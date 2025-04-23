from typing import TypedDict, cast, override

import librosa
import os
from jiwer import cer
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from pydantic import BaseModel, ConfigDict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.modeling_outputs import BaseModelOutput
from huggingface_hub import login
#login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))


class Batch(TypedDict):
    audio_features: torch.Tensor
    audio_masks: torch.Tensor
    input_ids: torch.Tensor
    input_masks: torch.Tensor
    assistant_token_ids: torch.Tensor
    assistant_token_mask: torch.Tensor
    output_ids: torch.Tensor
    output_masks: torch.Tensor


class Audio(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: str
    array: np.ndarray
    sampling_rate: int


class Item(BaseModel):
    audio: Audio
    transcription: str


type LLMTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


class Projector(nn.Module):
    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_encoder_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)

class SoftPrompt(nn.Module):
    def __init__(self, prompt_len, hidden_size):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(prompt_len, hidden_size))

    def forward(self, batch_size):
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


def downsample(features: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = features.shape
    if seq_len % k != 0:
        raise ValueError(
            "Sequence length must be divisible by the downsample factor")
    features = features.view(batch_size, seq_len // k, k, hidden_size)
    downsampled_features = features.reshape(
        batch_size, seq_len // k, k * hidden_size)
    return downsampled_features

def downsample_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Downsample attention mask by k.
    If any frame in a group of k has value 1, the group is 1.
    """
    batch_size, seq_len = mask.shape
    if seq_len % k != 0:
        pad_len = k - (seq_len % k)
        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        seq_len += pad_len
    mask = mask.view(batch_size, seq_len // k, k)
    return mask.max(dim=2).values  # shape: [B, T//k]


def tokenize_text(text: str, tokenizer: LLMTokenizer):
    return tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=512, add_special_tokens=True)


def get_token_embedding(token_ids: torch.Tensor, model: PreTrainedModel, /):
    embedding_layer = model.get_input_embeddings()
    return cast(torch.Tensor, embedding_layer(token_ids)).to(dtype=model.dtype)


def process_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    match audio_data.shape:
        case [_]:
            ...
        case [_, 2]:  # convert to mono if needed
            audio_data = librosa.to_mono(audio_data.T)
        case [2, _]:
            audio_data = librosa.to_mono(audio_data)
        case _:
            raise ValueError(f"Invalid audio data shape: {audio_data.shape}")

    if orig_sr != target_sr:  # Resample if needed
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    return audio_data


def data_collator(batch: list[Item], llm_tokenizer: LLMTokenizer, feature_extractor: WhisperFeatureExtractor) -> Batch:
    # Extract batch components
    audio_data = [item.audio for item in batch]
    # raw_input_prompts = [item["input_prompt"] for item in batch]
    # output_labels = [item["output_label"] for item in batch]
    raw_input_prompts = ["日本語で音声認識してください。"] * len(audio_data)
    output_labels = [item.transcription for item in batch]

    # Format input prompts with special tokens
    formatted_prompts = []
    for prompt in raw_input_prompts:
        formatted_prompt = (
            "<|im_start|>system\n"
            f"{prompt.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        formatted_prompts.append(formatted_prompt)

    # Process audio features
    processed_audio = [process_audio(audio.array, audio.sampling_rate) for audio in audio_data]
    audio_features = feature_extractor(
        processed_audio, sampling_rate=16000, return_tensors="pt", padding="max_length", return_attention_mask=True
    )

    # Add assistant token after audio
    assistant_token = ["<|im_end|>\n<|im_start|>assistant\n"] * len(audio_data)

    # Tokenize input prompts
    input_tokens = llm_tokenizer(
        formatted_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=False,  # 因为我们已经手动添加了特殊token
    )

    # Tokenize assistant token separately (will be used after audio features)
    assistant_tokens = llm_tokenizer(assistant_token, return_tensors="pt", add_special_tokens=False)

    # Tokenize output
    output_texts = [f"{label}{llm_tokenizer.eos_token}" for label in output_labels]
    output_tokens = llm_tokenizer(
        output_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        add_special_tokens=False,  # 因为我们手动添加了EOS
    )

    return {
        "audio_features": audio_features.input_features,
        "audio_masks": audio_features.attention_mask,
        "input_ids": input_tokens.input_ids,
        "input_masks": input_tokens.attention_mask,
        "assistant_token_ids": assistant_tokens.input_ids,
        "assistant_token_mask": assistant_tokens.attention_mask,
        "output_ids": output_tokens.input_ids,
        "output_masks": output_tokens.attention_mask,
    }


class ItemDataset[T](Dataset[T]):
    def __init__(self, data: list[T]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        return self.data[idx]


class DataInterface(L.LightningDataModule):
    def __init__(
        self,
        llm_tokenizer: LLMTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["llm_tokenizer", "feature_extractor"])
        self.llm_tokenizer = llm_tokenizer
        self.feature_extractor = feature_extractor

    @override
    def setup(self, stage: str):
        raw_train_dataset = load_dataset("japanese-asr/ja_asr.jsut_basic5000", split="test")
        train_dataset = ItemDataset([Item.model_validate(item) for item in raw_train_dataset])
        #raw_eval_dataset = load_dataset("RecoseleInc/middle_hardness_call_center", split="train").rename_column("transcription_kanji", "transcription")
        eval_dataset = ItemDataset([Item.model_validate(item) for item in raw_train_dataset.select(range(5))])
        self.train_dataset, self.val_dataset = train_dataset, eval_dataset

    def collate_fn(self, batch: list[Item]):
        return data_collator(batch, self.llm_tokenizer, self.feature_extractor)

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )


class SpeechLLMModel(L.LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        llm_path: str,
        speech_encoder_path: str = "openai/whisper-large-v3",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Whisper encoder
        self.speech_encoder = WhisperModel.from_pretrained(speech_encoder_path).encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(speech_encoder_path)
        assert isinstance(self.feature_extractor, WhisperFeatureExtractor)
        # Initialize LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.padding_side = "left"
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")
        #assert isinstance(self.llm_model, Qwen2ForCausalLM)
        # Initialize Projector
        self.downsample_factor = 4
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )
        self.soft_prompt_len = 20  
        self.soft_prompt_embeddings = SoftPrompt(self.soft_prompt_len, self.llm_model.config.hidden_size)
        
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.projector.load_state_dict({k.replace("projector.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("projector.")})
            self.soft_prompt_embeddings.load_state_dict({k.replace("soft_prompt.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("soft_prompt.")})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_encoder.to(device)
        self.projector.to(device)
        self.llm_model.to(device)
        self.soft_prompt_embeddings.to(device)

        # Freeze models except projector
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, batch: Batch):
        # Process speech input
        speech_features = cast(BaseModelOutput, self.speech_encoder(batch["audio_features"]))
        speech_embeds = speech_features.last_hidden_state
        speech_embeds = downsample(speech_embeds, self.downsample_factor)
        projected_speech = self.projector(speech_embeds)

        # Get input text embeddings
        input_embeds = get_token_embedding(batch["input_ids"], self.llm_model)

        # Combine embeddings
        combined_embeds = torch.cat([input_embeds, projected_speech], dim=1)
        combined_mask = torch.cat([batch["input_masks"], batch["audio_masks"][:, :: self.downsample_factor]], dim=1)

        return combined_embeds, combined_mask

    def training_step_with_llm(self, batch: Batch, batch_idx: int):
        # 处理语音输入
        speech_features = cast(BaseModelOutput, self.speech_encoder(batch["audio_features"]))
        speech_embeds = speech_features.last_hidden_state
        downsampled_embeds = downsample(speech_embeds, self.downsample_factor)
        projected_speech = self.projector(downsampled_embeds)

        # 获取各部分的嵌入
        input_embeds = get_token_embedding(batch["input_ids"], self.llm_model)
        assistant_embeds = get_token_embedding(batch["assistant_token_ids"], self.llm_model)

        # 拼接前缀部分（系统+音频+助理）
        prefix_embeds = torch.cat([input_embeds, projected_speech, assistant_embeds], dim=1)

        # 获取目标输出的嵌入
        target_embeds = get_token_embedding(batch["output_ids"], self.llm_model)
        # 输入序列使用除了最后一个token的部分
        input_target_embeds = target_embeds[:, :-1, :]

        # 完整的输入嵌入序列
        combined_embeds = torch.cat([prefix_embeds, input_target_embeds], dim=1)

        # 注意力掩码
        audio_mask_downsampled = batch["audio_masks"][:, :: self.downsample_factor]
        prefix_mask = torch.cat([batch["input_masks"], audio_mask_downsampled, batch["assistant_token_mask"]], dim=1)
        target_mask = batch["output_masks"][:, :-1]  # 对应input_target_embeds
        combined_mask = torch.cat([prefix_mask, target_mask], dim=1)

        # 构建标签序列
        batch_size = prefix_embeds.size(0)
        prefix_len = prefix_embeds.size(1)
        # 前缀部分的标签都设为-100，只计算目标输出部分的损失
        labels = torch.full((batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=combined_embeds.device)
        # 将目标序列从第二个token开始的部分作为标签（右移一位）
        labels[:, prefix_len:] = batch["output_ids"][:, 1:]

        if batch_idx == 0:  # Debug信息
            print("Shapes:")
            print(f"Prefix embeddings: {prefix_embeds.shape}")
            print(f"Target embeddings: {input_target_embeds.shape}")
            print(f"Combined embeddings: {combined_embeds.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Prefix length: {prefix_len}")
            print(f"Number of actual target tokens: {(labels != -100).sum().item()}")

        # 计算损失
        outputs = self.llm_model(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)

        loss = outputs.loss
        self.log("train_loss", loss)

        return loss

    def training_step(self, batch: Batch, batch_idx: int):
        # 1. 处理语音输入（frozen部分，用 no_grad 计算）
        with torch.no_grad():
            speech_features = cast(BaseModelOutput, self.speech_encoder(batch["audio_features"]))
            speech_embeds = speech_features.last_hidden_state
            downsampled_embeds = downsample(speech_embeds, self.downsample_factor)
        # 2. projector 的计算（需要梯度）
        projected_speech = self.projector(downsampled_embeds)

        # 3. 其他部分使用冻结模块的输出，但采用 .detach() 而不是 no_grad 包裹后续运算
        # 这样可以在不计算梯度的同时，仍让后续运算在正常的 autograd 环境下执行，从而保留 projector 输出的梯度流。
        input_embeds = get_token_embedding(batch["input_ids"], self.llm_model).detach()
        batch_size = input_embeds.size(0)
        soft_prompt_expanded = self.soft_prompt_embeddings(batch_size)
        assistant_embeds = get_token_embedding(batch["assistant_token_ids"], self.llm_model).detach()
        reply_embeds = get_token_embedding(batch["output_ids"], self.llm_model).detach()
        input_reply_embeds = reply_embeds[:, :-1, :]

        # 4. 拼接所有输入序列（注意此处不再处于 no_grad 环境）
        prefix_embeds = torch.cat([input_embeds, soft_prompt_expanded, projected_speech, assistant_embeds], dim=1)
        combined_embeds = torch.cat([prefix_embeds, input_reply_embeds], dim=1)

        # 5. 处理 attention masks（同理，操作应在正常环境下进行）
        soft_prompt_mask = torch.ones(batch_size, self.soft_prompt_len, device=self.device) 
        audio_mask_downsampled = downsample_mask(batch["audio_masks"].to(self.device), k=self.downsample_factor)
        prefix_mask = torch.cat([batch["input_masks"], soft_prompt_mask, audio_mask_downsampled, batch["assistant_token_mask"]], dim=1)
        reply_mask = batch["output_masks"][:, :-1]
        combined_mask = torch.cat([prefix_mask, reply_mask], dim=1)

        # 6. 设置标签：前缀部分设为 -100，只有回复部分计算 loss
        _, prefix_len, _ = prefix_embeds.size()
        labels = torch.full((batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=combined_embeds.device)
        labels[:, prefix_len:] = batch["output_ids"][:, 1:]

        # 7. 计算 loss（注意：LLM参数已冻结，即使在正常环境下计算也不会更新）
        outputs = self.llm_model(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)

        if batch_idx == 0:
            print("Shapes:")
            print(f"prefix_embeds: {prefix_embeds.shape}")
            print(f"input_reply_embeds: {input_reply_embeds.shape}")
            print(f"combined_embeds: {combined_embeds.shape}")
            print(f"labels: {labels.shape}")
            print(f"Number of actual target tokens: {(labels != -100).sum().item()}")

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        with torch.no_grad():
            # === 1. 语音编码 ===
            speech_features = self.speech_encoder(batch["audio_features"])
            speech_embeds = speech_features.last_hidden_state
            downsampled_embeds = downsample(speech_embeds, self.downsample_factor)
            projected_speech = self.projector(downsampled_embeds)

            # === 2. 获取文本部分的嵌入 ===
            input_embeds = get_token_embedding(batch["input_ids"], self.llm_model).detach()
            assistant_embeds = get_token_embedding(batch["assistant_token_ids"], self.llm_model).detach()
            reply_embeds = get_token_embedding(batch["output_ids"], self.llm_model).detach()
            input_reply_embeds = reply_embeds[:, :-1, :]

            # === 3. Soft Prompt + 拼接 ===
            batch_size = input_embeds.size(0)
            soft_prompt_expanded = self.soft_prompt_embeddings(batch_size)
            prefix_embeds = torch.cat([input_embeds, soft_prompt_expanded, projected_speech, assistant_embeds], dim=1)
            combined_embeds = torch.cat([prefix_embeds, input_reply_embeds], dim=1)

            # === 4. attention mask 拼接 ===
            soft_prompt_mask = torch.ones(batch_size, self.soft_prompt_len, device=self.device)
            audio_mask_downsampled = downsample_mask(batch["audio_masks"].to(self.device), k=self.downsample_factor)
            prefix_mask = torch.cat([batch["input_masks"], soft_prompt_mask, audio_mask_downsampled, batch["assistant_token_mask"]], dim=1)
            reply_mask = batch["output_masks"][:, :-1]
            combined_mask = torch.cat([prefix_mask, reply_mask], dim=1)

            # === 5. 标签设置 ===
            prefix_len = prefix_embeds.size(1)
            labels = torch.full((batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=self.device)
            labels[:, prefix_len:] = batch["output_ids"][:, 1:]

            # === 6. 验证损失计算 ===
            val_outputs = self.llm_model(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)
            val_loss = val_outputs.loss
            self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)

            # === 7. 推理输出 ===
            generated_ids = self.llm_model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                max_new_tokens=500,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                temperature=1.0,
                do_sample=False,
            )

            # === 8. 解码生成 & 参考文本 ===
            generated_text = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            reference_text = self.llm_tokenizer.batch_decode(batch["output_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if batch_idx == 0:
                print("\nValidation Shapes:")
                print(f"prefix_embeds: {prefix_embeds.shape}")
                print(f"combined_embeds: {combined_embeds.shape}")
                print(f"labels: {labels.shape}")
                print(f"Actual target tokens: {(labels != -100).sum().item()}")
                for i in range(min(2, len(generated_text))):
                    print(f"\nExample {i + 1}:")
                    print(f"Generated: \033[43m{generated_text[i]}\033[0m")
                    print(f"Reference: \033[44m{reference_text[i]}\033[0m")

            # === 9. 计算 CER ===
            cer_scores = [cer(ref, hyp) for ref, hyp in zip(reference_text, generated_text)]
            avg_cer = sum(cer_scores) / len(cer_scores)
            self.log("val_cer", avg_cer, prog_bar=True, sync_dist=True)

            return {"val_loss": val_loss, "val_cer": avg_cer, "generated_texts": generated_text, "reference_texts": reference_text}

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint.get('state_dict', {})

        # Exclude keys that start with 'speech_encoder' or 'llm_model'
        filtered_state_dict = {k: v for k, v in state_dict.items(
        ) if not k.startswith(('speech_encoder', 'llm_model'))}

        # Update the checkpoint with the filtered state_dict
        checkpoint['state_dict'] = filtered_state_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                        list(self.projector.parameters()) + list(self.soft_prompt_embeddings.parameters()),
                        lr=self.hparams.learning_rate
                    )

        def lr_lambda(current_step):
            warmup_steps = self.hparams.warmup_steps
            total_steps = self.trainer.estimated_stepping_batches or 10000
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("model.llm_tokenizer", "data.llm_tokenizer", apply_on="instantiate")
        parser.link_arguments("model.feature_extractor", "data.feature_extractor", apply_on="instantiate")


def main():
    cli = MyLightningCLI(
        SpeechLLMModel,
        DataInterface,
        trainer_defaults={
            "logger": {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": "tb_logs",
                    "name": "projector"
                }
            }
        }
    )



if __name__ == "__main__":
    print("start")
    torch.set_float32_matmul_precision("medium")
    main()

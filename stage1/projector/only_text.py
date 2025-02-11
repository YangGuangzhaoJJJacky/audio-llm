import os
import torch
import torch.distributed as dist
import torch.nn.functional as F 
import torch.nn as nn
from torchmetrics.text import CharErrorRate, WordErrorRate
import librosa
import torchaudio.transforms as T
from torch.optim.lr_scheduler import LambdaLR
import lightning as l
import unicodedata
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
)
from datasets import load_dataset


class Projector(nn.Module):
    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(speech_encoder_hidden_size),
            nn.Linear(speech_encoder_hidden_size, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


def downsample(features: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = features.shape
    # Check if seq_len is divisible by k to avoid losing data in reshaping
    if seq_len % k != 0:
        raise ValueError("Sequence length must be divisible by the downsample factor")
    # Reshape to group every k elements
    features = features.view(batch_size, seq_len // k, k, hidden_size)
    # Reshape to merge the grouped elements into a single feature vector
    downsampled_features = features.reshape(batch_size, seq_len // k, k * hidden_size)
    return downsampled_features


def tokenize_text(text, tokenizer):
    return tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    )


def get_text_embedding(text, model, tokenizer, return_attention_mask=False):
    tokens = tokenize_text(text, tokenizer)
    token_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(token_ids)

    if return_attention_mask:
        return embeddings.to(dtype=model.dtype), attention_mask
    else:
        return embeddings.to(dtype=model.dtype)


def get_token_embedding(token, model):
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(token)

    return embeddings.to(dtype=model.dtype)


def clean_text(text):
    chars_to_strip = "「」『』（）〔〕［］｛｝｟｠〈〉《》【】〖〗〘〙〚〛"
    text = unicodedata.normalize("NFKC", text)
    text = text.strip(chars_to_strip)
    text = (
        text.replace("」。", "").replace("』。", "").replace("♥", "").replace("�", "")
    )
    # text = text.lower()
    return text


def data_collator(batch, llm_tokenizer, feature_extractor):
    """
    Collate function for the ASR dataset that processes speech and text data.

    Args:
        batch: List of dictionaries containing input_audio, input_prompt and output_text
        llm_tokenizer: Tokenizer for the language model
        feature_extractor: Feature extractor for the speech model

    Returns:
        Dictionary containing processed batch data
    """
    speeches = [item["input_audio"] for item in batch]
    input_prompts = [item["input_prompt"] for item in batch]
    transcriptions = [item["audio_transcription"] for item in batch]

    # Process system prompt
    system_prompts = []
    for prompt in input_prompts:
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>"
        system_prompts.append(formatted_prompt)

    # Get embeddings for the system prompts
    all_prompt_embeds = []
    all_prompt_masks = []
    for prompt in system_prompts:
        prompt_embeds, prompt_mask = get_text_embedding(
            prompt, llm_tokenizer, return_attention_mask=True
        )
        all_prompt_embeds.append(prompt_embeds)
        all_prompt_masks.append(prompt_mask)

    user_prompt_embeds = torch.stack(all_prompt_embeds)
    user_prompt_masks = torch.stack(all_prompt_masks)

    # Process labels embding which is the audio transciption's first llm layer embeding
    labels = []
    for prompt, transcription in zip(system_prompts, transcriptions):
        label_prompt = f"{prompt}{transcription}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        labels.append(label_prompt)
    all_labels_embeds = []
    all_labels_masks = []
    for prompt in labels:
        label_embeds,label_mask = get_text_embedding(
            prompt, llm_tokenizer, return_attention_mask=True
        )
        all_labels_embeds.append(label_embeds)
        all_labels_masks.append(label_mask)
    labels_prompt_embeds = torch.stack(all_labels_embeds)
    labels_prompt_masks = torch.stack(all_labels_masks)

    # Process speech data
    speech_features = []
    speech_masks = []

    for speech in speeches:
        # Convert to mono if the audio has multiple channels
        if speech["array"].ndim > 1:
            speech_array = librosa.to_mono(speech["array"].T)
        else:
            speech_array = speech["array"]

        # Resample the audio to 16000 Hz using librosa
        speech_array = librosa.resample(
            speech_array, orig_sr=speech["sampling_rate"], target_sr=16000)
        speech["sampling_rate"] = 16000

        # Extract features and apply SpecAugment
        speech_feature = feature_extractor(
            speech_array, 
            sampling_rate=speech["sampling_rate"], 
            return_tensors="pt", 
            return_attention_mask=True
        )
        spectrogram = speech_feature.input_features.squeeze(0)

        # Apply SpecAugment for data augmentation
        specaug = T.SpecAugment(
            freq_mask_param=15, 
            time_mask_param=50, 
            n_freq_masks=2, 
            n_time_masks=10, 
            p=0.5
        )
        augmented_spectrogram = specaug(spectrogram)

        speech_features.append(augmented_spectrogram)
        speech_masks.append(speech_feature.attention_mask.squeeze(0))

    speech_features = torch.stack(speech_features)
    speech_masks = torch.stack(speech_masks)

    return {
        "speeches": speech_features,
        "speeches_masks": speech_masks,
        "labels": labels_prompt_embeds,
        "labels_mask": labels_prompt_masks,
        "user_prompt": user_prompt_embeds,
        "user_prompt_mask": user_prompt_masks,
        "original_labels": labels,
    }


class SLAM_ASR(l.LightningModule):
    def __init__(self, batch_size: int = 1, warmup_steps: int = 1000, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        speech_encoder_path = "openai/whisper-large-v3"
        self.speech_encoder = WhisperModel.from_pretrained(speech_encoder_path).encoder

        llm_path = "/home/recosele/Development/End2End/stage1/Llama-3-ELYZA-JP-8B"

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            speech_encoder_path
        )

        self.downsample_factor = 5
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )

        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.assistant_prompt_embeds, self.assistant_mask = get_text_embedding(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            self.llm_model,
            self.llm_tokenizer,
            return_attention_mask=True,
        )

    def setup(self, stage=None):
        combined_dataset = load_dataset(
            "withourai/withourai-asr-mix", token=True, split="train"
        )

        train_val_split = combined_dataset.train_test_split(0.001)
        self.train_dataset = train_val_split["train"]
        self.val_dataset = train_val_split["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(
                batch, self.llm_tokenizer, self.feature_extractor
            ),
            num_workers=20,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(
                batch, self.llm_tokenizer, self.feature_extractor
            ),
            num_workers=20,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
        )

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", {})

        # Exclude keys that start with 'speech_encoder' or 'llm_model'
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("speech_encoder", "llm_model"))
        }

        # Update the checkpoint with the filtered state_dict
        checkpoint["state_dict"] = filtered_state_dict

    def on_load_checkpoint(self, checkpoint):
        pretrained_dict = checkpoint.get("state_dict", {})
        model_dict = self.state_dict()

        pretrained_dict = {
            k: v for k, v in pretrained_dict["state_dict"].items() if k in model_dict
        }

        model_dict.update(pretrained_dict)

        # Load the filtered state_dict into the model
        self.load_state_dict(model_dict)

    def training_step(self, batch, batch_idx):
        batch_size = self.hparams.batch_size
        with torch.no_grad():
            user_prompt_embeds = batch["user_prompt"].to(
                device=self.llm_model.device, 
                dtype=self.llm_model.dtype
            )
            user_mask = batch["user_prompt_mask"].to(self.llm_model.device)


            assistant_prompt_embeds = self.assistant_prompt_embeds.expand(
                batch_size, -1, -1).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            assistant_mask = self.assistant_mask.expand(
                batch_size, -1).to(self.llm_model.device)

            labels_embeds = batch["labels"].to(self.llm_model.device)            

            speech_embeds = self.speech_encoder(batch["speeches"])
            speech_embeds = speech_embeds.last_hidden_state
            downsampled_embeds = downsample(
                speech_embeds, k=self.downsample_factor)

        projected_embeds = self.projector(downsampled_embeds)

        # Don't think the mask works for projected samples
        speech_mask = torch.ones_like(downsampled_embeds[:, :, 0])

        inputs_embeds = torch.cat([
            user_prompt_embeds,
            projected_embeds,
            assistant_prompt_embeds,
        ], dim=1)

        attention_masks = torch.cat([
            user_mask,
            speech_mask,
            assistant_mask,
        ], dim=1)

        
        loss = F.mse_loss(inputs_embeds, labels_embeds)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = self.hparams.batch_size
        with torch.no_grad():
            # 处理用户提示部分
            user_prompt_embeds = batch["user_prompt"].to(
                device=self.llm_model.device, 
                dtype=self.llm_model.dtype
            )
            user_mask = batch["user_prompt_mask"].to(self.llm_model.device)

            # 处理助手提示部分
            assistant_prompt_embeds = self.assistant_prompt_embeds.expand(
                batch_size, -1, -1
            ).to(device=self.llm_model.device, dtype=self.llm_model.dtype)
            assistant_mask = self.assistant_mask.expand(
                batch_size, -1
            ).to(self.llm_model.device)

            # 处理标签
            labels_embeds = batch["labels"].to(self.llm_model.device)

            # 处理语音部分
            speech_embeds = self.speech_encoder(batch["speeches"])
            speech_embeds = speech_embeds.last_hidden_state
            downsampled_embeds = downsample(speech_embeds, k=self.downsample_factor)

            # 投影语音特征
            projected_embeds = self.projector(downsampled_embeds)
            speech_mask = torch.ones_like(downsampled_embeds[:, :, 0])

            # 拼接所有部分
            inputs_embeds = torch.cat([
                user_prompt_embeds,
                projected_embeds,
                assistant_prompt_embeds,
            ], dim=1)

            attention_masks = torch.cat([
                user_mask,
                speech_mask,
                assistant_mask,
            ], dim=1)

            # 计算验证损失
            loss = F.mse_loss(inputs_embeds, labels_embeds)
            self.log("val_loss", loss)

            # 每个批次生成一个样本用于可视化效果
            input_embed = inputs_embeds[0].unsqueeze(0)
            attention_mask = attention_masks[0].unsqueeze(0)

            output = self.llm_model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                max_new_tokens=250,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                num_beams=2,
                early_stopping=True,
            )

            decoded_pred = self.llm_tokenizer.decode(
                output[0], skip_special_tokens=True
            )

            if dist.get_rank() == 0:
                print(f"\nInput Prompt: {batch['original_prompts'][0]}")  # 打印输入
                print(f"\nPredicted: {decoded_pred}")

            return {"val_loss": loss}
        
    def configure_optimizers(self):
        # Create the optimizer with the specified learning rate and no weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0
        )

        # Define a scheduler that warms up the learning rate for the first 1,000 steps
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return float(step) / float(max(1, self.hparams.warmup_steps))
            return 1.0

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # Update the learning rate after every step
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def cli_main():
    torch.set_float32_matmul_precision("medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    cli = LightningCLI(SLAM_ASR)


if __name__ == "__main__":
    cli_main()

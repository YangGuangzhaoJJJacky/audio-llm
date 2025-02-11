import os
import torch
import torch.nn as nn
import librosa
import lightning as L
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
    BitsAndBytesConfig
)
from datasets import load_dataset
from lightning.pytorch.cli import LightningCLI

class Projector(nn.Module):
    def __init__(self, speech_hidden, llm_hidden):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_hidden, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, llm_hidden),
            nn.LayerNorm(llm_hidden)
        )
    
    def forward(self, x):
        return self.proj(x)

def downsample(features: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, seq_len, hidden_size = features.shape
    if seq_len % k != 0:
        pad_len = k - (seq_len % k)
        features = torch.nn.functional.pad(features, (0, 0, 0, pad_len))
        seq_len = features.shape[1]
    features = features.view(batch_size, seq_len // k, k, hidden_size)
    downsampled_features = features.mean(dim=2)
    return downsampled_features

def tokenize_text(text, tokenizer):
    return tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )

def get_token_embedding(token_ids, model):
    embedding_layer = model.get_input_embeddings()
    return embedding_layer(token_ids).to(dtype=model.dtype)

def process_audio(audio_data, target_sr=16000):
    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = librosa.to_mono(audio_data.T)
    
    # Resample if needed
    if audio_data.shape[0] != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=audio_data.shape[0], target_sr=target_sr)
    
    return audio_data

def data_collator(batch, llm_tokenizer, feature_extractor):
    # Extract batch components
    audio_data = [item["audio"]["array"] for item in batch]
    raw_input_prompts = [item["input_prompt"] for item in batch]
    output_labels = [item["output_label"] for item in batch]

    # Format input prompts with special tokens
    formatted_prompts = []
    for prompt in raw_input_prompts:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            f"{prompt}"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
        )
        formatted_prompts.append(formatted_prompt)

    # Process audio features
    processed_audio = [process_audio(audio) for audio in audio_data]
    audio_features = feature_extractor(
        processed_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Add assistant token after audio
    assistant_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Tokenize input prompts
    input_tokens = llm_tokenizer(
        formatted_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=False  # 因为我们已经手动添加了特殊token
    )

    # Tokenize assistant token separately (will be used after audio features)
    assistant_tokens = llm_tokenizer(
        assistant_token,
        return_tensors="pt",
        add_special_tokens=False
    )

    # Tokenize output
    output_texts = [f"{label}{llm_tokenizer.eos_token}" for label in output_labels]
    output_tokens = llm_tokenizer(
        output_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        add_special_tokens=False  # 因为我们手动添加了EOS
    )

    return {
        "audio_features": audio_features.input_features,
        "audio_masks": audio_features.attention_mask,
        "input_ids": input_tokens.input_ids,
        "input_masks": input_tokens.attention_mask,
        "assistant_token_ids": assistant_tokens.input_ids,
        "assistant_token_mask": assistant_tokens.attention_mask,
        "output_ids": output_tokens.input_ids,
        "output_masks": output_tokens.attention_mask
    }

class SpeechLLMModel(L.LightningModule):
    def __init__(
        self,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        speech_encoder_path: str = "openai/whisper-large-v3",
        llm_path: str = "path/to/your/llm",
        dataset_path: str = "your/dataset/path",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize Whisper encoder
        self.speech_encoder = WhisperModel.from_pretrained(speech_encoder_path).encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(speech_encoder_path)
        
        # Initialize LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="auto"
        )
        
        # Initialize Projector
        self.downsample_factor = 4
        self.projector = Projector(
            self.speech_encoder.config.hidden_size,
            self.llm_model.config.hidden_size
        )
        
        # Freeze models except projector
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
    
    def setup(self, stage=None):
        # Load and split dataset
        dataset = load_dataset(self.hparams.dataset_path)
        train_val_split = dataset.train_test_split(test_size=0.1)
        self.train_dataset = train_val_split["train"]
        self.val_dataset = train_val_split["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(batch, self.llm_tokenizer, self.feature_extractor),
            num_workers=4,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda batch: data_collator(batch, self.llm_tokenizer, self.feature_extractor),
            num_workers=4
        )

    def forward(self, batch):
        # Process speech input
        speech_features = self.speech_encoder(batch["audio_features"])
        speech_embeds = speech_features.last_hidden_state
        speech_embeds = downsample(speech_embeds, self.downsample_factor)
        projected_speech = self.projector(speech_embeds)

        # Get input text embeddings
        input_embeds = get_token_embedding(batch["input_ids"], self.llm_model)

        # Combine embeddings
        combined_embeds = torch.cat([input_embeds, projected_speech], dim=1)
        combined_mask = torch.cat([
            batch["input_masks"],
            batch["audio_masks"][:, ::self.downsample_factor]
        ], dim=1)

        return combined_embeds, combined_mask

    def training_step_with_llm(self, batch, batch_idx):
        # 处理语音输入
        speech_features = self.speech_encoder(batch["audio_features"])
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
        audio_mask_downsampled = batch["audio_masks"][:, ::self.downsample_factor]
        prefix_mask = torch.cat([
            batch["input_masks"],
            audio_mask_downsampled,
            batch["assistant_token_mask"]
        ], dim=1)
        target_mask = batch["output_masks"][:, :-1]  # 对应input_target_embeds
        combined_mask = torch.cat([prefix_mask, target_mask], dim=1)
        
        # 构建标签序列
        batch_size = prefix_embeds.size(0)
        prefix_len = prefix_embeds.size(1)
        # 前缀部分的标签都设为-100，只计算目标输出部分的损失
        labels = torch.full((batch_size, combined_embeds.size(1)), -100, 
                        dtype=torch.long, device=combined_embeds.device)
        # 将目标序列从第二个token开始的部分作为标签（右移一位）
        labels[:, prefix_len:] = batch["output_ids"][:, 1:]
        
        if batch_idx == 0:  # Debug信息
            print(f"Shapes:")
            print(f"Prefix embeddings: {prefix_embeds.shape}")
            print(f"Target embeddings: {input_target_embeds.shape}")
            print(f"Combined embeddings: {combined_embeds.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Prefix length: {prefix_len}")
            print(f"Number of actual target tokens: {(labels != -100).sum().item()}")
        
        # 计算损失
        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels
        )
        
        loss = outputs.loss
        self.log("train_loss", loss)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # 1. 处理语音输入（frozen部分，用 no_grad 计算）
        with torch.no_grad():
            speech_features = self.speech_encoder(batch["audio_features"])
            speech_embeds = speech_features.last_hidden_state
            downsampled_embeds = downsample(speech_embeds, self.downsample_factor)
        # 2. projector 的计算（需要梯度）
        projected_speech = self.projector(downsampled_embeds)
        
        # 3. 其他部分使用冻结模块的输出，但采用 .detach() 而不是 no_grad 包裹后续运算
        # 这样可以在不计算梯度的同时，仍让后续运算在正常的 autograd 环境下执行，从而保留 projector 输出的梯度流。
        input_embeds = get_token_embedding(batch["input_ids"], self.llm_model).detach()
        assistant_embeds = get_token_embedding(batch["assistant_token_ids"], self.llm_model).detach()
        reply_embeds = get_token_embedding(batch["output_ids"], self.llm_model).detach()
        input_reply_embeds = reply_embeds[:, :-1, :]

        # 4. 拼接所有输入序列（注意此处不再处于 no_grad 环境）
        prefix_embeds = torch.cat([input_embeds, projected_speech, assistant_embeds], dim=1)
        combined_embeds = torch.cat([prefix_embeds, input_reply_embeds], dim=1)

        # 5. 处理 attention masks（同理，操作应在正常环境下进行）
        audio_mask_downsampled = batch["audio_masks"][:, ::self.downsample_factor]
        prefix_mask = torch.cat([batch["input_masks"], audio_mask_downsampled, batch["assistant_token_mask"]], dim=1)
        reply_mask = batch["output_masks"][:, :-1]
        combined_mask = torch.cat([prefix_mask, reply_mask], dim=1)

        # 6. 设置标签：前缀部分设为 -100，只有回复部分计算 loss
        batch_size = combined_embeds.size(0)
        prefix_len = prefix_embeds.size(1)
        labels = torch.full((batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=combined_embeds.device)
        labels[:, prefix_len:] = batch["output_ids"][:, 1:]

        # 7. 计算 loss（注意：LLM参数已冻结，即使在正常环境下计算也不会更新）
        outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels
        )
        
        if batch_idx == 0:
            print(f"Shapes:")
            print(f"prefix_embeds: {prefix_embeds.shape}")
            print(f"input_reply_embeds: {input_reply_embeds.shape}")
            print(f"combined_embeds: {combined_embeds.shape}")
            print(f"labels: {labels.shape}")
            print(f"Number of actual target tokens: {(labels != -100).sum().item()}")
        
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # 1. 处理语音输入（validation时也使用no_grad）
        with torch.no_grad():
            speech_features = self.speech_encoder(batch["audio_features"])
            speech_embeds = speech_features.last_hidden_state
            downsampled_embeds = downsample(speech_embeds, self.downsample_factor)
            projected_speech = self.projector(downsampled_embeds)
            
            # 2. 获取嵌入并detach
            input_embeds = get_token_embedding(batch["input_ids"], self.llm_model).detach()
            assistant_embeds = get_token_embedding(batch["assistant_token_ids"], self.llm_model).detach()
            reply_embeds = get_token_embedding(batch["output_ids"], self.llm_model).detach()
            input_reply_embeds = reply_embeds[:, :-1, :]
            
            # 3. 拼接所有输入序列
            prefix_embeds = torch.cat([input_embeds, projected_speech, assistant_embeds], dim=1)
            combined_embeds = torch.cat([prefix_embeds, input_reply_embeds], dim=1)
            
            # 4. 处理attention masks
            audio_mask_downsampled = batch["audio_masks"][:, ::self.downsample_factor]
            prefix_mask = torch.cat([batch["input_masks"], audio_mask_downsampled, batch["assistant_token_mask"]], dim=1)
            reply_mask = batch["output_masks"][:, :-1]
            combined_mask = torch.cat([prefix_mask, reply_mask], dim=1)
            
            # 5. 设置标签，与training_step保持一致
            batch_size = combined_embeds.size(0)
            prefix_len = prefix_embeds.size(1)
            labels = torch.full((batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=combined_embeds.device)
            labels[:, prefix_len:] = batch["output_ids"][:, 1:]
            
            # 6. 计算验证损失
            val_outputs = self.llm_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=labels
            )
            
            val_loss = val_outputs.loss
            self.log("val_loss", val_loss, sync_dist=True)
            
            # 7. 生成文本进行评估
            generated_ids = self.llm_model.generate(
                inputs_embeds=prefix_embeds,  # 注意这里只用prefix部分
                attention_mask=prefix_mask,    # 相应的mask也只用prefix部分
                max_new_tokens=500,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # 8. 解码生成的文本和参考文本
            generated_text = self.llm_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            reference_text = self.llm_tokenizer.batch_decode(
                batch["output_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 9. 打印调试信息
            if batch_idx == 0:
                print("\nValidation Shapes:")
                print(f"prefix_embeds: {prefix_embeds.shape}")
                print(f"input_reply_embeds: {input_reply_embeds.shape}")
                print(f"combined_embeds: {combined_embeds.shape}")
                print(f"labels: {labels.shape}")
                print(f"Number of actual target tokens: {(labels != -100).sum().item()}")
                
                print("\nValidation Examples:")
                for i in range(min(2, len(generated_text))):
                    print(f"\nExample {i+1}:")
                    print(f"Generated: {generated_text[i]}")
                    print(f"Reference: {reference_text[i]}")
                    print("-" * 50)
            
            return {
                "val_loss": val_loss,
                "generated_texts": generated_text,
                "reference_texts": reference_text
            }
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=self.hparams.learning_rate
        )
        
        scheduler = {
            'scheduler': LambdaLR(
                optimizer,
                lambda step: min(1.0, step / self.hparams.warmup_steps)
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]

def main():
    cli = LightningCLI(SpeechLLMModel)

if __name__ == "__main__":
    main()
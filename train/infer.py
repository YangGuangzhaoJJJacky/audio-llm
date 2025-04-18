import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, WhisperFeatureExtractor
from train.projector import Projector, process_audio, downsample, get_token_embedding  
import numpy as np
import librosa


@torch.no_grad()
def infer(
    audio_path: str,
    speech_encoder_path: str,
    llm_path: str,
    projector_path: str,
    prompt: str = "以下の音声データを日本語で音声認識してください。",
    assistant_token_text: str = "<|im_end|>\n<|im_start|>assistant\n",
    max_new_tokens: int = 256
):
    # Load models
    print("Loading models...")
    speech_encoder = WhisperModel.from_pretrained(speech_encoder_path).encoder.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(speech_encoder_path)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="right")
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto").eval()
    projector = Projector(speech_encoder.config.hidden_size, llm_model.config.hidden_size).eval()
    soft_prompt = SoftPrompt(length=20, dim=llm_model.config.hidden_size).eval()

    ckpt = torch.load(projector_path, map_location="cpu")
    projector.load_state_dict({k.replace("projector.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("projector.")})
    soft_prompt.load_state_dict({k.replace("soft_prompt.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("soft_prompt.")})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speech_encoder.to(device)
    projector.to(device)
    llm_model.to(device)
    soft_prompt.to(device)

    # Load and process audio
    print("Processing audio...")
    audio_data, sr = librosa.load(audio_path, sr=None)
    audio_array = process_audio(audio_data, orig_sr=sr, target_sr=16000)
    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length", return_attention_mask=True)
    audio_features = inputs.input_features.to(device)
    audio_mask = inputs.attention_mask.to(device)

    # Whisper encoder
    speech_outputs = speech_encoder(audio_features)
    speech_embeds = downsample(speech_outputs.last_hidden_state, k=4)
    projected_speech = projector(speech_embeds)

    # Prompt text + assistant token
    formatted_prompt = (
            "<|im_start|>system\n"
            f"{prompt.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
    assistant_token = assistant_token_text
    soft_prompt_expanded = soft_prompt(batch_size=1)

    # Tokenize prompt & assistant
    prompt_tokens = llm_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    assistant_tokens = llm_tokenizer(assistant_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Get token embeddings
    prompt_embeds = get_token_embedding(prompt_tokens, llm_model).detach()
    assistant_embeds = get_token_embedding(assistant_tokens, llm_model).detach()

    # Combine embedding sequence
    prefix_embeds = torch.cat([prompt_embeds, soft_prompt_expanded, projected_speech, assistant_embeds], dim=1)

    # Create attention mask
    prefix_mask = torch.cat([
        torch.ones(1, prompt_embeds.size(1), dtype=torch.long, device=device),
        torch.ones(1, soft_prompt.length, dtype=torch.long, device=device),
        audio_mask[:, ::4],
        torch.ones(1, assistant_embeds.size(1), dtype=torch.long, device=device)
    ], dim=1)


    # Generate
    print("Generating...")
    generated_ids = llm_model.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=prefix_mask,
        max_new_tokens=500,
        num_beams=1,                     
        early_stopping=True,
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9,
        do_sample=False                
    )



    result = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    print("\n🗣️ 推理完成：")
    print(f"\033[92m{result}\033[0m")

    return result


if __name__ == "__main__":
    import sys

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
    infer(
        audio_path=audio_path,
        speech_encoder_path="openai/whisper-large-v3",
        llm_path="pretrained_models/Qwen2.5-0.5B-Instruct",
        projector_path="/media/disk2/End2End/tb_logs/projector/version_3/checkpoints/epoch=28-step=18125.ckpt"
    )

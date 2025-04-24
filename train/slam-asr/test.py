from transformers import AutoTokenizer

# 加载 Qwen2.5 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 显示特殊 token
print("Special Tokens:")
print("bos_token:", tokenizer.bos_token)
print("eos_token:", tokenizer.eos_token)
print("pad_token:", tokenizer.pad_token)
print(tokenizer("<|im_start|>").input_ids)  # 应该只得到 1 个 token ID
# 额外重要的 system 指令格式
special_format = [
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|eot_id|>"
]

print("\n=== Token IDs ===")
for text in special_format:
    ids = tokenizer(text).input_ids
    print(f"{text!r} -> {ids}")
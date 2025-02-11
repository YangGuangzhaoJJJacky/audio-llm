from datasets import load_dataset, Dataset
import pandas as pd
import re

def process_text(text):
    # 提取 startofspeech 和 endofspeech 之间的内容
    voice_part_match = re.search(r'startofspeech(.*?)endofspeech', text, re.DOTALL)
    if voice_part_match:
        voice_part = voice_part_match.group(1)
        # 替换掉原文中的这部分（包括标记）得到text部分
        text_part = text.replace(f"startofspeech{voice_part}endofspeech", "")
    else:
        voice_part = ""
        text_part = text
    
    # 为text_part添加前缀
    text_part = "会話調で、できるだけ短く1、2文で回答してください。" + text_part
    
    return voice_part.strip(), text_part.strip()

def main():
    # 加载原始数据集
    print("Loading dataset...")
    original_dataset = load_dataset("RecoseleInc/E2E")
    
    # 获取原始数据集的所有列
    original_data = original_dataset['train'].to_pandas()
    
    # 添加新的处理列
    voice_parts = []
    text_parts = []
    
    for text in original_data['step1_text']:
        voice_part, text_part = process_text(text)
        voice_parts.append(voice_part)
        text_parts.append(text_part)
    
    # 添加新列到原始数据
    original_data['step1_voice_part'] = voice_parts
    original_data['step1_text_part'] = text_parts
    
    # 转换为Hugging Face Dataset格式
    print("Converting to Dataset format...")
    new_dataset = Dataset.from_pandas(original_data)
    
    # 推送到Hub
    print("Pushing to Hugging Face Hub...")
    new_dataset.push_to_hub(
        "RecoseleInc/E2E",
        split="train",
        commit_message="Add processed voice and text parts while preserving original columns"
    )
    
    print("Dataset successfully pushed to hub!")
    
    # 显示处理后的样本，包含所有列
    print("\nSample of processed data (first row):")
    print(new_dataset[0])

if __name__ == "__main__":
    main()
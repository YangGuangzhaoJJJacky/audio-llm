#!/home/recosele/miniconda3/envs/whisper——finetune_env/bin/python

from datasets import load_dataset
import pandas as pd
from datasets import Dataset, Audio, Features, Value
import datasets,aiohttp
from together import Together  
import re
from tqdm import tqdm
import os
import pykakasi
import asyncio
import concurrent.futures
from dotenv import load_dotenv
import numpy as np
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Optional

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
class APIRateLimiter:
    def __init__(self, max_requests: int = 1000, time_window: int = 60):
        """
        初始化速率限制器
        
        参数:
        - max_requests: 每个时间窗口允许的最大请求数
        - time_window: 时间窗口大小（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.semaphore = asyncio.Semaphore(100)  
    
    async def acquire(self):

        current_time = time.time()
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]

        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.time_window - current_time
            if sleep_time > 0:
                print(f"at limit,wait {sleep_time:.2f} s...")
                await asyncio.sleep(sleep_time)
        
        await self.semaphore.acquire()
        self.requests.append(time.time())
    
    def release(self):
        self.semaphore.release()

rate_limiter = APIRateLimiter(max_requests=1000, time_window=60)

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\sぁ-んァ-ン一-龥。、？！]", "", text) 
    text = re.sub(r"\s+", "", text)
    return text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def convert_numbers_to_kana(text: str) -> str:
    system_prompt = """
    これから提供するテキストを洗練させ、日本語の質問形式に整えてください。形式は以下の通りです：
          全体の質問形式は以下の要素で構成されます：

          {一部分の基本状況の説明} + [startofspeech] {残りの基本状況の説明と指示または質問内容} [endofspeech]
          以下は例です：「あなたは音声を理解できる対話アシスタントです。私の庭に芝生があります。[startofspeech]この芝生は二年前買ったものです。この芝生の手入れ方法を教えてください。[endofspeech]」

          なお、{基本状況の説明}と{残りの基本状況の説明と指示または質問内容}の内容の長さは同じとし、
          質問の内容を拡充・修正することができます。また、情報の一部は{指示または基本状況の説明}の中に反映させてください。
          {基本prompt}は自由生成してください。{残りの基本状況の説明と指示または基本状況の説明}の内容長すぎるしないでください。
          {基本状況の説明}と{残りの基本状況の説明と指示または質問内容}の内容のパランスは大切である。日本語以外のものを避けってください。
    """
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool,
                lambda: client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",  # 更改为 Together API 的模型
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<｜end▁of▁sentence｜>"],
                    stream=False
                )
            )
        result_text = response.choices[0].message.content.strip()
        return clean_text(result_text)
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error: {e}")
        return f"ERROR: {str(e)}" 
async def llm_convert(text):
    try:
        return await convert_numbers_to_kana(text)
    except Exception as e:
        print(f"Error in llm_convert: {e}")
        return f"ERROR: {str(e)}"

async def process_batch(batch_texts, max_concurrent: int = 50):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(text):
        async with semaphore:
            result = await llm_convert(text)
            return result
    
    tasks = [process_with_semaphore(text) for text in batch_texts]
    return await asyncio.gather(*tasks)

def get_processed_count(output_path):
    total_processed = 0
    last_shard_id = -1
    
    if os.path.exists(output_path):
        for dirname in os.listdir(output_path):
            if dirname.startswith('shard_'):
                shard_id = int(dirname.split('_')[1])
                shard_path = os.path.join(output_path, dirname)
                
                metadata_path = os.path.join(shard_path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        total_processed += metadata['sample_count']
                        last_shard_id = max(last_shard_id, shard_id)
    
    return total_processed, last_shard_id

def save_batch_data(processed_data, original_texts,  output_base_path, batch_id):

    shard_dir = os.path.join(output_base_path, f"shard_{batch_id:08d}")
    os.makedirs(shard_dir, exist_ok=True)
    features = Features({
        "original_text": Value("string"),
        "step1_text": Value("string"),
    })
    
    try:
        processed_dataset = Dataset.from_dict(
            {
                "original_text": original_texts,
                "step1_text": processed_data,
            },
            features=features
        )
        
        processed_dataset.save_to_disk(shard_dir)
        
        with open(os.path.join(shard_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "shard_id": batch_id,
                "sample_count": len(processed_data),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"保存分片 {batch_id} 时出错: {str(e)}")
        raise

def process_dataset(
    dataset_name="japanese-asr/ja_asr.reazon_speech_all", 
    output_path="processed_dataset_0",
    batch_size=4,
    max_samples=None,
    resume=False,
    cache_dir=None,
    num_proc=4,
    shard_size=1000,
    subset = 5
):
    try:
        if resume:
            processed_count, last_shard_id = get_processed_count(output_path)
            skip_samples = processed_count
            current_shard = last_shard_id + 1
            print(f"from checkpoint: finished {processed_count} ,start from {current_shard}")
        else:
            skip_samples = 0
            current_shard = 0
            processed_count = 0

        dataset = load_dataset(
            dataset_name, 
            streaming=True,
            cache_dir=cache_dir
        )
        
        os.makedirs(output_path, exist_ok=True)
        processed_data = []
        original_texts = []
        loop = asyncio.get_event_loop()

        if max_samples:
            total_samples = min(max_samples, skip_samples + max_samples)
        else:
            total_samples = None
            
        pbar = tqdm(total=total_samples, initial=processed_count,
                   desc="Processing dataset", unit="samples")
        
        skipped = 0
        for batch in dataset["train"].iter(batch_size):
            if resume and skipped < skip_samples:
                skipped += len(batch["input"])
                continue
                
            if max_samples and processed_count >= max_samples:
                print(f"\nat max: {max_samples}")
                break
            
            input_texts = batch["input"]
            processed_texts = loop.run_until_complete(process_batch(input_texts))
            
            
            processed_data.extend(processed_texts)
            original_texts.extend(input_texts)
            
            processed_count += len(input_texts)
            pbar.update(len(input_texts))
            
            if len(processed_data) >= shard_size:
                save_batch_data(
                    processed_data,
                    original_texts,
                    output_path,
                    current_shard
                )
                
                processed_data = []
                original_texts = []
                current_shard += 1
            
        if processed_data:
            save_batch_data(
                processed_data,
                original_texts,
                output_path,
                current_shard
            )
        
        print("\neverything done")

        
    except Exception as e:
        print(f"\n处理过程中出现错误: {str(e)}")
        raise
if __name__ == "__main__":
    # 设置参数
    subset = 5
    dataset_name = "LinhDuong/chatdoctor-200k"
    output_path = "/home/recosele/Development/End2End/dataset/step1"
    
    # 运行处理
    process_dataset(
        dataset_name=dataset_name,
        output_path=output_path,
        batch_size=100,           # 更大的批处理大小
        max_samples=None,        # 设置为None处理全部数据
        resume=True,             # 是否从断点继续
        cache_dir="./cache",     # 缓存目录
        num_proc=4,             # 并行处理数量
        shard_size=1000,       # 每个分片的样本数量
        subset = subset
    )
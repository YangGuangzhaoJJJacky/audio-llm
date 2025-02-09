from datasets import load_dataset, Dataset
import pyarrow.parquet as pq
import glob
import os

# 1. 首先导入所有parquet文件
parquet_files = sorted(glob.glob("/home/recosele/Development/End2End/dataset/step1parquet_dataset/train/*.parquet"))

# 2. 将所有parquet文件合并成一个dataset
# 方法1：使用datasets的load_dataset
dataset = load_dataset('parquet', 
                      data_files=parquet_files,
                      split='train')

# 或者方法2：手动读取并合并
# datasets = []
# for file in parquet_files:
#     table = pq.read_table(file)
#     dataset = Dataset.from_table(table)
#     datasets.append(dataset)
# dataset = concatenate_datasets(datasets)

# 3. 推送到hub
dataset.push_to_hub("RecoseleInc/E2E")  # private=False 如果想公开
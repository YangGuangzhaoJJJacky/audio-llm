from datasets import load_dataset, Dataset
import pyarrow.parquet as pq
import glob
import os

# Let's first print the files we find
parquet_files = sorted(glob.glob("/home/recosele/Development/End2End/dataset/step1/train/train*.parquet"))
print("Found parquet files:", parquet_files)

# If files are found, then proceed with loading
if parquet_files:
    dataset = load_dataset('parquet',
                          data_files=parquet_files,
                          split='train')
    print("Dataset loaded successfully")
    print("Dataset size:", len(dataset))
    # Push to hub
    dataset.push_to_hub("RecoseleInc/E2E")
else:
    print("No parquet files found in the specified directory")
    # Print the current working directory and list its contents
    print("Current directory:", os.getcwd())
    print("Directory contents:", os.listdir("/home/recosele/Development/End2End/dataset/step1/"))
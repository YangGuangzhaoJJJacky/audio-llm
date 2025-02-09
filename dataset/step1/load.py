from datasets import load_from_disk
from datasets import load_dataset

def load_and_print_dataset(dataset_path: str, split: str = "train", row_index: int = 0):
    """
    从本地磁盘加载 Hugging Face 数据集，并打印整体信息和指定行的数据
    :param dataset_path: 本地数据集路径 (如 "/path/to/dataset")
    :param split: 数据集的 split (如 "train", "test", "validation")，默认 "train"
    :param row_index: 需要查看的行索引，默认 0
    """
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    #dataset = load_dataset("parquet", data_files=dataset_path, split="train")


    # 打印数据集的整体信息
    print(f"=== 数据集: {dataset_path} ({split}) ===")
    print(f"数据量: {len(dataset)} 行")
    print(f"特征列: {dataset.column_names}")
    print(f"数据类型: {dataset.features}\n")

    # 打印指定行的数据
    if row_index >= len(dataset):
        raise IndexError(f"索引 {row_index} 超出范围，数据集长度为 {len(dataset)}")
    
    print(f"=== 第 {row_index} 行的数据 ===")
    for i in range(4):
        print(dataset[row_index+i])

# 示例调用：加载本地 `shard_0000` 并查看第 5 行
load_and_print_dataset("/home/recosele/Development/End2End/dataset/step1parquet_dataset/train", "train", 20)

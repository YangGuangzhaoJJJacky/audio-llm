import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import shutil, json
from huggingface_hub import HfApi, upload_file, upload_folder,upload_large_folder

# 配置路径和仓库信息
source_dir = Path(
    "/home/recosele/Development/End2End/dataset/step1"
)  # 原始分片存储目录
target_dir = Path(
    "/home/recosele/Development/End2End/dataset/step1parquet_dataset"
)  # 转换后的数据集目录
repo_id = "RecoseleInc/reazon_speech_all_gana"  # 替换为你的 Hugging Face 仓库 ID

# 创建目标目录结构
(target_dir / "train").mkdir(parents=True, exist_ok=True)

# 收集所有分片文件并按原始顺序排序
arrow_files = []
for shard_path in sorted(source_dir.glob("shard_*")):
    if shard_path.is_dir():
        arrow_files.extend(sorted(shard_path.glob("data-*.arrow")))

total_shards = len(arrow_files)
print(f"检测到总分片数: {total_shards}")

# 将每个分片转换为 Parquet 文件
for idx, arrow_file in enumerate(arrow_files):
    print(f"正在处理分片 {idx + 1}/{total_shards}: {arrow_file.name}")

    # 读取 Arrow 文件
    with open(arrow_file, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()

    # 保存为 Parquet 文件
    parquet_file_name = f"train-{idx:05d}-of-{total_shards:05d}.parquet"
    parquet_file_path = target_dir / "train" / parquet_file_name
    pq.write_table(table, parquet_file_path)

    print(f"已保存为: {parquet_file_path}")

# 处理配置文件（以第一个分片的配置文件为基础）
first_shard = next(source_dir.glob("shard_*"))
for config_file in ["dataset_info.json", "state.json", "metadata.json"]:
    if (first_shard / config_file).exists():
        shutil.copy(first_shard / config_file, target_dir)

# 更新 dataset_info.json 中的分片信息
dataset_info_path = target_dir / "dataset_info.json"
if dataset_info_path.exists():
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    
    # 更新每个 split 的分片信息
    for split_info in dataset_info.get("splits", {}).values():
        split_info["num_shards"] = total_shards
        # 清除可能需要重新计算的字段
        split_info.pop("shard_lengths", None)
        split_info.pop("num_bytes", None)
    
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)


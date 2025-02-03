#!/bin/bash

# 定义输入和输出目录
input_dir="wavs_22k"
output_dir="wavs"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 遍历输入目录中的所有文件
for input_file in "$input_dir"/*; do
	  # 获取文件的基本名称
	    base_name=$(basename "$input_file")
	      # 定义输出文件的路径
	        output_file="$output_dir/$base_name"
		  
		  # 使用 ffmpeg 转换音频采样率为 16kHz
		    ffmpeg -i "$input_file" -ar 16000 "$output_file"
	    done

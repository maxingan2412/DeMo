#!/bin/bash

# 你可以在运行脚本时传入 GPU ID，例如： ./run_with_timestamp.sh 1
gpu_id=${1:-0}  # 默认用 0 号卡，如果有参数就用传进来的

# 设置 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 构造输出路径：在 IDEA_RGBNT201 目录下创建时间戳子目录
output_dir="./DeMo_RGBNT201/${timestamp}"

# 创建目录
mkdir -p "$output_dir"

# 运行训练脚本，使用 config_file 并动态指定输出目录
python train_net.py --config_file configs/RGBNT201/DeMo.yml OUTPUT_DIR "$output_dir"

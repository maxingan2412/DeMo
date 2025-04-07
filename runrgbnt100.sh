#!/bin/bash

# 获取 GPU ID（默认是 0）
gpu_id=${1:-0}

# 获取实验名称（如果没给，就用 default）
exp_name=${2:-default}

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 构造输出路径（形如 my_exp_name_时间戳）
output_dir="./DeMo_RGBNT100/${exp_name}_${timestamp}"

# 创建输出目录
mkdir -p "$output_dir"

# 启动训练
python train_net.py --config_file configs/RGBNT100/DeMo.yml OUTPUT_DIR "$output_dir"

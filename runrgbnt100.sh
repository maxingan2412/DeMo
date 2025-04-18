#!/bin/bash

# 获取 GPU ID（默认是 0）
gpu_id=${1:-0}

# 获取实验名称（默认是 default）
exp_name=${2:-default}

# 获取 config 文件路径（默认是 configs/RGBNT100/DeMo.yml）
config_file=${3:-configs/RGBNT100/DeMo.yml}

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Using config: $config_file"

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 提取 dataset 和 model 名
dataset_name=$(basename $(dirname "$config_file"))  # RGBNT100
model_name=$(basename "$config_file" .yml)          # DeMo

# 构造输出路径：RGBNT100-DeMo/exp_name_时间戳
output_dir="./${dataset_name}-${model_name}/${exp_name}_${timestamp}"

# 创建输出目录
mkdir -p "$output_dir"

# 启动训练
python train_net.py --config_file "$config_file" OUTPUT_DIR "$output_dir" 2>&1 | tee "$output_dir/train.log"

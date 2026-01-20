#!/bin/bash
# PMOA 单模型运行脚本 - 可指定模型名称
# 用法: bash run_single_model.sh <model_name>
# 例如: bash run_single_model.sh Informer

export CUDA_VISIBLE_DEVICES=0

MODEL=${1:-Informer}

# 数据路径配置
ROOT_PATH="./data/"
DATA_PATH="pmoa"

echo "============================================"
echo "Running $MODEL on PMOA Classification Task"
echo "============================================"

python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model $MODEL \
  --data PMOA \
  --seq_len 64 \
  --enc_in 3 \
  --num_labels 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

echo "============================================"
echo "$MODEL completed!"
echo "============================================"

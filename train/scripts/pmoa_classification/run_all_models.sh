#!/bin/bash
# PMOA 多标签二分类任务 - 多模型运行脚本
# 预测任务: 后续k=8个事件是否在24小时内发生

export CUDA_VISIBLE_DEVICES=0

# 数据路径配置
ROOT_PATH="./data/"
DATA_PATH="pmoa"

# 通用参数
SEQ_LEN=64
ENC_IN=3
NUM_LABELS=8
BATCH_SIZE=32
TRAIN_EPOCHS=30
PATIENCE=10
LEARNING_RATE=0.001

echo "============================================"
echo "PMOA Multi-Label Classification Task"
echo "k=8 binary labels, H=24h threshold"
echo "============================================"

# ==================== Informer ====================
echo "[1/6] Running Informer..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model Informer \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

# ==================== TimesNet ====================
echo "[2/6] Running TimesNet..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model TimesNet \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --e_layers 2 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

# ==================== Autoformer ====================
echo "[3/6] Running Autoformer..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model Autoformer \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

# ==================== DLinear ====================
echo "[4/6] Running DLinear..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model DLinear \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

# ==================== PatchTST ====================
echo "[5/6] Running PatchTST..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model PatchTST \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --patch_len 8 \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

# ==================== iTransformer ====================
echo "[6/6] Running iTransformer..."
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id PMOA \
  --model iTransformer \
  --data PMOA \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --num_labels $NUM_LABELS \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --learning_rate $LEARNING_RATE \
  --des 'Exp' \
  --itr 1

echo "============================================"
echo "All models completed!"
echo "============================================"

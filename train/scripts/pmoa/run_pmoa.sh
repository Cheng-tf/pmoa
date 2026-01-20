#!/bin/bash

# ============================================
# PMOA事件发生时间预测任务运行脚本
# ============================================
# 使用标准long_term_forecast任务，支持多种模型
# 预测目标: 后续k个事件的发生时间 (小时)
# 评估指标: MSE, MAE
# ============================================

export CUDA_VISIBLE_DEVICES=0

# ============================================
# 数据预处理 (只需运行一次)
# ============================================
echo "=========================================="
echo "Step 1: Data Preprocessing"
echo "=========================================="

python data_provider/pmoa_preprocess.py \
    --input ../pmoa/tts_dataset.jsonl \
    --output ./dataset/pmoa \
    --k 8 \
    --seq_len 64 \
    --min_prefix_len 5

# ============================================
# 通用参数设置
# ============================================
SEQ_LEN=64
LABEL_LEN=32
PRED_LEN=8
ENC_IN=3
DEC_IN=3
C_OUT=3
BATCH_SIZE=64
TRAIN_EPOCHS=20
PATIENCE=5
LEARNING_RATE=0.001

# ============================================
# 运行不同模型
# ============================================

# --- Informer ---
echo ""
echo "=========================================="
echo "Training: Informer"
echo "=========================================="
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path pmoa \
    --model_id PMOA_Informer \
    --model Informer \
    --data PMOA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --factor 3 \
    --dropout 0.1 \
    --embed fixed \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --des Exp \
    --itr 1

# --- TimesNet ---
echo ""
echo "=========================================="
echo "Training: TimesNet"
echo "=========================================="
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path pmoa \
    --model_id PMOA_TimesNet \
    --model TimesNet \
    --data PMOA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --top_k 5 \
    --dropout 0.1 \
    --embed fixed \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --des Exp \
    --itr 1

# --- Autoformer ---
echo ""
echo "=========================================="
echo "Training: Autoformer"
echo "=========================================="
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path pmoa \
    --model_id PMOA_Autoformer \
    --model Autoformer \
    --data PMOA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --moving_avg 25 \
    --dropout 0.1 \
    --embed fixed \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --des Exp \
    --itr 1

# --- DLinear ---
echo ""
echo "=========================================="
echo "Training: DLinear"
echo "=========================================="
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path pmoa \
    --model_id PMOA_DLinear \
    --model DLinear \
    --data PMOA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --moving_avg 25 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --des Exp \
    --itr 1

# --- PatchTST ---
echo ""
echo "=========================================="
echo "Training: PatchTST"
echo "=========================================="
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path pmoa \
    --model_id PMOA_PatchTST \
    --model PatchTST \
    --data PMOA \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model 128 \
    --n_heads 16 \
    --e_layers 3 \
    --d_ff 256 \
    --dropout 0.2 \
    --embed fixed \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --des Exp \
    --itr 1

echo ""
echo "=========================================="
echo "All models training completed!"
echo "Results saved in ./results/"
echo "=========================================="

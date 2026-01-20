#!/bin/bash
# PMOA 多标签二分类任务 - 所有模型并行运行脚本
# 预测任务: 后续k=8个事件是否在24小时内发生
# 并行运行5个模型，使用A100显卡
#
# 使用方法:
#   bash scripts/pmoa_classification/run_all_models_par.sh
#
# 实时查看某个模型的日志:
#   tail -f logs/pmoa_classification/<MODEL>_<TIMESTAMP>.log

export CUDA_VISIBLE_DEVICES=0

# 数据路径配置
ROOT_PATH="./data/"
DATA_PATH="pmoa"

# 通用参数
SEQ_LEN=64
ENC_IN=3
NUM_LABELS=8
BATCH_SIZE=32
TRAIN_EPOCHS=10
PATIENCE=10
LEARNING_RATE=0.001

# 日志文件
LOG_DIR="./logs/pmoa_classification"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FINAL_LOG="${LOG_DIR}/all_models_results_${TIMESTAMP}.log"
PROGRESS_LOG="${LOG_DIR}/progress_${TIMESTAMP}.log"

# 创建命名管道用于实时输出
PIPE_DIR="/tmp/pmoa_pipes_$$"
mkdir -p $PIPE_DIR

echo "============================================"
echo "PMOA Multi-Label Classification Task"
echo "k=8 binary labels, H=24h threshold"
echo "Start time: $(date)"
echo "============================================"
echo ""
echo "Log directory: $LOG_DIR"
echo "Final results: $FINAL_LOG"
echo ""

# 写入最终日志头部
cat << EOF > $FINAL_LOG
============================================
PMOA Multi-Label Classification Task
k=8 binary labels, H=24h threshold
Start time: $(date)
============================================

EOF

# 支持classification的模型列表 (共22个)
MODELS=(
    "Informer"
    "TimesNet"
    "Autoformer"
    "DLinear"
    "PatchTST"
    "iTransformer"
    "Transformer"
    "Reformer"
    "FEDformer"
    "ETSformer"
    "Pyraformer"
    "Crossformer"
    "FiLM"
    "LightTS"
    "FreTS"
    "TimeMixer"
    "SegRNN"
    "TiDE"
    "Nonstationary_Transformer"
    "MSGNet"
    "TimeFilter"
    "WPMixer"
)

# 运行单个模型的函数
run_model() {
    local MODEL=$1
    local MODEL_LOG="${LOG_DIR}/${MODEL}_${TIMESTAMP}.log"
    local STATUS_FILE="${PIPE_DIR}/${MODEL}.status"
    
    # 根据模型设置特定参数
    local EXTRA_ARGS=""
    local D_MODEL=64
    local D_FF=256
    local E_LAYERS=2
    local D_LAYERS=1
    local N_HEADS=4
    
    case $MODEL in
        "TimesNet")
            D_MODEL=32
            D_FF=64
            EXTRA_ARGS="--top_k 3"
            ;;
        "PatchTST")
            D_MODEL=64
            D_FF=128
            EXTRA_ARGS="--patch_len 8"
            ;;
        "SegRNN")
            D_MODEL=64
            EXTRA_ARGS="--seg_len 8"
            ;;
        "TiDE")
            D_MODEL=64
            D_FF=256
            ;;
        "Crossformer")
            D_MODEL=64
            D_FF=128
            EXTRA_ARGS="--seg_len 8"
            ;;
        "Pyraformer")
            D_MODEL=64
            D_FF=128
            ;;
        "FiLM")
            D_MODEL=64
            D_FF=128
            ;;
        "TimeMixer")
            D_MODEL=32
            D_FF=64
            EXTRA_ARGS="--down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg"
            ;;
        "TimeFilter")
            D_MODEL=64
            D_FF=128
            ;;
        "MSGNet")
            D_MODEL=64
            D_FF=128
            ;;
        "WPMixer")
            D_MODEL=64
            D_FF=128
            ;;
        "FreTS")
            D_MODEL=64
            D_FF=128
            ;;
        "LightTS")
            D_MODEL=64
            D_FF=128
            ;;
        "ETSformer")
            D_MODEL=64
            D_FF=128
            ;;
        "Nonstationary_Transformer")
            D_MODEL=64
            D_FF=256
            EXTRA_ARGS="--p_hidden_dims 64 64 --p_hidden_layers 2"
            ;;
    esac
    
    # 运行模型，输出到日志文件
    {
        echo "============================================"
        echo "Model: $MODEL"
        echo "Start time: $(date)"
        echo "Parameters: d_model=$D_MODEL, d_ff=$D_FF, e_layers=$E_LAYERS"
        echo "Extra args: $EXTRA_ARGS"
        echo "============================================"
        echo ""
        
        python -u run.py \
            --task_name pmoa_classification \
            --is_training 1 \
            --root_path $ROOT_PATH \
            --data_path $DATA_PATH \
            --model_id PMOA \
            --model $MODEL \
            --data PMOA \
            --seq_len $SEQ_LEN \
            --enc_in $ENC_IN \
            --num_labels $NUM_LABELS \
            --e_layers $E_LAYERS \
            --d_layers $D_LAYERS \
            --d_model $D_MODEL \
            --d_ff $D_FF \
            --n_heads $N_HEADS \
            --batch_size $BATCH_SIZE \
            --train_epochs $TRAIN_EPOCHS \
            --patience $PATIENCE \
            --learning_rate $LEARNING_RATE \
            --des 'Exp' \
            --itr 1 \
            $EXTRA_ARGS
        
        local EXIT_CODE=$?
        
        echo ""
        echo "============================================"
        echo "Model: $MODEL"
        echo "End time: $(date)"
        echo "Exit code: $EXIT_CODE"
        echo "============================================"
        
        # 写入状态文件
        echo $EXIT_CODE > $STATUS_FILE
        
    } > $MODEL_LOG 2>&1
    
    # 读取退出状态
    local EXIT_CODE=$(cat $STATUS_FILE 2>/dev/null || echo "1")
    
    # 输出到终端
    if [ "$EXIT_CODE" -eq "0" ]; then
        echo "[$(date +%H:%M:%S)] ✓ $MODEL completed successfully"
    else
        echo "[$(date +%H:%M:%S)] ✗ $MODEL FAILED (exit code: $EXIT_CODE)"
    fi
    
    return $EXIT_CODE
}

# 并行运行一批模型
run_batch() {
    local start_idx=$1
    local batch_size=5
    local pids=()
    local models_in_batch=()
    
    echo ""
    echo "--------------------------------------------"
    
    # 启动一批模型
    for ((i=0; i<batch_size; i++)); do
        local idx=$((start_idx + i))
        if [ $idx -lt ${#MODELS[@]} ]; then
            local model=${MODELS[$idx]}
            models_in_batch+=("$model")
            echo "[$(date +%H:%M:%S)] Starting $model... (log: ${LOG_DIR}/${model}_${TIMESTAMP}.log)"
            run_model "$model" &
            pids+=($!)
        fi
    done
    
    echo "--------------------------------------------"
    echo "Batch models: ${models_in_batch[*]}"
    echo "Tip: Use 'tail -f ${LOG_DIR}/<MODEL>_${TIMESTAMP}.log' to view real-time log"
    echo "--------------------------------------------"
    
    # 等待所有进程完成
    local batch_success=0
    local batch_failed=0
    for i in "${!pids[@]}"; do
        wait ${pids[$i]}
        if [ $? -eq 0 ]; then
            ((batch_success++))
        else
            ((batch_failed++))
        fi
    done
    
    echo "--------------------------------------------"
    echo "Batch completed: $batch_success success, $batch_failed failed"
    echo "--------------------------------------------"
}

# 计算批次数
TOTAL_MODELS=${#MODELS[@]}
BATCH_SIZE=5
NUM_BATCHES=$(( (TOTAL_MODELS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Total models: $TOTAL_MODELS"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"

# 依次运行每批模型
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    start_idx=$((batch * BATCH_SIZE))
    echo ""
    echo "============================================"
    echo "Starting Batch $((batch + 1))/$NUM_BATCHES"
    echo "============================================"
    run_batch $start_idx
done

echo ""
echo "============================================"
echo "All models completed!"
echo "End time: $(date)"
echo "============================================"
echo ""

# 生成汇总结果
echo "Generating summary..."
echo ""

{
    echo ""
    echo "============================================"
    echo "DETAILED RESULTS"
    echo "============================================"
    echo ""
} >> $FINAL_LOG

# 收集每个模型的结果
SUCCESS_COUNT=0
FAILED_COUNT=0

for MODEL in "${MODELS[@]}"; do
    MODEL_LOG="${LOG_DIR}/${MODEL}_${TIMESTAMP}.log"
    STATUS_FILE="${PIPE_DIR}/${MODEL}.status"
    
    echo "=== $MODEL ===" >> $FINAL_LOG
    
    if [ -f "$MODEL_LOG" ]; then
        EXIT_CODE=$(cat $STATUS_FILE 2>/dev/null || echo "1")
        
        if [ "$EXIT_CODE" -eq "0" ]; then
            ((SUCCESS_COUNT++))
            echo "Status: SUCCESS" >> $FINAL_LOG
            # 提取结果
            grep -A 15 "PMOA Multi-Label Classification Results" $MODEL_LOG >> $FINAL_LOG 2>/dev/null || echo "Results section not found" >> $FINAL_LOG
        else
            ((FAILED_COUNT++))
            echo "Status: FAILED (exit code: $EXIT_CODE)" >> $FINAL_LOG
            echo "Last 20 lines of log:" >> $FINAL_LOG
            tail -20 $MODEL_LOG >> $FINAL_LOG 2>/dev/null
        fi
    else
        ((FAILED_COUNT++))
        echo "Status: NO LOG FILE" >> $FINAL_LOG
    fi
    echo "" >> $FINAL_LOG
done

# 生成汇总表格
{
    echo ""
    echo "============================================"
    echo "SUMMARY TABLE"
    echo "============================================"
    echo ""
    printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "Model" "Accuracy" "Precision" "Recall" "F1"
    printf "%-25s-+-%-10s-+-%-10s-+-%-10s-+-%-10s\n" "-------------------------" "----------" "----------" "----------" "----------"
} >> $FINAL_LOG

for MODEL in "${MODELS[@]}"; do
    MODEL_LOG="${LOG_DIR}/${MODEL}_${TIMESTAMP}.log"
    
    if [ -f "$MODEL_LOG" ]; then
        ACC=$(grep "^Accuracy:" $MODEL_LOG | tail -1 | awk '{print $2}')
        PREC=$(grep "^Precision:" $MODEL_LOG | tail -1 | awk '{print $2}')
        REC=$(grep "^Recall:" $MODEL_LOG | tail -1 | awk '{print $2}')
        F1=$(grep "^F1 Score:" $MODEL_LOG | tail -1 | awk '{print $3}')
        
        if [ -n "$ACC" ]; then
            printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "${ACC:-N/A}" "${PREC:-N/A}" "${REC:-N/A}" "${F1:-N/A}" >> $FINAL_LOG
        else
            printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "FAILED" "-" "-" "-" >> $FINAL_LOG
        fi
    else
        printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "NO LOG" "-" "-" "-" >> $FINAL_LOG
    fi
done

{
    echo ""
    echo "============================================"
    echo "FINAL STATISTICS"
    echo "============================================"
    echo "Total models: $TOTAL_MODELS"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAILED_COUNT"
    echo ""
    echo "End time: $(date)"
    echo "============================================"
} >> $FINAL_LOG

# 清理临时文件
rm -rf $PIPE_DIR

# 打印汇总表格到终端
echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo ""
printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "Model" "Accuracy" "Precision" "Recall" "F1"
printf "%-25s-+-%-10s-+-%-10s-+-%-10s-+-%-10s\n" "-------------------------" "----------" "----------" "----------" "----------"

for MODEL in "${MODELS[@]}"; do
    MODEL_LOG="${LOG_DIR}/${MODEL}_${TIMESTAMP}.log"
    
    if [ -f "$MODEL_LOG" ]; then
        ACC=$(grep "^Accuracy:" $MODEL_LOG | tail -1 | awk '{print $2}')
        PREC=$(grep "^Precision:" $MODEL_LOG | tail -1 | awk '{print $2}')
        REC=$(grep "^Recall:" $MODEL_LOG | tail -1 | awk '{print $2}')
        F1=$(grep "^F1 Score:" $MODEL_LOG | tail -1 | awk '{print $3}')
        
        if [ -n "$ACC" ]; then
            printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "${ACC:-N/A}" "${PREC:-N/A}" "${REC:-N/A}" "${F1:-N/A}"
        else
            printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "FAILED" "-" "-" "-"
        fi
    else
        printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL" "NO LOG" "-" "-" "-"
    fi
done

echo ""
echo "============================================"
echo "Total: $SUCCESS_COUNT success, $FAILED_COUNT failed"
echo "Results saved to: $FINAL_LOG"
echo "Individual logs in: $LOG_DIR"
echo "============================================"

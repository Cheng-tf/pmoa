# PMOA 多标签分类任务使用指南

## 任务描述

**预测任务**: 给定截止时间 t 之前的事件序列，预测后续 k=8 个事件是否会在 t+H=24小时内发生。

- **输入**: `[batch, seq_len, 3]` 时间序列特征
  - time_delta: 与前一个事件的时间差
  - cumulative_time: 累计时间
  - event_position: 归一化位置
- **输出**: `[batch, 8]` 多标签二分类 (8个独立的0/1标签)
- **损失函数**: BCEWithLogitsLoss
- **评估指标**: Accuracy, Precision, Recall, F1, AUC

## 使用步骤

### 1. 数据预处理

```bash
cd train

# 预处理PMOA数据集
python data_provider/pmoa_preprocess.py \
  --input ../pmoa/tts_dataset.jsonl \
  --output ./data/pmoa \
  --k 8 \
  --H 24 \
  --seq_len 64
```

预处理后的数据结构:
```
data/pmoa/
├── train_seq_x.npy      # 训练集输入 [N, 64, 3]
├── train_labels.npy     # 训练集标签 [N, 8]
├── train_stamp_x.npy    # 时间戳特征
├── val_seq_x.npy
├── val_labels.npy
├── val_stamp_x.npy
├── test_seq_x.npy
├── test_labels.npy
├── test_stamp_x.npy
├── metadata.json        # 元数据
└── scaler.pkl           # 标准化器
```

### 2. 运行模型

**方法1: 运行所有模型**
```bash
bash scripts/pmoa_classification/run_all_models.sh
```

**方法2: 运行单个模型**
```bash
# 使用默认Informer
bash scripts/pmoa_classification/run_single_model.sh

# 指定模型
bash scripts/pmoa_classification/run_single_model.sh TimesNet
bash scripts/pmoa_classification/run_single_model.sh Autoformer
bash scripts/pmoa_classification/run_single_model.sh DLinear
bash scripts/pmoa_classification/run_single_model.sh PatchTST
bash scripts/pmoa_classification/run_single_model.sh iTransformer
```

**方法3: 直接运行命令**
```bash
python -u run.py \
  --task_name pmoa_classification \
  --is_training 1 \
  --root_path ./data/ \
  --data_path pmoa \
  --model_id PMOA \
  --model Informer \
  --data PMOA \
  --seq_len 64 \
  --enc_in 3 \
  --num_labels 8 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 256 \
  --batch_size 32 \
  --train_epochs 30 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
```

### 3. 仅测试 (加载已训练模型)

```bash
python -u run.py \
  --task_name pmoa_classification \
  --is_training 0 \
  --root_path ./data/ \
  --data_path pmoa \
  --model_id PMOA \
  --model Informer \
  --data PMOA \
  --seq_len 64 \
  --enc_in 3 \
  --num_labels 8 \
  --des 'Exp'
```

## 支持的模型

| 模型 | 说明 |
|------|------|
| Informer | 长序列Transformer |
| TimesNet | 时间序列网络 |
| Autoformer | 自相关Transformer |
| DLinear | 线性模型 |
| PatchTST | Patch时间序列Transformer |
| iTransformer | 倒置Transformer |

其他可用模型: Transformer, Reformer, Crossformer, FEDformer, ETSformer, Pyraformer, SCINet, MICN, Koopa, SegRNN, FreTS, LightTS 等

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task_name` | 任务类型 | pmoa_classification |
| `--data` | 数据集类型 | PMOA |
| `--seq_len` | 输入序列长度 | 64 |
| `--enc_in` | 输入特征数 | 3 |
| `--num_labels` | 分类标签数(k) | 8 |
| `--batch_size` | 批次大小 | 32 |
| `--train_epochs` | 训练轮数 | 30 |
| `--patience` | 早停耐心值 | 10 |

## 输出结果

训练完成后，结果保存在:
- `checkpoints/`: 模型检查点
- `results/`: 评估指标文件
- `test_results/`: 预测结果 (predictions.npy, probabilities.npy, true_labels.npy)

结果示例:
```
============================================================
PMOA Multi-Label Classification Results
============================================================
Accuracy: 0.7523
Precision: 0.6845
Recall: 0.7012
F1 Score: 0.6928
AUC: 0.8134

Position-wise Metrics (k=8):
  Position 1: Acc=0.8123, F1=0.7856
  Position 2: Acc=0.7934, F1=0.7645
  ...
============================================================
```

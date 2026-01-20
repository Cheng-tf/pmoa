# PMOA-TTS 数据集介绍

## 概述

**PMOA-TTS** (Textual Time Series from PubMed Open Access) 是从 12.4 万份 PubMed 公开临床病例报告中提取的结构化文本时间序列数据集。

## 数据规模

| 数据集 | 样本数 | 格式 |
|--------|--------|------|
| train | 124,349 | .jsonl / .parquet |
| case_study_100 | 88 | .jsonl / .parquet |
| case_study_25k_L33 | 24,746 | .jsonl / .parquet |
| case_study_25k_DSR1 | 24,746 | .jsonl / .parquet |

## 数据结构

每条记录包含 **6 个字段**：

```json
{
  "pmc_id": "PMC006xxxxxx",
  "case_report_id": "PMC6417290",
  "textual_timeseries": [
    {"event": "admitted to hospital", "time": 0},
    {"event": "blood test performed", "time": 6},
    {"event": "discharged", "time": 120}
  ],
  "demographics": {
    "age": "53.0",
    "sex": "Male",
    "ethnicity": "Not Specified"
  },
  "diagnoses": ["Diabetes mellitus", "Hypertension"],
  "death_info": {
    "observed_time": 4320.0,
    "death_event_indicator": 0
  }
}
```

## 关键特征

### 1. 是否多变量？

**是** - 多维度数据集：
- 时间序列维度：事件时间序列 `{event, time}`
- 静态特征维度：年龄、性别、种族
- 诊断维度：诊断标签列表
- 标签维度：生存结果

### 2. 时间步长

| 属性 | 值 |
|------|-----|
| **时间单位** | 小时 (hours) |
| **时间基准** | 入院时刻 = 0 |
| **时间范围** | -34,560 ~ +13,140 小时 |
| **间隔特点** | **不规则间隔** (事件驱动) |

时间值示例：
- `time: -8760` → 入院前 1 年
- `time: 0` → 入院时刻
- `time: 24` → 入院后 1 天
- `time: 168` → 入院后 1 周

### 3. 序列长度

| 统计量 | 值 |
|--------|-----|
| 平均事件数/患者 | 55.25 |
| 最小事件数 | 1 |
| 最大事件数 | 138 |

## 预测任务定义

**任务**: 给定截止时间 t 之前的事件前缀，预测后续 k 个事件的发生时间

**设置**:
- `k = 8` (预测后续8个事件)
- `H = 24h` (用于评估: 是否在24h内发生)

**输入**: 时间 t 之前的事件时间特征序列
**输出**: 8 个连续值 (后续事件的发生时间，单位: 小时)
**评估指标**: MSE, MAE

## 转换为时序预测格式

预处理后的特征 (每个时间步):
1. `time_delta`: 与前一个事件的时间差 (小时)
2. `cumulative_time`: 从序列开始的累计时间 (小时)
3. `event_position`: 事件在序列中的归一化位置

## 文件路径

```
pmoa/
├── tts_dataset.jsonl                    # 训练集 (124K)
├── tts_dataset_case_study_100.jsonl     # 小规模测试 (88)
├── tts_dataset_case_study_25k_L33.jsonl # 中规模 (25K)
└── data/
    └── *.parquet                        # Parquet格式副本
```

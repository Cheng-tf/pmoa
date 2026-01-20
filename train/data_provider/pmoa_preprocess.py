"""
PMOA数据集预处理脚本 (多标签二分类任务)

将PMOA的文本事件时间序列转换为分类任务格式

预测任务: 给定截止时间t之前的事件序列，预测后续k个事件是否会在t+H小时内发生
- k = 8 (预测后续8个事件)
- H = 24h (判断阈值)
- 输出: 8个二分类标签 (0/1)
"""

import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle


def load_pmoa_jsonl(file_path):
    """加载PMOA JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_samples_for_classification(data, k=8, H=24, seq_len=64, min_prefix_len=5):
    """
    创建用于多标签二分类的样本

    输入特征 (每个时间步):
    - time_delta: 与前一个事件的时间差 (小时)
    - cumulative_time: 从序列开始的累计时间 (小时)
    - event_position: 事件计数 (归一化位置)

    预测目标:
    - k个二分类标签: 每个后续事件是否在H小时内发生 (0/1)

    Args:
        data: PMOA数据列表
        k: 预测后续k个事件
        H: 判断阈值 (小时)
        seq_len: 输入序列长度
        min_prefix_len: 最小前缀长度

    Returns:
        samples字典
    """
    all_seq_x = []      # [N, seq_len, num_features]
    all_labels = []     # [N, k] - 二分类标签
    all_stamp_x = []    # [N, seq_len, stamp_dim]
    all_case_ids = []

    num_features = 3  # time_delta, cumulative_time, event_position

    for record in tqdm(data, desc="创建样本"):
        events = record['textual_timeseries']
        events = sorted(events, key=lambda x: x['time'])
        n_events = len(events)

        if n_events < min_prefix_len + k:
            continue

        # 提取时间序列
        times = np.array([e['time'] for e in events], dtype=np.float32)

        # 计算时间差
        time_deltas = np.diff(times, prepend=times[0])
        time_deltas[0] = 0

        # 对于每个可能的截止位置
        for cut_idx in range(min_prefix_len, n_events - k):
            # 输入序列: cut_idx之前的事件
            start_idx = max(0, cut_idx - seq_len)
            input_times = times[start_idx:cut_idx]
            input_deltas = time_deltas[start_idx:cut_idx]

            # 截止时间
            t = times[cut_idx - 1]

            # 后续k个事件的时间 (相对于截止时间t)
            future_times = times[cut_idx:cut_idx + k] - t

            # 生成二分类标签: 是否在H小时内发生
            labels = (future_times <= H).astype(np.float32)

            # 构建输入特征
            actual_len = len(input_times)
            seq_x = np.zeros((seq_len, num_features), dtype=np.float32)

            # 填充特征 (右对齐，padding在左边)
            offset = seq_len - actual_len
            seq_x[offset:, 0] = input_deltas  # time_delta
            seq_x[offset:, 1] = input_times - input_times[0]  # cumulative_time
            seq_x[offset:, 2] = np.arange(actual_len) / actual_len  # event_position

            # 时间戳特征
            stamp_x = np.zeros((seq_len, 4), dtype=np.float32)
            stamp_x[offset:, 0] = (input_times - t) / 24  # 相对截止时间 (天)
            stamp_x[offset:, 1] = input_deltas / 24  # 时间差 (天)
            # stamp_x的其他维度保持为0

            all_seq_x.append(seq_x)
            all_labels.append(labels)
            all_stamp_x.append(stamp_x)
            all_case_ids.append(record['case_report_id'])

    return {
        'seq_x': np.array(all_seq_x),
        'labels': np.array(all_labels),
        'stamp_x': np.array(all_stamp_x),
        'case_ids': all_case_ids
    }


def preprocess_pmoa(
    input_file,
    output_dir,
    k=8,
    H=24,
    seq_len=64,
    min_prefix_len=5,
    train_ratio=0.7,
    val_ratio=0.1
):
    """
    预处理PMOA数据集为多标签二分类格式

    Args:
        input_file: PMOA JSONL文件路径
        output_dir: 输出目录
        k: 预测后续k个事件
        H: 判断阈值 (小时)
        seq_len: 输入序列长度
        min_prefix_len: 最小前缀长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载数据: {input_file}")
    data = load_pmoa_jsonl(input_file)
    print(f"总记录数: {len(data)}")

    print(f"创建样本 (k={k}, H={H}h)...")
    samples = create_samples_for_classification(data, k=k, H=H, seq_len=seq_len, min_prefix_len=min_prefix_len)

    n_samples = len(samples['seq_x'])
    print(f"总样本数: {n_samples}")

    # 按case_id分组，确保同一患者的样本在同一个split
    case_ids = list(set(samples['case_ids']))
    np.random.seed(42)
    np.random.shuffle(case_ids)

    n_train_cases = int(len(case_ids) * train_ratio)
    n_val_cases = int(len(case_ids) * val_ratio)

    train_cases = set(case_ids[:n_train_cases])
    val_cases = set(case_ids[n_train_cases:n_train_cases + n_val_cases])
    test_cases = set(case_ids[n_train_cases + n_val_cases:])

    # 创建索引
    train_idx = [i for i, cid in enumerate(samples['case_ids']) if cid in train_cases]
    val_idx = [i for i, cid in enumerate(samples['case_ids']) if cid in val_cases]
    test_idx = [i for i, cid in enumerate(samples['case_ids']) if cid in test_cases]

    print(f"训练集样本数: {len(train_idx)}")
    print(f"验证集样本数: {len(val_idx)}")
    print(f"测试集样本数: {len(test_idx)}")

    # 标准化输入特征 (仅在训练集上fit)
    scaler = StandardScaler()
    train_x = samples['seq_x'][train_idx].reshape(-1, samples['seq_x'].shape[-1])
    # 只对非零部分进行fit (避免padding影响)
    non_zero_mask = np.any(train_x != 0, axis=1)
    if non_zero_mask.sum() > 0:
        scaler.fit(train_x[non_zero_mask])
    else:
        scaler.fit(train_x)

    # 保存各个split
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        seq_x = samples['seq_x'][idx]
        labels = samples['labels'][idx]
        stamp_x = samples['stamp_x'][idx]

        # 标准化输入
        orig_shape = seq_x.shape
        seq_x_flat = seq_x.reshape(-1, orig_shape[-1])
        seq_x_scaled = scaler.transform(seq_x_flat).reshape(orig_shape)

        np.save(os.path.join(output_dir, f'{split}_seq_x.npy'), seq_x_scaled.astype(np.float32))
        np.save(os.path.join(output_dir, f'{split}_labels.npy'), labels.astype(np.float32))
        np.save(os.path.join(output_dir, f'{split}_stamp_x.npy'), stamp_x.astype(np.float32))

    # 保存元数据
    metadata = {
        'k': k,
        'H': H,
        'seq_len': seq_len,
        'num_features': samples['seq_x'].shape[-1],
        'stamp_dim': samples['stamp_x'].shape[-1],
        'num_classes': k,  # 8个二分类
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx)
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n数据已保存到: {output_dir}")

    # 打印标签分布
    all_labels = samples['labels']
    print(f"\n标签分布 (k={k}, H={H}h):")
    print(f"  总正样本比例: {all_labels.mean():.3f}")
    for i in range(k):
        pos_ratio = all_labels[:, i].mean()
        print(f"  位置{i+1}: 正样本比例 = {pos_ratio:.3f}")

    return metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预处理PMOA数据集 (多标签二分类)')
    parser.add_argument('--input', type=str, required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--k', type=int, default=8, help='预测后续k个事件')
    parser.add_argument('--H', type=int, default=24, help='判断阈值 (小时)')
    parser.add_argument('--seq_len', type=int, default=64, help='输入序列长度')
    parser.add_argument('--min_prefix_len', type=int, default=5, help='最小前缀长度')

    args = parser.parse_args()

    preprocess_pmoa(
        input_file=args.input,
        output_dir=args.output,
        k=args.k,
        H=args.H,
        seq_len=args.seq_len,
        min_prefix_len=args.min_prefix_len
    )

"""
PMOA数据集加载器 (多标签二分类任务)

返回格式:
- batch_x: [batch, seq_len, num_features] 输入时间序列
- labels: [batch, k] 多标签二分类目标
- padding_mask: [batch, seq_len] padding掩码

预测任务: 后续k个事件是否在H小时内发生 (8个独立的0/1标签)
"""

import os
import json
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PMOADataset(Dataset):
    """
    PMOA多标签二分类数据集

    适配代码库的classification任务格式
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='pmoa',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        Args:
            args: 参数对象
            root_path: 数据根目录
            flag: 'train', 'val', 'test', 'TRAIN', 'TEST'
            data_path: 数据子目录名
            其他参数为兼容性保留
        """
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale

        # 统一flag格式
        if flag.upper() == 'TRAIN':
            self.flag = 'train'
        elif flag.upper() == 'TEST':
            self.flag = 'test'
        else:
            self.flag = flag.lower()

        # 序列长度参数
        if size is not None:
            self.seq_len = size[0]
        else:
            self.seq_len = getattr(args, 'seq_len', 64)

        self.__read_data__()

    def __read_data__(self):
        """读取预处理后的数据"""
        data_dir = os.path.join(self.root_path, self.data_path)

        # 加载元数据
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'k': 8, 'num_features': 3}

        # 根据flag选择数据集
        prefix = self.flag

        # 加载数据
        self.seq_x = np.load(os.path.join(data_dir, f'{prefix}_seq_x.npy'))
        self.labels = np.load(os.path.join(data_dir, f'{prefix}_labels.npy'))
        self.stamp_x = np.load(os.path.join(data_dir, f'{prefix}_stamp_x.npy'))

        # 获取数据维度
        self.num_features = self.seq_x.shape[-1]
        self.num_classes = self.labels.shape[-1]  # k=8
        self.max_seq_len = self.seq_x.shape[1]

        # 用于兼容classification任务的属性
        self.feature_df = type('obj', (object,), {'shape': [len(self.seq_x) * self.max_seq_len, self.num_features]})()
        self.class_names = [f'pos_{i}' for i in range(self.num_classes)]

        print(f"{self.flag}: samples={len(self.seq_x)}, "
              f"seq_len={self.max_seq_len}, features={self.num_features}, "
              f"num_labels={self.num_classes}")

    def __getitem__(self, index):
        """
        返回一个样本

        Returns:
            batch_x: [seq_len, num_features] 输入序列
            labels: [k] 多标签目标
            padding_mask: [seq_len] padding掩码 (1=有效, 0=padding)
        """
        # 输入序列
        batch_x = self.seq_x[index]  # [seq_len, num_features]

        # 标签
        labels = self.labels[index]  # [k]

        # 创建padding mask (非零位置为有效)
        # 检查每个时间步是否全为0
        padding_mask = np.any(batch_x != 0, axis=-1).astype(np.float32)

        return (
            torch.from_numpy(batch_x.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
            torch.from_numpy(padding_mask)
        )

    def __len__(self):
        return len(self.seq_x)

    def inverse_transform(self, data):
        """逆变换 (如果需要)"""
        return data

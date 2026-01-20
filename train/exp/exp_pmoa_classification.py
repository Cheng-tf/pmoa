"""
PMOA多标签二分类实验类

预测任务: 给定截止时间t之前的事件序列，预测后续k个事件是否会在t+H小时内发生
- 输入: [batch, seq_len, num_features] 时间序列
- 输出: [batch, k] 多标签二分类 (k=8)
- 损失函数: BCEWithLogitsLoss
- 评估指标: Accuracy, F1, AUC
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')


class Exp_PMOA_Classification(Exp_Basic):
    """PMOA多标签二分类实验"""

    def __init__(self, args):
        super(Exp_PMOA_Classification, self).__init__(args)

    def _build_model(self):
        """构建模型"""
        # 设置为分类任务
        self.args.task_name = 'classification'
        self.args.num_class = self.args.num_labels  # k=8

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """获取数据"""
        # 临时恢复task_name为pmoa_classification，以使用正确的data_provider分支
        original_task_name = self.args.task_name
        self.args.task_name = 'pmoa_classification'
        data_set, data_loader = data_provider(self.args, flag)
        self.args.task_name = original_task_name  # 恢复为classification（模型需要）
        return data_set, data_loader

    def _select_optimizer(self):
        """选择优化器"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """选择损失函数 - 多标签二分类使用BCEWithLogitsLoss"""
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """验证"""
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, labels, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                labels = labels.float().to(self.device)

                # 模型前向传播 (classification模式)
                outputs = self.model(batch_x, padding_mask, None, None)

                # 计算损失
                loss = criterion(outputs, labels)
                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(labels.detach().cpu())

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        # 计算指标
        metrics = self._compute_metrics(preds, trues)

        self.model.train()
        return total_loss, metrics

    def _compute_metrics(self, preds, trues):
        """计算评估指标"""
        probs = torch.sigmoid(preds).numpy()
        predictions = (probs > 0.5).astype(int)
        trues_np = trues.numpy()

        # 整体指标
        acc = accuracy_score(trues_np.flatten(), predictions.flatten())
        precision = precision_score(trues_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
        recall = recall_score(trues_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
        f1 = f1_score(trues_np.flatten(), predictions.flatten(), average='macro', zero_division=0)

        # AUC
        try:
            auc = roc_auc_score(trues_np.flatten(), probs.flatten())
        except:
            auc = 0.0

        # 每个位置的指标
        position_acc = []
        position_f1 = []
        for i in range(trues_np.shape[1]):
            pos_acc = accuracy_score(trues_np[:, i], predictions[:, i])
            pos_f1 = f1_score(trues_np[:, i], predictions[:, i], zero_division=0)
            position_acc.append(pos_acc)
            position_f1.append(pos_f1)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'position_acc': position_acc,
            'position_f1': position_f1
        }

    def train(self, setting):
        """训练"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, labels, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                labels = labels.float().to(self.device)

                # 前向传播
                outputs = self.model(batch_x, padding_mask, None, None)

                # 计算损失
                loss = criterion(outputs, labels)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Vali Loss: {vali_loss:.4f} Vali Acc: {vali_metrics['accuracy']:.4f} Vali F1: {vali_metrics['f1']:.4f} | "
                  f"Test Loss: {test_loss:.4f} Test Acc: {test_metrics['accuracy']:.4f} Test F1: {test_metrics['f1']:.4f}")

            # 使用验证集F1进行早停
            early_stopping(-vali_metrics['f1'], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """测试"""
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, labels, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach().cpu())
                trues.append(labels.detach().cpu())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # 计算指标
        metrics = self._compute_metrics(preds, trues)

        # 保存结果
        result_folder = './results/' + setting + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        print('\n' + '='*60)
        print('PMOA Multi-Label Classification Results')
        print('='*60)
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1 Score: {metrics["f1"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        print(f'\nPosition-wise Metrics (k={len(metrics["position_acc"])}):')
        for i in range(len(metrics["position_acc"])):
            print(f'  Position {i+1}: Acc={metrics["position_acc"][i]:.4f}, F1={metrics["position_f1"][i]:.4f}')
        print('='*60)

        # 保存到文件
        file_name = 'result_pmoa_classification.txt'
        with open(os.path.join(result_folder, file_name), 'a') as f:
            f.write(setting + "\n")
            f.write(f'Accuracy: {metrics["accuracy"]:.4f}\n')
            f.write(f'Precision: {metrics["precision"]:.4f}\n')
            f.write(f'Recall: {metrics["recall"]:.4f}\n')
            f.write(f'F1 Score: {metrics["f1"]:.4f}\n')
            f.write(f'AUC: {metrics["auc"]:.4f}\n')
            f.write(f'Position Accuracy: {metrics["position_acc"]}\n')
            f.write(f'Position F1: {metrics["position_f1"]}\n')
            f.write('\n')

        # 保存预测结果
        probs = torch.sigmoid(preds).numpy()
        predictions = (probs > 0.5).astype(int)
        np.save(os.path.join(folder_path, 'predictions.npy'), predictions)
        np.save(os.path.join(folder_path, 'probabilities.npy'), probs)
        np.save(os.path.join(folder_path, 'true_labels.npy'), trues.numpy())

        return metrics

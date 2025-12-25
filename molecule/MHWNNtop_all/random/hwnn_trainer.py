import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import pickle
import torch.nn.functional as F
from datetime import datetime


class HWNNTrainer:
    """HWNN超图神经网络的训练器（支持节点任务/图任务/分子任务，适配注意力池化，包含RMSE指标）"""

    def __init__(self, model, processed_data, hyper_data, device, log_path,
                 save_model=True,
                 # 注意力权重保存参数（适配注意力池化）
                 save_attn_weights=False,
                 attn_save_interval=10):
        self.model = model
        self.model = self.model.to(device)
        self.processed_data = processed_data
        self.hyper_data = hyper_data
        self.device = device
        self.log_path = log_path
        self.save_model = save_model

        # 注意力权重保存配置（增强：区分卷积注意力和池化注意力）
        self.save_attn_weights = save_attn_weights
        self.attn_save_interval = attn_save_interval
        self.attn_weights_history = {}  # 存储：{SMILES: [{epoch, attn_weights, pred, target, valid_mask}]}
        # 新增：判断模型是否启用注意力池化
        self.use_pool_attention = hasattr(self.model, 'use_pool_attention') and self.model.use_pool_attention

        # 任务类型判断
        self.task_type = hyper_data['task_type']
        self.is_mol_task = hyper_data.get('is_mol_task', False)

        # 结果记录
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_metric = float('inf') if 'regression' in self.task_type else -float('inf')
        self.best_model_state = None
        self.optimizer = None
        self.scheduler = None

        # 创建模型保存目录
        if save_model:
            self.model_save_path = os.path.join(log_path, "models")
            os.makedirs(self.model_save_path, exist_ok=True)

        # 分子任务+注意力池化：创建注意力权重保存目录（增强校验）
        if self.save_attn_weights and self.is_mol_task and self.use_pool_attention:
            self.attn_save_path = os.path.join(log_path, "attention_weights")
            os.makedirs(self.attn_save_path, exist_ok=True)
            logging.info(
                f"[注意力保存] 已启用池化注意力权重保存（间隔{attn_save_interval}轮），路径：{self.attn_save_path}")
        elif self.save_attn_weights and (not self.is_mol_task or not self.use_pool_attention):
            logging.warning(f"[注意力保存] 跳过保存：仅分子任务+启用注意力池化时支持权重保存")

    def set_optimizer(self, lr=0.001, weight_decay=1e-5, grad_clip=0, lr_scheduler=True, lr_scheduler_patience=5):
        """配置优化器和学习率调度器"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        if lr_scheduler:
            # 移除了verbose=True参数，解决警告问题
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min' if 'regression' in self.task_type else 'max',
                factor=0.5, patience=lr_scheduler_patience
            )

    def _get_node_task_batch(self):
        """获取节点任务的完整数据（无batch）"""
        theta_snapshots = [snap['Theta'] for snap in self.hyper_data['snapshots']]
        return {
            'features': self.processed_data['features'],
            'theta_snapshots': theta_snapshots,
            'labels': self.processed_data['labels'],
            'train_mask': self.processed_data['train_mask'],
            'val_mask': self.processed_data['val_mask'],
            'test_mask': self.processed_data['test_mask']
        }

    def _get_graph_batch(self, split, batch_indices):
        """获取图任务/分子任务的batch数据"""
        batch_data = [self.hyper_data['hyper_data'][split][i] for i in batch_indices]
        graphs = [self.processed_data[split][i] for i in batch_indices]

        # 提取超图拉普拉斯矩阵快照（按batch中最大节点数padding）
        theta_snapshots_list = []
        max_nodes = max(data['num_nodes'] for data in batch_data)
        min_snapshots = min(len(data['snapshots']) for data in batch_data)
        if min_snapshots == 0:
            raise ValueError("批量数据中存在无快照的图，请检查超图构建逻辑")

        for data in batch_data:
            snapshots = [snap['Theta'] for snap in data['snapshots'][:min_snapshots]]
            padded_snapshots = []
            for theta in snapshots:
                pad_size = max_nodes - theta.size(0)
                if pad_size > 0:
                    theta = F.pad(theta, (0, pad_size, 0, pad_size), mode='constant', value=0)
                padded_snapshots.append(theta)
            theta_snapshots_list.append(padded_snapshots)

        # 转置为 [snapshot_idx][batch_idx][nodes][nodes]
        num_snapshots = min_snapshots
        theta_snapshots = []
        for s_idx in range(num_snapshots):
            batch_theta_list = []
            for b_idx in range(len(batch_indices)):
                if s_idx < len(theta_snapshots_list[b_idx]):
                    batch_theta_list.append(theta_snapshots_list[b_idx][s_idx])
                else:
                    zero_theta = torch.zeros(max_nodes, max_nodes, device=self.device)
                    batch_theta_list.append(zero_theta)
            batch_theta = torch.stack(batch_theta_list).to(self.device)
            theta_snapshots.append(batch_theta)

        # 提取节点特征+有效原子掩码
        features_list = []
        valid_masks_list = []
        for graph in graphs:
            feat = graph['features']
            pad_size = max_nodes - feat.size(0)
            if pad_size > 0:
                feat = F.pad(feat, (0, 0, 0, pad_size), mode='constant', value=0)
            features_list.append(feat)

            if self.is_mol_task:
                mask = graph['valid_mask']
                if pad_size > 0:
                    mask = F.pad(mask, (0, pad_size), mode='constant', value=False)
                valid_masks_list.append(mask)

        batch_features = torch.stack(features_list).to(self.device)
        labels = torch.stack([torch.tensor(data['label'], device=self.device) for data in batch_data])

        result = {
            'features': batch_features,
            'theta_snapshots': theta_snapshots,
            'labels': labels,
            'num_nodes': max_nodes,
            'is_mol_task': self.is_mol_task,
            'batch_indices': batch_indices,
            'split': split,
            # 传递原始图的有效原子数（用于过滤padding权重）
            'raw_valid_counts': [graph['valid_mask'].sum().item() for graph in graphs] if self.is_mol_task else None
        }

        if self.is_mol_task and valid_masks_list:
            result['valid_masks'] = torch.stack(valid_masks_list).to(self.device)

        return result

    def _compute_metrics(self, pred, target):
        """根据任务类型计算评估指标（包含RMSE指标）"""
        if 'classification' in self.task_type:
            pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
            target_labels = target.cpu().numpy()
            accuracy = accuracy_score(target_labels, pred_labels)
            macro_f1 = f1_score(target_labels, pred_labels, average='macro')
            return {'accuracy': accuracy, 'macro-f1': macro_f1}, macro_f1
        else:  # 回归任务（含分子溶解度预测）：包含RMSE
            pred = pred.squeeze().cpu().detach().numpy()
            target = target.cpu().numpy()
            mse = mean_squared_error(target, pred)
            mae = mean_absolute_error(target, pred)
            r2 = r2_score(target, pred)
            rmse = np.sqrt(mse)  # 基于MSE计算RMSE
            return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}, mse

    def _save_attention_weights(self, epoch, batch_data, attn_weights):
        """保存注意力池化权重（过滤填充原子，仅保留有效权重）"""
        # 前置校验：仅在满足所有条件时保存
        if not (self.save_attn_weights
                and self.is_mol_task
                and self.use_pool_attention
                and epoch % self.attn_save_interval == 0):
            return

        # 提取关键信息
        batch_indices = batch_data['batch_indices']
        split = batch_data['split']
        valid_masks = batch_data.get('valid_masks')  # (B, N)：有效原子掩码
        raw_valid_counts = batch_data.get('raw_valid_counts')  # 原始有效原子数（无padding）
        pred = batch_data.get('pred')  # 模型预测值
        target = batch_data.get('labels')  # 真实标签

        # 校验必要数据
        if None in [valid_masks, raw_valid_counts, pred, target]:
            logging.warning(f"[注意力保存] 跳过第{epoch}轮：batch数据缺少必要字段")
            return

        # 提取当前batch的SMILES（用于唯一标识分子）
        try:
            smiles_list = [self.hyper_data['hyper_data'][split][i]['smiles'] for i in batch_indices]
        except KeyError:
            logging.error(f"[注意力保存] 无法获取SMILES：hyper_data['hyper_data']['{split}']缺少'smiles'键")
            return

        # 逐个分子保存注意力权重（过滤padding）
        for idx, (smiles, raw_count) in enumerate(zip(smiles_list, raw_valid_counts)):
            # 获取当前分子的注意力权重和有效掩码
            mol_attn = attn_weights[idx].cpu().detach().numpy()  # (N,)：含padding的权重
            mol_valid_mask = valid_masks[idx].cpu().numpy()  # (N,)：有效原子掩码
            mol_pred = pred[idx].item()  # 预测值
            mol_target = target[idx].item()  # 真实值

            # 过滤padding：仅保留有效原子的权重（与原始分子原子数一致）
            valid_attn = mol_attn[mol_valid_mask]  # (raw_count,)：过滤后有效权重
            if len(valid_attn) != raw_count:
                logging.warning(f"[注意力保存] 分子{smiles}权重过滤异常：预期{raw_count}个，实际{len(valid_attn)}个")
                continue

            # 存入历史记录
            if smiles not in self.attn_weights_history:
                self.attn_weights_history[smiles] = []
            self.attn_weights_history[smiles].append({
                'epoch': epoch,
                'split': split,  # 数据集类型（train/val/test）
                'attn_weights': valid_attn,  # 过滤后的有效原子权重
                'pred': mol_pred,
                'target': mol_target,
                'valid_atom_count': raw_count,  # 有效原子数
                'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # 按间隔批量保存历史（减少IO操作）
        if epoch % (self.attn_save_interval * 10) == 0:
            save_filename = f"attn_weights_epoch_{epoch}_{split}.pkl"
            save_path = os.path.join(self.attn_save_path, save_filename)
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.attn_weights_history, f)
                logging.info(
                    f"[注意力保存] 第{epoch}轮权重历史已保存至：{save_path}（含{len(self.attn_weights_history)}个分子）")
            except Exception as e:
                logging.error(f"[注意力保存] 保存失败：{str(e)}", exc_info=True)

    def train(self, epochs=100, patience=20, batch_size=32, eval_interval=1, verbose=True, record_attn_every=10):
        """训练模型（包含RMSE日志展示）"""
        self.model.train()
        early_stop_counter = 0
        best_epoch = 0

        # 准备数据
        if self.task_type == 'node_classification':
            data = self._get_node_task_batch()
            train_mask = data['train_mask']
            val_mask = data['val_mask']
            train_labels = data['labels'][train_mask]
            val_labels = data['labels'][val_mask]
        else:
            num_train = len(self.hyper_data['hyper_data']['train'])
            train_indices = np.arange(num_train)
            num_batches = (num_train + batch_size - 1) // batch_size

        # 导入datetime用于时间戳（注意力保存需要）
        from datetime import datetime

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            if self.task_type == 'node_classification':
                # 节点任务训练
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.model.loss(pred[train_mask], train_labels)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_train_loss = loss.item()
            else:
                # 图任务/分子任务训练
                np.random.shuffle(train_indices)
                for b_idx in range(num_batches):
                    batch_start = b_idx * batch_size
                    batch_end = min((b_idx + 1) * batch_size, num_train)
                    batch_indices = train_indices[batch_start:batch_end]
                    batch_data = self._get_graph_batch('train', batch_indices)

                    self.optimizer.zero_grad()
                    # 模型前向：会自动将注意力权重存入 batch_data['node_att_weights']
                    pred = self.model(batch_data)
                    loss = self.model.loss(pred, batch_data['labels'])
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                    # 累加训练损失
                    total_train_loss += loss.item() * len(batch_indices)

                    # 获取注意力权重并保存（训练集）
                    if self.use_pool_attention and 'node_att_weights' in batch_data:
                        batch_data['pred'] = pred  # 存入预测值
                        self._save_attention_weights(epoch, batch_data, batch_data['node_att_weights'])

                total_train_loss /= num_train  # 计算平均损失

            # 记录训练损失
            self.train_losses.append(total_train_loss)

            # 评估与注意力记录（按间隔）
            if epoch % eval_interval == 0:
                # 训练集指标计算
                if self.task_type == 'node_classification':
                    train_pred = pred[train_mask].detach()
                    train_metrics, _ = self._compute_metrics(train_pred, train_labels)
                else:
                    _, train_metrics, _ = self.evaluate(split='train', batch_size=batch_size, current_epoch=epoch)
                self.train_metrics.append(train_metrics)

                # 验证集评估
                val_loss, val_metrics, val_metric = self.evaluate(
                    split='val', batch_size=batch_size, current_epoch=epoch
                )
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)

                # 打印训练日志（包含RMSE展示）
                if verbose:
                    if 'classification' in self.task_type:
                        logging.info(
                            f"Epoch {epoch}/{epochs} | "
                            f"Train Loss: {total_train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Train Acc: {train_metrics['accuracy']:.4f} | "
                            f"Val Acc: {val_metrics['accuracy']:.4f} | "
                            f"Val F1: {val_metrics['macro-f1']:.4f}"
                        )
                    else:
                        logging.info(
                            f"Epoch {epoch}/{epochs} | "
                            f"Train Loss: {total_train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Train MSE: {train_metrics['mse']:.4f} | "
                            f"Train RMSE: {train_metrics['rmse']:.4f} | "  # 训练集RMSE
                            f"Val MSE: {val_metrics['mse']:.4f} | "
                            f"Val RMSE: {val_metrics['rmse']:.4f} | "  # 验证集RMSE
                            f"Val R²: {val_metrics['r2']:.4f}"
                        )

                # 学习率调度：添加学习率变化监控
                if self.scheduler is not None:
                    # 记录调整前的学习率
                    old_lr = self.scheduler.get_last_lr()
                    self.scheduler.step(val_metric)
                    # 获取调整后的学习率
                    new_lr = self.scheduler.get_last_lr()
                    # 当学习率发生变化时打印信息
                    if new_lr != old_lr:
                        logging.info(f"学习率调整: 从 {old_lr[0]:.6f} 调整为 {new_lr[0]:.6f}")

                # 保存最佳模型
                is_better = (val_metric < self.best_val_metric) if 'regression' in self.task_type else (
                            val_metric > self.best_val_metric)
                if is_better:
                    self.best_val_metric = val_metric
                    self.best_model_state = self.model.state_dict()
                    best_epoch = epoch
                    early_stop_counter = 0
                    if self.save_model:
                        torch.save(self.best_model_state, os.path.join(self.model_save_path, "best_model.pth"))
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logging.info(f"早停触发：在第{epoch}轮，最佳模型在第{best_epoch}轮")
                        break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.best_val_metric

    def evaluate(self, split='test', batch_size=32, current_epoch=None):
        """评估模型性能（支持RMSE计算）"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            if self.task_type == 'node_classification':
                # 节点任务评估
                data = self._get_node_task_batch()
                mask = data[f'{split}_mask']
                labels = data['labels'][mask]
                pred = self.model(data)[mask]
                loss = self.model.loss(pred, labels).item()
                metrics, metric = self._compute_metrics(pred, labels)
                return loss, metrics, metric
            else:
                # 图任务/分子任务评估
                num_samples = len(self.hyper_data['hyper_data'][split])
                indices = np.arange(num_samples)
                num_batches = (num_samples + batch_size - 1) // batch_size

                for b_idx in range(num_batches):
                    batch_start = b_idx * batch_size
                    batch_end = min((b_idx + 1) * batch_size, num_samples)
                    batch_indices = indices[batch_start:batch_end]
                    batch_data = self._get_graph_batch(split, batch_indices)

                    # 模型前向：获取预测值和注意力权重
                    pred = self.model(batch_data)
                    loss = self.model.loss(pred, batch_data['labels']).item()
                    total_loss += loss * len(batch_indices)

                    # 评估阶段保存注意力权重（val/test集）
                    if self.use_pool_attention and 'node_att_weights' in batch_data and current_epoch is not None:
                        batch_data['pred'] = pred  # 存入预测值
                        self._save_attention_weights(current_epoch, batch_data, batch_data['node_att_weights'])

                    # 收集预测值和真实标签
                    all_preds.append(pred.view(-1).cpu().detach())
                    all_targets.append(batch_data['labels'].view(-1).cpu())

                # 计算平均损失和评估指标
                total_loss /= num_samples
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                metrics, metric = self._compute_metrics(all_preds, all_targets)

                return total_loss, metrics, metric

    def evaluate_test_rmse(self, batch_size=32, verbose=True):
        """
        专门计算测试集的RMSE（及其他回归指标）
        :param batch_size: 评估时的batch大小
        :param verbose: 是否打印详细结果
        :return: 测试集RMSE值（float），及完整指标字典（含mse/mae/r2/rmse）
        """
        # 1. 校验任务类型（仅回归任务有RMSE）
        if 'regression' not in self.task_type:
            logging.error(f"[测试集RMSE计算] 任务类型{self.task_type}不支持RMSE，仅回归任务可用")
            return None, None

        # 2. 调用evaluate方法评估测试集（current_epoch=None：不保存注意力权重）
        test_loss, test_metrics, _ = self.evaluate(
            split='test',  # 明确评估测试集
            batch_size=batch_size,
            current_epoch=None  # 测试阶段无需记录epoch对应的注意力权重
        )

        # 3. 提取并打印RMSE（核心指标）
        test_rmse = test_metrics.get('rmse', None)
        if test_rmse is None:
            logging.error(f"[测试集RMSE计算] 未从测试集指标中获取到RMSE，请检查_compute_metrics方法")
            return None, test_metrics

        # 4. 打印详细测试结果（含RMSE）
        if verbose:
            logging.info("=" * 50)
            logging.info("          测试集最终评估结果（回归任务）          ")
            logging.info("=" * 50)
            logging.info(f"测试集损失（Loss）: {test_loss:.4f}")
            logging.info(f"测试集均方误差（MSE）: {test_metrics['mse']:.4f}")
            logging.info(f"测试集平均绝对误差（MAE）: {test_metrics['mae']:.4f}")
            logging.info(f"测试集决定系数（R²）: {test_metrics['r2']:.4f}")
            logging.info(f"测试集均方根误差（RMSE）: {test_rmse:.4f}")  # 重点输出RMSE
            logging.info("=" * 50)

        return test_rmse, test_metrics

    def plot_learning_curves(self, save_path=None, show_rmse=True):
        """绘制学习曲线（支持RMSE可视化）"""
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        attn_tag = "（启用注意力池化）" if self.use_pool_attention else "（无注意力池化）"
        plt.title(f'Learning Curves {attn_tag}')
        plt.legend()

        # 指标曲线（支持MSE/RMSE切换）
        plt.subplot(1, 2, 2)
        if 'classification' in self.task_type:
            train_metrics = [m['accuracy'] for m in self.train_metrics]
            val_metrics = [m['accuracy'] for m in self.val_metrics]
            plt.ylabel('Accuracy')
        else:
            if show_rmse:
                # 显示RMSE
                train_metrics = [m['rmse'] for m in self.train_metrics]
                val_metrics = [m['rmse'] for m in self.val_metrics]
                plt.ylabel('RMSE')
            else:
                # 显示MSE
                train_metrics = [m['mse'] for m in self.train_metrics]
                val_metrics = [m['mse'] for m in self.val_metrics]
                plt.ylabel('MSE')

        plt.plot(range(1, len(train_metrics) + 1), train_metrics, label='Train Metric')
        plt.plot(range(1, len(val_metrics) + 1), val_metrics, label='Validation Metric')
        plt.xlabel('Epoch')
        metric_tag = "RMSE" if (show_rmse and 'regression' in self.task_type) else "Metric"
        plt.title(f'{metric_tag} Curves {attn_tag}')
        plt.legend()

        plt.tight_layout()
        if save_path:
            save_suffix = f'_rmse_{show_rmse}_attn_pool_{self.use_pool_attention}.png'
            save_path = save_path.replace('.png', save_suffix)
            plt.savefig(save_path)
            logging.info(f"学习曲线已保存至：{save_path}")
        else:
            plt.show()

    def plot_attn_distribution(self, smiles, save_path=None):
        """绘制单个分子的注意力权重分布（仅分子任务+注意力池化）"""
        if not (self.is_mol_task and self.use_pool_attention and smiles in self.attn_weights_history):
            logging.warning(f"[注意力可视化] 无法绘制：分子{smiles}无注意力记录或未启用注意力池化")
            return

        # 提取该分子的所有注意力记录（按epoch排序）
        attn_records = sorted(self.attn_weights_history[smiles], key=lambda x: x['epoch'])
        if len(attn_records) == 0:
            logging.warning(f"[注意力可视化] 分子{smiles}无有效注意力记录")
            return

        # 绘制注意力权重分布（按epoch分个子图）
        num_plots = min(len(attn_records), 5)  # 最多显示5个epoch的结果
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]  # 兼容单子图情况

        for idx, record in enumerate(attn_records[:num_plots]):
            epoch = record['epoch']
            attn_weights = record['attn_weights']
            valid_count = record['valid_atom_count']
            pred = record['pred']
            target = record['target']

            # 绘制条形图（原子索引为x轴，注意力权重为y轴）
            axes[idx].bar(range(valid_count), attn_weights, color='#1f77b4', alpha=0.7)
            axes[idx].set_xlabel('Atom Index')
            axes[idx].set_ylabel('Attention Weight')
            axes[idx].set_title(f'SMILES: {smiles[:20]}...\nEpoch {epoch} | Pred: {pred:.3f} | Target: {target:.3f}')
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            save_path = os.path.join(self.log_path, f"attn_dist_{smiles[:10]}_epoch_{epoch}.png")
            plt.savefig(save_path)
            logging.info(f"分子注意力分布已保存至：{save_path}")
        else:
            plt.show()

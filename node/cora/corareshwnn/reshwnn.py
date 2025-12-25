import os
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
import math
import matplotlib.pyplot as plt
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import json

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 固定数据分割的seed（仅用42）
DATA_SPLIT_SEED = 42


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run HWNN (10 global seeds, fixed data split seed=42).")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--filters", type=int, default=128, help="Filters (neurons) in convolution. ")
    parser.add_argument("--dropout", type=float, default=0.01, help="Dropout probability. ")
    parser.add_argument("--start-global-seed", type=int, default=16, help="Start global random seed. ")
    parser.add_argument("--num-global-seeds", type=int, default=10, help="Number of global seeds to train.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Adam weight decay.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in the HWNN.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha parameter for initial residual connection.")
    parser.add_argument("--lamda", type=float, default=1, help="Lambda parameter for beta calculation.")
    parser.add_argument("--use-mixed-precision", action="store_true", default=False,
                        help="Enable mixed precision training.")
    return parser.parse_args()


# 全局seed固定函数（控制模型初始化、dropout等）
def set_global_seed(seed):
    """固定全局随机种子（模型相关随机过程）"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)


class HWNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device, K1=2, K2=2, approx=False, data=None):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(ncount))
        self.par = torch.nn.Parameter(torch.Tensor(K1 + K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(self.device)
        Theta = data.hypergraph_snapshot[snap_index]["Theta"].to(self.device)
        Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.ncount, device=self.device)
            Theta_mul = torch.eye(self.ncount, device=self.device)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly += self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.ncount, device=self.device)
            Theta_mul = torch.eye(self.ncount, device=self.device)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t
                poly_t += self.par[ind] * Theta_mul

            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            wavelets = data.hypergraph_snapshot[snap_index]["wavelets"].to(self.device)
            wavelets_inverse = data.hypergraph_snapshot[snap_index]["wavelets_inv"].to(self.device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        return local_fea_1


class HWNN(torch.nn.Module):
    def __init__(self, args, ncount, feature_number, class_number, device, data):
        super(HWNN, self).__init__()
        self.args = args
        self.ncount = ncount
        self.feature_number = feature_number
        self.class_number = class_number
        self.device = device
        self.data = data
        self.num_layers = args.num_layers
        self.hyper_snapshot_num = len(data.hypergraph_snapshot)
        logging.info(f"There are {self.hyper_snapshot_num} hypergraphs")

        self.setup_layers()
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def setup_layers(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.feature_number, self.args.filters))
        try:
            pre_trained = np.load('M_layer_1.npy')
            self.layers[0].weight.data = torch.from_numpy(pre_trained).float()
        except FileNotFoundError:
            logging.warning("Layer 1 pre-trained weights not found, using random initialization.")

        for i in range(1, self.num_layers + 1):
            layer = HWNNLayer(self.args.filters, self.args.filters, self.ncount, self.device, K1=2, K2=2, approx=True,
                              data=self.data)
            self.layers.append(layer)
            try:
                pre_trained = np.load(f'M_layer_{i + 1}.npy')
                layer.weight_matrix.data = torch.from_numpy(pre_trained).float()
            except FileNotFoundError:
                logging.warning(f"Layer {i + 1} pre-trained weights not found, using random initialization.")

        self.layers.append(torch.nn.Linear(self.args.filters, self.class_number))

    def forward(self, features):
        features = features.to(self.device)
        h = F.dropout(features, self.args.dropout, training=self.training)
        h = F.relu(self.layers[0](h))
        h0 = h

        for i, layer in enumerate(self.layers[1:-1]):
            h = F.dropout(h, self.args.dropout, training=self.training)
            beta = math.log(self.args.lamda / (i + 1) + 1)
            alpha = self.args.alpha

            with autocast():
                h_layer = F.relu(layer(h, 0, self.data))

            h = (1 - alpha) * h_layer + alpha * h0
            h = (1 - beta) * h + beta * torch.matmul(h, layer.weight_matrix)

        h = F.dropout(h, self.args.dropout, training=self.training)
        h = self.layers[-1](h)
        h = F.log_softmax(h, dim=1)
        return h


class HWNNTrainer(object):
    def __init__(self, args, global_seed, features, target, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.global_seed = global_seed  # 当前训练的全局seed
        self.data = data
        self.features = features
        self.ncount = self.features.size(0)
        self.feature_number = self.features.size(1)
        self.target = target.to(self.device)
        self.class_number = self.data.class_num

        self.setup_model()
        self.train_valid_test_split()  # 恢复train/valid/test分割

        # 指标记录列表 - 恢复test相关，保留泛化误差
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.test_micro_f1s = []
        self.test_macro_f1s = []
        self.test_precision_scores = []
        self.test_recall_scores = []
        self.generalization_errors = []  # 泛化误差记录

        # 最佳模型指标（恢复test相关，新增泛化误差）
        self.best_metrics = {
            'global_seed': global_seed,
            'data_split_seed': DATA_SPLIT_SEED,
            'valid_accuracy': 0.0,
            'valid_loss': float('inf'),  # 初始化为无穷大
            'generalization_error': float('inf'),  # 泛化误差
            'test_accuracy': 0.0,
            'test_micro_f1': 0.0,
            'test_macro_f1': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'best_epoch': 0
        }
        self.best_model_path = f"best_model_global_seed_{global_seed}.pt"
        self.scaler = GradScaler(enabled=args.use_mixed_precision)

        # 用于跟踪每个epoch的最佳valacc和epoch
        self.current_best_valacc = 0.0
        self.current_best_epoch = 0
        self.current_best_valloss = float('inf')
        self.current_best_gen_error = float('inf')

    def setup_model(self):
        self.model = HWNN(self.args, self.ncount, self.feature_number, self.class_number, self.device, self.data).to(
            self.device)

    def train_valid_test_split(self):
        """恢复train/valid/test分割（固定DATA_SPLIT_SEED=42）"""
        nodes = list(range(self.ncount))
        # 强制使用DATA_SPLIT_SEED，不随全局seed变化
        train_nodes, temp_nodes = train_test_split(nodes, test_size=0.4, random_state=DATA_SPLIT_SEED)
        valid_nodes, test_nodes = train_test_split(temp_nodes, test_size=0.5, random_state=DATA_SPLIT_SEED)
        self.train_nodes = torch.LongTensor(train_nodes).to(self.device)
        self.valid_nodes = torch.LongTensor(valid_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(test_nodes).to(self.device)
        logging.info(
            f"Data split done with fixed seed {DATA_SPLIT_SEED} (train:{len(train_nodes)}, valid:{len(valid_nodes)}, test:{len(test_nodes)})")

    def compute_metrics(self, prediction):
        """恢复所有指标计算（train/valid/test + 泛化误差）"""
        with torch.no_grad():
            # 计算损失
            loss_train = F.cross_entropy(prediction[self.train_nodes], self.target[self.train_nodes])
            loss_valid = F.cross_entropy(prediction[self.valid_nodes], self.target[self.valid_nodes])
            loss_test = F.cross_entropy(prediction[self.test_nodes], self.target[self.test_nodes])

            # 计算准确率
            _, train_pred = prediction[self.train_nodes].max(dim=1)
            accuracy_train = train_pred.eq(self.target[self.train_nodes]).sum().item() / max(1, len(self.train_nodes))
            _, valid_pred = prediction[self.valid_nodes].max(dim=1)
            valid_accuracy = valid_pred.eq(self.target[self.valid_nodes]).sum().item() / max(1, len(self.valid_nodes))
            _, test_pred = prediction[self.test_nodes].max(dim=1)
            accuracy_test = test_pred.eq(self.target[self.test_nodes]).sum().item() / max(1, len(self.test_nodes))

            # 计算泛化误差（|训练准确率 - 验证准确率|）
            generalization_error = abs(accuracy_train - valid_accuracy)

            # 计算测试集其他指标
            test_target_cpu = self.target[self.test_nodes].cpu()
            test_pred_cpu = test_pred.cpu()
            test_micro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='micro', zero_division=1)
            test_macro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)
            test_precision = metrics.precision_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)
            test_recall = metrics.recall_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)

        return {
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'loss_test': loss_test,
            'acc_train': accuracy_train,
            'acc_valid': valid_accuracy,
            'acc_test': accuracy_test,
            'generalization_error': generalization_error,
            'micro_f1': test_micro_f1,
            'macro_f1': test_macro_f1,
            'precision': test_precision,
            'recall': test_recall
        }

    def fit(self):
        """单次全局seed训练，模型选择基于valid，同时评估test"""
        logging.info(f"\n===== Start training for Global Seed {self.global_seed} =====")
        logging.info(f"Training strategy: Validation-based model selection (Val Acc > Val Loss > Gen Error)")
        logging.info(f"Test set evaluation: Computed for best model (selected by validation metrics)")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()

            # 训练步骤 - 保持训练模式
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.use_mixed_precision):
                prediction = self.model(self.features)
                loss_train = F.cross_entropy(prediction[self.train_nodes], self.target[self.train_nodes])

            if self.args.use_mixed_precision:
                self.scaler.scale(loss_train).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_train.backward()
                self.optimizer.step()

            # 评估步骤 - 切换到评估模式
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(self.features)
                metrics_dict = self.compute_metrics(prediction)

            # 更新当前最佳指标跟踪（保留泛化误差）
            if metrics_dict['acc_valid'] > self.current_best_valacc:
                self.current_best_valacc = metrics_dict['acc_valid']
                self.current_best_epoch = epoch
                self.current_best_valloss = metrics_dict['loss_valid'].item()
                self.current_best_gen_error = metrics_dict['generalization_error']
            elif metrics_dict['acc_valid'] == self.current_best_valacc:
                if metrics_dict['loss_valid'].item() < self.current_best_valloss:
                    self.current_best_valloss = metrics_dict['loss_valid'].item()
                    self.current_best_epoch = epoch
                    self.current_best_gen_error = metrics_dict['generalization_error']
                elif metrics_dict['loss_valid'].item() == self.current_best_valloss:
                    if metrics_dict['generalization_error'] < self.current_best_gen_error:
                        self.current_best_gen_error = metrics_dict['generalization_error']
                        self.current_best_epoch = epoch

            # ========== 核心逻辑：模型选择仅基于valid（val acc > val loss > 泛化误差） ==========
            should_update = False
            update_reason = ""

            # 第一层：验证准确率更高
            if metrics_dict['acc_valid'] > self.best_metrics['valid_accuracy']:
                # 1. 保存更新前的旧值
                old_best_val_acc = self.best_metrics['valid_accuracy']
                # 2. 构造更新理由（基于旧值）
                update_reason = f"Better val acc: {metrics_dict['acc_valid']:.4f} > {old_best_val_acc:.4f}"
                # 3. 更新最佳指标（包含test）
                self.best_metrics.update({
                    'valid_accuracy': metrics_dict['acc_valid'],
                    'valid_loss': metrics_dict['loss_valid'].item(),
                    'generalization_error': metrics_dict['generalization_error'],
                    'test_accuracy': metrics_dict['acc_test'],
                    'test_micro_f1': metrics_dict['micro_f1'],
                    'test_macro_f1': metrics_dict['macro_f1'],
                    'test_precision': metrics_dict['precision'],
                    'test_recall': metrics_dict['recall'],
                    'best_epoch': epoch
                })
                should_update = True

            # 第二层：验证准确率相同，验证损失更低
            elif metrics_dict['acc_valid'] == self.best_metrics['valid_accuracy']:
                if metrics_dict['loss_valid'].item() < self.best_metrics['valid_loss']:
                    # 1. 保存更新前的旧值
                    old_best_val_loss = self.best_metrics['valid_loss']
                    # 2. 构造更新理由（基于旧值）
                    update_reason = f"Same val acc ({metrics_dict['acc_valid']:.4f}) but better loss: {metrics_dict['loss_valid']:.4f} < {old_best_val_loss:.4f}"
                    # 3. 更新最佳指标（包含test）
                    self.best_metrics.update({
                        'valid_loss': metrics_dict['loss_valid'].item(),
                        'generalization_error': metrics_dict['generalization_error'],
                        'test_accuracy': metrics_dict['acc_test'],
                        'test_micro_f1': metrics_dict['micro_f1'],
                        'test_macro_f1': metrics_dict['macro_f1'],
                        'test_precision': metrics_dict['precision'],
                        'test_recall': metrics_dict['recall'],
                        'best_epoch': epoch
                    })
                    should_update = True

                # 第三层：验证准确率和损失都相同，泛化误差更小
                elif metrics_dict['loss_valid'].item() == self.best_metrics['valid_loss']:
                    if metrics_dict['generalization_error'] < self.best_metrics['generalization_error']:
                        # 1. 保存更新前的旧值
                        old_best_gen_error = self.best_metrics['generalization_error']
                        # 2. 构造更新理由（基于旧值）
                        update_reason = f"Same val acc ({metrics_dict['acc_valid']:.4f}) and val loss ({metrics_dict['loss_valid']:.4f}) but better gen error: {metrics_dict['generalization_error']:.4f} < {old_best_gen_error:.4f}"
                        # 3. 更新最佳指标（包含test）
                        self.best_metrics.update({
                            'generalization_error': metrics_dict['generalization_error'],
                            'test_accuracy': metrics_dict['acc_test'],
                            'test_micro_f1': metrics_dict['micro_f1'],
                            'test_macro_f1': metrics_dict['macro_f1'],
                            'test_precision': metrics_dict['precision'],
                            'test_recall': metrics_dict['recall'],
                            'best_epoch': epoch
                        })
                        should_update = True

            if should_update:
                logging.info(f"Epoch {epoch}: {update_reason}")
                # 保存最佳模型
                self.save_best_model()

            # 记录所有指标（恢复test，保留泛化误差）
            self.train_losses.append(metrics_dict['loss_train'].item())
            self.valid_losses.append(metrics_dict['loss_valid'].item())
            self.test_losses.append(metrics_dict['loss_test'].item())
            self.train_accuracies.append(metrics_dict['acc_train'])
            self.valid_accuracies.append(metrics_dict['acc_valid'])
            self.test_accuracies.append(metrics_dict['acc_test'])
            self.test_micro_f1s.append(metrics_dict['micro_f1'])
            self.test_macro_f1s.append(metrics_dict['macro_f1'])
            self.test_precision_scores.append(metrics_dict['precision'])
            self.test_recall_scores.append(metrics_dict['recall'])
            self.generalization_errors.append(metrics_dict['generalization_error'])

            # 打印epoch日志（恢复test，新增泛化误差）
            epoch_time = time.time() - epoch_start_time
            log_msg = (
                f"Epoch:{epoch:3d}/{self.args.epochs} | "
                f"Train Loss:{metrics_dict['loss_train']:.4f} | "
                f"Valid Loss:{metrics_dict['loss_valid']:.4f} | "
                f"Test Loss:{metrics_dict['loss_test']:.4f} | "
                f"Train Acc:{metrics_dict['acc_train']:.4f} | "
                f"Valid Acc:{metrics_dict['acc_valid']:.4f} | "
                f"Test Acc:{metrics_dict['acc_test']:.4f} | "
                f"Gen Error:{metrics_dict['generalization_error']:.4f} | "
                f"Current Best Val Acc:{self.current_best_valacc:.4f} (epoch {self.current_best_epoch})"
            )

            if should_update:
                log_msg += f" | [BEST MODEL UPDATED]"

            log_msg += f" | Time:{epoch_time:.2f}s"
            logging.info(log_msg)

        # 训练结束，恢复训练模式
        self.model.train()

        # 最终验证：加载最佳模型并重新评估（包含test）
        self.final_validation()

        # 绘制学习曲线（恢复test，新增泛化误差）
        self.plot_learning_curves()

        # 输出当前全局seed的最佳指标
        logging.info(f"\n{'=' * 60}")
        logging.info(f"BEST MODEL SUMMARY - Global Seed: {self.global_seed}")
        logging.info(f"{'=' * 60}")
        logging.info(f"Data Split Seed: {DATA_SPLIT_SEED}")
        logging.info(f"Best Epoch: {self.best_metrics['best_epoch']}")
        logging.info(f"Valid Accuracy: {self.best_metrics['valid_accuracy']:.4f}")
        logging.info(f"Valid Loss: {self.best_metrics['valid_loss']:.4f}")
        logging.info(f"Generalization Error (|train_acc - val_acc|): {self.best_metrics['generalization_error']:.4f}")
        logging.info(f"Test Accuracy: {self.best_metrics['test_accuracy']:.4f}")
        logging.info(f"Test Micro F1: {self.best_metrics['test_micro_f1']:.4f}")
        logging.info(f"Test Macro F1: {self.best_metrics['test_macro_f1']:.4f}")
        logging.info(f"Test Precision: {self.best_metrics['test_precision']:.4f}")
        logging.info(f"Test Recall: {self.best_metrics['test_recall']:.4f}")
        logging.info(f"{'=' * 60}")

        return self.best_metrics

    def final_validation(self):
        """加载最佳模型并进行最终验证（包含test）"""
        try:
            # 加载保存的最佳模型
            checkpoint = torch.load(self.best_model_path, map_location=self.device)

            # 确保模型是评估模式
            self.model.eval()

            # 加载状态字典
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 进行最终评估
            with torch.no_grad():
                prediction = self.model(self.features)
                metrics_dict = self.compute_metrics(prediction)

            # 验证与保存的指标是否一致
            logging.info(f"\n===== Final Validation =====")
            logging.info(f"Saved Valid Accuracy: {checkpoint['best_metrics']['valid_accuracy']:.4f}")
            logging.info(f"Re-evaluated Valid Accuracy: {metrics_dict['acc_valid']:.4f}")
            logging.info(f"Saved Test Accuracy: {checkpoint['best_metrics']['test_accuracy']:.4f}")
            logging.info(f"Re-evaluated Test Accuracy: {metrics_dict['acc_test']:.4f}")
            logging.info(f"Saved Generalization Error: {checkpoint['best_metrics']['generalization_error']:.4f}")
            logging.info(f"Re-evaluated Generalization Error: {metrics_dict['generalization_error']:.4f}")

            # 更新为最新评估结果
            if abs(metrics_dict['acc_valid'] - checkpoint['best_metrics']['valid_accuracy']) > 0.0001:
                logging.warning(
                    f"Validation accuracy mismatch: saved={checkpoint['best_metrics']['valid_accuracy']:.4f}, re-evaluated={metrics_dict['acc_valid']:.4f}")

            self.best_metrics.update({
                'valid_accuracy': metrics_dict['acc_valid'],
                'valid_loss': metrics_dict['loss_valid'].item(),
                'generalization_error': metrics_dict['generalization_error'],
                'test_accuracy': metrics_dict['acc_test'],
                'test_micro_f1': metrics_dict['micro_f1'],
                'test_macro_f1': metrics_dict['macro_f1'],
                'test_precision': metrics_dict['precision'],
                'test_recall': metrics_dict['recall']
            })

        except Exception as e:
            logging.error(f"Error in final validation: {e}")

    def save_best_model(self):
        """保存当前全局seed的最佳模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_metrics': self.best_metrics.copy(),
            'args': self.args,
            'global_seed': self.global_seed,
            'data_split_seed': DATA_SPLIT_SEED
        }

        torch.save(checkpoint, self.best_model_path)
        logging.info(f"Best model saved to {self.best_model_path} (epoch {self.best_metrics['best_epoch']})")

    def plot_learning_curves(self):
        """绘制学习曲线（恢复test，新增泛化误差）"""
        epochs = range(1, self.args.epochs + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss')
        axes[0, 0].plot(epochs, self.valid_losses, label='Valid Loss')
        axes[0, 0].plot(epochs, self.test_losses, label='Test Loss', alpha=0.7)
        axes[0, 0].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5,
                           label=f'Best Epoch ({self.best_metrics["best_epoch"]})')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'Loss Curves (Seed: {self.global_seed})')

        # 准确率曲线
        axes[0, 1].plot(epochs, self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(epochs, self.valid_accuracies, label='Valid Accuracy')
        axes[0, 1].plot(epochs, self.test_accuracies, label='Test Accuracy', alpha=0.7)
        axes[0, 1].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5,
                           label=f'Best Epoch ({self.best_metrics["best_epoch"]})')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title(f'Accuracy Curves (Seed: {self.global_seed})')

        # 泛化误差曲线
        axes[0, 2].plot(epochs, self.generalization_errors, label='Generalization Error', color='orange')
        axes[0, 2].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5,
                           label=f'Best Epoch ({self.best_metrics["best_epoch"]})')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('|Train Acc - Valid Acc|')
        axes[0, 2].legend()
        axes[0, 2].set_title(f'Generalization Error Curve (Seed: {self.global_seed})')

        # F1分数
        axes[1, 0].plot(epochs, self.test_micro_f1s, label='Micro F1', color='blue')
        axes[1, 0].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Micro F1')
        axes[1, 0].legend()
        axes[1, 0].set_title(f'Micro F1 Score (Seed: {self.global_seed})')

        axes[1, 1].plot(epochs, self.test_macro_f1s, label='Macro F1', color='purple')
        axes[1, 1].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Macro F1')
        axes[1, 1].legend()
        axes[1, 1].set_title(f'Macro F1 Score (Seed: {self.global_seed})')

        # 精确率和召回率
        axes[1, 2].plot(epochs, self.test_precision_scores, label='Precision', color='red')
        axes[1, 2].plot(epochs, self.test_recall_scores, label='Recall', color='green')
        axes[1, 2].axvline(x=self.best_metrics['best_epoch'], color='r', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Epochs')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].set_title(f'Precision & Recall (Seed: {self.global_seed})')

        plt.suptitle(
            f'Learning Curves - Global Seed: {self.global_seed}, Data Split Seed: {DATA_SPLIT_SEED} | Best Val Acc: {self.best_metrics["valid_accuracy"]:.4f} (epoch {self.best_metrics["best_epoch"]})',
            fontsize=14)
        plt.tight_layout()
        plt.savefig(f'learning_curves_global_seed_{self.global_seed}.png')
        plt.close()


def calculate_mean_metrics(all_metrics):
    """计算所有全局seed的指标均值和标准差（包含test和泛化误差）"""
    mean_metrics = {}
    metrics_keys = [
        'valid_accuracy', 'valid_loss', 'generalization_error',
        'test_accuracy', 'test_micro_f1', 'test_macro_f1', 'test_precision', 'test_recall'
    ]

    for key in metrics_keys:
        values = [metrics[key] for metrics in all_metrics]
        mean_metrics[f'mean_{key}'] = np.mean(values)
        mean_metrics[f'std_{key}'] = np.std(values)

    return mean_metrics


if __name__ == "__main__":
    total_start_time = time.time()

    # 1. 解析参数
    args = parameter_parser()
    start_global_seed = args.start_global_seed
    num_global_seeds = args.num_global_seeds
    # 生成全局seed列表：start_global_seed ~ start_global_seed+num_global_seeds-1
    global_seeds = [start_global_seed + i for i in range(num_global_seeds)]

    # 2. 配置全局日志（追加模式）
    logging.basicConfig(
        filename='multi_global_seeds_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'
    )
    logging.info(f"\n{'=' * 80}")
    logging.info(f"START NEW TRAINING SESSION")
    logging.info(f"{'=' * 80}")
    logging.info(f"Fixed data split seed: {DATA_SPLIT_SEED}")
    logging.info(f"Number of global seeds: {num_global_seeds}")
    logging.info(f"Global seeds: {global_seeds}")
    logging.info(f"Model selection: Valid Accuracy > Valid Loss > Generalization Error (|train_acc - val_acc|)")
    logging.info(f"Test set evaluation: Computed for best model (selected by validation metrics)")

    # 3. 加载数据（只加载一次）
    from data import Data

    data = Data()
    data.load(data_path='./data/cora/', data_name='cora')
    target = data.nodes_labels_sequence.type(torch.LongTensor)
    features = data.X_0.type(torch.FloatTensor)
    if torch.cuda.is_available():
        features = features.to('cuda')
        target = target.to('cuda')

    # 4. 循环训练不同全局seed
    all_metrics = []
    for global_seed in global_seeds:
        # 设置当前全局seed（控制模型相关随机过程）
        set_global_seed(global_seed)
        # 初始化训练器并训练
        trainer = HWNNTrainer(args, global_seed, features, target, data)
        best_metrics = trainer.fit()
        all_metrics.append(best_metrics)

    # 5. 计算均值和标准差（包含test和泛化误差）
    mean_metrics = calculate_mean_metrics(all_metrics)

    # 6. 输出汇总结果
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Fixed data split seed: {DATA_SPLIT_SEED}")
    logging.info(f"Number of global seeds: {num_global_seeds}")

    # 详细输出每个seed的结果
    logging.info("\n" + "-" * 80)
    logging.info("DETAILED RESULTS PER SEED:")
    logging.info("-" * 80)
    for i, metrics_dict in enumerate(all_metrics):
        logging.info(f"Seed {metrics_dict['global_seed']}: Best Epoch={metrics_dict['best_epoch']}, "
                     f"Val Acc={metrics_dict['valid_accuracy']:.4f}, "
                     f"Gen Error={metrics_dict['generalization_error']:.4f}, "
                     f"Test Acc={metrics_dict['test_accuracy']:.4f}, "
                     f"Test Macro F1={metrics_dict['test_macro_f1']:.4f}")

    logging.info("\n" + "-" * 80)
    logging.info("MEAN METRICS (± Std):")
    logging.info("-" * 80)
    logging.info(
        f"Mean Valid Accuracy: {mean_metrics['mean_valid_accuracy']:.4f} ± {mean_metrics['std_valid_accuracy']:.4f}")
    logging.info(
        f"Mean Generalization Error: {mean_metrics['mean_generalization_error']:.4f} ± {mean_metrics['std_generalization_error']:.4f}")
    logging.info(
        f"Mean Test Accuracy: {mean_metrics['mean_test_accuracy']:.4f} ± {mean_metrics['std_test_accuracy']:.4f}")
    logging.info(
        f"Mean Test Micro F1: {mean_metrics['mean_test_micro_f1']:.4f} ± {mean_metrics['std_test_micro_f1']:.4f}")
    logging.info(
        f"Mean Test Macro F1: {mean_metrics['mean_test_macro_f1']:.4f} ± {mean_metrics['std_test_macro_f1']:.4f}")
    logging.info(
        f"Mean Test Precision: {mean_metrics['mean_test_precision']:.4f} ± {mean_metrics['std_test_precision']:.4f}")
    logging.info(
        f"Mean Test Recall: {mean_metrics['mean_test_recall']:.4f} ± {mean_metrics['std_test_recall']:.4f}")

    # 打印到控制台
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Fixed data split seed: {DATA_SPLIT_SEED}")
    print(f"Number of global seeds: {num_global_seeds}")

    print("\n" + "-" * 80)
    print("DETAILED RESULTS PER SEED:")
    print("-" * 80)
    for i, metrics_dict in enumerate(all_metrics):
        print(f"Seed {metrics_dict['global_seed']}: Best Epoch={metrics_dict['best_epoch']}, "
              f"Val Acc={metrics_dict['valid_accuracy']:.4f}, "
              f"Gen Error={metrics_dict['generalization_error']:.4f}, "
              f"Test Acc={metrics_dict['test_accuracy']:.4f}, "
              f"Test Macro F1={metrics_dict['test_macro_f1']:.4f}")

    print("\n" + "-" * 80)
    print("MEAN METRICS (± Std):")
    print("-" * 80)
    print(f"Mean Valid Accuracy: {mean_metrics['mean_valid_accuracy']:.4f} ± {mean_metrics['std_valid_accuracy']:.4f}")
    print(f"Mean Generalization Error: {mean_metrics['mean_generalization_error']:.4f} ± {mean_metrics['std_generalization_error']:.4f}")
    print(f"Mean Test Accuracy: {mean_metrics['mean_test_accuracy']:.4f} ± {mean_metrics['std_test_accuracy']:.4f}")
    print(f"Mean Test Micro F1: {mean_metrics['mean_test_micro_f1']:.4f} ± {mean_metrics['std_test_micro_f1']:.4f}")
    print(f"Mean Test Macro F1: {mean_metrics['mean_test_macro_f1']:.4f} ± {mean_metrics['std_test_macro_f1']:.4f}")
    print(f"Mean Test Precision: {mean_metrics['mean_test_precision']:.4f} ± {mean_metrics['std_test_precision']:.4f}")
    print(f"Mean Test Recall: {mean_metrics['mean_test_recall']:.4f} ± {mean_metrics['std_test_recall']:.4f}")

    # 7. 保存所有指标到JSON文件
    with open('multi_global_seeds_metrics.json', 'w') as f:
        json.dump({
            'fixed_data_split_seed': DATA_SPLIT_SEED,
            'global_seeds': global_seeds,
            'all_seeds_metrics': all_metrics,
            'mean_metrics': mean_metrics,
            'training_strategy': 'Validation-based model selection (Val Acc > Val Loss > Generalization Error) + Test set evaluation'
        }, f, indent=4)

    total_time = time.time() - total_start_time
    logging.info(f"\nTotal training time for {num_global_seeds} global seeds: {total_time:.2f} seconds")
    print(f"\nTotal training time: {total_time:.2f} seconds")
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETED")
    logging.info("=" * 80)
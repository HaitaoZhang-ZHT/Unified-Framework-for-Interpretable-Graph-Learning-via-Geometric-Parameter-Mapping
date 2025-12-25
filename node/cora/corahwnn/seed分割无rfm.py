import os
import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run HWNN with vectorized operations and mixed precision.")

    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 150.")

    parser.add_argument("--filters",
                        type=int,
                        default=128,
                        help="Filters (neurons) in convolution. Default is 1433.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.01,
                        help="Dropout probability. Default is 0.01")

    parser.add_argument("--seed",
                        type=int,
                        default=16,
                        help="Base random seed (for model randomness). Default is 38.")

    parser.add_argument("--repeat",
                        type=int,
                        default=10,
                        help="Number of repeated runs (seed 38-47). Default is 10.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.0001,
                        help="Adam weight decay. Default is 0.0001.")

    parser.add_argument("--use-mixed-precision",
                        action="store_true",
                        help="Enable mixed precision training.")

    return parser.parse_args()


class HWNNLayer(torch.nn.Module):
    """超图小波神经网络层（HWNN核心层）"""

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

        # 可训练参数
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(self.device)
        features = features.to(self.device)
        Theta = data.hypergraph_snapshot[snap_index]["Theta"].to(self.device)

        if self.approx:
            # 多项式近似小波变换
            Theta_pows = torch.stack([torch.matrix_power(Theta, k) for k in range(self.K1)], dim=0)
            poly = torch.sum(self.par[:self.K1, None, None] * Theta_pows, dim=0)

            Theta_t = torch.transpose(Theta, 0, 1)
            Theta_t_pows = torch.stack([torch.matrix_power(Theta_t, k) for k in range(self.K2)], dim=0)
            poly_t = torch.sum(self.par[self.K1:, None, None] * Theta_t_pows, dim=0)

            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            # 精确小波变换（预计算）
            wavelets = self.data.hypergraph_snapshot[snap_index]["wavelets"].to(self.device)
            wavelets_inverse = self.data.hypergraph_snapshot[snap_index]["wavelets_inv"].to(self.device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        return local_fea_1


class HWNN(torch.nn.Module):
    """超图小波神经网络（整体模型）"""

    def __init__(self, args, ncount, feature_number, class_number, device, data):
        super(HWNN, self).__init__()
        self.args = args
        self.ncount = ncount
        self.feature_number = feature_number
        self.class_number = class_number
        self.device = device
        self.data = data

        self.hyper_snapshot_num = len(self.data.hypergraph_snapshot)
        print(f"Detected {self.hyper_snapshot_num} hypergraph snapshots")

        # 构建网络层
        self.setup_layers()

        # 快照融合参数
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def setup_layers(self):
        # 两层卷积（特征提取 + 分类）
        self.convolution_1 = HWNNLayer(self.feature_number,
                                       self.args.filters,
                                       self.ncount,
                                       self.device,
                                       K1=2,
                                       K2=2,
                                       approx=True,
                                       data=self.data)

        self.convolution_2 = HWNNLayer(self.args.filters,
                                       self.class_number,
                                       self.ncount,
                                       self.device,
                                       K1=2,
                                       K2=2,
                                       approx=True,
                                       data=self.data)

    def forward(self, features):
        features = features.to(self.device)
        batch_features = features.unsqueeze(0).repeat(self.hyper_snapshot_num, 1, 1)

        # 第一层卷积（特征提取）
        conv1_outputs = []
        for snap_index in range(self.hyper_snapshot_num):
            conv1 = self.convolution_1(batch_features[snap_index], snap_index, self.data)
            conv1_relu = F.relu(conv1)
            conv1_dropout = F.dropout(conv1_relu, self.args.dropout)
            conv1_outputs.append(conv1_dropout)
        conv1_stack = torch.stack(conv1_outputs, dim=0)

        # 第二层卷积（分类）
        conv2_outputs = []
        for snap_index in range(self.hyper_snapshot_num):
            conv2 = self.convolution_2(conv1_stack[snap_index], snap_index, self.data)
            conv2_logsoftmax = F.log_softmax(conv2, dim=1)
            conv2_outputs.append(conv2_logsoftmax)
        conv2_stack = torch.stack(conv2_outputs, dim=0)

        # 快照融合
        deep_features_3 = torch.sum(self.par[:, None, None] * conv2_stack, dim=0)

        return deep_features_3


class HWNNTrainer(object):
    """HWNN训练器：封装训练/验证/测试逻辑"""

    def __init__(self, args, features, target, data, seed):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data = data
        self.features = features
        self.ncount = self.features.size()[0]
        self.feature_number = self.features.size()[1]
        self.target = target.to(self.device)
        self.class_number = self.data.class_num
        self.seed = seed

        # 初始化模型和数据分割
        self.setup_model()
        self.train_test_split()

        # 训练过程记录（新增test指标）
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.test_micro_f1s = []
        self.test_macro_f1s = []
        self.test_precisions = []
        self.test_recalls = []

        # 混合精度训练
        self.scaler = GradScaler(enabled=args.use_mixed_precision)

        # 跟踪当前最佳指标
        self.current_best_valacc = 0.0
        self.current_best_valacc_epoch = 0
        self.current_best_valloss = float('inf')
        self.current_best_valloss_epoch = 0
        self.current_best_generalization_error = float('inf')  # 泛化误差

    def setup_model(self):
        """初始化模型并移至设备"""
        self.model = HWNN(self.args,
                          self.ncount,
                          self.feature_number,
                          self.class_number,
                          self.device, self.data).to(self.device)

    def train_test_split(self):
        """固定seed的数据分割（train/valid/test=60%/20%/20%）"""
        DATA_SPLIT_SEED = 42
        np.random.seed(DATA_SPLIT_SEED)
        torch.manual_seed(DATA_SPLIT_SEED)

        nodes = list(range(self.ncount))
        train_nodes, temp_nodes = train_test_split(nodes, test_size=0.4, random_state=DATA_SPLIT_SEED)
        valid_nodes, test_nodes = train_test_split(temp_nodes, test_size=0.5, random_state=DATA_SPLIT_SEED)

        self.train_nodes = torch.LongTensor(train_nodes).to(self.device)
        self.valid_nodes = torch.LongTensor(valid_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(test_nodes).to(self.device)

        # 打印数据分割信息
        logging.info(
            f"Data split (seed={DATA_SPLIT_SEED}): Train={len(train_nodes)}, Valid={len(valid_nodes)}, Test={len(test_nodes)}")

    def compute_metrics(self, prediction):
        """计算训练/验证/测试指标（含泛化误差+测试集分类指标）"""
        with torch.no_grad():
            # 损失计算
            loss_train = F.cross_entropy(prediction[self.train_nodes], self.target[self.train_nodes])
            loss_valid = F.cross_entropy(prediction[self.valid_nodes], self.target[self.valid_nodes])
            loss_test = F.cross_entropy(prediction[self.test_nodes], self.target[self.test_nodes])

            # 准确率计算
            _, train_pred = prediction[self.train_nodes].max(dim=1)
            accuracy_train = train_pred.eq(self.target[self.train_nodes]).sum().item() / len(self.train_nodes)

            _, valid_pred = prediction[self.valid_nodes].max(dim=1)
            valid_accuracy = valid_pred.eq(self.target[self.valid_nodes]).sum().item() / len(self.valid_nodes)

            _, test_pred = prediction[self.test_nodes].max(dim=1)
            test_accuracy = test_pred.eq(self.target[self.test_nodes]).sum().item() / len(self.test_nodes)

            # 泛化误差（train_acc与val_acc的绝对差）
            generalization_error = abs(accuracy_train - valid_accuracy)

            # 测试集分类指标（F1/Precision/Recall）
            test_target_cpu = self.target[self.test_nodes].cpu().numpy()
            test_pred_cpu = test_pred.cpu().numpy()

            test_micro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='micro', zero_division=1)
            test_macro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)
            test_precision = metrics.precision_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)
            test_recall = metrics.recall_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)

        return {
            'loss_train': loss_train.item(),
            'loss_valid': loss_valid.item(),
            'loss_test': loss_test.item(),
            'acc_train': accuracy_train,
            'acc_valid': valid_accuracy,
            'acc_test': test_accuracy,
            'generalization_error': generalization_error,
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        }

    def evaluate_best_model(self, best_model_path):
        """加载最佳模型并评估测试集（最终验证）"""
        logging.info(f"\n===== Evaluating best model on test set (Seed: {self.seed}) =====")

        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(self.features)
            metrics_dict = self.compute_metrics(outputs)

        # 输出测试集最终结果
        logging.info(f"Final Test Accuracy: {metrics_dict['acc_test']:.4f}")
        logging.info(f"Final Test Micro F1: {metrics_dict['test_micro_f1']:.4f}")
        logging.info(f"Final Test Macro F1: {metrics_dict['test_macro_f1']:.4f}")
        logging.info(f"Final Test Precision: {metrics_dict['test_precision']:.4f}")
        logging.info(f"Final Test Recall: {metrics_dict['test_recall']:.4f}")

        return metrics_dict

    def fit(self):
        """核心训练逻辑（含test评估）"""
        logging.info(f"\n===== Starting training for seed {self.seed} =====")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()

        # 创建模型保存目录
        model_dir = "./models"
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, f"best_model_seed_{self.seed}_dataSplit42.pt")

        # 初始化最佳指标（含test）
        best_valid_accuracy = 0.0
        best_valid_loss = float('inf')
        best_generalization_error = float('inf')
        best_test_accuracy = 0.0
        best_test_macro_f1 = 0.0
        best_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()

            # 前向传播 + 反向传播
            self.optimizer.zero_grad()
            with autocast(enabled=self.args.use_mixed_precision):
                outputs = self.model(self.features)
                train_loss = F.cross_entropy(outputs[self.train_nodes], self.target[self.train_nodes])

            # 梯度更新
            if self.args.use_mixed_precision:
                self.scaler.scale(train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                train_loss.backward()
                self.optimizer.step()

            # 验证+测试阶段（无梯度）
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.features)
                metrics_dict = self.compute_metrics(outputs)
            self.model.train()

            # 更新当前最佳指标跟踪
            if metrics_dict['acc_valid'] > self.current_best_valacc:
                self.current_best_valacc = metrics_dict['acc_valid']
                self.current_best_valacc_epoch = epoch
                self.current_best_valloss = metrics_dict['loss_valid']
                self.current_best_valloss_epoch = epoch
                self.current_best_generalization_error = metrics_dict['generalization_error']
            elif metrics_dict['acc_valid'] == self.current_best_valacc:
                if metrics_dict['loss_valid'] < self.current_best_valloss:
                    self.current_best_valloss = metrics_dict['loss_valid']
                    self.current_best_valloss_epoch = epoch
                    self.current_best_generalization_error = metrics_dict['generalization_error']
                elif metrics_dict['loss_valid'] == self.current_best_valloss:
                    if metrics_dict['generalization_error'] < self.current_best_generalization_error:
                        self.current_best_generalization_error = metrics_dict['generalization_error']

            # ========== 核心修复：三层判断统一遵循「存旧值→构理由→更新值」 ==========
            update_reason = "No update"
            should_save_model = False

            # 第一层判断：验证准确率更高
            if metrics_dict['acc_valid'] > best_valid_accuracy:
                # 1. 保存更新前的旧值
                old_best_val_acc = best_valid_accuracy
                # 2. 构造更新理由（基于旧值）
                update_reason = f"Higher valid accuracy: {metrics_dict['acc_valid']:.4f} > {old_best_val_acc:.4f}"
                # 3. 更新最佳指标（含test）
                best_valid_accuracy = metrics_dict['acc_valid']
                best_valid_loss = metrics_dict['loss_valid']
                best_generalization_error = metrics_dict['generalization_error']
                best_test_accuracy = metrics_dict['acc_test']
                best_test_macro_f1 = metrics_dict['test_macro_f1']
                best_epoch = epoch
                should_save_model = True

            # 第二层判断：验证准确率相同，验证损失更低
            elif metrics_dict['acc_valid'] == best_valid_accuracy:
                if metrics_dict['loss_valid'] < best_valid_loss:
                    # 1. 保存更新前的旧值
                    old_best_val_loss = best_valid_loss
                    # 2. 构造更新理由（基于旧值）
                    update_reason = f"Same valid accuracy ({metrics_dict['acc_valid']:.4f}) but better loss: {metrics_dict['loss_valid']:.4f} < {old_best_val_loss:.4f}"
                    # 3. 更新最佳指标（含test）
                    best_valid_loss = metrics_dict['loss_valid']
                    best_generalization_error = metrics_dict['generalization_error']
                    best_test_accuracy = metrics_dict['acc_test']
                    best_test_macro_f1 = metrics_dict['test_macro_f1']
                    best_epoch = epoch
                    should_save_model = True

                # 第三层判断：验证准确率和损失都相同，泛化误差更小
                elif metrics_dict['loss_valid'] == best_valid_loss:
                    if metrics_dict['generalization_error'] < best_generalization_error:
                        # 1. 保存更新前的旧值
                        old_best_gen_error = best_generalization_error
                        # 2. 构造更新理由（基于旧值）
                        update_reason = f"Same val_acc ({metrics_dict['acc_valid']:.4f}) and val_loss ({metrics_dict['loss_valid']:.4f}) but better generalization error: {metrics_dict['generalization_error']:.4f} < {old_best_gen_error:.4f}"
                        # 3. 更新最佳指标（含test）
                        best_generalization_error = metrics_dict['generalization_error']
                        best_test_accuracy = metrics_dict['acc_test']
                        best_test_macro_f1 = metrics_dict['test_macro_f1']
                        best_epoch = epoch
                        should_save_model = True

            # 保存最佳模型
            if should_save_model:
                if self.args.use_mixed_precision:
                    self.model.eval()
                    with torch.no_grad():
                        model_to_save = self.model.float()
                        torch.save(model_to_save.state_dict(), best_model_path)
                        if next(self.model.parameters()).dtype == torch.float16:
                            self.model = self.model.half()
                    self.model.train()
                else:
                    torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"Epoch {epoch}: Saving best model - {update_reason}")

            # 记录训练过程指标（新增test）
            self.train_losses.append(metrics_dict['loss_train'])
            self.valid_losses.append(metrics_dict['loss_valid'])
            self.test_losses.append(metrics_dict['loss_test'])
            self.train_accuracies.append(metrics_dict['acc_train'])
            self.valid_accuracies.append(metrics_dict['acc_valid'])
            self.test_accuracies.append(metrics_dict['acc_test'])
            self.test_micro_f1s.append(metrics_dict['test_micro_f1'])
            self.test_macro_f1s.append(metrics_dict['test_macro_f1'])
            self.test_precisions.append(metrics_dict['test_precision'])
            self.test_recalls.append(metrics_dict['test_recall'])

            # 打印epoch日志（新增test指标）
            epoch_duration = time.time() - epoch_start
            log_msg = (
                f"Epoch {epoch:3d}/{self.args.epochs:3d} | "
                f"Train Loss: {metrics_dict['loss_train']:.4f} | "
                f"Valid Loss: {metrics_dict['loss_valid']:.4f} | "
                f"Test Loss: {metrics_dict['loss_test']:.4f} | "
                f"Train Acc: {metrics_dict['acc_train']:.4f} | "
                f"Valid Acc: {metrics_dict['acc_valid']:.4f} | "
                f"Test Acc: {metrics_dict['acc_test']:.4f} | "
                f"Gen Error: {metrics_dict['generalization_error']:.4f} | "
                f"Current Best Val Acc: {self.current_best_valacc:.4f} (epoch {self.current_best_valacc_epoch})"
            )
            if should_save_model:
                log_msg += f" | [BEST MODEL SAVED]"
            log_msg += f" | Time:{epoch_duration:.2f}s"
            logging.info(log_msg)

        # 绘制学习曲线（新增test）
        self.plot_learning_curves()

        # 加载最佳模型并评估测试集
        final_test_metrics = self.evaluate_best_model(best_model_path)

        # 训练结束日志（含test）
        logging.info(f"\n{'=' * 60}")
        logging.info(f"BEST MODEL SUMMARY - Seed: {self.seed}")
        logging.info(f"{'=' * 60}")
        logging.info(f"Best Epoch: {best_epoch}")
        logging.info(f"Best Valid Accuracy: {best_valid_accuracy:.4f}")
        logging.info(f"Best Valid Loss: {best_valid_loss:.4f}")
        logging.info(f"Best Generalization Error (|train_acc - val_acc|): {best_generalization_error:.4f}")
        logging.info(f"Best Test Accuracy (at best epoch): {best_test_accuracy:.4f}")
        logging.info(f"Final Test Accuracy (loaded model): {final_test_metrics['acc_test']:.4f}")
        logging.info(f"Final Test Macro F1: {final_test_metrics['test_macro_f1']:.4f}")
        logging.info(f"Final Test Precision: {final_test_metrics['test_precision']:.4f}")
        logging.info(f"Final Test Recall: {final_test_metrics['test_recall']:.4f}")
        logging.info(f"{'=' * 60}")

        return {
            'seed': self.seed,
            'best_epoch': best_epoch,
            'best_valid_accuracy': best_valid_accuracy,
            'best_valid_loss': best_valid_loss,
            'best_generalization_error': best_generalization_error,
            'best_test_accuracy': best_test_accuracy,
            'final_test_accuracy': final_test_metrics['acc_test'],
            'final_test_micro_f1': final_test_metrics['test_micro_f1'],
            'final_test_macro_f1': final_test_metrics['test_macro_f1'],
            'final_test_precision': final_test_metrics['test_precision'],
            'final_test_recall': final_test_metrics['test_recall']
        }

    def plot_learning_curves(self):
        """绘制学习曲线（新增test损失/准确率/F1）"""
        epochs = range(1, self.args.epochs + 1)
        plt.figure(figsize=(20, 10))

        # 1. 损失曲线（train/valid/test）
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss', linestyle='-')
        plt.plot(epochs, self.valid_losses, label='Valid Loss', linestyle='--')
        plt.plot(epochs, self.test_losses, label='Test Loss', linestyle='-.', alpha=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curves (Seed: {self.seed})')

        # 2. 准确率曲线（train/valid/test）
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Acc', linestyle='-')
        plt.plot(epochs, self.valid_accuracies, label='Valid Acc', linestyle='--')
        plt.plot(epochs, self.test_accuracies, label='Test Acc', linestyle='-.', alpha=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy Curves (Seed: {self.seed})')

        # 3. 泛化误差曲线
        plt.subplot(2, 3, 3)
        generalization_errors = [abs(t - v) for t, v in zip(self.train_accuracies, self.valid_accuracies)]
        plt.plot(epochs, generalization_errors, label='Generalization Error', color='orange', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('|Train Acc - Valid Acc|')
        plt.legend()
        plt.title(f'Generalization Error (Seed: {self.seed})')

        # 4. 测试集Micro F1
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.test_micro_f1s, label='Test Micro F1', color='green', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Micro F1')
        plt.legend()
        plt.title(f'Test Micro F1 (Seed: {self.seed})')

        # 5. 测试集Macro F1
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.test_macro_f1s, label='Test Macro F1', color='red', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.title(f'Test Macro F1 (Seed: {self.seed})')

        # 6. 测试集Precision & Recall
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.test_precisions, label='Test Precision', color='purple', linestyle='-')
        plt.plot(epochs, self.test_recalls, label='Test Recall', color='blue', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.title(f'Test Precision & Recall (Seed: {self.seed})')

        plt.tight_layout()
        plt.savefig(f'learning_curves_seed_{self.seed}_dataSplit42.png')
        plt.close()


if __name__ == "__main__":
    # 全局计时
    total_start = time.time()

    # 日志配置
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'
    )

    # 解析参数
    args = parameter_parser()

    # 加载数据（需确保data.py存在并正确实现Data类）
    from data import Data

    data = Data()
    data.load(data_path='./data/cora/', data_name='cora')
    target = data.nodes_labels_sequence.type(torch.LongTensor)
    features = data.X_0.type(torch.FloatTensor)

    # 移至GPU（如果可用）
    if torch.cuda.is_available():
        features = features.cuda()
        target = target.cuda()

    # 存储所有seed的训练结果
    all_results = []

    # 训练开始日志
    logging.info(f"\n{'=' * 80}")
    logging.info(f"START NEW TRAINING SESSION")
    logging.info(f"{'=' * 80}")
    logging.info(f"Base seed: {args.seed}, Number of repeats: {args.repeat}")
    logging.info(f"Model selection criteria: Val Acc > Val Loss > Generalization Error (|train_acc - val_acc|)")
    logging.info(f"Data split seed: 42 (fixed for all repeats)")
    logging.info(f"Test set evaluation: Enabled (final evaluation on best model)")

    # 多seed重复训练
    for i in range(args.repeat):
        current_seed = args.seed + i
        logging.info(f"\n\n===== Starting training for Seed {current_seed} =====")
        print(f"\n\n===== Starting training for Seed {current_seed} =====")

        # 设置全局随机种子（保证可复现）
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 初始化训练器并开始训练
        trainer = HWNNTrainer(args, features, target, data, current_seed)
        results = trainer.fit()
        all_results.append(results)

        # 打印当前seed结果（含test）
        print(f"\nSeed {current_seed} Final Results:")
        print(f"Best Epoch: {results['best_epoch']}")
        print(f"Best Valid Accuracy: {results['best_valid_accuracy']:.4f}")
        print(f"Best Valid Loss: {results['best_valid_loss']:.4f}")
        print(f"Best Generalization Error: {results['best_generalization_error']:.4f}")
        print(f"Final Test Accuracy: {results['final_test_accuracy']:.4f}")
        print(f"Final Test Macro F1: {results['final_test_macro_f1']:.4f}")

    # 统计多seed平均结果（含test）
    if args.repeat > 1:
        # 验证集指标统计
        avg_valid_accuracy = np.mean([r['best_valid_accuracy'] for r in all_results])
        avg_valid_loss = np.mean([r['best_valid_loss'] for r in all_results])
        avg_generalization_error = np.mean([r['best_generalization_error'] for r in all_results])
        std_valid_accuracy = np.std([r['best_valid_accuracy'] for r in all_results])
        std_valid_loss = np.std([r['best_valid_loss'] for r in all_results])
        std_generalization_error = np.std([r['best_generalization_error'] for r in all_results])

        # 测试集指标统计
        avg_test_accuracy = np.mean([r['final_test_accuracy'] for r in all_results])
        avg_test_macro_f1 = np.mean([r['final_test_macro_f1'] for r in all_results])
        avg_test_precision = np.mean([r['final_test_precision'] for r in all_results])
        avg_test_recall = np.mean([r['final_test_recall'] for r in all_results])
        std_test_accuracy = np.std([r['final_test_accuracy'] for r in all_results])
        std_test_macro_f1 = np.std([r['final_test_macro_f1'] for r in all_results])
        std_test_precision = np.std([r['final_test_precision'] for r in all_results])
        std_test_recall = np.std([r['final_test_recall'] for r in all_results])

        # 日志输出平均结果（含test）
        logging.info("\n" + "=" * 80)
        logging.info("AVERAGE RESULTS (across all seeds):")
        logging.info("=" * 80)
        logging.info(f"[Validation Set]")
        logging.info(f"Average Valid Accuracy: {avg_valid_accuracy:.4f} ± {std_valid_accuracy:.4f}")
        logging.info(f"Average Valid Loss: {avg_valid_loss:.4f} ± {std_valid_loss:.4f}")
        logging.info(f"Average Generalization Error: {avg_generalization_error:.4f} ± {std_generalization_error:.4f}")
        logging.info(f"\n[Test Set]")
        logging.info(f"Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        logging.info(f"Average Test Macro F1: {avg_test_macro_f1:.4f} ± {std_test_macro_f1:.4f}")
        logging.info(f"Average Test Precision: {avg_test_precision:.4f} ± {std_test_precision:.4f}")
        logging.info(f"Average Test Recall: {avg_test_recall:.4f} ± {std_test_recall:.4f}")

        # 控制台输出平均结果（含test）
        print("\n" + "=" * 80)
        print("AVERAGE RESULTS (across all seeds):")
        print("=" * 80)
        print(f"[Validation Set]")
        print(f"Average Valid Accuracy: {avg_valid_accuracy:.4f} ± {std_valid_accuracy:.4f}")
        print(f"Average Valid Loss: {avg_valid_loss:.4f} ± {std_valid_loss:.4f}")
        print(f"Average Generalization Error: {avg_generalization_error:.4f} ± {std_generalization_error:.4f}")
        print(f"\n[Test Set]")
        print(f"Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        print(f"Average Test Macro F1: {avg_test_macro_f1:.4f} ± {std_test_macro_f1:.4f}")
        print(f"Average Test Precision: {avg_test_precision:.4f} ± {std_test_precision:.4f}")
        print(f"Average Test Recall: {avg_test_recall:.4f} ± {std_test_recall:.4f}")

    # 总训练时间
    total_duration = time.time() - total_start
    logging.info(f"\nTotal Training Time ({args.repeat} runs): {total_duration:.2f} seconds")
    print(f"\nTotal Training Time ({args.repeat} runs): {total_duration:.2f} seconds")
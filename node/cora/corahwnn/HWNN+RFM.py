import os
import torch
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建模型保存目录（若不存在）
MODEL_SAVE_DIR = "./saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run HWNN with vectorized operations.")

    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--filters", type=int, default=1433, help="Filters in convolution.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Ratio of test samples.")  # 新增test比例
    parser.add_argument("--valid-size", type=float, default=0.2,
                        help="Ratio of validation samples (on remaining data).")
    parser.add_argument("--dropout", type=float, default=0.01, help="Dropout probability.")
    parser.add_argument("--start_seed", type=int, default=16, help="Base random seed (for model randomness).")
    parser.add_argument("--repeat", type=int, default=10, help="Number of repeated runs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Adam weight decay.")
    parser.add_argument("--use-mixed-precision", action="store_true", help="Enable mixed precision training.")

    return parser.parse_args()


class HWNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device, K1=2, K2=2, approx=True, data=None, weight_init=None):
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
        self.weight_init = weight_init
        self.init_parameters()

    def init_parameters(self):
        if self.weight_init is not None:
            weight_matrix_init = np.load(self.weight_init)
            self.weight_matrix.data = torch.Tensor(weight_matrix_init)
        else:
            torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(self.device)
        features = features.to(self.device)
        Theta = data.hypergraph_snapshot[snap_index]["Theta"].to(self.device)

        if self.approx:
            Theta_pows = torch.stack([torch.matrix_power(Theta, k) for k in range(self.K1)], dim=0)
            poly = torch.sum(self.par[:self.K1, None, None] * Theta_pows, dim=0)

            Theta_t = torch.transpose(Theta, 0, 1)
            Theta_t_pows = torch.stack([torch.matrix_power(Theta_t, k) for k in range(self.K2)], dim=0)
            poly_t = torch.sum(self.par[self.K1:, None, None] * Theta_t_pows, dim=0)

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
        self.hyper_snapshot_num = len(data.hypergraph_snapshot)
        print(f"there are {self.hyper_snapshot_num} hypergraphs")
        self.setup_layers()
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def setup_layers(self):
        self.convolution_1 = HWNNLayer(
            self.feature_number, self.args.filters, self.ncount, self.device,
            K1=2, K2=2, approx=True, data=self.data, weight_init='M_layer_1.npy'
        )
        self.convolution_2 = HWNNLayer(
            self.args.filters, self.class_number, self.ncount, self.device,
            K1=2, K2=2, approx=True, data=self.data
        )

    def forward(self, features):
        features = features.to(self.device)
        batch_features = features.unsqueeze(0).repeat(self.hyper_snapshot_num, 1, 1)

        conv1_outputs = []
        for snap_index in range(self.hyper_snapshot_num):
            conv1 = self.convolution_1(batch_features[snap_index], snap_index, self.data)
            conv1_relu = F.relu(conv1)
            conv1_dropout = F.dropout(conv1_relu, self.args.dropout)
            conv1_outputs.append(conv1_dropout)

        conv1_stack = torch.stack(conv1_outputs, dim=0)

        conv2_outputs = []
        for snap_index in range(self.hyper_snapshot_num):
            conv2 = self.convolution_2(conv1_stack[snap_index], snap_index, self.data)
            conv2_logsoftmax = F.log_softmax(conv2, dim=1)
            conv2_outputs.append(conv2_logsoftmax)

        conv2_stack = torch.stack(conv2_outputs, dim=0)

        deep_features_3 = torch.sum(self.par[:, None, None] * conv2_stack, dim=0)

        return deep_features_3


class HWNNTrainer(object):
    def __init__(self, args, features, target, data, seed):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data = data
        self.features = features
        self.ncount = self.features.size()[0]
        self.feature_number = self.features.size()[1]
        self.target = target.to(self.device)
        self.class_number = data.class_num
        self.seed = seed
        self.setup_model()
        self.train_valid_test_split()  # 替换为与RFM一致的划分逻辑

        # 指标记录列表（新增Precision/Recall）
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.test_macro_f1s = []  # 新增test F1
        self.test_micro_f1s = []
        self.test_precisions = []  # 新增：Test Precision（宏平均）
        self.test_recalls = []     # 新增：Test Recall（宏平均）

        self.scaler = GradScaler(enabled=args.use_mixed_precision)

        # 跟踪当前最佳指标（新增泛化误差）
        self.current_best_valacc = 0.0
        self.current_best_valacc_epoch = 0
        self.current_best_valloss = float('inf')
        self.current_best_valloss_epoch = 0
        self.current_best_generalization_error = float('inf')  # 泛化误差

    def setup_model(self):
        self.model = HWNN(self.args, self.ncount, self.feature_number, self.class_number, self.device, self.data).to(
            self.device)

    def train_valid_test_split(self):
        """
        逻辑：先划Train(60%) → 剩余40%划分为Val(20%)+Test(20%)
        """
        DATA_SPLIT_SEED = 42  # 与RFM保持一致的分割种子
        np.random.seed(DATA_SPLIT_SEED)
        torch.manual_seed(DATA_SPLIT_SEED)

        nodes = list(range(self.ncount))
        # 第一步：先划Train集（60%），剩余40%为临时集
        train_nodes, temp_nodes = train_test_split(
            nodes,
            test_size=0.4,  # 40%剩余 → Train=60%
            random_state=DATA_SPLIT_SEED
        )
        # 第二步：临时集对半分 → Val=20%，Test=20%
        valid_nodes, test_nodes = train_test_split(
            temp_nodes,
            test_size=0.5,  # 40%×50%=20%
            random_state=DATA_SPLIT_SEED
        )

        self.train_nodes = torch.LongTensor(train_nodes).to(self.device)
        self.valid_nodes = torch.LongTensor(valid_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(test_nodes).to(self.device)

        # 打印分割信息（日志+控制台）
        logging.info(
            f"Data split (seed={DATA_SPLIT_SEED}, align with RFM): Train={len(train_nodes)}, Valid={len(valid_nodes)}, Test={len(test_nodes)}")
        print(
            f"Data split (align with RFM): Train={len(train_nodes)}, Valid={len(valid_nodes)}, Test={len(test_nodes)}")

    def compute_metrics(self, prediction):
        """计算训练/验证/测试指标（新增Precision、Recall）"""
        with torch.no_grad():
            # 计算损失（新增test_loss）
            loss_train = F.cross_entropy(prediction[self.train_nodes], self.target[self.train_nodes])
            loss_valid = F.cross_entropy(prediction[self.valid_nodes], self.target[self.valid_nodes])
            loss_test = F.cross_entropy(prediction[self.test_nodes], self.target[self.test_nodes])

            # 计算准确率（新增test_acc）
            _, train_pred = prediction[self.train_nodes].max(dim=1)
            accuracy_train = train_pred.eq(self.target[self.train_nodes]).sum().item() / len(self.train_nodes)

            _, valid_pred = prediction[self.valid_nodes].max(dim=1)
            valid_accuracy = valid_pred.eq(self.target[self.valid_nodes]).sum().item() / len(self.valid_nodes)

            _, test_pred = prediction[self.test_nodes].max(dim=1)
            test_accuracy = test_pred.eq(self.target[self.test_nodes]).sum().item() / len(self.test_nodes)

            # 计算test集分类指标（核心修改：新增Precision、Recall）
            test_target_cpu = self.target[self.test_nodes].cpu().numpy()
            test_pred_cpu = test_pred.cpu().numpy()
            test_macro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)
            test_micro_f1 = metrics.f1_score(test_target_cpu, test_pred_cpu, average='micro', zero_division=1)
            test_precision = metrics.precision_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)  # 新增
            test_recall = metrics.recall_score(test_target_cpu, test_pred_cpu, average='macro', zero_division=1)       # 新增

            # 泛化误差（仅train/valid）
            generalization_error = abs(accuracy_train - valid_accuracy)

        return {
            'loss_train': loss_train.item(),
            'loss_valid': loss_valid.item(),
            'loss_test': loss_test.item(),
            'acc_train': accuracy_train,
            'acc_valid': valid_accuracy,
            'acc_test': test_accuracy,
            'test_macro_f1': test_macro_f1,
            'test_micro_f1': test_micro_f1,
            'test_precision': test_precision,  # 新增
            'test_recall': test_recall,        # 新增
            'generalization_error': generalization_error
        }

    def evaluate_best_model(self, best_model_path):
        """加载最佳模型（基于valid选的），评估test集最终性能（新增Precision、Recall）"""
        logging.info(f"\n===== Evaluating best model on TEST set (Seed: {self.seed}) =====")
        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(self.features)
            test_metrics = self.compute_metrics(outputs)

        # 输出test集最终结果（新增Precision、Recall）
        logging.info(f"Final Test Accuracy: {test_metrics['acc_test']:.4f}")
        logging.info(f"Final Test Macro F1: {test_metrics['test_macro_f1']:.4f}")
        logging.info(f"Final Test Micro F1: {test_metrics['test_micro_f1']:.4f}")
        logging.info(f"Final Test Precision (Macro): {test_metrics['test_precision']:.4f}")  # 新增
        logging.info(f"Final Test Recall (Macro): {test_metrics['test_recall']:.4f}")        # 新增
        return test_metrics

    def fit(self):
        logging.info(f"\n===== Starting training for seed {self.seed} =====")
        logging.info(
            f"Model selection criteria: Valid Accuracy > Valid Loss > Generalization Error (|train_acc - val_acc|)")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        self.model.train()

        # 最佳模型指标初始化（新增test相关，但模型选择仍基于valid）
        best_valid_accuracy = 0.0
        best_valid_loss = float('inf')
        best_generalization_error = float('inf')
        best_test_accuracy = 0.0  # 记录最佳valid对应的test acc（仅参考）
        best_epoch = 0

        # 模型保存路径
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_seed_{self.seed}_dataSplit42.pt")

        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()

            self.optimizer.zero_grad()
            with autocast(enabled=self.args.use_mixed_precision):
                outputs = self.model(self.features)
                train_loss = F.cross_entropy(outputs[self.train_nodes], self.target[self.train_nodes])

            if self.args.use_mixed_precision:
                self.scaler.scale(train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                train_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.features)
                metrics_dict = self.compute_metrics(outputs)

            self.model.train()

            # 更新当前最佳指标跟踪（仅基于valid）
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

            update_reason = "No update"
            should_save_model = False

            # ========== 核心：模型选择仍仅基于valid，test仅记录 ==========
            # 第一层：验证准确率更高
            if metrics_dict['acc_valid'] > best_valid_accuracy:
                old_best_val_acc = best_valid_accuracy
                update_reason = f"Higher valid accuracy: {metrics_dict['acc_valid']:.4f} > {old_best_val_acc:.4f}"
                best_valid_accuracy = metrics_dict['acc_valid']
                best_valid_loss = metrics_dict['loss_valid']
                best_generalization_error = metrics_dict['generalization_error']
                best_test_accuracy = metrics_dict['acc_test']  # 记录此时的test acc
                best_epoch = epoch
                should_save_model = True

            # 第二层：验证准确率相同，验证损失更低
            elif metrics_dict['acc_valid'] == best_valid_accuracy:
                if metrics_dict['loss_valid'] < best_valid_loss:
                    old_best_val_loss = best_valid_loss
                    update_reason = f"Same valid accuracy ({metrics_dict['acc_valid']:.4f}) but better loss: {metrics_dict['loss_valid']:.4f} < {old_best_val_loss:.4f}"
                    best_valid_loss = metrics_dict['loss_valid']
                    best_generalization_error = metrics_dict['generalization_error']
                    best_test_accuracy = metrics_dict['acc_test']
                    best_epoch = epoch
                    should_save_model = True

                # 第三层：验证准确率和损失都相同，泛化误差更小
                elif metrics_dict['loss_valid'] == best_valid_loss:
                    if metrics_dict['generalization_error'] < best_generalization_error:
                        old_best_gen_error = best_generalization_error
                        update_reason = f"Same val_acc ({metrics_dict['acc_valid']:.4f}) and val_loss ({metrics_dict['loss_valid']:.4f}) but better generalization error: {metrics_dict['generalization_error']:.4f} < {old_best_gen_error:.4f}"
                        best_generalization_error = metrics_dict['generalization_error']
                        best_test_accuracy = metrics_dict['acc_test']
                        best_epoch = epoch
                        should_save_model = True

            if should_save_model:
                try:
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
                except Exception as e:
                    logging.error(f"Failed to save model: {str(e)}")
                    print(f"Warning: Failed to save model - {str(e)}")

            # 记录指标（新增Precision、Recall）
            self.train_losses.append(metrics_dict['loss_train'])
            self.valid_losses.append(metrics_dict['loss_valid'])
            self.test_losses.append(metrics_dict['loss_test'])
            self.train_accuracies.append(metrics_dict['acc_train'])
            self.valid_accuracies.append(metrics_dict['acc_valid'])
            self.test_accuracies.append(metrics_dict['acc_test'])
            self.test_macro_f1s.append(metrics_dict['test_macro_f1'])
            self.test_micro_f1s.append(metrics_dict['test_micro_f1'])
            self.test_precisions.append(metrics_dict['test_precision'])  # 新增
            self.test_recalls.append(metrics_dict['test_recall'])        # 新增

            epoch_duration = time.time() - start_time

            # 打印epoch日志（新增test指标）
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

        self.plot_learning_curves()

        # 加载最佳模型，评估test集最终性能（核心：test仅最终评估）
        final_test_metrics = self.evaluate_best_model(best_model_path)

        # 训练结束后，记录最佳模型信息（新增test）
        logging.info(f"\n{'=' * 60}")
        logging.info(f"BEST MODEL SUMMARY - Seed: {self.seed}")
        logging.info(f"{'=' * 60}")
        logging.info(f"Best Epoch: {best_epoch}")
        logging.info(f"Best Valid Accuracy: {best_valid_accuracy:.4f}")
        logging.info(f"Best Valid Loss: {best_valid_loss:.4f}")
        logging.info(f"Best Generalization Error (|train_acc - val_acc|): {best_generalization_error:.4f}")
        logging.info(f"Test Accuracy at Best Epoch: {best_test_accuracy:.4f} (reference only)")
        logging.info(f"Final Test Accuracy (Loaded Model): {final_test_metrics['acc_test']:.4f}")
        logging.info(f"Final Test Macro F1: {final_test_metrics['test_macro_f1']:.4f}")
        logging.info(f"Final Test Precision (Macro): {final_test_metrics['test_precision']:.4f}")  # 新增
        logging.info(f"Final Test Recall (Macro): {final_test_metrics['test_recall']:.4f}")        # 新增
        logging.info(f"{'=' * 60}")

        return {
            'seed': self.seed,
            'best_epoch': best_epoch,
            'best_valid_accuracy': best_valid_accuracy,
            'best_valid_loss': best_valid_loss,
            'best_generalization_error': best_generalization_error,
            'best_test_accuracy': best_test_accuracy,
            'final_test_accuracy': final_test_metrics['acc_test'],
            'final_test_macro_f1': final_test_metrics['test_macro_f1'],
            'final_test_micro_f1': final_test_metrics['test_micro_f1'],
            'final_test_precision': final_test_metrics['test_precision'],  # 新增
            'final_test_recall': final_test_metrics['test_recall']        # 新增
        }

    def plot_learning_curves(self):
        """绘制学习曲线（新增Precision/Recall曲线）"""
        PLOT_SAVE_DIR = "./learning_curves"
        os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

        epochs = range(1, self.args.epochs + 1)
        plt.figure(figsize=(20, 10))  # 扩大画布，容纳更多子图

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

        # 4. Test Macro F1曲线
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.test_macro_f1s, label='Test Macro F1', color='red', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.title(f'Test Macro F1 (Seed: {self.seed})')

        # 5. Test Precision曲线（新增）
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.test_precisions, label='Test Precision (Macro)', color='purple', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.title(f'Test Precision (Seed: {self.seed})')

        # 6. Test Recall曲线（新增）
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.test_recalls, label='Test Recall (Macro)', color='green', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.title(f'Test Recall (Seed: {self.seed})')

        plt.tight_layout()
        plot_path = os.path.join(PLOT_SAVE_DIR, f'learning_curves_seed_{self.seed}_dataSplit42.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Learning curves saved to {plot_path}")


if __name__ == "__main__":
    start_time = time.time()

    logging.basicConfig(filename='training.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a')  # 追加模式

    args = parameter_parser()

    from data import Data

    data = Data()
    data.load(data_path='./data/cora/', data_name='cora')
    target = data.nodes_labels_sequence.type(torch.LongTensor)
    features = data.X_0.type(torch.FloatTensor)

    if torch.cuda.is_available():
        features = features.cuda()
        target = target.cuda()

    all_results = []

    # 训练开始的总日志（更新说明）
    logging.info(f"\n{'=' * 80}")
    logging.info(f"START NEW TRAINING SESSION")
    logging.info(f"{'=' * 80}")
    logging.info(f"Base seed: {args.start_seed}, Number of repeats: {args.repeat}")
    logging.info(
        f"Model selection criteria: Valid Accuracy > Valid Loss > Generalization Error (|train_acc - val_acc|)")
    logging.info(f"Data split: Train=60%, Valid=20%, Test=20% (seed=42, align with RFM)")
    logging.info(f"Test set: Only used for final evaluation (no participation in model selection)")

    for i in range(args.repeat):
        current_seed = args.start_seed + i
        logging.info(f"\n\n===== Starting training for Seed {current_seed} =====")
        print(f"\n\n===== Starting training for Seed {current_seed} =====")

        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        trainer = HWNNTrainer(args, features, target, data, current_seed)
        results = trainer.fit()
        all_results.append(results)

        # 打印当前seed的最终结果（新增Precision、Recall）
        print(f"\nSeed {current_seed} Final Results:")
        print(f"Best Epoch: {results['best_epoch']}")
        print(f"Best Valid Accuracy: {results['best_valid_accuracy']:.4f}")
        print(f"Best Valid Loss: {results['best_valid_loss']:.4f}")
        print(f"Best Generalization Error: {results['best_generalization_error']:.4f}")
        print(f"Final Test Accuracy: {results['final_test_accuracy']:.4f}")
        print(f"Final Test Macro F1: {results['final_test_macro_f1']:.4f}")
        print(f"Final Test Precision (Macro): {results['final_test_precision']:.4f}")  # 新增
        print(f"Final Test Recall (Macro): {results['final_test_recall']:.4f}")        # 新增

    if len(all_results) > 1:
        # 计算平均结果（新增Precision、Recall）
        avg_valid_accuracy = np.mean([r['best_valid_accuracy'] for r in all_results])
        avg_valid_loss = np.mean([r['best_valid_loss'] for r in all_results])
        avg_generalization_error = np.mean([r['best_generalization_error'] for r in all_results])
        avg_test_accuracy = np.mean([r['final_test_accuracy'] for r in all_results])
        avg_test_macro_f1 = np.mean([r['final_test_macro_f1'] for r in all_results])
        avg_test_precision = np.mean([r['final_test_precision'] for r in all_results])  # 新增
        avg_test_recall = np.mean([r['final_test_recall'] for r in all_results])        # 新增

        std_valid_accuracy = np.std([r['best_valid_accuracy'] for r in all_results])
        std_valid_loss = np.std([r['best_valid_loss'] for r in all_results])
        std_generalization_error = np.std([r['best_generalization_error'] for r in all_results])
        std_test_accuracy = np.std([r['final_test_accuracy'] for r in all_results])
        std_test_macro_f1 = np.std([r['final_test_macro_f1'] for r in all_results])
        std_test_precision = np.std([r['final_test_precision'] for r in all_results])   # 新增
        std_test_recall = np.std([r['final_test_recall'] for r in all_results])         # 新增

        # 输出每个seed的详细结果
        logging.info("\n" + "=" * 80)
        logging.info("DETAILED RESULTS PER SEED:")
        logging.info("=" * 80)
        for result in all_results:
            logging.info(f"Seed {result['seed']}: "
                         f"Best Epoch={result['best_epoch']}, "
                         f"Val Acc={result['best_valid_accuracy']:.4f}, "
                         f"Val Loss={result['best_valid_loss']:.4f}, "
                         f"Gen Error={result['best_generalization_error']:.4f}, "
                         f"Test Acc={result['final_test_accuracy']:.4f}, "
                         f"Test Macro F1={result['final_test_macro_f1']:.4f}, "
                         f"Test Precision={result['final_test_precision']:.4f}, "  # 新增
                         f"Test Recall={result['final_test_recall']:.4f}")         # 新增

        # 输出平均结果（新增Precision、Recall）
        logging.info("\n" + "=" * 80)
        logging.info("AVERAGE RESULTS (across all seeds):")
        logging.info("=" * 80)
        logging.info(f"Average Valid Accuracy: {avg_valid_accuracy:.4f} ± {std_valid_accuracy:.4f}")
        logging.info(f"Average Valid Loss: {avg_valid_loss:.4f} ± {std_valid_loss:.4f}")
        logging.info(f"Average Generalization Error: {avg_generalization_error:.4f} ± {std_generalization_error:.4f}")
        logging.info(f"Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        logging.info(f"Average Test Macro F1: {avg_test_macro_f1:.4f} ± {std_test_macro_f1:.4f}")
        logging.info(f"Average Test Precision: {avg_test_precision:.4f} ± {std_test_precision:.4f}")  # 新增
        logging.info(f"Average Test Recall: {avg_test_recall:.4f} ± {std_test_recall:.4f}")          # 新增

        print("\n" + "=" * 80)
        print("AVERAGE RESULTS (across all seeds):")
        print("=" * 80)
        print(f"Average Valid Accuracy: {avg_valid_accuracy:.4f} ± {std_valid_accuracy:.4f}")
        print(f"Average Valid Loss: {avg_valid_loss:.4f} ± {std_valid_loss:.4f}")
        print(f"Average Generalization Error: {avg_generalization_error:.4f} ± {std_generalization_error:.4f}")
        print(f"Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        print(f"Average Test Macro F1: {avg_test_macro_f1:.4f} ± {std_test_macro_f1:.4f}")
        print(f"Average Test Precision: {avg_test_precision:.4f} ± {std_test_precision:.4f}")  # 新增
        print(f"Average Test Recall: {avg_test_recall:.4f} ± {std_test_recall:.4f}")          # 新增

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"\nTotal training time for Model: {total_duration:.2f} seconds")
    print(f"\nTotal training time for Model: {total_duration:.2f} seconds")
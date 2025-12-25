import os
import numpy as np
import torch
from tqdm import tqdm
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def euclidean_distances_M(samples, centers, M, squared=True):
    # 确保所有输入在同一设备
    samples = samples.to(torch.float32)
    centers = centers.to(torch.float32)
    M = M.to(torch.float32)

    samples_norm = (samples @ M) * samples
    samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = (centers @ M) * centers
        centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(M @ centers.T)
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)

    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances


def laplacian_M(samples, centers, bandwidth, M):
    # 确保所有输入在同一设备
    samples = samples.to(torch.float32)
    centers = centers.to(torch.float32)
    M = M.to(torch.float32)

    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1.0 / bandwidth
    kernel_mat = kernel_mat.float()
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def get_agop_gradients(molecular_features, sol, bandwidth, M):
    """直接返回AGOP矩阵，使用向量运算替代循环，移除手动分批逻辑"""
    n, C = molecular_features.shape
    # 将sol转移到与分子特征相同的设备，并确保维度正确
    sol_tensor = torch.from_numpy(sol).to(torch.float32).to(molecular_features.device)

    # 确保sol_tensor是1D张量并调整维度以匹配分子特征
    if sol_tensor.dim() > 1:
        sol_tensor = sol_tensor.squeeze()  # 移除多余的维度
    # 确保sol_tensor的形状为(N,)以匹配molecular_features的第一维度
    assert sol_tensor.shape[0] == n, f"sol维度不匹配: {sol_tensor.shape} vs {n}"

    # 计算梯度 (N, C)：扩展sol维度以便广播
    grad = sol_tensor.unsqueeze(1) * molecular_features

    # 向量运算替代循环：直接计算所有样本的外积和
    # 数学等价于：sum(grad[i].outer(grad[i]) for i in range(n)) / n
    agop_matrix = (grad.T @ grad) / n  # (C, C)

    return agop_matrix


def get_weighted_feature_vector(M, top_k=1, threshold=0):
    """
    从矩阵M的特征分解中筛选特征向量并按特征值加权

    参数:
        M: 待分解的矩阵 (C, C)
        top_k: 保留前k个特征向量（若指定则忽略threshold）
        threshold: 特征值累计占比阈值（默认保留90%能量）
    返回:
        weighted_vector: 加权融合后的特征向量 (C,)
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(M)  # 特征值升序排列
    # 从大到小排序（反转顺序）
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]  # 列向量对应排序后的特征值

    # 筛选特征向量
    if top_k is not None:
        # 直接保留前top_k个
        selected_eigenvalues = eigenvalues[:top_k]
        selected_vectors = eigenvectors[:, :top_k]
    else:
        # 按累计占比筛选（默认阈值0.9，若用户传入0则调整为0.9）
        threshold = 0.9 if threshold == 0 else threshold
        total = torch.sum(eigenvalues)
        if total == 0:  # 处理特殊情况，避免除以零
            logger.warning("特征值总和为0，返回第一个特征向量")
            return eigenvectors[:, 0]
        cumulative = torch.cumsum(eigenvalues, dim=0) / total
        # 找到满足阈值的最小k
        k = torch.where(cumulative >= threshold)[0][0] + 1  # +1是因为索引从0开始
        selected_eigenvalues = eigenvalues[:k]
        selected_vectors = eigenvectors[:, :k]

    # 按特征值加权（归一化权重）
    weights = selected_eigenvalues / torch.sum(selected_eigenvalues)  # 权重归一化
    weighted_vector = torch.sum(selected_vectors * weights, dim=1)  # 加权求和

    return weighted_vector


class RFM:
    def __init__(self, device=None, feature_vector_top_k=None, feature_vector_threshold=None):
        # 自动选择设备，优先GPU
        self.device = device if device is not None else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.M = None  # 特征权重矩阵 (C, C)
        self.alphas = None  # 核回归系数 (1, N)
        self.molecular_features_train = None  # 训练集分子特征 (N, C)
        self.bandwidth = None  # 核带宽
        self.reg = None  # 正则化系数
        self.pooling_iters = None  # 外层池化迭代次数
        self.inner_iters = None  # 内层M更新迭代次数
        # 特征向量选择参数
        self.feature_vector_top_k = feature_vector_top_k  # 保留前k个特征向量
        self.feature_vector_threshold = feature_vector_threshold  # 特征值累计占比阈值

    def fit(self, X_train, y_train, atom_mask=None, pooling_iters=3, inner_iters=2,
            bandwidth=10.0, reg=1e-3, verbose=False, feature_vector_top_k=None, feature_vector_threshold=None):
        self.bandwidth = bandwidth
        self.reg = reg
        self.pooling_iters = pooling_iters
        self.inner_iters = inner_iters
        # 优先使用fit方法传入的参数，若未传入则使用初始化时的参数
        self.feature_vector_top_k = feature_vector_top_k if feature_vector_top_k is not None else self.feature_vector_top_k
        self.feature_vector_threshold = feature_vector_threshold if feature_vector_threshold is not None else self.feature_vector_threshold

        N, K, C = X_train.shape  # N=样本数, K=原子数, C=特征数

        self.M = torch.eye(C, dtype=torch.float32, device=self.device)
        X_train_tensor = torch.from_numpy(X_train).to(torch.float32).to(self.device)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32).to(self.device)

        # 处理原子掩码（默认全1，即无padding）
        if atom_mask is not None:
            self.atom_mask_train = torch.from_numpy(atom_mask).to(torch.float32).to(self.device)  # [N, K]
        else:
            self.atom_mask_train = torch.ones(N, K, dtype=torch.float32, device=self.device)

        for pool_epoch in range(pooling_iters):
            if verbose:
                logger.info(f"\n===== Pooling Iteration {pool_epoch + 1}/{pooling_iters} =====")

            # 计算加权特征向量
            weighted_vector = get_weighted_feature_vector(
                self.M, top_k=self.feature_vector_top_k, threshold=self.feature_vector_threshold
            )

            # 用掩码屏蔽padding原子的得分
            atom_scores = torch.sum(X_train_tensor * weighted_vector, dim=2)  # [N, K]
            atom_scores = atom_scores * self.atom_mask_train  # padding位置得分归0

            atom_weights = torch.softmax(atom_scores, dim=1)  # [N, K]（padding权重为0）
            molecular_features = torch.sum(X_train_tensor * atom_weights.unsqueeze(2), dim=1)  # [N, C]
            self.molecular_features_train = molecular_features

            # 内层迭代
            for inner_epoch in range(inner_iters):
                K_train = laplacian_M(molecular_features, molecular_features, self.bandwidth, self.M)
                K_reg = K_train + self.reg * torch.eye(N, device=self.device)
                y_train_reshaped = y_train_tensor.reshape(-1, 1)
                self.alphas = torch.linalg.solve(K_reg, y_train_reshaped).squeeze()

                # 调用优化后的AGOP计算函数（无分批循环）
                agop_matrix = get_agop_gradients(
                    molecular_features, self.alphas.cpu().numpy(), self.bandwidth, self.M
                )
                self.M = agop_matrix + self.reg * torch.eye(C, dtype=torch.float32, device=self.device)

                if verbose:
                    trace = torch.trace(self.M).item()
                    preds = (self.alphas.unsqueeze(0) @ K_train).squeeze()
                    mse = torch.mean((preds - y_train_tensor) ** 2).item()
                    logger.info(f"  Inner Iter {inner_epoch + 1}/{inner_iters} | M Trace: {trace:.2f} | MSE: {mse:.4f}")

        logger.info("模型训练完成")
        return self

    def predict(self, X_test, atom_mask=None):
        X_test_tensor = torch.from_numpy(X_test).to(torch.float32).to(self.device)
        N_test, K, _ = X_test_tensor.shape

        # 处理测试集原子掩码
        if atom_mask is not None:
            atom_mask_test = torch.from_numpy(atom_mask).to(torch.float32).to(self.device)  # [N_test, K]
        else:
            atom_mask_test = torch.ones(N_test, K, dtype=torch.float32, device=self.device)

        # 计算加权特征向量（与训练一致）
        weighted_vector = get_weighted_feature_vector(
            self.M, top_k=self.feature_vector_top_k, threshold=self.feature_vector_threshold
        )

        # 用掩码屏蔽测试集padding原子
        atom_scores = torch.sum(X_test_tensor * weighted_vector, dim=2)  # [N_test, K]
        atom_scores = atom_scores * atom_mask_test  # padding位置得分归0

        atom_weights = torch.softmax(atom_scores, dim=1)  # [N_test, K]，即原子重要性
        molecular_features_test = torch.sum(
            X_test_tensor * atom_weights.unsqueeze(2), dim=1
        )  # [N_test, C]

        # 计算核矩阵并预测
        K_test = laplacian_M(
            self.molecular_features_train, molecular_features_test, self.bandwidth, self.M
        )  # [N_train, N_test]

        y_pred = (self.alphas.unsqueeze(0) @ K_test).squeeze().cpu().numpy()
        # 同时返回预测结果和测试集原子重要性（已转为CPU的numpy数组）
        return y_pred, atom_weights.cpu().numpy()

    def get_feature_importance(self):
        """获取特征重要性（基于加权特征向量）"""
        weighted_vector = get_weighted_feature_vector(
            self.M,
            top_k=self.feature_vector_top_k,
            threshold=self.feature_vector_threshold
        )
        return weighted_vector.cpu().numpy()

    def get_weighted_diag_matrix(self):
        """
        根据feature_vector_top_k筛选特征向量，按特征值加权得到向量后转换为对角矩阵
        返回：对角矩阵（对角线元素为加权向量的值）
        """
        if self.M is None:
            raise ValueError("模型尚未训练，无法获取M矩阵")

        # 1. 获取加权特征向量（基于训练好的M矩阵和top_k参数）
        weighted_vector = get_weighted_feature_vector(
            self.M,
            top_k=self.feature_vector_top_k,
            threshold=self.feature_vector_threshold
        )  # 形状: (C,)

        # 2. 将加权向量转换为对角矩阵（对角线元素为weighted_vector的值）
        weighted_diag_matrix = torch.diag(weighted_vector)  # 形状: (C, C)

        return weighted_diag_matrix

    def save_weighted_diag_matrix(self, save_path, format='npy'):
        """
        保存加权对角矩阵到文件

        参数:
            save_path: 保存路径（含文件名，不含后缀）
            format: 保存格式，可选 'npy'（默认）或 'pt'
        """
        if format not in ['npy', 'pt']:
            raise ValueError("支持的格式：'npy' 或 'pt'")

        # 获取对角矩阵
        diag_matrix = self.get_weighted_diag_matrix()

        # 确保矩阵在CPU上（避免设备问题）
        diag_matrix = diag_matrix.cpu()

        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"创建保存目录：{save_dir}")

        # 保存文件
        if format == 'npy':
            np.save(f"{save_path}.npy", diag_matrix.numpy())
            logger.info(f"对角矩阵已保存为Numpy格式：{save_path}.npy")
        elif format == 'pt':
            torch.save(diag_matrix, f"{save_path}.pt")
            logger.info(f"对角矩阵已保存为PyTorch格式：{save_path}.pt")


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据（示例）
    N_train, K, C = 100, 10, 5  # 100个训练样本，每个样本10个原子，每个原子5个特征
    X_train = np.random.randn(N_train, K, C).astype(np.float32)
    y_train = np.random.randn(N_train).astype(np.float32)
    atom_mask_train = np.ones((N_train, K), dtype=np.float32)  # 无padding

    N_test = 20
    X_test = np.random.randn(N_test, K, C).astype(np.float32)
    atom_mask_test = np.ones((N_test, K), dtype=np.float32)

    # 初始化并训练模型
    model = RFM(
        feature_vector_top_k=3,  # 保留前3个特征向量
        feature_vector_threshold=0.9  # 若未指定top_k，使用累计占比0.9
    )
    model.fit(
        X_train=X_train,
        y_train=y_train,
        atom_mask=atom_mask_train,
        pooling_iters=3,
        inner_iters=2,
        bandwidth=10.0,
        reg=1e-3,
        verbose=True
    )

    # 预测
    y_pred, atom_weights_test = model.predict(X_test, atom_mask=atom_mask_test)
    logger.info(f"预测结果形状：{y_pred.shape}")
    logger.info(f"测试集原子权重形状：{atom_weights_test.shape}")

    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    logger.info(f"特征重要性向量形状：{feature_importance.shape}")
    logger.info(f"特征重要性值：{feature_importance}")

    # 获取加权对角矩阵
    diag_matrix = model.get_weighted_diag_matrix()
    logger.info(f"加权对角矩阵形状：{diag_matrix.shape}")
    logger.info(f"加权对角矩阵：\n{diag_matrix.cpu().numpy()}")

    # 保存加权对角矩阵
    save_path = "./saved_matrices/weighted_diag_matrix"
    model.save_weighted_diag_matrix(save_path, format='npy')
    model.save_weighted_diag_matrix(save_path, format='pt')

    # 验证保存的文件
    if os.path.exists(f"{save_path}.npy"):
        loaded_npy = np.load(f"{save_path}.npy")
        logger.info(f"加载的npy文件形状：{loaded_npy.shape}")
    if os.path.exists(f"{save_path}.pt"):
        loaded_pt = torch.load(f"{save_path}.pt")
        logger.info(f"加载的pt文件形状：{loaded_pt.shape}")
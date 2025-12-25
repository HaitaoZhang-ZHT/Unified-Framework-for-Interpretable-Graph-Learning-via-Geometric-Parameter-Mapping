import argparse
import os
import numpy as np
import time  # 导入时间模块
import eigenpro_rfm as rfm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def format_time(seconds):
    """将秒数转换为 时:分:秒 格式，便于阅读"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def load_pubmed_data(data_path, random_seed=42):
    """加载pubmed数据集的节点特征和标签，支持指定随机种子"""
    # 读取内容文件
    content_file = os.path.join(data_path, "pubmed.content")
    with open(content_file, 'r') as f:
        lines = f.readlines()

    # 解析节点ID、特征和标签
    node_ids = []
    features = []
    labels = []

    for line in lines:
        parts = line.strip().split('\t')
        node_ids.append(parts[0])
        features.append([float(x) for x in parts[1:-1]])
        labels.append(parts[-1])

    # 转换为numpy数组
    X = np.array(features, dtype=np.float32)

    # 标签编码
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # 60%训练集，40%待分割
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=random_seed)

    # 将剩余40%对半分，得到20%验证集和20%测试集
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_seed)

    return X, y, train_idx, val_idx, test_idx, len(encoder.classes_)


def transform_features(X, M, normalize=True):
    """使用RFM变换矩阵M转换特征"""
    X_transformed = X @ M
    
    if normalize:
        norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除以零
        X_transformed = X_transformed / norms
        
    return X_transformed


def main():
    # 记录程序总开始时间
    start_total = time.perf_counter()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default="data/pubmed", type=str, help="pubmed数据集目录")
    parser.add_argument('-file', default="6Mpd.log", type=str, help="输出文件")
    parser.add_argument('--ep_iter', default=10, type=int, help="每次迭代的Epoch数")
    parser.add_argument('--max_iter', default=5, type=int, help="最大迭代次数")
    parser.add_argument('--start_seed', default=42, type=int, help="起始随机种子")
    parser.add_argument('--num_seeds', default=1, type=int, help="随机种子数量")
    parser.add_argument('--save_dir', default="models", type=str, help="模型保存目录")
    parser.add_argument('--depth', default=3, type=int, help="RFM层数")
    parser.add_argument('--load_existing', action='store_true', help="加载已保存的模型进行验证")

    args = parser.parse_args()
    data_dir = args.dir
    ep_iter = args.ep_iter
    max_iter = args.max_iter
    start_seed = args.start_seed
    num_seeds = args.num_seeds
    save_dir = args.save_dir
    depth = args.depth
    load_existing = args.load_existing

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 创建输出文件
    outf = open(args.file, "w")
    header = "Seed\tDataset\tSize\tNumFeatures\tNumClasses"
    for i in range(depth):
        header += f"\tLayer{i+1} Val Acc\tLayer{i+1} EpIter\tLayer{i+1} Iter\tLayer{i+1} Test Acc"
    header += "\tFinal Test Acc\tTotal Seed Time"  # 新增种子总时间列
    print(header, file=outf)

    # 运行多个随机种子的实验
    all_test_acc = []

    for seed in range(start_seed, start_seed + num_seeds):
        # 记录当前种子的开始时间
        start_seed_time = time.perf_counter()
        print(f"\n===== 运行实验 with seed={seed}, depth={depth} =====")

        # 加载pubmed数据（使用当前种子）
        print(f"加载pubmed数据集从 {data_dir}")
        load_start = time.perf_counter()
        X, y, train_idx, val_idx, test_idx, num_classes = load_pubmed_data(data_dir, random_seed=seed)
        load_time = time.perf_counter() - load_start
        print(f"数据加载完成，耗时: {format_time(load_time)}")

        print(f"节点数: {X.shape[0]}")
        print(f"特征维度: {X.shape[1]}")
        print(f"类别数: {num_classes}")
        print(f"训练样本数: {len(train_idx)}")
        print(f"验证样本数: {len(val_idx)}")
        print(f"测试样本数: {len(test_idx)}")

        # 超参数选择
        normalize_options = [True]
        
        # 存储每层的最佳参数
        best_M_list = []
        best_iter_list = []
        best_ep_iter_list = []
        best_normalize_list = []
        best_val_acc_list = []
        layer_test_acc_list = []
        layer_time_list = []  # 记录每层耗时

        # 当前层的输入特征
        X_train_current = X[train_idx]
        X_val_current = X[val_idx]
        X_test_current = X[test_idx]

        for layer in range(depth):
            # 记录当前层的开始时间
            start_layer = time.perf_counter()
            print(f"\n===== 开始第 {layer+1}/{depth} 层训练 =====")
            
            if load_existing:
                # 加载已保存的模型
                model_path = os.path.join(save_dir, f"pubmed_layer{layer+1}_best_model_seed_{seed}.npy")
                try:
                    best_M = np.load(model_path)
                    print(f"成功加载第 {layer+1} 层模型: {model_path}")
                    
                    # 假设这些参数与训练时相同
                    best_iter = max_iter
                    best_ep_iter = ep_iter
                    best_normalize = False
                    
                    # 评估已加载模型
                    test_acc = rfm.train(
                        X_train_current, y[train_idx],
                        X_test_current, y[test_idx],
                        num_classes, best_M,
                        iters=best_iter, ep_iter=best_ep_iter, L=1,
                        normalize=best_normalize
                    )
                    layer_test_acc_list.append(test_acc)
                    print(f"加载的第 {layer+1} 层模型测试准确率: {test_acc:.4f}")
                    
                    # 转换特征
                    X_train_current = transform_features(X_train_current, best_M, best_normalize)
                    X_val_current = transform_features(X_val_current, best_M, best_normalize)
                    X_test_current = transform_features(X_test_current, best_M, best_normalize)
                    
                except FileNotFoundError:
                    print(f"未找到第 {layer+1} 层模型文件，将重新训练")
                    load_existing = False  # 后续层也重新训练
                    continue  # 进入重新训练流程
            
            # 正常训练流程（如果未加载模型或加载失败）
            if not load_existing:
                best_acc, best_iter, best_M = 0, 0, None
                best_ep_iter = 0
                best_normalize = False

                for normalize in normalize_options:
                    print(f"尝试 normalize={normalize}")
                    acc, iter_v, M, ep_iter_val = rfm.hyperparam_train(
                        X_train_current, y[train_idx],
                        X_val_current, y[val_idx], num_classes,
                        ep_iter=ep_iter, iters=max_iter, L=1, normalize=normalize
                    )

                    if acc > best_acc:
                        best_acc = acc
                        best_iter = iter_v
                        best_M = M
                        best_normalize = normalize
                        best_ep_iter = ep_iter_val+1

                print(f"第 {layer+1} 层最佳验证准确率: {best_acc:.4f}")
                print(f"第 {layer+1} 层最佳参数: ep_iter={best_ep_iter}, iters={best_iter}, normalize={best_normalize}")

                # 保存当前层的最佳参数
                best_M_list.append(best_M)
                best_iter_list.append(best_iter)
                best_ep_iter_list.append(best_ep_iter)
                best_normalize_list.append(best_normalize)
                best_val_acc_list.append(best_acc)

                # 在当前层训练并评估测试集性能
                test_acc = rfm.train(
                    X_train_current, y[train_idx],
                    X_test_current, y[test_idx],
                    num_classes, best_M,
                    iters=best_iter, ep_iter=best_ep_iter, L=12,
                    normalize=best_normalize
                )
                layer_test_acc_list.append(test_acc)
                print(f"第 {layer+1} 层测试准确率: {test_acc:.4f}")

                # 保存当前层的M矩阵为npy文件
                model_path = os.path.join(save_dir, f"pubmed_layer{layer+1}_best_model_seed_{seed}.npy")
                np.save(model_path, best_M)
                print(f"第 {layer+1} 层最佳模型参数已保存到: {model_path}")
                
                # 验证保存的模型是否可以正确加载
                loaded_M = np.load(model_path)
                if np.array_equal(best_M, loaded_M):
                    print(f"验证通过: 保存的模型与原始模型一致")
                else:
                    print(f"警告: 保存的模型与原始模型不一致")

                # 使用当前层的M转换特征，作为下一层的输入
                X_train_current = transform_features(X_train_current, best_M, best_normalize)
                X_val_current = transform_features(X_val_current, best_M, best_normalize)
                X_test_current = transform_features(X_test_current, best_M, best_normalize)
            
            # 计算当前层耗时并记录
            layer_time = time.perf_counter() - start_layer
            layer_time_list.append(layer_time)
            print(f"第 {layer+1} 层训练完成，耗时: {format_time(layer_time)}")

        # 计算当前种子的总耗时
        seed_total_time = time.perf_counter() - start_seed_time
        print(f"\nSeed {seed} 实验完成，总耗时: {format_time(seed_total_time)}")

        # 最终测试准确率是最后一层的结果
        final_test_acc = layer_test_acc_list[-1] if layer_test_acc_list else 0
        all_test_acc.append(final_test_acc)

        # 输出结果到文件（包含当前种子耗时）
        dataset_name = "pubmed"
        result_line = f"{seed}\t{dataset_name}\t{X.shape[0]}\t{X.shape[1]}\t{num_classes}"
        for i in range(depth):
            result_line += f"\t{best_val_acc_list[i]:.4f}\t{best_ep_iter_list[i]}\t{best_iter_list[i]}\t{layer_test_acc_list[i]:.4f}"
        result_line += f"\t{final_test_acc * 100:.2f}\t{format_time(seed_total_time)}"
        print(result_line, file=outf)

    # 计算总运行时间
    total_time = time.perf_counter() - start_total
    print(f"\n===== 所有实验完成 =====")
    print(f"总运行时间: {format_time(total_time)}")
    print(f"平均测试准确率: {np.mean(all_test_acc) * 100:.2f}% ± {np.std(all_test_acc) * 100:.2f}%")
    print(f"测试准确率范围: [{min(all_test_acc) * 100:.2f}%, {max(all_test_acc) * 100:.2f}%]")
    print(f"所有准确率: {[f'{acc * 100:.2f}%' for acc in all_test_acc]}")

    # 将总时间写入文件
    print(f"\n总运行时间: {format_time(total_time)}", file=outf)
    print(f"结果已保存到 {args.file}")
    outf.close()


if __name__ == "__main__":
    main()
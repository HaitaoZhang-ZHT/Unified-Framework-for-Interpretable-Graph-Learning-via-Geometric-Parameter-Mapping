import argparse
import torch
import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """解析命令行参数，返回统一配置字典（适配稀疏注意力机制和M矩阵初始化）"""
    parser = argparse.ArgumentParser(
        description="HWNN (Heterogeneous Hypergraph Wavelet Neural Network) 配置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------------------------- 核心任务配置 --------------------------
    parser.add_argument("--task-type", type=str, default="graph_regression",
                        choices=["node_classification", "graph_classification", "graph_regression"],
                        help="任务类型：node_classification(节点分类)、graph_classification(图分类)、graph_regression(图回归)")
    parser.add_argument("--dataset", type=str, default="esol",
                        help="数据集名称（节点任务：cora/pubmed/dblp；图任务：mutag/ptc/esol/freesolv/lipo/qm9）")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="数据集存储路径（ESOL需放入 ./data/ESOL/ 目录）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="计算设备（auto：自动检测CUDA）")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子（控制模型初始化、数据打乱等）")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="数据集分割种子（单独控制分割可复现性，与全局种子分离）")
    parser.add_argument("--log-path", type=str, default="./logs",
                        help="日志、模型、注意力权重的保存根路径")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="实验名称（默认自动生成，格式：任务缩写_数据集_参数标识_注意力配置_M初始化配置）")

    # -------------------------- 数据预处理配置 --------------------------
    # 节点任务专用
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="节点任务训练集比例（仅node_classification有效，需与val/test比例总和为1）")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="节点任务验证集比例（仅node_classification有效）")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="节点任务测试集比例（仅node_classification有效）")

    # 分子任务专用（增强骨架分割配置）
    parser.add_argument("--split-type", type=str, default="random",
                        choices=["random", "scaffold"],
                        help="分子数据集分割方式：random（随机分割）、scaffold（骨架分割）")
    parser.add_argument("--split-balanced", action="store_true", default=False,
                        help="骨架分割是否启用平衡模式（避免大骨架组集中在某一集合，仅split-type=scaffold有效）")
    parser.add_argument("--mol-feat-type", type=str, default="rdkit+graph",
                        choices=["rdkit", "graph", "rdkit+graph"],
                        help="分子原子特征类型：rdkit（化学属性）、graph（拓扑嵌入）、rdkit+graph（融合特征）")
    parser.add_argument("--normalize-mol-props", action="store_true", default=False,
                        help="是否标准化分子属性（如分子量，仅分子任务有效）")

    # 通用预处理
    parser.add_argument("--normalize-features", action="store_true", default=False,
                        help="是否对节点/原子特征进行Z-score归一化（仅用训练集统计量）")
    parser.add_argument("--use-node-degree", action="store_true", default=False,
                        help="是否将节点度作为额外特征（分子任务中为原子连接度）")

    # -------------------------- 超图构建配置 --------------------------
    # 节点任务超边类型
    parser.add_argument("--node-hyper-types", type=str, default="neighbor,attribute,cluster,community",
                        help="节点任务超边类型（逗号分隔，支持：neighbor(邻居)、attribute(属性)、cluster(聚类)、community(社区)）")
    parser.add_argument("--node-hyper-repeats", type=str, default="3,1,1,1",
                        help="节点任务每种超边的重复次数（需与node-hyper-types长度一致）")

    # 图任务/分子任务超边类型
    parser.add_argument("--graph-hyper-types", type=str, default="bond,functional_group,substructure,similarity",
                        help="图任务超边类型（逗号分隔，分子任务推荐：bond(化学键)、functional_group(官能团)、substructure(子结构)、similarity(特征相似)、hydrogen_bond(氢键)）")
    parser.add_argument("--graph-hyper-repeats", type=str, default="1,1,1,1",
                        help="图任务每种超边的重复次数（需与graph-hyper-types长度一致）")
    parser.add_argument("--graph-edge-weight", type=str, default="bond_strength",
                        choices=["uniform", "similarity", "bond_strength"],
                        help="图任务超边权重计算方式：uniform（均匀权重）、similarity（特征相似度）、bond_strength（化学键强度，仅分子任务有效）")

    # 通用超图参数
    parser.add_argument("--phi-hop", type=int, default=1,
                        help="节点任务邻居超边的φ跳数（控制邻居范围，1跳=直接邻居）")
    parser.add_argument("--n-clusters", type=int, default=100,
                        help="节点任务聚类超边的簇数（KMeans聚类数量）")
    parser.add_argument("--hyperedge-size-range", type=str, default="2,20",
                        help="超边大小范围（最小,最大），过滤过小/过大的超边（分子任务推荐2-20）")
    parser.add_argument("--substructure-depth", type=int, default=4,
                        help="分子子结构超边的深度（控制子结构范围，1=单原子+直接邻居，2=扩展1层）")

    # -------------------------- 稀疏注意力配置（新增） --------------------------
    parser.add_argument("--sparse-attention-threshold", type=int, default=5000,
                        help="启用稀疏注意力的节点数阈值（节点数超过此值自动使用稀疏注意力，避免显存溢出）")
    parser.add_argument("--force-sparse-attention", action="store_true", default=False,
                        help="强制使用稀疏注意力（即使节点数较少，用于测试或特殊需求）")
    parser.add_argument("--sparse-matrix-threshold", type=int, default=5000,
                        help="超图构建时启用稀疏矩阵存储的节点数阈值（与稀疏注意力阈值保持一致）")

    # -------------------------- 模型核心配置（适配稀疏注意力和M矩阵初始化） --------------------------
    parser.add_argument("--input-dim", type=int, default=None,
                        help="输入特征维度（自动从数据获取，无需手动设置）")
    parser.add_argument("--hidden-dim", type=int, default=260,
                        help="HWNN卷积层隐藏维度（分子任务推荐256-512，适配复杂分子结构）")
    parser.add_argument("--output-dim", type=int, default=None,
                        help="输出维度（分类任务=类别数，回归任务=1，自动从数据获取）")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="HWNN卷积层数（分子任务推荐2-3层，避免过拟合）")
    parser.add_argument("--K1", type=int, default=3,
                        help="多项式近似Theta的正幂次阶数（控制超图卷积的局部性，推荐2-3）")
    parser.add_argument("--K2", type=int, default=3,
                        help="多项式近似Theta^T的幂次阶数（与K1配合，推荐相同值）")
    parser.add_argument("--approx", action="store_true", default=True,
                        help="是否使用多项式近似（避免拉普拉斯特征分解，大幅提升效率）")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout概率（卷积层+池化层共享，分子任务推荐0.2-0.3）")

    # M矩阵初始化配置（新增）
    parser.add_argument("--use-M-init-linear-input", action="store_true", default=True,
                        help="输入层是否使用M矩阵初始化线性权重（推荐启用，利用预训练知识）")
    parser.add_argument("--use-M-init-linear-hidden", action="store_true", default=False,
                        help="隐藏层是否使用M矩阵初始化线性权重（可选，通常Xavier初始化效果更好）")
    parser.add_argument("--M-matrix-path", type=str, default="./logs/esol/esol_rfm_embedding_model.npz",
                        help="M矩阵文件路径（包含预训练嵌入矩阵的npz文件）")

    # 卷积注意力与池化注意力参数
    parser.add_argument("--use-conv-attention", action="store_true", default=True,
                        help="是否启用超图卷积层的节点注意力（捕捉节点间拓扑关联重要性，推荐启用）")
    parser.add_argument("--conv-attention-heads", type=int, default=2,
                        help="卷积层注意力头数（多头部捕捉不同拓扑关联，需能被hidden-dim整除，推荐2-4）")
    parser.add_argument("--use-pool-attention", action="store_true", default=False,
                        help="是否启用图/分子任务的注意力池化（节点聚合阶段，学习原子重要性，分子任务推荐启用）")

    # -------------------------- 训练配置 --------------------------
    # 图任务/分子任务专用
    parser.add_argument("--batch-size", type=int, default=8,
                        help="图任务批次大小（分子任务推荐16-32，适配GPU内存）")

    # 通用训练参数
    parser.add_argument("--epochs", type=int, default=5000,
                        help="训练总轮次（分子任务推荐5000-10000，早停机制会自动终止）")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="初始学习率（分子任务推荐0.0005-0.001，Adam优化器）")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减（L2正则化，控制过拟合，推荐1e-4-1e-3）")
    parser.add_argument("--patience", type=int, default=100,
                        help="早停耐心值（验证指标连续下降N轮后停止，分子任务推荐40-100）")
    parser.add_argument("--lr-scheduler", type=str, default="reduce_on_plateau",
                        choices=["none", "step", "reduce_on_plateau"],
                        help="学习率调度器：none（无）、step（固定步长衰减）、reduce_on_plateau（验证指标停滞时衰减）")
    parser.add_argument("--lr-scheduler-patience", type=int, default=30,
                        help="调度器耐心值（仅reduce_on_plateau有效，推荐为早停耐心值的2-3倍）")

    # 分子任务训练增强
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="梯度裁剪阈值（防止梯度爆炸，分子任务必设，推荐5.0-10.0）")
    parser.add_argument("--record-attn-interval", type=int, default=10,
                        help="注意力权重记录间隔（每N轮保存一次卷积+池化注意力，仅图回归任务有效）")

    # -------------------------- 输出配置 --------------------------
    parser.add_argument("--save-model", action="store_true", default=True,
                        help="是否保存最佳模型（基于验证集指标，保存在 log-path/models/ 目录）")
    parser.add_argument("--plot-curves", action="store_true", default=True,
                        help="是否绘制学习曲线（损失+指标，保存在 log-path/ 目录，含注意力配置标识）")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="验证间隔（每N轮进行一次验证，推荐1-5，小数据集可设1）")
    parser.add_argument("--save-attn-weights", action="store_true", default=True,
                        help="是否保存注意力权重（含卷积层注意力+池化层原子权重，保存在 log-path/attention_weights/ 目录）")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="是否打印详细训练日志（每轮输出损失、指标，含注意力配置信息）")

    # 解析参数
    args = parser.parse_args()
    config = vars(args)  # 转换为字典

    # -------------------------- 配置后处理与校验（增强版：适配稀疏注意力和M矩阵初始化） --------------------------
    # 1. 设备自动选择
    if config["device"] == "auto":
        old_device = "cuda" if torch.cuda.is_available() else "cpu"
        config["device"] = old_device
        if old_device == "cuda":
            print(f"[设备选择] 检测到CUDA可用，使用设备：cuda（设备编号：{torch.cuda.current_device()}）")
        else:
            print(f"[设备选择] 未检测到CUDA，使用设备：cpu（注意：分子任务训练可能较慢）")
    else:
        print(f"[设备选择] 手动指定设备：{config['device']}")

    # 2. 超边类型与重复次数校验
    # 节点任务
    config["node_hyper_types"] = [t.strip() for t in config["node_hyper_types"].split(",")]
    config["node_hyper_repeats"] = list(map(int, config["node_hyper_repeats"].split(",")))
    if len(config["node_hyper_types"]) != len(config["node_hyper_repeats"]):
        raise ValueError(
            f"节点任务超边配置错误：node-hyper-types（{len(config['node_hyper_types'])}种）与 "
            f"node-hyper-repeats（{len(config['node_hyper_repeats'])}次）长度必须一致\n"
            f"当前超边类型：{config['node_hyper_types']}\n当前重复次数：{config['node_hyper_repeats']}"
        )

    # 图任务/分子任务
    config["graph_hyper_types"] = [t.strip() for t in config["graph_hyper_types"].split(",")]
    config["graph_hyper_repeats"] = list(map(int, config["graph_hyper_repeats"].split(",")))
    if len(config["graph_hyper_types"]) != len(config["graph_hyper_repeats"]):
        raise ValueError(
            f"图任务超边配置错误：graph-hyper-types（{len(config['graph_hyper_types'])}种）与 "
            f"graph-hyper-repeats（{len(config['graph_hyper_repeats'])}次）长度必须一致\n"
            f"当前超边类型：{config['graph_hyper_types']}\n当前重复次数：{config['graph_hyper_repeats']}"
        )

    # 3. 超边大小范围处理
    try:
        min_h, max_h = map(int, config["hyperedge_size_range"].split(","))
    except ValueError:
        raise ValueError(
            f"超边大小范围格式错误：{config['hyperedge_size_range']}，需为逗号分隔的两个整数（如 2,20）"
        )
    config["hyperedge_size_range"] = (min_h, max_h)
    if min_h < 2 or max_h < min_h:
        raise ValueError(
            f"超边大小范围无效：{min_h},{max_h}，需满足 最小≥2 且 最大≥最小\n"
            f"分子任务推荐范围：2-20（过小超边无意义，过大超边计算效率低）"
        )

    # 4. 稀疏注意力配置校验（新增）
    if config["sparse_attention_threshold"] < 100:
        print(f"⚠️ [稀疏注意力警告] 阈值{config['sparse_attention_threshold']}过低，可能影响小图性能，建议≥1000")

    if config["force_sparse_attention"]:
        print(f"[稀疏注意力] 强制启用稀疏注意力模式（即使节点数较少）")

    # 确保稀疏矩阵阈值与稀疏注意力阈值一致
    config["sparse_matrix_threshold"] = config["sparse_attention_threshold"]
    print(f"[稀疏配置] 稀疏注意力阈值：{config['sparse_attention_threshold']}节点 | "
          f"稀疏矩阵阈值：{config['sparse_matrix_threshold']}节点")

    # 5. M矩阵路径校验（新增）
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        if not os.path.exists(config["M_matrix_path"]):
            print(f"⚠️ [M矩阵警告] 指定的M矩阵文件不存在：{config['M_matrix_path']}")
            print(f"  将回退到Xavier初始化")
            # 自动禁用M矩阵初始化
            config["use_M_init_linear_input"] = False
            config["use_M_init_linear_hidden"] = False
        else:
            print(f"[M矩阵配置] 使用M矩阵文件：{config['M_matrix_path']}")
            print(f"  输入层M初始化：{'启用' if config['use_M_init_linear_input'] else '禁用'}")
            print(f"  隐藏层M初始化：{'启用' if config['use_M_init_linear_hidden'] else '禁用'}")

    # 6. 实验名称自动生成（增强：加入稀疏注意力和M矩阵初始化标识）
    if config["experiment_name"] is None:
        # 任务缩写映射
        task_abbr = {
            "node_classification": "nc",
            "graph_classification": "gc",
            "graph_regression": "gr"
        }[config["task_type"]]

        # 分子任务补充分割标识
        split_suffix = ""
        if config["task_type"] == "graph_regression" and config["dataset"].lower() in ["esol", "freesolv", "lipo",
                                                                                       "qm9"]:
            balance_suffix = "_balanced" if config["split_balanced"] else "_unbalanced"
            split_suffix = f"_{config['split_type']}{balance_suffix}"

        # 注意力配置标识
        attn_suffix = f"_cattn_{1 if config['use_conv_attention'] else 0}_pattn_{1 if config['use_pool_attention'] else 0}"

        # 稀疏注意力标识（新增）
        sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                         "sparse_attention_threshold"] < 10000 else ""
        if config["force_sparse_attention"]:
            sparse_suffix = "_force_sparse"

        # M矩阵初始化标识（新增）
        M_init_suffix = ""
        if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
            input_flag = "1" if config["use_M_init_linear_input"] else "0"
            hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
            M_init_suffix = f"_Minit_{input_flag}{hidden_flag}"

        # 生成最终名称
        config["experiment_name"] = (
            f"{task_abbr}_{config['dataset']}_l{config['num_layers']}_d{config['hidden_dim']}_s{config['seed']}"
            f"{split_suffix}{attn_suffix}{sparse_suffix}{M_init_suffix}"
        )
    print(f"[实验配置] 实验名称：{config['experiment_name']}")

    # 7. 日志路径构建
    if config["task_type"] == "graph_regression" and config["dataset"].lower() in ["esol", "freesolv", "lipo", "qm9"]:
        task_suffix = f"graph_reg_{config['dataset']}_{config['split_type']}"
    else:
        task_suffix = {
            "node_classification": "node_cls",
            "graph_classification": "graph_cls",
            "graph_regression": "graph_reg"
        }[config["task_type"]]

    config["log_path"] = os.path.join(
        config["log_path"],
        task_suffix,
        config["experiment_name"]
    )
    os.makedirs(config["log_path"], exist_ok=True)
    os.makedirs(os.path.join(config["log_path"], "models"), exist_ok=True)
    if config["save_attn_weights"] and config["task_type"] in ["graph_classification", "graph_regression"]:
        os.makedirs(os.path.join(config["log_path"], "attention_weights"), exist_ok=True)
    print(f"[路径配置] 日志根路径：{config['log_path']}")
    print(f"[路径配置] 模型保存路径：{os.path.join(config['log_path'], 'models')}")
    if config["save_attn_weights"]:
        print(f"[路径配置] 注意力权重路径：{os.path.join(config['log_path'], 'attention_weights')}")

    # 8. 数据集与任务匹配校验
    node_datasets = {"cora", "pubmed", "dblp", "citeseer"}
    graph_cls_datasets = {"mutag", "ptc", "proteins", "imdb", "collab"}
    graph_reg_datasets = {"esol", "freesolv", "lipo", "qm9"}

    if config["task_type"] == "node_classification" and config["dataset"] not in node_datasets:
        print(f"⚠️ [数据集警告] 节点分类推荐数据集：{sorted(node_datasets)}，当前使用 {config['dataset']}\n"
              f"  若需使用自定义数据集，请确保数据格式与Cora一致（含 .content 和 .cites 文件）")

    if config["task_type"] == "graph_classification" and config["dataset"] not in graph_cls_datasets:
        print(f"⚠️ [数据集警告] 图分类推荐数据集：{sorted(graph_cls_datasets)}，当前使用 {config['dataset']}\n"
              f"  若需使用自定义数据集，请确保pkl文件包含 nodes/edges/edge_types/features/label 字段")

    if config["task_type"] == "graph_regression" and config["dataset"] not in graph_reg_datasets:
        print(f"⚠️ [数据集警告] 图回归（分子）推荐数据集：{sorted(graph_reg_datasets)}，当前使用 {config['dataset']}\n"
              f"  若需使用自定义分子数据集，请确保pkl文件包含 smiles/labels/all_atoms/embeddings 字段")

    # 9. 分子任务特殊参数校验
    if config["task_type"] == "graph_regression" and config["dataset"].lower() in graph_reg_datasets:
        # 特征类型校验
        if config["mol_feat_type"] not in ["rdkit", "graph", "rdkit+graph"]:
            raise ValueError(
                f"分子任务特征类型错误：{config['mol_feat_type']}，仅支持 rdkit/graph/rdkit+graph\n"
                f"  - rdkit：化学属性（原子序数、电荷等）\n"
                f"  - graph：拓扑嵌入（预计算的原子嵌入）\n"
                f"  - rdkit+graph：融合特征（推荐，兼顾化学属性与拓扑信息）"
            )

        # 稀疏注意力推荐（新增）
        if config["use_conv_attention"]:
            print(f"[稀疏注意力] 分子任务推荐启用稀疏注意力（当前阈值：{config['sparse_attention_threshold']}节点）\n"
                  f"  可防止大分子图的显存溢出，同时保持小分子的计算效率")

        # 池化注意力推荐
        if not config["use_pool_attention"]:
            print(f"⚠️ [分子任务警告] 分子任务推荐启用注意力池化（添加 --use-pool-attention 参数）\n"
                  f"  禁用可能导致模型无法区分关键原子（如羟基/羧基）与非关键原子，预测性能可能大幅下降")

        # 卷积注意力推荐
        if not config["use_conv_attention"]:
            print(f"⚠️ [分子任务警告] 分子任务推荐启用卷积层注意力（--use-conv-attention）\n"
                  f"  禁用会导致模型无法捕捉原子间拓扑关联的重要性，建议启用以提升性能")

        # 骨架分割配置校验
        if config["split_type"] == "scaffold":
            print(f"[分子分割] 启用骨架分割（Scaffold Split）\n"
                  f"  - 分割种子：{config['split_seed']}\n"
                  f"  - 平衡模式：{'启用' if config['split_balanced'] else '禁用'}\n"
                  f"  - 功能：确保训练/验证/测试集无相同骨架，真实评估泛化能力")
        else:
            print(f"[分子分割] 启用随机分割（Random Split）\n"
                  f"  提示：骨架分割（--split-type scaffold）更适合分子任务的泛化能力评估")

        # 梯度裁剪强制校验
        if config["grad_clip"] <= 0:
            raise ValueError(
                f"分子任务梯度裁剪错误：{config['grad_clip']}，需设为正数（推荐5.0-10.0）\n"
                f"  分子任务特征维度高，易出现梯度爆炸，梯度裁剪是必要措施"
            )

    # 10. 训练集比例校验
    if config["task_type"] == "node_classification":
        total_ratio = config["train_ratio"] + config["val_ratio"] + config["test_ratio"]
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(
                f"节点任务比例错误：训练{config['train_ratio']} + 验证{config['val_ratio']} + 测试{config['test_ratio']} = {total_ratio:.4f}\n"
                f"  需确保比例总和为1（允许±0.01的浮点数误差），当前偏差过大"
            )

    # 11. 模型参数合理性校验
    if config["use_conv_attention"] and config["hidden_dim"] % config["conv_attention_heads"] != 0:
        raise ValueError(
            f"卷积注意力配置错误：隐藏维度 {config['hidden_dim']} 无法被卷积注意力头数 {config['conv_attention_heads']} 整除\n"
            f"  需满足 hidden_dim % conv_attention_heads == 0（每个注意力头的维度需一致）\n"
            f"  推荐组合：hidden_dim=256 → heads=2/4/8；hidden_dim=512 → heads=2/4/8/16"
        )

    # 12. 稀疏注意力与节点数兼容性检查（新增）
    if config["task_type"] == "node_classification" and config["use_conv_attention"]:
        # 常见节点数据集的节点数预估
        dataset_node_counts = {
            "cora": 2708,
            "pubmed": 19717,
            "dblp": 17716,
            "citeseer": 3327
        }

        expected_nodes = dataset_node_counts.get(config["dataset"].lower(), 0)
        if expected_nodes > 0:
            if expected_nodes > config["sparse_attention_threshold"]:
                print(
                    f"[稀疏兼容性] 数据集{config['dataset']}预计有{expected_nodes}节点 > 阈值{config['sparse_attention_threshold']}\n"
                    f"  将自动启用稀疏注意力，避免显存溢出")
            else:
                print(
                    f"[稀疏兼容性] 数据集{config['dataset']}预计有{expected_nodes}节点 ≤ 阈值{config['sparse_attention_threshold']}\n"
                    f"  将使用稠密注意力，保持最佳性能")

    # 13. 池化注意力专属校验
    if config["use_pool_attention"] and config["task_type"] == "node_classification":
        print(f"⚠️ [注意力配置警告] 节点任务无需启用注意力池化（--use-pool-attention=True）\n"
              f"  节点任务直接输出节点级预测，无需聚合为图特征，建议设置 --no-use-pool-attention 禁用")

    # -------------------------- 配置摘要打印（增强：显示稀疏注意力和M矩阵初始化状态） --------------------------
    print("\n" + "=" * 80)
    print(f"HWNN 实验配置摘要 | 实验名称：{config['experiment_name']} | 任务类型：{config['task_type']}")
    print("=" * 80)

    # 基础配置
    base_info = (
        f"基础配置: \n"
        f"  数据集：{config['dataset']} | 数据路径：{config['data_path']}\n"
        f"  设备：{config['device']} | 全局种子：{config['seed']} | 分割种子：{config['split_seed']}\n"
        f"  日志路径：{config['log_path']}"
    )
    if config["task_type"] == "graph_regression" and config["dataset"].lower() in graph_reg_datasets:
        base_info += f"\n  分割方式：{config['split_type']} | 平衡分割：{'是' if config['split_balanced'] else '否'}"
    print(base_info)

    # 稀疏注意力配置（新增）
    if config["use_conv_attention"]:
        sparse_mode = "强制稀疏" if config["force_sparse_attention"] else "自动切换"
        print(f"稀疏注意力: 模式={sparse_mode} | 阈值={config['sparse_attention_threshold']}节点")

    # M矩阵初始化配置（新增）
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        print(f"M矩阵初始化: 输入层={'启用' if config['use_M_init_linear_input'] else '禁用'} | "
              f"隐藏层={'启用' if config['use_M_init_linear_hidden'] else '禁用'}")

    # 特征与预处理配置
    print(f"特征配置: " + {
        "node_classification": (
            f"归一化：{'是' if config['normalize_features'] else '否'} | 节点度特征：{'是' if config['use_node_degree'] else '否'}\n"
            f"  数据划分：训练{config['train_ratio']} | 验证{config['val_ratio']} | 测试{config['test_ratio']}"
        ),
        "graph_classification": (
            f"归一化：{'是' if config['normalize_features'] else '否'} | 节点度特征：{'是' if config['use_node_degree'] else '否'}"
        ),
        "graph_regression": (
            f"分子特征：{config['mol_feat_type']} | 属性归一化：{'是' if config['normalize_mol_props'] else '否'}\n"
            f"  节点度特征：{'是' if config['use_node_degree'] else '否'} | 子结构深度：{config['substructure_depth']}"
        )
    }[config["task_type"]])

    # 超图配置
    print(f"超图配置: " + {
        "node_classification": (
            f"超边类型：{config['node_hyper_types']} | 重复次数：{config['node_hyper_repeats']}\n"
            f"  邻居跳数：{config['phi_hop']} | 聚类数：{config['n_clusters']} | 超边大小范围：{config['hyperedge_size_range']}"
        ),
        "graph_classification": (
            f"超边类型：{config['graph_hyper_types']} | 重复次数：{config['graph_hyper_repeats']}\n"
            f"  权重方式：{config['graph_edge_weight']} | 超边大小范围：{config['hyperedge_size_range']}"
        ),
        "graph_regression": (
            f"超边类型：{config['graph_hyper_types']} | 重复次数：{config['graph_hyper_repeats']}\n"
            f"  权重方式：{config['graph_edge_weight']} | 超边大小范围：{config['hyperedge_size_range']} | 子结构深度：{config['substructure_depth']}"
        )
    }[config["task_type"]])

    # 模型配置（动态显示稀疏注意力和M矩阵初始化状态）
    conv_attn_info = f"启用（头数：{config['conv_attention_heads']}）" if config["use_conv_attention"] else "禁用"
    pool_attn_info = f"启用" if config["use_pool_attention"] else "禁用"
    sparse_info = f" | 稀疏阈值：{config['sparse_attention_threshold']}节点" if config["use_conv_attention"] else ""

    # M矩阵初始化信息（新增）
    M_init_info = ""
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        M_init_info = f"\n  M矩阵初始化：输入层={'是' if config['use_M_init_linear_input'] else '否'} | 隐藏层={'是' if config['use_M_init_linear_hidden'] else '否'}"

    print(f"模型配置: "
          f"层数：{config['num_layers']} | 隐藏维度：{config['hidden_dim']} | "
          f"多项式阶数：({config['K1']},{config['K2']}) | Dropout：{config['dropout']}\n"
          f"  卷积层注意力：{conv_attn_info}{sparse_info} | 池化层注意力：{pool_attn_info} | "
          f"近似计算：{'启用' if config['approx'] else '禁用'}{M_init_info}")

    # 训练配置
    print(f"训练配置: "
          f"总轮次：{config['epochs']} | 学习率：{config['lr']} | 权重衰减：{config['weight_decay']}\n"
          f"  早停耐心值：{config['patience']} | 调度器：{config['lr_scheduler']} | " + {
              "node_classification": f"验证间隔：{config['eval_interval']}",
              "graph_classification": f"批次大小：{config['batch_size']} | 验证间隔：{config['eval_interval']}",
              "graph_regression": (
                  f"批次大小：{config['batch_size']} | 梯度裁剪：{config['grad_clip']} | "
                  f"验证间隔：{config['eval_interval']} | 注意力记录间隔：{config['record_attn_interval']}"
              )
          }[config["task_type"]])

    # 输出配置
    print(f"输出配置: "
          f"保存模型：{'是' if config['save_model'] else '否'} | 绘制曲线：{'是' if config['plot_curves'] else '否'}\n"
          f"  保存注意力：{'是' if config['save_attn_weights'] else '否'}（卷积层拓扑注意力"
          f"{' + 池化层原子权重' if config['use_pool_attention'] else ''}） | "
          f"详细日志：{'是' if config['verbose'] else '否'}")
    print("=" * 80 + "\n")

    return config


if __name__ == "__main__":
    # 测试配置解析（调试用）
    config = get_config()
import os
import torch
import numpy as np
import logging
import pickle
from datetime import datetime
import traceback

from data_preprocessor import DataPreprocessor
from hypergraph_builder import HypergraphBuilder
from hwnn_model import HWNN
from hwnn_trainer import HWNNTrainer
from hwnn_config import get_config


def get_task_abbr(task_type):
    """统一生成任务类型缩写，避免重复定义"""
    task_abbr_map = {
        "node_classification": "nc",
        "graph_classification": "gc",
        "graph_regression": "gr"
    }
    if task_type not in task_abbr_map:
        raise ValueError(f"不支持的任务类型：{task_type}，可选类型：{list(task_abbr_map.keys())}")
    return task_abbr_map[task_type]


def validate_and_create_dir(file_path, description):
    """通用目录验证与创建函数"""
    if not file_path:
        raise ValueError(f"{description}路径不能为空")

    dir_path = os.path.dirname(file_path)

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"[目录创建成功] {description}目录：{dir_path}")

        # 验证目录可写性
        test_file = os.path.join(dir_path, ".write_test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True

    except PermissionError:
        raise PermissionError(f"没有权限在{dir_path}创建{description}目录，请检查文件夹权限")
    except OSError as e:
        raise OSError(f"创建{description}目录失败：{str(e)}\n路径：{dir_path}")


def setup_logging(config):
    """配置日志（适配稀疏注意力配置和M矩阵初始化）"""
    # 使用统一函数生成任务缩写
    task_abbr = get_task_abbr(config["task_type"])

    # 简化的任务子路径：使用缩写和更短的标识
    if config["task_type"] == "graph_regression" and config["dataset"].lower() in ["esol", "freesolv", "lipo", "qm9"]:
        # 平衡模式缩写：b=balanced, ub=unbalanced
        balance_suffix = "_b" if config["split_balanced"] else "_ub"
        # 注意力配置缩写：ca=conv_attn, pa=pool_attn; 1=启用,0=禁用
        attn_suffix = f"_ca{1 if config['use_conv_attention'] else 0}_pa{1 if config['use_pool_attention'] else 0}"
        # 稀疏注意力标识
        sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                         "sparse_attention_threshold"] < 10000 else ""
        if config["force_sparse_attention"]:
            sparse_suffix = "_fsparse"
        # M矩阵初始化标识（新增）
        M_init_suffix = ""
        if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
            input_flag = "1" if config["use_M_init_linear_input"] else "0"
            hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
            M_init_suffix = f"_M{input_flag}{hidden_flag}"

        # 数据集名取前4个字符
        dataset_abbr = config["dataset"].lower()[:4]
        # 分割方式取前3个字符
        split_abbr = config["split_type"][:3] if config["split_type"] else "rnd"

        task_suffix = f"{task_abbr}_{dataset_abbr}_{split_abbr}{balance_suffix}{attn_suffix}{sparse_suffix}{M_init_suffix}"
    else:
        # 节点任务也添加稀疏注意力标识和M矩阵标识
        sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                         "sparse_attention_threshold"] < 10000 else ""
        if config["force_sparse_attention"]:
            sparse_suffix = "_fsparse"
        # M矩阵初始化标识（新增）
        M_init_suffix = ""
        if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
            input_flag = "1" if config["use_M_init_linear_input"] else "0"
            hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
            M_init_suffix = f"_M{input_flag}{hidden_flag}"
        task_suffix = f"{task_abbr}{sparse_suffix}{M_init_suffix}"

    # 日志文件名：包含稀疏注意力和M矩阵信息
    exp_name_short = config["experiment_name"].split('_')[0]  # 只取实验名的第一部分
    log_filename = (
        f"hwnn_{task_suffix}_{exp_name_short}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    )
    log_filepath = os.path.join(config["log_path"], log_filename)

    # 验证路径长度
    if len(log_filepath) > 250:  # 预留一些空间
        print(f"[警告] 路径长度接近Windows限制({len(log_filepath)}/260)：{log_filepath}")

    # 验证并创建目录
    try:
        validate_and_create_dir(log_filepath, "日志")
    except Exception as e:
        print(f"[致命错误] 日志目录准备失败：{str(e)}")
        traceback.print_exc()
        raise

    # 配置日志处理器
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"[日志配置] 日志文件已保存至：{log_filepath}")
    logger.info(f"[路径信息] 路径长度：{len(log_filepath)}字符")
    return logger


def set_seed(config):
    """设置全局随机种子（添加稀疏注意力和M矩阵初始化信息）"""
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if config["device"] == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cuda_info = f" | CUDA设备：{torch.cuda.get_device_name(0)}"
    else:
        cuda_info = ""

    # 稀疏注意力配置信息
    sparse_info = ""
    if config["use_conv_attention"]:
        sparse_mode = "强制稀疏" if config["force_sparse_attention"] else "自动切换"
        sparse_info = f" | 稀疏注意力：{sparse_mode}（阈值：{config['sparse_attention_threshold']}节点）"

    # M矩阵初始化信息（新增）
    M_init_info = ""
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        M_init_info = f" | M矩阵初始化：输入层={'启用' if config['use_M_init_linear_input'] else '禁用'} | 隐藏层={'启用' if config['use_M_init_linear_hidden'] else '禁用'}"

    if config["task_type"] == "graph_regression" and config["dataset"].lower() in ["esol", "freesolv", "lipo", "qm9"]:
        logging.info(
            f"[种子配置] 全局种子：{config['seed']} | 分割种子：{config['split_seed']} | "
            f"分割方式：{config['split_type']} | 平衡模式：{'启用' if config['split_balanced'] else '禁用'}{sparse_info}{M_init_info} | "
            f"注意力配置：卷积={config['use_conv_attention']}（头数{config['conv_attention_heads']}） | 池化={config['use_pool_attention']} | "
            f"设备：{config['device']}{cuda_info}"
        )
    else:
        logging.info(f"[种子配置] 全局种子：{config['seed']}{sparse_info}{M_init_info} | 设备：{config['device']}{cuda_info}")


def save_mol_config(config, hypergraph_data, save_path):
    """保存分子任务配置（添加稀疏注意力和M矩阵初始化信息）"""
    if config["dataset"].lower() == "esol" and config["task_type"] == "graph_regression":
        mol_config = {
            "fg_config": hypergraph_data.get("fg_config", {}),
            "hbond_config": hypergraph_data.get("hbond_config", {}),
            "graph_hyper_types": config["graph_hyper_types"],
            "graph_hyper_repeats": config["graph_hyper_repeats"],
            "graph_edge_weight": config["graph_edge_weight"],
            "hyperedge_size_range": config["hyperedge_size_range"],
            "substructure_depth": config["substructure_depth"],
            "split_type": config["split_type"],
            "split_balanced": config["split_balanced"],
            "split_seed": config["split_seed"],
            "train_ratio": config.get("train_ratio", 0.6),
            "val_ratio": config.get("val_ratio", 0.2),
            "test_ratio": config.get("test_ratio", 0.2),
            "mol_feat_type": config["mol_feat_type"],
            "normalize_mol_props": config["normalize_mol_props"],
            "normalize_features": config["normalize_features"],
            "use_conv_attention": config["use_conv_attention"],
            "conv_attention_heads": config["conv_attention_heads"],
            "use_pool_attention": config["use_pool_attention"],
            # 稀疏注意力配置
            "sparse_attention_threshold": config["sparse_attention_threshold"],
            "force_sparse_attention": config["force_sparse_attention"],
            "sparse_matrix_threshold": config["sparse_matrix_threshold"],
            # M矩阵初始化配置（新增）
            "use_M_init_linear_input": config["use_M_init_linear_input"],
            "use_M_init_linear_hidden": config["use_M_init_linear_hidden"],
            "M_matrix_path": config["M_matrix_path"],
            "experiment_name": config["experiment_name"],
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 简化配置文件名（添加稀疏标识和M矩阵标识）
        balance_suffix = "_b" if config["split_balanced"] else "_ub"
        attn_suffix = f"_ca{1 if config['use_conv_attention'] else 0}_pa{1 if config['use_pool_attention'] else 0}"
        sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                         "sparse_attention_threshold"] < 10000 else ""
        if config["force_sparse_attention"]:
            sparse_suffix = "_fsparse"
        # M矩阵初始化标识（新增）
        M_init_suffix = ""
        if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
            input_flag = "1" if config["use_M_init_linear_input"] else "0"
            hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
            M_init_suffix = f"_M{input_flag}{hidden_flag}"
        split_abbr = config["split_type"][:3] if config["split_type"] else "rnd"
        save_filename = f"esol_cfg_{split_abbr}{balance_suffix}{attn_suffix}{sparse_suffix}{M_init_suffix}.pkl"
        save_filepath = os.path.join(save_path, save_filename)

        validate_and_create_dir(save_filepath, "配置文件")

        with open(save_filepath, "wb") as f:
            pickle.dump(mol_config, f)

        logging.info(f"[配置保存] 分子任务配置已保存至：{save_filepath}")


def main():
    # 1. 加载配置
    config = get_config()

    # 提前验证基础日志目录
    try:
        if not os.path.isdir(config["log_path"]):
            os.makedirs(config["log_path"], exist_ok=True)
            print(f"[初始化] 已创建基础日志目录：{config['log_path']}")
        else:
            print(f"[初始化] 基础日志目录已存在：{config['log_path']}")
    except Exception as e:
        print(f"[致命错误] 基础日志目录验证失败：{str(e)}")
        traceback.print_exc()
        return

    # 2. 初始化日志
    try:
        logger = setup_logging(config)
    except Exception as e:
        print(f"[致命错误] 日志系统初始化失败：{str(e)}")
        return

    # 3. 设置随机种子
    set_seed(config)

    # 打印实验启动信息（添加稀疏注意力和M矩阵初始化信息）
    logger.info("\n" + "=" * 80)
    logger.info(f"HWNN 实验启动 | 实验名称：{config['experiment_name']}")
    logger.info(f"核心配置：任务类型={config['task_type']} | 数据集={config['dataset']} | 设备={config['device']}")

    # 稀疏注意力信息
    if config["use_conv_attention"]:
        sparse_mode = "强制稀疏" if config["force_sparse_attention"] else "自动切换"
        logger.info(f"稀疏注意力：模式={sparse_mode} | 阈值={config['sparse_attention_threshold']}节点")

    # M矩阵初始化信息（新增）
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        logger.info(f"M矩阵初始化：输入层={'启用' if config['use_M_init_linear_input'] else '禁用'} | "
                   f"隐藏层={'启用' if config['use_M_init_linear_hidden'] else '禁用'}")

    if config["task_type"] in ["graph_classification", "graph_regression"]:
        logger.info(
            f"注意力配置：卷积层={'启用' if config['use_conv_attention'] else '禁用'}（头数{config['conv_attention_heads']}） | "
            f"池化层={'启用' if config['use_pool_attention'] else '禁用'}"
        )
    if config["task_type"] == "graph_regression" and config["dataset"].lower() in ["esol", "freesolv", "lipo", "qm9"]:
        logger.info(
            f"分子分割：方式={config['split_type']} | 平衡模式={'启用' if config['split_balanced'] else '禁用'} | 分割种子={config['split_seed']}")
    logger.info("=" * 80)

    # 4. 数据预处理
    logger.info(f"\n[Step 1/4] 数据预处理 | 数据集：{config['dataset']} | 任务：{config['task_type']}")
    try:
        if config["task_type"] == "node_classification":
            preprocessor = DataPreprocessor(
                dataset_name=config["dataset"],
                root_data_path=config["data_path"],
                train_ratio=config["train_ratio"],
                val_ratio=config["val_ratio"],
                test_ratio=config["test_ratio"],
                split_seed=config["split_seed"],
                normalize_features=config["normalize_features"],
                device=config["device"],
                n_clusters=config["n_clusters"],
                use_node_degree=config["use_node_degree"]
            )
        else:
            preprocessor = DataPreprocessor(
                dataset_name=config["dataset"],
                root_data_path=config["data_path"],
                normalize_features=config["normalize_features"],
                device=config["device"],
                task_type=config["task_type"],
                mol_feat_type=config["mol_feat_type"],
                normalize_mol_props=config["normalize_mol_props"],
                split_seed=config["split_seed"],
                split_type=config["split_type"],
                split_balanced=config["split_balanced"]
            )

        processed_data = preprocessor.run()

        if processed_data.get("is_mol_task", False):
            train_size = len(processed_data["train"])
            val_size = len(processed_data["val"])
            test_size = len(processed_data["test"])
            total_size = train_size + val_size + test_size
            logger.info(f"[分子分割结果] 总分子数：{total_size} | 训练集：{train_size}（{train_size / total_size:.1%}） | "
                        f"验证集：{val_size}（{val_size / total_size:.1%}） | 测试集：{test_size}（{test_size / total_size:.1%}）")
            logger.info(
                f"[分子分割结果] 分割方式：{processed_data['split_type']} | 平衡模式：{'启用' if processed_data['split_balanced'] else '禁用'}")

    except Exception as e:
        logger.error(f"[数据预处理失败] 原因：{str(e)}", exc_info=True)
        raise

    # 5. 更新模型维度配置
    logger.info(f"\n[Step 2/4] 模型维度配置")
    try:
        if config["task_type"] == "node_classification":
            config["input_dim"] = processed_data["features"].shape[1]
            config["output_dim"] = processed_data["num_classes"]
            logger.info(f"[节点任务] 输入维度：{config['input_dim']} | 输出维度：{config['output_dim']}（类别数）")
        else:
            config["input_dim"] = processed_data["num_features"]
            config["output_dim"] = processed_data["graph_label_dim"] if config[
                                                                            "task_type"] == "graph_classification" else 1

            if processed_data.get("is_mol_task", False):
                if config["mol_feat_type"] == "rdkit+graph" and config["input_dim"] < 4:
                    logger.warning(
                        f"[分子特征警告] 融合特征维度={config['input_dim']}，可能低于预期（RDKit特征4维+嵌入维度）")
                logger.info(
                    f"[分子任务] 原子特征维度：{config['input_dim']} | 输出维度：{config['output_dim']}（回归目标）")
                logger.info(
                    f"[分子任务] 有效原子掩码：{'已包含' if 'valid_mask' in processed_data['train'][0] else '缺失'}")
            else:
                logger.info(f"[图任务] 输入维度：{config['input_dim']} | 输出维度：{config['output_dim']}")

    except KeyError as e:
        logger.error(f"[维度配置失败] 预处理数据缺少键：{str(e)}", exc_info=True)
        raise

    # 6. 超图构建（添加稀疏矩阵阈值配置）
    hyper_types = config.get("node_hyper_types") if config["task_type"] == "node_classification" else config.get(
        "graph_hyper_types")

    # 稀疏矩阵信息
    sparse_info = f" | 稀疏矩阵阈值：{config['sparse_matrix_threshold']}节点" if config["use_conv_attention"] else ""
    logger.info(
        f"\n[Step 3/4] 异构超图构建 | 超边类型：{hyper_types} | 超边大小范围：{config['hyperedge_size_range']}{sparse_info}")

    try:
        if config["task_type"] == "node_classification":
            builder = HypergraphBuilder(
                processed_data=processed_data,
                task_type=config["task_type"],
                hyper_types=config["node_hyper_types"],
                hyper_repeats=config["node_hyper_repeats"],
                phi_hop=config["phi_hop"],
                n_clusters=config["n_clusters"],
                hyperedge_size_range=config["hyperedge_size_range"],
                sparse_threshold=config["sparse_matrix_threshold"],  # 稀疏矩阵阈值
                epsilon=1e-5,
                device=config["device"]
            )
        else:
            builder = HypergraphBuilder(
                processed_data=processed_data,
                task_type=config["task_type"],
                graph_edge_weight=config["graph_edge_weight"],
                substructure_depth=config["substructure_depth"],
                graph_hyper_types=config["graph_hyper_types"],
                graph_hyper_repeats=config["graph_hyper_repeats"],
                hyperedge_size_range=config["hyperedge_size_range"],
                sparse_threshold=config["sparse_matrix_threshold"],  # 稀疏矩阵阈值
                epsilon=1e-5,
                device=config["device"]
            )

        hypergraph_data = builder.run()
        save_mol_config(config, hypergraph_data, config["log_path"])

        # 稀疏矩阵统计信息
        if hypergraph_data.get('sparse_enabled', False):
            logger.info(f"[稀疏矩阵] 超图构建已完成，部分或全部数据启用稀疏矩阵存储")
        else:
            logger.info(f"[稀疏矩阵] 超图构建已完成，使用标准稠密矩阵存储")

    except Exception as e:
        logger.error(f"[超图构建失败] 原因：{str(e)}", exc_info=True)
        raise

    # 7. 模型初始化与训练（添加稀疏注意力阈值和M矩阵初始化配置）
    logger.info(f"\n[Step 4/4] 模型初始化与训练")
    try:
        model = HWNN(
            task_type=config["task_type"],
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            num_layers=config["num_layers"],
            K1=config["K1"],
            K2=config["K2"],
            use_M_init_linear_input=config["use_M_init_linear_input"],
            use_M_init_linear_hidden=config["use_M_init_linear_hidden"],
            M_matrix_path=config['M_matrix_path'],
            approx=config["approx"],
            dropout=config["dropout"],
            device=config["device"],
            use_conv_attention=config["use_conv_attention"],
            conv_attention_heads=config["conv_attention_heads"],
            use_pool_attention=config["use_pool_attention"],
            sparse_attention_threshold=config["sparse_attention_threshold"]
        )

        conv_attn_info = f"启用（头数：{config['conv_attention_heads']}）" if config["use_conv_attention"] else "禁用"
        pool_attn_info = f"启用" if config["use_pool_attention"] else "禁用"
        sparse_info = f" | 稀疏阈值：{config['sparse_attention_threshold']}节点" if config["use_conv_attention"] else ""

        # M矩阵初始化信息（新增）
        M_init_info = ""
        if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
            M_init_info = f" | M矩阵初始化：输入层={'是' if config['use_M_init_linear_input'] else '否'} | 隐藏层={'是' if config['use_M_init_linear_hidden'] else '否'}"

        logger.info(f"[模型配置] 层数：{config['num_layers']} | 隐藏维度：{config['hidden_dim']} | "
                    f"Dropout：{config['dropout']} | 卷积注意力：{conv_attn_info}{sparse_info} | 池化注意力：{pool_attn_info} | "
                    f"多项式阶数：({config['K1']},{config['K2']}){M_init_info}")

    except Exception as e:
        logger.error(f"[模型初始化失败] 原因：{str(e)}", exc_info=True)
        raise

    try:
        trainer = HWNNTrainer(
            model=model,
            processed_data=processed_data,
            hyper_data=hypergraph_data,
            device=config["device"],
            log_path=config["log_path"],
            save_model=config["save_model"],
            save_attn_weights=config["save_attn_weights"],
            attn_save_interval=config["record_attn_interval"]
        )

        lr_scheduler_enable = config["lr_scheduler"] != "none"
        trainer.set_optimizer(
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            grad_clip=config["grad_clip"],
            lr_scheduler=lr_scheduler_enable,
            lr_scheduler_patience=config["lr_scheduler_patience"]
        )
        logger.info(
            f"[训练配置] 学习率：{config['lr']} | 权重衰减：{config['weight_decay']} | 梯度裁剪：{config['grad_clip']} | "
            f"调度器：{'启用' if lr_scheduler_enable else '禁用'}（类型：{config['lr_scheduler']}）")

        if config["task_type"] == "node_classification":
            best_val_metric = trainer.train(
                epochs=config["epochs"],
                patience=config["patience"],
                eval_interval=config["eval_interval"],
                verbose=config["verbose"]
            )
        else:
            # 训练信息添加稀疏注意力状态
            sparse_status = "（启用稀疏注意力）" if (config["use_conv_attention"] and
                                                   hypergraph_data.get('sparse_enabled', False)) else ""
            logger.info(
                f"[训练启动] 批次大小：{config['batch_size']} | 总轮次：{config['epochs']} | 早停耐心值：{config['patience']}{sparse_status} | "
                f"验证间隔：{config['eval_interval']} | 注意力记录间隔：{config['record_attn_interval']}")

            best_val_metric = trainer.train(
                epochs=config["epochs"],
                patience=config["patience"],
                batch_size=config["batch_size"],
                eval_interval=config["eval_interval"],
                verbose=config["verbose"],
                record_attn_every=config["record_attn_interval"]
            )

        logger.info(
            f"\n[测试集评估] 分割方式：{config.get('split_type', 'random')} | 批次大小：{config.get('batch_size', '全量')} | "
            f"注意力配置：卷积={config['use_conv_attention']} | 池化={config['use_pool_attention']}")

        # 稀疏注意力状态信息
        if config["use_conv_attention"] and hypergraph_data.get('sparse_enabled', False):
            logger.info(f"[稀疏注意力] 测试阶段使用稀疏注意力计算")

        test_loss, test_results, test_metric = trainer.evaluate(
            split='test',
            batch_size=config["batch_size"] if config["task_type"] != "node_classification" else None
        )

        # 根据任务类型打印测试集结果，回归任务增加RMSE
        if 'regression' in config["task_type"]:
            test_rmse = test_results.get('rmse', None)
            if test_rmse is not None:
                logger.info(
                    f"[测试集结果] 损失：{test_loss:.4f} | 主要指标（MSE）：{test_metric:.4f} | RMSE：{test_rmse:.4f}")
            else:
                logger.warning(
                    f"[测试集结果] 损失：{test_loss:.4f} | 主要指标（MSE）：{test_metric:.4f} | 未获取到 RMSE（请检查任务类型配置）")
        else:
            # 分类任务保持原逻辑
            logger.info(f"[测试集结果] 损失：{test_loss:.4f} | 主要指标（准确率/Macro-F1）：{test_metric:.4f}")

        if config["plot_curves"]:
            # 简化图像文件名（添加稀疏标识和M矩阵标识）
            if processed_data.get("is_mol_task", False):
                balance_suffix = "_b" if config["split_balanced"] else "_ub"
                attn_suffix = f"_ca{1 if config['use_conv_attention'] else 0}_pa{1 if config['use_pool_attention'] else 0}"
                sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                                 "sparse_attention_threshold"] < 10000 else ""
                if config["force_sparse_attention"]:
                    sparse_suffix = "_fsparse"
                # M矩阵初始化标识（新增）
                M_init_suffix = ""
                if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
                    input_flag = "1" if config["use_M_init_linear_input"] else "0"
                    hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
                    M_init_suffix = f"_M{input_flag}{hidden_flag}"
                split_abbr = config["split_type"][:3] if config["split_type"] else "rnd"
                curve_suffix = f"esol_{split_abbr}{balance_suffix}{attn_suffix}{sparse_suffix}{M_init_suffix}"
            else:
                # 使用统一函数获取任务缩写
                task_abbr = get_task_abbr(config["task_type"])
                sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                                 "sparse_attention_threshold"] < 10000 else ""
                if config["force_sparse_attention"]:
                    sparse_suffix = "_fsparse"
                # M矩阵初始化标识（新增）
                M_init_suffix = ""
                if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
                    input_flag = "1" if config["use_M_init_linear_input"] else "0"
                    hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
                    M_init_suffix = f"_M{input_flag}{hidden_flag}"
                curve_suffix = f"{task_abbr}{sparse_suffix}{M_init_suffix}"

            curve_path = os.path.join(config["log_path"], f"lc_{curve_suffix}.png")
            os.makedirs(os.path.dirname(curve_path), exist_ok=True)
            trainer.plot_learning_curves(save_path=curve_path)
            logger.info(f"[学习曲线] 已保存至：{curve_path}")

        if processed_data.get("is_mol_task", False) and config["save_attn_weights"] and hasattr(trainer,
                                                                                                "attn_weights_history"):
            mol_smiles = next(iter(trainer.attn_weights_history.keys()), None)
            if mol_smiles:
                # 简化注意力图像文件名（添加稀疏标识和M矩阵标识）
                smiles_short = mol_smiles[:6]
                sparse_suffix = f"_sparse{config['sparse_attention_threshold'] // 1000}k" if config[
                                                                                                 "sparse_attention_threshold"] < 10000 else ""
                if config["force_sparse_attention"]:
                    sparse_suffix = "_fsparse"
                # M矩阵初始化标识（新增）
                M_init_suffix = ""
                if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
                    input_flag = "1" if config["use_M_init_linear_input"] else "0"
                    hidden_flag = "1" if config["use_M_init_linear_hidden"] else "0"
                    M_init_suffix = f"_M{input_flag}{hidden_flag}"
                attn_plot_path = os.path.join(config["log_path"], f"attn_{smiles_short}{sparse_suffix}{M_init_suffix}.png")
                os.makedirs(os.path.dirname(attn_plot_path), exist_ok=True)
                trainer.plot_attn_distribution(smiles=mol_smiles, save_path=attn_plot_path)
                logger.info(f"[注意力可视化] 分子{mol_smiles[:20]}...的权重分布已保存至：{attn_plot_path}")

    except Exception as e:
        logger.error(f"[训练/评估失败] 原因：{str(e)}", exc_info=True)
        raise

    # 8. 输出最终结果（添加稀疏注意力和M矩阵初始化性能分析）
    logger.info("\n" + "=" * 80)

    # 稀疏注意力性能分析
    sparse_performance_note = ""
    if config["use_conv_attention"] and hypergraph_data.get('sparse_enabled', False):
        sparse_performance_note = " | 稀疏注意力：已启用（显存优化）"
    elif config["use_conv_attention"] and not hypergraph_data.get('sparse_enabled', False):
        sparse_performance_note = " | 稀疏注意力：未启用（节点数较少）"

    # M矩阵初始化状态（新增）
    M_init_status = ""
    if config["use_M_init_linear_input"] or config["use_M_init_linear_hidden"]:
        M_init_status = f" | M矩阵初始化：输入层={'启用' if config['use_M_init_linear_input'] else '禁用'} | 隐藏层={'启用' if config['use_M_init_linear_hidden'] else '禁用'}"

    logger.info(
        f"实验结束 | 数据集：{config['dataset']} | 分割方式：{config.get('split_type', 'random')}{sparse_performance_note}{M_init_status} | "
        f"注意力配置：卷积={config['use_conv_attention']} | 池化={config['use_pool_attention']}")
    logger.info("=" * 80)

    if config["task_type"] in ["node_classification", "graph_classification"]:
        logger.info(f"[最佳验证结果] Macro-F1：{best_val_metric:.4f}")
        logger.info(f"[测试集结果] 准确率：{test_results['accuracy']:.4f} | Macro-F1：{test_results['macro-f1']:.4f}")
        if test_results['accuracy'] < 0.5:
            logger.warning(f"[结果警告] 测试准确率低于50%，建议检查：\n"
                           f"  1. 卷积注意力是否启用（--use-conv-attention True）\n"
                           f"  2. 超边类型配置是否合理（如增加community/similarity超边）\n"
                           f"  3. 隐藏维度与注意力头数匹配性（hidden_dim % conv_attention_heads == 0）")
    else:
        logger.info(f"[最佳验证结果] MSE：{best_val_metric:.4f}")
        logger.info(f"[测试集详细结果]")
        logger.info(f"  - MSE（均方误差）：{test_results['mse']:.4f}（越小越好）")
        logger.info(f"  - RMSE（均方根误差）：{test_results['rmse']:.4f}（越小越好，与目标值同量级，更直观）")
        logger.info(f"  - MAE（平均绝对误差）：{test_results['mae']:.4f}（越小越好）")
        logger.info(f"  - R²（决定系数）：{test_results['r2']:.4f}（接近1表示模型拟合优度高）")

        if config["dataset"].lower() == "esol":
            # 稀疏注意力和M矩阵初始化对ESOL任务的影响分析（新增）
            if config["use_conv_attention"] and hypergraph_data.get('sparse_enabled', False):
                logger.info(f"[稀疏注意力效果] ESOL任务中稀疏注意力已启用，有效防止大分子图的显存溢出")

            # M矩阵初始化效果分析（新增）
            if config["use_M_init_linear_input"]:
                logger.info(f"[M矩阵初始化效果] 输入层使用预训练M矩阵初始化，有助于模型快速收敛")

            # 增加基于RMSE的评价
            if test_results['rmse'] < 0.5:
                logger.info(f"[结果评价] RMSE < 0.5，模型对ESOL溶解度的预测精度极高")
            elif test_results['rmse'] < 1.0:
                logger.info(f"[结果评价] RMSE < 1.0，模型对ESOL溶解度的预测精度良好")
            else:
                logger.warning(f"[结果警告] RMSE > 1.0，模型预测精度需优化（建议调整注意力头数或超边类型）")

            if test_results['r2'] > 0.8:
                logger.info(f"[结果评价] R² > 0.8，模型对ESOL溶解度的预测能力优秀\n"
                            f"  注意力配置生效：卷积注意力捕捉原子拓扑关联，池化注意力突出关键官能团（如-OH/-COOH）")
            elif test_results['r2'] > 0.6:
                logger.info(f"[结果评价] R² > 0.6，模型对ESOL溶解度的预测能力良好\n"
                            f"  可尝试优化：增加卷积注意力头数（--conv-attention-heads 8）或调整池化层Dropout（--dropout 0.2）")
            else:
                logger.warning(f"[结果警告] R² < 0.6，模型预测能力一般，建议调整：\n"
                               f"  1. 确保启用注意力池化（--use-pool-attention True，分子任务必选）\n"
                               f"  2. 尝试融合特征（--mol-feat-type rdkit+graph，兼顾化学属性与拓扑）\n"
                               f"  3. 增加超边类型（如--graph-hyper-types bond,functional_group,hydrogen_bond）\n"
                               f"  4. 调整隐藏维度（--hidden-dim 512）与训练轮次（--epochs 10000）")

    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logging.getLogger().hasHandlers():
            logging.error(f"[实验整体失败] 原因：{str(e)}", exc_info=True)
        else:
            print(f"[实验整体失败] 原因：{str(e)}")
        exit(1)
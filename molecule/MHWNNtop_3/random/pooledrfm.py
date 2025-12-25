import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import logging
import argparse
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# 假设你的 RFM 类在 mrfmmore.py 文件中，并且已经包含了 save_weighted_diag_matrix 方法
from mrfmmore import RFM
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

# 配置中文字体，解决中文显示为方框的问题
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
import seaborn as sns
from itertools import product
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/esol/training.log", encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------
# 骨架分割依赖的RDKit导入（处理ImportError）
# ----------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors  # 新增：分子级属性提取需要
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("请安装RDKit以支持骨架分割和分子属性提取功能：pip install rdkit-pypi")


# ========== 新增：分子级属性提取函数 ==========
def extract_molecular_properties(smiles_list: list) -> tuple[np.ndarray, list]:
    """
    从SMILES列表提取分子级属性（分子量、脂水分配系数等）
    返回：属性矩阵（N, M）和属性名称列表（M个属性）
    """
    # 定义要提取的分子属性（可扩展）
    prop_names = [
        "MolWt",  # 分子量
        "MolLogP",  # 脂水分配系数
        "NumHDonors",  # 氢键供体数
        "NumHAcceptors",  # 氢键受体数
        "TPSA"  # 拓扑极性表面积
    ]

    # 提取属性值
    mol_props = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 无效分子用均值填充（后续会被过滤）
                props = [np.nan] * len(prop_names)
            else:
                # 提取每个属性值
                props = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol)
                ]
        except Exception as e:
            logger.warning(f"分子属性提取失败：{smiles[:20]} | 错误：{str(e)[:30]}")
            props = [np.nan] * len(prop_names)
        mol_props.append(props)

    # 转换为数组并处理NaN值（用列均值填充）
    mol_props = np.array(mol_props, dtype=np.float32)
    for col in range(mol_props.shape[1]):
        col_mean = np.nanmean(mol_props[:, col])
        mol_props[np.isnan(mol_props[:, col]), col] = col_mean

    logger.info(f"[分子属性提取] 提取{len(prop_names)}种分子级属性，形状：{mol_props.shape}")
    return mol_props, prop_names


# ========== 新增：分子级属性标准化函数 ==========
def normalize_molecular_properties(
        mol_props_train: np.ndarray,
        mol_props_val: np.ndarray,
        mol_props_test: np.ndarray,
        prop_names: list  # 新增：接收属性名称列表
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    基于训练集对分子级属性进行标准化
    """
    scaler = StandardScaler()
    mol_props_train_norm = scaler.fit_transform(mol_props_train)
    mol_props_val_norm = scaler.transform(mol_props_val)
    mol_props_test_norm = scaler.transform(mol_props_test)

    # 记录标准化统计信息（此时prop_names已通过参数传入，可正常引用）
    for i, prop_name in enumerate(prop_names):
        logger.info(
            f"[属性标准化] {prop_name} - 均值: {scaler.mean_[i]:.4f} | 标准差: {scaler.scale_[i]:.4f}"
        )

    return mol_props_train_norm, mol_props_val_norm, mol_props_test_norm, scaler


# ========== RDKit原子特征提取函数 ==========
def _extract_rdkit_atom_feats(smiles: str, max_num_atoms: int) -> np.ndarray:
    """从SMILES提取RDKit原子特征（原子序数、形式电荷、价电子数、芳香性）"""
    rdkit_feat_dim = 4  # 固定4维RDKit特征
    atom_feats = np.zeros((max_num_atoms, rdkit_feat_dim), dtype=np.float32)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"SMILES无效，RDKit特征用0填充：{smiles[:20]}")
            return atom_feats

        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_num_atoms:
            logger.warning(f"分子原子数({num_atoms})超过最大限制({max_num_atoms})，截断后填充：{smiles[:20]}")
            num_atoms = max_num_atoms  # 截断超出部分

        # 提取每个原子的特征
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_feats[atom_idx] = [
                atom.GetAtomicNum(),  # 原子序数
                atom.GetFormalCharge(),  # 形式电荷
                atom.GetTotalValence(),  # 价电子数
                1.0 if atom.GetIsAromatic() else 0.0  # 芳香性（0/1）
            ]
    except Exception as e:
        logger.warning(f"RDKit特征提取失败，用0填充：{smiles[:20]} | 错误：{str(e)[:20]}")

    return atom_feats


# ========== 特征融合函数（增强：支持分子级属性融合） ==========
def fuse_mol_features(
        smiles_list: list,
        graph_feats: np.ndarray,  # 预计算的原子嵌入（graph特征）
        max_num_atoms: int,
        mol_feat_type: str = "rdkit+graph",
        mol_props: np.ndarray = None,  # 新增：分子级属性
        use_mol_props: bool = False  # 新增：是否使用分子级属性
) -> np.ndarray:
    """融合RDKit特征、graph特征和分子级属性"""
    N = len(smiles_list)
    assert N == graph_feats.shape[0], f"SMILES数({N})与graph特征数({graph_feats.shape[0]})不匹配"
    assert graph_feats.shape[
               1] == max_num_atoms, f"graph特征原子数({graph_feats.shape[1]})与最大原子数({max_num_atoms})不匹配"

    # 1. 提取RDKit特征（批量处理）
    rdkit_feats = []
    for smiles in smiles_list:
        feats = _extract_rdkit_atom_feats(smiles, max_num_atoms)
        rdkit_feats.append(feats)
    rdkit_feats = np.array(rdkit_feats, dtype=np.float32)  # 形状 (N, max_num_atoms, 4)
    logger.info(f"RDKit特征形状：{rdkit_feats.shape}（4维固定特征）")

    # 2. 基础特征融合（RDKit + graph）
    if mol_feat_type == "rdkit":
        base_feats = rdkit_feats
        logger.info(f"基础特征类型：仅RDKit特征（维度4）")
    elif mol_feat_type == "graph":
        base_feats = graph_feats
        logger.info(f"基础特征类型：仅预计算嵌入（维度{graph_feats.shape[2]}）")
    else:  # rdkit+graph
        base_feats = np.concatenate([rdkit_feats, graph_feats], axis=2)
        logger.info(f"基础特征类型：RDKit+graph融合（维度{4 + graph_feats.shape[2]} = 4 + {graph_feats.shape[2]}）")

    # 3. 融合分子级属性（若启用）
    if use_mol_props and mol_props is not None:
        # 分子级属性形状：(N, M)，需要扩展为(N, 1, M)才能与原子级特征拼接
        mol_props_expanded = np.expand_dims(mol_props, axis=1)  # (N, 1, M)
        # 复制到所有原子位置：(N, max_num_atoms, M)
        mol_props_broadcast = np.broadcast_to(mol_props_expanded, (N, max_num_atoms, mol_props.shape[1]))
        # 与基础特征融合：在特征维度拼接
        fused_feats = np.concatenate([base_feats, mol_props_broadcast], axis=2)
        logger.info(f"特征融合：添加{mol_props.shape[1]}种分子级属性，最终维度{base_feats.shape[2] + mol_props.shape[1]}")
        return fused_feats
    else:
        return base_feats


# 导入数据加载函数
def load_esol_data(save_path: str, format: str = "pkl"):
    """复用嵌入数据解析逻辑"""
    if format == "pkl":
        import pickle
        if not os.path.exists(f"{save_path}.pkl"):
            raise FileNotFoundError(f"嵌入文件 {save_path}.pkl 不存在，请检查路径")
        with open(f"{save_path}.pkl", "rb") as f:
            return pickle.load(f)
    elif format == "npz":
        if not os.path.exists(f"{save_path}.npz"):
            raise FileNotFoundError(f"嵌入文件 {save_path}.npz 不存在，请检查路径")
        data = np.load(f"{save_path}.npz", allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "smiles": data["smiles"].tolist(),
            "labels": data["labels"],
            "original_indices": data["original_indices"],
            "all_atoms": data["all_atoms"],
            "element_map": dict(zip(data["elem_symbols"], data["elem_codes"])),
            "max_num_atoms": data["max_num_atoms"],
            "embed_dim": data["embed_dim"]
        }
    else:
        raise ValueError("支持格式: 'pkl' 或 'npz'")


# 骨架提取函数
def _get_murcko_scaffold(smiles):
    """从SMILES提取Murcko骨架"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"invalid_smiles_{smiles[:20]}"
    except Exception as e:
        return f"parse_error_{smiles[:20]}_{str(e)[:10]}"

    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
    except Exception as e:
        return f"scaffold_error_{smiles[:20]}_{str(e)[:10]}"

    if not scaffold_smiles:
        return f"linear_mol_{smiles[:20]}"

    return scaffold_smiles


# 骨架分割核心函数
def scaffold_split(
        smiles_list,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        split_balanced=True,
        random_seed=42
):
    """基于分子骨架的数据集分割（无效样本全部放入训练集）"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "三集比例总和必须为1"
    total_samples = len(smiles_list)
    if total_samples == 0:
        raise ValueError("SMILES列表为空，无法分割")

    # 分离有效样本和无效样本
    valid_indices = []
    invalid_indices = []
    scaffold_groups = {}

    for idx, smiles in enumerate(smiles_list):
        scaffold = _get_murcko_scaffold(smiles)
        if any(prefix in scaffold for prefix in ["invalid_", "parse_error_", "scaffold_error_"]):
            invalid_indices.append(idx)
        else:
            valid_indices.append(idx)
            if scaffold not in scaffold_groups:
                scaffold_groups[scaffold] = []
            scaffold_groups[scaffold].append(idx)

    logger.info(
        f"[骨架分组] 有效样本: {len(valid_indices)} | 无效样本: {len(invalid_indices)} | 骨架组数: {len(scaffold_groups)}")

    # 计算有效样本的分割比例
    valid_total = len(valid_indices)
    train_valid = int(valid_total * train_ratio)
    val_valid = int(valid_total * val_ratio)
    test_valid = valid_total - train_valid - val_valid

    # 处理有效样本分组
    groups_list = list(scaffold_groups.values())
    group_sizes = [len(g) for g in groups_list] if groups_list else []
    if group_sizes:
        logger.info(
            f"[分组大小] 最大: {max(group_sizes)} | 最小: {min(group_sizes)} | 平均: {np.mean(group_sizes):.1f}")

    # 平衡分组处理
    random.seed(random_seed)
    if split_balanced and len(groups_list) > 0:
        val_test_total = val_valid + test_valid
        big_group_thresh = val_test_total / 2
        big_groups = [g for g in groups_list if len(g) > big_group_thresh]
        small_groups = [g for g in groups_list if len(g) <= big_group_thresh]
        random.shuffle(big_groups)
        random.shuffle(small_groups)
        groups_list = big_groups + small_groups
        logger.info(f"[平衡处理] 大组: {len(big_groups)} | 小组: {len(small_groups)} | 阈值: {big_group_thresh:.1f}")
    else:
        groups_list = sorted(groups_list, key=lambda x: len(x), reverse=True)
        logger.info("[排序处理] 按骨架组大小降序排列（非平衡模式）")

    # 分割有效样本为训练/验证/测试
    train_idx_valid = []
    rest_idx_valid = []
    for group in groups_list:
        if len(train_idx_valid) + len(group) <= train_valid:
            train_idx_valid.extend(group)
        else:
            rest_idx_valid.extend(group)

    # 分割剩余有效样本为验证和测试
    val_idx_valid = []
    test_idx_valid = []
    rest_groups = {}
    for idx in rest_idx_valid:
        scaffold = _get_murcko_scaffold(smiles_list[idx])
        if scaffold not in rest_groups:
            rest_groups[scaffold] = []
        rest_groups[scaffold].append(idx)
    rest_groups_list = sorted(list(rest_groups.values()), key=lambda x: len(x), reverse=True)

    for group in rest_groups_list:
        if len(val_idx_valid) + len(group) <= val_valid:
            val_idx_valid.extend(group)
        else:
            test_idx_valid.extend(group)

    # 合并无效样本到训练集
    train_idx = train_idx_valid + invalid_indices
    val_idx = val_idx_valid
    test_idx = test_idx_valid

    # 验证骨架无重叠
    def get_valid_scaffolds(idx_list):
        scaffolds = set()
        for idx in idx_list:
            if idx in valid_indices:  # 只检查有效样本
                scaffold = _get_murcko_scaffold(smiles_list[idx])
                scaffolds.add(scaffold)
        return scaffolds

    train_scaffolds = get_valid_scaffolds(train_idx)
    val_scaffolds = get_valid_scaffolds(val_idx)
    test_scaffolds = get_valid_scaffolds(test_idx)

    assert len(train_scaffolds & val_scaffolds) == 0, "训练集与验证集存在骨架重叠！"
    assert len(train_scaffolds & test_scaffolds) == 0, "训练集与测试集存在骨架重叠！"
    assert len(val_scaffolds & test_scaffolds) == 0, "验证集与测试集存在骨架重叠！"

    logger.info("\n[骨架分割结果]")
    logger.info(f"训练集：{len(train_idx)}样本（含{len(invalid_indices)}无效样本）| {len(train_scaffolds)}有效骨架")
    logger.info(f"验证集：{len(val_idx)}样本 | {len(val_scaffolds)}有效骨架")
    logger.info(f"测试集：{len(test_idx)}样本 | {len(test_scaffolds)}有效骨架")

    return train_idx, val_idx, test_idx


# 随机分割函数
def random_split(
        total_samples: int,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
):
    """基于随机的数据集分割（按比例拆分，无骨架约束）"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "三集比例总和必须为1"
    if total_samples == 0:
        raise ValueError("样本数为空，无法分割")

    # 第一步：分割训练集 和 （验证集+测试集）
    all_indices = np.arange(total_samples)
    train_idx, rest_idx = train_test_split(
        all_indices,
        train_size=train_ratio,
        shuffle=True,
        random_state=random_seed  # 固定seed保证可复现
    )

    # 第二步：分割验证集 和 测试集（基于剩余样本）
    val_test_ratio = val_ratio / (val_ratio + test_ratio)  # 剩余样本中验证集占比
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_test_ratio,
        shuffle=True,
        random_state=random_seed  # 同一seed保证拆分稳定
    )

    logger.info("\n[随机分割结果]")
    logger.info(f"训练集：{len(train_idx)}样本 | 验证集：{len(val_idx)}样本 | 测试集：{len(test_idx)}样本")
    return train_idx, val_idx, test_idx


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"已设置随机种子: {seed}")


# 加载嵌入数据（增强：集成分子级属性提取）
def load_embedding_data(
        embedding_dir: str,
        embedding_filename: str,
        mol_feat_type: str = "rdkit+graph",
        format: str = "pkl",
        use_mol_emb: bool = False,
        use_mol_props: bool = False  # 新增：是否使用分子级属性
):
    """加载嵌入数据+特征融合+分子级属性提取"""
    if not os.path.exists(embedding_dir):
        raise NotADirectoryError(f"嵌入目录 {embedding_dir} 不存在，请确认路径")

    embedding_path = os.path.join(embedding_dir, embedding_filename)
    logger.info(f"加载嵌入数据: {embedding_path}.{format}")
    embedding_data = load_esol_data(embedding_path, format=format)

    # 提取核心数据
    graph_feats = embedding_data["embeddings"]  # 预计算的graph特征（原子级）
    y = embedding_data["labels"]
    all_atoms = embedding_data["all_atoms"]
    smiles_list = embedding_data["smiles"]
    max_num_atoms = embedding_data["max_num_atoms"]
    original_embed_dim = embedding_data["embed_dim"]  # 原始graph特征维度

    # 新增：提取分子级属性
    mol_props = None
    prop_names = []
    if use_mol_props:
        mol_props, prop_names = extract_molecular_properties(smiles_list)

    # 特征融合（RDKit + graph + 分子级属性）
    X = fuse_mol_features(
        smiles_list=smiles_list,
        graph_feats=graph_feats,
        max_num_atoms=max_num_atoms,
        mol_feat_type=mol_feat_type,
        mol_props=mol_props,  # 传入分子级属性
        use_mol_props=use_mol_props  # 控制是否融合
    )
    fused_embed_dim = X.shape[2]  # 融合后的特征维度

    # 生成原子掩码
    atom_mask = (all_atoms != 0).astype(np.float32)

    # 可选：聚合为分子级嵌入（对融合后的特征生效）
    if use_mol_emb:
        atom_mask_3d = atom_mask[..., np.newaxis]  # 形状 (N, max_num_atoms, 1)
        masked_emb = X * atom_mask_3d  # 过滤无效原子
        mol_emb = masked_emb.sum(axis=1) / (atom_mask.sum(axis=1, keepdims=True) + 1e-8)  # 按分子平均
        X = mol_emb
        logger.info(f"[嵌入聚合] 已转为分子级嵌入，形状: {X.shape}（维度{mol_emb.shape[1]}）")
    else:
        logger.info(
            f"[嵌入格式] 保留原子级嵌入，形状: {X.shape}（N={X.shape[0]}, 原子数={X.shape[1]}, 维度={X.shape[2]}）")

    # 整理元信息
    embed_info = {
        "max_num_atoms": max_num_atoms,
        "original_embed_dim": original_embed_dim,  # 原始graph维度
        "fused_embed_dim": fused_embed_dim,  # 融合后维度
        "num_samples": len(X),
        "use_mol_emb": use_mol_emb,
        "use_mol_props": use_mol_props,  # 新增：记录是否使用分子属性
        "num_mol_props": len(prop_names) if use_mol_props else 0,  # 分子属性数量
        "mol_prop_names": prop_names,  # 分子属性名称列表
        "atom_mask": atom_mask,
        "smiles": smiles_list,
        "mol_feat_type": mol_feat_type,
        "mol_props": mol_props  # 保存原始分子属性（未标准化）
    }

    # 数据有效性校验
    assert len(X) == len(y), f"特征数({len(X)})与标签数({len(y)})不匹配"
    assert len(X) == len(smiles_list), f"特征数({len(X)})与SMILES数({len(smiles_list)})不匹配"
    logger.info(
        f"[数据加载完成] 有效样本数={embed_info['num_samples']} | "
        f"原始graph维度={original_embed_dim} | 融合后维度={fused_embed_dim} | "
        f"特征类型={mol_feat_type} | 分子属性数={len(prop_names)}"
    )

    return X, y, embed_info


# 特征标准化函数（增强：支持分子级属性标准化）
def normalize_features(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        atom_mask_train: np.ndarray = None,
        atom_mask_val: np.ndarray = None,
        atom_mask_test: np.ndarray = None,
        use_mol_emb: bool = False,
        mol_props_train: np.ndarray = None,
        mol_props_val: np.ndarray = None,
        mol_props_test: np.ndarray = None,
        normalize_mol_props: bool = False,
        prop_names: list = None  # 新增：接收属性名称列表，传递给标准化函数
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """标准化特征和分子级属性（若启用）"""
    # 1. 初始化标准化器
    feat_scaler = StandardScaler()
    mol_prop_scaler = StandardScaler() if (normalize_mol_props and mol_props_train is not None) else None

    # 2. 特征标准化（原子级/分子级嵌入）
    if use_mol_emb:
        # 分子级特征：直接拟合（形状 (N, D)）
        X_train_norm = feat_scaler.fit_transform(X_train)
        X_val_norm = feat_scaler.transform(X_val)
        X_test_norm = feat_scaler.transform(X_test)
        logger.info(f"[特征标准化] 分子级特征标准化完成（基于{X_train.shape[0]}个训练样本）")
    else:
        # 原子级特征：需reshape为 (N*A, D) 拟合
        N_train, A, D = X_train.shape
        N_val = X_val.shape[0]
        N_test = X_test.shape[0]

        # 用掩码过滤无效原子（仅对训练集拟合时过滤）
        train_mask_flat = atom_mask_train.flatten() == 1.0  # 有效原子的掩码
        X_train_flat = X_train.reshape(-1, D)  # (N_train*A, D)
        X_train_valid = X_train_flat[train_mask_flat]  # 仅用有效原子拟合scaler

        # 拟合scaler
        feat_scaler.fit(X_train_valid)

        # 对所有集进行转换
        X_train_norm = feat_scaler.transform(X_train_flat).reshape(N_train, A, D)
        X_val_norm = feat_scaler.transform(X_val.reshape(-1, D)).reshape(N_val, A, D)
        X_test_norm = feat_scaler.transform(X_test.reshape(-1, D)).reshape(N_test, A, D)

        logger.info(
            f"[特征标准化] 原子级特征标准化完成（基于{X_train_valid.shape[0]}个有效原子样本，维度{D}）")

    # 3. 分子级属性标准化（若启用）
    if normalize_mol_props and mol_props_train is not None:
        mol_props_train_norm, mol_props_val_norm, mol_props_test_norm, mol_prop_scaler = normalize_molecular_properties(
            mol_props_train, mol_props_val, mol_props_test, prop_names  # 补充传入prop_names（原代码遗漏，此处修复）
        )

        # 将标准化后的分子属性重新融合到特征中（仅原子级需要）
        if not use_mol_emb:
            # 分子属性在特征中的起始索引
            prop_start_idx = X_train.shape[2] - mol_props_train.shape[1]

            # 替换特征中的分子属性部分为标准化后的值
            X_train_norm[..., prop_start_idx:] = np.broadcast_to(
                np.expand_dims(mol_props_train_norm, axis=1),
                (X_train.shape[0], X_train.shape[1], mol_props_train.shape[1])
            )
            X_val_norm[..., prop_start_idx:] = np.broadcast_to(
                np.expand_dims(mol_props_val_norm, axis=1),
                (X_val.shape[0], X_val.shape[1], mol_props_val.shape[1])
            )
            X_test_norm[..., prop_start_idx:] = np.broadcast_to(
                np.expand_dims(mol_props_test_norm, axis=1),
                (X_test.shape[0], X_test.shape[1], mol_props_test.shape[1])
            )
            logger.info(f"[特征更新] 已将标准化后的分子属性融合到原子级特征中")

    # 记录标准化统计量
    logger.info(f"[特征标准化统计] 均值范围：{np.min(feat_scaler.mean_):.4f} ~ {np.max(feat_scaler.mean_):.4f}")
    logger.info(f"[特征标准化统计] 标准差范围：{np.min(feat_scaler.scale_):.4f} ~ {np.max(feat_scaler.scale_):.4f}")
    return X_train_norm, X_val_norm, X_test_norm, feat_scaler, mol_prop_scaler


# 超参数优化模块（适配分子属性+RFM双返回值）
def objective(X, y, atom_mask, mol_props, inner_iters, pooling_iters, bandwidth, reg, feature_vector_top_k,
              feature_vector_threshold, n_splits=5, use_mol_emb=False, use_mol_props=False):
    """单个参数组合的交叉验证评估（适配分子属性+RFM predict双返回值）"""
    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics = {'mse': [], 'mae': [], 'rmse': [], 'r2': []}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            atom_mask_train = atom_mask[train_idx] if not use_mol_emb else None

            # 提取分子属性的训练/验证子集（若启用）
            mol_props_train = mol_props[train_idx] if (use_mol_props and mol_props is not None) else None
            mol_props_val = mol_props[val_idx] if (use_mol_props and mol_props is not None) else None

            model = RFM(device=device)
            model.fit(
                X_train, y_train,
                atom_mask=atom_mask_train,
                inner_iters=inner_iters,
                pooling_iters=pooling_iters,
                bandwidth=bandwidth,
                reg=reg,
                verbose=False,
                feature_vector_top_k=feature_vector_top_k,
                feature_vector_threshold=feature_vector_threshold
            )

            # 适配RFM predict双返回值：交叉验证暂不关注原子重要性，用_忽略
            atom_mask_val = atom_mask[val_idx] if not use_mol_emb else None
            y_pred, _ = model.predict(X_val, atom_mask=atom_mask_val)

            # 计算 metrics
            metrics['mse'].append(mean_squared_error(y_val, y_pred))
            metrics['mae'].append(mean_absolute_error(y_val, y_pred))
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics['r2'].append(r2_score(y_val, y_pred))

        return {k: np.mean(v) for k, v in metrics.items()}
    except Exception as e:
        logger.warning(
            f"参数组合失败: {inner_iters}, {pooling_iters}, {bandwidth}, {reg}, {feature_vector_top_k}, {feature_vector_threshold} | 错误: {str(e)}")
        return None


def grid_search(X, y, atom_mask, mol_props, param_grid, use_mol_emb=False, use_mol_props=False, n_jobs=-1):
    """并行化超参数网格搜索（适配分子属性）"""
    param_combinations = list(product(
        param_grid['inner_iters'],
        param_grid['pooling_iters'],
        param_grid['bandwidth'],
        param_grid['reg'],
        param_grid['feature_vector_top_k'],
        param_grid['feature_vector_threshold']
    ))

    total_combinations = len(param_combinations)
    logger.info(f"超参数搜索开始，共{total_combinations}个组合，使用{n_jobs}个进程")

    # 并行执行（传入分子属性相关参数）
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(objective)(
            X, y, atom_mask, mol_props, inner_iters, pooling_iters, bandwidth, reg,
            feature_vector_top_k, feature_vector_threshold,
            use_mol_emb=use_mol_emb,
            use_mol_props=use_mol_props  # 新增：控制是否使用分子属性
        ) for inner_iters, pooling_iters, bandwidth, reg, feature_vector_top_k, feature_vector_threshold in
        param_combinations
    )

    # 整理有效结果
    results = []
    for i, res in enumerate(results_list):
        if res is not None:
            params = param_combinations[i]
            results.append({
                'inner_iters': params[0],
                'pooling_iters': params[1],
                'bandwidth': params[2],
                'reg': params[3],
                'feature_vector_top_k': params[4],
                'feature_vector_threshold': params[5],
                'mse': res['mse'],
                'mae': res['mae'],
                'rmse': res['rmse'],
                'r2': res['r2']
            })

    if not results:
        raise ValueError("所有参数组合都失败了，请检查模型或参数范围")

    best_result = min(results, key=lambda x: x['mse'])
    return best_result, results


# 可视化模块
def plot_hyperparameter_results(results, log_dir):
    """超参数搜索结果可视化"""
    df = pd.DataFrame(results)
    df['feature_vector_threshold'] = df['feature_vector_threshold'].apply(
        lambda x: 'None' if x is None else x
    )

    plt.figure(figsize=(18, 22))

    # 1. 内层迭代次数 vs MSE
    plt.subplot(5, 2, 1)
    sns.boxplot(x='inner_iters', y='mse', data=df)
    plt.title('内层迭代次数 vs MSE')

    # 2. 外层池化迭代次数 vs MSE
    plt.subplot(5, 2, 2)
    sns.boxplot(x='pooling_iters', y='mse', data=df)
    plt.title('外层池化迭代次数 vs MSE')

    # 3. 带宽 vs MSE
    plt.subplot(5, 2, 3)
    sns.scatterplot(x='bandwidth', y='mse', data=df)
    plt.title('带宽 vs MSE')

    # 4. 正则化系数 vs MSE
    plt.subplot(5, 2, 4)
    sns.scatterplot(x='reg', y='mse', data=df)
    plt.title('正则化系数 vs MSE')

    # 5. feature_vector_top_k vs MSE
    plt.subplot(5, 2, 5)
    sns.boxplot(x='feature_vector_top_k', y='mse', data=df)
    plt.title('feature_vector_top_k vs MSE')

    # 6. feature_vector_threshold vs MSE
    plt.subplot(5, 2, 6)
    threshold_cats = ['None'] + [str(t) for t in
                                 sorted([v for v in df['feature_vector_threshold'].unique() if v != 'None'])]
    df['feature_vector_threshold'] = pd.Categorical(
        df['feature_vector_threshold'],
        categories=threshold_cats
    )
    sns.boxplot(x='feature_vector_threshold', y='mse', data=df)
    plt.title('feature_vector_threshold vs MSE')

    # 7. feature_vector_top_k vs R²
    plt.subplot(5, 2, 7)
    sns.boxplot(x='feature_vector_top_k', y='r2', data=df)
    plt.title('feature_vector_top_k vs R²')

    # 8. feature_vector_threshold vs R²
    plt.subplot(5, 2, 8)
    sns.boxplot(x='feature_vector_threshold', y='r2', data=df)
    plt.title('feature_vector_threshold vs R²')

    # 9. 带宽 vs R²
    plt.subplot(5, 2, 9)
    sns.scatterplot(x='bandwidth', y='r2', data=df)
    plt.title('带宽 vs R²')

    # 10. 正则化系数 vs R²
    plt.subplot(5, 2, 10)
    sns.scatterplot(x='reg', y='r2', data=df)
    plt.title('正则化系数 vs R²')

    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "hyperparameter_search_results_embedding.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"超参图表已保存到: {save_path}")


def plot_prediction_scatter(y_true, y_pred, title, save_path):
    """绘制预测值vs实际值散点图"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"预测散点图已保存到: {save_path}")


# 新增：原子重要性结果保存函数
def save_atom_importance(first_mol_smiles, first_mol_importance, log_dir):
    """保存测试集第一个分子的原子重要性到文件"""
    save_path = os.path.join(log_dir, "test_first_mol_atom_importance.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"测试集第一个分子的SMILES: {first_mol_smiles}\n")
        f.write("原子索引 | 原子重要性\n")
        f.write("-" * 30 + "\n")
        for atom_idx, importance in first_mol_importance:
            f.write(f"{atom_idx:8d} | {importance:.6f}\n")
    logger.info(f"测试集第一个分子的原子重要性已保存到: {save_path}")


# 新增：绘制原子重要性图函数
def plot_atom_importance(first_mol_smiles, first_mol_importance_sorted, log_dir, max_atoms_to_show=20):
    """绘制测试集第一个分子的原子重要性条形图"""
    # 提取原子索引和重要性值
    if len(first_mol_importance_sorted) == 0:
        logger.warning("没有有效的原子重要性数据可绘制")
        return

    # 如果原子数量太多，只显示最重要的前max_atoms_to_show个
    if len(first_mol_importance_sorted) > max_atoms_to_show:
        atom_data = first_mol_importance_sorted[:max_atoms_to_show]
        title_suffix = f" (Top {max_atoms_to_show})"
    else:
        atom_data = first_mol_importance_sorted
        title_suffix = ""

    atom_indices = [f"Atom {idx}" for idx, _ in atom_data]
    importance_values = [imp for _, imp in atom_data]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 使用seaborn绘制条形图
    colors = plt.cm.viridis(np.linspace(0, 1, len(atom_data)))
    bars = plt.barh(atom_indices, importance_values, color=colors, alpha=0.7)

    # 添加数值标签
    for bar, value in zip(bars, importance_values):
        width = bar.get_width()
        plt.text(width + 0.001 * max(importance_values), bar.get_y() + bar.get_height() / 2,
                 f'{value:.4f}', ha='left', va='center', fontsize=9)

    # 美化图形
    plt.xlabel('原子重要性', fontsize=12)
    plt.ylabel('原子索引', fontsize=12)
    plt.title(
        f'测试集第一个分子的原子重要性{title_suffix}\nSMILES: {first_mol_smiles[:50]}{"..." if len(first_mol_smiles) > 50 else ""}',
        fontsize=13, pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(log_dir, "test_first_mol_atom_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"原子重要性图已保存到: {save_path}")


# 新增：绘制原子重要性热力图函数（可选）
def plot_atom_importance_heatmap(first_mol_smiles, first_mol_importance, log_dir):
    """绘制原子重要性的热力图风格图"""
    if len(first_mol_importance) == 0:
        return

    # 提取数据
    atom_indices = [idx for idx, _ in first_mol_importance]
    importance_values = [imp for _, imp in first_mol_importance]

    # 创建热力图数据
    n_atoms = len(atom_indices)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1：条形图
    bars = ax1.bar(range(n_atoms), importance_values,
                   color=plt.cm.RdYlBu_r(np.linspace(0, 1, n_atoms)))
    ax1.set_xlabel('原子索引')
    ax1.set_ylabel('重要性值')
    ax1.set_title(f'原子重要性分布 - {first_mol_smiles[:40]}{"..." if len(first_mol_smiles) > 40 else ""}')
    ax1.grid(True, alpha=0.3)

    # 在条形上添加数值
    for i, (bar, value) in enumerate(zip(bars, importance_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{value:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)

    # 子图2：热力图风格
    im = ax2.imshow([importance_values], cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(n_atoms))
    ax2.set_xticklabels([f'Atom {idx}' for idx in atom_indices], rotation=45)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['重要性'])
    ax2.set_title('原子重要性热力图')

    # 添加颜色条
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 在热力图上添加数值
    for i, value in enumerate(importance_values):
        ax2.text(i, 0, f'{value:.3f}', ha='center', va='center',
                 color='black' if value < max(importance_values) * 0.7 else 'white', fontsize=9)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(log_dir, "test_first_mol_atom_importance_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"原子重要性热力图已保存到: {save_path}")


# 主训练流程（增强：支持分子级属性标准化+原子重要性提取）
def main(args):
    # 初始化目录
    log_dir = os.path.join(args.log_dir, "esol")
    os.makedirs(log_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 1. 加载嵌入数据（含分子级属性提取）
    X, y, embed_info = load_embedding_data(
        embedding_dir=args.embedding_dir,
        embedding_filename=args.embedding_filename,
        mol_feat_type=args.mol_feat_type,
        format=args.embedding_format,
        use_mol_emb=args.use_mol_emb,
        use_mol_props=args.use_mol_props  # 控制是否提取分子级属性
    )
    atom_mask = embed_info["atom_mask"]
    smiles_list = embed_info["smiles"]
    use_mol_emb = embed_info["use_mol_emb"]
    use_mol_props = embed_info["use_mol_props"]
    mol_props = embed_info["mol_props"]  # 原始分子属性（未标准化）
    mol_prop_names = embed_info["mol_prop_names"]
    max_num_atoms = embed_info["max_num_atoms"]

    logger.info(f"[数据概览] X={X.shape}, y={y.shape}, atom_mask={atom_mask.shape}, "
                f"分子属性数={len(mol_prop_names)}, SMILES数={len(smiles_list)}, 最大原子数={max_num_atoms}")

    # 2. 执行分割
    logger.info(f"\n[开始分割] 分割方式：{args.split_type} | 随机种子：{args.seed}")
    if args.split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            smiles_list=smiles_list,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_balanced=args.split_balanced,
            random_seed=args.seed
        )
    else:  # 随机分割
        train_idx, val_idx, test_idx = random_split(
            total_samples=len(smiles_list),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )

    # 3. 切分数据（训练/验证/测试）
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    atom_mask_train = atom_mask[train_idx] if not use_mol_emb else None
    atom_mask_val = atom_mask[val_idx] if not use_mol_emb else None
    atom_mask_test = atom_mask[test_idx] if not use_mol_emb else None

    # 切分分子级属性（若启用）
    mol_props_train = mol_props[train_idx] if (use_mol_props and mol_props is not None) else None
    mol_props_val = mol_props[val_idx] if (use_mol_props and mol_props is not None) else None
    mol_props_test = mol_props[test_idx] if (use_mol_props and mol_props is not None) else None

    logger.info("\n[分割后数据形状]")
    logger.info(f"训练集：X={X_train.shape}, y={y_train.shape}")
    logger.info(f"验证集：X={X_val.shape}, y={y_val.shape}")
    logger.info(f"测试集：X={X_test.shape}, y={y_test.shape}")
    if use_mol_props and mol_props is not None:
        logger.info(
            f"分子属性形状：训练集{mol_props_train.shape} | 验证集{mol_props_val.shape} | 测试集{mol_props_test.shape}")

    # 4. 特征标准化（含分子级属性标准化）
    feat_scaler = None
    mol_prop_scaler = None
    if args.normalize_features or (args.normalize_mol_props and use_mol_props):
        logger.info("\n[开始特征标准化] 基于训练集统计量拟合...")
        X_train, X_val, X_test, feat_scaler, mol_prop_scaler = normalize_features(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            atom_mask_train=atom_mask_train,
            atom_mask_val=atom_mask_val,
            atom_mask_test=atom_mask_test,
            use_mol_emb=use_mol_emb,
            mol_props_train=mol_props_train,
            mol_props_val=mol_props_val,
            mol_props_test=mol_props_test,
            normalize_mol_props=args.normalize_mol_props,
            prop_names=embed_info["mol_prop_names"]  # 传入属性名称列表
        )
        logger.info(f"[标准化完成] 训练集X均值：{np.mean(X_train):.4f}（标准化后应接近0）")
    else:
        logger.info("\n[跳过特征标准化]（--normalize_features和--normalize_mol_props均未启用）")

    # 5. 超参数网格配置
    param_grid = {
        'inner_iters': args.inner_iters,
        'pooling_iters': args.pooling_iters,
        'bandwidth': args.bandwidth,
        'reg': args.reg,
        'feature_vector_top_k': args.feature_vector_top_k,
        'feature_vector_threshold': args.feature_vector_threshold
    }
    logger.info(f"超参数网格: {param_grid}")

    # 6. 超参数网格搜索（传入分子属性）
    logger.info("\n[超参数搜索] 开始网格搜索（基于训练集KFold交叉验证）...")
    best_params, all_results = grid_search(
        X_train, y_train, atom_mask_train, mol_props_train,  # 传入分子属性
        param_grid,
        use_mol_emb=use_mol_emb,
        use_mol_props=use_mol_props,  # 控制是否使用分子属性
        n_jobs=args.n_jobs
    )

    # 保存超参结果
    results_df = pd.DataFrame(all_results)
    results_df_path = os.path.join(log_dir, "hyperparameter_results_embedding.xlsx")
    results_df.to_excel(results_df_path, index=False)
    logger.info(f"\n[结果保存] 超参搜索结果已保存到: {results_df_path}")

    # 打印最佳超参数
    logger.info("\n===== 最佳超参数 =====")
    for param, value in best_params.items():
        if param in ['inner_iters', 'pooling_iters', 'bandwidth', 'reg', 'feature_vector_top_k',
                     'feature_vector_threshold']:
            logger.info(f"{param}: {value}")

    # 打印交叉验证性能
    logger.info("\n===== 最佳参数交叉验证性能（训练集） =====")
    for metric in ['mse', 'mae', 'rmse', 'r2']:
        logger.info(f"{metric}: {best_params[metric]:.4f}")

    # 7. 超参结果可视化
    plot_hyperparameter_results(all_results, log_dir)

    # 8. 训练最终模型
    logger.info("\n===== 训练最终模型（基于最佳超参数） =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RFM(device=device)
    model.fit(
        X_train, y_train,
        atom_mask=atom_mask_train,
        inner_iters=best_params['inner_iters'],
        pooling_iters=best_params['pooling_iters'],
        bandwidth=best_params['bandwidth'],
        reg=best_params['reg'],
        verbose=True,
        feature_vector_top_k=best_params['feature_vector_top_k'],
        feature_vector_threshold=best_params['feature_vector_threshold']
    )

    # ==========================================================================
    # 新添加的代码：保存加权对角矩阵
    # ==========================================================================
    logger.info("\n===== 保存加权对角矩阵 =====")
    try:
        # 定义矩阵保存路径
        diag_matrix_save_path = os.path.join(log_dir, "weighted_diag_matrix")

        # 保存为 numpy 格式 (.npy)
        model.save_weighted_diag_matrix(diag_matrix_save_path, format='npy')

        # 也可以保存为 PyTorch 格式 (.pt)
        model.save_weighted_diag_matrix(diag_matrix_save_path, format='pt')

        logger.info(f"加权对角矩阵已成功保存到: {diag_matrix_save_path}.npy 和 {diag_matrix_save_path}.pt")
    except Exception as e:
        logger.error(f"保存加权对角矩阵时发生错误: {e}")
    # ==========================================================================

    # 9. 评估最终模型 + 提取测试集第一个分子的原子重要性
    logger.info("\n===== 模型评估（验证集+测试集） =====")
    # 验证集评估：适配RFM predict双返回值
    y_val_pred, _ = model.predict(X_val, atom_mask=atom_mask_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)

    # 测试集评估：获取预测结果和原子重要性
    y_test_pred, atom_weights_test = model.predict(X_test, atom_mask=atom_mask_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info(f"[验证集性能] MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
    logger.info(f"[测试集性能] MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

    # 绘制预测散点图
    plot_prediction_scatter(
        y_val, y_val_pred,
        f"验证集预测 vs 实际 (R²={val_r2:.4f})",
        os.path.join(log_dir, "val_pred_scatter.png")
    )
    plot_prediction_scatter(
        y_test, y_test_pred,
        f"测试集预测 vs 实际 (R²={test_r2:.4f})",
        os.path.join(log_dir, "test_pred_scatter.png")
    )

    # 10. 提取并输出测试集第一个分子的原子重要性
    logger.info("\n===== 测试集第一个分子的原子重要性 =====")
    if len(test_idx) == 0:
        logger.warning("测试集为空，无法提取原子重要性")
    else:
        # 获取测试集第一个分子的原始信息
        first_test_idx = test_idx[0]  # 测试集第一个分子在原始数据中的索引
        first_mol_smiles = smiles_list[first_test_idx]  # 第一个分子的SMILES
        first_mol_atom_mask = atom_mask_test[0]  # 第一个分子的原子掩码（过滤padding）
        first_mol_atom_weights = atom_weights_test[0]  # 第一个分子的原子重要性

        # 过滤无效原子（padding原子：掩码为0，重要性接近0）
        valid_atom_mask = first_mol_atom_mask > 1e-8  # 掩码阈值（避免浮点误差）
        valid_atom_indices = np.where(valid_atom_mask)[0]
        valid_atom_importance = first_mol_atom_weights[valid_atom_mask]

        # 整理有效原子的（索引，重要性）对
        first_mol_importance = list(zip(valid_atom_indices, valid_atom_importance))
        first_mol_importance_sorted = sorted(first_mol_importance, key=lambda x: x[1], reverse=True)  # 按重要性降序

        # 日志输出
        logger.info(f"测试集第一个分子的SMILES: {first_mol_smiles}")
        logger.info(f"有效原子数量: {len(valid_atom_indices)}（总原子数上限: {max_num_atoms}）")
        logger.info("原子索引（降序） | 原子重要性")
        logger.info("-" * 35)
        for atom_idx, importance in first_mol_importance_sorted:
            logger.info(f"{atom_idx:15d} | {importance:.6f}")

        # 保存原子重要性到文件
        save_atom_importance(first_mol_smiles, first_mol_importance_sorted, log_dir)

        # ==== 新增：绘制原子重要性图 ====
        try:
            # 绘制标准条形图
            plot_atom_importance(first_mol_smiles, first_mol_importance_sorted, log_dir)

            # 可选：绘制热力图风格（如果原子数量适中）
            if len(first_mol_importance_sorted) <= 30:  # 原子数太多热力图会拥挤
                plot_atom_importance_heatmap(first_mol_smiles, first_mol_importance_sorted, log_dir)

            logger.info("原子重要性可视化完成")
        except Exception as e:
            logger.warning(f"原子重要性绘图失败: {str(e)}")

    # 11. 特征重要性输出（增强：若使用分子属性，标记其重要性）
    try:
        logger.info("\n===== Top10 融合特征重要性 =====")
        feature_importance = model.get_feature_importance()
        importance_sorted = sorted(enumerate(feature_importance), key=lambda x: abs(x[1]), reverse=True)

        # 标记分子属性对应的特征维度
        if use_mol_props and len(mol_prop_names) > 0:
            # 计算分子属性在特征中的起始索引（分子级嵌入/原子级嵌入不同）
            if use_mol_emb:
                prop_start_idx = X_train.shape[1] - len(mol_prop_names)
            else:
                prop_start_idx = X_train.shape[2] - len(mol_prop_names)

            for i, (dim_idx, imp) in enumerate(importance_sorted[:10]):
                if prop_start_idx <= dim_idx < prop_start_idx + len(mol_prop_names):
                    prop_idx = dim_idx - prop_start_idx
                    logger.info(f"Top {i + 1}: 分子属性[{mol_prop_names[prop_idx]}] → 重要性: {imp:.4f}")
                else:
                    logger.info(f"Top {i + 1}: 特征维度{dim_idx} → 重要性: {imp:.4f}")
        else:
            for i, (dim_idx, imp) in enumerate(importance_sorted[:10]):
                logger.info(f"Top {i + 1}: 特征维度{dim_idx} → 重要性: {imp:.4f}")
    except AttributeError:
        logger.warning("\n[警告] RFM模型不支持get_feature_importance方法，跳过特征重要性输出")

    # 12. 保存模型、scaler与结果
    model_state = {
        'M': model.M.cpu().numpy(),
        'alphas': model.alphas.cpu().numpy() if model.alphas is not None else None,
        'molecular_features_train': model.molecular_features_train.cpu().numpy() if model.molecular_features_train is not None else None,
        'bandwidth': model.bandwidth,
        'reg': model.reg,
        'feature_vector_top_k': model.feature_vector_top_k,
        'feature_vector_threshold': model.feature_vector_threshold,
        'max_num_atoms': max_num_atoms  # 新增：保存最大原子数（后续加载预测需用）
    }
    # 保存特征标准化器
    if args.normalize_features and feat_scaler is not None:
        model_state['feat_scaler_mean'] = feat_scaler.mean_
        model_state['feat_scaler_scale'] = feat_scaler.scale_

    # 保存分子属性标准化器（若启用）
    if args.normalize_mol_props and mol_prop_scaler is not None:
        model_state['mol_prop_scaler_mean'] = mol_prop_scaler.mean_
        model_state['mol_prop_scaler_scale'] = mol_prop_scaler.scale_
        model_state['mol_prop_names'] = mol_prop_names  # 保存属性名称

    model_path = os.path.join(log_dir, "esol_rfm_embedding_model.npz")
    np.savez(model_path, **model_state)
    logger.info(f"\n[模型保存] 完整模型状态已保存到: {model_path}")

    # 保存最佳参数和性能（含原子重要性信息）
    params_path = os.path.join(log_dir, "best_hyperparameters_embedding.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("===== 嵌入数据信息 =====\n")
        for key, val in embed_info.items():
            if key not in ["smiles", "atom_mask", "mol_props"]:
                f.write(f"{key}: {val}\n")

        f.write("\n===== 数据处理配置 =====\n")
        f.write(f"mol_feat_type: {args.mol_feat_type}\n")
        f.write(f"use_mol_props: {use_mol_props}\n")
        f.write(f"normalize_features: {args.normalize_features}\n")
        f.write(f"normalize_mol_props: {args.normalize_mol_props}\n")

        if args.normalize_features and feat_scaler is not None:
            f.write(f"特征标准化均值范围: {np.min(feat_scaler.mean_):.4f} ~ {np.max(feat_scaler.mean_):.4f}\n")
            f.write(f"特征标准化标准差范围: {np.min(feat_scaler.scale_):.4f} ~ {np.max(feat_scaler.scale_):.4f}\n")

        if args.normalize_mol_props and mol_prop_scaler is not None:
            f.write("\n===== 分子属性标准化统计 =====\n")
            for i, prop_name in enumerate(mol_prop_names):
                f.write(
                    f"{prop_name} - 均值: {mol_prop_scaler.mean_[i]:.4f} | 标准差: {mol_prop_scaler.scale_[i]:.4f}\n")

        f.write("\n===== 分割配置 =====\n")
        f.write(f"split_type: {args.split_type}\n")
        if args.split_type == "scaffold":
            f.write(f"split_balanced: {args.split_balanced}\n")
        f.write(f"train_ratio: {args.train_ratio} | val_ratio: {args.val_ratio} | test_ratio: {args.test_ratio}\n")
        f.write(f"random_seed: {args.seed}\n")

        f.write("\n===== 最佳超参数 =====\n")
        for param, value in best_params.items():
            if param in ['inner_iters', 'pooling_iters', 'bandwidth', 'reg', 'feature_vector_top_k',
                         'feature_vector_threshold']:
                f.write(f"{param}: {value}\n")

        f.write("\n===== 交叉验证性能（训练集） =====\n")
        for metric in ['mse', 'mae', 'rmse', 'r2']:
            f.write(f"{metric}: {best_params[metric]:.4f}\n")

        f.write("\n===== 最终模型性能 =====\n")
        f.write(f"[验证集] MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}\n")
        f.write(f"[测试集] MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}\n")

        # 新增：写入测试集第一个分子的原子重要性摘要
        if len(test_idx) > 0:
            f.write("\n===== 测试集第一个分子原子重要性摘要 =====\n")
            f.write(f"SMILES: {first_mol_smiles}\n")
            f.write(f"有效原子数: {len(valid_atom_indices)}\n")
            f.write("Top5 重要原子：\n")
            for i, (atom_idx, importance) in enumerate(first_mol_importance_sorted[:5]):
                f.write(f"  Top{i + 1}: 原子{atom_idx} → 重要性{importance:.6f}\n")
            f.write(f"完整原子重要性文件：test_first_mol_atom_importance.txt\n")

        # 新增：记录加权对角矩阵保存路径
        f.write("\n===== 模型衍生文件 =====\n")
        f.write(f"加权对角矩阵 (Numpy格式): weighted_diag_matrix.npy\n")
        f.write(f"加权对角矩阵 (PyTorch格式): weighted_diag_matrix.pt\n")

    logger.info(f"[结果保存] 最佳参数与性能已保存到: {params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM模型训练与评估（支持分子级属性标准化+原子重要性提取）")
    # 数据路径参数
    parser.add_argument("--embedding_dir", type=str, default="./data/ESOL", help="嵌入文件目录")
    parser.add_argument("--embedding_filename", type=str, default="esol_embeddings_with_atom_codes",
                        help="嵌入文件名（无后缀）")
    parser.add_argument("--embedding_format", type=str, default="pkl", choices=["pkl", "npz"], help="嵌入文件格式")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志与结果保存目录")

    # 特征融合与标准化参数
    parser.add_argument("--mol_feat_type", type=str, default="rdkit+graph",
                        choices=["rdkit", "graph", "rdkit+graph"],
                        help="分子特征类型：rdkit（仅RDKit特征）、graph（仅预计算嵌入）、rdkit+graph（融合）")
    parser.add_argument("--normalize_features", action="store_true", default=False,
                        help="是否启用特征标准化（基于训练集）")

    # 新增：分子级属性相关参数
    parser.add_argument("--use_mol_props", action="store_true", default=False,
                        help="是否使用分子级属性（分子量、MolLogP等）")
    parser.add_argument("--normalize_mol_props", action="store_true", default=True,
                        help="是否标准化分子级属性（仅当--use_mol_props启用时有效）")

    # 数据处理参数
    parser.add_argument("--use_mol_emb", action="store_true", help="是否使用分子级嵌入（默认原子级）")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--split_balanced", action="store_true", default=False,
                        help="是否启用平衡骨架分割（仅scaffold分割有效）")
    parser.add_argument("--split_type", type=str, default="random",
                        choices=["scaffold", "random"],
                        help="数据集分割方式：scaffold（分子骨架分割）、random（随机分割）")

    # 超参数搜索参数
    parser.add_argument("--inner_iters", type=int, nargs="+", default=[23], help="内层迭代次数列表")
    parser.add_argument("--pooling_iters", type=int, nargs="+", default=[26], help="外层池化迭代次数列表")
    parser.add_argument("--bandwidth", type=float, nargs="+", default=[85], help="核带宽列表")
    parser.add_argument("--reg", type=float, nargs="+", default=[1e-1], help="正则化系数列表")
    parser.add_argument("--feature_vector_top_k", type=int, nargs="+", default=[5], help="保留特征向量数量列表")
    parser.add_argument("--feature_vector_threshold", type=float, nargs="+", default=[None], help="特征值累计阈值列表")

    # 其他参数
    parser.add_argument("--n_jobs", type=int, default=15, help="并行进程数（-1表示全部）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    main(args)
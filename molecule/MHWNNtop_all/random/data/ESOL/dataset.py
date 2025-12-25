import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random  # Python原生随机模块
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from torch.cuda.amp import autocast


# -------------------------- 修复：随机种子设置函数（移除旧RDKit不支持的API） --------------------------
def set_random_seed(seed: int = 42, disable_rdkit_log: bool = True):
    """
    设置全流程随机种子，保证实验可重复性（适配旧版RDKit）
    Args:
        seed: 随机种子值（默认42，常用且推荐）
        disable_rdkit_log: 是否关闭RDKit冗余日志（可选，减少输出干扰）
    """
    # 1. Python原生随机模块（RDKit部分操作依赖）
    random.seed(seed)
    # 2. NumPy随机模块（RDKit构象生成、ETKDG等核心随机逻辑依赖）
    np.random.seed(seed)
    # 3. PyTorch CPU随机种子
    torch.manual_seed(seed)
    # 4. PyTorch GPU随机种子（单GPU/多GPU均适用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时覆盖所有设备
    # 5. CuDNN参数（避免GPU算法随机性）
    torch.backends.cudnn.deterministic = True  # 强制确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用自动选最快算法（可能引入随机）
    # 可选：关闭RDKit警告日志（避免构象生成时的冗余输出）
    if disable_rdkit_log:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')


# 扩展的元素类型映射（保持不变）
EXTENDED_ELEMENT_MAP = {
    'H': 1, 'He': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'P': 9, 'S': 10, 'Cl': 11, 'Ar': 12, 'As': 13, 'Se': 14, 'Br': 15,
    'Kr': 16, 'Te': 17, 'I': 18, 'Xe': 19,
    'Li': 20, 'Be': 21, 'Na': 22, 'Mg': 23, 'K': 24, 'Ca': 25, 'Rb': 26,
    'Sr': 27, 'Cs': 28, 'Ba': 29,
    'Sc': 30, 'Ti': 31, 'V': 32, 'Cr': 33, 'Mn': 34, 'Fe': 35, 'Co': 36,
    'Ni': 37, 'Cu': 38, 'Zn': 39, 'Y': 40, 'Zr': 41, 'Nb': 42, 'Mo': 43,
    'Tc': 44, 'Ru': 45, 'Rh': 46, 'Pd': 47, 'Ag': 48, 'Cd': 49, 'Hf': 50,
    'Ta': 51, 'W': 52, 'Re': 53, 'Os': 54, 'Ir': 55, 'Pt': 56, 'Au': 57,
    'Hg': 58,
    'Al': 59, 'Ga': 60, 'In': 61, 'Sn': 62, 'Tl': 63, 'Pb': 64, 'Bi': 65,
}


# 高斯基函数层（保持不变）
class GaussianLayer(nn.Module):
    def __init__(self, k: int, edge_types: int, cutoff: float = 5.0):
        super().__init__()
        self.k = k
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0, cutoff, k))
        self.widths = nn.Parameter(torch.full((k,), cutoff / (k - 1) if k > 1 else 1.0))
        self.type_bias = nn.Embedding(edge_types, k)

    def forward(self, dist: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        cutoff_mask = (dist <= self.cutoff).float()
        dist = dist.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        widths = self.widths.view(1, 1, 1, -1)
        gaussian = torch.exp(-((dist - centers) / widths) ** 2)
        type_bias = self.type_bias(edge_type)
        return gaussian * cutoff_mask.unsqueeze(-1) + type_bias


# 分子编码器（保持不变）
class MolecularEncoderExtended(nn.Module):
    def __init__(
            self,
            embed_dim: int = None,
            max_atom_types: int = len(EXTENDED_ELEMENT_MAP) + 1,
            gbf_k: int = 64,
            cutoff_dist: float = 5.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_atom_types = max_atom_types
        self.atom_encoder = nn.Embedding(
            num_embeddings=max_atom_types,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.edge_types = max_atom_types * max_atom_types
        self.gbf = GaussianLayer(k=gbf_k, edge_types=self.edge_types, cutoff=cutoff_dist)
        self.edge_proj = nn.Linear(gbf_k, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, atoms: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_size, max_num_atoms = atoms.shape
        padding_mask = atoms.eq(0)
        edge_type = atoms.view(batch_size, max_num_atoms, 1) * self.max_atom_types + \
                    atoms.view(batch_size, 1, max_num_atoms)
        delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = delta_pos.norm(dim=-1)
        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        atom_3d_feature = edge_features.sum(dim=2)
        atom_3d_feature_proj = self.edge_proj(atom_3d_feature)
        atom_type_embed = self.atom_encoder(atoms)
        fused_emb = atom_type_embed + atom_3d_feature_proj
        fused_emb = self.norm(fused_emb)
        fused_emb = fused_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return fused_emb


# 数据预处理（保持不变）
def smiles_to_features_extended(smiles: str, max_num_atoms: int) -> tuple:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol = Chem.AddHs(mol)
        # 关键：ETKDG构象生成依赖numpy种子，已通过set_random_seed固定
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        atoms = []
        pos = []
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            atoms.append(EXTENDED_ELEMENT_MAP.get(elem, 0))
            coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            pos.append([coords.x, coords.y, coords.z])
        pad_length = max_num_atoms - len(atoms)
        if pad_length > 0:
            atoms += [0] * pad_length
            pos += [[0.0, 0.0, 0.0]] * pad_length
        elif pad_length < 0:
            atoms = atoms[:max_num_atoms]
            pos = pos[:max_num_atoms]
        return np.array(atoms, dtype=np.int32), np.array(pos, dtype=np.float32)
    except Exception as e:
        print(f"处理SMILES失败: {smiles}, 错误: {e}")
        return None, None


# 批量处理函数（保持不变）
def process_esol_extended(
        csv_path: str,
        max_num_atoms: int = 55,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    df = pd.read_csv(csv_path)
    required_cols = ["SMILES", "(solubility:mol/L)"]
    if not set(required_cols).issubset(df.columns):
        existing_cols = df.columns.tolist()
        raise ValueError(f"需要列: {required_cols}, 实际列: {existing_cols}")

    raw_data = df[required_cols].reset_index()
    smiles_list = raw_data["SMILES"].tolist()
    labels = raw_data["(solubility:mol/L)"].values
    original_indices = raw_data["index"].tolist()

    all_atoms_list = []
    all_pos_list = []
    valid_smiles = []
    valid_labels = []
    valid_original_indices = []

    print("预处理分子并保存原子编码...")
    for smiles, label, orig_idx in tqdm(zip(smiles_list, labels, original_indices),
                                        total=len(smiles_list)):
        atoms, pos = smiles_to_features_extended(smiles, max_num_atoms)
        if atoms is not None and pos is not None:
            all_atoms_list.append(atoms)
            all_pos_list.append(pos)
            valid_smiles.append(smiles)
            valid_labels.append(label)
            valid_original_indices.append(orig_idx)

    print(f"预处理完成: 有效分子 {len(valid_smiles)}/{len(smiles_list)}")

    all_atoms = np.array(all_atoms_list, dtype=np.int32)
    all_pos = np.array(all_pos_list, dtype=np.float32)

    encoder = MolecularEncoderExtended(
        embed_dim=256,
        max_atom_types=len(EXTENDED_ELEMENT_MAP) + 1,
        gbf_k=64,
        cutoff_dist=5.0
    ).to(device)
    encoder.eval()

    all_embeddings = []
    num_batches = (len(all_atoms) + batch_size - 1) // batch_size

    print(f"在{device}上生成分子嵌入...")
    with torch.no_grad():
        for b in tqdm(range(num_batches), total=num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, len(all_atoms))

            batch_atoms = torch.from_numpy(all_atoms[start:end]).to(device)
            batch_pos = torch.from_numpy(all_pos[start:end]).to(device)

            with autocast():
                batch_emb = encoder(batch_atoms, batch_pos)
            all_embeddings.append(batch_emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return {
        "embeddings": all_embeddings,
        "smiles": valid_smiles,
        "labels": np.array(valid_labels),
        "original_indices": valid_original_indices,
        "all_atoms": all_atoms,
        "element_map": EXTENDED_ELEMENT_MAP,
        "max_num_atoms": max_num_atoms,
        "embed_dim": 256
    }


# 保存函数（保持不变）
def save_esol_embeddings(result: dict, save_path: str, format: str = "pkl") -> None:
    if format == "npz":
        np.savez(
            f"{save_path}.npz",
            embeddings=result["embeddings"],
            smiles=np.array(result["smiles"], dtype=str),
            labels=result["labels"],
            original_indices=result["original_indices"],
            all_atoms=result["all_atoms"],
            elem_symbols=np.array(list(result["element_map"].keys()), dtype=str),
            elem_codes=np.array(list(result["element_map"].values()), dtype=int),
            max_num_atoms=result["max_num_atoms"],
            embed_dim=result["embed_dim"]
        )
        print(f"NPZ文件保存完成: {save_path}.npz")
    elif format == "pkl":
        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"PKL文件保存完成: {save_path}.pkl")
    else:
        raise ValueError("支持格式: 'npz' 或 'pkl'")


# 加载函数（保持不变）
def load_esol_data(save_path: str, format: str = "pkl"):
    if format == "pkl":
        with open(f"{save_path}.pkl", "rb") as f:
            return pickle.load(f)
    elif format == "npz":
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


# 验证原子嵌入对应关系（保持不变）
def verify_atom_embedding_mapping(esol_data):
    element_map = esol_data["element_map"]
    code_to_element = {v: k for k, v in element_map.items()}

    # np.random.choice依赖numpy种子，已通过set_random_seed固定
    sample_indices = np.random.choice(len(esol_data["smiles"]), 5, replace=False)
    for idx in sample_indices:
        smiles = esol_data["smiles"][idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol_with_h = Chem.AddHs(mol)
        expected_atoms = [atom.GetSymbol() for atom in mol_with_h.GetAtoms()]
        expected_num = len(expected_atoms)

        atom_codes = esol_data["all_atoms"][idx]
        valid_indices = np.where(atom_codes != 0)[0]
        valid_elements = [code_to_element[code] for code in atom_codes[valid_indices]]
        actual_num = len(valid_elements)

        atom_embeddings = esol_data["embeddings"][idx][valid_indices]

        print(f"\n分子SMILES: {smiles}")
        print(f"预期原子数（含氢）: {expected_num}, 实际原子数: {actual_num}")
        print(f"原子是否匹配: {expected_num == actual_num}")
        print(f"前3个原子（预期）: {expected_atoms[:3]}")
        print(f"前3个原子（实际）: {valid_elements[:3]}")
        print(f"对应嵌入形状: {atom_embeddings[:3].shape}")


# 提取单个分子的原子嵌入（保持不变）
def get_atom_embeddings_for_molecule(esol_data, molecule_idx: int):
    atom_codes = esol_data["all_atoms"][molecule_idx]
    valid_indices = np.where(atom_codes != 0)[0]
    code_to_element = {v: k for k, v in esol_data["element_map"].items()}

    return {
        "smiles": esol_data["smiles"][molecule_idx],
        "elements": [code_to_element[atom_codes[i]] for i in valid_indices],
        "embeddings": esol_data["embeddings"][molecule_idx][valid_indices],
        "num_atoms": len(valid_indices)
    }


# 主函数：先设置Seed再执行
if __name__ == "__main__":
    # 1. 优先设置随机种子（必须在所有随机操作前执行！）
    set_random_seed(seed=42, disable_rdkit_log=True)  # 可修改seed值（如123、666）

    # 2. 后续流程（与原代码一致）
    ESOL_CSV_PATH = "ESOL.csv"  # 替换为你的数据集实际路径
    SAVE_PATH = "esol_embeddings_with_atom_codes"
    esol_data = process_esol_extended(ESOL_CSV_PATH)
    save_esol_embeddings(esol_data, SAVE_PATH)

    loaded_data = load_esol_data(SAVE_PATH)
    verify_atom_embedding_mapping(loaded_data)

    first_mol = get_atom_embeddings_for_molecule(loaded_data, 0)
    print(f"\n第一个分子的原子嵌入:")
    print(f"元素列表: {first_mol['elements']}")
    print(f"嵌入形状: {first_mol['embeddings'].shape}")
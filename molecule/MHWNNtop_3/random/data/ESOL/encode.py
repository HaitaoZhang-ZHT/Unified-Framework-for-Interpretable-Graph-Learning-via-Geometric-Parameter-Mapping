import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from torch.cuda.amp import autocast

# 扩展的元素类型映射
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


# 高斯基函数层
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


# 分子编码器
class MolecularEncoderExtended(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
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


# 数据预处理（增加SMILES验证）
def smiles_to_features_extended(smiles: str, max_num_atoms: int) -> tuple:
    try:
        # 验证SMILES有效性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无效SMILES: {smiles}")
            return None, None

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        atoms = []
        pos = []
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            atoms.append(EXTENDED_ELEMENT_MAP.get(elem, 0))
            coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            pos.append([coords.x, coords.y, coords.z])

        # 处理长度
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


# 批量处理函数（确保SMILES与嵌入一一对应）
def process_esol_extended(
        csv_path: str,
        max_num_atoms: int = 100,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    # 加载数据集
    df = pd.read_csv(csv_path)
    required_cols = ["SMILES", "(solubility:mol/L)"]
    if not set(required_cols).issubset(df.columns):
        existing_cols = df.columns.tolist()
        raise ValueError(f"需要列: {required_cols}, 实际列: {existing_cols}")

    # 提取原始数据（保留索引用于追溯）
    raw_data = df[required_cols].reset_index()  # 保留原始索引
    smiles_list = raw_data["SMILES"].tolist()
    labels = raw_data["(solubility:mol/L)"].values
    original_indices = raw_data["index"].tolist()  # 原始数据集索引

    # 预处理并保存有效SMILES
    all_atoms = []
    all_pos = []
    valid_smiles = []  # 与嵌入一一对应的SMILES
    valid_labels = []
    valid_original_indices = []  # 原始索引，方便回溯

    print("预处理分子并保存SMILES...")
    for idx, (smiles, label, orig_idx) in tqdm(enumerate(zip(smiles_list, labels, original_indices)),
                                               total=len(smiles_list)):
        atoms, pos = smiles_to_features_extended(smiles, max_num_atoms)
        if atoms is not None and pos is not None:
            all_atoms.append(atoms)
            all_pos.append(pos)
            valid_smiles.append(smiles)  # 保存有效SMILES
            valid_labels.append(label)
            valid_original_indices.append(orig_idx)

    print(f"预处理完成: 有效分子 {len(valid_smiles)}/{len(smiles_list)}")

    # 生成嵌入
    encoder = MolecularEncoderExtended(
        embed_dim=128,
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
            batch_atoms = torch.tensor(all_atoms[start:end], dtype=torch.long).to(device)
            batch_pos = torch.tensor(all_pos[start:end], dtype=torch.float32).to(device)
            with autocast():
                batch_emb = encoder(batch_atoms, batch_pos)
            all_embeddings.append(batch_emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # 验证SMILES与嵌入数量一致
    assert len(valid_smiles) == len(all_embeddings), \
        f"SMILES数量({len(valid_smiles)})与嵌入数量({len(all_embeddings)})不匹配"

    return {
        "embeddings": all_embeddings,  # 分子嵌入
        "smiles": valid_smiles,  # 对应SMILES（用于构建图）
        "labels": np.array(valid_labels),  # 溶解度标签
        "original_indices": valid_original_indices,  # 原始数据集索引
        "element_map": EXTENDED_ELEMENT_MAP,
        "max_num_atoms": max_num_atoms,
        "embed_dim": 128
    }


# 保存函数（优化SMILES存储格式）
def save_esol_embeddings(result: dict, save_path: str, format: str = "pkl") -> None:
    """推荐使用pkl格式，完美保存字符串列表和字典"""
    if format == "npz":
        # NPZ保存时确保SMILES为字符串数组
        np.savez(
            f"{save_path}.npz",
            embeddings=result["embeddings"],
            smiles=np.array(result["smiles"], dtype=str),  # 显式指定字符串类型
            labels=result["labels"],
            original_indices=result["original_indices"],
            elem_symbols=np.array(list(result["element_map"].keys()), dtype=str),
            elem_codes=np.array(list(result["element_map"].values()), dtype=int),
            max_num_atoms=result["max_num_atoms"],
            embed_dim=result["embed_dim"]
        )
        print(f"NPZ文件保存完成: {save_path}.npz")
    elif format == "pkl":
        # PKL格式保留原始数据结构，更适合后续图构建
        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"PKL文件保存完成: {save_path}.pkl")
    else:
        raise ValueError("支持格式: 'npz' 或 'pkl'（推荐pkl）")


# 后续构建图的示例函数（基于保存的SMILES）
def build_molecular_graphs_from_smiles(smiles_list: list):
    """示例：从SMILES构建分子图（节点=原子，边=化学键）"""
    graphs = []
    for smiles in tqdm(smiles_list, desc="从SMILES构建图"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 节点特征：原子类型（可与之前的element_map映射）
        nodes = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # 边特征：化学键类型（单键=1，双键=2，三键=3，芳香键=4）
        edges = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # 转换为数值
            edges.append((start, end, bond_type))
            edges.append((end, start, bond_type))  # 无向图

        graphs.append({
            "smiles": smiles,
            "nodes": nodes,
            "edges": edges
        })
    return graphs


if __name__ == "__main__":
    ESOL_CSV_PATH = "ESOL.csv"  # 替换为你的数据集路径
    SAVE_PATH = "esol_embeddings_with_smiles"
    MAX_NUM_ATOMS = 100
    BATCH_SIZE = 64
    SAVE_FORMAT = "pkl"  # 推荐用pkl保存，方便后续图构建

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    if not os.path.exists(ESOL_CSV_PATH):
        raise FileNotFoundError(f"未找到数据集: {ESOL_CSV_PATH}")

    # 处理并保存（包含SMILES）
    esol_data = process_esol_extended(
        csv_path=ESOL_CSV_PATH,
        max_num_atoms=MAX_NUM_ATOMS,
        batch_size=BATCH_SIZE,
        device=device
    )
    save_esol_embeddings(esol_data, SAVE_PATH, SAVE_FORMAT)

    # 示例：加载并使用SMILES构建图
    print("\n示例：从保存的SMILES构建分子图...")
    with open(f"{SAVE_PATH}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    graphs = build_molecular_graphs_from_smiles(loaded_data["smiles"][:5])  # 处理前5个分子
    print(f"成功构建{len(graphs)}个分子图")
    print("第一个分子图信息:")
    print(f"SMILES: {graphs[0]['smiles']}")
    print(f"原子节点: {graphs[0]['nodes']}")
    print(f"化学键边: {graphs[0]['edges'][:5]}...")  # 显示前5条边

import numpy as np
import torch
import os
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from community import community_louvain
import networkx as nx

# 分子任务必需：RDKit用于分子特征处理（新增骨架提取工具）
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold  # 新增：用于提取分子骨架
except ImportError:
    raise ImportError("请安装RDKit以处理分子数据：pip install rdkit-pypi")


class DataPreprocessor:
    """数据预处理类，支持节点任务/图任务（含ESOL分子任务，增强版Scaffold Split）"""

    def __init__(self, dataset_name, root_data_path="./data",
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 normalize_features=None, device="cpu",
                 n_clusters=100,
                 # 分子任务专属参数
                 task_type=None,
                 mol_feat_type="rdkit+graph",  # 分子特征类型：rdkit/graph/rdkit+graph
                 normalize_mol_props=True,
                 split_seed=None,
                 # 分割方式控制（增强：支持平衡骨架分割）
                 split_type="random",  # random=随机分割，scaffold=骨架分割
                 split_balanced=False):  # 是否启用平衡骨架分割（避免大组集中）
        self.dataset_name = dataset_name
        self.root_data_path = root_data_path
        # 任务类型标识（区分节点/图/分子任务）
        self.task_type = task_type
        # 分子任务专属参数
        self.mol_feat_type = mol_feat_type
        self.normalize_mol_props = normalize_mol_props
        self.split_seed = split_seed if split_seed is not None else 42  # 默认种子确保可复现
        self.split_type = split_type
        self.split_balanced = split_balanced  # 新增：平衡分割开关

        # 任务类型判断（兼容原有逻辑+分子任务）
        self.is_graph_task = self.dataset_name in ["graph_classification", "graph_regression"] or \
                             (self.task_type is not None and "graph" in self.task_type)
        self.is_mol_task = self.dataset_name.lower() == "esol" and self.task_type == "graph_regression"

        # 数据集路径配置
        if self.is_mol_task:
            self.mol_data_path = os.path.join(root_data_path, "ESOL", "esol_embeddings_with_atom_codes.pkl")
        elif self.is_graph_task:
            self.graph_data_path = os.path.join(root_data_path, "random")
        else:
            self.node_data_folder = os.path.join(root_data_path, dataset_name.capitalize())

        # 通用参数
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize_features = normalize_features
        self.device = device
        self.n_clusters = n_clusters

        # 数据存储变量（原有逻辑+分子任务扩展）
        self.features = None
        self.labels = None
        self.adj = None
        self.attributes = None
        self.clusters = None
        self.communities = None
        self.num_nodes = None
        self.num_classes = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.train_graphs = None
        self.val_graphs = None
        self.test_graphs = None
        self.num_features = None
        self.graph_label_dim = 1
        self.mol_valid_masks = None  # 分子有效原子掩码

        # 工具变量
        self.scaler = None
        self.attr_scaler = None
        self.mol_prop_scaler = StandardScaler() if self.normalize_mol_props else None

        # 确保目录存在
        os.makedirs(root_data_path, exist_ok=True)
        if self.is_mol_task:
            os.makedirs(os.path.dirname(self.mol_data_path), exist_ok=True)
        elif self.is_graph_task:
            os.makedirs(self.graph_data_path, exist_ok=True)

    def _load_raw_data(self):
        """加载原始数据（分支：节点任务/普通图任务/分子任务）"""
        if self.is_mol_task:
            self._load_mol_esol_data()
        elif self.is_graph_task:
            self._load_graph_task_data()
        else:
            self._load_node_task_data()

    # -------------------------- 增强：分子骨架提取（优化异常处理） --------------------------
    def _get_murcko_scaffold(self, smiles):
        """
        从SMILES提取Murcko骨架（增强版：完善异常处理）
        返回：骨架SMILES字符串（无效分子返回带标识的临时骨架）
        """
        # 1. 解析SMILES（捕获RDKit可能抛出的异常）
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"invalid_smiles_{smiles[:20]}"  # 截断长SMILES避免标识过长
        except Exception as e:
            return f"parse_error_{smiles[:20]}_{str(e)[:10]}"  # 记录解析错误原因

        # 2. 提取Murcko骨架（含环结构，去除侧链）
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
        except Exception as e:
            return f"scaffold_error_{smiles[:20]}_{str(e)[:10]}"  # 记录骨架提取错误

        # 3. 处理线性分子（无环结构，无骨架SMILES）
        if not scaffold_smiles:
            return f"linear_mol_{smiles[:20]}"

        return scaffold_smiles

    # -------------------------- 增强：ESOL分子数据加载（添加分割日志） --------------------------
    def _load_mol_esol_data(self):
        """加载ESOL分子数据集（增强：打印分割配置详情）"""
        # 1. 检查文件存在性
        if not os.path.exists(self.mol_data_path):
            raise FileNotFoundError(
                f"ESOL文件缺失：{self.mol_data_path}\n"
                f"请将ESOL数据文件放入 ./data/ESOL/ 目录，文件格式需包含：\n"
                f"smiles, labels, all_atoms, element_map, max_num_atoms, embeddings 字段"
            )

        # 2. 加载数据
        print(f"[ESOL加载] 从 {self.mol_data_path} 读取数据...")
        with open(self.mol_data_path, 'rb') as f:
            esol_data = pickle.load(f)

        # 3. 解析核心字段（增强：添加字段校验）
        required_fields = ["smiles", "labels", "all_atoms", "element_map", "max_num_atoms", "embeddings"]
        for field in required_fields:
            if field not in esol_data:
                raise KeyError(f"ESOL数据缺失必需字段：{field}，请检查数据格式")

        smiles_list = esol_data["smiles"]
        labels = esol_data["labels"].astype(np.float32)
        all_atoms = esol_data["all_atoms"]
        max_num_atoms = esol_data["max_num_atoms"]
        embeddings = esol_data["embeddings"]
        self.num_features = embeddings.shape[2]  # 初始特征维度（嵌入维度）

        # 4. 生成分子图结构
        mol_graphs = []
        for mol_idx in range(len(smiles_list)):
            smiles = smiles_list[mol_idx]
            atom_codes = all_atoms[mol_idx]
            atom_embeds = embeddings[mol_idx]

            # 4.1 有效原子掩码（过滤填充原子）
            valid_mask = (atom_codes != 0).astype(bool)

            # 4.2 原子特征生成（按mol_feat_type）- 核心修复：graph分支补充mol定义
            try:
                if self.mol_feat_type == "rdkit":
                    mol = Chem.MolFromSmiles(smiles)
                    atom_features = self._extract_rdkit_atom_feats(mol, max_num_atoms)
                elif self.mol_feat_type == "graph":
                    # 修复：即使使用graph特征，也需解析SMILES获取mol（用于后续提取化学键）
                    mol = Chem.MolFromSmiles(smiles)
                    atom_features = atom_embeds
                else:  # rdkit+graph 融合
                    mol = Chem.MolFromSmiles(smiles)
                    rdkit_feats = self._extract_rdkit_atom_feats(mol, max_num_atoms)
                    atom_features = np.concatenate([rdkit_feats, atom_embeds], axis=1)
                    self.num_features = atom_features.shape[1]  # 更新融合后维度
            except Exception as e:
                raise ValueError(
                    f"分子 {mol_idx + 1}（SMILES: {smiles}）特征生成失败：{str(e)}\n"
                    f"请检查SMILES有效性或RDKit版本"
                )

            # 4.3 化学键（边）提取 - 此时mol已在所有分支定义，不会报未定义错误
            edges, edge_types = self._extract_mol_edges(mol, max_num_atoms)

            # 4.4 存储分子图结构
            mol_graphs.append({
                "nodes": list(range(max_num_atoms)),
                "edges": edges,
                "edge_types": edge_types,
                "features": atom_features.astype(np.float32),
                "label": labels[mol_idx],
                "valid_mask": valid_mask,
                "smiles": smiles,
                "num_nodes": max_num_atoms,
                "num_edges": len(edges)
            })

        # 5. 数据集分割（增强：调用修改后的骨架分割逻辑，分两步：train/rest → val/test）
        print(f"[ESOL分割] 配置：分割方式={self.split_type} | 平衡分割={self.split_balanced} | "
              f"种子={self.split_seed} | 比例={self.train_ratio}:{self.val_ratio}:{self.test_ratio}")

        # 第一步：分割训练集和剩余集（剩余集包含验证+测试）
        train_graphs, rest_graphs = self._split_graph_data(
            graphs=mol_graphs,
            target_split="train_rest",  # 标识当前是train/rest分割
            total_ratio=(self.train_ratio, self.val_ratio + self.test_ratio)
        )

        # 第二步：分割剩余集为验证集和测试集
        val_graphs, test_graphs = self._split_graph_data(
            graphs=rest_graphs,
            target_split="val_test",  # 标识当前是val/test分割
            total_ratio=(self.val_ratio / (self.val_ratio + self.test_ratio),
                         self.test_ratio / (self.val_ratio + self.test_ratio))
        )

        # 6. 分子属性标准化（如需要）
        if self.normalize_mol_props:
            train_mol_props = []
            for graph in train_graphs:
                mol = Chem.MolFromSmiles(graph["smiles"])
                train_mol_props.append([Descriptors.MolWt(mol)])  # 以分子量为例
            self.mol_prop_scaler.fit(np.array(train_mol_props))
            print(f"[ESOL标准化] 分子属性标准化完成（基于训练集分子量拟合）")

        # 7. 特征归一化（仅训练集拟合）
        if self.normalize_features:
            train_graphs, val_graphs, test_graphs = self._normalize_graph_features(
                train_graphs, val_graphs, test_graphs
            )

        # 8. 转换为PyTorch张量
        self.train_graphs = self._graphs_to_torch(train_graphs)
        self.val_graphs = self._graphs_to_torch(val_graphs)
        self.test_graphs = self._graphs_to_torch(test_graphs)
        self.graph_label_dim = 1
        self.num_features = self.train_graphs[0]["features"].shape[1]

        # 9. 最终分割校验（确保无骨架重叠）
        def get_valid_scaffolds(graph_list):
            """获取有效骨架集合（排除无效标识）"""
            scaffolds = set()
            for g in graph_list:
                s = self._get_murcko_scaffold(g["smiles"])
                if not any(prefix in s for prefix in ["invalid_", "parse_error_", "scaffold_error_"]):
                    scaffolds.add(s)
            return scaffolds

        train_scaffolds = get_valid_scaffolds(train_graphs)
        val_scaffolds = get_valid_scaffolds(val_graphs)
        test_scaffolds = get_valid_scaffolds(test_graphs)

        # 关键修改：仅当split-type为scaffold时，才校验骨架无重叠
        if self.split_type == "scaffold":
            # 断言无重叠（严格校验）
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0, "训练集与验证集存在骨架重叠！"
            assert len(train_scaffolds.intersection(test_scaffolds)) == 0, "训练集与测试集存在骨架重叠！"
            assert len(val_scaffolds.intersection(test_scaffolds)) == 0, "验证集与测试集存在骨架重叠！"
            print(f"[分割校验] 已完成骨架无重叠校验（有效骨架：训练{len(train_scaffolds)} | 验证{len(val_scaffolds)} | 测试{len(test_scaffolds)}）")
        else:
            # random分割时跳过校验，打印说明日志
            print(f"[分割校验] 因split-type={self.split_type}，跳过骨架重叠校验（有效骨架统计：训练{len(train_scaffolds)} | 验证{len(val_scaffolds)} | 测试{len(test_scaffolds)}）")

    # -------------------------- 原有工具方法（增强：添加类型校验） --------------------------
    def _extract_rdkit_atom_feats(self, mol, max_num_atoms):
        """提取RDKit原子特征（增强：添加原子数校验）"""
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_num_atoms:
            raise ValueError(
                f"分子原子数（{num_atoms}）超过最大限制（{max_num_atoms}），"
                f"请检查ESOL数据的max_num_atoms配置"
            )
        rdkit_feat_dim = 4  # 原子序数、形式电荷、价电子数、芳香性
        atom_feats = np.zeros((max_num_atoms, rdkit_feat_dim), dtype=np.float32)
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_feats[atom_idx] = [
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                atom.GetTotalValence(),
                1.0 if atom.GetIsAromatic() else 0.0
            ]
        return atom_feats

    def _extract_mol_edges(self, mol, max_num_atoms):
        """提取分子化学键（增强：过滤无效原子索引）"""
        edges = []
        edge_types = []
        bond_type_map = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            # 过滤超出max_num_atoms的索引（避免填充原子干扰）
            if 0 <= u < max_num_atoms and 0 <= v < max_num_atoms:
                edges.append([u, v])
                edge_types.append(bond_type_map[bond.GetBondType()])
        return np.array(edges, dtype=np.int64), np.array(edge_types, dtype=np.int64)

    # -------------------------- 核心修改：骨架分割逻辑（对齐之前的方法） --------------------------
    def _split_graph_data(self, graphs, target_split, total_ratio):
        """
        核心分割函数（修改版）：实现严格骨架分割
        :param graphs: 待分割的分子图列表
        :param target_split: 分割目标（train_rest=train/剩余，val_test=val/test）
        :param total_ratio: 分割比例（如train_rest时为[train_ratio, rest_ratio]）
        :return: 分割后的两个子集（如train和rest，或val和test）
        """
        # 1. 非分子任务或随机分割：沿用原有逻辑
        if not self.is_mol_task or self.split_type == "random":
            subset1, subset2 = train_test_split(
                graphs,
                test_size=total_ratio[1],  # 第二个比例是subset2的占比
                random_state=self.split_seed,
                shuffle=True
            )
            print(f"[随机分割] {target_split}：subset1={len(subset1)} | subset2={len(subset2)}")
            return subset1, subset2

        # 2. 分子任务+骨架分割：核心逻辑（对齐之前的方法）
        elif self.is_mol_task and self.split_type == "scaffold":
            # 2.1 按骨架分组（同一骨架的分子归为一组）
            scaffold_groups = {}
            invalid_count = 0  # 统计无效分子数
            for graph in graphs:
                scaffold = self._get_murcko_scaffold(graph["smiles"])
                # 统计无效骨架（便于调试）
                if any(prefix in scaffold for prefix in ["invalid_", "parse_error_", "scaffold_error_"]):
                    invalid_count += 1
                # 加入分组（键=骨架，值=分子图列表）
                if scaffold not in scaffold_groups:
                    scaffold_groups[scaffold] = []
                scaffold_groups[scaffold].append(graph)

            # 分组统计日志
            groups_list = list(scaffold_groups.values())
            group_sizes = [len(g) for g in groups_list]
            print(f"[骨架分组] {target_split}：总分组数={len(groups_list)} | 无效分子数={invalid_count} | "
                  f"最大组大小={max(group_sizes) if group_sizes else 0} | 平均组大小={np.mean(group_sizes):.1f}")

            # 2.2 平衡分组处理（避免大组集中在某一子集）
            if self.split_balanced and len(groups_list) > 0:
                # 计算大组阈值：参考之前的方法，用subset2的1/2作为阈值
                total_mols = len(graphs)
                subset2_size = int(total_mols * total_ratio[1])
                big_group_thresh = subset2_size / 2  # 超过该阈值的视为大组

                # 拆分大组和小组
                big_groups = [g for g in groups_list if len(g) > big_group_thresh]
                small_groups = [g for g in groups_list if len(g) <= big_group_thresh]

                # 分别打乱（确保随机性，避免大组顺序影响）
                random.seed(self.split_seed)
                random.shuffle(big_groups)
                random.shuffle(small_groups)

                # 合并分组（大组在前，优先分配到subset1，避免大组集中在subset2）
                groups_list = big_groups + small_groups
                print(
                    f"[平衡处理] {target_split}：大组数量={len(big_groups)} | 小组数量={len(small_groups)} | 阈值={big_group_thresh:.1f}")
            else:
                # 非平衡模式：按组大小降序排序（优先分配大组，保证比例更准确）
                groups_list = sorted(groups_list, key=lambda x: len(x), reverse=True)
                print(f"[排序处理] {target_split}：按组大小降序排列（共{len(groups_list)}组）")

            # 2.3 按比例分配分组（核心：同一骨架不跨集）
            subset1 = []  # 如train/val
            subset2 = []  # 如rest/test
            subset1_cutoff = int(len(graphs) * total_ratio[0])  # subset1的数量上限

            for group in groups_list:
                # 若当前subset1加入该组后不超过上限：加入subset1
                if len(subset1) + len(group) <= subset1_cutoff:
                    subset1.extend(group)
                # 否则：加入subset2
                else:
                    subset2.extend(group)

            # 2.4 分割结果校验与日志
            print(f"[分割结果] {target_split}：subset1={len(subset1)}（目标{subset1_cutoff}） | subset2={len(subset2)}")

            # 检查骨架重叠（仅有效骨架）
            def get_valid_scaffolds(subset):
                scaffolds = set()
                for g in subset:
                    s = self._get_murcko_scaffold(g["smiles"])
                    if not any(prefix in s for prefix in ["invalid_", "parse_error_", "scaffold_error_"]):
                        scaffolds.add(s)
                return scaffolds

            subset1_scaffolds = get_valid_scaffolds(subset1)
            subset2_scaffolds = get_valid_scaffolds(subset2)
            overlap = subset1_scaffolds.intersection(subset2_scaffolds)
            if overlap:
                raise ValueError(f"[分割错误] {target_split}存在有效骨架重叠：{len(overlap)}个骨架跨集！")
            else:
                print(
                    f"[分割校验] {target_split}：有效骨架无重叠（subset1={len(subset1_scaffolds)}个 | subset2={len(subset2_scaffolds)}个）")

            return subset1, subset2

        # 3. 无效分割方式
        else:
            raise ValueError(
                f"不支持的分割方式：{self.split_type}\n"
                f"分子任务仅支持 'random'（随机分割）或 'scaffold'（骨架分割）"
            )

    # -------------------------- 原有任务加载逻辑（保持不变，添加日志） --------------------------
    def _load_node_task_data(self):
        print(f"[节点任务] 从 {self.node_data_folder} 加载 {self.dataset_name} 数据...")
        if self.dataset_name.lower() == "cora":
            self._load_cora()
        elif self.dataset_name.lower() == "pubmed":
            self._load_pubmed()
        elif self.dataset_name.lower() == "dblp":
            self._load_dblp()
        else:
            raise NotImplementedError(f"节点任务数据集 {self.dataset_name} 未实现加载逻辑")

    def _load_cora(self):
        content_path = os.path.join(self.node_data_folder, "Cora.content")
        cites_path = os.path.join(self.node_data_folder, "Cora.cites")
        self._check_node_data_files([content_path, cites_path])

        node_features, node_labels, node_ids, label_map = [], [], [], {}
        current_label_id = 0
        with open(content_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 1435:
                    raise ValueError(f"Cora.content格式错误：每行应含1435个元素，当前行含{len(parts)}个")
                node_id = parts[0]
                features = list(map(int, parts[1:-1]))
                label = parts[-1]
                if label not in label_map:
                    label_map[label] = current_label_id
                    current_label_id += 1
                node_ids.append(node_id)
                node_features.append(features)
                node_labels.append(label_map[label])

        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        self.num_classes = len(label_map)
        self.features = np.array(node_features, dtype=np.float32)
        self.labels = np.array(node_labels, dtype=np.int64)
        self.adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        with open(cites_path, 'r', encoding='utf-8') as f:
            for line in f:
                cited_id, citing_id = line.strip().split()
                if cited_id in node_id_to_idx and citing_id in node_id_to_idx:
                    i = node_id_to_idx[cited_id]
                    j = node_id_to_idx[citing_id]
                    self.adj[i, j] = 1.0
                    self.adj[j, i] = 1.0

        print(f"[Cora加载完成] {self.num_nodes}节点 | {self.features.shape[1]}维特征 | {self.num_classes}类别")

    def _load_pubmed(self):
        content_path = os.path.join(self.node_data_folder, "Pubmed.content")
        cites_path = os.path.join(self.node_data_folder, "Pubmed.cites")
        self._check_node_data_files([content_path, cites_path])

        node_features, node_labels, node_ids, label_map = [], [], [], {}
        current_label_id = 0
        with open(content_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 502:
                    raise ValueError(f"Pubmed.content格式错误：每行应含502个元素，当前行含{len(parts)}个")
                node_id = parts[0]
                features = list(map(int, parts[1:-1]))
                label = parts[-1]
                if label not in label_map:
                    label_map[label] = current_label_id
                    current_label_id += 1
                node_ids.append(node_id)
                node_features.append(features)
                node_labels.append(label_map[label])

        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        self.num_classes = len(label_map)
        self.features = np.array(node_features, dtype=np.float32)
        self.labels = np.array(node_labels, dtype=np.int64)
        self.adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        with open(cites_path, 'r', encoding='utf-8') as f:
            for line in f:
                cited_id, citing_id = line.strip().split()
                if cited_id in node_id_to_idx and citing_id in node_id_to_idx:
                    i = node_id_to_idx[cited_id]
                    j = node_id_to_idx[citing_id]
                    self.adj[i, j] = 1.0
                    self.adj[j, i] = 1.0

        print(f"[Pubmed加载完成] {self.num_nodes}节点 | {self.features.shape[1]}维特征 | {self.num_classes}类别")

    def _load_dblp(self):
        content_path = os.path.join(self.node_data_folder, "DBLP_authors.content")
        cites_path = os.path.join(self.node_data_folder, "DBLP_authors.cites")
        self._check_node_data_files([content_path, cites_path])

        node_features, node_labels, node_ids = [], [], []
        label_map = {"database": 0, "data_mining": 1, "machine_learning": 2, "information_retrieval": 3}
        with open(content_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                node_id = parts[0]
                features = list(map(float, parts[1:-1]))
                label = parts[-1]
                if label not in label_map:
                    raise ValueError(f"DBLP标签错误：未知研究领域 {label}")
                node_ids.append(node_id)
                node_features.append(features)
                node_labels.append(label_map[label])

        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        self.num_classes = 4
        self.features = np.array(node_features, dtype=np.float32)
        self.labels = np.array(node_labels, dtype=np.int64)
        self.adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        with open(cites_path, 'r', encoding='utf-8') as f:
            for line in f:
                author1_id, author2_id = line.strip().split()
                if author1_id in node_id_to_idx and author2_id in node_id_to_idx:
                    i = node_id_to_idx[author1_id]
                    j = node_id_to_idx[author2_id]
                    self.adj[i, j] = 1.0
                    self.adj[j, i] = 1.0

        print(f"[DBLP加载完成] {self.num_nodes}作者节点 | {self.features.shape[1]}维特征 | {self.num_classes}研究领域")

    def _load_graph_task_data(self):
        pkl_filename = "graph_classification_complete.pkl" if self.dataset_name == "graph_classification" else "graph_regression_complete.pkl"
        pkl_path = os.path.join(self.graph_data_path, pkl_filename)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"图任务文件缺失：{pkl_path}\n请放入 {self.graph_data_path} 目录")

        print(f"[图任务] 从 {pkl_path} 加载 {self.dataset_name} 数据...")
        with open(pkl_path, 'rb') as f:
            raw_graphs = pickle.load(f)

        # 校验图数据格式
        required_keys = ["nodes", "edges", "edge_types", "features", "label"]
        for idx, graph in enumerate(raw_graphs):
            if not all(key in graph for key in required_keys):
                raise ValueError(f"第{idx + 1}个图缺失键：{set(required_keys) - set(graph.keys())}")
            if self.num_features is None:
                self.num_features = graph["features"].shape[1] if isinstance(graph["features"], np.ndarray) else len(
                    graph["features"][0])
            else:
                current_dim = graph["features"].shape[1] if isinstance(graph["features"], np.ndarray) else len(
                    graph["features"][0])
                if current_dim != self.num_features:
                    raise ValueError(f"图{idx + 1}特征维度错误：应为{self.num_features}，实际{current_dim}")

        # 分割数据（非分子任务，沿用原有逻辑）
        train_graphs, rest_graphs = train_test_split(
            raw_graphs, test_size=1 - self.train_ratio, random_state=self.split_seed
        )
        val_graphs, test_graphs = train_test_split(
            rest_graphs, test_size=1 - (self.val_ratio / (self.val_ratio + self.test_ratio)),
            random_state=self.split_seed
        )

        # 特征归一化
        if self.normalize_features:
            train_graphs, val_graphs, test_graphs = self._normalize_graph_features(train_graphs, val_graphs,
                                                                                   test_graphs)

        # 转换为张量
        self.train_graphs = self._graphs_to_torch(train_graphs)
        self.val_graphs = self._graphs_to_torch(val_graphs)
        self.test_graphs = self._graphs_to_torch(test_graphs)
        self.graph_label_dim = len(
            set(g["label"] for g in raw_graphs)) if self.dataset_name == "graph_classification" else 1

        print(f"[图任务加载完成] \n"
              f"  训练：{len(self.train_graphs)}个图 | 验证：{len(self.val_graphs)}个图 | 测试：{len(self.test_graphs)}个图\n"
              f"  节点特征维度：{self.num_features} | 标签维度：{self.graph_label_dim}")

    # -------------------------- 原有工具方法（保持不变） --------------------------
    def _normalize_graph_features(self, train_graphs, val_graphs, test_graphs):
        train_all_features = []
        for graph in train_graphs:
            features = graph["features"]
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            train_all_features.extend(features)
        train_all_features = np.array(train_all_features, dtype=np.float32)

        self.scaler = StandardScaler()
        self.scaler.fit(train_all_features)

        def normalize_single_graph(graph):
            features = graph["features"]
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            graph["features"] = self.scaler.transform(features)
            return graph

        train_graphs = [normalize_single_graph(g) for g in train_graphs]
        val_graphs = [normalize_single_graph(g) for g in val_graphs]
        test_graphs = [normalize_single_graph(g) for g in test_graphs]
        print("[特征归一化] 图任务特征归一化完成（仅用训练集统计量）")
        return train_graphs, val_graphs, test_graphs

    def _graphs_to_torch(self, graphs):
        torch_graphs = []
        for graph in graphs:
            features = torch.FloatTensor(graph["features"]).to(self.device)
            edges = torch.LongTensor(graph["edges"]).to(self.device) if len(graph["edges"]) > 0 else torch.empty((0, 2),
                                                                                                                 dtype=torch.long).to(
                self.device)
            edge_types = torch.LongTensor(graph["edge_types"]).to(self.device) if len(
                graph["edge_types"]) > 0 else torch.empty(0, dtype=torch.long).to(self.device)
            label_dtype = torch.long if (
                        self.dataset_name == "graph_classification" or not self.is_mol_task) else torch.float32
            label = torch.tensor(graph["label"], dtype=label_dtype).to(self.device)
            valid_mask = torch.BoolTensor(graph.get("valid_mask", np.ones(features.shape[0], dtype=bool))).to(
                self.device)

            torch_graphs.append({
                "nodes": graph["nodes"],
                "edges": edges,
                "edge_types": edge_types,
                "features": features,
                "label": label,
                "num_nodes": features.shape[0],
                "num_edges": edges.shape[0],
                "valid_mask": valid_mask,
                "smiles": graph.get("smiles", "")
            })
        return torch_graphs

    def _check_node_data_files(self, file_paths):
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"节点文件缺失：{path}\n请放入 {self.node_data_folder} 目录")

    def _split_train_val_test(self):
        if self.is_graph_task or self.is_mol_task:
            return
        indices = np.arange(self.num_nodes)
        train_indices, rest_indices = train_test_split(indices, test_size=1 - self.train_ratio,
                                                       random_state=self.split_seed, stratify=self.labels)
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_indices, test_indices = train_test_split(rest_indices, test_size=1 - val_ratio_adjusted,
                                                     random_state=self.split_seed, stratify=self.labels[rest_indices])

        self.train_mask = np.zeros(self.num_nodes, dtype=bool)
        self.val_mask = np.zeros(self.num_nodes, dtype=bool)
        self.test_mask = np.zeros(self.num_nodes, dtype=bool)
        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True

        print(f"[节点分割] 完成：训练{sum(self.train_mask)}节点 | 验证{sum(self.val_mask)}节点 | 测试{sum(self.test_mask)}节点")

    def _normalize_features(self):
        if not self.normalize_features or self.is_graph_task or self.is_mol_task:
            return
        self.scaler = StandardScaler()
        self.features[self.train_mask] = self.scaler.fit_transform(self.features[self.train_mask])
        self.features[self.val_mask] = self.scaler.transform(self.features[self.val_mask])
        self.features[self.test_mask] = self.scaler.transform(self.features[self.test_mask])
        print("[节点归一化] 节点任务特征归一化完成")

    def _cluster_based_on_train_features(self):
        if self.is_graph_task or self.is_mol_task:
            return
        train_features = self.features[self.train_mask]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.split_seed, n_init=10)
        train_clusters = kmeans.fit_predict(train_features)
        val_clusters = kmeans.predict(self.features[self.val_mask])
        test_clusters = kmeans.predict(self.features[self.test_mask])

        self.clusters = np.zeros(self.num_nodes, dtype=np.int64)
        self.clusters[self.train_mask] = train_clusters
        self.clusters[self.val_mask] = val_clusters
        self.clusters[self.test_mask] = test_clusters
        print(f"[节点聚类] K-means聚类完成：{self.n_clusters}个簇")

    def _community_based_on_train_adj(self):
        if self.is_graph_task or self.is_mol_task:
            return
        train_indices = np.where(self.train_mask)[0]
        train_idx_to_sub_idx = {idx: i for i, idx in enumerate(train_indices)}
        train_adj_sub = self.adj[np.ix_(train_indices, train_indices)]

        train_graph = nx.Graph()
        train_graph.add_nodes_from(range(len(train_indices)))
        for i in range(len(train_indices)):
            for j in range(i + 1, len(train_indices)):
                if train_adj_sub[i, j] > 0:
                    train_graph.add_edge(i, j)

        if train_graph.number_of_edges() == 0:
            train_communities = np.arange(len(train_indices), dtype=np.int64)
        else:
            partition = community_louvain.best_partition(train_graph, random_state=self.split_seed)
            train_communities = np.array([partition[i] for i in range(len(train_indices))])

        self.attributes = np.zeros_like(self.features[:, :1433])
        train_attributes = self.features[self.train_mask, :1433]
        self.attr_scaler = StandardScaler()
        self.attributes[self.train_mask] = self.attr_scaler.fit_transform(train_attributes)
        self.attributes[self.val_mask] = self.attr_scaler.transform(self.features[self.val_mask, :1433])
        self.attributes[self.test_mask] = self.attr_scaler.transform(self.features[self.test_mask, :1433])

        self.communities = np.full(self.num_nodes, -1, dtype=np.int64)
        self.communities[train_indices] = train_communities
        next_community_id = max(train_communities) + 1 if len(train_communities) > 0 else 0

        for idx in np.where(self.val_mask)[0]:
            connected_train_idx = train_indices[np.where(self.adj[idx, train_indices] > 0)[0]]
            if len(connected_train_idx) > 0:
                self.communities[idx] = np.bincount(self.communities[connected_train_idx]).argmax()
            else:
                self.communities[idx] = next_community_id
                next_community_id += 1

        for idx in np.where(self.test_mask)[0]:
            connected_train_idx = train_indices[np.where(self.adj[idx, train_indices] > 0)[0]]
            if len(connected_train_idx) > 0:
                self.communities[idx] = np.bincount(self.communities[connected_train_idx]).argmax()
            else:
                self.communities[idx] = next_community_id
                next_community_id += 1

        print(f"[节点社区] Louvain社区检测完成：共{next_community_id}个社区")

    def _convert_to_torch(self):
        if self.is_graph_task or self.is_mol_task:
            return
        self.features = torch.FloatTensor(self.features).to(self.device)
        self.labels = torch.LongTensor(self.labels).to(self.device)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        self.attributes = torch.FloatTensor(self.attributes).to(self.device)
        self.clusters = torch.LongTensor(self.clusters).to(self.device)
        self.communities = torch.LongTensor(self.communities).to(self.device)
        self.train_mask = torch.BoolTensor(self.train_mask).to(self.device)
        self.val_mask = torch.BoolTensor(self.val_mask).to(self.device)
        self.test_mask = torch.BoolTensor(self.test_mask).to(self.device)

    def run(self):
        print(f"\n[预处理启动] 数据集：{self.dataset_name} | 任务类型：{self.task_type} | 设备：{self.device}")
        self._load_raw_data()

        if not self.is_graph_task and not self.is_mol_task:
            self._split_train_val_test()
            self._normalize_features()
            self._cluster_based_on_train_features()
            self._community_based_on_train_adj()
            self._convert_to_torch()
        else:
            print(f"[预处理跳过] 图任务/分子任务加载时已完成数据转换，无需额外步骤")

        print(f"\n{self.dataset_name} 预处理完成\n")

        # 返回结果（适配任务类型）
        if self.is_mol_task:
            return {
                'train': self.train_graphs,
                'val': self.val_graphs,
                'test': self.test_graphs,
                'num_features': self.num_features,
                'graph_label_dim': self.graph_label_dim,
                'task_type': self.task_type,
                'is_mol_task': True,
                'split_type': self.split_type,
                'split_balanced': self.split_balanced
            }
        elif self.is_graph_task:
            return {
                'train': self.train_graphs,
                'val': self.val_graphs,
                'test': self.test_graphs,
                'num_features': self.num_features,
                'graph_label_dim': self.graph_label_dim,
                'task_type': self.task_type
            }
        else:
            return {
                'features': self.features,
                'labels': self.labels,
                'adj': self.adj,
                'attributes': self.attributes,
                'clusters': self.clusters,
                'communities': self.communities,
                'train_mask': self.train_mask,
                'val_mask': self.val_mask,
                'test_mask': self.test_mask,
                'num_nodes': self.num_nodes,
                'num_classes': self.num_classes,
                'task_type': self.task_type
            }
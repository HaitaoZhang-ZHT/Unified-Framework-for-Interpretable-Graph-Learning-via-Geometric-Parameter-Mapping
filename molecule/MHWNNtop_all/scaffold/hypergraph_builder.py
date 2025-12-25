import numpy as np
import torch
import logging
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_classif

# 分子任务必需：RDKit用于官能团识别
try:
    from rdkit import Chem
    from rdkit.Chem import MolFromSmarts, GetDistanceMatrix
except ImportError:
    raise ImportError("请安装RDKit以处理分子数据：pip install rdkit-pypi")


class HypergraphBuilder:
    """超图构建类，支持节点任务/图任务（含ESOL分子任务），新增氢键网络超边和稀疏矩阵支持"""

    def __init__(self, processed_data, task_type="node_classification",
                 # 节点任务参数
                 hyper_types=['neighbor', 'attribute', 'cluster', 'community'],
                 hyper_repeats=[1, 1, 1, 1],
                 phi_hop=3,
                 n_clusters=100,
                 # 图任务（含分子）专属参数
                 graph_hyper_types=["bond", "functional_group", "hydrogen_bond"],
                 graph_hyper_repeats=[1, 1, 1],
                 graph_edge_weight="similarity",
                 substructure_depth=2,
                 hyperedge_size_range=(2, 20),
                 # 氢键超边专属参数
                 hbond_distance_threshold=3.0,
                 hbond_angle_threshold=120.0,
                 # 稀疏矩阵支持参数
                 sparse_threshold=5000,  # 节点数超过此值启用稀疏矩阵
                 # 通用参数
                 epsilon=1e-5,
                 device="cpu"):
        self.processed_data = processed_data
        self.task_type = task_type
        self.epsilon = epsilon
        self.device = device
        self.hyperedge_size_range = hyperedge_size_range
        self.sparse_threshold = sparse_threshold  # 稀疏矩阵阈值

        # 节点任务参数初始化
        if self.task_type == "node_classification":
            self.hyper_types = hyper_types
            self.hyper_repeats = hyper_repeats
            self.phi_hop = phi_hop
            self.n_clusters = n_clusters
            # 节点任务必要键检查
            required_keys = ['features', 'adj', 'attributes', 'clusters', 'communities', 'labels']
            for key in required_keys:
                if key not in self.processed_data:
                    raise KeyError(f"节点任务的processed_data缺少必要键：{key}，请检查数据预处理")
            self.num_nodes = processed_data['features'].shape[0]
            self.hypergraph_snapshots = []
            if len(self.hyper_types) != len(self.hyper_repeats):
                raise ValueError(f"超边类型数量({len(self.hyper_types)})与重复次数({len(self.hyper_repeats)})不匹配")

        # 图任务参数初始化
        else:
            self.graph_hyper_types = graph_hyper_types
            self.graph_hyper_repeats = graph_hyper_repeats
            if len(self.graph_hyper_types) != len(self.graph_hyper_repeats):
                raise ValueError(
                    f"图任务超边类型数量({len(self.graph_hyper_types)})与重复次数({len(self.graph_hyper_repeats)})不匹配")

            # 图任务必要分割检查
            required_splits = ['train', 'val', 'test']
            for split in required_splits:
                if split not in self.processed_data:
                    raise KeyError(f"图任务的processed_data缺少必要的数据集分割：{split}\n"
                                   f"请检查DataPreprocessor是否正确生成了训练/验证/测试集")

            self.graphs = {
                'train': processed_data['train'],
                'val': processed_data['val'],
                'test': processed_data['test']
            }

            # 检查分割数据非空
            for split in required_splits:
                if not self.graphs[split]:
                    logging.warning(f"{split}集数据为空，可能导致后续处理失败")

            self.graph_hyper_data = {
                'train': [],
                'val': [],
                'test': []
            }
            self.graph_edge_weight = graph_edge_weight
            self.substructure_depth = substructure_depth

            # 提取所有边类型
            all_edge_types = set()
            for split in ['train', 'val', 'test']:
                for graph in self.graphs[split]:
                    if 'edge_types' not in graph:
                        raise KeyError(f"{split}集中的图缺少'edge_types'键，请检查数据预处理")
                    all_edge_types.update(graph['edge_types'].cpu().numpy())
            self.all_edge_types = list(all_edge_types)

            # 分子任务官能团和氢键配置
            self.is_mol_task = processed_data.get("is_mol_task", False)
            if self.is_mol_task:
                self.fg_smarts = self._get_default_fg_smarts()
                self.fg_type_map = {name: idx for idx, name in enumerate(self.fg_smarts.keys())}
                self.fg_weights = {
                    '羟基(-OH)': 0.8, '羧基(-COOH)': 0.9, '氨基(-NH2)': 0.7,
                    '羰基(C=O)': 0.5, '醚键(C-O-C)': 0.3, '酯基(-COO-)': 0.2,
                    '烷基(-CH3)': 0.1, '芳香环': 0.05
                }

                # 氢键参数
                self.hbond_distance_threshold = hbond_distance_threshold
                self.hbond_angle_threshold = hbond_angle_threshold
                self.hbond_type_map = {
                    'O-H...O': 0,
                    'O-H...N': 1,
                    'N-H...O': 2,
                    'N-H...N': 3
                }
                self.hbond_weights = {
                    0: 0.6,
                    1: 0.5,
                    2: 0.55,
                    3: 0.4
                }

                print(f"分子任务初始化：\n"
                      f"边类型：{self.all_edge_types}（共{len(self.all_edge_types)}种）\n"
                      f"官能团类型：{list(self.fg_smarts.keys())}（共{len(self.fg_smarts)}种）\n"
                      f"氢键类型：{list(self.hbond_type_map.keys())}（共{len(self.hbond_type_map)}种）\n"
                      f"超边类型配置：{self.graph_hyper_types}\n"
                      f"超边重复次数：{self.graph_hyper_repeats}")
            else:
                print(f"图任务初始化：\n"
                      f"边类型：{self.all_edge_types}（共{len(self.all_edge_types)}种）\n"
                      f"超边类型配置：{self.graph_hyper_types}\n"
                      f"超边重复次数：{self.graph_hyper_repeats}")

    def _get_default_fg_smarts(self):
        """影响溶解度的常见官能团SMARTS模式（RDKit可识别）"""
        return {
            '羟基(-OH)': '[OX2H]',
            '羧基(-COOH)': '[CX3](=O)[OX2H1]',
            '氨基(-NH2)': '[NX3H2]',
            '羰基(C=O)': '[CX3]=[OX1]',
            '醚键(C-O-C)': '[CX3]-[OX2]-[CX3]',
            '酯基(-COO-)': '[CX3](=O)[OX2][CX3]',
            '烷基(-CH3)': '[CX3H3]',
            '芳香环': '[c]1[c][c][c][c][c]1'
        }

    # 新增：稀疏矩阵转换方法
    def _dense_to_sparse(self, dense_matrix):
        """将稠密矩阵转换为稀疏格式 (edge_index, edge_attr)"""
        if isinstance(dense_matrix, torch.Tensor):
            dense_np = dense_matrix.cpu().numpy()
        else:
            dense_np = dense_matrix

        # 找到非零元素
        rows, cols = np.where(dense_np > 1e-5)
        values = dense_np[rows, cols]

        if len(rows) == 0:
            # 如果没有边，创建空的稀疏矩阵
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 1), dtype=torch.float, device=self.device)
        else:
            edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(values, dtype=torch.float, device=self.device).unsqueeze(-1)

        return (edge_index, edge_attr)

    # 节点任务超边构建方法（保持不变）
    def _hypergraph_neighbor_based(self):
        adj = self.processed_data['adj'].cpu().numpy()
        hyperedges = []
        adj_power = np.linalg.matrix_power(adj, self.phi_hop)
        for i in range(self.num_nodes):
            neighbors = np.where(adj_power[i] > 0)[0]
            if len(neighbors) > 0:
                hyperedge = np.unique(np.concatenate([[i], neighbors]))
                if self.hyperedge_size_range[0] <= len(hyperedge) <= self.hyperedge_size_range[1]:
                    hyperedges.append(hyperedge)
        return np.array(hyperedges, dtype=object)

    def _hypergraph_attribute_based(self):
        attributes = self.processed_data['attributes'].cpu().numpy()
        num_attributes = attributes.shape[1]
        hyperedges = []
        for attr_idx in range(num_attributes):
            nodes_with_attr = np.where(attributes[:, attr_idx] > 0)[0]
            if len(nodes_with_attr) >= self.hyperedge_size_range[0]:
                hyperedges.append(nodes_with_attr)
        return np.array(hyperedges, dtype=object)

    def _hypergraph_cluster_based(self):
        clusters = self.processed_data['clusters'].cpu().numpy()
        unique_clusters = np.unique(clusters)
        hyperedges = []
        for cluster_id in unique_clusters:
            cluster_nodes = np.where(clusters == cluster_id)[0]
            if len(cluster_nodes) >= self.hyperedge_size_range[0]:
                hyperedges.append(cluster_nodes)
        return np.array(hyperedges, dtype=object)

    def _hypergraph_community_based(self):
        communities = self.processed_data['communities'].cpu().numpy()
        unique_communities = np.unique(communities)
        hyperedges = []
        for community_id in unique_communities:
            community_nodes = np.where(communities == community_id)[0]
            if len(community_nodes) >= self.hyperedge_size_range[0]:
                hyperedges.append(community_nodes)
        return np.array(hyperedges, dtype=object)

    # 图任务超边构建 - 新增超边类型映射，包含氢键
    def _get_graph_hyperedge_builders(self):
        """获取超边类型与构建方法的映射，新增氢键超边构建器"""
        return {
            'bond': self._hypergraph_edge_type_based,
            'functional_group': self._hypergraph_functional_group_based,
            'substructure': self._hypergraph_substructure_based,
            'similarity': self._hypergraph_similarity_based,
            'hydrogen_bond': self._hypergraph_hydrogen_bond_based
        }

    def _hypergraph_edge_type_based(self, graph):
        """基于边类型构建超边（支持分子化学键强度权重）"""
        required_graph_keys = ['edges', 'edge_types', 'num_nodes', 'valid_mask']
        for key in required_graph_keys:
            if key not in graph:
                raise KeyError(f"图数据缺少必要键：{key}，请检查数据预处理")

        edges = graph['edges'].cpu().numpy()
        edge_types = graph['edge_types'].cpu().numpy()
        unique_types = np.unique(edge_types)
        num_nodes = graph['num_nodes']
        valid_mask = graph['valid_mask'].cpu().numpy()

        hyperedges_dict = {t: [] for t in unique_types}
        for idx, (u, v) in enumerate(edges):
            # 过滤无效原子（分子任务）
            if self.is_mol_task and not (valid_mask[u] and valid_mask[v]):
                continue
            # 过滤超边大小（2节点边）
            if not (self.hyperedge_size_range[0] <= 2 <= self.hyperedge_size_range[1]):
                continue
            t = edge_types[idx]
            hyperedges_dict[t].append(np.array([u, v], dtype=np.int64))

        # 过滤空超边
        hyperedges_by_type = {t: np.array(edges, dtype=object) for t, edges in hyperedges_dict.items() if
                              len(edges) > 0}
        return hyperedges_by_type

    def _hypergraph_functional_group_based(self, graph):
        """基于分子官能团构建超边（仅分子任务有效）"""
        if not self.is_mol_task:
            return {}

        if 'smiles' not in graph:
            logging.warning("图数据缺少'smiles'键，无法构建官能团超边")
            return {}

        smiles = graph['smiles']
        if not smiles:
            return {}
        num_nodes = graph['num_nodes']
        valid_mask = graph['valid_mask'].cpu().numpy()

        # 解析SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"SMILES {smiles} 无效，跳过官能团超边构建")
            return {}

        fg_hyperedges = {}
        for fg_name, fg_smarts in self.fg_smarts.items():
            fg_type = self.fg_type_map[fg_name]
            fg_pattern = MolFromSmarts(fg_smarts)
            matches = mol.GetSubstructMatches(fg_pattern)
            if not matches:
                continue

            # 过滤有效原子并构建超边
            for match in matches:
                valid_atoms = [atom_idx for atom_idx in match
                               if 0 <= atom_idx < num_nodes and valid_mask[atom_idx]]
                if self.hyperedge_size_range[0] <= len(valid_atoms) <= self.hyperedge_size_range[1]:
                    if fg_type not in fg_hyperedges:
                        fg_hyperedges[fg_type] = []
                    fg_hyperedges[fg_type].append(np.array(valid_atoms, dtype=np.int64))

        # 过滤空超边
        return {t: np.array(edges, dtype=object) for t, edges in fg_hyperedges.items() if len(edges) > 0}

    def _hypergraph_substructure_based(self, graph):
        """基于分子子结构构建超边（支持指定深度）"""
        if not self.is_mol_task or 'smiles' not in graph:
            return {}

        smiles = graph['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"SMILES {smiles} 无效，跳过子结构超边构建")
            return {}

        num_nodes = graph['num_nodes']
        valid_mask = graph['valid_mask'].cpu().numpy()
        substruct_hyperedges = []

        # 提取指定深度的子结构
        for depth in range(1, self.substructure_depth + 1):
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                if not valid_mask[atom_idx]:
                    continue
                substruct_atoms = self._bfs_substruct(mol, atom_idx, depth)
                valid_substruct = [idx for idx in substruct_atoms if valid_mask[idx]]
                if self.hyperedge_size_range[0] <= len(valid_substruct) <= self.hyperedge_size_range[1]:
                    substruct_hyperedges.append(np.array(valid_substruct, dtype=np.int64))

        return {'substructure': np.array(substruct_hyperedges, dtype=object)} if substruct_hyperedges else {}

    def _bfs_substruct(self, mol, start_idx, depth):
        visited = set()
        queue = [(start_idx, 0)]
        while queue:
            atom_idx, d = queue.pop(0)
            if atom_idx in visited or d > depth:
                continue
            visited.add(atom_idx)
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    queue.append((neighbor_idx, d + 1))
        return list(visited)

    def _hypergraph_similarity_based(self, graph):
        """基于节点特征相似度构建超边"""
        required_keys = ['features', 'num_nodes', 'valid_mask']
        for key in required_keys:
            if key not in graph:
                raise KeyError(f"图数据缺少必要键：{key}，无法构建相似度超边")

        features = graph['features'].cpu().numpy()
        num_nodes = graph['num_nodes']
        valid_mask = graph['valid_mask'].cpu().numpy()
        valid_nodes = np.where(valid_mask)[0]

        if len(valid_nodes) < self.hyperedge_size_range[0]:
            return {}

        # 计算相似度矩阵
        sim_matrix = cosine_similarity(features[valid_nodes])
        similarity_hyperedges = []

        # 为每个节点构建相似节点超边
        for i, node in enumerate(valid_nodes):
            sim_scores = sim_matrix[i]
            sorted_indices = np.argsort(sim_scores)[::-1]
            top_k = max(2, self.hyperedge_size_range[0])
            similar_nodes = valid_nodes[sorted_indices[1:top_k]]
            hyperedge = np.unique(np.concatenate([[node], similar_nodes]))
            if self.hyperedge_size_range[0] <= len(hyperedge) <= self.hyperedge_size_range[1]:
                similarity_hyperedges.append(hyperedge)

        return {'similarity': np.array(similarity_hyperedges, dtype=object)} if similarity_hyperedges else {}

    def _hypergraph_hydrogen_bond_based(self, graph):
        """基于分子内氢键网络构建超边（仅分子任务有效）"""
        if not self.is_mol_task or 'smiles' not in graph:
            return {}

        smiles = graph['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"SMILES {smiles} 无效，跳过氢键超边构建")
            return {}

        # 生成3D构象以计算原子间距离
        try:
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception as e:
            logging.warning(f"生成分子{smiles}的3D构象失败：{str(e)}，使用拓扑距离近似")

        num_nodes = graph['num_nodes']
        valid_mask = graph['valid_mask'].cpu().numpy()
        hbond_hyperedges = {}

        # 识别氢键供体和受体
        donors, acceptors = self._identify_hbond_donors_acceptors(mol)

        if not donors or not acceptors:
            return {}

        # 获取原子间距离矩阵
        try:
            dist_matrix = AllChem.Get3DDistanceMatrix(mol)
        except:
            dist_matrix = GetDistanceMatrix(mol)

        # 识别潜在氢键对
        hbond_pairs = []
        for (d_idx, d_type) in donors:
            for (a_idx, a_type) in acceptors:
                if d_idx == a_idx:
                    continue

                distance = dist_matrix[d_idx][a_idx] if d_idx < len(dist_matrix) and a_idx < len(
                    dist_matrix) else float('inf')
                if distance > self.hbond_distance_threshold:
                    continue

                if self._check_hbond_angle(mol, d_idx, a_idx):
                    hbond_type = f"{d_type}-H...{a_type}"
                    if hbond_type in self.hbond_type_map:
                        hbond_pairs.append((d_idx, a_idx, self.hbond_type_map[hbond_type]))

        if not hbond_pairs:
            return {}

        # 构建氢键网络超边
        hbond_graph = {i: [] for i in range(num_nodes)}
        for d_idx, a_idx, hb_type in hbond_pairs:
            hbond_graph[d_idx].append((a_idx, hb_type))
            hbond_graph[a_idx].append((d_idx, hb_type))

        # 使用连通分量识别氢键网络
        visited = set()
        for node in range(num_nodes):
            if node not in visited and node in hbond_graph and valid_mask[node]:
                component = []
                queue = [node]
                visited.add(node)

                while queue:
                    current = queue.pop(0)
                    component.append(current)

                    for neighbor, hb_type in hbond_graph.get(current, []):
                        if neighbor not in visited and valid_mask[neighbor]:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if self.hyperedge_size_range[0] <= len(component) <= self.hyperedge_size_range[1]:
                    component_hb_types = [hb_type for d, a, hb_type in hbond_pairs
                                          if d in component or a in component]
                    if component_hb_types:
                        main_hb_type = max(set(component_hb_types), key=component_hb_types.count)
                        if main_hb_type not in hbond_hyperedges:
                            hbond_hyperedges[main_hb_type] = []
                        hbond_hyperedges[main_hb_type].append(np.array(component, dtype=np.int64))

        return {t: np.array(edges, dtype=object) for t, edges in hbond_hyperedges.items() if len(edges) > 0}

    def _identify_hbond_donors_acceptors(self, mol):
        """识别分子中的氢键供体（D-H）和受体（A）"""
        donors = []
        acceptors = []

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()

            if atom_symbol == 'H':
                neighbors = atom.GetNeighbors()
                if neighbors:
                    neighbor = neighbors[0]
                    neighbor_symbol = neighbor.GetSymbol()
                    if neighbor_symbol in ['O', 'N']:
                        donors.append((neighbor.GetIdx(), neighbor_symbol))

            elif atom_symbol in ['O', 'N']:
                if self._has_lone_pairs(atom):
                    acceptors.append((atom_idx, atom_symbol))

        return donors, acceptors

    def _has_lone_pairs(self, atom):
        """判断原子是否有孤对电子（基于元素类型和价态）"""
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        valence = atom.GetTotalValence()

        if symbol == 'O':
            return valence < 4 or formal_charge != 0
        elif symbol == 'N':
            return valence < 5 or formal_charge != 0
        return False

    def _check_hbond_angle(self, mol, donor_idx, acceptor_idx):
        """简化版氢键键角检查"""
        donor_atom = mol.GetAtomWithIdx(donor_idx)
        h_atoms = [n.GetIdx() for n in donor_atom.GetNeighbors() if n.GetSymbol() == 'H']
        return bool(h_atoms)

    # 超边权重计算（扩展氢键权重）
    def _compute_hyperedge_weights(self, hyperedges, hyper_type, task_data=None):
        """计算超边权重（新增氢键权重计算）"""
        weights = []
        if self.task_type == "node_classification":
            features = self.processed_data['features'].cpu().numpy()
            labels = self.processed_data['labels'].cpu().numpy()
            attributes = self.processed_data['attributes'].cpu().numpy()
        else:
            if task_data is None:
                raise ValueError("图任务计算权重时必须提供task_data")
            features = task_data['features'].cpu().numpy()
            attributes = None
            labels = None
            smiles = task_data.get('smiles', '')
            valid_mask = task_data.get('valid_mask', torch.ones(features.shape[0], dtype=torch.bool)).cpu().numpy()

        for nodes in hyperedges:
            if len(nodes) == 0:
                weights.append(0.1)
                continue

            # 节点任务权重计算
            if self.task_type == "node_classification":
                if hyper_type == 'neighbor':
                    w = 1.0 / np.log1p(len(nodes))
                elif hyper_type == 'attribute':
                    attr_idx = np.where(attributes[nodes[0]] > 0)[0][0] if len(nodes) > 0 else 0
                    mi = mutual_info_classif(self.processed_data['attributes'][:, [attr_idx]].cpu().numpy(), labels,
                                             random_state=42)[0]
                    w = mi if mi > 0 else 0.1
                elif hyper_type == 'cluster':
                    sim = cosine_similarity(features[nodes])
                    w = np.mean(sim) if len(nodes) > 1 else 0.1
                elif hyper_type == 'community':
                    w = 1.0 / np.sqrt(len(nodes))
                else:
                    w = 1.0

            # 图任务权重计算
            else:
                if self.is_mol_task and hyper_type in self.hbond_type_map.values():
                    base_w = self.hbond_weights.get(hyper_type, 0.4)
                    w = base_w / np.log1p(len(nodes))

                elif hyper_type in self.all_edge_types:
                    if self.graph_edge_weight == "bond_strength" and self.is_mol_task:
                        bond_strength_map = {0: 1.0, 1: 1.5, 2: 2.0, 3: 1.2}
                        w = bond_strength_map.get(hyper_type, 1.0)
                    elif self.graph_edge_weight == "similarity":
                        if len(nodes) == 2:
                            u, v = nodes
                            if u < features.shape[0] and v < features.shape[0] and valid_mask[u] and valid_mask[v]:
                                sim = cosine_similarity(features[[u]], features[[v]])[0][0]
                                w = sim if sim > 0 else 0.1
                            else:
                                w = 0.1
                        else:
                            w = 1.0 / np.log1p(len(nodes))
                    else:
                        w = 1.0

                elif hyper_type in self.fg_type_map.values() and self.is_mol_task:
                    fg_name = [name for name, idx in self.fg_type_map.items() if idx == hyper_type][0]
                    base_w = self.fg_weights.get(fg_name, 0.3)
                    w = base_w / np.log1p(len(nodes))

                else:
                    w = 1.0

            weights.append(w)
        return np.array(weights)

    # 通用方法（添加稀疏矩阵支持）
    def _build_hypergraph_matrix(self, hyperedges, num_nodes):
        """构建超图邻接矩阵（稀疏矩阵）"""
        num_hyperedges = len(hyperedges)
        row, col, data = [], [], []
        for e_idx, nodes in enumerate(hyperedges):
            valid_nodes = [n for n in nodes if 0 <= n < num_nodes]
            if not valid_nodes:
                continue
            row.extend(valid_nodes)
            col.extend([e_idx] * len(valid_nodes))
            data.extend([1.0] * len(valid_nodes))
        return csr_matrix((data, (row, col)), shape=(num_nodes, num_hyperedges))

    def _compute_theta_matrix(self, H, W, num_nodes):
        """计算Theta矩阵（超图拉普拉斯相关矩阵）"""
        Dv = np.array(H @ W @ np.ones((W.shape[0], 1))).flatten()
        Dv_sqrt_inv = np.diag(1.0 / np.sqrt(Dv + self.epsilon))
        De = np.array(H.T @ np.ones((num_nodes, 1))).flatten()
        De_inv = np.diag(1.0 / (De + self.epsilon))
        Theta = Dv_sqrt_inv @ H.toarray() @ W.toarray() @ De_inv @ H.toarray().T @ Dv_sqrt_inv
        Theta[np.isinf(Theta)] = 0
        Theta[np.isnan(Theta)] = 0
        Theta += self.epsilon * np.eye(num_nodes)
        return torch.FloatTensor(Theta).to(self.device)

    def _compute_wavelets(self, Theta):
        """计算小波矩阵及其逆矩阵"""
        num_nodes = Theta.shape[0]
        Laplacian = torch.eye(num_nodes, device=self.device) - Theta
        lambda_, fourier_v = torch.linalg.eigh(Laplacian)
        s = 1.0
        wavelets = fourier_v @ torch.diag(torch.exp(-lambda_ * s)) @ fourier_v.T
        wavelets_inv = fourier_v @ torch.diag(torch.exp(lambda_ * s)) @ fourier_v.T
        wavelets[wavelets.abs() < 1e-5] = 0
        wavelets_inv[wavelets_inv.abs() < 1e-5] = 0
        return wavelets, wavelets_inv

    # 节点任务快照构建（添加稀疏矩阵支持）
    def _build_node_snapshots(self):
        hyperedge_builders = {
            'neighbor': self._hypergraph_neighbor_based,
            'attribute': self._hypergraph_attribute_based,
            'cluster': self._hypergraph_cluster_based,
            'community': self._hypergraph_community_based
        }

        # 检查是否启用稀疏矩阵
        use_sparse = self.num_nodes > self.sparse_threshold
        if use_sparse:
            print(f"[稀疏矩阵] 节点数({self.num_nodes}) > 阈值({self.sparse_threshold})，启用稀疏矩阵")

        for hyper_type, repeat in zip(self.hyper_types, self.hyper_repeats):
            for r in range(repeat):
                print(f"构建节点任务超边：{hyper_type}（第{r + 1}/{repeat}次）")
                hyperedges = hyperedge_builders[hyper_type]()
                if len(hyperedges) == 0:
                    print(f"警告：{hyper_type}类型未生成超边，跳过")
                    continue
                weights = self._compute_hyperedge_weights(hyperedges, hyper_type)
                H = self._build_hypergraph_matrix(hyperedges, self.num_nodes)
                W = csr_matrix(np.diag(weights))
                Theta = self._compute_theta_matrix(H, W, self.num_nodes)
                wavelets, wavelets_inv = self._compute_wavelets(Theta)

                # 新增：构建稀疏邻接矩阵
                sparse_adj = None
                if use_sparse:
                    sparse_adj = self._dense_to_sparse(Theta)

                snapshot = {
                    'type': hyper_type,
                    'H': H,
                    'W': W,
                    'Theta': Theta,
                    'wavelets': wavelets,
                    'wavelets_inv': wavelets_inv,
                    'num_hyperedges': len(hyperedges)
                }

                # 添加稀疏矩阵支持
                if sparse_adj is not None:
                    snapshot['sparse_adj'] = sparse_adj
                    snapshot['use_sparse'] = True
                else:
                    snapshot['use_sparse'] = False

                self.hypergraph_snapshots.append(snapshot)

        print(f"节点任务超图构建完成，共{len(self.hypergraph_snapshots)}个快照")
        if use_sparse:
            print(f"[稀疏矩阵] 所有快照已启用稀疏矩阵支持")

    # 图任务超图构建 - 支持氢键超边和稀疏矩阵
    def _build_graph_hyperdata(self):
        hyperedge_builders = self._get_graph_hyperedge_builders()

        for split in ['train', 'val', 'test']:
            graphs = self.graphs[split]
            print(f"开始构建{split}集超图（共{len(graphs)}个图）")

            for graph_idx, graph in enumerate(graphs):
                if graph_idx % 10 == 0:
                    print(f"  处理第{graph_idx + 1}/{len(graphs)}个图...")

                # 基本结构检查
                if 'num_nodes' not in graph:
                    logging.warning(f"{split}集第{graph_idx}个图缺少'num_nodes'键，跳过")
                    continue
                if 'label' not in graph:
                    logging.warning(f"{split}集第{graph_idx}个图缺少'label'键，跳过")
                    continue

                num_nodes = graph['num_nodes']
                graph_label = graph['label'].cpu().numpy()
                graph_snapshots = []

                # 检查是否启用稀疏矩阵
                use_sparse = num_nodes > self.sparse_threshold

                # 遍历配置的超边类型和重复次数
                for hyper_type, repeat in zip(self.graph_hyper_types, self.graph_hyper_repeats):
                    if hyper_type not in hyperedge_builders:
                        logging.warning(f"不支持的超边类型：{hyper_type}，跳过")
                        continue

                    for r in range(repeat):
                        try:
                            hyperedges_dict = hyperedge_builders[hyper_type](graph)
                        except Exception as e:
                            logging.warning(
                                f"{split}集第{graph_idx}个图构建{hyper_type}超边失败（第{r + 1}次）：{str(e)}，跳过")
                            continue

                        for subtype, hyperedges in hyperedges_dict.items():
                            if len(hyperedges) == 0:
                                continue

                            weights = self._compute_hyperedge_weights(
                                hyperedges,
                                hyper_type=subtype if hyper_type != 'bond' else subtype,
                                task_data=graph
                            )
                            H = self._build_hypergraph_matrix(hyperedges, num_nodes)
                            W = csr_matrix(np.diag(weights))
                            Theta = self._compute_theta_matrix(H, W, num_nodes)
                            wavelets, wavelets_inv = self._compute_wavelets(Theta)

                            # 新增：构建稀疏邻接矩阵
                            sparse_adj = None
                            if use_sparse:
                                sparse_adj = self._dense_to_sparse(Theta)

                            # 生成快照信息
                            snapshot_info = {
                                'type': hyper_type,
                                'subtype': subtype,
                                'H': H,
                                'W': W,
                                'Theta': Theta,
                                'wavelets': wavelets,
                                'wavelets_inv': wavelets_inv,
                                'num_hyperedges': len(hyperedges),
                                'repeat': r + 1
                            }

                            # 添加稀疏矩阵支持
                            if sparse_adj is not None:
                                snapshot_info['sparse_adj'] = sparse_adj
                                snapshot_info['use_sparse'] = True
                            else:
                                snapshot_info['use_sparse'] = False

                            graph_snapshots.append(snapshot_info)

                # 保存有效快照
                if graph_snapshots:
                    graph_info = {
                        'graph_idx': graph_idx,
                        'label': graph_label,
                        'num_nodes': num_nodes,
                        'snapshots': graph_snapshots,
                        'num_snapshots': len(graph_snapshots),
                        'smiles': graph.get('smiles', ''),
                        'use_sparse': use_sparse  # 记录是否使用稀疏矩阵
                    }
                    self.graph_hyper_data[split].append(graph_info)
                else:
                    logging.warning(f"图 {graph_idx}（SMILES: {graph.get('smiles', '未知')}）未生成任何超图快照，已跳过")

            print(f"{split}集超图构建完成，有效图数量：{len(self.graph_hyper_data[split])}")

    # 公共接口（保持不变）
    def run(self):
        """执行超图构建流程并返回结果"""
        print(f"开始构建{self.task_type}任务超图...")
        if self.task_type == "node_classification":
            self._build_node_snapshots()
            return self._get_node_hyperdata()
        elif self.task_type in ["graph_classification", "graph_regression"]:
            self._build_graph_hyperdata()
            # 输出有效图统计
            for split in ['train', 'val', 'test']:
                total = len(self.graphs[split])
                valid = len(self.graph_hyper_data[split])
                sparse_count = sum(1 for graph in self.graph_hyper_data[split] if graph.get('use_sparse', False))
                logging.info(f"{split}集：原始图{total}个，有效图{valid}个（其中{sparse_count}个使用稀疏矩阵）")
            return self._get_graph_hyperdata()
        else:
            raise ValueError(
                f"不支持的任务类型：{self.task_type}，支持类型：node_classification, graph_classification, graph_regression")

    def _get_node_hyperdata(self):
        """返回节点任务超图数据（添加稀疏矩阵信息）"""
        sparse_count = sum(1 for snap in self.hypergraph_snapshots if snap.get('use_sparse', False))
        print(f"[稀疏矩阵统计] 节点任务：{len(self.hypergraph_snapshots)}个快照中{sparse_count}个启用稀疏矩阵")

        return {
            'snapshots': self.hypergraph_snapshots,
            'num_nodes': self.num_nodes,
            'hyper_types': self.hyper_types,
            'task_type': 'node_classification',
            'hyperedge_size_range': self.hyperedge_size_range,
            'sparse_threshold': self.sparse_threshold,
            'sparse_enabled': sparse_count > 0  # 标记是否启用了稀疏矩阵
        }

    def _get_graph_hyperdata(self):
        """返回图任务超图数据，包含氢键配置和稀疏矩阵信息"""
        sparse_counts = {}
        for split in ['train', 'val', 'test']:
            sparse_counts[split] = sum(1 for graph in self.graph_hyper_data[split] if graph.get('use_sparse', False))

        print(
            f"[稀疏矩阵统计] 图任务：训练集{sparse_counts['train']}个，验证集{sparse_counts['val']}个，测试集{sparse_counts['test']}个图启用稀疏矩阵")

        return {
            'hyper_data': self.graph_hyper_data,
            'edge_types': self.all_edge_types,
            'num_edge_types': len(self.all_edge_types),
            'fg_config': {
                'fg_type_map': self.fg_type_map if self.is_mol_task else None,
                'fg_weights': self.fg_weights if self.is_mol_task else None
            },
            'hbond_config': {
                'hbond_type_map': self.hbond_type_map if self.is_mol_task else None,
                'hbond_weights': self.hbond_weights if self.is_mol_task else None,
                'distance_threshold': self.hbond_distance_threshold,
                'angle_threshold': self.hbond_angle_threshold
            },
            'task_type': self.task_type,
            'is_mol_task': self.is_mol_task,
            'hyper_types': self.graph_hyper_types,
            'hyper_repeats': self.graph_hyper_repeats,
            'hyperedge_size_range': self.hyperedge_size_range,
            'sparse_threshold': self.sparse_threshold,
            'sparse_enabled': any(count > 0 for count in sparse_counts.values())  # 标记是否启用了稀疏矩阵
        }
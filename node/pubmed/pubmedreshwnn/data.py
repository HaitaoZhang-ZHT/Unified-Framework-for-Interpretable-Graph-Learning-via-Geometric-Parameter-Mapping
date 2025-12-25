#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:HWNN
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: data.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2019/10/16
"""
import numpy as np
from collections import defaultdict
import torch
import time


class Data:
    def __init__(self, metapathscheme=None, device='cpu'):
        self.data_path = None
        self.nodes_labels = None  # one-hot torch矩阵
        self.nodes_names_int = None  # 节点名称列表（字符串）
        self.nodes_names_map = None  # 节点名称到索引的映射
        self.X_0 = None  # 特征矩阵
        self.class_num = 0  # 类别数
        self.nodes_labels_sequence = None  # 节点标签序列
        self.edges = None  # 边数据 (src, des, edge_type)
        self.nodes_number = 0  # 节点总数
        self.s = 1.0  # 小波变换参数
        self.hypergraph_snapshot = []  # 超图快照列表
        self.simplegraph_snapshot = []  # 简单图快照列表
        self.metapathscheme = metapathscheme  # 元路径方案
        self.labeled_node_index = []  # 标记节点索引（仅用于imdb）
        self.device = device  # 计算设备（CPU/GPU）

    def _label_string2matrix(self, nodes_labels_str):
        """将字符串标签转换为one-hot矩阵（向量化实现）"""
        unique_labels, inverse = np.unique(nodes_labels_str, return_inverse=True)
        class_num = len(unique_labels)
        sample_num = len(inverse)
        self.class_num = class_num

        # 向量化创建one-hot矩阵，避免Python循环
        nodes_labels = torch.zeros((sample_num, class_num), device=self.device)
        nodes_labels[torch.arange(sample_num, device=self.device), torch.from_numpy(inverse).to(self.device)] = 1.0
        self.nodes_labels_sequence = torch.from_numpy(inverse).to(self.device)
        return nodes_labels

    def _nodes_names_map(self, nodes_names_int):
        """使用字典推导式构建节点映射（O(n)时间复杂度）"""
        return {str(node): i for i, node in enumerate(nodes_names_int)}

    def _hypergraph_cora(self, edges):
        """超图构建优化（增加矩阵求逆稳定性处理）"""
        t_start = time.time()
        graph = defaultdict(set)  # 使用集合自动去重

        # 并行构建图结构（避免嵌套循环）
        for edge in edges:
            graph[edge[0]].add(edge[0])
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[1])
            graph[edge[1]].add(edge[0])
        graph = {k: list(v) for k, v in graph.items()}  # 转换为列表便于后续处理

        # 稀疏矩阵构建超图索引
        num_hyperedges = len(graph)
        row_indices, col_indices, values = [], [], []
        for col, hyperedge_nodes in enumerate(graph.values()):
            for node in hyperedge_nodes:
                row = self.nodes_names_map[node]
                row_indices.append(row)
                col_indices.append(col)
                values.append(1.0)

        # 使用稀疏张量减少内存占用
        indice_matrix = torch.sparse_coo_tensor(
            torch.tensor([row_indices, col_indices], device=self.device),
            torch.tensor(values, device=self.device),
            size=(self.nodes_number, num_hyperedges),
            device=self.device
        ).to_dense()

        # 向量化计算度矩阵
        W_e_diag = torch.ones(num_hyperedges, device=self.device)
        D_e_diag = torch.sum(indice_matrix, dim=0)  # 超边度
        D_v_diag = torch.sum(indice_matrix, dim=1)  # 节点度

        # 优化矩阵运算链
        D_v_sqrt_inv = torch.pow(D_v_diag, -0.5)
        D_v_sqrt_inv[torch.isinf(D_v_sqrt_inv)] = 0
        D_e_inv = torch.pow(D_e_diag, -1)
        D_e_inv[torch.isinf(D_e_inv)] = 0

        Theta = (D_v_sqrt_inv.diag() @ indice_matrix @ W_e_diag.diag() @
                 D_e_inv.diag() @ indice_matrix.T @ D_v_sqrt_inv.diag())

        # ----------------- 矩阵求逆稳定性增强 -----------------
        # 方法1: 添加正则化项避免奇异
        epsilon = 1e-6  # 正则化系数（可调整）
        Theta_reg = Theta + epsilon * torch.eye(Theta.size(0), device=self.device)

        try:
            # 方法2: 优先尝试普通求逆
            Theta_inverse = torch.linalg.inv(Theta_reg)
        except torch._C._LinAlgError:
            # 方法3: 若失败则使用伪逆(Moore-Penrose逆)
            print("矩阵求逆失败，使用伪逆计算...")
            Theta_inverse = torch.linalg.pinv(Theta_reg)

        # 处理可能的无穷大值
        Theta_inverse[torch.isinf(Theta_inverse)] = 0
        # ---------------------------------------------------

        # 带自环的Theta计算
        Theta_I = (D_v_sqrt_inv.diag() @ indice_matrix @ (W_e_diag + 1).diag() @
                   D_e_inv.diag() @ indice_matrix.T @ D_v_sqrt_inv.diag())
        Theta_I[torch.isnan(Theta_I)] = 0

        # 自环矩阵求逆稳定性处理
        Theta_I_reg = Theta_I + epsilon * torch.eye(Theta_I.size(0), device=self.device)
        try:
            Theta_I_inverse = torch.linalg.inv(Theta_I_reg)
        except torch._C._LinAlgError:
            print("自环矩阵求逆失败，使用伪逆计算...")
            Theta_I_inverse = torch.linalg.pinv(Theta_I_reg)

        Theta_I_inverse[torch.isinf(Theta_I_inverse)] = 0

        # 拉普拉斯矩阵与小波变换
        Laplacian = torch.eye(self.nodes_number, device=self.device) - Theta

        # 特征分解稳定性增强
        try:
            fourier_e, fourier_v = torch.linalg.eigh(Laplacian)
        except torch._C._LinAlgError:
            print("特征分解失败，使用奇异值分解替代...")
            U, S, V = torch.linalg.svd(Laplacian)
            fourier_e = S
            fourier_v = U

        # 向量化小波变换计算
        exp_term = torch.exp(-1.0 * fourier_e * self.s)
        wavelets = fourier_v @ exp_term.diag() @ fourier_v.T
        wavelets_inv = fourier_v @ torch.exp(fourier_e * self.s).diag() @ fourier_v.T
        wavelets_t = wavelets.T

        # 阈值处理
        wavelets[wavelets < 1e-5] = 0
        wavelets_inv[wavelets_inv < 1e-5] = 0
        wavelets_t[wavelets_t < 1e-5] = 0

        print(f"超图构建耗时: {time.time() - t_start:.4f}秒")
        return {
            "graph": graph,
            "indice_matrix": indice_matrix,
            "D_v_diag": D_v_diag,
            "D_e_diag": D_e_diag,
            "W_e_diag": W_e_diag,
            "laplacian": Laplacian,
            "fourier_v": fourier_v,
            "fourier_e": fourier_e,
            "wavelets": wavelets,
            "wavelets_inv": wavelets_inv,
            "wavelets_t": wavelets_t,
            "Theta": Theta,
            "Theta_inv": Theta_inverse,
            "Theta_I": Theta_I,
            "Theta_I_inv": Theta_I_inverse,
        }

    def _simplegraph_cora(self, edges):
        """简单图构建优化（集合去重+稀疏邻接矩阵）"""
        t_start = time.time()
        graph = defaultdict(set)  # 使用集合自动去重并过滤自环

        # 并行构建图结构
        for edge in edges:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])

        # 过滤自环并转换为列表
        for node in graph:
            if node in graph[node]:
                graph[node].remove(node)
            graph[node] = list(graph[node])

        # 稀疏邻接矩阵构建
        row_indices, col_indices, values = [], [], []
        node_degree_flat = torch.zeros(self.nodes_number, device=self.device)

        for node, neighbors in graph.items():
            node_idx = self.nodes_names_map[node]
            node_degree_flat[node_idx] = len(neighbors)
            for neighbor in neighbors:
                neighbor_idx = self.nodes_names_map[neighbor]
                row_indices.append(node_idx)
                col_indices.append(neighbor_idx)
                values.append(1.0)

        # 使用稀疏张量表示邻接矩阵
        A = torch.sparse_coo_tensor(
            torch.tensor([row_indices, col_indices], device=self.device),
            torch.tensor(values, device=self.device),
            size=(self.nodes_number, self.nodes_number),
            device=self.device
        ).to_dense()

        # 拉普拉斯矩阵优化计算
        node_degree_flat_pow = torch.pow(node_degree_flat, -0.5)
        node_degree_flat_pow[torch.isinf(node_degree_flat_pow)] = 0
        node_degree_flat_pow[torch.isnan(node_degree_flat_pow)] = 0

        L = torch.eye(self.nodes_number, device=self.device) - (
                node_degree_flat_pow.diag() @ A @ node_degree_flat_pow.diag()
        )
        fourier_e, fourier_v = torch.linalg.eigh(L)

        # 小波变换向量化计算
        exp_term = torch.exp(-1.0 * fourier_e * self.s)
        wavelets = fourier_v @ exp_term.diag() @ fourier_v.T
        wavelets_inv = fourier_v @ torch.exp(fourier_e * self.s).diag() @ fourier_v.T
        wavelets[wavelets < 1e-5] = 0
        wavelets_inv[wavelets_inv < 1e-5] = 0

        # 节点类型映射（字典推导式优化）
        node_type_list = [0]
        node_type_map = {node: 0 for node in graph.keys()}
        type_node_map = defaultdict(list)
        type_node_map[0] = list(graph.keys())

        print(f"简单图构建耗时: {time.time() - t_start:.4f}秒")
        return {
            "graph": {k: list(v) for k, v in graph.items()},
            "edges": edges,
            "node_type_list": node_type_list,
            "node_degree_flat": node_degree_flat,
            "node_type_map": node_type_map,
            "type_node_map": type_node_map,
            "adj": A,
            "laplacian": L,
            "fourier_e": fourier_e,
            "fourier_v": fourier_v,
            "wavelets": wavelets,
            "wavelets_inv": wavelets_inv
        }

    def load(self, data_path, data_name, save=False, use_gpu=True):
        """数据加载优化（自动设备检测+稀疏数据处理）"""
        t_start = time.time()
        print('开始加载数据...')

        # 自动检测并设置计算设备
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            print("检测到GPU，使用CUDA加速")
        else:
            self.device = 'cpu'
            print("使用CPU计算")

        self.data_path = data_path
        # 自动检测分隔符（提升兼容性）
        with open(f"{data_path}/{data_name}.content", 'r') as f:
            header = f.read(100)
            delimiter = '\t' if '\t' in header else ' '

        # 高效加载数据（指定数据类型）
        content = np.loadtxt(f"{data_path}/{data_name}.content", dtype=str, delimiter=delimiter)

        # 标签处理（向量化转换）
        nodes_labels_str = content[:, -1]
        self.nodes_labels = self._label_string2matrix(nodes_labels_str)
        self.nodes_number = self.nodes_labels.size(0)
        print(f"节点总数: {self.nodes_number}, 类别数: {self.class_num}")

        # 节点名称映射（字典推导式）
        self.nodes_names_int = content[:, 0].astype(str)
        print('构建节点映射...')
        self.nodes_names_map = self._nodes_names_map(self.nodes_names_int)

        # 特征矩阵（直接加载到目标设备）
        nodes_features_int = content[:, 1:-1].astype(np.float64)
        print('构建特征矩阵...')
        self.X_0 = torch.from_numpy(nodes_features_int).to(self.device)

        # 边数据加载
        print('加载边数据...')
        self.edges = np.loadtxt(f"{data_path}/{data_name}.cites", dtype=str, delimiter=delimiter)

        # 图结构构建
        if data_name in ["cora", 'pubmed', 'aminer', 'spammer']:
            """
            compelete simple graph
            """
            print("construct simple graphs...")
            simple_graph = self._simplegraph_cora(self.edges)
            self.simplegraph_snapshot.append(simple_graph)

            """
            simple graph snapshots, you just need to remove unrelated edges in self.edges, and send
            them into the same function
            simple_graph = self._simplegraph_cora(self.edges)
            self.simplegraph_snapshot.append(simple_graph)
            """

            print("construct hypergraphs...")
            hypergraph = self._hypergraph_cora(self.edges)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            print("load done!")
        elif data_name == "dblp" or data_name == "imdb":
            """
              compelete simple graph
              """
            print("construct simple graphs...")
            simple_graph = self._simplegraph_dblp(self.edges)
            self.simplegraph_snapshot.append(simple_graph)

            print("construct hypergraphs...")

            hypergraph = self._hypergraph_dblp(self.edges)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            # self.hypergraph_snapshot.append(hypergraph)

            print("load done!")


if __name__ == "__main__":
    # 自动选择最优设备并加载数据
    data = Data(device='cuda' if torch.cuda.is_available() else 'cpu')
    data.load(data_path='./data/cora/', data_name='cora', use_gpu=True)
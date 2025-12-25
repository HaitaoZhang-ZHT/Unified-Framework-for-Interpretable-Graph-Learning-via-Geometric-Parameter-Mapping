import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from torch_sparse import spspmm, coalesce  # 新增：稀疏矩阵操作


class SparseAttentionPooling(nn.Module):
    """稀疏版节点级注意力池化层（适配大图）"""

    def __init__(self, hidden_dim, dropout=0.0, device="cpu", k_neighbors=50):
        super(SparseAttentionPooling, self).__init__()
        self.device = device
        self.dropout = dropout
        self.k_neighbors = k_neighbors  # 每个节点的最大邻居数

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # 输入是节点对特征
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)

    def forward(self, node_feats, valid_masks=None, sparse_adj=None):
        """
        sparse_adj: 稀疏邻接矩阵 (edge_index, edge_attr)
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] 或 None
        """
        batch_size, num_nodes, feat_dim = node_feats.shape

        if sparse_adj is None:
            # 回退到稠密注意力（小图情况）
            return self._dense_attention(node_feats, valid_masks)

        edge_index = sparse_adj[0]  # [2, num_edges]
        edge_attr = sparse_adj[1] if len(sparse_adj) > 1 else None

        # 为每个batch处理稀疏注意力
        batch_att_weights = []
        batch_graph_feats = []

        for b in range(batch_size):
            # 提取当前batch的节点特征
            node_feats_b = node_feats[b]  # [num_nodes, feat_dim]

            # 构建节点对特征：只计算有边的节点对
            src_nodes = edge_index[0]  # 源节点索引
            dst_nodes = edge_index[1]  # 目标节点索引
            src_feats = node_feats_b[src_nodes]  # [num_edges, feat_dim]
            dst_feats = node_feats_b[dst_nodes]  # [num_edges, feat_dim]

            # 拼接节点对特征
            pair_feats = torch.cat([src_feats, dst_feats], dim=1)  # [num_edges, feat_dim*2]

            # 计算注意力分数
            att_scores = self.attention_net(pair_feats).squeeze(-1)  # [num_edges]

            # 应用边权重（如果有）
            if edge_attr is not None:
                att_scores = att_scores * edge_attr.squeeze(-1)

            # 创建稀疏注意力矩阵
            att_weights_sparse = torch.sparse_coo_tensor(
                edge_index,
                F.softmax(att_scores, dim=0),
                (num_nodes, num_nodes)
            )

            # 稀疏矩阵乘法：聚合邻居特征
            graph_feat_sparse = torch.sparse.mm(att_weights_sparse, node_feats_b)  # [num_nodes, feat_dim]

            # 池化得到图特征（均值池化）
            if valid_masks is not None:
                valid_mask_b = valid_masks[b]  # [num_nodes]
                valid_feats = graph_feat_sparse[valid_mask_b]
                if len(valid_feats) > 0:
                    graph_feat = valid_feats.mean(dim=0)
                else:
                    graph_feat = graph_feat_sparse.mean(dim=0)
            else:
                graph_feat = graph_feat_sparse.mean(dim=0)

            batch_graph_feats.append(graph_feat)
            batch_att_weights.append(att_weights_sparse)

        graph_feats = torch.stack(batch_graph_feats)  # [batch_size, feat_dim]
        att_weights = torch.stack(batch_att_weights)  # [batch_size, num_nodes, num_nodes] (稀疏)

        return graph_feats, att_weights

    def _dense_attention(self, node_feats, valid_masks=None):
        """回退到稠密注意力（小图使用）"""
        batch_size, num_nodes, _ = node_feats.shape
        att_scores = self.attention_net(
            torch.cat([node_feats.unsqueeze(2).expand(-1, -1, num_nodes, -1),
                       node_feats.unsqueeze(1).expand(-1, num_nodes, -1, -1)], dim=-1)
        ).squeeze(-1)

        if valid_masks is not None:
            valid_masks_expanded = valid_masks.unsqueeze(1) & valid_masks.unsqueeze(2)
            att_scores = att_scores.masked_fill(~valid_masks_expanded, -1e9)

        att_weights = F.softmax(att_scores, dim=2)
        graph_feat = torch.bmm(att_weights, node_feats).mean(dim=1)

        return graph_feat, att_weights


class HypergraphConv(nn.Module):
    """超图卷积层（支持稀疏注意力机制）"""

    def __init__(self, in_channels, out_channels, use_attention=False, attention_heads=1,
                 dropout=0.0, device="cpu", force_xavier=False,
                 approx=False, K1=2, K2=2, sparse_attention_threshold=5000, task_type=None,
                 use_M_init_linear=False, M_matrix_path=None):  # 新增：是否用M矩阵初始化线性权重
        super(HypergraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_heads = attention_heads if use_attention else 1
        self.dropout = dropout
        self.device = device
        self.force_xavier = force_xavier
        self.sparse_attention_threshold = sparse_attention_threshold  # 启用稀疏注意力的节点数阈值
        self.use_M_init_linear = use_M_init_linear  # 新增：控制是否用M矩阵初始化线性权重
        self.M_matrix_path = M_matrix_path

        # 多项式近似参数
        self.approx = approx
        self.K1 = K1
        self.K2 = K2
        self.task_type = task_type

        # 基础卷积权重
        self.weight = Parameter(torch.Tensor(
            in_channels, out_channels * self.attention_heads
        ).to(device))
        self.bias = Parameter(torch.Tensor(out_channels * self.attention_heads).to(device))

        # 注意力机制参数
        self.att_drop = nn.Dropout(dropout)
        if self.use_attention:
            # 初始化注意力权重（全部使用Xavier）
            self.att_weight1 = Parameter(torch.Tensor(2 * out_channels, out_channels // 2).to(device))
            self.att_weight2 = Parameter(torch.Tensor(out_channels // 2, 1).to(device))
            nn.init.xavier_uniform_(self.att_weight1)
            nn.init.xavier_uniform_(self.att_weight2)

        # 多项式近似系数
        if self.approx:
            self.par_k1 = Parameter(torch.Tensor(self.K1).to(device))
            self.par_k2 = Parameter(torch.Tensor(self.K2).to(device))
            nn.init.uniform_(self.par_k1, 0, 0.99)
            nn.init.uniform_(self.par_k2, 0, 0.99)

        self.reset_parameters()

    def reset_parameters(self):
        """重置参数 - 修改为使用M矩阵初始化线性权重"""
        if self.use_M_init_linear:
            # 使用M矩阵初始化线性权重
            self._init_weight_with_M()
        else:
            # 使用Xavier初始化
            nn.init.xavier_uniform_(self.weight)

        nn.init.zeros_(self.bias)

    def _init_weight_with_M(self):
        """使用M矩阵初始化线性权重"""
        # 如果没有提供路径，使用默认路径（向后兼容）
        if self.M_matrix_path is None:
            self.M_matrix_path = "None"
            print(f"[警告] 未提供M矩阵路径，使用默认路径：{self.M_matrix_path}")

        try:
            # 1. 加载M矩阵文件
            if self.M_matrix_path.endswith('.npz'):
                # 处理.npz文件
                with np.load(self.M_matrix_path, allow_pickle=True) as npz_data:
                    if "M" not in npz_data:
                        raise KeyError(f"npz文件中未找到键'M'，当前包含键：{list(npz_data.keys())}")
                    M_np = npz_data["M"].astype(np.float32)
            else:
                # 处理.npy文件或其他格式
                M_np = np.load(self.M_matrix_path).astype(np.float32)

            # 2. 校验M矩阵维度
            if M_np.ndim != 2 or M_np.shape[0] != M_np.shape[1]:
                raise ValueError(f"矩阵M必须是2D方阵，当前形状：{M_np.shape}")

            M_size = M_np.shape[0]  # M矩阵的边长

            # 3. 计算需要的堆叠次数
            required_rows = self.in_channels
            required_cols = self.out_channels * self.attention_heads

            # 行方向堆叠次数
            row_stack_times = (required_rows + M_size - 1) // M_size  # 向上取整
            # 列方向堆叠次数
            col_stack_times = (required_cols + M_size - 1) // M_size  # 向上取整

            # 4. 堆叠M矩阵
            M_np_stacked = np.tile(M_np, (row_stack_times, col_stack_times))

            # 5. 裁剪到目标形状
            M_np_final = M_np_stacked[:required_rows, :required_cols]

            # 6. 转换为tensor并赋值
            self.weight.data = torch.tensor(M_np_final, device=self.device, dtype=torch.float32)

            print(f"[HypergraphConv] 成功用M矩阵初始化线性权重，路径：{self.M_matrix_path}")
            print(f"  原始M矩阵形状：{M_np.shape}，目标形状：[{required_rows}, {required_cols}]")
            print(f"  堆叠模式：[{row_stack_times}, {col_stack_times}]")

        except FileNotFoundError as e:
            print(f"[警告] 未找到M矩阵文件：{str(e)} → 改用Xavier初始化线性权重")
            nn.init.xavier_uniform_(self.weight)
        except KeyError as e:
            print(f"[警告] M矩阵加载失败：{str(e)} → 改用Xavier初始化线性权重")
            nn.init.xavier_uniform_(self.weight)
        except ValueError as e:
            print(f"[警告] M矩阵维度非法：{str(e)} → 改用Xavier初始化线性权重")
            nn.init.xavier_uniform_(self.weight)
        except Exception as e:
            print(f"[警告] M矩阵加载未知错误：{str(e)} → 改用Xavier初始化线性权重")
            nn.init.xavier_uniform_(self.weight)

    def _should_use_sparse_attention(self, num_nodes, batch_size=1):
        """判断是否使用稀疏注意力"""
        if self.task_type == "node_classification":
            total_nodes = num_nodes  # 节点任务：num_nodes就是总节点数
        else:
            total_nodes = num_nodes  # 图任务：只关注单个图的节点数

        return total_nodes > self.sparse_attention_threshold and self.use_attention

    def _compute_sparse_attention(self, x, theta, batch_mode):
        """稀疏注意力计算 - 核心优化"""
        if batch_mode:
            batch_size, num_nodes, heads, feat_dim = x.shape
            x_flat = x.reshape(batch_size * num_nodes, heads, feat_dim)
        else:
            num_nodes, heads, feat_dim = x.shape
            batch_size = 1
            x_flat = x.reshape(num_nodes, heads, feat_dim)

        # 将theta转换为稀疏格式
        if batch_mode:
            # 批量处理：取第一个样本的theta作为拓扑结构（假设批量内拓扑相似）
            theta_sample = theta[0].cpu().numpy()
        else:
            theta_sample = theta.cpu().numpy()

        # 获取非零元素（边）
        rows, cols = np.where(theta_sample > 1e-5)
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=self.device)
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            # 没有边，返回均值池化
            if batch_mode:
                return x.mean(dim=2)  # [batch_size, num_nodes, feat_dim]
            else:
                return x.mean(dim=1)  # [num_nodes, feat_dim]

        # 计算稀疏注意力分数
        src_feats = x_flat[edge_index[0]]  # [num_edges, heads, feat_dim]
        dst_feats = x_flat[edge_index[1]]  # [num_edges, heads, feat_dim]

        # 多头部注意力计算
        head_outputs = []
        for head in range(heads):
            src_head = src_feats[:, head, :]  # [num_edges, feat_dim]
            dst_head = dst_feats[:, head, :]  # [num_edges, feat_dim]

            # 拼接特征并计算注意力
            pair_feats = torch.cat([src_head, dst_head], dim=1)  # [num_edges, feat_dim*2]
            att_scores = F.leaky_relu(torch.matmul(pair_feats, self.att_weight1))  # [num_edges, feat_dim//2]
            att_scores = torch.matmul(att_scores, self.att_weight2).squeeze(-1)  # [num_edges]

            # 应用softmax
            att_weights = torch.sparse_coo_tensor(
                edge_index,
                F.softmax(att_scores, dim=0),
                (num_nodes * batch_size, num_nodes * batch_size)
            )

            # 稀疏矩阵乘法
            x_head = x_flat[:, head, :]  # [num_nodes*batch_size, feat_dim]
            attended_head = torch.sparse.mm(att_weights, x_head)  # [num_nodes*batch_size, feat_dim]
            head_outputs.append(attended_head)

        # 合并多头结果
        x_out = torch.stack(head_outputs, dim=1)  # [num_nodes*batch_size, heads, feat_dim]

        if batch_mode:
            x_out = x_out.reshape(batch_size, num_nodes, heads, feat_dim)
            x_out = x_out.mean(dim=2)  # [batch_size, num_nodes, feat_dim]
        else:
            x_out = x_out.reshape(num_nodes, heads, feat_dim)
            x_out = x_out.mean(dim=1)  # [num_nodes, feat_dim]

        return x_out

    def _compute_dense_attention(self, x, theta, batch_mode):
        """原始稠密注意力计算（小图使用）"""
        num_nodes = x.size(-3) if batch_mode else x.size(0)

        if batch_mode:
            # 修复维度问题：正确处理批量模式
            batch_size, num_nodes, heads, feat_dim = x.shape
            # 重塑为 (batch_size, heads, num_nodes, feat_dim)
            x_reshaped = x.permute(0, 2, 1, 3)  # [B, N, H, C] → [B, H, N, C]

            # 构建节点对特征
            x_i = x_reshaped.unsqueeze(3).expand(batch_size, heads, num_nodes, num_nodes, feat_dim)
            x_j = x_reshaped.unsqueeze(2).expand(batch_size, heads, num_nodes, num_nodes, feat_dim)
            x_pair = torch.cat([x_i, x_j], dim=-1)  # [B, H, N, N, 2*feat_dim]

            # 重塑为二维矩阵以便矩阵乘法
            x_pair_flat = x_pair.reshape(batch_size * heads * num_nodes * num_nodes, 2 * feat_dim)

            # 计算注意力分数
            att_intermediate = torch.matmul(x_pair_flat, self.att_weight1)  # [B*H*N*N, feat_dim//2]
            att_scores = torch.matmul(att_intermediate, self.att_weight2)  # [B*H*N*N, 1]
            att_scores = att_scores.reshape(batch_size, heads, num_nodes, num_nodes)
            att_scores = F.leaky_relu(att_scores)

            # 应用拓扑约束
            theta_expanded = theta.unsqueeze(1)  # [B, 1, N, N]
            mask = (theta_expanded == 0)
            att_scores = att_scores.masked_fill(mask, -1e9)

            att_weights = F.softmax(att_scores, dim=-1)  # [B, H, N, N]
            return att_weights

        else:
            # 单图模式
            num_nodes, heads, feat_dim = x.shape
            x_reshaped = x.permute(1, 0, 2)  # [N, H, C] → [H, N, C]

            # 构建节点对特征
            x_i = x_reshaped.unsqueeze(2).expand(heads, num_nodes, num_nodes, feat_dim)
            x_j = x_reshaped.unsqueeze(1).expand(heads, num_nodes, num_nodes, feat_dim)
            x_pair = torch.cat([x_i, x_j], dim=-1)  # [H, N, N, 2*feat_dim]

            # 重塑为二维矩阵
            x_pair_flat = x_pair.reshape(heads * num_nodes * num_nodes, 2 * feat_dim)

            # 计算注意力分数
            att_intermediate = torch.matmul(x_pair_flat, self.att_weight1)  # [H*N*N, feat_dim//2]
            att_scores = torch.matmul(att_intermediate, self.att_weight2)  # [H*N*N, 1]
            att_scores = att_scores.reshape(heads, num_nodes, num_nodes)
            att_scores = F.leaky_relu(att_scores)

            # 应用拓扑约束
            theta_expanded = theta.unsqueeze(0)  # [1, N, N]
            mask = (theta_expanded == 0)
            att_scores = att_scores.masked_fill(mask, -1e9)

            return F.softmax(att_scores, dim=-1)  # [H, N, N]

    def _compute_polynomial(self, theta):
        """多项式近似计算（保持不变）"""
        batch_mode = theta.dim() == 3
        theta_pows = []

        for k in range(self.K1):
            if batch_mode:
                pow_k = torch.stack([torch.matrix_power(theta[i], k) for i in range(theta.size(0))])
            else:
                pow_k = torch.matrix_power(theta, k)
            theta_pows.append(pow_k)

        theta_pows = torch.stack(theta_pows, dim=0)
        poly = torch.sum(self.par_k1[:, None, None, None] * theta_pows, dim=0)

        theta_t = theta.transpose(-2, -1)
        theta_t_pows = []
        for k in range(self.K2):
            if batch_mode:
                pow_k = torch.stack([torch.matrix_power(theta_t[i], k) for i in range(theta_t.size(0))])
            else:
                pow_k = torch.matrix_power(theta_t, k)
            theta_t_pows.append(pow_k)

        theta_t_pows = torch.stack(theta_t_pows, dim=0)
        poly_t = torch.sum(self.par_k2[:, None, None, None] * theta_t_pows, dim=0)

        return poly @ poly_t

    def forward(self, x, theta):
        batch_mode = theta.dim() == 3
        num_nodes = theta.size(-1)

        # 线性变换 - 这里使用的self.weight现在可以用M矩阵初始化
        x = torch.matmul(x, self.weight) + self.bias
        if batch_mode:
            x = x.view(-1, num_nodes, self.attention_heads, self.out_channels)
        else:
            x = x.view(num_nodes, self.attention_heads, self.out_channels)

        # 多项式近似处理
        if self.approx:
            theta_processed = self._compute_polynomial(theta)
        else:
            theta_processed = theta

        # 注意力机制或普通卷积
        if self.use_attention:
            # 根据节点数决定使用稀疏还是稠密注意力
            if self._should_use_sparse_attention(num_nodes, batch_size=x.size(0) if batch_mode else 1):
                x = self._compute_sparse_attention(x, theta_processed, batch_mode)
            else:
                att_weights = self._compute_dense_attention(x, theta_processed, batch_mode)
                att_weights = self.att_drop(att_weights)

                if batch_mode:
                    # 批量模式注意力聚合
                    batch_size, num_nodes, heads, feat_dim = x.shape
                    x_reshaped = x.permute(0, 2, 1, 3)  # [B, N, H, C] → [B, H, N, C]
                    x_attended = torch.matmul(att_weights, x_reshaped)  # [B, H, N, C]
                    x = x_attended.permute(0, 2, 1, 3)  # [B, H, N, C] → [B, N, H, C]
                    x = x.mean(dim=2)  # [B, N, C]
                else:
                    # 单图模式注意力聚合
                    x_reshaped = x.permute(1, 0, 2)  # [N, H, C] → [H, N, C]
                    x_attended = torch.matmul(att_weights, x_reshaped)  # [H, N, C]
                    x = x_attended.permute(1, 0, 2)  # [H, N, C] → [N, H, C]
                    x = x.mean(dim=1)  # [N, C]
        else:
            # 普通超图卷积
            if batch_mode:
                x = torch.matmul(theta_processed, x.squeeze(2))
            else:
                x = torch.matmul(theta_processed, x.squeeze(1))

        return x


class HWNN(nn.Module):
    """超图神经网络（支持稀疏注意力）"""

    def __init__(self, task_type, input_dim, hidden_dim, output_dim,
                 num_layers=2, K1=3, K2=3, approx=False, dropout=0.0,
                 device="cpu", use_conv_attention=False, conv_attention_heads=1,
                 use_pool_attention=True, sparse_attention_threshold=5000,
                 use_M_init_linear_input=False, use_M_init_linear_hidden=False
                 ,M_matrix_path = None):  # 新增：控制各层是否用M初始化
        super(HWNN, self).__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.dropout = dropout
        self.device = device
        self.sparse_attention_threshold = sparse_attention_threshold
        self.M_matrix_path = M_matrix_path

        self.use_conv_attention = use_conv_attention
        self.conv_attention_heads = conv_attention_heads if use_conv_attention else 1
        self.use_pool_attention = use_pool_attention

        if self.task_type != "node_classification" and self.use_pool_attention:
            self.attention_pooling = SparseAttentionPooling(
                hidden_dim=hidden_dim,
                dropout=dropout,
                device=device
            )

        # 输入卷积层 - 使用M矩阵初始化线性权重
        self.input_conv = HypergraphConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            use_attention=use_conv_attention,
            attention_heads=conv_attention_heads,
            dropout=dropout,
            device=device,
            force_xavier=False,  # 不再用于控制注意力权重
            approx=self.approx,
            K1=self.K1,
            K2=self.K2,
            sparse_attention_threshold=sparse_attention_threshold,
            task_type=self.task_type,
            use_M_init_linear=use_M_init_linear_input,  # 控制输入层是否用M初始化
            M_matrix_path = M_matrix_path
        )

        # 隐藏卷积层
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_convs.append(HypergraphConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                use_attention=use_conv_attention,
                attention_heads=conv_attention_heads,
                dropout=dropout,
                device=device,
                force_xavier=False,  # 不再用于控制注意力权重
                approx=self.approx,
                K1=self.K1,
                K2=self.K2,
                sparse_attention_threshold=sparse_attention_threshold,
                task_type=self.task_type,
                use_M_init_linear=use_M_init_linear_hidden,  # 控制隐藏层是否用M初始化
                M_matrix_path = M_matrix_path
            ))

        # 输出层（保持不变）
        if task_type in ["node_classification", "graph_classification"]:
            self.output_layer = nn.Linear(hidden_dim, output_dim).to(device)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Flatten()
            ).to(device)

        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data):
        x = data['features']
        theta_snapshots = data['theta_snapshots']
        valid_masks = data.get('valid_masks', None)
        sparse_adj = data.get('sparse_adj', None)  # 新增：稀疏邻接矩阵

        # 调整掩码维度（保持不变）
        if valid_masks is not None and x.dim() == 3:
            batch_size_x, num_nodes_x, _ = x.shape
            batch_size_mask, num_nodes_mask = valid_masks.shape
            if batch_size_mask != batch_size_x:
                raise ValueError(f"批次维度不匹配：特征({batch_size_x}) vs 掩码({batch_size_mask})")
            if num_nodes_mask != num_nodes_x:
                valid_masks = valid_masks[:, :num_nodes_x]
                if valid_masks.shape[1] < num_nodes_x:
                    pad_size = num_nodes_x - valid_masks.shape[1]
                    valid_masks = F.pad(valid_masks, (0, pad_size), mode='constant', value=False)
            data['valid_masks'] = valid_masks

        # 超图卷积前向
        x = self._layer_forward(x, theta_snapshots, self.input_conv)
        for conv in self.hidden_convs:
            x = self._layer_forward(x, theta_snapshots, conv)

        # 任务处理
        if self.task_type == "node_classification":
            output_device = next(self.output_layer.parameters()).device
            x = x.to(output_device)
            out = self.output_layer(x)
            return out
        else:
            graph_feat = None
            if self.use_pool_attention:
                # 使用稀疏池化
                graph_feat, node_att_weights = self.attention_pooling(
                    x, valid_masks, sparse_adj
                )
                data['node_att_weights'] = node_att_weights
            else:
                if valid_masks is not None:
                    valid_masks = valid_masks.unsqueeze(-1)
                    x = x * valid_masks
                    num_valid = valid_masks.sum(dim=1, keepdim=True) + 1e-5
                    graph_feat = (x.sum(dim=1) / num_valid.squeeze(1))
                else:
                    graph_feat = x.mean(dim=1)

            output_device = next(self.output_layer.parameters()).device
            graph_feat = graph_feat.to(output_device)
            out = self.output_layer(graph_feat)
            return out.view(-1)

    def _layer_forward(self, x, theta_snapshots, conv_layer):
        """层前向传播（保持不变）"""
        snapshot_outs = []
        for theta in theta_snapshots:
            h = conv_layer(x, theta)
            snapshot_outs.append(h)
        x = torch.mean(torch.stack(snapshot_outs), dim=0)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x

    def loss(self, pred, target):
        """损失函数（保持不变）"""
        if self.task_type in ["node_classification", "graph_classification"]:
            target = target.long()
            return F.cross_entropy(pred, target)
        else:
            pred = pred.view(-1)
            target = target.view(-1).float()
            return F.mse_loss(pred, target)
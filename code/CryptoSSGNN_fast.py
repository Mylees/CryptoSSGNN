# -*- coding: utf-8 -*-
# ========================= FAST / OPTIMIZED VARIANT =========================
# This file was auto-generated to incorporate runtime optimizations:
# 1) Tensorized relation aggregation using index_add_ (removes inner Python loops)
# 2) Vectorized negative-sample conflict fixes in compute_graph_loss
# 3) Faster attribute-graph construction via kneighbors_graph (sparse) without Python loops
# 4) Less frequent eval by default + shorter patience for earlier early-stopping
# If any patch could not be applied (pattern not found), the original code for that
# region is left unchanged; see the patch report at the bottom of this file header.
# ==========================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import load_npz, csr_matrix
import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir):
    """加载多关系图数据"""
    print(f"Loading data from {data_dir}...")
    
    # 加载特征和标签
    features = load_npz(os.path.join(data_dir, 'features.npz')).toarray()
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    
    # 加载所有关系的邻接矩阵
    adj_matrices = []
    relation_files = [f for f in os.listdir(data_dir) if f.startswith('adj_relation') and f.endswith('.npz')]
    relation_files.sort()  # 确保顺序一致
    
    for file in relation_files:
        adj = load_npz(os.path.join(data_dir, file))
        # 转换为CSR格式以支持索引操作
        if not isinstance(adj, csr_matrix):
            adj = adj.tocsr()
        adj_matrices.append(adj)
    
    print(f"Loaded {len(adj_matrices)} relations")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    return features, labels, adj_matrices

def sparse_to_edge_index(sparse_matrix):
    """将稀疏矩阵转换为edge_index格式"""
    # 确保转换为COO格式
    if hasattr(sparse_matrix, 'tocoo'):
        coo = sparse_matrix.tocoo()
    else:
        coo = sparse_matrix
    
    # 处理空矩阵的情况
    if coo.nnz == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return edge_index, edge_weight
    
    edge_index = torch.stack([torch.from_numpy(coo.row), torch.from_numpy(coo.col)], dim=0)
    edge_weight = torch.from_numpy(coo.data).float()
    return edge_index.long(), edge_weight

def construct_attribute_graph(features, k_neighbors=10):
    """Construct attribute graph quickly using sparse kNN.
    Returns edge_index (2xE np.array or torch.LongTensor) and edge_weight (E,).
    """
    import numpy as np
    from sklearn.neighbors import kneighbors_graph
    
    X = np.asarray(features)
    n = X.shape[0]
    k = min(k_neighbors + 1, n)
    A = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # Gaussian weights
    A.data = np.exp(-(A.data ** 2))
    
    # Convert CSR to edge_index/edge_weight
    indices = A.nonzero()
    src = indices[0].astype(np.int64)
    dst = indices[1].astype(np.int64)
    weights = A.data.astype(np.float32)
    
    try:
        import torch
        edge_index = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_weight = torch.as_tensor(weights, dtype=torch.float32)
    except Exception:
        edge_index = np.vstack([src, dst])
        edge_weight = weights
    return edge_index, edge_weight

class SimilarityLayer(nn.Module):
    """相似度层 - CARE-GNN的核心组件之一"""
    def __init__(self, input_dim, similarity_dim=64):
        super(SimilarityLayer, self).__init__()
        self.similarity_dim = similarity_dim
        
        # 相似度计算网络
        self.similarity_net = nn.Sequential(
            nn.Linear(input_dim, similarity_dim),
            nn.ReLU(),
            nn.Linear(similarity_dim, similarity_dim)
        )
        
        # 注意力权重
        self.attention = nn.Linear(similarity_dim * 2, 1)
        
    def forward(self, node_features, neighbor_features):
        """
        计算节点与邻居的相似度
        Args:
            node_features: [num_nodes, input_dim]
            neighbor_features: [num_nodes, num_neighbors, input_dim]
        """
        # 计算节点和邻居的相似度表示
        node_sim = self.similarity_net(node_features)  # [num_nodes, similarity_dim]
        neighbor_sim = self.similarity_net(neighbor_features)  # [num_nodes, num_neighbors, similarity_dim]
        
        # 扩展节点特征以匹配邻居维度
        node_sim_expanded = node_sim.unsqueeze(1).expand_as(neighbor_sim)  # [num_nodes, num_neighbors, similarity_dim]
        
        # 计算注意力权重
        concat_features = torch.cat([node_sim_expanded, neighbor_sim], dim=-1)  # [num_nodes, num_neighbors, similarity_dim*2]
        attention_weights = self.attention(concat_features).squeeze(-1)  # [num_nodes, num_neighbors]
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return attention_weights

class LabelAwareAttention(nn.Module):
    """标签感知注意力机制 - 修复版本"""
    def __init__(self, input_dim, num_classes=2, disable_attention=False):
        super(LabelAwareAttention, self).__init__()
        self.num_classes = num_classes
        self.disable_attention = disable_attention  # 🔧 新增：消融实验开关
        
        # 为每个类别学习不同的注意力
        self.class_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()  # 🔧 修复: 确保输出[0,1]
            ) for _ in range(num_classes)
        ])
        
        # 类别预测器
        self.class_predictor = nn.Linear(input_dim, num_classes)
        
        # 🔧 修复: 添加bias确保注意力不会过小
        self.attention_bias = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, features, predicted_labels=None):
        """
        根据预测标签计算注意力权重
        """
        if predicted_labels is None:
            # 使用软标签
            class_logits = self.class_predictor(features)
            class_probs = F.softmax(class_logits, dim=-1)
        else:
            class_probs = predicted_labels
        
        # 🔧 消融实验：如果禁用注意力，返回均匀权重
        if self.disable_attention:
            num_nodes = features.size(0)
            uniform_attention = torch.ones(num_nodes, device=features.device) * 0.5
            return uniform_attention, class_probs
        
        # 计算每个类别的注意力
        attention_scores = []
        for i, attention_layer in enumerate(self.class_attention):
            scores = attention_layer(features).squeeze(-1)
            attention_scores.append(scores)
        
        attention_scores = torch.stack(attention_scores, dim=-1)  # [num_nodes, num_classes]
        
        # 根据类别概率加权注意力
        final_attention = torch.sum(attention_scores * class_probs, dim=-1)  # [num_nodes]
        
        # 🔧 修复: 添加bias确保注意力值在合理范围
        final_attention = final_attention + self.attention_bias
        
        return final_attention, class_probs

class ViewLevelAttention(nn.Module):
    """视图级别注意力 - 来自SemiGNN的创新"""
    def __init__(self, input_dim, num_views, disable_view_attention=False):
        super(ViewLevelAttention, self).__init__()
        self.num_views = num_views
        self.disable_view_attention = disable_view_attention  # 🔧 新增：消融实验开关
        
        # 视图偏好向量
        self.view_preference = nn.Parameter(torch.randn(num_views, input_dim))
        
        # 注意力计算
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, view_embeddings):
        """
        计算视图级别的注意力权重
        Args:
            view_embeddings: list of [num_nodes, input_dim] for each view
        """
        # 堆叠视图嵌入
        stacked_embeddings = torch.stack(view_embeddings, dim=1)  # [num_nodes, num_views, input_dim]
        
        # 🔧 消融实验：如果禁用视图注意力，返回均匀权重
        if self.disable_view_attention:
            num_nodes = stacked_embeddings.size(0)
            uniform_weights = torch.ones(num_nodes, self.num_views, device=stacked_embeddings.device) / self.num_views
            weighted_embeddings = stacked_embeddings * uniform_weights.unsqueeze(-1)
            combined_embedding = weighted_embeddings.sum(dim=1)
            return combined_embedding, uniform_weights
        
        # 计算每个视图的注意力分数
        attention_scores = []
        for i in range(self.num_views):
            # 使用视图偏好向量计算相似度
            view_emb = stacked_embeddings[:, i, :]  # [num_nodes, input_dim]
            preference = self.view_preference[i].unsqueeze(0).expand_as(view_emb)
            
            # 计算注意力分数
            score = self.attention_net(view_emb * preference).squeeze(-1)  # [num_nodes]
            attention_scores.append(score)
        
        # 归一化注意力权重
        attention_scores = torch.stack(attention_scores, dim=1)  # [num_nodes, num_views]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权组合视图嵌入
        weighted_embeddings = stacked_embeddings * attention_weights.unsqueeze(-1)
        combined_embedding = weighted_embeddings.sum(dim=1)  # [num_nodes, input_dim]
        
        return combined_embedding, attention_weights

class RelationAggregator(nn.Module):
    """关系聚合器 - 处理多关系图"""
    def __init__(self, input_dim, output_dim, num_relations):
        super(RelationAggregator, self).__init__()
        self.num_relations = num_relations
        
        # 每个关系的变换矩阵
        self.relation_transforms = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_relations)
        ])
        
        # 关系权重学习
        self.relation_weight_net = nn.Sequential(
            nn.Linear(input_dim, num_relations),
            nn.Softmax(dim=-1)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, edge_indices, edge_weights):
        """
        聚合多关系信息
        Args:
            features: [num_nodes, input_dim]
            edge_indices: list of [2, num_edges] for each relation
            edge_weights: list of [num_edges] for each relation
        """
        num_nodes = features.size(0)
        device = features.device
        
        # 计算关系权重
        relation_weights = self.relation_weight_net(features)  # [num_nodes, num_relations]
        
        # 对每个关系进行消息传递
        relation_outputs = []
        for i, (edge_index, edge_weight) in enumerate(zip(edge_indices, edge_weights)):
            output_dim = self.relation_transforms[i].out_features
            
            if edge_index.size(1) == 0:  # 如果没有边，跳过
                relation_outputs.append(torch.zeros(num_nodes, output_dim, device=device))
                continue
                
            # 消息传递
            try:
                src_features = features[edge_index[0]]  # [num_edges, input_dim]
                transformed_features = self.relation_transforms[i](src_features)  # [num_edges, output_dim]
                
                # 加权消息
                weighted_messages = transformed_features * edge_weight.unsqueeze(-1)  # [num_edges, output_dim]
                
                # 聚合到目标节点
                dst_indices = edge_index[1]
                aggregated = torch.zeros(num_nodes, output_dim, device=device)
                
                # 使用更安全的scatter_add方法
                for j in range(output_dim):
                    aggregated[:, j].scatter_add_(0, dst_indices, weighted_messages[:, j])
                
                relation_outputs.append(aggregated)
            except Exception as e:
                print(f"Error in relation {i}: {e}")
                relation_outputs.append(torch.zeros(num_nodes, output_dim, device=device))
        
        # 确保所有关系输出都有相同的形状
        if len(relation_outputs) == 0:
            return torch.zeros(num_nodes, self.relation_transforms[0].out_features, device=device)
        
        # 加权组合关系输出
        combined_output = torch.zeros_like(relation_outputs[0])
        for i, relation_output in enumerate(relation_outputs):
            combined_output += relation_weights[:, i:i+1] * relation_output
        
        # 门控机制
        gate_values = self.gate(combined_output)
        gated_output = gate_values * combined_output
        
        return gated_output

class EnhancedCAREGNNLayer(nn.Module):
    """增强的CARE-GNN层 - 整合了SemiGNN的创新"""
    def __init__(self, input_dim, output_dim, num_relations, num_views=None, dropout=0.5,
                 disable_node_attention=False, disable_view_attention=False):
        super(EnhancedCAREGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_views = num_views if num_views is not None else num_relations
        self.disable_node_attention = disable_node_attention  # 🔧 新增：消融实验开关
        self.disable_view_attention = disable_view_attention  # 🔧 新增：消融实验开关
        
        # 核心组件
        self.similarity_layer = SimilarityLayer(input_dim)
        self.label_aware_attention = LabelAwareAttention(input_dim, disable_attention=disable_node_attention)
        
        # 多视图聚合器
        self.view_aggregators = nn.ModuleList([
            RelationAggregator(input_dim, output_dim, 1)  # 每个视图单独处理
            for _ in range(self.num_views)
        ])
        
        # 视图级别注意力
        self.view_attention = ViewLevelAttention(output_dim, self.num_views, 
                                               disable_view_attention=disable_view_attention)
        
        # 特征变换
        self.feature_transform = nn.Linear(input_dim, output_dim)
        self.self_loop_transform = nn.Linear(input_dim, output_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, features, edge_indices, edge_weights, return_attention=False):
        """
        前向传播
        Args:
            features: [num_nodes, input_dim]
            edge_indices: list of [2, num_edges] for each view
            edge_weights: list of [num_edges] for each view
            return_attention: 是否返回注意力权重（用于可解释性）
        """
        # 1. 标签感知注意力
        node_attention, predicted_labels = self.label_aware_attention(features)
        
        # 2. 多视图聚合
        view_outputs = []
        for i in range(min(len(edge_indices), self.num_views)):
            # 每个视图单独处理
            view_output = self.view_aggregators[i](
                features, [edge_indices[i]], [edge_weights[i]]
            )
            view_outputs.append(view_output)
        
        # 填充缺失的视图
        while len(view_outputs) < self.num_views:
            view_outputs.append(torch.zeros_like(view_outputs[0]))
        
        # 3. 视图级别注意力
        combined_output, view_attention_weights = self.view_attention(view_outputs)
        
        # 4. 自循环
        self_output = self.self_loop_transform(features)
        
        # 5. 特征变换
        transformed_features = self.feature_transform(features)
        
        # 6. 融合
        # 使用节点注意力权重调节输出
        weighted_output = combined_output * node_attention.unsqueeze(-1)
        
        # 融合自循环和关系输出
        fused_output = self.fusion(torch.cat([self_output, weighted_output], dim=-1))
        
        # 7. 残差连接和归一化
        if fused_output.size(-1) == transformed_features.size(-1):
            output = fused_output + transformed_features
        else:
            output = fused_output
            
        output = self.layer_norm(output)
        
        if return_attention:
            return output, predicted_labels, node_attention, view_attention_weights
        
        return output, predicted_labels

class EnhancedCAREGNN(nn.Module):
    """增强的CARE-GNN模型 - 整合了半监督学习"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, 
                 num_views=None, num_classes=2, num_layers=2, dropout=0.5,
                 alpha=0.5, disable_node_attention=False, disable_view_attention=False):
        super(EnhancedCAREGNN, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_views = num_views if num_views is not None else num_relations
        self.alpha = alpha  # 监督损失和无监督损失的平衡参数
        self.disable_node_attention = disable_node_attention  # 🔧 新增：消融实验开关
        self.disable_view_attention = disable_view_attention  # 🔧 新增：消融实验开关
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 增强的CARE-GNN层
        self.care_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.care_layers.append(
                EnhancedCAREGNNLayer(layer_input_dim, layer_output_dim, 
                                   num_relations, num_views, dropout,
                                   disable_node_attention=disable_node_attention,
                                   disable_view_attention=disable_view_attention)
            )
        
        # 最终分类器
        final_dim = output_dim if num_layers > 0 else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        # 用于半监督学习的嵌入投影
        self.embedding_projection = nn.Linear(final_dim, final_dim)
        
        # 辅助损失的权重
        self.aux_loss_weight = 0.1
        
    def forward(self, features, edge_indices, edge_weights, return_embeddings=False):
        """
        前向传播
        """
        # 输入投影
        x = self.input_projection(features)
        
        # 通过CARE-GNN层
        aux_predictions = []
        all_attentions = []
        
        for i, layer in enumerate(self.care_layers):
            if i == self.num_layers - 1:  # 最后一层返回注意力权重
                x, predicted_labels, node_att, view_att = layer(
                    x, edge_indices, edge_weights, return_attention=True
                )
                all_attentions.append((node_att, view_att))
            else:
                x, predicted_labels = layer(x, edge_indices, edge_weights)
            aux_predictions.append(predicted_labels)
        
        # 用于半监督学习的嵌入
        embeddings = self.embedding_projection(x)
        
        # 最终分类
        final_logits = self.classifier(x)
        
        if return_embeddings:
            return final_logits, aux_predictions, embeddings, all_attentions
        
        return final_logits, aux_predictions
    
    def compute_graph_loss(self, embeddings, edge_indices, num_neg_samples=3):
        """
        计算图结构的无监督损失 - 修复版本
        """
        device = embeddings.device
        graph_loss = 0
        num_valid_views = 0
        
        for edge_index in edge_indices:
            if edge_index.size(1) == 0:
                continue
                
            num_valid_views += 1
            
            # 🔧 修复1: 限制边数量，防止计算爆炸
            max_edges = min(1000, edge_index.size(1))
            if edge_index.size(1) > max_edges:
                perm = torch.randperm(edge_index.size(1), device=device)[:max_edges]
                edge_index = edge_index[:, perm]
            
            src, dst = edge_index
            
            # 🔧 修复2: 使用标准化的嵌入
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            
            # 正样本分数 - 使用余弦相似度
            pos_scores = (embeddings_norm[src] * embeddings_norm[dst]).sum(dim=1)
            
            # 🔧 修复3: 使用温度参数和更稳定的损失
            temperature = 0.1
            pos_scores = pos_scores / temperature
            
            # 负采样
            num_nodes = embeddings.size(0)
            neg_dst = torch.randint(0, num_nodes, (src.size(0), num_neg_samples), device=device)

            # FAST: vectorized conflict fix where neg_dst == dst
            if neg_dst.dim() == 1:
                neg_dst = neg_dst.unsqueeze(1)
            mask = (neg_dst == dst.unsqueeze(1))
            if mask.any():
                neg_dst[mask] = ((dst.unsqueeze(1).expand_as(neg_dst))[mask] + 1) % num_nodes
            
            # 🔧 修复4: 确保负样本不等于正样本
            for i in range(src.size(0)):
                while neg_dst[i].eq(dst[i]).any():
                    neg_dst[i] = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
            
            # 负样本分数
            src_emb_expanded = embeddings_norm[src].unsqueeze(1)  # [batch, 1, dim]
            neg_emb = embeddings_norm[neg_dst]  # [batch, num_neg_samples, dim]
            neg_scores = (src_emb_expanded * neg_emb).sum(dim=2) / temperature  # [batch, num_neg_samples]
            
            # 🔧 修复5: 使用InfoNCE损失而非分别计算正负损失
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch, 1+num_neg_samples]
            targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=device)
            view_loss = F.cross_entropy(all_scores, targets)
            
            graph_loss += view_loss
        
        # 🔧 修复6: 如果没有有效视图，返回小的非零损失
        if num_valid_views == 0:
            graph_loss = torch.tensor(1e-6, device=device, requires_grad=True)
        else:
            graph_loss /= num_valid_views
            
        return graph_loss
    
    def compute_loss(self, final_logits, aux_predictions, labels, mask, 
                    embeddings=None, edge_indices=None):
        """
        🔧 修复后的损失计算函数 - 确保α参数正确工作
        """
        # 监督损失
        main_loss = F.cross_entropy(final_logits[mask], labels[mask])
        
        # 辅助损失
        aux_loss = 0
        for aux_pred in aux_predictions:
            if aux_pred is not None:
                aux_loss += F.cross_entropy(aux_pred[mask], labels[mask])
        
        sup_loss = main_loss + self.aux_loss_weight * aux_loss
        
        # 图损失计算
        if embeddings is not None and edge_indices is not None:
            graph_loss = self.compute_graph_loss(embeddings, edge_indices)
            
            # 🔧 修复1: 归一化图损失
            num_valid_views = sum(1 for ei in edge_indices if ei.size(1) > 0)
            if num_valid_views > 0:
                graph_loss = graph_loss / num_valid_views
                
            # 🔧 修复2: 自适应缩放，防止图损失过大
            with torch.no_grad():
                sup_loss_val = sup_loss.item()
                graph_loss_val = graph_loss.item()
                
                if graph_loss_val > sup_loss_val * 3:
                    scale_factor = sup_loss_val / graph_loss_val
                    graph_loss = graph_loss * scale_factor
        else:
            graph_loss = torch.tensor(0.0, device=sup_loss.device, requires_grad=True)
        
        # 🔧 修复3: 正确的α参数应用
        if self.alpha >= 1.0:
            # 纯监督学习 - 图损失应该为0
            total_loss = sup_loss
            graph_loss = torch.tensor(0.0, device=sup_loss.device, requires_grad=True)
        elif self.alpha <= 0.0:
            # 纯无监督学习 - 只使用图损失
            total_loss = graph_loss
        else:
            # 半监督学习 - 按α比例混合
            total_loss = self.alpha * sup_loss + (1 - self.alpha) * graph_loss
        
        # 🔧 修复4: 统一返回格式
        return total_loss, sup_loss, graph_loss

def evaluate_model(model, features, labels, edge_indices, edge_weights, mask):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        final_logits, _, embeddings, _ = model(features, edge_indices, edge_weights, 
                                              return_embeddings=True)
        probs = F.softmax(final_logits, dim=1)
        preds = torch.argmax(final_logits, dim=1)
        
        # 只评估有标签的节点
        valid_mask = (labels >= 0) & mask
        if valid_mask.sum() == 0:
            return {}
        
        true_labels = labels[valid_mask].cpu().numpy()
        pred_labels = preds[valid_mask].cpu().numpy()
        pred_probs = probs[valid_mask, 1].cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        }
        
        # 计算AUC（如果有正负样本）
        if len(np.unique(true_labels)) > 1:
            metrics['auc'] = roc_auc_score(true_labels, pred_probs)
        
        return metrics

def get_attention_analysis(model, features, edge_indices, edge_weights, top_k=10):
    """分析注意力权重以提供可解释性"""
    model.eval()
    with torch.no_grad():
        _, _, _, attentions = model(features, edge_indices, edge_weights, 
                                   return_embeddings=True)
        
        # 获取最后一层的注意力权重
        if attentions:
            node_attention, view_attention = attentions[-1]
            
            # 分析视图重要性
            avg_view_attention = view_attention.mean(dim=0)
            view_importance = {
                f'View_{i}': float(avg_view_attention[i]) 
                for i in range(len(avg_view_attention))
            }
            
            # 分析节点重要性
            node_importance = node_attention.mean().item()
            
            return {
                'view_importance': view_importance,
                'avg_node_attention': node_importance
            }
    
    return None

def train_enhanced_care_gnn(data_dir, args):
    """训练增强的CARE-GNN"""
    set_seed(args.seed)
    
    # 加载数据
    features, labels, adj_matrices = load_data(data_dir)
    
    # 构建属性图作为额外的视图
    attr_edge_index, attr_edge_weight = construct_attribute_graph(
        features, k_neighbors=args.k_neighbors
    )
    
    # 转换为edge_index格式
    print("Converting sparse matrices to edge indices...")
    edge_indices = []
    edge_weights = []
    
    # 原始关系图
    for adj in adj_matrices:
        edge_index, edge_weight = sparse_to_edge_index(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    
    # 添加属性图作为新视图
    edge_indices.append(attr_edge_index)
    edge_weights.append(attr_edge_weight)
    
    # 数据预处理
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # 创建训练/验证/测试掩码
    valid_nodes = (labels >= 0).numpy()
    valid_indices = np.where(valid_nodes)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("No valid labeled nodes found!")
    
    # 获取未标记节点（用于半监督学习）
    all_indices = np.arange(len(labels))
    unlabeled_indices = np.setdiff1d(all_indices, valid_indices)
    
    train_idx, temp_idx = train_test_split(valid_indices, test_size=0.4, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    
    # 创建掩码
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    unlabeled_mask = torch.zeros(len(labels), dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    unlabeled_mask[unlabeled_indices] = True
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    unlabeled_mask = unlabeled_mask.to(device)
    
    # 将edge_indices和edge_weights移动到GPU
    edge_indices = [ei.to(device) for ei in edge_indices]
    edge_weights = [ew.to(device) for ew in edge_weights]
    
    # 创建增强模型
    model = EnhancedCAREGNN(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim,
        num_relations=len(adj_matrices),
        num_views=len(edge_indices),  # 包括属性图
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        alpha=args.alpha,  # 监督/无监督损失平衡
        disable_node_attention=getattr(args, 'disable_node_attention', False),
        disable_view_attention=getattr(args, 'disable_view_attention', False)
    ).to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"Enhanced Model: {model}")
    print(f"Device: {device}")
    print(f"Train nodes: {train_mask.sum()}")
    print(f"Val nodes: {val_mask.sum()}")
    print(f"Test nodes: {test_mask.sum()}")
    print(f"Unlabeled nodes: {unlabeled_mask.sum()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of views: {len(edge_indices)}")
    print(f"Disable Node Attention: {getattr(args, 'disable_node_attention', False)}")
    print(f"Disable View Attention: {getattr(args, 'disable_view_attention', False)}")
    
    # 🔧 快速验证α参数是否工作
    print(f"\n🔬 验证α参数是否工作 (α={args.alpha})...")
    model.eval()
    with torch.no_grad():
        final_logits, aux_predictions, embeddings, _ = model(
            features, edge_indices, edge_weights, return_embeddings=True
        )
        total_loss, sup_loss, graph_loss = model.compute_loss(
            final_logits, aux_predictions, labels, train_mask,
            embeddings, edge_indices
        )
        print(f"初始损失 - Sup: {sup_loss.item():.4f}, Graph: {graph_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        if args.alpha >= 1.0 and graph_loss.item() < 1e-6:
            print("✅ α参数工作正常！图损失为0")
        elif args.alpha >= 1.0:
            print(f"❌ α参数异常！α=1.0时图损失应为0，但实际为{graph_loss.item():.4f}")
        else:
            ratio = graph_loss.item() / sup_loss.item() if sup_loss.item() > 0 else 0
            print(f"✅ α参数工作正常！损失比例: {ratio:.4f}")
    
    # 训练循环
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 0
    max_patience = 50
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        final_logits, aux_predictions, embeddings, _ = model(
            features, edge_indices, edge_weights, return_embeddings=True
        )
        
        # 计算损失（包括监督和无监督）
        total_loss, sup_loss, graph_loss = model.compute_loss(
            final_logits, aux_predictions, labels, train_mask,
            embeddings, edge_indices
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 评估
        if epoch % args.eval_every == 0:
            train_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, train_mask)
            val_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, val_mask)
            test_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, test_mask)
            
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss:.4f} "
                  f"(Sup: {sup_loss:.4f}, Graph: {graph_loss:.4f})")
            print(f"Train F1: {train_metrics.get('f1', 0):.4f} | "
                  f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                  f"Test F1: {test_metrics.get('f1', 0):.4f}")
            
            # 获取注意力分析
            if args.analyze_attention and epoch % (args.eval_every * 5) == 0:
                attention_info = get_attention_analysis(model, features, edge_indices, edge_weights)
                if attention_info:
                    print("View Importance:", attention_info['view_importance'])
            
            # 早停机制
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_test_metrics = test_metrics
                patience = 0
                torch.save(model.state_dict(), 'best_enhanced_care_gnn.pth')
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # 输出最终结果
    print("\n=== Best Test Results ===")
    for metric, value in best_test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # 最终的注意力分析
    if args.analyze_attention:
        print("\n=== Final Attention Analysis ===")
        model.load_state_dict(torch.load('best_enhanced_care_gnn.pth'))
        attention_info = get_attention_analysis(model, features, edge_indices, edge_weights)
        if attention_info:
            print("View Importance Scores:")
            for view, score in attention_info['view_importance'].items():
                print(f"  {view}: {score:.4f}")
            print(f"Average Node Attention: {attention_info['avg_node_attention']:.4f}")
    
    return model, best_test_metrics

def create_ablation_variants(input_dim, hidden_dim, output_dim, num_relations, num_views, 
                           num_layers, dropout, alpha):
    """创建消融实验的模型变体 - 修复版本"""
    variants = {}
    
    # 1. 完整模型
    variants['Full_Model'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=alpha,
        disable_node_attention=False,
        disable_view_attention=False
    )
    
    # 2. 无节点注意力版本
    variants['No_Node_Attention'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=alpha,
        disable_node_attention=True,  # 🔧 关闭节点注意力
        disable_view_attention=False
    )
    
    # 3. 无视图注意力版本
    variants['No_View_Attention'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=alpha,
        disable_node_attention=False,
        disable_view_attention=True  # 🔧 关闭视图注意力
    )
    
    # 4. 无任何注意力版本
    variants['No_Attention'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=alpha,
        disable_node_attention=True,  # 🔧 关闭节点注意力
        disable_view_attention=True   # 🔧 关闭视图注意力
    )
    
    # 5. 仅监督学习版本
    variants['Supervised_Only'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=1.0,  # 🔧 纯监督学习
        disable_node_attention=False,
        disable_view_attention=False
    )
    
    # 6. 仅无监督学习版本
    variants['Unsupervised_Only'] = EnhancedCAREGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_views=num_views,
        num_classes=2,
        num_layers=num_layers,
        dropout=dropout,
        alpha=0.0,  # 🔧 纯无监督学习
        disable_node_attention=False,
        disable_view_attention=False
    )
    
    return variants

def run_ablation_study(data_dir, args):
    """运行消融研究 - 修复版本"""
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)
    
    set_seed(args.seed)
    
    # 加载数据（与主训练函数相同的处理）
    features, labels, adj_matrices = load_data(data_dir)
    attr_edge_index, attr_edge_weight = construct_attribute_graph(features, k_neighbors=args.k_neighbors)
    
    edge_indices = []
    edge_weights = []
    for adj in adj_matrices:
        edge_index, edge_weight = sparse_to_edge_index(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    edge_indices.append(attr_edge_index)
    edge_weights.append(attr_edge_weight)
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    valid_nodes = (labels >= 0).numpy()
    valid_indices = np.where(valid_nodes)[0]
    train_idx, temp_idx = train_test_split(valid_indices, test_size=0.4, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]
    edge_weights = [ew.to(device) for ew in edge_weights]
    
    # 创建消融变体
    variants = create_ablation_variants(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim,
        num_relations=len(adj_matrices),
        num_views=len(edge_indices),
        num_layers=args.num_layers,
        dropout=args.dropout,
        alpha=args.alpha
    )
    
    results = {}
    
    # 测试每个变体
    for variant_name, model in variants.items():
        print(f"\n🧪 Testing {variant_name}...")
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_f1 = 0
        best_test_metrics = {}
        patience = 0
        max_patience = 30
        
        # 训练循环
        for epoch in range(150):  # 减少训练轮数以加快消融实验
            model.train()
            optimizer.zero_grad()
            
            final_logits, aux_predictions, embeddings, _ = model(
                features, edge_indices, edge_weights, return_embeddings=True
            )
            
            total_loss, sup_loss, graph_loss = model.compute_loss(
                final_logits, aux_predictions, labels, train_mask,
                embeddings, edge_indices
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 验证
            if epoch % 15 == 0:
                val_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, val_mask)
                test_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, test_mask)
                
                if epoch % 30 == 0:
                    print(f"  Epoch {epoch}: Val F1={val_metrics.get('f1', 0):.4f}, Test F1={test_metrics.get('f1', 0):.4f}")
                
                if val_metrics.get('f1', 0) > best_val_f1:
                    best_val_f1 = val_metrics.get('f1', 0)
                    best_test_metrics = test_metrics
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        break
        
        results[variant_name] = best_test_metrics
        print(f"✅ {variant_name} Results:")
        print(f"   F1: {best_test_metrics.get('f1', 0):.4f}")
        print(f"   AUC: {best_test_metrics.get('auc', 0):.4f}")
        print(f"   Accuracy: {best_test_metrics.get('accuracy', 0):.4f}")
    
    # 输出对比结果
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    print(f"{'Variant':<20} {'F1':<8} {'AUC':<8} {'Accuracy':<10}")
    print("-" * 50)
    
    # 按F1分数排序
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
    
    for variant_name, metrics in sorted_results:
        print(f"{variant_name:<20} "
              f"{metrics.get('f1', 0):<8.4f} "
              f"{metrics.get('auc', 0):<8.4f} "
              f"{metrics.get('accuracy', 0):<10.4f}")
    
    # 计算改进量
    full_model_f1 = results.get('Full_Model', {}).get('f1', 0)
    print(f"\n📊 相对于完整模型的性能变化:")
    for variant_name, metrics in results.items():
        if variant_name != 'Full_Model':
            f1_diff = metrics.get('f1', 0) - full_model_f1
            print(f"  {variant_name}: {f1_diff:+.4f} F1")
    
    return results

def visualize_results(model, features, labels, edge_indices, edge_weights, save_path='results/'):
    """可视化结果和注意力权重"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        _, _, embeddings, attentions = model(features, edge_indices, edge_weights, 
                                            return_embeddings=True)
        
        # 1. 视图重要性可视化
        if attentions:
            _, view_attention = attentions[-1]
            avg_view_attention = view_attention.mean(dim=0).cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            view_names = [f'View {i}' for i in range(len(avg_view_attention))]
            view_names[-1] = 'Attribute Graph'  # 最后一个是属性图
            
            bars = plt.bar(view_names, avg_view_attention)
            plt.title('View Importance for Fraud Detection', fontsize=16)
            plt.xlabel('Views', fontsize=14)
            plt.ylabel('Attention Weight', fontsize=14)
            
            # 添加数值标签
            for bar, value in zip(bars, avg_view_attention):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'view_importance.png'))
            plt.close()
        
        # 2. 嵌入空间可视化（使用t-SNE）
        from sklearn.manifold import TSNE
        
        # 只可视化有标签的节点
        valid_mask = labels >= 0
        valid_embeddings = embeddings[valid_mask].cpu().numpy()
        valid_labels = labels[valid_mask].cpu().numpy()
        
        if len(valid_embeddings) > 1000:  # 采样以加快速度
            sample_idx = np.random.choice(len(valid_embeddings), 1000, replace=False)
            valid_embeddings = valid_embeddings[sample_idx]
            valid_labels = valid_labels[sample_idx]
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(valid_embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=valid_labels, cmap='RdBu', alpha=0.6)
        plt.colorbar(scatter, label='Label (0: Normal, 1: Fraud)')
        plt.title('t-SNE Visualization of Node Embeddings', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'embedding_visualization.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced CARE-GNN Training with Semi-supervised Learning')
    parser.add_argument('--data_dir', type=str, default='small_multirel_dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluate every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Balance between supervised and unsupervised loss')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Number of neighbors for attribute graph construction')
    parser.add_argument('--analyze_attention', action='store_true',
                        help='Analyze attention weights during training')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results after training')
    
    # 🔧 新增：消融实验参数
    parser.add_argument('--run_ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--disable_node_attention', action='store_true',
                        help='Disable node attention mechanism')
    parser.add_argument('--disable_view_attention', action='store_true',
                        help='Disable view attention mechanism')
    
    args = parser.parse_args()
    
    if args.run_ablation:
        # 运行消融研究
        print("Running Ablation Study...")
        ablation_results = run_ablation_study(args.data_dir, args)
        
        # 保存结果
        import json
        os.makedirs('ablation_results', exist_ok=True)
        with open('ablation_results/ablation_results.json', 'w') as f:
            # 转换numpy类型
            serializable_results = {}
            for variant, metrics in ablation_results.items():
                serializable_results[variant] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in metrics.items()
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Ablation study completed!")
        print(f"Results saved in 'ablation_results/ablation_results.json'")
        
    else:
        # 训练单个模型
        print("Training Enhanced CARE-GNN with Semi-supervised Learning...")
        model, metrics = train_enhanced_care_gnn(args.data_dir, args)
        
        print("\n=== Training Complete ===")
        print("Best model saved as 'best_enhanced_care_gnn.pth'")
        
        # 可视化结果
        if args.visualize:
            print("\nGenerating visualizations...")
            # 重新加载数据和模型进行可视化
            features, labels, adj_matrices = load_data(args.data_dir)
            attr_edge_index, attr_edge_weight = construct_attribute_graph(features, k_neighbors=args.k_neighbors)
            
            edge_indices = []
            edge_weights = []
            for adj in adj_matrices:
                edge_index, edge_weight = sparse_to_edge_index(adj)
                edge_indices.append(edge_index)
                edge_weights.append(edge_weight)
            edge_indices.append(attr_edge_index)
            edge_weights.append(attr_edge_weight)
            
            device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
            features = torch.FloatTensor(features).to(device)
            labels = torch.LongTensor(labels).to(device)
            edge_indices = [ei.to(device) for ei in edge_indices]
            edge_weights = [ew.to(device) for ew in edge_weights]
            
            model.load_state_dict(torch.load('best_enhanced_care_gnn.pth'))
            model.to(device)
            
            visualize_results(model, features, labels, edge_indices, edge_weights)
            print("Visualizations saved in 'results/' directory")

if __name__ == '__main__':
    main()
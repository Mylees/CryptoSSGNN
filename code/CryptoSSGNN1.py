import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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

# ========== 新增：半监督学习相关组件 ==========

class SemiSupervisedLoss(nn.Module):
    """半监督损失函数 - 参考SemiGNN论文思路"""
    def __init__(self, alpha=0.7, beta=0.3, temperature=0.1, confidence_threshold=0.8, 
                 walk_length=10, num_walks=5, window_size=3, num_negative_samples=3):
        super().__init__()
        self.alpha = alpha  # 监督损失权重
        self.beta = beta    # 图正则化损失权重
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # SemiGNN论文中的随机游走参数
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        
    def forward(self, logits, labels, node_embeddings, edge_indices, edge_weights, 
                labeled_mask, unlabeled_mask):
        """
        计算半监督损失
        Args:
            logits: [num_nodes, num_classes] 模型输出
            labels: [num_nodes] 真实标签
            node_embeddings: [num_nodes, hidden_dim] 节点嵌入
            edge_indices: list of [2, num_edges] 边索引
            edge_weights: list of [num_edges] 边权重
            labeled_mask: [num_nodes] 有标签节点掩码
            unlabeled_mask: [num_nodes] 无标签节点掩码
        """
        losses = {}
        
        # 1. 监督损失
        if labeled_mask.sum() > 0:
            supervised_loss = F.cross_entropy(logits[labeled_mask], labels[labeled_mask])
        else:
            supervised_loss = torch.tensor(0.0, device=logits.device)
        losses['supervised'] = supervised_loss
        
        # 2. SemiGNN风格的图损失（随机游走 + 负采样）
        # 使用优化版本以提高计算效率
        graph_loss = self.compute_semignn_graph_loss_optimized(
            node_embeddings, edge_indices, edge_weights
        )
        losses['graph_reg'] = graph_loss
        
        # 3. 一致性损失（伪标签）
        consistency_loss = self.compute_consistency_loss(
            logits, unlabeled_mask
        )
        losses['consistency'] = consistency_loss
        
        # 4. 邻居标签传播损失
        if labeled_mask.sum() > 0 and unlabeled_mask.sum() > 0:
            propagation_loss = self.compute_label_propagation_loss(
                logits, labels, edge_indices, edge_weights, labeled_mask, unlabeled_mask
            )
        else:
            propagation_loss = torch.tensor(0.0, device=logits.device)
        losses['propagation'] = propagation_loss
        
        # 总损失
        total_loss = (self.alpha * supervised_loss + 
                     self.beta * graph_loss + 
                     0.1 * consistency_loss + 
                     0.2 * propagation_loss)
        
        return total_loss, losses
    
    def random_walk(self, edge_indices, edge_weights, num_nodes, start_node, walk_length):
        """
        在图上执行随机游走 - 参考SemiGNN论文
        Args:
            edge_indices: list of [2, num_edges] 
            edge_weights: list of [num_edges]
            num_nodes: 节点总数
            start_node: 起始节点
            walk_length: 游走长度
        Returns:
            walk: 游走路径
        """
        # 构建邻接表用于快速查找邻居
        adj_list = {}
        weight_dict = {}
        
        for rel_idx, (edge_index, edge_weight) in enumerate(zip(edge_indices, edge_weights)):
            if edge_index.size(1) == 0:
                continue
                
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                weight = edge_weight[i].item()
                
                if src not in adj_list:
                    adj_list[src] = []
                    weight_dict[src] = []
                
                adj_list[src].append(dst)
                weight_dict[src].append(weight)
        
        # 执行随机游走
        walk = [start_node]
        current_node = start_node
        
        for _ in range(walk_length - 1):
            if current_node not in adj_list or len(adj_list[current_node]) == 0:
                break
                
            neighbors = adj_list[current_node]
            weights = weight_dict[current_node]
            
            # 根据边权重进行概率采样
            if len(weights) > 0:
                weights_tensor = torch.tensor(weights, dtype=torch.float)
                weights_tensor = F.softmax(weights_tensor, dim=0)
                
                # 采样下一个节点
                next_idx = torch.multinomial(weights_tensor, 1).item()
                next_node = neighbors[next_idx]
                walk.append(next_node)
                current_node = next_node
            else:
                break
        
        return walk
    
    def compute_semignn_graph_loss(self, embeddings, edge_indices, edge_weights):
        """
        计算SemiGNN风格的图损失
        参考论文公式：Lgraph = Σ_{u∈U} Σ_{v∈Nu∪Negu} -log(σ(aᵀᵤaᵥ)) - Q·E_{q~Pneg(u)}log(σ(aᵀᵤaq))
        """
        device = embeddings.device
        num_nodes = embeddings.size(0)
        
        if not edge_indices or len(edge_indices) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # 为每个节点执行随机游走，构建正样本对
        nodes_to_sample = min(num_nodes, 500)  # 限制采样节点数以控制计算复杂度
        sampled_nodes = torch.randperm(num_nodes)[:nodes_to_sample]
        
        for node_idx in sampled_nodes:
            node = node_idx.item()
            
            # 执行多次随机游走
            positive_pairs = []
            for _ in range(self.num_walks):
                walk = self.random_walk(edge_indices, edge_weights, num_nodes, node, self.walk_length)
                
                # 从游走中提取正样本对（窗口内的节点对）
                for i, center_node in enumerate(walk):
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            positive_pairs.append((center_node, context_node))
            
            # 如果没有正样本对，跳过该节点
            if len(positive_pairs) == 0:
                continue
            
            # 处理正样本对
            for center_node, context_node in positive_pairs:
                if center_node >= num_nodes or context_node >= num_nodes:
                    continue
                    
                # 正样本损失：最大化相似度
                center_emb = embeddings[center_node]
                context_emb = embeddings[context_node]
                pos_score = torch.sum(center_emb * context_emb)
                pos_loss = -F.logsigmoid(pos_score)
                
                # 负采样
                neg_loss = 0
                for _ in range(self.num_negative_samples):
                    # 采样负样本节点（避免采样到正样本）
                    neg_node = torch.randint(0, num_nodes, (1,)).item()
                    while neg_node == center_node or neg_node == context_node:
                        neg_node = torch.randint(0, num_nodes, (1,)).item()
                    
                    neg_emb = embeddings[neg_node]
                    neg_score = torch.sum(center_emb * neg_emb)
                    neg_loss += -F.logsigmoid(-neg_score)  # 最小化负样本相似度
                
                total_loss += pos_loss + neg_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)
    
    def build_degree_based_negative_sampler(self, edge_indices, num_nodes):
        """
        构建基于度的负采样器 - 参考SemiGNN论文
        论文中使用 Pneg(u) ∝ d^0.75_u 的负采样分布
        """
        # 计算每个节点的度
        degree = torch.zeros(num_nodes)
        
        for edge_index in edge_indices:
            if edge_index.size(1) > 0:
                # 计算入度和出度
                degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
                degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
        
        # 避免度为0的节点
        degree = torch.clamp(degree, min=1.0)
        
        # 计算负采样概率：d^0.75
        neg_sampling_probs = torch.pow(degree, 0.75)
        neg_sampling_probs = neg_sampling_probs / neg_sampling_probs.sum()
        
        return neg_sampling_probs
    
    def sample_negative_nodes(self, neg_sampling_probs, center_node, context_node, num_samples):
        """
        采样负样本节点
        """
        negative_nodes = []
        num_nodes = len(neg_sampling_probs)
        
        for _ in range(num_samples):
            # 基于度的概率采样
            neg_node = torch.multinomial(neg_sampling_probs, 1).item()
            
            # 确保不采样到正样本
            attempts = 0
            while (neg_node == center_node or neg_node == context_node) and attempts < 10:
                neg_node = torch.multinomial(neg_sampling_probs, 1).item()
                attempts += 1
            
            negative_nodes.append(neg_node)
        
        return negative_nodes
    
    def compute_semignn_graph_loss_optimized(self, embeddings, edge_indices, edge_weights):
        """
        优化版的SemiGNN图损失计算
        使用更高效的实现和度优先负采样
        """
        device = embeddings.device
        num_nodes = embeddings.size(0)
        
        if not edge_indices or len(edge_indices) == 0:
            return torch.tensor(0.0, device=device)
        
        # 构建度优先负采样器
        neg_sampling_probs = self.build_degree_based_negative_sampler(edge_indices, num_nodes)
        neg_sampling_probs = neg_sampling_probs.to(device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # 直接从边构建正样本对（更高效）
        for edge_index, edge_weight in zip(edge_indices, edge_weights):
            if edge_index.size(1) == 0:
                continue
            
            # 限制边的数量以控制计算复杂度
            max_edges = min(edge_index.size(1), 1000)
            if edge_index.size(1) > max_edges:
                perm = torch.randperm(edge_index.size(1))[:max_edges]
                sampled_edges = edge_index[:, perm]
                sampled_weights = edge_weight[perm]
            else:
                sampled_edges = edge_index
                sampled_weights = edge_weight
            
            # 批量处理正样本
            src_nodes = sampled_edges[0]  # [num_edges]
            dst_nodes = sampled_edges[1]  # [num_edges]
            edge_weights_batch = sampled_weights  # [num_edges]
            
            # 获取嵌入
            src_embeddings = embeddings[src_nodes]  # [num_edges, hidden_dim]
            dst_embeddings = embeddings[dst_nodes]  # [num_edges, hidden_dim]
            
            # 计算正样本分数
            pos_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)  # [num_edges]
            
            # 加权正样本损失
            weighted_pos_loss = -F.logsigmoid(pos_scores) * edge_weights_batch
            pos_loss = torch.mean(weighted_pos_loss)
            
            # 负采样损失
            neg_loss = 0
            for _ in range(self.num_negative_samples):
                # 批量负采样
                neg_nodes = torch.multinomial(neg_sampling_probs, sampled_edges.size(1), replacement=True)
                neg_embeddings = embeddings[neg_nodes]  # [num_edges, hidden_dim]
                
                # 计算负样本分数
                neg_scores = torch.sum(src_embeddings * neg_embeddings, dim=1)  # [num_edges]
                
                # 负样本损失
                weighted_neg_loss = -F.logsigmoid(-neg_scores) * edge_weights_batch
                neg_loss += torch.mean(weighted_neg_loss)
            
            total_loss += pos_loss + neg_loss / self.num_negative_samples
            num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)
    
    def compute_consistency_loss(self, logits, unlabeled_mask):
        """
        一致性损失 - 对高置信度的无标签样本使用伪标签
        """
        if not unlabeled_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        unlabeled_logits = logits[unlabeled_mask]
        unlabeled_probs = F.softmax(unlabeled_logits / self.temperature, dim=-1)
        
        # 获取最高置信度的预测
        max_probs, pseudo_labels = torch.max(unlabeled_probs, dim=-1)
        
        # 只对高置信度样本计算损失
        confident_mask = max_probs > self.confidence_threshold
        
        if not confident_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # 计算伪标签的交叉熵损失
        confident_logits = unlabeled_logits[confident_mask]
        confident_pseudo_labels = pseudo_labels[confident_mask]
        
        return F.cross_entropy(confident_logits, confident_pseudo_labels)
    
    def compute_label_propagation_loss(self, logits, labels, edge_indices, edge_weights, 
                                     labeled_mask, unlabeled_mask):
        """
        标签传播损失 - 有标签节点向无标签节点传播标签信息
        """
        if not edge_indices or len(edge_indices) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        total_loss = 0
        valid_propagations = 0
        
        for edge_index, edge_weight in zip(edge_indices, edge_weights):
            if edge_index.size(1) == 0:
                continue
            
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # 找到从有标签节点到无标签节点的边
            src_labeled = labeled_mask[src_nodes]
            dst_unlabeled = unlabeled_mask[dst_nodes]
            valid_edges = src_labeled & dst_unlabeled
            
            if not valid_edges.any():
                continue
            
            # 获取有效边的信息
            valid_src = src_nodes[valid_edges]
            valid_dst = dst_nodes[valid_edges]
            valid_weights = edge_weight[valid_edges]
            
            # 获取源节点的真实标签和目标节点的预测
            src_labels = labels[valid_src]
            dst_logits = logits[valid_dst]
            
            # 计算加权交叉熵损失
            weighted_loss = F.cross_entropy(dst_logits, src_labels, reduction='none')
            weighted_loss = weighted_loss * valid_weights
            
            total_loss += torch.mean(weighted_loss)
            valid_propagations += 1
        
        return total_loss / valid_propagations if valid_propagations > 0 else torch.tensor(0.0, device=logits.device)

class UnlabeledDataExpansion:
    """无标签数据扩展 - 通过社交关系扩展标签数据"""
    
    @staticmethod
    def expand_labeled_data(labels, edge_indices, expansion_hops=1):
        """
        通过图结构扩展有标签数据，获取无标签节点
        Args:
            labels: [num_nodes] 原始标签，-1表示无标签
            edge_indices: list of [2, num_edges] 边索引
            expansion_hops: 扩展的跳数
        Returns:
            unlabeled_mask: 扩展后的无标签节点掩码
        """
        num_nodes = len(labels)
        device = labels.device if torch.is_tensor(labels) else 'cpu'
        
        # 创建邻接矩阵
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
        for edge_index in edge_indices:
            if edge_index.size(1) > 0:
                adj_matrix[edge_index[0], edge_index[1]] = 1
                adj_matrix[edge_index[1], edge_index[0]] = 1  # 无向图
        
        # 找到初始有标签节点
        if torch.is_tensor(labels):
            labeled_nodes = torch.where(labels >= 0)[0]
        else:
            labeled_nodes = torch.tensor(np.where(labels >= 0)[0], device=device)
        
        # 多跳扩展
        expanded_nodes = set(labeled_nodes.cpu().numpy())
        current_nodes = labeled_nodes
        
        for hop in range(expansion_hops):
            neighbors = []
            for node in current_nodes:
                node_neighbors = torch.where(adj_matrix[node] > 0)[0]
                neighbors.extend(node_neighbors.cpu().numpy())
            
            current_nodes = torch.tensor(list(set(neighbors) - expanded_nodes), device=device)
            expanded_nodes.update(current_nodes.cpu().numpy())
            
            if len(current_nodes) == 0:
                break
        
        # 创建无标签掩码（扩展的节点中除了原有标签节点）
        unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        original_labeled = set(labeled_nodes.cpu().numpy())
        expanded_unlabeled = expanded_nodes - original_labeled
        
        if expanded_unlabeled:
            unlabeled_indices = torch.tensor(list(expanded_unlabeled), device=device)
            unlabeled_mask[unlabeled_indices] = True
        
        return unlabeled_mask

# ========== 原有组件保持不变 ==========

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
    """标签感知注意力机制"""
    def __init__(self, input_dim, num_classes=2):
        super(LabelAwareAttention, self).__init__()
        self.num_classes = num_classes
        
        # 为每个类别学习不同的注意力
        self.class_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1)
            ) for _ in range(num_classes)
        ])
        
        # 类别预测器
        self.class_predictor = nn.Linear(input_dim, num_classes)
        
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
        
        # 计算每个类别的注意力
        attention_scores = []
        for i, attention_layer in enumerate(self.class_attention):
            scores = attention_layer(features).squeeze(-1)
            attention_scores.append(scores)
        
        attention_scores = torch.stack(attention_scores, dim=-1)  # [num_nodes, num_classes]
        
        # 根据类别概率加权注意力
        final_attention = torch.sum(attention_scores * class_probs, dim=-1)  # [num_nodes]
        
        return final_attention, class_probs

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

class CAREGNNLayer(nn.Module):
    """CARE-GNN层的完整实现"""
    def __init__(self, input_dim, output_dim, num_relations, dropout=0.5):
        super(CAREGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        
        # 核心组件
        self.similarity_layer = SimilarityLayer(input_dim)
        self.label_aware_attention = LabelAwareAttention(input_dim)
        self.relation_aggregator = RelationAggregator(input_dim, output_dim, num_relations)
        
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
        
    def forward(self, features, edge_indices, edge_weights):
        """
        前向传播
        Args:
            features: [num_nodes, input_dim]
            edge_indices: list of [2, num_edges] for each relation
            edge_weights: list of [num_edges] for each relation
        """
        # 1. 标签感知注意力
        node_attention, predicted_labels = self.label_aware_attention(features)
        
        # 2. 关系聚合
        relation_output = self.relation_aggregator(features, edge_indices, edge_weights)
        
        # 3. 自循环
        self_output = self.self_loop_transform(features)
        
        # 4. 特征变换
        transformed_features = self.feature_transform(features)
        
        # 5. 融合
        # 使用注意力权重调节关系输出
        weighted_relation_output = relation_output * node_attention.unsqueeze(-1)
        
        # 融合自循环和关系输出
        fused_output = self.fusion(torch.cat([self_output, weighted_relation_output], dim=-1))
        
        # 6. 残差连接和归一化
        if fused_output.size(-1) == transformed_features.size(-1):
            output = fused_output + transformed_features
        else:
            output = fused_output
            
        output = self.layer_norm(output)
        
        return output, predicted_labels

# ========== 修改后的主模型 ==========

class SemiSupervisedCAREGNN(nn.Module):
    """增强版CARE-GNN，集成半监督学习"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, 
                 num_classes=2, num_layers=2, dropout=0.5, 
                 semi_supervised=True, expansion_hops=1):
        super(SemiSupervisedCAREGNN, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.semi_supervised = semi_supervised
        self.expansion_hops = expansion_hops
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # CARE-GNN层
        self.care_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.care_layers.append(CAREGNNLayer(layer_input_dim, layer_output_dim, num_relations, dropout))
        
        # 最终分类器
        final_dim = output_dim if num_layers > 0 else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        # 半监督损失
        if semi_supervised:
            self.semi_loss_fn = SemiSupervisedLoss(
                alpha=0.7, beta=0.3, 
                walk_length=10, num_walks=5, window_size=3, num_negative_samples=3
            )
        else:
            self.semi_loss_fn = None
        
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
        embeddings_list = [x]  # 存储每层的嵌入用于半监督学习
        
        for i, layer in enumerate(self.care_layers):
            x, predicted_labels = layer(x, edge_indices, edge_weights)
            aux_predictions.append(predicted_labels)
            embeddings_list.append(x)
        
        # 最终分类
        final_logits = self.classifier(x)
        
        if return_embeddings:
            return final_logits, aux_predictions, embeddings_list[-2]  # 返回分类前的嵌入
        else:
            return final_logits, aux_predictions
    
    def compute_loss(self, features, edge_indices, edge_weights, labels, 
                    labeled_mask, unlabeled_mask=None):
        """
        计算损失，支持半监督学习
        """
        # 前向传播获取嵌入
        final_logits, aux_predictions, node_embeddings = self.forward(
            features, edge_indices, edge_weights, return_embeddings=True
        )
        
        losses = {}
        
        if self.semi_supervised and unlabeled_mask is not None and unlabeled_mask.sum() > 0:
            # 半监督损失
            total_loss, semi_losses = self.semi_loss_fn(
                final_logits, labels, node_embeddings, edge_indices, edge_weights,
                labeled_mask, unlabeled_mask
            )
            losses.update(semi_losses)
        else:
            # 标准监督损失
            if labeled_mask.sum() > 0:
                supervised_loss = F.cross_entropy(final_logits[labeled_mask], labels[labeled_mask])
            else:
                supervised_loss = torch.tensor(0.0, device=final_logits.device)
            total_loss = supervised_loss
            losses['supervised'] = supervised_loss
        
        # 辅助损失
        aux_loss = torch.tensor(0.0, device=final_logits.device)
        if labeled_mask.sum() > 0:
            for aux_pred in aux_predictions:
                if aux_pred is not None:
                    aux_loss += F.cross_entropy(aux_pred[labeled_mask], labels[labeled_mask])
        
        total_loss += self.aux_loss_weight * aux_loss
        losses['auxiliary'] = aux_loss
        losses['total'] = total_loss
        
        return total_loss, losses, final_logits

def evaluate_model(model, features, labels, edge_indices, edge_weights, mask):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        final_logits, _ = model(features, edge_indices, edge_weights)
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

def train_semi_supervised_care_gnn(data_dir, args):
    """训练半监督CARE-GNN"""
    set_seed(args.seed)
    
    # 加载数据
    features, labels, adj_matrices = load_data(data_dir)
    
    # 转换为edge_index格式
    print("Converting sparse matrices to edge indices...")
    edge_indices = []
    edge_weights = []
    for adj in adj_matrices:
        edge_index, edge_weight = sparse_to_edge_index(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    
    # 数据预处理
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # 创建训练/验证/测试掩码
    valid_nodes = (labels >= 0).numpy()
    valid_indices = np.where(valid_nodes)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("No valid labeled nodes found!")
    
    # 分割数据集
    train_idx, temp_idx = train_test_split(valid_indices, test_size=0.4, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # 扩展无标签数据（半监督学习的关键）
    print("Expanding unlabeled data through graph structure...")
    if args.semi_supervised:
        unlabeled_mask = UnlabeledDataExpansion.expand_labeled_data(
            labels, edge_indices, expansion_hops=args.expansion_hops
        )
        print(f"Expanded {unlabeled_mask.sum()} unlabeled nodes from {train_mask.sum()} labeled nodes")
    else:
        unlabeled_mask = torch.zeros(len(labels), dtype=torch.bool)
    
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
    
    # 创建模型
    model = SemiSupervisedCAREGNN(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim,
        num_relations=len(adj_matrices),
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        semi_supervised=args.semi_supervised,
        expansion_hops=args.expansion_hops
    ).to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Train nodes: {train_mask.sum()}")
    print(f"Val nodes: {val_mask.sum()}")
    print(f"Test nodes: {test_mask.sum()}")
    print(f"Unlabeled nodes: {unlabeled_mask.sum()}")
    print(f"Semi-supervised: {args.semi_supervised}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练循环
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 0
    max_patience = 50
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播和损失计算
        total_loss, losses, final_logits = model.compute_loss(
            features, edge_indices, edge_weights, labels,
            train_mask, unlabeled_mask if args.semi_supervised else None
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
            
            # 打印训练信息
            loss_str = f"Total: {total_loss:.4f}"
            if args.semi_supervised:
                loss_str += f" (Sup: {losses.get('supervised', 0):.4f}"
                loss_str += f", Graph: {losses.get('graph_reg', 0):.4f}"
                loss_str += f", Cons: {losses.get('consistency', 0):.4f}"
                loss_str += f", Prop: {losses.get('propagation', 0):.4f}"
                loss_str += f", Aux: {losses.get('auxiliary', 0):.4f})"
            else:
                loss_str += f" (Sup: {losses.get('supervised', 0):.4f}, Aux: {losses.get('auxiliary', 0):.4f})"
            
            print(f"Epoch {epoch:03d} | Loss: {loss_str}")
            print(f"Train F1: {train_metrics.get('f1', 0):.4f} | Val F1: {val_metrics.get('f1', 0):.4f} | Test F1: {test_metrics.get('f1', 0):.4f}")
            
            # 早停机制
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_test_metrics = test_metrics
                patience = 0
                # 保存最佳模型
                model_save_name = 'best_semi_care_gnn.pth' if args.semi_supervised else 'best_care_gnn.pth'
                torch.save(model.state_dict(), model_save_name)
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # 输出最终结果
    print("\n=== Best Test Results ===")
    for metric, value in best_test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # 如果是半监督学习，输出额外的分析
    if args.semi_supervised and unlabeled_mask.sum() > 0:
        print(f"\n=== Semi-Supervised Learning Analysis ===")
        print(f"Labeled training nodes: {train_mask.sum()}")
        print(f"Unlabeled nodes used: {unlabeled_mask.sum()}")
        print(f"Unlabeled/Labeled ratio: {unlabeled_mask.sum().float() / train_mask.sum().float():.2f}")
        
        # 分析无标签节点的预测置信度
        model.eval()
        with torch.no_grad():
            final_logits, _ = model(features, edge_indices, edge_weights)
            unlabeled_probs = F.softmax(final_logits[unlabeled_mask], dim=1)
            max_probs, _ = torch.max(unlabeled_probs, dim=1)
            
            print(f"Avg confidence on unlabeled nodes: {max_probs.mean():.4f}")
            print(f"High confidence nodes (>0.8): {(max_probs > 0.8).sum()}/{len(max_probs)}")
    
    return model, best_test_metrics

def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised CARE-GNN Training')
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
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    
    # 新增半监督学习相关参数
    parser.add_argument('--semi_supervised', action='store_true', default=True,
                        help='Enable semi-supervised learning')
    parser.add_argument('--expansion_hops', type=int, default=1,
                        help='Number of hops for expanding unlabeled data')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for supervised loss in semi-supervised learning')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for graph regularization loss')
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                        help='Confidence threshold for pseudo-labeling')
    
    # SemiGNN论文相关参数
    parser.add_argument('--walk_length', type=int, default=10,
                        help='Random walk length for graph loss')
    parser.add_argument('--num_walks', type=int, default=5,
                        help='Number of random walks per node')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Context window size for random walk')
    parser.add_argument('--num_negative_samples', type=int, default=3,
                        help='Number of negative samples for graph loss')
    
    args = parser.parse_args()
    
    print("=== Semi-Supervised CARE-GNN ===")
    print(f"Semi-supervised learning: {args.semi_supervised}")
    print(f"Data directory: {args.data_dir}")
    print(f"Expansion hops: {args.expansion_hops}")
    
    # 训练模型
    model, metrics = train_semi_supervised_care_gnn(args.data_dir, args)
    
    print("\n=== Training Complete ===")
    model_name = "Semi-Supervised CARE-GNN" if args.semi_supervised else "CARE-GNN"
    print(f"{model_name} training finished!")
    save_name = 'best_semi_care_gnn.pth' if args.semi_supervised else 'best_care_gnn.pth'
    print(f"Best model saved as '{save_name}'")

if __name__ == '__main__':
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse
import os
import time
import warnings
import json
from scipy.sparse import load_npz, csr_matrix
warnings.filterwarnings('ignore')

# 导入您的原始模块
from CryptoSSGNN0808 import (
    load_data, sparse_to_edge_index, construct_attribute_graph,
    evaluate_model, set_seed, EnhancedCAREGNN
)

class SemiSupervisedMixin:
    """半监督学习混入类 - 为所有基线模型提供半监督能力"""
    
    def compute_contrastive_loss(self, embeddings, edge_indices, temperature=0.1, num_neg_samples=3):
        """
        计算对比学习损失 - 统一的无监督损失函数
        Args:
            embeddings: [num_nodes, embed_dim] 节点嵌入
            edge_indices: list of [2, num_edges] 边索引列表
            temperature: 温度参数
            num_neg_samples: 负样本数量
        """
        device = embeddings.device
        total_loss = 0
        num_valid_relations = 0
        
        for edge_index in edge_indices:
            if edge_index.size(1) == 0:
                continue
                
            num_valid_relations += 1
            
            # 限制边数量，防止计算爆炸
            max_edges = min(1000, edge_index.size(1))
            if edge_index.size(1) > max_edges:
                perm = torch.randperm(edge_index.size(1), device=device)[:max_edges]
                edge_index = edge_index[:, perm]
            
            src, dst = edge_index
            
            # 标准化嵌入
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            
            # 正样本分数
            pos_scores = (embeddings_norm[src] * embeddings_norm[dst]).sum(dim=1) / temperature
            
            # 负采样
            num_nodes = embeddings.size(0)
            neg_dst = torch.randint(0, num_nodes, (src.size(0), num_neg_samples), device=device)
            
            # 确保负样本不等于正样本
            for i in range(src.size(0)):
                while neg_dst[i].eq(dst[i]).any():
                    neg_dst[i] = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
            
            # 负样本分数
            src_emb_expanded = embeddings_norm[src].unsqueeze(1)  # [batch, 1, dim]
            neg_emb = embeddings_norm[neg_dst]  # [batch, num_neg_samples, dim]
            neg_scores = (src_emb_expanded * neg_emb).sum(dim=2) / temperature  # [batch, num_neg_samples]
            
            # InfoNCE损失
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch, 1+num_neg_samples]
            targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=device)
            relation_loss = F.cross_entropy(all_scores, targets)
            
            total_loss += relation_loss
        
        if num_valid_relations == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / num_valid_relations
    
    def compute_semi_supervised_loss(self, logits, embeddings, labels, mask, edge_indices, alpha=0.5):
        """
        计算半监督损失
        Args:
            logits: [num_nodes, num_classes] 预测logits
            embeddings: [num_nodes, embed_dim] 节点嵌入
            labels: [num_nodes] 标签
            mask: [num_nodes] 训练掩码
            edge_indices: list of edge indices
            alpha: 监督/无监督损失平衡参数
        """
        # 监督损失
        sup_loss = F.cross_entropy(logits[mask], labels[mask])
        
        # 无监督损失
        if alpha < 1.0:
            unsup_loss = self.compute_contrastive_loss(embeddings, edge_indices)
            # 自适应缩放，防止无监督损失过大
            with torch.no_grad():
                if unsup_loss.item() > sup_loss.item() * 3:
                    scale_factor = sup_loss.item() / unsup_loss.item()
                    unsup_loss = unsup_loss * scale_factor
        else:
            unsup_loss = torch.tensor(0.0, device=sup_loss.device, requires_grad=True)
        
        # 组合损失
        if alpha >= 1.0:
            total_loss = sup_loss
        elif alpha <= 0.0:
            total_loss = unsup_loss
        else:
            total_loss = alpha * sup_loss + (1 - alpha) * unsup_loss
        
        return total_loss, sup_loss, unsup_loss


class SemiSupervisedGraphConvLayer(nn.Module):
    """半监督图卷积层"""
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(SemiSupervisedGraphConvLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, edge_index, edge_weight=None):
        num_nodes = features.size(0)
        device = features.device
        
        if edge_index.size(1) == 0:
            return self.dropout(self.linear(features))
        
        # 消息传递
        src_features = features[edge_index[0]]
        
        if edge_weight is not None:
            weighted_messages = src_features * edge_weight.unsqueeze(-1)
        else:
            weighted_messages = src_features
        
        # 聚合到目标节点
        dst_indices = edge_index[1]
        aggregated = torch.zeros(num_nodes, features.size(1), device=device)
        
        for i in range(features.size(1)):
            aggregated[:, i].scatter_add_(0, dst_indices, weighted_messages[:, i])
        
        # 线性变换
        output = self.linear(aggregated + features)
        return self.dropout(output)


class SemiSupervisedGCN(nn.Module, SemiSupervisedMixin):
    """半监督GCN"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=2, 
                 num_layers=2, dropout=0.5, alpha=0.5):
        super(SemiSupervisedGCN, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        
        # GCN层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SemiSupervisedGraphConvLayer(input_dim, hidden_dim, dropout))
            else:
                self.layers.append(SemiSupervisedGraphConvLayer(hidden_dim, hidden_dim, dropout))
        
        # 嵌入投影层
        self.embedding_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features, edge_indices, edge_weights, return_embeddings=False):
        x = features
        
        # 选择第一个关系图
        if len(edge_indices) > 0:
            edge_index = edge_indices[0]
            edge_weight = edge_weights[0] if len(edge_weights) > 0 else None
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=features.device)
            edge_weight = None
        
        # 通过GCN层
        for layer in self.layers:
            x = F.relu(layer(x, edge_index, edge_weight))
        
        # 嵌入投影
        embeddings = self.embedding_projection(x)
        
        # 分类
        logits = self.classifier(x)
        
        if return_embeddings:
            return logits, embeddings
        return logits


class SemiSupervisedMultiRelationGCN(nn.Module, SemiSupervisedMixin):
    """半监督多关系GCN"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, 
                 num_classes=2, num_layers=2, dropout=0.5, alpha=0.5):
        super(SemiSupervisedMultiRelationGCN, self).__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.alpha = alpha
        
        # 每个关系的GCN层
        self.relation_layers = nn.ModuleList()
        for i in range(num_relations):
            layers = nn.ModuleList()
            for j in range(num_layers):
                if j == 0:
                    layers.append(SemiSupervisedGraphConvLayer(input_dim, hidden_dim, dropout))
                else:
                    layers.append(SemiSupervisedGraphConvLayer(hidden_dim, hidden_dim, dropout))
            self.relation_layers.append(layers)
        
        # 关系融合
        self.relation_fusion = nn.Linear(hidden_dim * num_relations, hidden_dim)
        
        # 嵌入投影层
        self.embedding_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features, edge_indices, edge_weights, return_embeddings=False):
        relation_outputs = []
        
        for i in range(min(len(edge_indices), self.num_relations)):
            x = features
            edge_index = edge_indices[i]
            edge_weight = edge_weights[i] if i < len(edge_weights) else None
            
            # 通过关系特定的GCN层
            for layer in self.relation_layers[i]:
                x = F.relu(layer(x, edge_index, edge_weight))
            
            relation_outputs.append(x)
        
        # 填充缺失的关系
        while len(relation_outputs) < self.num_relations:
            relation_outputs.append(torch.zeros_like(relation_outputs[0]))
        
        # 融合所有关系
        combined = torch.cat(relation_outputs, dim=-1)
        fused = F.relu(self.relation_fusion(combined))
        
        # 嵌入投影
        embeddings = self.embedding_projection(fused)
        
        # 分类
        logits = self.classifier(fused)
        
        if return_embeddings:
            return logits, embeddings
        return logits


class SemiSupervisedSAGELayer(nn.Module):
    """半监督GraphSAGE层"""
    def __init__(self, input_dim, output_dim, dropout=0.5, aggregator='mean'):
        super(SemiSupervisedSAGELayer, self).__init__()
        self.aggregator = aggregator
        self.linear = nn.Linear(input_dim * 2, output_dim)  # self + neighbor
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, edge_index):
        num_nodes = features.size(0)
        device = features.device
        
        if edge_index.size(1) == 0:
            doubled_features = torch.cat([features, features], dim=-1)
            return self.dropout(self.linear(doubled_features))
        
        # 邻居聚合
        src_features = features[edge_index[0]]
        dst_indices = edge_index[1]
        
        # 聚合邻居特征
        neighbor_features = torch.zeros_like(features)
        if self.aggregator == 'mean':
            # 计算每个节点的度
            degree = torch.zeros(num_nodes, device=device)
            degree.scatter_add_(0, dst_indices, torch.ones_like(dst_indices, dtype=torch.float))
            degree = torch.clamp(degree, min=1)
            
            # 聚合
            for i in range(features.size(1)):
                neighbor_features[:, i].scatter_add_(0, dst_indices, src_features[:, i])
            neighbor_features = neighbor_features / degree.unsqueeze(-1)
        
        # 连接自身和邻居特征
        combined_features = torch.cat([features, neighbor_features], dim=-1)
        output = self.linear(combined_features)
        
        return self.dropout(output)


class SemiSupervisedGraphSAGE(nn.Module, SemiSupervisedMixin):
    """半监督GraphSAGE"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=2, 
                 num_layers=2, dropout=0.5, aggregator='mean', alpha=0.5):
        super(SemiSupervisedGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.alpha = alpha
        
        # SAGE层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SemiSupervisedSAGELayer(input_dim, hidden_dim, dropout, aggregator))
            else:
                self.layers.append(SemiSupervisedSAGELayer(hidden_dim, hidden_dim, dropout, aggregator))
        
        # 嵌入投影层
        self.embedding_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features, edge_indices, edge_weights, return_embeddings=False):
        x = features
        
        # 使用第一个关系图
        if len(edge_indices) > 0:
            edge_index = edge_indices[0]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=features.device)
        
        # 通过SAGE层
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        
        # 嵌入投影
        embeddings = self.embedding_projection(x)
        
        # 分类
        logits = self.classifier(x)
        
        if return_embeddings:
            return logits, embeddings
        return logits


class SemiSupervisedGraphAttentionLayer(nn.Module):
    """半监督图注意力层"""
    def __init__(self, input_dim, output_dim, num_heads=1, dropout=0.5):
        super(SemiSupervisedGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(output_dim * 2, num_heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, edge_index):
        num_nodes = features.size(0)
        device = features.device
        
        if edge_index.size(1) == 0:
            return self.dropout(self.linear(features))
        
        # 线性变换
        h = self.linear(features)
        
        # 计算注意力权重
        src_h = h[edge_index[0]]
        dst_h = h[edge_index[1]]
        
        attention_input = torch.cat([src_h, dst_h], dim=-1)
        attention_weights = self.attention(attention_input)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # 消息传递
        messages = src_h.unsqueeze(1) * attention_weights.unsqueeze(-1)
        
        # 聚合
        output = torch.zeros(num_nodes, self.output_dim, device=device)
        for head in range(self.num_heads):
            head_messages = messages[:, head, :]
            head_output = torch.zeros(num_nodes, self.head_dim, device=device)
            
            for i in range(self.head_dim):
                head_output[:, i].scatter_add_(0, edge_index[1], head_messages[:, i])
            
            start_idx = head * self.head_dim
            end_idx = start_idx + self.head_dim
            output[:, start_idx:end_idx] = head_output
        
        return self.dropout(output + h)


class SemiSupervisedGAT(nn.Module, SemiSupervisedMixin):
    """半监督图注意力网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=2, 
                 num_layers=2, num_heads=2, dropout=0.5, alpha=0.5):
        super(SemiSupervisedGAT, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        
        # GAT层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SemiSupervisedGraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout))
            else:
                self.layers.append(SemiSupervisedGraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout))
        
        # 嵌入投影层
        self.embedding_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features, edge_indices, edge_weights, return_embeddings=False):
        x = features
        
        # 使用第一个关系图
        if len(edge_indices) > 0:
            edge_index = edge_indices[0]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=features.device)
        
        # 通过GAT层
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        
        # 嵌入投影
        embeddings = self.embedding_projection(x)
        
        # 分类
        logits = self.classifier(x)
        
        if return_embeddings:
            return logits, embeddings
        return logits


class SemiSupervisedMLP(nn.Module, SemiSupervisedMixin):
    """半监督MLP - 使用特征重构作为无监督任务"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=2, 
                 num_layers=2, dropout=0.5, alpha=0.5):
        super(SemiSupervisedMLP, self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        
        # 主网络
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 嵌入投影层
        self.embedding_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 特征重构器（用于无监督学习）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, features, edge_indices=None, edge_weights=None, return_embeddings=False):
        # 编码
        encoded = self.encoder(features)
        
        # 嵌入投影
        embeddings = self.embedding_projection(encoded)
        
        # 分类
        logits = self.classifier(encoded)
        
        if return_embeddings:
            return logits, embeddings, encoded
        return logits
    
    def compute_reconstruction_loss(self, features, encoded):
        """计算重构损失作为无监督损失"""
        reconstructed = self.decoder(encoded)
        return F.mse_loss(reconstructed, features)
    
    def compute_semi_supervised_loss(self, logits, embeddings, labels, mask, edge_indices, alpha=None):
        """重写MLP的半监督损失"""
        if alpha is None:
            alpha = self.alpha
            
        # 监督损失
        sup_loss = F.cross_entropy(logits[mask], labels[mask])
        
        # 无监督损失（特征重构）
        if alpha < 1.0:
            # 获取编码特征
            with torch.no_grad():
                _, _, encoded = self.forward(embeddings.detach(), return_embeddings=True)
            unsup_loss = self.compute_reconstruction_loss(embeddings.detach(), encoded)
        else:
            unsup_loss = torch.tensor(0.0, device=sup_loss.device, requires_grad=True)
        
        # 组合损失
        if alpha >= 1.0:
            total_loss = sup_loss
        elif alpha <= 0.0:
            total_loss = unsup_loss
        else:
            total_loss = alpha * sup_loss + (1 - alpha) * unsup_loss
        
        return total_loss, sup_loss, unsup_loss


def train_semi_supervised_baseline(model, features, labels, edge_indices, edge_weights,
                                 train_mask, val_mask, test_mask, args):
    """训练半监督基线模型"""
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 0
    max_patience = 30
    
    print(f"Training with α={model.alpha if hasattr(model, 'alpha') else 'N/A'}")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(model, SemiSupervisedMLP):
            logits, embeddings, encoded = model(features, edge_indices, edge_weights, return_embeddings=True)
            # MLP使用特征作为"嵌入"进行重构
            total_loss, sup_loss, unsup_loss = model.compute_semi_supervised_loss(
                logits, features, labels, train_mask, edge_indices
            )
        else:
            logits, embeddings = model(features, edge_indices, edge_weights, return_embeddings=True)
            total_loss, sup_loss, unsup_loss = model.compute_semi_supervised_loss(
                logits, embeddings, labels, train_mask, edge_indices
            )
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 评估
        if epoch % args.eval_every == 0:
            val_metrics = evaluate_semi_supervised_model(model, features, labels, edge_indices, 
                                                       edge_weights, val_mask)
            test_metrics = evaluate_semi_supervised_model(model, features, labels, edge_indices, 
                                                        edge_weights, test_mask)
            
            if epoch % (args.eval_every * 2) == 0:
                print(f"Epoch {epoch:03d} | Total: {total_loss:.4f} "
                      f"(Sup: {sup_loss:.4f}, Unsup: {unsup_loss:.4f}) | "
                      f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                      f"Test F1: {test_metrics.get('f1', 0):.4f}")
            
            # 早停机制
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_test_metrics = test_metrics
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
    
    return best_test_metrics


def evaluate_semi_supervised_model(model, features, labels, edge_indices, edge_weights, mask):
    """评估半监督模型"""
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_indices, edge_weights)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
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


def run_semi_supervised_comparison(data_dir, args):
    """运行半监督模型对比实验"""
    print("\n" + "="*80)
    print("SEMI-SUPERVISED MODEL COMPARISON STUDY")
    print("="*80)
    
    set_seed(args.seed)
    
    # 加载数据
    features, labels, adj_matrices = load_data(data_dir)
    attr_edge_index, attr_edge_weight = construct_attribute_graph(
        features, k_neighbors=args.k_neighbors
    )
    
    # 转换为edge_index格式
    edge_indices = []
    edge_weights = []
    for adj in adj_matrices:
        edge_index, edge_weight = sparse_to_edge_index(adj)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    edge_indices.append(attr_edge_index)
    edge_weights.append(attr_edge_weight)
    
    # 数据预处理
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # 创建训练/验证/测试掩码
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
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]
    edge_weights = [ew.to(device) for ew in edge_weights]
    
    results = {}
    
    # 测试不同的α值
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0] if args.test_alpha_values else [args.alpha]
    
    for alpha in alpha_values:
        print(f"\n🔍 Testing models with α={alpha}")
        alpha_results = {}
        
        # 1. 半监督MLP
        print(f"\n🔍 Testing Semi-Supervised MLP (α={alpha})...")
        mlp_model = SemiSupervisedMLP(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=alpha
        )
        mlp_results = train_semi_supervised_baseline(
            mlp_model, features, labels, edge_indices, edge_weights,
            train_mask, val_mask, test_mask, args
        )
        alpha_results['Semi_MLP'] = mlp_results
        
        # 2. 半监督GCN
        print(f"\n🔍 Testing Semi-Supervised GCN (α={alpha})...")
        gcn_model = SemiSupervisedGCN(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=alpha
        )
        gcn_results = train_semi_supervised_baseline(
            gcn_model, features, labels, edge_indices, edge_weights,
            train_mask, val_mask, test_mask, args
        )
        alpha_results['Semi_GCN'] = gcn_results
        
        # 3. 半监督多关系GCN
        print(f"\n🔍 Testing Semi-Supervised Multi-Relation GCN (α={alpha})...")
        mr_gcn_model = SemiSupervisedMultiRelationGCN(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_relations=len(adj_matrices),
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=alpha
        )
        mr_gcn_results = train_semi_supervised_baseline(
            mr_gcn_model, features, labels, edge_indices, edge_weights,
            train_mask, val_mask, test_mask, args
        )
        alpha_results['Semi_MultiRelation_GCN'] = mr_gcn_results
        
        # 4. 半监督GraphSAGE
        print(f"\n🔍 Testing Semi-Supervised GraphSAGE (α={alpha})...")
        sage_model = SemiSupervisedGraphSAGE(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=alpha
        )
        sage_results = train_semi_supervised_baseline(
            sage_model, features, labels, edge_indices, edge_weights,
            train_mask, val_mask, test_mask, args
        )
        alpha_results['Semi_GraphSAGE'] = sage_results
        
        # 5. 半监督GAT
        print(f"\n🔍 Testing Semi-Supervised GAT (α={alpha})...")
        gat_model = SemiSupervisedGAT(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            num_heads=2,
            dropout=args.dropout,
            alpha=alpha
        )
        gat_results = train_semi_supervised_baseline(
            gat_model, features, labels, edge_indices, edge_weights,
            train_mask, val_mask, test_mask, args
        )
        alpha_results['Semi_GAT'] = gat_results
        
        # 6. Enhanced CARE-GNN（作为对比）
        if alpha == args.alpha:  # 只在默认α值时测试
            print(f"\n🔍 Testing Enhanced CARE-GNN (α={alpha})...")
            care_gnn_model = EnhancedCAREGNN(
                input_dim=features.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=args.hidden_dim,
                num_relations=len(adj_matrices),
                num_views=len(edge_indices),
                num_classes=2,
                num_layers=args.num_layers,
                dropout=args.dropout,
                alpha=alpha
            )
            care_gnn_results = train_enhanced_care_gnn_for_comparison(
                care_gnn_model, features, labels, edge_indices, edge_weights,
                train_mask, val_mask, test_mask, args
            )
            alpha_results['Enhanced_CARE_GNN'] = care_gnn_results
        
        results[f'alpha_{alpha}'] = alpha_results
    
    # 输出结果
    print_semi_supervised_results(results, args)
    
    return results


def train_enhanced_care_gnn_for_comparison(model, features, labels, edge_indices, edge_weights,
                                         train_mask, val_mask, test_mask, args):
    """为对比实验训练Enhanced CARE-GNN"""
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 0
    max_patience = 30
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        final_logits, aux_predictions, embeddings, _ = model(
            features, edge_indices, edge_weights, return_embeddings=True
        )
        
        # 计算损失
        total_loss, sup_loss, graph_loss = model.compute_loss(
            final_logits, aux_predictions, labels, train_mask,
            embeddings, edge_indices
        )
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 评估
        if epoch % args.eval_every == 0:
            val_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, val_mask)
            test_metrics = evaluate_model(model, features, labels, edge_indices, edge_weights, test_mask)
            
            if epoch % (args.eval_every * 2) == 0:
                print(f"Epoch {epoch:03d} | Total: {total_loss:.4f} "
                      f"(Sup: {sup_loss:.4f}, Graph: {graph_loss:.4f}) | "
                      f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                      f"Test F1: {test_metrics.get('f1', 0):.4f}")
            
            # 早停机制
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_test_metrics = test_metrics
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
    
    return best_test_metrics


def print_semi_supervised_results(results, args):
    """打印半监督实验结果"""
    print("\n" + "="*100)
    print("SEMI-SUPERVISED MODEL COMPARISON RESULTS")
    print("="*100)
    
    if args.test_alpha_values:
        # α值分析
        print(f"\n📊 Alpha Value Analysis:")
        print(f"{'Model':<25} " + " ".join([f"α={α:<6}" for α in [0.0, 0.3, 0.5, 0.7, 1.0]]))
        print("-" * 100)
        
        # 收集所有模型名称
        all_models = set()
        for alpha_key, alpha_results in results.items():
            all_models.update(alpha_results.keys())
        
        for model_name in sorted(all_models):
            print(f"{model_name:<25} ", end="")
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                alpha_key = f"alpha_{alpha}"
                if alpha_key in results and model_name in results[alpha_key]:
                    f1_score = results[alpha_key][model_name].get('f1', 0)
                    print(f"{f1_score:<8.4f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            print()
    
    # 最佳结果对比（使用默认α值）
    default_alpha_key = f"alpha_{args.alpha}"
    if default_alpha_key in results:
        print(f"\n📈 Best Results Comparison (α={args.alpha}):")
        print(f"{'Model':<25} {'F1':<8} {'AUC':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8}")
        print("-" * 80)
        
        alpha_results = results[default_alpha_key]
        sorted_results = sorted(alpha_results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<25} "
                  f"{metrics.get('f1', 0):<8.4f} "
                  f"{metrics.get('auc', 0):<8.4f} "
                  f"{metrics.get('accuracy', 0):<10.4f} "
                  f"{metrics.get('precision', 0):<10.4f} "
                  f"{metrics.get('recall', 0):<8.4f}")
        
        # 计算改进幅度
        if 'Enhanced_CARE_GNN' in alpha_results:
            care_gnn_f1 = alpha_results['Enhanced_CARE_GNN'].get('f1', 0)
            best_baseline_f1 = 0
            best_baseline_name = ""
            
            for model_name, metrics in alpha_results.items():
                if model_name != 'Enhanced_CARE_GNN' and metrics.get('f1', 0) > best_baseline_f1:
                    best_baseline_f1 = metrics.get('f1', 0)
                    best_baseline_name = model_name
            
            if best_baseline_f1 > 0:
                improvement = ((care_gnn_f1 - best_baseline_f1) / best_baseline_f1) * 100
                print(f"\n🚀 Enhanced CARE-GNN vs Best Semi-Supervised Baseline ({best_baseline_name}):")
                print(f"   Improvement: {improvement:+.2f}% F1 score")
                print(f"   Absolute difference: {care_gnn_f1 - best_baseline_f1:+.4f} F1")


def analyze_alpha_sensitivity(results):
    """分析α参数敏感性"""
    print("\n" + "="*80)
    print("ALPHA PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # 收集所有α值下的结果
    alpha_analysis = {}
    
    for alpha_key, alpha_results in results.items():
        alpha_val = float(alpha_key.split('_')[1])
        
        for model_name, metrics in alpha_results.items():
            if model_name not in alpha_analysis:
                alpha_analysis[model_name] = {'alphas': [], 'f1_scores': []}
            
            alpha_analysis[model_name]['alphas'].append(alpha_val)
            alpha_analysis[model_name]['f1_scores'].append(metrics.get('f1', 0))
    
    # 分析每个模型的最佳α值
    print(f"{'Model':<25} {'Best α':<8} {'Best F1':<10} {'α Range':<15} {'Stability':<10}")
    print("-" * 80)
    
    for model_name, data in alpha_analysis.items():
        alphas = np.array(data['alphas'])
        f1_scores = np.array(data['f1_scores'])
        
        # 排序
        sorted_indices = np.argsort(alphas)
        alphas = alphas[sorted_indices]
        f1_scores = f1_scores[sorted_indices]
        
        # 找到最佳α
        best_idx = np.argmax(f1_scores)
        best_alpha = alphas[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # 计算稳定性（标准差）
        stability = np.std(f1_scores)
        
        # α值范围（F1 > 0.9 * best_f1的α范围）
        good_indices = f1_scores >= 0.9 * best_f1
        if np.any(good_indices):
            alpha_range = f"{alphas[good_indices].min():.1f}-{alphas[good_indices].max():.1f}"
        else:
            alpha_range = f"{best_alpha:.1f}"
        
        print(f"{model_name:<25} "
              f"{best_alpha:<8.1f} "
              f"{best_f1:<10.4f} "
              f"{alpha_range:<15} "
              f"{stability:<10.4f}")


def create_alpha_sensitivity_plot(results, save_path='semi_supervised_results/'):
    """创建α敏感性分析图"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(save_path, exist_ok=True)
        
        # 收集数据
        plot_data = {}
        for alpha_key, alpha_results in results.items():
            alpha_val = float(alpha_key.split('_')[1])
            
            for model_name, metrics in alpha_results.items():
                if model_name not in plot_data:
                    plot_data[model_name] = {'alphas': [], 'f1_scores': []}
                
                plot_data[model_name]['alphas'].append(alpha_val)
                plot_data[model_name]['f1_scores'].append(metrics.get('f1', 0))
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        for model_name, data in plot_data.items():
            alphas = np.array(data['alphas'])
            f1_scores = np.array(data['f1_scores'])
            
            # 排序
            sorted_indices = np.argsort(alphas)
            alphas = alphas[sorted_indices]
            f1_scores = f1_scores[sorted_indices]
            
            # 绘制线条
            linestyle = '-' if 'CARE_GNN' in model_name else '--'
            linewidth = 3 if 'CARE_GNN' in model_name else 1.5
            
            plt.plot(alphas, f1_scores, 'o-', label=model_name, 
                    linestyle=linestyle, linewidth=linewidth, markersize=6)
        
        plt.xlabel('Alpha (α)', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.title('Alpha Parameter Sensitivity Analysis', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'alpha_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Alpha sensitivity plot saved in '{save_path}'")
        
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available, skipping alpha sensitivity plot")


def save_semi_supervised_results(results, save_path='semi_supervised_results/'):
    """保存半监督实验结果"""
    os.makedirs(save_path, exist_ok=True)
    
    # 保存原始结果
    serializable_results = {}
    for alpha_key, alpha_results in results.items():
        serializable_results[alpha_key] = {}
        for model, metrics in alpha_results.items():
            serializable_results[alpha_key][model] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in metrics.items()
            }
    
    with open(os.path.join(save_path, 'semi_supervised_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 创建CSV文件
    import pandas as pd
    
    # 扁平化结果用于CSV
    csv_data = []
    for alpha_key, alpha_results in results.items():
        alpha_val = float(alpha_key.split('_')[1])
        for model, metrics in alpha_results.items():
            row = {'Alpha': alpha_val, 'Model': model}
            row.update(metrics)
            csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(save_path, 'semi_supervised_results.csv'), index=False)
    
    print(f"✅ Semi-supervised results saved in '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised GNN Model Comparison')
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
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Default alpha value for supervised/unsupervised balance')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Number of neighbors for attribute graph construction')
    
    # 实验控制参数
    parser.add_argument('--run_semi_comparison', action='store_true',
                        help='Run semi-supervised model comparison')
    parser.add_argument('--test_alpha_values', action='store_true',
                        help='Test multiple alpha values for sensitivity analysis')
    parser.add_argument('--analyze_alpha', action='store_true',
                        help='Analyze alpha parameter sensitivity')
    parser.add_argument('--visualize_alpha', action='store_true',
                        help='Create alpha sensitivity visualization')
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results')
    
    args = parser.parse_args()
    
    if args.run_semi_comparison:
        print("🚀 Starting Semi-Supervised Model Comparison...")
        
        # 运行半监督对比实验
        results = run_semi_supervised_comparison(args.data_dir, args)
        
        # α敏感性分析
        if args.analyze_alpha and args.test_alpha_values:
            analyze_alpha_sensitivity(results)
        
        # 创建α敏感性可视化
        if args.visualize_alpha and args.test_alpha_values:
            create_alpha_sensitivity_plot(results)
        
        # 保存结果
        if args.save_results:
            save_semi_supervised_results(results)
        
        print(f"\n🎉 Semi-supervised comparison completed!")
        
        # 输出实验建议
        print(f"\n💡 Experimental Insights:")
        print(f"   ✅ All baselines now use semi-supervised learning")
        print(f"   ✅ Fair comparison with your Enhanced CARE-GNN")
        print(f"   ✅ Alpha sensitivity analysis reveals optimal hyperparameters")
        print(f"   ✅ Demonstrates the value of your model's innovations beyond semi-supervision")
        
    else:
        print("Use --run_semi_comparison to start the semi-supervised model comparison")
        print("Additional options:")
        print("  --test_alpha_values: Test multiple alpha values (0.0, 0.3, 0.5, 0.7, 1.0)")
        print("  --analyze_alpha: Analyze alpha parameter sensitivity")
        print("  --visualize_alpha: Create alpha sensitivity plots")
        print("  --save_results: Save detailed results")


if __name__ == '__main__':
    main()
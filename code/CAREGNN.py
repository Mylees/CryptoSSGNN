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

class CAREGNN(nn.Module):
    """完整的CARE-GNN模型"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_classes=2, num_layers=2, dropout=0.5):
        super(CAREGNN, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        
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
        
        # 辅助损失的权重
        self.aux_loss_weight = 0.1
        
    def forward(self, features, edge_indices, edge_weights):
        """
        前向传播
        """
        # 输入投影
        x = self.input_projection(features)
        
        # 通过CARE-GNN层
        aux_predictions = []
        for i, layer in enumerate(self.care_layers):
            x, predicted_labels = layer(x, edge_indices, edge_weights)
            aux_predictions.append(predicted_labels)
        
        # 最终分类
        final_logits = self.classifier(x)
        
        return final_logits, aux_predictions
    
    def compute_loss(self, final_logits, aux_predictions, labels, mask):
        """
        计算总损失，包括主损失和辅助损失
        """
        # 主损失
        main_loss = F.cross_entropy(final_logits[mask], labels[mask])
        
        # 辅助损失 - 每一层的标签预测损失
        aux_loss = 0
        for aux_pred in aux_predictions:
            if aux_pred is not None:
                aux_loss += F.cross_entropy(aux_pred[mask], labels[mask])
        
        # 总损失
        total_loss = main_loss + self.aux_loss_weight * aux_loss
        
        return total_loss, main_loss, aux_loss

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

def train_care_gnn(data_dir, args):
    """训练CARE-GNN"""
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
    
    # 将edge_indices和edge_weights移动到GPU
    edge_indices = [ei.to(device) for ei in edge_indices]
    edge_weights = [ew.to(device) for ew in edge_weights]
    
    # 创建模型
    model = CAREGNN(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim,
        num_relations=len(adj_matrices),
        num_classes=2,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Train nodes: {train_mask.sum()}")
    print(f"Val nodes: {val_mask.sum()}")
    print(f"Test nodes: {test_mask.sum()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练循环
    best_val_f1 = 0
    best_test_metrics = {}
    patience = 0
    max_patience = 50
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        final_logits, aux_predictions = model(features, edge_indices, edge_weights)
        
        # 计算损失
        total_loss, main_loss, aux_loss = model.compute_loss(
            final_logits, aux_predictions, labels, train_mask
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
            
            print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} (Main: {main_loss:.4f}, Aux: {aux_loss:.4f})")
            print(f"Train F1: {train_metrics.get('f1', 0):.4f} | Val F1: {val_metrics.get('f1', 0):.4f} | Test F1: {test_metrics.get('f1', 0):.4f}")
            
            # 早停机制
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_test_metrics = test_metrics
                patience = 0
                torch.save(model.state_dict(), 'best_care_gnn.pth')
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # 输出最终结果
    print("\n=== Best Test Results ===")
    for metric, value in best_test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return model, best_test_metrics

def main():
    parser = argparse.ArgumentParser(description='CARE-GNN Training')
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
    
    args = parser.parse_args()
    
    # 训练模型
    model, metrics = train_care_gnn(args.data_dir, args)
    
    print("\n=== Training Complete ===")
    print("Best model saved as 'best_care_gnn.pth'")

if __name__ == '__main__':
    main()
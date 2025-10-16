#!/usr/bin/env python3


import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from torch_geometric.data import Data
import random
import os
from collections import defaultdict, deque

class SemiSupervisedAwareNeighborSampler:
    """半监督学习感知的邻居采样器"""
    
    def __init__(self, adj_matrices, labels, sample_sizes=[15, 10, 5]):
        self.adj_matrices = adj_matrices
        self.labels = labels
        self.sample_sizes = sample_sizes
        self.num_hops = len(sample_sizes)
        
        # 预处理：构建邻接列表
        self.adj_lists = {}
        for rel_type, adj_matrix in adj_matrices.items():
            adj_list = defaultdict(list)
            row, col = adj_matrix.nonzero()
            for src, dst in zip(row, col):
                adj_list[src].append(dst)
            self.adj_lists[rel_type] = dict(adj_list)
        
        # 计算节点重要性
        self.node_importance = self._compute_node_importance()
        
    def _compute_node_importance(self):
        """计算节点重要性 - 结合度中心性和标签信息"""
        num_nodes = len(self.labels)
        
        # 1. 度中心性
        total_degree = np.zeros(num_nodes)
        for adj_matrix in self.adj_matrices.values():
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            total_degree += degree
        
        degree_importance = total_degree / (total_degree.max() + 1e-8)
        
        # 2. 标签重要性 - 有标签节点更重要
        label_importance = np.zeros(num_nodes)
        labeled_mask = self.labels != -1
        label_importance[labeled_mask] = 1.0
        
        # 3. 多样性重要性 - 避免过度采样某一类
        diversity_importance = np.ones(num_nodes)
        if np.any(labeled_mask):
            unique_labels = np.unique(self.labels[labeled_mask])
            for label in unique_labels:
                label_nodes = np.where(self.labels == label)[0]
                # 类别越大，单个节点重要性越低
                diversity_importance[label_nodes] = 1.0 / np.sqrt(len(label_nodes))
        
        # 综合重要性：30%度中心性 + 40%标签重要性 + 30%多样性
        importance = (0.3 * degree_importance + 
                     0.4 * label_importance + 
                     0.3 * diversity_importance)
        
        return importance
    
    def label_aware_seed_selection(self, target_seeds=800):
        """标签感知的种子选择 - 保持原始分布"""
        labeled_nodes = np.where(self.labels != -1)[0]
        unique_labels = np.unique(self.labels[labeled_nodes])
        
        seed_nodes = []
        
        # 为每个类别按原始比例分配种子数量
        for label in unique_labels:
            label_nodes = labeled_nodes[self.labels[labeled_nodes] == label]
            
            # 保持原始比例
            original_ratio = len(label_nodes) / len(labeled_nodes)
            target_for_label = max(20, int(target_seeds * original_ratio))  # 最少20个
            target_for_label = min(target_for_label, len(label_nodes))
            
            # 基于重要性选择，而非随机
            label_importance = self.node_importance[label_nodes]
            
            # 重要性采样：70%按重要性，30%随机
            num_importance = int(target_for_label * 0.7)
            num_random = target_for_label - num_importance
            
            # 选择重要性最高的节点
            if num_importance > 0:
                top_indices = np.argsort(label_importance)[-num_importance:]
                seed_nodes.extend(label_nodes[top_indices])
            
            # 随机选择剩余节点
            if num_random > 0:
                remaining_nodes = np.setdiff1d(label_nodes, label_nodes[top_indices] if num_importance > 0 else [])
                if len(remaining_nodes) >= num_random:
                    random_selected = np.random.choice(remaining_nodes, num_random, replace=False)
                    seed_nodes.extend(random_selected)
                else:
                    seed_nodes.extend(remaining_nodes)
            
            print(f"Label {label}: {len(label_nodes)} 个节点 -> {len(seed_nodes) - (len(seed_nodes) - target_for_label)} 个种子")
        
        return seed_nodes
    
    def adaptive_neighbor_sampling(self, nodes, rel_type, base_sample_size, hop):
        """自适应邻居采样 - 根据节点特性调整采样策略"""
        sampled_neighbors = []
        adj_list = self.adj_lists[rel_type]
        
        for node in nodes:
            if node not in adj_list:
                continue
                
            neighbors = adj_list[node]
            node_degree = len(neighbors)
            
            # 自适应采样策略
            if node_degree <= 5:
                # 低度节点：保留所有邻居
                sample_size = len(neighbors)
            elif node_degree <= 20:
                # 中等度节点：适度采样
                sample_size = min(base_sample_size, len(neighbors))
            else:
                # 高度节点：减少采样避免hub偏差
                sample_size = min(base_sample_size // 2, len(neighbors))
            
            # 随hop递减
            sample_size = max(1, sample_size // (hop + 1))
            
            if sample_size >= len(neighbors):
                sampled_neighbors.extend(neighbors)
            else:
                # 重要性采样
                neighbor_importance = self.node_importance[neighbors]
                
                # 混合策略：50%重要性采样 + 50%随机采样
                num_importance = sample_size // 2
                num_random = sample_size - num_importance
                
                if num_importance > 0 and len(neighbors) > num_importance:
                    # 按重要性选择
                    top_indices = np.argsort(neighbor_importance)[-num_importance:]
                    sampled_neighbors.extend([neighbors[i] for i in top_indices])
                    
                    # 随机选择剩余
                    remaining_neighbors = [neighbors[i] for i in range(len(neighbors)) if i not in top_indices]
                    if len(remaining_neighbors) >= num_random:
                        random_selected = random.sample(remaining_neighbors, num_random)
                        sampled_neighbors.extend(random_selected)
                    else:
                        sampled_neighbors.extend(remaining_neighbors)
                else:
                    # 完全随机
                    sampled_neighbors.extend(random.sample(neighbors, sample_size))
        
        return list(set(sampled_neighbors))
    
    def semi_supervised_sampling(self, target_size):
        """半监督学习专用的采样方法"""
        
        # 1. 标签感知的种子选择
        seed_nodes = self.label_aware_seed_selection(target_seeds=min(1000, target_size//10))
        
        all_sampled_nodes = set(seed_nodes)
        current_layer_nodes = list(seed_nodes)
        
        print(f"[Semi-supervised Sampling] 种子节点: {len(seed_nodes)}")
        print(f"[Semi-supervised Sampling] 种子节点标签分布: {np.bincount(self.labels[seed_nodes][self.labels[seed_nodes] != -1].astype(int))}")
        
        # 2. 多跳自适应采样
        for hop in range(self.num_hops):
            if len(all_sampled_nodes) >= target_size:
                break
            
            next_layer_nodes = []
            base_sample_size = self.sample_sizes[hop]
            
            print(f"[Hop {hop+1}] 开始采样，当前节点数: {len(all_sampled_nodes)}")
            
            # 对每种关系类型进行自适应采样
            for rel_type in self.adj_matrices.keys():
                neighbors = self.adaptive_neighbor_sampling(
                    current_layer_nodes, rel_type, base_sample_size, hop
                )
                next_layer_nodes.extend(neighbors)
            
            # 去重并移除已采样节点
            next_layer_nodes = list(set(next_layer_nodes) - all_sampled_nodes)
            
            # 容量控制 - 基于重要性选择
            remaining_capacity = target_size - len(all_sampled_nodes)
            if remaining_capacity > 0:
                if len(next_layer_nodes) > remaining_capacity:
                    # 重要性采样
                    node_importance = self.node_importance[next_layer_nodes]
                    importance_probs = node_importance / node_importance.sum()
                    
                    selected_indices = np.random.choice(
                        len(next_layer_nodes), 
                        remaining_capacity, 
                        replace=False, 
                        p=importance_probs
                    )
                    next_layer_nodes = [next_layer_nodes[i] for i in selected_indices]
                
                all_sampled_nodes.update(next_layer_nodes)
                current_layer_nodes = next_layer_nodes
                
                print(f"[Hop {hop+1}] 采样邻居: {len(next_layer_nodes)}, 累计: {len(all_sampled_nodes)}")
            else:
                break
        
        # 3. 图连通性修复
        final_nodes = list(all_sampled_nodes)
        final_nodes = self._repair_graph_connectivity(final_nodes, target_size)
        
        return final_nodes
    
    def _repair_graph_connectivity(self, nodes, target_size):
        """修复图连通性 - 确保标签传播路径"""
        if len(nodes) >= target_size:
            return nodes[:target_size]
        
        current_nodes = set(nodes)
        labeled_nodes = [n for n in nodes if self.labels[n] != -1]
        unlabeled_nodes = [n for n in nodes if self.labels[n] == -1]
        
        print(f"[连通性修复] 当前: {len(nodes)} 节点, 标签: {len(labeled_nodes)}, 无标签: {len(unlabeled_nodes)}")
        
        # 确保每个无标签节点都有到标签节点的路径
        bridge_nodes = set()
        
        for unlabeled_node in unlabeled_nodes:
            # 检查是否需要桥接节点
            has_labeled_neighbor = False
            
            for rel_type, adj_list in self.adj_lists.items():
                if unlabeled_node in adj_list:
                    neighbors = adj_list[unlabeled_node]
                    for neighbor in neighbors:
                        if neighbor in labeled_nodes:
                            has_labeled_neighbor = True
                            break
                    if has_labeled_neighbor:
                        break
            
            # 如果没有直接连接到标签节点，寻找桥接节点
            if not has_labeled_neighbor:
                for rel_type, adj_list in self.adj_lists.items():
                    if unlabeled_node in adj_list:
                        neighbors = adj_list[unlabeled_node]
                        for neighbor in neighbors:
                            if neighbor not in current_nodes and neighbor < len(self.labels):
                                # 检查这个邻居是否连接到标签节点
                                for rel_type2, adj_list2 in self.adj_lists.items():
                                    if neighbor in adj_list2:
                                        neighbor_neighbors = adj_list2[neighbor]
                                        if any(nn in labeled_nodes for nn in neighbor_neighbors):
                                            bridge_nodes.add(neighbor)
                                            break
        
        # 添加桥接节点
        remaining_capacity = target_size - len(current_nodes)
        bridge_nodes = list(bridge_nodes)[:remaining_capacity]
        
        final_nodes = list(current_nodes) + bridge_nodes
        
        print(f"[连通性修复] 添加 {len(bridge_nodes)} 个桥接节点")
        
        return final_nodes


def create_edge_type_from_attr(edge_attr):
    """
    基于edge_attr（金额、时间差）自动划分4类关系
    edge_attr: tensor of shape [num_edges, 2], 其中第0列是金额，第1列是时间差
    
    返回:
    - edge_types: numpy array，每条边的关系类型 (0-3)
    - relation_info: dict，记录每种关系的含义
    """
    
    # 假设edge_attr[:, 0]是金额，edge_attr[:, 1]是时间差
    amounts = edge_attr[:, 0].numpy() if hasattr(edge_attr, 'numpy') else edge_attr[:, 0]
    time_deltas = edge_attr[:, 1].numpy() if hasattr(edge_attr, 'numpy') else edge_attr[:, 1]
    
    print(f"金额范围: [{amounts.min():.2f}, {amounts.max():.2f}]")
    print(f"时间差范围: [{time_deltas.min():.2f}, {time_deltas.max():.2f}]")
    
    # 使用中位数进行二值划分
    amount_median = np.median(amounts)
    time_median = np.median(time_deltas)
    
    print(f"金额中位数: {amount_median:.2f}")
    print(f"时间差中位数: {time_median:.2f}")
    
    # 创建二进制标记
    amount_high = amounts >= amount_median  # True=高额, False=低额
    time_high = time_deltas >= time_median   # True=高频(时间差大), False=低频
    
    # 组合成4类关系
    edge_types = np.zeros(len(amounts), dtype=int)
    
    # Rel-0: low-amount & low-freq (低额-低频)
    mask_0 = (~amount_high) & (~time_high)
    edge_types[mask_0] = 0
    
    # Rel-1: low-amount & high-freq (低额-高频) 
    mask_1 = (~amount_high) & time_high
    edge_types[mask_1] = 1
    
    # Rel-2: high-amount & low-freq (高额-低频)
    mask_2 = amount_high & (~time_high)
    edge_types[mask_2] = 2
    
    # Rel-3: high-amount & high-freq (高额-高频)
    mask_3 = amount_high & time_high
    edge_types[mask_3] = 3
    
    # 统计每种关系的边数
    relation_info = {}
    for i in range(4):
        count = np.sum(edge_types == i)
        relation_info[i] = {
            'count': count,
            'percentage': count / len(edge_types) * 100,
            'description': [
                'low-amount & low-freq (低额-低频)',
                'low-amount & high-freq (低额-高频)', 
                'high-amount & low-freq (高额-低频)',
                'high-amount & high-freq (高额-高频)'
            ][i]
        }
        print(f"Rel-{i}: {relation_info[i]['description']} - {count:,} 边 ({relation_info[i]['percentage']:.1f}%)")
    
    return edge_types, relation_info


def create_fixed_dataset(original_data_path="data.pt", output_dir="fixed_semisup_dataset"):
    """创建修复后的数据集"""
    
    print("=" * 60)
    print("创建半监督学习优化的数据集（基于edge_attr自动划分关系）")
    print("=" * 60)
    
    # 1. 加载原始数据
    data = torch.load(original_data_path, weights_only=False)
    edge_index = data['edge_index']
    num_nodes = data['num_nodes'] if 'num_nodes' in data else int(edge_index.max().item()) + 1
    
    # 提取特征和标签
    features = data['X'].numpy()
    if 'labels' in data:
        labels = data['labels'].numpy()
    elif hasattr(data, 'y'):
        labels = data.y.numpy()
    else:
        raise KeyError("未找到标签数据")
    
    print(f"原始数据: {num_nodes} 节点, {features.shape[1]} 特征维度")
    
    # 2. 检查并处理edge_attr
    if 'edge_attr' not in data or data['edge_attr'] is None:
        print("⚠️  未找到edge_attr，将生成模拟的金额和时间差数据")
        # 生成模拟的edge_attr: [金额, 时间差]
        num_edges = edge_index.shape[1]
        # 金额：对数正态分布模拟真实交易
        amounts = np.random.lognormal(mean=5, sigma=2, size=num_edges)
        # 时间差：指数分布模拟交易间隔
        time_deltas = np.random.exponential(scale=10, size=num_edges)
        edge_attr = torch.tensor(np.column_stack([amounts, time_deltas]), dtype=torch.float32)
        print(f"生成模拟edge_attr: {edge_attr.shape}")
    else:
        edge_attr = data['edge_attr']
        print(f"找到edge_attr: {edge_attr.shape}")
        
        # 确保edge_attr至少有2列（金额、时间）
        if edge_attr.shape[1] < 2:
            print("⚠️  edge_attr列数不足，补充时间差列")
            # 如果只有1列，假设是金额，补充时间差
            time_deltas = np.random.exponential(scale=10, size=edge_attr.shape[0])
            edge_attr = torch.cat([edge_attr, torch.tensor(time_deltas).unsqueeze(1)], dim=1)
    
    # 3. 基于edge_attr自动划分关系类型
    print(f"\n基于edge_attr自动划分关系类型...")
    edge_types, relation_info = create_edge_type_from_attr(edge_attr)
    
    # 4. 构建多关系邻接矩阵
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    adj_matrices = {}
    
    for rel_type in range(4):  # 固定4种关系
        mask = edge_types == rel_type
        if not np.any(mask):
            print(f"⚠️  关系类型 {rel_type} 没有边，跳过")
            continue
            
        src_rel = src[mask]
        dst_rel = dst[mask]
        adj_rel = coo_matrix((np.ones(len(src_rel)), (src_rel, dst_rel)), 
                            shape=(num_nodes, num_nodes))
        adj_matrices[rel_type] = adj_rel
        print(f"关系 {rel_type} ({relation_info[rel_type]['description']}): {adj_rel.nnz:,} 条边")
    
    # 5. 使用改进的采样器
    TARGET_NODES = 15000
    sampler = SemiSupervisedAwareNeighborSampler(
        adj_matrices, labels, sample_sizes=[20, 15, 10]
    )
    
    print(f"\n开始半监督优化采样...")
    selected_nodes = sampler.semi_supervised_sampling(TARGET_NODES)
    
    print(f"最终采样节点数: {len(selected_nodes)}")
    
    # 6. 构建子图
    node_id_map = {old: new for new, old in enumerate(selected_nodes)}
    sub_features = features[selected_nodes]
    sub_labels = labels[selected_nodes]
    
    # 分析标签分布保持情况
    original_labeled = labels[labels != -1]
    sub_labeled = sub_labels[sub_labels != -1]
    
    correlation = 0
    if len(original_labeled) > 0 and len(sub_labeled) > 0:
        original_dist = np.bincount(original_labeled.astype(int))
        sub_dist = np.bincount(sub_labeled.astype(int))
        
        # 计算分布相似性
        min_len = min(len(original_dist), len(sub_dist))
        if min_len > 1:
            correlation = np.corrcoef(
                original_dist[:min_len] / original_dist[:min_len].sum(),
                sub_dist[:min_len] / sub_dist[:min_len].sum()
            )[0, 1]
            print(f"类别分布保持相似性: {correlation:.4f}")
    
    # 7. 构建子图邻接矩阵
    sub_adj_matrices = {}
    total_edges = 0
    
    for rel_type, adj_matrix in adj_matrices.items():
        row, col = adj_matrix.nonzero()
        
        # 筛选子图边
        mask = np.isin(row, selected_nodes) & np.isin(col, selected_nodes)
        src_sub = [node_id_map[s] for s in row[mask]]
        dst_sub = [node_id_map[d] for d in col[mask]]
        
        adj_sub = coo_matrix((np.ones(len(src_sub)), (src_sub, dst_sub)),
                            shape=(len(selected_nodes), len(selected_nodes)))
        sub_adj_matrices[rel_type] = adj_sub
        total_edges += adj_sub.nnz
        
        print(f"子图关系 {rel_type} ({relation_info[rel_type]['description']}): {adj_sub.nnz:,} 条边")
    
    # 8. 分析图质量
    analyze_graph_quality_for_semisupervised(sub_adj_matrices, sub_labels, len(selected_nodes))
    
    # 9. 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    
    for rel_type, adj_sub in sub_adj_matrices.items():
        save_npz(os.path.join(output_dir, f"adj_relation{rel_type}.npz"), adj_sub)
    
    save_npz(os.path.join(output_dir, "features.npz"), csr_matrix(sub_features))
    np.save(os.path.join(output_dir, "labels.npy"), sub_labels)
    
    # 保存关系信息和改进信息
    improvement_info = {
        'sampling_method': 'semi_supervised_aware_neighbor_sampling',
        'relation_method': 'edge_attr_based_4_relations',
        'relation_info': relation_info,
        'improvements': [
            'label_aware_seed_selection',
            'adaptive_neighbor_sampling',
            'importance_based_selection',
            'graph_connectivity_repair',
            'edge_attr_based_relation_types'
        ],
        'target_nodes': TARGET_NODES,
        'actual_nodes': len(selected_nodes),
        'distribution_correlation': correlation,
        'total_edges': total_edges
    }
    
    np.save(os.path.join(output_dir, "improvement_info.npy"), improvement_info)
    
    print(f"\n{'='*60}")
    print(f"半监督优化数据集已保存到: {output_dir}/")
    print(f"{'='*60}")
    print(f"🔧 主要改进:")
    print(f"  ✅ 基于edge_attr自动划分4类关系（金额×时间）")
    print(f"  ✅ 标签感知的种子选择（保持原始分布）")
    print(f"  ✅ 自适应邻居采样（避免hub偏差）")
    print(f"  ✅ 重要性引导的节点选择")
    print(f"  ✅ 图连通性修复（确保标签传播路径）")
    print(f"\n📊 关系类型分布:")
    for rel_type, info in relation_info.items():
        print(f"  - Rel-{rel_type}: {info['description']} ({info['percentage']:.1f}%)")
    print(f"\n📊 数据集质量:")
    print(f"  - 节点数: {len(selected_nodes):,}")
    print(f"  - 边数: {total_edges:,}")
    print(f"  - 类别分布相似性: {correlation:.4f}" if correlation > 0 else "  - 类别分布相似性: N/A")
    
    return output_dir


def analyze_graph_quality_for_semisupervised(adj_matrices, labels, num_nodes):
    """分析图质量对半监督学习的适用性"""
    
    print(f"\n图质量分析（半监督学习视角）:")
    print("-" * 40)
    
    # 1. 标签传播路径分析
    labeled_nodes = np.where(labels != -1)[0]
    unlabeled_nodes = np.where(labels == -1)[0]
    
    print(f"有标签节点: {len(labeled_nodes)} ({len(labeled_nodes)/num_nodes*100:.1f}%)")
    print(f"无标签节点: {len(unlabeled_nodes)} ({len(unlabeled_nodes)/num_nodes*100:.1f}%)")
    
    # 2. 连通性分析
    # 合并所有关系的邻接矩阵
    combined_adj = None
    for adj in adj_matrices.values():
        if combined_adj is None:
            combined_adj = adj
        else:
            combined_adj = combined_adj + adj
    
    # 简单的连通性检查
    row, col = combined_adj.nonzero()
    edge_count = len(row)
    density = edge_count / (num_nodes * num_nodes)
    avg_degree = edge_count * 2 / num_nodes
    
    print(f"图密度: {density:.6f}")
    print(f"平均度数: {avg_degree:.2f}")
    
    # 3. 标签-无标签连接分析
    labeled_to_unlabeled_edges = 0
    unlabeled_to_unlabeled_edges = 0
    
    for src, dst in zip(row, col):
        if src in labeled_nodes and dst in unlabeled_nodes:
            labeled_to_unlabeled_edges += 1
        elif src in unlabeled_nodes and dst in unlabeled_nodes:
            unlabeled_to_unlabeled_edges += 1
    
    if len(unlabeled_nodes) > 0:
        labeled_unlabeled_ratio = labeled_to_unlabeled_edges / len(unlabeled_nodes)
        print(f"标签-无标签连接比例: {labeled_unlabeled_ratio:.2f} (每个无标签节点平均连接到标签节点的边数)")
        
        if labeled_unlabeled_ratio < 0.5:
            print("⚠️  警告: 标签-无标签连接过少，可能影响标签传播效果")
        else:
            print("✅ 标签-无标签连接充足，有利于半监督学习")


def main():
    """主函数"""
    
    print("开始创建半监督学习优化的数据集...")
    
    try:
        # 创建修复后的数据集
        output_dir = create_fixed_dataset()
        
        print(f"\n🎉 数据集创建完成!")
        print(f"📁 位置: {output_dir}")
        print(f"\n💡 使用建议:")
        print(f"1. 用这个新数据集重新训练Enhanced CARE-GNN")
        print(f"2. 对比原始数据集vs修复数据集的半监督学习效果")
        print(f"3. 检查alpha参数搜索结果是否有改善")
        print(f"4. 观察注意力权重是否更加稳定")
        
        print(f"\n🔬 预期改进:")
        print(f"- 4种关系类型基于真实交易特征（金额×时间）更有意义")
        print(f"- 半监督学习效果应该显著提升")
        print(f"- 注意力权重数值应该更稳定")  
        print(f"- 不同alpha值之间的性能差异应该更明显")
        print(f"- 图损失应该对模型性能有正向贡献")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()

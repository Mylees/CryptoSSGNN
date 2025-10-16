# -*- coding: utf-8 -*-
"""
简化版传统机器学习对比实验
解决导入问题的最简版本
"""

import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# 直接导入所有需要的库
from scipy.sparse import load_npz, csr_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("📚 Libraries imported successfully!")

def load_and_preprocess_data(data_dir):
    """加载和预处理数据"""
    print(f"📁 Loading data from: {data_dir}")
    
    # 加载特征和标签
    features = load_npz(os.path.join(data_dir, 'features.npz')).toarray()
    labels = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)
    
    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Unique labels: {np.unique(labels)}")
    
    # 过滤有效数据
    valid_mask = labels >= 0
    X = features[valid_mask]
    y = labels[valid_mask]
    
    print(f"✅ Valid samples: {len(X)}")
    print(f"✅ Class distribution: {np.bincount(y)}")
    
    return X, y

def run_ml_models(X, y):
    """运行机器学习模型"""
    print("\n🤖 Starting ML model comparison...")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300)
    }
    
    results = {}
    
    print("\n" + "="*60)
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        try:
            start_time = time.time()
            
            # 训练
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            
            # 获取概率预测 (用于AUC计算)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test_scaled)
                # 将决策分数转换为概率 (sigmoid函数)
                y_prob = 1 / (1 + np.exp(-decision_scores))
            else:
                y_prob = y_pred.astype(float)
            
            # 计算所有评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # 计算AUC
            try:
                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = 0.0
            except Exception as e:
                print(f"   Warning: AUC calculation failed for {name}: {e}")
                auc = 0.0
            
            training_time = time.time() - start_time
            
            # 存储完整指标
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'time': training_time
            }
            
            print(f"✅ {name} completed:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1: {f1:.4f}")
            print(f"   AUC: {auc:.4f}")
            print(f"   Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
            results[name] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 
                'f1': 0, 'auc': 0, 'time': 0, 'error': str(e)
            }
    
    return results

def print_summary(results):
    """打印完整的结果汇总"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    # 过滤掉有错误的结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("❌ No valid results to display!")
        return
    
    # 按F1分数排序
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # 详细结果表格
    print(f"{'Rank':<4} {'Model':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'Time(s)':<8}")
    print("-" * 82)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {name:<18} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f} {metrics['time']:<8.2f}")
    
    # 最佳模型详细信息
    best_model = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best_model[0]}")
    print("="*50)
    print(f"📈 Accuracy:  {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
    print(f"🎯 Precision: {best_model[1]['precision']:.4f}")
    print(f"🔍 Recall:    {best_model[1]['recall']:.4f}")
    print(f"⚖️  F1 Score:  {best_model[1]['f1']:.4f}")
    print(f"📊 AUC:       {best_model[1]['auc']:.4f}")
    print(f"⏱️  Time:      {best_model[1]['time']:.2f} seconds")
    
    # 性能统计
    print(f"\n📊 PERFORMANCE STATISTICS")
    print("="*50)
    
    all_accuracies = [metrics['accuracy'] for metrics in valid_results.values()]
    all_f1s = [metrics['f1'] for metrics in valid_results.values()]
    all_aucs = [metrics['auc'] for metrics in valid_results.values()]
    
    print(f"Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Average F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    print(f"Average AUC:      {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"Best Accuracy:    {np.max(all_accuracies):.4f}")
    print(f"Best F1 Score:    {np.max(all_f1s):.4f}")
    print(f"Best AUC:         {np.max(all_aucs):.4f}")
    
    # 模型比较洞察
    print(f"\n🔍 MODEL COMPARISON INSIGHTS")
    print("="*50)
    
    # 找出在不同指标上表现最佳的模型
    best_acc_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
    best_f1_model = max(valid_results.items(), key=lambda x: x[1]['f1'])
    best_auc_model = max(valid_results.items(), key=lambda x: x[1]['auc'])
    fastest_model = min(valid_results.items(), key=lambda x: x[1]['time'])
    
    print(f"Best Accuracy:  {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.4f})")
    print(f"Best F1 Score:  {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
    print(f"Best AUC:       {best_auc_model[0]} ({best_auc_model[1]['auc']:.4f})")
    print(f"Fastest Model:  {fastest_model[0]} ({fastest_model[1]['time']:.2f}s)")
    
    # 保存详细结果到文件
    try:
        import json
        os.makedirs('ml_results', exist_ok=True)
        
        # 创建详细的结果字典
        detailed_results = {
            'summary': {
                'best_model': best_model[0],
                'best_f1': float(best_model[1]['f1']),
                'num_models_tested': len(valid_results),
                'average_accuracy': float(np.mean(all_accuracies)),
                'average_f1': float(np.mean(all_f1s)),
                'average_auc': float(np.mean(all_aucs))
            },
            'detailed_results': {}
        }
        
        for model_name, metrics in valid_results.items():
            detailed_results['detailed_results'][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'auc': float(metrics['auc']),
                'training_time': float(metrics['time'])
            }
        
        with open('ml_results/comprehensive_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: ml_results/comprehensive_results.json")
        
    except Exception as e:
        print(f"❌ Could not save results: {e}")
    
    # 错误报告（如果有的话）
    error_results = {k: v for k, v in results.items() if 'error' in v}
    if error_results:
        print(f"\n⚠️  MODELS WITH ERRORS")
        print("="*50)
        for model_name, result in error_results.items():
            print(f"❌ {model_name}: {result['error']}")
        

def main():
    """主函数"""
    print("🚀 Simple ML Baseline Comparison")
    print("="*50)
    
    # 修复命令行参数解析
    data_dir = "small_multirel_dataset"  # 默认值
    
    # 简单的参数解析
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--data_dir" and i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
                break
            elif not arg.startswith("--") and arg != sys.argv[0]:
                # 如果是非选项参数，就当作数据目录
                data_dir = arg
                break
    
    print(f"📁 Using data directory: {data_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ Directory '{data_dir}' does not exist!")
        print("Available directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        return
    
    try:
        # 加载数据
        X, y = load_and_preprocess_data(data_dir)
        
        # 运行模型
        results = run_ml_models(X, y)
        
        # 打印结果
        print_summary(results)
        
        print(f"\n✅ Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
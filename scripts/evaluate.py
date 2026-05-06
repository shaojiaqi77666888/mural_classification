"""
壁画主题分类 - 评估脚本
功能：混淆矩阵、各类别F1/精度/召回、可视化
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from configs.config import CLASS_NAMES, IDX_TO_CLASS
from scripts.dataset import create_data_loaders, load_annotations
from scripts.model import get_model


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path = None,
    normalize: bool = True
):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, 0)
        fmt = '.2f'
        title_suffix = '(Normalized)'
    else:
        fmt = 'd'
        title_suffix = '(Count)'
    
    # 只显示实际出现的类别
    present_classes = sorted(set(y_true) | set(y_pred))
    labels = [class_names[i] for i in present_classes]
    cm_display = cm[np.ix_(present_classes, present_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(f'Confusion Matrix {title_suffix}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    plt.show()


def plot_training_history(history_path: Path, save_path: Path = None):
    """绘制训练历史曲线"""
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史已保存: {save_path}")
    plt.show()


def plot_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Path = None
):
    """绘制各类别指标柱状图"""
    present_classes = sorted(set(y_true))
    labels = [class_names[i] for i in present_classes]
    
    precision = precision_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别指标图已保存: {save_path}")
    plt.show()


def evaluate_model(
    model_path: Path,
    test_loader,
    device: str = "cuda",
    results_dir: Path = None
):
    """
    评估模型并生成完整报告
    
    Args:
        model_path: 模型权重路径
        test_loader: 测试数据加载器
        device: 计算设备
        results_dir: 结果保存目录
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint推断类别数
    state_dict = checkpoint['model_state_dict']
    num_classes = state_dict['classifier.6.weight'].shape[0]
    
    model = get_model(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 收集预测
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # 使用实际出现的类别名
    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    
    # 1. 总体指标
    print("\n[总体指标]")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    # 2. 分类报告
    print("\n[分类报告]")
    print("-" * 40)
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))
    
    # 3. 保存详细报告
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩阵
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            save_path=results_dir / 'confusion_matrix.png',
            normalize=True
        )
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            save_path=results_dir / 'confusion_matrix_count.png',
            normalize=False
        )
        
        # 类别指标
        plot_class_metrics(
            y_true, y_pred, CLASS_NAMES,
            save_path=results_dir / 'class_metrics.png'
        )
        
        # 训练历史（如果存在）
        history_path = results_dir / 'training_history.json'
        if history_path.exists():
            plot_training_history(
                history_path,
                save_path=results_dir / 'training_history.png'
            )
        
        # 保存文本报告
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
            output_dict=True
        )
        
        with open(results_dir / 'evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存预测
        np.savez(
            results_dir / 'predictions.npz',
            labels=y_true,
            preds=y_pred,
            probs=y_prob
        )
        
        print(f"\n所有结果已保存到: {results_dir}")
    
    return y_true, y_pred, y_prob


def quick_eval(
    model_path: str,
    json_path: str,
    image_dir: str,
    batch_size: int = 16
):
    """快速评估入口"""
    from configs.config import TrainConfig
    
    model_path = Path(model_path)
    json_path = Path(json_path)
    image_dir = Path(image_dir)
    
    # 加载测试数据
    annotations = load_annotations(json_path)
    
    # 使用测试集划分（假设已保存）
    splits_dir = json_path.parent / "splits"
    if (splits_dir / "test.json").exists():
        with open(splits_dir / "test.json", 'r', encoding='utf-8') as f:
            test_ann = json.load(f)
    else:
        # 如果没有划分文件，使用全部数据
        test_ann = annotations
    
    _, _, test_loader, _ = create_data_loaders(
        [], [], test_ann,
        image_dir=image_dir,
        batch_size=batch_size,
        use_weighted_sampler=False
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = model_path.parent.parent / "results"
    
    evaluate_model(model_path, test_loader, device, results_dir)


if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent
    
    model_path = project_root / "models" / "best_model.pth"
    json_path = project_root / "data" / "annotations.json"
    image_dir = project_root / "data" / "images"
    
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        json_path = Path(sys.argv[2])
    
    quick_eval(str(model_path), str(json_path), str(image_dir))

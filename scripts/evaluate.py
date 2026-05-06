"""
壁画主题分类 - 评估脚本（修复版）
兼容四省合并数据集
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

from configs.config import CLASS_NAMES, IDX_TO_CLASS
from scripts.dataset import create_data_loaders, load_annotations
from scripts.model import get_model


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, 0)
        fmt = '.2f'
    else:
        fmt = 'd'

    present = sorted(set(y_true) | set(y_pred))
    labels = [class_names[i] for i in present]
    cm_display = cm[np.ix_(present, present)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, square=True)
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_training_history(history_path, save_path=None):
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_title('Learning Rate'); axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_metrics(y_true, y_pred, class_names, save_path=None):
    present = sorted(set(y_true))
    labels = [class_names[i] for i in present]

    precision = precision_score(y_true, y_pred, average=None, labels=present, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=present, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=present, zero_division=0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1', color='seagreen')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(); ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model_path, test_loader, device="cuda", results_dir=None):
    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 推断类别数
    if 'classifier.6.weight' in state_dict:
        num_classes = state_dict['classifier.6.weight'].shape[0]
    else:
        num_classes = 10

    model = get_model(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

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

    present = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES[i] for i in present]

    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                              save_path=results_dir / 'confusion_matrix.png', normalize=True)
        plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                              save_path=results_dir / 'confusion_matrix_count.png', normalize=False)
        plot_class_metrics(y_true, y_pred, CLASS_NAMES,
                           save_path=results_dir / 'class_metrics.png')

        history_path = results_dir / 'training_history.json'
        if history_path.exists():
            plot_training_history(history_path, save_path=results_dir / 'training_history.png')

        report = classification_report(y_true, y_pred, target_names=target_names,
                                       digits=4, zero_division=0, output_dict=True)
        with open(results_dir / 'evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        np.savez(results_dir / 'predictions.npz', labels=y_true, preds=y_pred, probs=y_prob)
        print(f"\n结果保存到: {results_dir}")

    return y_true, y_pred, y_prob


def quick_eval(model_path, json_path, image_dir, batch_size=16):
    from configs.config import TrainConfig

    model_path, json_path, image_dir = Path(model_path), Path(json_path), Path(image_dir)

    annotations = load_annotations(json_path)

    splits_dir = json_path.parent / "splits"
    if (splits_dir / "test.json").exists():
        with open(splits_dir / "test.json", 'r', encoding='utf-8') as f:
            test_ann = json.load(f)
    else:
        from scripts.dataset import split_dataset
        _, _, test_ann = split_dataset(annotations)

    _, _, test_loader, _ = create_data_loaders(
        [], [], test_ann, image_dir=image_dir,
        batch_size=batch_size, use_weighted_sampler=False
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

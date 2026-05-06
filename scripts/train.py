"""
壁画主题分类 - 训练脚本（修复版）
关键修复:
1. 修正freeze逻辑：model.py里分类头叫classifier不是fc，原来代码会冻住整个网络
2. 增加两阶段训练：先冻主干练分类头，再解冻微调
3. 增加训练历史记录
4. 修正JSON路径为四省合并版
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from configs.config import TrainConfig, CLASS_NAMES
from scripts.dataset import create_data_loaders, load_annotations, save_splits, split_dataset
from scripts.model import get_model


# ================= 早停 =================
class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        return False

    def restore(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


# ================= 训练一个epoch =================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


# ================= 验证 =================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100 * correct / total, np.array(all_preds), np.array(all_labels)


# ================= 主训练 =================
def train(data_dir, image_dir, json_path, config):
    print("=" * 60)
    print("壁画主题分类 - ResNet50 训练 (四省合并版)")
    print("=" * 60)

    # ---------- 数据 ----------
    print("\n[1/5] 加载数据...")
    annotations = load_annotations(json_path)
    print(f"总样本: {len(annotations)}")

    train_ann, val_ann, test_ann = split_dataset(annotations)
    save_splits(train_ann, val_ann, test_ann, Path(data_dir) / "splits")

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_ann, val_ann, test_ann, image_dir,
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # ---------- 模型 ----------
    print("\n[2/5] 初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        device=str(device)
    )

    # ---- 关键修复：正确的freeze逻辑 ----
    # model.py里backbone.fc已经被替换为Identity，真正的分类头是model.classifier
    # 原来代码 if not name.startswith("fc") 会冻住classifier所有层，导致无法训练！

    # 阶段1：冻结backbone，只训练分类头
    print("\n[阶段1] 冻结主干网络，训练分类头...")
    for name, param in model.named_parameters():
        if "classifier" in name:  # 只解冻分类头
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ---------- 损失 ----------
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ---------- 阶段1优化器 ----------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR * 3,   # 阶段1用较高学习率
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    early_stop = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)

    # ---------- 训练循环 ----------
    print(f"\n[3/5] 开始训练 (最多 {config.EPOCHS} epochs)...")
    print("-" * 60)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    writer = SummaryWriter(Path(data_dir).parent / "logs") if HAS_TENSORBOARD else None

    # 阶段1训练（冻结主干）
    freeze_epochs = getattr(config, 'FREEZE_EPOCHS', 10)

    for epoch in range(1, config.EPOCHS + 1):
        # 阶段2：解冻主干（在freeze_epochs之后）
        if epoch == freeze_epochs + 1:
            print(f"\n>>> 阶段2：解冻主干网络，全网络微调...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # 重新设置优化器（分层学习率）
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config.LR * 0.1},
                {"params": model.classifier.parameters(), "lr": config.LR}
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - freeze_epochs, eta_min=1e-6)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
              f"LR {current_lr:.2e}")

        # 保存最佳
        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = Path(data_dir).parent / "models"
            save_dir.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": {k: v for k, v in vars(config).items() if not k.startswith("_")}
            }, save_dir / "best_model.pth")

        # 早停
        if early_stop(val_loss, model):
            print(f"\n早停触发 (patience={config.EARLY_STOP_PATIENCE})")
            break

    if writer:
        writer.close()

    # 恢复最佳
    early_stop.restore(model)

    # 保存训练历史
    results_dir = Path(data_dir).parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # ---------- 测试 ----------
    print("\n[4/5] 最终测试评估...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # 保存测试预测
    np.savez(results_dir / "test_predictions.npz",
             preds=test_preds, labels=test_labels)

    # 各类别准确率
    from sklearn.metrics import classification_report, confusion_matrix
    present_classes = sorted(set(test_labels))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    print("\n[5/5] 分类报告:")
    print(classification_report(test_labels, test_preds, target_names=target_names, digits=4, zero_division=0))

    print("\n" + "=" * 60)
    print(f"训练完成! Best Val Acc: {best_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print(f"模型保存: models/best_model.pth")
    print("=" * 60)
    return model


# ================= 入口 =================
if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"  # 四省合并版

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"[错误] 标注文件不存在: {json_path}")
        sys.exit(1)

    train(data_dir, image_dir, json_path, TrainConfig())

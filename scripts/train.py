import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from configs.config import TrainConfig
from scripts.dataset import create_data_loaders, load_annotations, save_splits, split_dataset
from scripts.model import get_model


# ================= 早停 =================
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = 1e9
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_model:
            model.load_state_dict(self.best_model)


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

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


# ================= 主训练 =================
def train(data_dir, image_dir, json_path, config):
    print("===== 开始训练 =====")

    # ---------- 数据 ----------
    annotations = load_annotations(json_path)

    train_ann, val_ann, test_ann = split_dataset(annotations)

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_ann, val_ann, test_ann, image_dir
    )

    # ---------- 模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        num_classes=TrainConfig.NUM_CLASSES,
        pretrained=True,
        device=str(device)
    )

    # 🔥 小数据：冻结 backbone（非常关键）
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    # ---------- 损失 ----------
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ---------- 优化器 ----------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,   # 🔥 提高学习率
        weight_decay=1e-4
    )

    # ---------- 调度器 ----------
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TrainConfig.EPOCHS,
        eta_min=1e-5   # 🔥 防止学习率太低
    )

    early_stop = EarlyStopping(patience=15)

    best_acc = 0

    # ================= 训练循环 =================
    for epoch in range(TrainConfig.EPOCHS):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Epoch {epoch+1} | "
              f"Train Acc {train_acc:.2f}% | "
              f"Val Acc {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")

        # 早停
        if early_stop(val_loss, model):
            print("早停触发")
            break

    # 恢复最佳模型
    early_stop.restore(model)

    # ================= 测试 =================
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print("===== 训练完成 =====")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")

    return model


# ================= 入口 =================
if __name__ == "__main__":

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "henan_final_categorized.json"

    train(data_dir, image_dir, json_path)
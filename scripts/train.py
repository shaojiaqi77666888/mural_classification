"""
壁画主题分类 - 训练脚本（10类强化版）
核心改进（不改变类别数）:
1. Focal Loss 替代 CrossEntropy - 专门解决类别不平衡
2. 更激进的类别权重 - 1/count 而非 1/sqrt(count)
3. 更高的学习率 - 阶段2 backbone 1e-4, head 1e-3
4. 更长的冻结阶段 - 20 epochs
5. 更强的数据增强 + 随机擦除
6. 更大的 batch_size - RTX 4060 可以上 24
7. Label Smoothing 降低到 0.05（壁画类别边界模糊，不要太软化）
8. 增加 Mixup 数据增强
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from configs.config import TrainConfig, CLASS_NAMES
from scripts.dataset import create_data_loaders, load_annotations, save_splits, split_dataset
from scripts.model import get_model


# ================= Focal Loss =================
class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    对难分类样本加大权重，对易分类样本降低权重
    特别适合类别极度不平衡的情况
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数，越大对易分样本抑制越强
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        
        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        # 加上类别权重 alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ================= 早停 =================
class EarlyStopping:
    def __init__(self, patience=25, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None

    def __call__(self, val_acc, model):
        score = val_acc
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


# ================= CutMix =================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================= 训练一个epoch =================
def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Mixup
        if use_mixup and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
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
    print("壁画主题分类 - ResNet50 训练 (10类强化版)")
    print("改进: Focal Loss + Mixup + 更高LR + 更强增强")
    print("=" * 60)

    # ---------- 数据 ----------
    print("\n[1/5] 加载数据...")
    annotations = load_annotations(json_path)
    print(f"总样本: {len(annotations)}")

    # 打印类别分布
    from collections import Counter
    cats = Counter(item['category'] for item in annotations)
    print("类别分布:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    train_ann, val_ann, test_ann = split_dataset(annotations)
    save_splits(train_ann, val_ann, test_ann, Path(data_dir) / "splits")

    img_size = getattr(config, 'IMG_SIZE', 448)  # ← 加这行
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_ann, val_ann, test_ann, image_dir,
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        img_size=img_size  # ← 加这个参数
    )

    # ---------- 模型 ----------
    print("\n[2/5] 初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    dropout = getattr(config, 'DROPOUT', 0.7)
    model = get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout=dropout,  # ← 加上这行
        device=str(device)
    )

    # ---- 阶段1：冻结主干 ----
    freeze_epochs = getattr(config, 'FREEZE_EPOCHS', 20)
    print(f"\n[阶段1] 冻结主干 {freeze_epochs} epochs，只训练分类头...")
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ---------- 损失函数：Focal Loss + 类别权重 ----------
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # 用Focal Loss替代CrossEntropy
    # gamma=2.0 对易分样本抑制适中
    # alpha 用类别权重
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"使用 Focal Loss (gamma=2.0) + 类别权重")

    # ---------- 阶段1优化器（高学习率） ----------
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW(
        classifier_params,
        lr=config.LR * 5,      # 1.5e-3
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs, eta_min=1e-5)
    early_stop = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)

    # ---------- 训练循环 ----------
    print(f"\n[3/5] 开始训练 (最多 {config.EPOCHS} epochs)...")
    print("-" * 60)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    writer = SummaryWriter(Path(data_dir).parent / "logs") if HAS_TENSORBOARD else None

    for epoch in range(1, config.EPOCHS + 1):
        # 阶段2：解冻主干
        if epoch == freeze_epochs + 1:
            print(f"\n>>> 阶段2：解冻主干，全网络微调...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # 分层学习率：主干1e-4，分类头1e-3
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config.LR},
                {"params": model.classifier.parameters(), "lr": config.LR * 10}
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS - freeze_epochs, eta_min=1e-6
            )

        # 前30轮使用Mixup，之后不用
        use_mixup = epoch <= 30
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup
        )
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
                "val_acc": val_acc,
            }, save_dir / "best_model.pth")
            print(f"  >> 最佳模型保存 (Val Acc: {best_acc:.2f}%)")

        # 早停
        if early_stop(val_acc, model):
            print(f"\n早停触发 (patience={config.EARLY_STOP_PATIENCE})")
            break

    if writer:
        writer.close()
    early_stop.restore(model)

    # 保存历史
    results_dir = Path(data_dir).parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # ---------- 测试 ----------
    print("\n[4/5] 最终测试评估...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    np.savez(results_dir / "test_predictions.npz", preds=test_preds, labels=test_labels)

    # 各类别详细报告
    from sklearn.metrics import classification_report, confusion_matrix
    present = sorted(set(test_labels))
    target_names = [CLASS_NAMES[i] for i in present]
    print("\n[5/5] 分类报告:")
    print(classification_report(test_labels, test_preds, target_names=target_names, digits=4, zero_division=0))

    # 打印混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    print("\n混淆矩阵:")
    print("        ", end="")
    for i in present:
        print(f"{CLASS_NAMES[i][:4]:6s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:4]:6s}", end="")
        for val in row:
            print(f"{val:6d}", end="")
        print()

    print("\n" + "=" * 60)
    print(f"训练完成! Best Val Acc: {best_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print("=" * 60)
    return model


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"[错误] 标注文件不存在: {json_path}")
        sys.exit(1)

    train(data_dir, image_dir, json_path, TrainConfig())

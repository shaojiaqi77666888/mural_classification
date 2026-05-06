"""
壁画主题分类 - 数据集处理模块（优化版）
支持：数据加载、增强、划分、采样
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from configs.config import CLASS_TO_IDX, TrainConfig


# ================= 数据集 =================
class MuralDataset(Dataset):
    def __init__(
        self,
        annotations: List[Dict],
        image_dir: Path,
        transform=None,
        phase: str = "train"
    ):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform
        self.phase = phase

        self.valid_annotations = []
        for item in annotations:
            img_path = self.image_dir / item["image"]
            if img_path.exists():
                self.valid_annotations.append(item)
            else:
                # 自动修复后缀
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                    alt = self.image_dir / (Path(item["image"]).stem + ext)
                    if alt.exists():
                        item = item.copy()
                        item["image"] = alt.name
                        self.valid_annotations.append(item)
                        break

        print(f"[{phase}] 有效样本: {len(self.valid_annotations)} / {len(annotations)}")

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        item = self.valid_annotations[idx]

        img_path = self.image_dir / item["image"]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[item["category"]]
        return image, label

    def get_labels(self):
        return [CLASS_TO_IDX[item["category"]] for item in self.valid_annotations]


# ================= 数据增强 =================
def get_transforms(phase="train", img_size=224):

    if phase == "train" and TrainConfig.AUGMENT:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),

            transforms.RandomCrop(img_size),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),

            transforms.RandomRotation(20),

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ================= 数据划分 =================
def split_dataset(
    annotations: List[Dict],
    test_size=0.15,
    val_size=0.15,
    seed=42
):
    labels = [CLASS_TO_IDX[item["category"]] for item in annotations]

    train_val, test = train_test_split(
        annotations,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=[CLASS_TO_IDX[item["category"]] for item in train_val],
        random_state=seed
    )

    print(f"数据划分: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


# ================= DataLoader =================
def create_data_loaders(
    train_ann,
    val_ann,
    test_ann,
    image_dir,
    batch_size=16,
    num_workers=0,
    use_weighted_sampler=True
):

    train_dataset = MuralDataset(train_ann, image_dir, get_transforms("train"), "train")
    val_dataset = MuralDataset(val_ann, image_dir, get_transforms("val"), "val")
    test_dataset = MuralDataset(test_ann, image_dir, get_transforms("test"), "test")

    sampler = None
    class_weights = None

    if use_weighted_sampler:
        labels = train_dataset.get_labels()
        class_counts = np.bincount(labels)

        weights = 1.0 / np.sqrt(class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)

        class_weights = torch.FloatTensor(weights)

        sample_weights = [weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print("类别权重:", weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_weights


# ================= JSON =================
def load_annotations(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_splits(train, val, test, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, data in zip(["train", "val", "test"], [train, val, test]):
        with open(save_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已保存: {name}.json")
"""
壁画主题分类 - 数据集处理模块（10类强化版）
核心改进:
1. 更强数据增强: RandomResizedCrop + 随机擦除 + AutoAugment
2. 支持Mixup
3. 递归子目录搜索
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from configs.config import CLASS_TO_IDX, TrainConfig


class MuralDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None, phase="train"):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform
        self.phase = phase

        self.file_map = {}
        self._scan_image_dir(image_dir)

        self.valid_annotations = []
        for item in annotations:
            resolved = self._resolve_image_path(item["image"])
            if resolved is not None:
                item = item.copy()
                item["_resolved_path"] = str(resolved)
                self.valid_annotations.append(item)
            else:
                found = False
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                    alt = self.image_dir / (Path(item["image"]).stem + ext)
                    if alt.exists():
                        item = item.copy()
                        item["image"] = alt.name
                        item["_resolved_path"] = str(alt)
                        self.valid_annotations.append(item)
                        found = True
                        break
                if not found:
                    resolved = self._resolve_image_path(Path(item["image"]).stem)
                    if resolved is not None:
                        item = item.copy()
                        item["image"] = resolved.name
                        item["_resolved_path"] = str(resolved)
                        self.valid_annotations.append(item)
                    else:
                        print(f"[警告] 图片不存在: {item['image']}")

        print(f"[{phase}] 有效样本: {len(self.valid_annotations)} / {len(annotations)}")

    def _scan_image_dir(self, image_dir):
        if not image_dir.exists():
            return
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.gif", "*.GIF"]:
            for f in image_dir.rglob(ext):
                if f.name not in self.file_map:
                    self.file_map[f.name] = f

    def _resolve_image_path(self, image_name):
        if image_name in self.file_map:
            return self.file_map[image_name]
        for fname, fpath in self.file_map.items():
            if fname.lower() == image_name.lower():
                return fpath
        return None

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        item = self.valid_annotations[idx]
        try:
            image = Image.open(Path(item["_resolved_path"])).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_IDX.get(item["category"], 0)
        return image, label

    def get_labels(self):
        return [CLASS_TO_IDX.get(item["category"], 0) for item in self.valid_annotations]


def get_transforms(phase="train", img_size=448):
    if phase == "train" and TrainConfig.AUGMENT:
        return transforms.Compose([
            # 随机缩放裁剪 - 比RandomCrop更强
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            # 强颜色抖动
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 随机擦除 - 模拟壁画损坏/剥落
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def split_dataset(annotations, test_size=0.15, val_size=0.15, seed=42):
    labels = [CLASS_TO_IDX.get(item["category"], 0) for item in annotations]
    train_val, test = train_test_split(annotations, test_size=test_size, stratify=labels, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size),
        stratify=[CLASS_TO_IDX.get(item["category"], 0) for item in train_val],
        random_state=seed
    )
    print(f"数据划分: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def create_data_loaders(train_ann, val_ann, test_ann, image_dir,
                       batch_size=12, num_workers=0, use_weighted_sampler=True,
                       img_size=448):
    train_dataset = MuralDataset(train_ann, image_dir, get_transforms("train", img_size), "train")
    val_dataset = MuralDataset(val_ann, image_dir, get_transforms("val", img_size), "val")
    test_dataset = MuralDataset(test_ann, image_dir, get_transforms("test", img_size), "test")

    sampler = None
    class_weights = None

    if use_weighted_sampler:
        labels = train_dataset.get_labels()
        class_counts = np.bincount(labels, minlength=TrainConfig.NUM_CLASSES)
        
        # 更激进的权重: 1/count (原来用 1/sqrt(count))
        weights = np.zeros(TrainConfig.NUM_CLASSES)
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = 1.0 / count
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.FloatTensor(weights)

        sample_weights = [weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print("类别权重:", {i: f"{w:.3f}" for i, w in enumerate(weights)})

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights


def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_splits(train, val, test, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, data in zip(["train", "val", "test"], [train, val, test]):
        with open(save_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存: {name}.json")

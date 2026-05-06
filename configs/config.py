"""
壁画主题分类 - 配置文件（10类原版）
关键调整:
1. 保持10大类别不变
2. 提高batch_size到24（RTX 4060 8GB可以hold住）
3. 延长冻结阶段到20 epochs
4. 提高学习率到3e-4
5. 增加早停耐心到25
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
JSON_PATH = DATA_DIR / "annotations.json"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

class TrainConfig:
    IMG_SIZE = 224
    BATCH_SIZE = 24            # ← RTX 4060 可以上24
    NUM_WORKERS = 4
    
    NUM_CLASSES = 10             # ← 保持10类不变
    BACKBONE = "resnet50"
    PRETRAINED = True
    
    EPOCHS = 150
    LR = 3e-4                    # ← 提高3倍
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 25     # ← 更耐心的早停
    
    FREEZE_EPOCHS = 20           # ← 冻结更久
    
    LR_SCHEDULER = "cosine"
    AUGMENT = True
    
    DEVICE = "cuda"
    SEED = 42

# 10大类别（不变）
CLASS_NAMES = [
    "人物肖像",      # 0
    "神话升仙",      # 1
    "礼仪文化",      # 2
    "车马出行",      # 3
    "乐舞百戏",      # 4
    "花鸟装饰",      # 5
    "生产经济",      # 6
    "天文星象",      # 7
    "宗教题材",      # 8
    "建筑场景",      # 9
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

NUM_CLASSES = 10

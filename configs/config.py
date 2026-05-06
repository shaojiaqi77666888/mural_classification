"""
ResNet50 壁画主题分类 - 配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"          # 壁画图片存放目录
JSON_PATH = DATA_DIR / "annotations.json"  # 标注文件路径
SPLITS_DIR = DATA_DIR / "splits"         # 数据划分文件

# 模型与日志路径
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# 训练参数
class TrainConfig:
    # 数据参数
    IMG_SIZE = 224                # 输入图像尺寸
    BATCH_SIZE = 16               # 批量大小（根据显存调整）
    NUM_WORKERS = 4               # 数据加载线程数
    
    # 模型参数
    NUM_CLASSES = 10              # 壁画主题类别数
    BACKBONE = "resnet50"         # 主干网络
    PRETRAINED = True             # 是否使用ImageNet预训练权重
    
    # 训练参数
    EPOCHS = 100                  # 最大训练轮数
    LR = 1e-4                     # 初始学习率
    WEIGHT_DECAY = 1e-4           # 权重衰减
    EARLY_STOP_PATIENCE = 15      # 早停耐心值
    
    # 学习率调度
    LR_SCHEDULER = "cosine"       # cosine / step / plateau
    WARMUP_EPOCHS = 5             # 预热轮数
    
    # 数据增强
    AUGMENT = True                # 是否使用数据增强
    MIXUP_ALPHA = 0.2             # Mixup参数（0表示不使用）
    CUTMIX_ALPHA = 0.0            # Cutmix参数（0表示不使用）
    
    # 设备
    DEVICE = "cuda"               # cuda / cpu
    SEED = 42                     # 随机种子

# 类别数量
NUM_CLASSES = 10

# 类别映射
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

# 类别权重（用于处理类别不平衡）
# 将根据实际数据分布自动计算
CLASS_WEIGHTS = None

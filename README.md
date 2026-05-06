# 壁画主题分类 - ResNet50

基于 ResNet50 的中国古代墓葬壁画主题自动分类系统，支持 10 大主题类别。

## 项目结构

```
mural_classification/
├── configs/                  # 配置文件
│   └── config.py             # 训练参数、类别映射
├── scripts/                  # 核心模块
│   ├── dataset.py            # 数据集处理、增强、采样
│   ├── model.py              # ResNet50 模型定义
│   ├── train.py              # 训练循环、早停、日志
│   ├── evaluate.py           # 评估指标、可视化
│   └── predict.py            # 单图/批量预测
├── data/                     # 数据目录
│   ├── images/               # 壁画图片（需自行放置）
│   ├── splits/               # 数据集划分
│   └── annotations.json      # 标注文件（由prepare命令复制）
├── models/                   # 模型权重保存
├── logs/                     # TensorBoard 日志
├── results/                  # 评估结果、图表
├── run.py                    # 统一入口脚本
└── requirements.txt          # 依赖
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将壁画图片放入 `data/images/` 目录，并准备标注JSON文件（格式见下方）。

```bash
python run.py prepare --json henan_final_categorized.json --images ./壁画图片目录
```

JSON 格式示例（每条记录必须包含 `image` 和 `category` 字段）：

```json
[
  {
    "image": "henan-001.JPG",
    "title": "青龙壁画",
    "category": "神话升仙",
    ...
  }
]
```

### 3. 训练模型

```bash
python run.py train --epochs 100 --batch-size 16
```

训练参数（可在 `configs/config.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IMG_SIZE | 224 | 输入图像尺寸 |
| BATCH_SIZE | 16 | 批量大小 |
| EPOCHS | 100 | 最大训练轮数 |
| LR | 1e-4 | 初始学习率 |
| EARLY_STOP_PATIENCE | 15 | 早停耐心值 |
| PRETRAINED | True | 使用ImageNet预训练权重 |

**关键特性：**
- 分层学习率：主干网络 `LR×0.1`，分类头 `LR`
- 加权随机采样：自动处理类别不平衡
- 余弦退火学习率调度
- 早停机制 + 梯度裁剪
- 数据增强：随机裁剪、翻转、旋转、颜色抖动、仿射变换

### 4. 评估模型

```bash
python run.py eval
```

输出内容：
- 混淆矩阵热力图（归一化 + 计数）
- 各类别 Precision / Recall / F1 柱状图
- 训练历史曲线（Loss / Accuracy / LR）
- 详细JSON报告

### 5. 预测新图片

```bash
python run.py predict image.jpg --top_k 3
```

输出示例：
```
图片: image.jpg
预测结果:
  1. 神话升仙     0.9234 ██████████████████████████████
  2. 天文星象     0.0456 ██
  3. 人物肖像     0.0189 █
```

## 主题类别定义

| ID | 类别 | 说明 |
|----|------|------|
| 0 | 人物肖像 | 墓主、官吏、侍从、门吏等 |
| 1 | 神话升仙 | 四神、伏羲女娲、羽人、升仙图等 |
| 2 | 礼仪文化 | 仪仗、拜谒、跪拜等礼仪场景 |
| 3 | 车马出行 | 出行队列、轺车、导骑等 |
| 4 | 乐舞百戏 | 舞蹈、伎乐、狩猎、斗鸡等 |
| 5 | 花鸟装饰 | 花卉、鸟类、装饰图案等 |
| 6 | 生产经济 | 农耕、庖厨、宰牲、备茶等 |
| 7 | 天文星象 | 日月、星宿、金乌、蟾蜍等 |
| 8 | 宗教题材 | 礼佛、涅槃、天王等 |
| 9 | 建筑场景 | 楼阁、院落、家具陈设等 |

## 数据分布（河南数据集示例）

| 类别 | 数量 | 占比 |
|------|------|------|
| 人物肖像 | 74 | 33.0% |
| 神话升仙 | 45 | 20.1% |
| 礼仪文化 | 27 | 12.1% |
| 车马出行 | 21 | 9.4% |
| 乐舞百戏 | 15 | 6.7% |
| 花鸟装饰 | 14 | 6.2% |
| 生产经济 | 12 | 5.4% |
| 天文星象 | 9 | 4.0% |
| 宗教题材 | 4 | 1.8% |
| 建筑场景 | 3 | 1.3% |

## 技术细节

### 模型架构
- **主干网络**: ResNet50 (ImageNet 预训练)
- **分类头**: Linear(2048→512) → BN → ReLU → Dropout → Linear(512→256) → BN → ReLU → Dropout → Linear(256→10)
- **参数量**: ~24M (可训练 ~1M 分类头参数)

### 训练策略
1. **阶段1**（前10 epoch）：冻结主干，仅训练分类头
2. **阶段2**（后续）：解冻主干，使用分层学习率微调
3. 类别不平衡处理：加权随机采样 + 加权交叉熵损失
4. 标签平滑（Label Smoothing = 0.1）

### 数据增强
- RandomCrop + RandomHorizontalFlip
- RandomRotation(±15°)
- ColorJitter（亮度/对比度/饱和度/色调）
- RandomAffine（平移 + 缩放）

### 监控
```bash
tensorboard --logdir logs/
```

## 扩展其他省份

本项目框架支持任意省份数据集，只需：

1. 准备带 `category` 字段的 JSON 标注文件
2. 将图片放入 `data/images/`
3. 更新 `configs/config.py` 中的 `CLASS_NAMES`（如有不同分类体系）
4. 运行 `python run.py prepare --json 新标注.json`
5. 运行 `python run.py train`

## 许可证

本项目用于学术研究，壁画数据版权归各博物馆/考古机构所有。

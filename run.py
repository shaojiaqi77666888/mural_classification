#!/usr/bin/env python3
"""
壁画主题分类 - 统一入口脚本
整合：数据准备、训练、评估、预测
"""
import argparse
import shutil
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TrainConfig
from scripts.dataset import create_data_loaders, load_annotations, save_splits, split_dataset
from scripts.evaluate import evaluate_model, plot_training_history
from scripts.model import get_model
from scripts.predict import MuralPredictor
from scripts.train import train


def cmd_prepare(args):
    """准备数据：复制JSON标注，检查图片"""
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"
    
    print("=" * 60)
    print("数据准备")
    print("=" * 60)
    
    # 如果指定了JSON路径，复制到项目目录
    if args.json and Path(args.json).exists():
        src_json = Path(args.json)
        shutil.copy2(src_json, json_path)
        print(f"[OK] 标注文件已复制: {src_json} -> {json_path}")
    
    if not json_path.exists():
        print(f"[错误] 标注文件不存在: {json_path}")
        print("请使用 --json 指定标注文件路径")
        return 1
    
    # 加载并检查
    annotations = load_annotations(json_path)
    print(f"[OK] 加载 {len(annotations)} 条标注")
    
    # 检查图片
    if args.images:
        src_img_dir = Path(args.images)
        if src_img_dir.exists():
            image_dir.mkdir(parents=True, exist_ok=True)
            # 复制图片
            img_files = list(src_img_dir.glob('*.jpg')) + \
                       list(src_img_dir.glob('*.png')) + \
                       list(src_img_dir.glob('*.jpeg')) + \
                       list(src_img_dir.glob('*.JPG')) + \
                       list(src_img_dir.glob('*.PNG'))
            for f in img_files:
                dst = image_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
            print(f"[OK] 已复制 {len(img_files)} 张图片到 {image_dir}")
    
    # 检查匹配情况
    if image_dir.exists():
        missing = 0
        found = 0
        for item in annotations:
            img_path = image_dir / item['image']
            if not img_path.exists():
                # 尝试不同扩展名
                found_alt = False
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    alt = image_dir / (Path(item['image']).stem + ext)
                    if alt.exists():
                        found_alt = True
                        break
                if not found_alt:
                    missing += 1
            else:
                found += 1
        
        print(f"[OK] 图片匹配: {found} 成功, {missing} 缺失")
        if missing > 0:
            print(f"[提示] 请将缺失的图片放入: {image_dir}")
    else:
        print(f"[提示] 图片目录不存在，请创建并放入图片: {image_dir}")
    
    # 类别分布
    from collections import Counter
    cats = Counter(item['category'] for item in annotations)
    print("\n类别分布:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    
    return 0


def cmd_train(args):
    """训练模型"""
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"
    
    if not json_path.exists():
        print(f"[错误] 标注文件不存在，请先运行: python run.py prepare --json <path>")
        return 1
    
    if not image_dir.exists() or not any(image_dir.iterdir()):
        print(f"[错误] 图片目录为空，请先放入图片: {image_dir}")
        return 1
    
    # 更新配置
    config = TrainConfig()
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LR = args.lr
    if args.no_pretrained:
        config.PRETRAINED = False
    
    # 启动训练
    train(data_dir, image_dir, json_path, config)
    return 0


def cmd_eval(args):
    """评估模型"""
    from scripts.evaluate import quick_eval
    
    model_path = args.model or PROJECT_ROOT / "models" / "best_model.pth"
    json_path = PROJECT_ROOT / "data" / "annotations.json"
    image_dir = PROJECT_ROOT / "data" / "images"
    
    quick_eval(str(model_path), str(json_path), str(image_dir))
    return 0


def cmd_predict(args):
    """预测单图"""
    model_path = args.model or PROJECT_ROOT / "models" / "best_model.pth"
    
    if not Path(model_path).exists():
        print(f"[错误] 模型不存在: {model_path}")
        return 1
    
    predictor = MuralPredictor(
        model_path=Path(model_path),
        device=args.device
    )
    
    image_path = Path(args.image)
    result = predictor.predict(image_path, top_k=args.top_k)
    
    print(f"\n图片: {image_path}")
    print("预测结果:")
    for i, (cls, prob) in enumerate(result['top_k'], 1):
        bar = '█' * int(prob * 30)
        print(f"  {i}. {cls:12s} {prob:.4f} {bar}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='壁画主题分类 - ResNet50',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 准备数据
  python run.py prepare --json henan_final_categorized.json --images ./pics
  
  # 训练
  python run.py train --epochs 80 --batch-size 16
  
  # 评估
  python run.py eval
  
  # 预测单图
  python run.py predict image.jpg --top_k 3
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # prepare
    p_prepare = subparsers.add_parser('prepare', help='准备数据')
    p_prepare.add_argument('--json', type=str, help='标注JSON文件路径')
    p_prepare.add_argument('--images', type=str, help='图片目录路径')
    
    # train
    p_train = subparsers.add_parser('train', help='训练模型')
    p_train.add_argument('--epochs', type=int, help='训练轮数')
    p_train.add_argument('--batch-size', type=int, help='批量大小')
    p_train.add_argument('--lr', type=float, help='学习率')
    p_train.add_argument('--no-pretrained', action='store_true', help='不使用预训练权重')
    
    # eval
    p_eval = subparsers.add_parser('eval', help='评估模型')
    p_eval.add_argument('--model', type=str, help='模型路径（默认models/best_model.pth）')
    
    # predict
    p_predict = subparsers.add_parser('predict', help='预测图片')
    p_predict.add_argument('image', type=str, help='图片路径')
    p_predict.add_argument('--model', type=str, help='模型路径')
    p_predict.add_argument('--top_k', type=int, default=3)
    p_predict.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        'prepare': cmd_prepare,
        'train': cmd_train,
        'eval': cmd_eval,
        'predict': cmd_predict,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

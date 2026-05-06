#!/usr/bin/env python3
"""
壁画主题分类 - 统一入口脚本（四省合并修复版）
关键修复:
1. prepare支持递归搜索子目录（彩色_河北/彩色_河南等）
2. prepare支持逗号分隔多目录
3. train增加--freeze-epochs参数控制解冻时机
4. 默认加载四省合并的annotations.json
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TrainConfig
from scripts.dataset import create_data_loaders, load_annotations, save_splits, split_dataset
from scripts.evaluate import evaluate_model
from scripts.model import get_model
from scripts.predict import MuralPredictor
from scripts.train import train


def cmd_prepare(args):
    """准备数据：支持递归子目录和多目录"""
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"

    print("=" * 60)
    print("数据准备 - 四省壁画合并版")
    print("=" * 60)

    # 复制JSON
    if args.json and Path(args.json).exists():
        src_json = Path(args.json)
        shutil.copy2(src_json, json_path)
        print(f"[OK] 标注文件已复制: {src_json} -> {json_path}")

    if not json_path.exists():
        print(f"[错误] 标注文件不存在: {json_path}")
        return 1

    annotations = load_annotations(json_path)
    print(f"[OK] 加载 {len(annotations)} 条标注")

    # ---- 关键修复：递归搜索子目录并复制图片 ----
    if args.images:
        src_dirs = [p.strip() for p in args.images.split(",")]
        image_dir.mkdir(parents=True, exist_ok=True)
        total_copied = 0

        for src_str in src_dirs:
            src_path = Path(src_str)
            if not src_path.exists():
                print(f"[跳过] 目录不存在: {src_path}")
                continue

            # 递归搜索所有子目录中的图片
            img_exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.gif', '*.GIF')
            img_files = []
            for ext in img_exts:
                img_files.extend(list(src_path.rglob(ext)))

            # 去重：按文件名去重
            seen_names = set()
            for f in img_files:
                if f.name in seen_names:
                    continue
                seen_names.add(f.name)
                dst = image_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    total_copied += 1

            print(f"[OK] {src_path}: 找到 {len(img_files)} 张，复制 {total_copied} 张新图片")

    # 检查匹配情况（递归搜索子目录）
    if image_dir.exists():
        from scripts.dataset import MuralDataset
        # 用Dataset的扫描逻辑测试匹配
        test_ds = MuralDataset(annotations[:10], image_dir, None, "check")
        # 这里只是为了触发扫描
        del test_ds

        # 手动统计
        file_map = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for f in image_dir.rglob(ext):
                file_map[f.name] = f

        missing = 0
        found = 0
        for item in annotations:
            img_name = item['image']
            if img_name in file_map:
                found += 1
            else:
                # 尝试不同扩展名
                stem = Path(img_name).stem
                alt_found = False
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    if (stem + ext) in file_map:
                        alt_found = True
                        break
                if not alt_found:
                    missing += 1

        print(f"[OK] 图片匹配: {found} 成功, {missing} 缺失")
        if missing > 0:
            print(f"[提示] 缺失图片请放入: {image_dir}")
    else:
        print(f"[提示] 图片目录不存在: {image_dir}")

    # 类别分布
    from collections import Counter
    cats = Counter(item['category'] for item in annotations)
    print("\n类别分布:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # 省份分布
    provinces = Counter()
    for item in annotations:
        img = item.get('image', '')
        if img.startswith('hebei'): provinces['河北'] += 1
        elif img.startswith('henan'): provinces['河南'] += 1
        elif img.startswith('shanxi'): provinces['山西'] += 1
        elif img.startswith('shandong'): provinces['山东'] += 1
    if provinces:
        print("\n省份分布:")
        for p, c in provinces.most_common():
            print(f"  {p}: {c}")

    return 0


def cmd_train(args):
    """训练模型"""
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "images"
    json_path = data_dir / "annotations.json"

    if not json_path.exists():
        print(f"[错误] 标注文件不存在: {json_path}")
        print("请先运行: python run.py prepare --json <path> --images <path>")
        return 1

    if not image_dir.exists() or not any(image_dir.iterdir()):
        print(f"[错误] 图片目录为空: {image_dir}")
        return 1

    config = TrainConfig()
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LR = args.lr
    if args.no_pretrained:
        config.PRETRAINED = False
    if args.freeze_epochs is not None:
        config.FREEZE_EPOCHS = args.freeze_epochs

    print(f"训练配置: epochs={config.EPOCHS}, batch={config.BATCH_SIZE}, lr={config.LR}")
    print(f"阶段1冻结主干训练 {config.FREEZE_EPOCHS} epochs，之后全网络微调")

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

    predictor = MuralPredictor(model_path=Path(model_path), device=args.device)
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
        description='壁画主题分类 - ResNet50 (四省合并版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 准备数据（四省图片在各自子目录）
  python run.py prepare --json data/annotations.json --images "E:/壁画图片"

  # 或四省图片在不同位置
  python run.py prepare --json data/annotations.json --images "E:/河北,E:/河南,E:/山西,E:/山东"

  # 训练（先冻10epoch再解冻）
  python run.py train --epochs 100 --batch-size 16 --freeze-epochs 10

  # 评估
  python run.py eval

  # 预测
  python run.py predict image.jpg --top_k 3
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # prepare
    p_prepare = subparsers.add_parser('prepare', help='准备数据')
    p_prepare.add_argument('--json', type=str, help='标注JSON文件路径')
    p_prepare.add_argument('--images', type=str, help='图片目录路径（逗号分隔多个目录）')

    # train
    p_train = subparsers.add_parser('train', help='训练模型')
    p_train.add_argument('--epochs', type=int, help='训练轮数')
    p_train.add_argument('--batch-size', type=int, help='批量大小')
    p_train.add_argument('--lr', type=float, help='学习率')
    p_train.add_argument('--no-pretrained', action='store_true', help='不使用预训练权重')
    p_train.add_argument('--freeze-epochs', type=int, default=10, help='冻结主干训练的轮数')

    # eval
    p_eval = subparsers.add_parser('eval', help='评估模型')
    p_eval.add_argument('--model', type=str, help='模型路径')

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

    commands = {'prepare': cmd_prepare, 'train': cmd_train, 'eval': cmd_eval, 'predict': cmd_predict}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

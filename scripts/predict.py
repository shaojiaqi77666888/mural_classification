"""
壁画主题分类 - 单图预测脚本
支持：单张图片预测、批量预测、Top-K概率输出
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.config import CLASS_NAMES, IDX_TO_CLASS
from scripts.model import get_model


class MuralPredictor:
    """壁画主题分类预测器"""
    
    def __init__(
        self,
        model_path: Path,
        num_classes: int = 10,
        img_size: int = 224,
        device: str = "cuda"
    ):
        """
        Args:
            model_path: 模型权重路径
            num_classes: 类别数
            img_size: 输入图像尺寸
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 自动推断类别数
        state_dict = checkpoint['model_state_dict']
        if 'classifier.6.weight' in state_dict:
            self.num_classes = state_dict['classifier.6.weight'].shape[0]
        else:
            self.num_classes = num_classes
        
        self.model = get_model(
            num_classes=self.num_classes,
            pretrained=False,
            device=str(self.device)
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"[预测器] 模型已加载: {model_path}")
        print(f"[预测器] 类别数: {self.num_classes}")
    
    def predict(self, image_path: Path, top_k: int = 3) -> dict:
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            top_k: 返回前K个预测结果
        
        Returns:
            {
                'top1_class': str,
                'top1_confidence': float,
                'top_k': [(class_name, probability), ...]
            }
        """
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Top-K
        probs, indices = probabilities.topk(min(top_k, self.num_classes))
        probs = probs.cpu().numpy()
        indices = indices.cpu().numpy()
        
        top_k_results = []
        for idx, prob in zip(indices, probs):
            class_name = IDX_TO_CLASS.get(int(idx), f"class_{idx}")
            top_k_results.append((class_name, float(prob)))
        
        return {
            'top1_class': top_k_results[0][0],
            'top1_confidence': top_k_results[0][1],
            'top_k': top_k_results
        }
    
    def predict_batch(self, image_paths: list, top_k: int = 3) -> list:
        """批量预测"""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, top_k)
                result['image'] = str(path)
                results.append(result)
            except Exception as e:
                print(f"[错误] 预测失败 {path}: {e}")
                results.append({
                    'image': str(path),
                    'error': str(e)
                })
        return results


def main():
    parser = argparse.ArgumentParser(description='壁画主题分类预测')
    parser.add_argument('image', type=str, help='图片路径或目录')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='模型权重路径')
    parser.add_argument('--top_k', type=int, default=3,
                       help='显示前K个预测结果')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return
    
    predictor = MuralPredictor(
        model_path=model_path,
        device=args.device
    )
    
    image_path = Path(args.image)
    
    if image_path.is_file():
        # 单图预测
        result = predictor.predict(image_path, top_k=args.top_k)
        print(f"\n图片: {image_path}")
        print(f"预测结果:")
        for i, (cls, prob) in enumerate(result['top_k'], 1):
            bar = '█' * int(prob * 30)
            print(f"  {i}. {cls:12s} {prob:.4f} {bar}")
    
    elif image_path.is_dir():
        # 批量预测
        image_paths = list(image_path.glob('*.jpg')) + \
                     list(image_path.glob('*.png')) + \
                     list(image_path.glob('*.jpeg'))
        
        print(f"\n找到 {len(image_paths)} 张图片")
        results = predictor.predict_batch(image_paths, top_k=args.top_k)
        
        for r in results:
            if 'error' in r:
                print(f"\n{r['image']}: 预测失败")
            else:
                print(f"\n{r['image']}: {r['top1_class']} ({r['top1_confidence']:.4f})")
    
    else:
        print(f"[错误] 路径不存在: {image_path}")


if __name__ == "__main__":
    main()

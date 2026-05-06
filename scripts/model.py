"""
壁画主题分类 - ResNet50 模型定义
支持：预训练权重加载、特征提取、分类头自定义
"""
import torch
import torch.nn as nn
from torchvision import models

from configs.config import NUM_CLASSES  # 10个类别


class ResNet50Classifier(nn.Module):
    """ResNet50 壁画主题分类器"""
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: 分类类别数
            pretrained: 是否使用ImageNet预训练权重
            dropout: Dropout概率
            freeze_backbone: 是否冻结主干网络（用于特征提取微调）
        """
        super(ResNet50Classifier, self).__init__()
        
        # 加载预训练ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # 获取特征维度
        in_features = self.backbone.fc.in_features  # 2048
        
        # 移除原始全连接层
        self.backbone.fc = nn.Identity()
        
        # 冻结主干（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes)
        )
        
        # 初始化分类头权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He初始化分类头"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（用于可视化/分析）"""
        return self.backbone(x)
    
    def unfreeze_backbone(self, lr_ratio: float = 0.1):
        """
        解冻主干网络，用于微调
        
        Args:
            lr_ratio: 主干学习率相对于分类头的比例
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # 返回参数组配置（不同学习率）
        param_groups = [
            {'params': self.backbone.parameters(), 'lr': None},  # 稍后设置
            {'params': self.classifier.parameters(), 'lr': None}
        ]
        return param_groups


def get_model(
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.5,
    device: str = "cuda"
) -> ResNet50Classifier:
    """获取模型实例"""
    model = ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"[模型] 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[模型] 使用CPU训练")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = get_model(num_classes=10, pretrained=False, device="cpu")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
    assert out.shape == (2, 10), f"输出维度错误: {out.shape}"
    print("模型测试通过!")

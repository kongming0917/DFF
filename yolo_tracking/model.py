#!/usr/bin/env python3
"""
YOLO Tracking 모델
- yolo_sim 모델 재사용
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/hai/home/jdj/dvs')

# yolo_sim 모델 재사용
try:
    from yolo_sim.model import YOLOv3Tiny, decode_predictions, get_laser_center
    print("✅ Imported YOLO model from yolo_sim")
except ImportError:
    print("⚠️ Could not import from yolo_sim, defining locally")
    
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )
        
        def forward(self, x):
            return self.conv(x)
    
    class YOLOv3Tiny(nn.Module):
        """YOLOv3-Tiny for tracking"""
        def __init__(self, input_channels=5, num_classes=1, num_anchors=3):
            super().__init__()
            self.num_classes = num_classes
            self.num_anchors = num_anchors
            
            self.layer1 = nn.Sequential(ConvBlock(input_channels, 16, 3, 1, 1), nn.MaxPool2d(2, 2))
            self.layer2 = nn.Sequential(ConvBlock(16, 32, 3, 1, 1), nn.MaxPool2d(2, 2))
            self.layer3 = nn.Sequential(ConvBlock(32, 64, 3, 1, 1), nn.MaxPool2d(2, 2))
            self.layer4 = nn.Sequential(ConvBlock(64, 128, 3, 1, 1), nn.MaxPool2d(2, 2))
            self.layer5 = nn.Sequential(ConvBlock(128, 256, 3, 1, 1), nn.MaxPool2d(2, 2))
            self.layer6 = nn.Sequential(ConvBlock(256, 512, 3, 1, 1), ConvBlock(512, 256, 1, 1, 0))
            
            out_channels = num_anchors * (5 + num_classes)
            self.detect = nn.Conv2d(256, out_channels, 1, 1, 0)
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.detect(x)
            return x


def get_yolo_tracking_model(input_channels=5, num_classes=1, num_anchors=3):
    """YOLO tracking 모델 생성"""
    return YOLOv3Tiny(input_channels=input_channels, num_classes=num_classes, num_anchors=num_anchors)


def count_parameters(model):
    """모델 파라미터 수"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == "__main__":
    print("🧪 Testing YOLO Tracking Model")
    model = get_yolo_tracking_model(input_channels=5)
    params = count_parameters(model)
    print(f"✅ Parameters: {params['total']:,}")
    
    test_input = torch.randn(2, 5, 512, 512)
    output = model(test_input)
    print(f"✅ Input: {test_input.shape} → Output: {output.shape}")


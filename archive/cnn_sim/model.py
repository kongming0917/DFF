#!/usr/bin/env python3
"""
DVS 레이저 중심점 탐지를 위한 경량 CNN 모델들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from torchvision.models import MobileNet_V2_Weights
from typing import Tuple, Dict, Any

class BasicCNN(nn.Module):
    """기본 CNN 모델 - 간단한 구조"""
    
    def __init__(self, input_channels: int = 1, output_dim: int = 2):
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Sigmoid로 0-1 범위 제한 (좌표 회귀)
        x = torch.sigmoid(x)
        
        return x




def get_model(model_name: str, **kwargs) -> nn.Module:
    """모델 팩토리 함수 - 다양한 모델 지원"""
    
    models = {
        'basic': BasicCNN,
        'mobilenet_v2': MobileNetV2Regressor,
        'mobilenet_v2_light': MobileNetV2LightRegressor
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델의 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 Testing DVS CNN Models")
    print("=" * 40)
    
    # 테스트 입력 (배치크기=2, 채널=1, 높이=384, 너비=384)
    test_input = torch.randn(2, 1, 384, 384)
    
    for model_name in ['basic']:
        print(f"\n📊 {model_name.upper()} Model:")
        
        # 모델 생성
        model = get_model(model_name)
        model.eval()
        
        # 파라미터 수 계산
        params = count_parameters(model)
        print(f"   Parameters: {params['total']:,}")
        
        # 추론 테스트
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            output = model(test_input)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                print(f"   Inference time: {inference_time:.2f}ms")
            
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"   Sample output: {output[0].tolist()}")


class MobileNetV2Regressor(nn.Module):
    """MobileNetV2 기반 회귀 모델 - 특징 추출기 + 커스텀 분류기"""
    
    def __init__(self, input_channels: int = 1, output_dim: int = 2, pretrained: bool = True):
        super(MobileNetV2Regressor, self).__init__()
        
        # MobileNetV2 특징 추출기 로드
        #self.backbone = models.mobilenet_v2(pretrained=pretrained)
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # 입력 채널 수 조정 (RGB 3채널 → temporal 채널 수)
        if input_channels != 3:
            # MobileNetV2의 첫 번째 레이어는 Conv2dNormActivation 구조
            # 원본 첫 번째 레이어의 출력 채널 수 확인
            original_conv = self.backbone.features[0][0]  # Conv2d 부분
            original_out_channels = original_conv.out_channels
            
            # 새로운 Conv2d 레이어 생성
            new_conv = nn.Conv2d(
                input_channels, original_out_channels, 
                kernel_size=3, stride=2, padding=1, bias=False
            )
            
            # 가중치 초기화
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # 첫 번째 레이어 교체
            self.backbone.features[0][0] = new_conv
        
        # 원본 분류기 제거하고 특징 추출기만 사용
        self.feature_extractor = self.backbone.features
        
        # 특징 차원 계산 (512x512 입력 기준)
        # MobileNetV2의 마지막 특징 맵 크기: 512x512 → 16x16 (32배 축소)
        self.feature_dim = 1280  # MobileNetV2의 마지막 채널 수
        
        # 커스텀 회귀 헤드
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """회귀 헤드 가중치 초기화"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 특징 추출
        features = self.feature_extractor(x)
        
        # 회귀 예측
        output = self.regressor(features)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MobileNetV2Regressor',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': 'MobileNetV2',
            'feature_dim': self.feature_dim,
            'input_channels': self.regressor[3].in_features if hasattr(self.regressor[3], 'in_features') else 'Unknown'
        }


class MobileNetV2LightRegressor(nn.Module):
    """경량화된 MobileNetV2 기반 회귀 모델"""
    
    def __init__(self, input_channels: int = 1, output_dim: int = 2, pretrained: bool = True):
        super(MobileNetV2LightRegressor, self).__init__()
        
        # MobileNetV2 특징 추출기 로드
        #self.backbone = models.mobilenet_v2(pretrained=pretrained)
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # 입력 채널 수 조정 (RGB 3채널 → temporal 채널 수)
        if input_channels != 3:
            # MobileNetV2의 첫 번째 레이어는 Conv2dNormActivation 구조
            # 원본 첫 번째 레이어의 출력 채널 수 확인
            original_conv = self.backbone.features[0][0]  # Conv2d 부분
            original_out_channels = original_conv.out_channels
            
            # 새로운 Conv2d 레이어 생성
            new_conv = nn.Conv2d(
                input_channels, original_out_channels, 
                kernel_size=3, stride=2, padding=1, bias=False
            )
            
            # 가중치 초기화
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # 첫 번째 레이어 교체
            self.backbone.features[0][0] = new_conv
        
        # 특징 추출기만 사용
        self.feature_extractor = self.backbone.features
        
        # 더 간단한 회귀 헤드
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1280, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """회귀 헤드 가중치 초기화"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MobileNetV2LightRegressor',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': 'MobileNetV2',
            'feature_dim': 1280,
            'input_channels': 'Custom'
        }
    
    print(f"\n✅ All models tested successfully!")
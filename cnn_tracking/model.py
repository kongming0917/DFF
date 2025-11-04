#!/usr/bin/env python3
"""
DVS Laser Tracking 모델
- 기존 CNN 모델 재사용 + Tracking 특화 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import sys
sys.path.append('/hai/home/jdj/dvs')

# 기존 cnn_sim 모델 import
try:
    from cnn_sim.model import BasicCNN, MobileNetV2Regressor, MobileNetV2LightRegressor
except ImportError:
    print("⚠️ Could not import from cnn_sim, defining models locally")
    
    class BasicCNN(nn.Module):
        """기본 CNN 모델"""
        def __init__(self, input_channels: int = 1, output_dim: int = 2):
            super(BasicCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, output_dim)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            x = torch.sigmoid(x)
            return x


class LSTMTrackingCNN(nn.Module):
    """
    CNN + LSTM 기반 Tracking 모델
    - CNN으로 각 프레임의 특징 추출
    - LSTM으로 시간적 패턴 학습
    """
    
    def __init__(
        self, 
        input_channels: int = 5,
        output_dim: int = 2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2
    ):
        super().__init__()
        
        # CNN 특징 추출기 (각 프레임에 적용)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.feature_dim = 128 * 4 * 4  # 2048
        
        # LSTM (시간적 패턴 학습)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.3 if lstm_num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, temporal_frames, H, W]
        Returns:
            output: [batch, 2] (x, y) coordinates
        """
        batch_size, temporal_frames, H, W = x.shape
        
        # 각 프레임에서 특징 추출
        # [batch*temporal_frames, 1, H, W]
        x = x.view(batch_size * temporal_frames, 1, H, W)
        features = self.feature_extractor(x)  # [batch*temporal, 128, 4, 4]
        features = features.view(batch_size * temporal_frames, -1)  # [batch*temporal, 2048]
        
        # Temporal sequence로 재구성
        features = features.view(batch_size, temporal_frames, -1)  # [batch, temporal, 2048]
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # [batch, temporal, hidden_size]
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # 최종 예측
        output = self.fc(last_output)  # [batch, 2]
        
        return output


class TransformerTrackingCNN(nn.Module):
    """
    CNN + Transformer 기반 Tracking 모델
    - CNN으로 각 프레임의 특징 추출
    - Transformer로 시간적 attention 학습
    """
    
    def __init__(
        self, 
        input_channels: int = 5,
        output_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2
    ):
        super().__init__()
        
        # CNN 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.feature_dim = 128 * 4 * 4
        
        # Feature를 d_model 차원으로 프로젝션
        self.feature_projection = nn.Linear(self.feature_dim, d_model)
        
        # Positional encoding (시간 정보)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_channels, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, temporal_frames, H, W]
        Returns:
            output: [batch, 2]
        """
        batch_size, temporal_frames, H, W = x.shape
        
        # 각 프레임에서 특징 추출
        x = x.view(batch_size * temporal_frames, 1, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size * temporal_frames, -1)
        
        # Temporal sequence로 재구성
        features = features.view(batch_size, temporal_frames, -1)
        
        # d_model 차원으로 프로젝션
        features = self.feature_projection(features)  # [batch, temporal, d_model]
        
        # Positional encoding 추가
        features = features + self.pos_encoding[:, :temporal_frames, :]
        
        # Transformer
        transformer_out = self.transformer(features)  # [batch, temporal, d_model]
        
        # 마지막 타임스텝 사용
        last_output = transformer_out[:, -1, :]
        
        # 최종 예측
        output = self.fc(last_output)
        
        return output


def get_tracking_model(model_name: str, **kwargs) -> nn.Module:
    """Tracking 모델 팩토리 함수"""
    
    models = {
        'basic_tracking': BasicCNN,
        'mobilenet_v2': MobileNetV2Regressor,
        'mobilenet_v2_light': MobileNetV2LightRegressor,
        'lstm_tracking': LSTMTrackingCNN,
        'transformer_tracking': TransformerTrackingCNN,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_class = models[model_name]
    
    # 모델별로 받을 수 있는 파라미터만 전달
    # BasicCNN, MobileNet은 lstm 파라미터를 받지 않음
    if model_class in [BasicCNN, MobileNetV2Regressor, MobileNetV2LightRegressor]:
        kwargs.pop('lstm_hidden_size', None)
        kwargs.pop('lstm_num_layers', None)
    
    return model_class(**kwargs)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


if __name__ == "__main__":
    print("🧪 Testing Tracking Models")
    print("=" * 50)
    
    # 테스트 입력 (batch=2, temporal=5, H=384, W=384)
    test_input = torch.randn(2, 5, 384, 384)
    
    for model_name in ['basic_tracking', 'mobilenet_v2_light', 'mobilenet_v2', 'lstm_tracking', 'transformer_tracking']:
        print(f"\n📊 {model_name.upper()}:")
        
        try:
            # 모델 생성
            model = get_tracking_model(model_name, input_channels=5, output_dim=2)
            model.eval()
            
            # 파라미터 수
            params = count_parameters(model)
            print(f"   Parameters: {params['total']:,}")
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   Input: {test_input.shape}")
            print(f"   Output: {output.shape}")
            print(f"   Sample output: ({output[0, 0]:.3f}, {output[0, 1]:.3f})")
            print(f"   ✅ Test passed!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ All model tests completed!")


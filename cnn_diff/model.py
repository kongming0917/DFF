#!/usr/bin/env python3
"""
model.py: DVS 레이저 중심점 탐지를 위한 Differentiable Logic Network 모델
CIFAR-10용 LogicTreeNet 구조를 베이스로 하되, 회귀 헤드를 사용합니다.
"""

import torch
import torch.nn as nn
import sys
import os

# birel 패키지 경로 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
birel_path = os.path.join(dvs_root, 'birel')
if birel_path not in sys.path:
    sys.path.insert(0, birel_path)

# difflogic 및 birel 모듈 import
try:
    from difflogic import FusedLogicTreeBlock, LogicLayer
    from birel.model import RegressionLayer, MultiOutputRegressionLayer
except ImportError as e:
    print(f"Warning: Could not import difflogic or birel modules: {e}")
    print("Please ensure birel package is installed: pip install -e ./birel")
    raise


class LogicDVSNet(nn.Module):
    """
    DVS 레이저 중심점 탐지를 위한 Differentiable Logic Network
    
    CIFAR-10용 4-stage FusedLogicTreeBlock 구조를 사용하고,
    마지막에 회귀 헤드를 적용하여 (x, y) 좌표를 예측합니다.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_neurons: int = 64,
        output_dim: int = 2,
        tree_depth: int = 3,
        groups: int = 1,
        tau: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Args:
            input_channels: 입력 채널 수 (DVS 이벤트 프레임)
            num_neurons: 기본 뉴런 수 (k)
            output_dim: 출력 차원 (좌표 수, 기본값 2 = x, y)
            tree_depth: Logic tree 깊이
            groups: 그룹 수
            tau: 초기 temperature 파라미터
            device: 연산 디바이스
        """
        super().__init__()
        
        self.k = num_neurons
        self.output_dim = output_dim
        self.tau = tau
        self.device = device
        
        # CIFAR-10 스타일의 4-Stage Logic Backbone
        # Stage 1: Input -> k
        # Stage 2: k -> 4k
        # Stage 3: 4k -> 16k
        # Stage 4: 16k -> 32k
        self.features = nn.Sequential(
            # Stage 1: Input -> k (첫 레이어는 groups=1)
            FusedLogicTreeBlock(
                input_channels, self.k,
                kernel_size=3, padding=1,
                tree_depth=tree_depth, groups=1,
                tau=tau, device=device
            ),
            
            # Stage 2: k -> 4k
            FusedLogicTreeBlock(
                self.k, 4*self.k,
                kernel_size=3, padding=1,
                tree_depth=tree_depth, groups=groups,
                tau=tau, device=device
            ),
            
            # Stage 3: 4k -> 16k
            FusedLogicTreeBlock(
                4*self.k, 16*self.k,
                kernel_size=3, padding=1,
                tree_depth=tree_depth, groups=groups,
                tau=tau, device=device
            ),
            
            # Stage 4: 16k -> 32k
            FusedLogicTreeBlock(
                16*self.k, 32*self.k,
                kernel_size=3, padding=1,
                tree_depth=tree_depth, groups=groups,
                tau=tau, device=device
            )
        )
        
        # Global Average Pooling으로 공간 차원 축소
        # FusedLogicTreeBlock은 내부적으로 pooling을 포함할 수 있으므로
        # 입력 크기에 따라 최종 feature map 크기가 결정됩니다.
        # 편의상 AdaptiveAvgPool을 사용하여 차원을 고정합니다.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Regression Head
        # 입력 차원: 32*k (pooling 후)
        # 중간 레이어를 통해 feature reduction 후 회귀 출력 생성
        feature_dim = 32 * self.k
        
        # 중간 LogicLayer들 (feature reduction)
        self.reg_head = nn.Sequential(
            LogicLayer(feature_dim, 512, tau=tau, device=device),
            LogicLayer(512, 128, tau=tau, device=device),
        )
        
        # 최종 회귀 레이어: 128 -> output_dim (x, y)
        # MultiOutputRegressionLayer 사용 시 입력 차원을 output_dim의 배수로 맞춰야 함
        # 간단하게 Linear 레이어를 사용하는 것이 더 안정적
        self.reg_output = nn.Linear(128, output_dim)
        
        # 또는 MultiOutputRegressionLayer 사용 (더 복잡하지만 logic gate 기반)
        # 입력 차원을 output_dim의 배수로 조정 필요
        # self.reg_output_dim = (128 // output_dim) * output_dim  # 128 -> 128 (output_dim=2인 경우)
        # self.reg_output = MultiOutputRegressionLayer(
        #     k=output_dim,
        #     tau=tau,
        #     device=device,
        #     use_ternary=True
        # )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (B, C, H, W)
            
        Returns:
            출력 텐서 (B, output_dim) - (x, y) 좌표
        """
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.pool(x)
        x = self.flatten(x)
        
        # Regression head
        x = self.reg_head(x)
        
        # 최종 회귀 출력
        # Linear 레이어 사용 (간단하고 안정적)
        output = self.reg_output(x)
        
        # 출력이 (B, output_dim) 형태임을 보장
        assert output.shape == (x.shape[0], self.output_dim), \
            f"Output shape mismatch: {output.shape} != ({x.shape[0]}, {self.output_dim})"
        
        return output
    
    def set_tau(self, tau: float):
        """
        학습 중 tau 스케줄링을 위한 함수
        
        Args:
            tau: 새로운 temperature 값
        """
        self.tau = tau
        for m in self.modules():
            if hasattr(m, 'tau'):
                m.tau = tau
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'LogicDVSNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_neurons': self.k,
            'output_dim': self.output_dim,
            'tau': self.tau
        }


def get_model(
    input_channels: int = 1,
    num_neurons: int = 64,
    output_dim: int = 2,
    **kwargs
) -> LogicDVSNet:
    """
    모델 팩토리 함수
    
    Args:
        input_channels: 입력 채널 수
        num_neurons: 기본 뉴런 수
        output_dim: 출력 차원
        **kwargs: 추가 모델 파라미터
        
    Returns:
        LogicDVSNet 모델 인스턴스
    """
    return LogicDVSNet(
        input_channels=input_channels,
        num_neurons=num_neurons,
        output_dim=output_dim,
        **kwargs
    )


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 Testing LogicDVSNet Model")
    print("=" * 50)
    
    # 테스트 입력 (배치=2, 채널=1, 높이=128, 너비=128)
    test_input = torch.randn(2, 1, 128, 128)
    print(f"Input shape: {test_input.shape}")
    
    # 모델 생성
    model = LogicDVSNet(
        input_channels=1,
        num_neurons=32,  # 작은 모델로 테스트
        output_dim=2,
        tau=1.0
    )
    
    # 모델 정보 출력
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Forward pass 테스트
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"\n✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"   Sample output: {output[0].tolist()}")
    
    # Tau 스케줄링 테스트
    print(f"\n🔄 Testing tau scheduling...")
    model.set_tau(0.5)
    print(f"   Tau updated to: {model.tau}")


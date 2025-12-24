#!/usr/bin/env python3
"""
model.py: DVS 레이저 중심점 탐지를 위한 Differentiable Logic Network 모델
CIFAR-10용 LogicTreeNet 구조를 베이스로 하되, 회귀 헤드를 사용합니다.
"""

import torch
import torch.nn as nn
import sys
import os

# difflogic 및 birel 모듈 import
try:
    from birel.conv import Crossbar1x1Conv, TreeConvLayer, ORPool2d
    from birel.model import RegressionLayer, LogicLayer
except ImportError as e:
    print(f"Warning: Could not import difflogic or birel modules: {e}")
    print("Please ensure birel package is installed: pip install -e ./birel")
    raise


class LogicDVSNet(nn.Module):
    def __init__(self, input_channels=1, num_neurons=64, output_dim=2, tau=1.0, **kwargs):
        """
        CIFAR-10 im2col 아키텍처 기반의 DVS 레이저 중심점 탐지 모델
        
        구조 특징:
        - Crossbar1x1Conv (Channel Mixing) -> TreeConvLayer (Spatial Logic Conv) -> ORPool2d (Downsampling)
        - 4 Stages 구성
        - Regression Head (좌표 예측)
        """
        super().__init__()

        self.output_dim = output_dim
        self.tau = tau
        self.k = num_neurons
        k = num_neurons
        
        # LogicLayer 공통 설정 (im2col 모드에서는 implementation='cuda' 사용)
        base_logic_layer_kw = dict(
            ste=False,
            implementation='cuda', 
            init='residual',
            tau=tau
        )
        
        # Crossbar의 num_blocks 설정을 위한 안전장치 (최소 1)
        # 원본 코드에서는 k//16을 사용하지만, k가 작을 경우를 대비해 max(1, ...) 처리 권장
        nb = max(1, k // 16)

        self.features = nn.Sequential(
            # --- Stage 1 ---
            # Input -> k
            # Crossbar: Channel 정보를 섞어서 TreeConv 입력에 맞게 뻥튀기 (x2)
            Crossbar1x1Conv(in_channels=input_channels, out_channels=k*2, num_blocks=1, connections='unique'),
            TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, 
                          k=k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(kernel_size=2, stride=2), # 128 -> 64
            
            # --- Stage 2 ---
            # k -> 4k
            Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks=nb, connections='unique'),
            TreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1, 
                          k=4*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(kernel_size=2, stride=2), # 64 -> 32
            
            # --- Stage 3 ---
            # 4k -> 16k
            Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks=nb, connections='unique'),
            TreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1, 
                          k=16*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(kernel_size=2, stride=2), # 32 -> 16
            
            # --- Stage 4 ---
            # 16k -> 32k
            Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks=nb, connections='unique'),
            TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, 
                          k=32*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(kernel_size=2, stride=2)  # 16 -> 8
        )
        
        # --- Regression Head ---
        # 최종 Feature Map 크기: (Batch, 32*k, H', W')
        # DVS 입력(128x128) 기준 -> Stage 4 통과 후 (8x8) 예상
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Classifier 대신 좌표 예측용 Regressor 구성
        self.regressor = nn.Sequential(
            LogicLayer(32*k, 512, **base_logic_layer_kw),
            LogicLayer(512, 128, **base_logic_layer_kw),
            # 마지막은 실수 좌표를 출력해야 하므로 RegressionLayer 사용
            RegressionLayer(output_dim) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x.view(-1, self.output_dim)

    def set_tau(self, tau):
        """학습 중 모든 Logic Layer의 tau 값을 업데이트"""
        for m in self.modules():
            # TreeConvLayer, LogicLayer 등 tau 속성이 있는 모든 모듈 업데이트
            if hasattr(m, 'tau'):
                m.tau = tau
            # Crossbar1x1Conv나 TreeConvLayer 내부의 로직 레이어들도 재귀적으로 처리됨
            # 하지만 명시적으로 내부 kwargs 등을 업데이트해야 할 수도 있음
            if hasattr(m, 'logic_layer_kwargs'):
                m.logic_layer_kwargs['tau'] = tau
    
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

def get_model(
    input_channels: int = 1,
    num_neurons: int = 64,
    output_dim: int = 2,
    tau: float = 1.0,  # [수정] 명시적 인자 추가
    **kwargs
) -> LogicDVSNet:
    return LogicDVSNet(
        input_channels=input_channels,
        num_neurons=num_neurons,
        output_dim=output_dim,
        tau=tau,
        **kwargs
    )


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 Testing LogicDVSNet Model")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 테스트 입력 (Batch, Input Channels, Height, Width))
    test_input = torch.randn(2, 5, 512, 512).to(device)
    print(f"Input shape: {test_input.shape}")
    
    # 모델 생성
    model = get_model(
        input_channels=5,
        num_neurons=32, 
        output_dim=2,
        tau=1.0
    ).to(device)
    
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
        print(f"   Sample output: {output[0].tolist()}")
    
    # Tau 스케줄링 테스트
    print(f"\n🔄 Testing tau scheduling...")
    model.set_tau(0.5)
    print(f"   Tau updated to: {model.tau}")


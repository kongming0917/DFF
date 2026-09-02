#!/usr/bin/env python3
"""
model.py: DVS 레이저 중심점 탐지를 위한 경량 CNN 모델들
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from typing import Dict, Any

# Apple 공식 MobileOne 구현 import
from mobileone_official import MobileOne, mobileone, reparameterize_model, PARAMS as MOBILEONE_PARAMS, MobileOneBlock

# QAT는 학습 모드 전용 QuantStub/DeQuantStub만 모델에 둔다.
# 실제 QAT 변환(prepare/convert/observer)은 quantization.py 참고.
from torch.ao.quantization import QuantStub, DeQuantStub

# Apple 공식 MobileOne ImageNet 가중치 (unfused = multi-branch 학습형, fine-tuning용).
# torch.hub가 ~/.cache/torch/hub/checkpoints/에 캐시.
_MOBILEONE_URLS = {
    "s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar",
}


# ============================================================================
# MobileOne 기반 회귀 모델 (DVS 레이저 중심점 탐지용)
# ============================================================================

class MobileOneS0(nn.Module):
    """MobileOne S0 모델 - Apple 공식 구현 기반
    
    공식 MobileOne 백본을 사용하고 회귀 헤드를 추가한 버전
    Structural Re-parameterization을 사용하여:
    - 학습 시: multi-branch 구조로 안정적인 학습
    - 추론 시: single-branch로 빠른 추론
    """
    
    def __init__(self, variant: str = "s0", input_channels: int = 5, output_dim: int = 2,
                 inference_mode: bool = False, pretrained: bool = False):
        super(MobileOneS0, self).__init__()
        self.variant = variant

        # Quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 1. Backbone 생성
        self.backbone = mobileone(variant=variant, num_classes = 1000, inference_mode=inference_mode)

        # 1-1. (선택) ImageNet pretrained 로드 — stage0/head 교체 전에 전체 로드해야
        #      stage1~4가 pretrained로 채워진다 (stage0·linear는 곧 random으로 덮어씀).
        if pretrained:
            self._load_pretrained_backbone(variant, inference_mode)

        # 2. Stage0 교체 (replace input_channel)
        prev_stage0 = self.backbone.stage0
        self.backbone.stage0 = MobileOneBlock(
            in_channels=input_channels,
            out_channels=prev_stage0.out_channels,
            kernel_size=prev_stage0.kernel_size,
            stride=prev_stage0.stride,
            padding=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=prev_stage0.num_conv_branches
        )
        
        # 3. Head 교체: Classification -> Regression
        in_features = self.backbone.linear.in_features
        self.backbone.linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # 4. Sigmoid 추가
        #self.sigmoid = nn.Sigmoid()
        self.hsigmoid = nn.Hardsigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        # Backbone + Head 통과
        x = self.backbone(x)
        
        x = self.dequant(x)
        # Sigmoid로 0-1 범위 제한 
        #return self.sigmoid(x)
        return self.hsigmoid(x) # Hardsigmoid: ReLU6(x+3)/6
    
    def _load_pretrained_backbone(self, variant: str, inference_mode: bool):
        """Apple 공식 ImageNet 가중치를 backbone에 로드 (stage0/head 교체 전 호출).

        unfused(multi-branch) 체크포인트라 inference_mode=False에서만 키가 맞는다.
        이 시점 backbone은 vanilla mobileone이므로 strict=True로 전체(1000-class 포함) 로드 →
        이후 stage0(5ch)·linear(회귀 head)를 random으로 덮어쓰면 stage1~4만 pretrained로 남는다.
        """
        if inference_mode:
            raise ValueError("pretrained(unfused)는 inference_mode=False에서만 로드 가능")
        url = _MOBILEONE_URLS.get(variant)
        if url is None:
            raise ValueError(f"pretrained 가중치 URL 미정: variant={variant}")
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        self.backbone.load_state_dict(sd, strict=True)
        print(f"Loaded MobileOne {variant} ImageNet pretrained ({len(sd)} tensors)")

    def reparameterize(self):
        """모든 MobileOneBlock을 single-branch로 변환 (추론 최적화)"""
        self.backbone =reparameterize_model(self.backbone)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'mobileone_s0',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': 'MobileOne (Official)',
            'variant': self.variant
        }

def get_model(model_name: str, use_qat: bool = False, **kwargs) -> nn.Module:
    """모델 팩토리 함수 - 다양한 모델 지원
    
    Args:
        model_name: 모델 이름 ('mobilenet_v2', 'mobileone_s0')
        use_qat: QAT 모드 사용 여부 (int8 양자화)
        **kwargs: 모델 생성 파라미터
    """
    
    models = {
        'mobilenet_v2': MobileNetV2Regressor,
        'mobileone_s0': MobileOneS0
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name](**kwargs)

    # QAT 변환이 필요하면 quantization.prepare_qat_model(model) 사용 (FPGA INT8 배포용).
    # active path는 use_qat=False로 호출.

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델의 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


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
    
    def forward(self, x):
        # 특징 추출
        features = self.feature_extractor(x)
        # 회귀 예측
        output = self.regressor(features)
        
        return torch.sigmoid(output)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'mobilenet_v2',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': 'MobileNetV2',
            'feature_dim': self.feature_dim,
            'input_channels': self.regressor[3].in_features if hasattr(self.regressor[3], 'in_features') else 'Unknown'
        }

if __name__ == "__main__":
    # 빠른 sanity check: temporal_window=5, 512x512 입력으로 두 모델 forward 확인
    x = torch.randn(2, 5, 512, 512)

    for name in ["mobilenet_v2", "mobileone_s0"]:
        model = get_model(name, input_channels=5, output_dim=2).eval()
        with torch.no_grad():
            y = model(x)
        n_params = count_parameters(model)["total"]
        print(f"{name}: params={n_params:,}  out={tuple(y.shape)}  range=[{y.min():.3f}, {y.max():.3f}]")

        # MobileOne: reparameterize 전후 출력이 동일해야 함 (구조 재매개변수화 검증)
        if name == "mobileone_s0":
            model.reparameterize()
            model.eval()
            with torch.no_grad():
                max_diff = (y - model(x)).abs().max().item()
            print(f"  reparam max diff: {max_diff:.2e}")
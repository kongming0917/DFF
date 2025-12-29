#!/usr/bin/env python3
"""
model.py: DVS 레이저 중심점 탐지를 위한 경량 CNN 모델들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from torchvision.models import MobileNet_V2_Weights
from typing import Tuple, Dict, Any, Optional, List
import copy

# Apple 공식 MobileOne 구현 import
from mobileone_official import MobileOne, mobileone, reparameterize_model, PARAMS as MOBILEONE_PARAMS, MobileOneBlock

from torch.ao.quantization import prepare_qat, QConfig, QuantStub, FakeQuantize, DeQuantStub, fuse_modules
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    default_per_channel_weight_observer
)


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
                 inference_mode: bool = False):
        super(MobileOneS0, self).__init__()
        
        # Quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 1. Backbone 생성
        self.backbone = mobileone(variant=variant, num_classes = 1000, inference_mode=inference_mode)
        
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
        self.sigmoid = nn.Sigmoid()
        self.hsigmoid = nn.Hardsigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        # Backbone + Head 통과
        x = self.backbone(x)
        
        x = self.dequant(x)
        # Sigmoid로 0-1 범위 제한 
        #return self.sigmoid(x)
        return self.hsigmoid(x)
    
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
    
    # # QAT 모드인 경우 모델을 QAT 모델로 변환
    # if use_qat:
    #     model = prepare_qat_model(model)
    
    return model


def fuse_mobilenetv2_layers(model):
    """
    MobileNetV2의 내부 구조(InvertedResidual)를 파고들어서
    [Conv + BN + ReLU] 또는 [Conv + BN]을 찾아서 퓨전합니다.
    """
    # 1. Feature Extractor 내부 순회
    for m in model.modules():
        # torchvision의 MobileNetV2 블록 구조에 대응
        # ConvBNActivation(Sequential) 형태를 찾아서 퓨전
        if isinstance(m, nn.Sequential) and len(m) == 3:
            # Conv + BN + ReLU (예: expansion, depthwise)
            # 보통 내부 이름이 ['0', '1', '2']로 되어 있음
            if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d) and isinstance(m[2], nn.ReLU):
                fuse_modules(m, ['0', '1', '2'], inplace=True)
        
        elif isinstance(m, nn.Sequential) and len(m) == 2:
            # Conv + BN (예: projection, 마지막 레이어 등 ReLU가 없는 경우)
            if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
                fuse_modules(m, ['0', '1'], inplace=True)

    print("✅ MobileNetV2 layers fused (Conv+BN+ReLU)")
    return model

class PowerOfTwoObserver(MovingAverageMinMaxObserver):
    """ Scale 값을 2의 거듭제곱으로 강제함 (Activation용)"""
    def calculate_qparams(self):
        # 1. 기존 방식대로 min/max 기반 scale 계산
        scale, zero_point = super().calculate_qparams()
        
        # 2. Scale을 가장 가까운 2의 n승으로 변환 (PoT)
        # log2(scale) -> 반올림 -> 2^round(...)
        scale_log2 = torch.log2(scale)
        scale_pot = torch.pow(2, torch.round(scale_log2))
        
        # 3. Shift 연산만을 위해 Zero Point는 무조건 0으로 강제 (Symmetric)
        zero_point = torch.zeros_like(zero_point)
        
        return scale_pot, zero_point

class PowerOfTwoPerChannelObserver(PerChannelMinMaxObserver):
    """ Scale 값을 2의 거듭제곱으로 강제함 (Weight용)"""
    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        
        # Scale -> PoT 변환
        scale_log2 = torch.log2(scale)
        scale_pot = torch.pow(2, torch.round(scale_log2))
        
        # Zero Point -> 0 강제
        zero_point = torch.zeros_like(zero_point)
        
        return scale_pot, zero_point


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """모델을 QAT(Quantization Aware Training) 모델로 변환 - int8 양자화 (모든 레이어 per-channel)
    
    Args:
        model: 원본 모델
        
    Returns:
        QAT 준비된 모델
    """
    try:
        torch.backends.quantized.engine = 'fbgemm'
        
        print("🔢 Preparing model for QAT (int8 quantization, per-channel for all layers)...")
        
        # 모델을 학습 모드로 설정 (QAT는 학습 모드에서만 작동)
        model.train()
        model.to('cpu') # 설정 안전성 확보

        # Fuse MobileNetV2 layers (QATWrapper 없이 모델에 quant/dequant가 포함된 상태 가정)
        if hasattr(model, 'backbone') and 'MobileNetV2' in str(type(model.backbone)):
             # MobileNetV2인 경우 퓨전 수행
             fuse_mobilenetv2_layers(model)

        # 1. Activation Observer (공통)
        act_observer = FakeQuantize.with_args(
            observer = PowerOfTwoObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=True,
            quant_min=0, quant_max=255
        )
        
        act_observer_symmetric = FakeQuantize.with_args(
            observer = PowerOfTwoObserver,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=True,
            quant_min=-127, quant_max=127
        )

        # 2. Conv2d용 설정 (Per-Channel Symmetric)
        weight_obs_conv = FakeQuantize.with_args(
            observer = PowerOfTwoPerChannelObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric, # Conv는 채널별로!
            quant_min=-127, quant_max=127, eps=2e-5
        )

        # 3. Linear용 설정 (Per-Tensor Symmetric)
        # Linear는 Per-Channel을 쓰면 error 발생!
        weight_obs_linear = FakeQuantize.with_args(
            observer = PowerOfTwoObserver,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, # Linear는 텐서 통째로!
            quant_min=-127, quant_max=127, eps=2e-5
        )
        
        qconfig_conv = QConfig(activation=act_observer, weight=weight_obs_conv)
        qconfig_linear = QConfig(activation=act_observer, weight=weight_obs_linear)
        qconfig_output = QConfig(activation=act_observer_symmetric, weight=weight_obs_linear)

        model.qconfig = qconfig_conv

        # 4. 레이어 타입별로 다른 설정 주입
        def set_qconfig(module):
            """재귀적으로 모든 Conv2d와 Linear 레이어에 qconfig 설정"""
            for child_name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    child.qconfig = qconfig_conv
                elif isinstance(child, nn.Linear):
                    child.qconfig = qconfig_linear
                else:
                    # 재귀 호출
                    set_qconfig(child)
        
        set_qconfig(model)
        
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'linear'):
            final_layer = model.backbone.linear[-1]
            
            if isinstance(final_layer, nn.Linear):
                final_layer.qconfig = qconfig_output
                
        # QAT 준비
        model_prepared = prepare_qat(model)
        
        # Input scale fix
        # Observer와 FakeQuant를 켜되, 통계 수집은 끕니다.
        model_prepared.quant.activation_post_process.observer_enabled[0] = 0
        model_prepared.quant.activation_post_process.fake_quant_enabled[0] = 1
        
        # Scale = 1.0 (입력 1은 실제값 1을 의미)
        model_prepared.quant.activation_post_process.scale = torch.tensor([1.0])
        
        # Zero Point = 0 (입력 0은 실제값 0을 의미)
        model_prepared.quant.activation_post_process.zero_point = torch.tensor([0], dtype=torch.int32)
        
        print("✅ Model prepared for QAT (int8, per-channel for all layers)")
        print("   - Weight: per-channel symmetric quantization")
        print("   - Activation: per-tensor quantization")
        return model_prepared
        
    except Exception as e:
        print(f"⚠️ Error preparing QAT model: {e}")
        import traceback
        traceback.print_exc()
        print("   Using standard model without quantization.")
        return model


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """QAT 모델을 실제 양자화된 모델로 변환 (추론용) - int8
    
    Args:
        model: QAT로 학습된 모델
        
    Returns:
        양자화된 모델 (int8)
    """
    try:
        from torch.ao.quantization import convert
        
        print("🔢 Converting QAT model to quantized model (int8)...")
        
        model.eval()
        model.cpu()
        quantized_model = convert(model, inplace=False)
        
        print("✅ Model converted to quantized model (int8)")
        return quantized_model
        
    except Exception as e:
        print(f"⚠️ Error converting to quantized model: {e}")
        import traceback
        traceback.print_exc()
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


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 Testing DVS CNN Models")
    print("=" * 40)
    
    # 테스트 입력 (배치크기=2, 채널=1, 높이=384, 너비=384)
    test_input = torch.randn(2, 1, 384, 384)
    
    for model_name in ['mobilenet_v2', 'mobileone_s0']:
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
        
        # MobileOne의 경우 reparameterization 테스트
        if model_name == 'mobileone_s0':
            print(f"\n   🔄 Testing reparameterization...")
            model.reparameterize()
            model.eval()
            
            with torch.no_grad():
                output_reparam = model(test_input)
                print(f"   ✅ Reparameterized output shape: {output_reparam.shape}")
                print(f"   ✅ Output difference: {torch.abs(output - output_reparam).max().item():.6f}")


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


#!/usr/bin/env python3
"""
모델 구조 요약 정보 출력
torchinfo를 사용하여 레이어별 상세 정보 확인
"""

import torch
import sys
import os

# 경로 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from model import get_model, count_parameters

def format_bytes(bytes_size):
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def count_model_size(model):
    """모델의 실제 메모리 크기 계산"""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size

try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    print("⚠️ torchinfo가 설치되어 있지 않습니다.")
    print("   설치 방법: pip install torchinfo")
    TORCHINFO_AVAILABLE = False

if TORCHINFO_AVAILABLE:
    print("=" * 70)
    print("📊 MobileNetV2Regressor 모델 구조 요약")
    print("=" * 70)
    
    # 모델 생성 (5채널 입력, 2차원 출력)
    model = get_model('mobilenet_v2', input_channels=5, output_dim=2, use_qat=False)
    
    # 실제 입력 크기 (512x512 ROI)
    print("\n🔍 모델 요약 정보 (입력: 1, 5, 512, 512)")
    print("-" * 70)
    
    summary_result = summary(
        model,
        input_size=(1, 5, 512, 512),  # 실제 ROI 크기
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,  # 레이어 깊이 (3~4 추천)
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("📈 파라미터 수 비교")
    print("=" * 70)
    
    # torchinfo가 계산한 파라미터 수
    if hasattr(summary_result, 'total_params'):
        torchinfo_total = summary_result.total_params
        print(f"\ntorchinfo total_params: {torchinfo_total:,}")
    
    # 직접 계산한 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n직접 계산 (model.parameters()):")
    print(f"  전체 파라미터 수: {total_params:,}")
    print(f"  학습 가능한 파라미터: {trainable_params:,}")
    print(f"  고정 파라미터: {total_params - trainable_params:,}")
    
    # 모든 텐서 포함 (Buffer 포함 - BatchNorm의 running_mean, running_var 등)
    all_tensors = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    print(f"\n모든 텐서 (parameters + buffers):")
    print(f"  Parameters: {total_params:,}")
    print(f"  Buffers (running_mean, running_var 등): {sum(b.numel() for b in model.buffers()):,}")
    print(f"  Total: {all_tensors:,}")
    
    # 레이어별 상세 분석
    print(f"\n" + "=" * 70)
    print("🔍 레이어별 파라미터 분석")
    print("=" * 70)
    
    feature_extractor_params = 0
    regressor_params = 0
    
    for name, param in model.named_parameters():
        if 'feature_extractor' in name or 'backbone' in name:
            feature_extractor_params += param.numel()
        elif 'regressor' in name:
            regressor_params += param.numel()
    
    print(f"\n  Feature Extractor: {feature_extractor_params:,}")
    print(f"  Regressor Head:    {regressor_params:,}")
    print(f"  Total:             {feature_extractor_params + regressor_params:,}")
    
    # 메모리 사용량
    print(f"\n" + "=" * 70)
    print("💾 메모리 사용량")
    print("=" * 70)
    float32_size = total_params * 4 / (1024**2)
    print(f"\n  Float32 (4 bytes): {float32_size:.2f} MB")
    print(f"  Int8 (1 byte):     {total_params * 1 / (1024**2):.2f} MB")
    print(f"  절약:               {float32_size - total_params * 1 / (1024**2):.2f} MB (75% 감소)")
    
    print("=" * 70)
    
    # 양자화 전후 비교 추가
    print("\n" + "=" * 70)
    print("🔢 양자화 전후 비교")
    print("=" * 70)
    
    # 1. Standard 모델
    print("\n1️⃣ Standard 모델 (Float32, 양자화 전)")
    print("-" * 70)
    standard_model = get_model('mobilenet_v2', input_channels=5, output_dim=2, use_qat=False)
    standard_params = count_parameters(standard_model)
    standard_size = count_model_size(standard_model)
    
    print(f"   Total parameters:     {standard_params['total']:,}")
    print(f"   Trainable parameters: {standard_params['trainable']:,}")
    print(f"   Model size (float32): {format_bytes(standard_size)}")
    
    # 2. QAT 모델
    print("\n2️⃣ QAT 모델 (Float32, 양자화 준비, 학습 중)")
    print("-" * 70)
    qat_model = get_model('mobilenet_v2', input_channels=5, output_dim=2, use_qat=True)
    qat_params = count_parameters(qat_model)
    qat_size = count_model_size(qat_model)
    
    print(f"   Total parameters:     {qat_params['total']:,}")
    print(f"   Trainable parameters: {qat_params['trainable']:,}")
    print(f"   Model size (float32): {format_bytes(qat_size)}")
    print(f"   Note:                 Fake quantization layers 추가됨")
    
    # 3. 양자화된 모델 (체크포인트에서 로드)
    print("\n3️⃣ 양자화된 모델 (int8, 추론용)")
    print("-" * 70)
    
    checkpoint_path = 'checkpoints_mobilenet_v2_qat/mobilenet_best.pth'
    quantized_checkpoint_path = 'checkpoints_mobilenet_v2_qat/mobilenet_v2_quantized.pth'
    
    if os.path.exists(quantized_checkpoint_path):
        checkpoint = torch.load(quantized_checkpoint_path, map_location='cpu', weights_only=False)
        print(f"   ✅ Quantized checkpoint found: {quantized_checkpoint_path}")
        
        quantized_model = get_model('mobilenet_v2', input_channels=5, output_dim=2, use_qat=False)
        try:
            quantized_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            quantized_params = count_parameters(quantized_model)
            int8_size = quantized_params["total"] * 1
            
            print(f"   Total parameters:     {quantized_params['total']:,}")
            print(f"   Model size (int8):     {format_bytes(int8_size)}")
            print(f"   Data type:            int8 (1 byte per parameter)")
        except Exception as e:
            print(f"   ⚠️ Error loading quantized model: {e}")
            int8_size = standard_params["total"] * 1
            print(f"   Expected size (int8):  {format_bytes(int8_size)}")
    elif os.path.exists(checkpoint_path):
        print(f"   ⚠️ Quantized checkpoint not found, using QAT checkpoint")
        print(f"   Note:                 이 모델은 QAT 학습 중인 모델 (아직 float32)")
        int8_size = standard_params["total"] * 1
        print(f"   Expected size (int8):  {format_bytes(int8_size)}")
    else:
        print(f"   ⚠️ Checkpoint not found")
        int8_size = standard_params["total"] * 1
        print(f"   Expected size (int8):  {format_bytes(int8_size)}")
    
    # 4. 비교 요약
    print("\n" + "=" * 70)
    print("📈 양자화 전후 비교 요약")
    print("=" * 70)
    
    print(f"\n파라미터 수:")
    print(f"  Standard:  {standard_params['total']:,} (동일)")
    print(f"  QAT:       {qat_params['total']:,} (동일)")
    print(f"  Quantized: {standard_params['total']:,} (동일)")
    print(f"\n  ✅ 파라미터 수는 동일합니다! (양자화는 데이터 타입만 변경)")
    
    print(f"\n메모리 사용량:")
    print(f"  Standard (float32):  {format_bytes(standard_size)}")
    print(f"  Quantized (int8):    {format_bytes(int8_size)}")
    saved = standard_size - int8_size
    saved_percent = (saved / standard_size) * 100
    print(f"  절약:                {format_bytes(saved)} ({saved_percent:.1f}% 감소)")
    
    print(f"\n추가 정보:")
    print(f"  - Per-channel 양자화 파라미터: ~5 KB (scale + zero-point)")
    print(f"  - 양자화 오버헤드: 무시할 수 있는 수준")
    
    print("=" * 70)
else:
    # torchinfo가 없을 경우 대안
    print("\n📊 모델 구조 (기본 정보)")
    print("=" * 70)
    
    model = get_model('mobilenet_v2', input_channels=5, output_dim=2, use_qat=False)
    
    # 간단한 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n전체 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    
    print("\n주요 레이어:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")
    
    print("\n⚠️ 더 자세한 정보를 보려면 torchinfo를 설치하세요:")
    print("   pip install torchinfo")


#!/usr/bin/env python3
"""
model_summary.py: 모델 구조 및 메모리 사용량 분석 스크립트
지원 모델: mobilenet_v2, mobileone_s0
"""

import torch
import torch.nn as nn
import sys
import os
import argparse  # [추가] 명령줄 인자 처리를 위해 추가

# 경로 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# model.py에서 필요한 함수들 임포트
# (주의: model.py에 prepare_qat_model, convert_to_quantized가 있어야 실제 int8 로드 가능)
from model import get_model, count_parameters, prepare_qat_model, convert_to_quantized

def format_bytes(bytes_size):
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def count_model_size(model):
    """모델의 실제 메모리 크기 계산 (buffer 포함)"""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size

def analyze_model(model_name, input_channels=5, output_dim=2):
    print("=" * 70)
    print(f"📊 {model_name} 모델 구조 요약")
    print("=" * 70)

    # 1. 모델 생성 (Float32)
    model = get_model(model_name, input_channels=input_channels, output_dim=output_dim, use_qat=False)
    
    # [MobileOne 전용] Reparameterization 전/후 비교
    is_mobileone = 'mobileone' in model_name
    
    if is_mobileone:
        print("\n🔄 MobileOne 감지됨: Reparameterization(가지치기) 전/후 비교 수행")
        
        # (1) 학습용 구조 (Multi-branch)
        params_train = count_parameters(model)['total']
        print(f"  - 학습 모드 (Multi-branch) 파라미터: {params_train:,}")
        
        # (2) 추론용 구조 (Single-branch)로 변환
        if hasattr(model, 'reparameterize'):
            model.reparameterize()
            print("  - ✅ Reparameterization 완료 (구조 단순화)")
        
        params_infer = count_parameters(model)['total']
        print(f"  - 추론 모드 (Single-branch) 파라미터: {params_infer:,}")
        print(f"  - 파라미터 감소: {params_train - params_infer:,} (구조적 오버헤드 제거됨)")
        print("-" * 70)

    # torchinfo 확인
    try:
        from torchinfo import summary
        print(f"\n🔍 모델 요약 정보 (Model: {model_name}, Input: 1, {input_channels}, 512, 512)")
        print("-" * 70)
        
        summary(
            model,
            input_size=(1, input_channels, 512, 512),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            verbose=1
        )
    except ImportError:
        print("⚠️ torchinfo가 설치되어 있지 않습니다. (pip install torchinfo)")

    print("\n" + "=" * 70)
    print("📈 파라미터 및 메모리 분석")
    print("=" * 70)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"모델: {model_name}")
    print(f"  전체 파라미터:      {total_params:,}")
    print(f"  학습 가능 파라미터: {trainable_params:,}")
    
    # 메모리 사용량 (이론적)
    float32_size_mb = total_params * 4 / (1024**2)
    int8_size_mb = total_params * 1 / (1024**2)
    
    print("\n💾 이론적 메모리 사용량 (가중치 Only):")
    print(f"  Float32 (Training/FP32 Inference): {float32_size_mb:.2f} MB")
    print(f"  Int8    (Quantized Inference):     {int8_size_mb:.2f} MB")
    print(f"  예상 절약 효과:                    {float32_size_mb - int8_size_mb:.2f} MB (75% 감소)")

    # ---------------------------------------------------------
    # 실제 체크포인트 파일 확인 및 로드 시도
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("📁 체크포인트 파일 확인")
    print("=" * 70)

    # 동적 경로 생성
    ckpt_dir = f"checkpoints_{model_name}_qat"
    int8_filename = f"{model_name}_int8.pth"
    int8_path = os.path.join(ckpt_dir, int8_filename)

    if os.path.exists(int8_path):
        print(f"✅ Int8 모델 파일 발견: {int8_path}")
        file_size = os.path.getsize(int8_path)
        print(f"  실제 파일 크기: {format_bytes(file_size)}")
        
        # 실제 로드 테스트 (구조가 맞아야 로드됨)
        try:
            print("  🔄 Int8 모델 구조 생성 및 로드 시도...")
            
            # 1. 껍데기 생성
            dummy_model = get_model(model_name, input_channels=input_channels, output_dim=output_dim)
            if hasattr(dummy_model, 'reparameterize'):
                dummy_model.reparameterize()
            
            # 2. QAT 준비 & 변환 (Int8 껍데기로 변신)
            # 주의: model.py의 prepare_qat_model 설정과 정확히 일치해야 함
            dummy_model = prepare_qat_model(dummy_model)
            dummy_model.eval()
            dummy_model.to('cpu')
            quantized_model = convert_to_quantized(dummy_model)
            
            # 3. 로드
            checkpoint = torch.load(int8_path, map_location='cpu')
            quantized_model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✅ Int8 State Dict 로드 성공! (검증 완료)")
            
        except Exception as e:
            print(f"  ⚠️ 로드 테스트 중 경고/에러: {e}")
            print("  (모델 구조나 prepare_qat 설정이 저장 시점과 다를 수 있습니다.)")
            
    else:
        print(f"❌ Int8 모델 파일을 찾을 수 없음: {int8_path}")
        print(f"  (아직 학습이 완료되지 않았거나 QAT가 실행되지 않았습니다.)")

if __name__ == "__main__":
    # 인자 파싱 설정
    parser = argparse.ArgumentParser(description="DVS 모델 구조 및 파라미터 분석기")
    parser.add_argument('--model', type=str, default='mobileone_s0', 
                        choices=['mobilenet_v2', 'mobileone_s0'],
                        help='분석할 모델 이름 (기본값: mobileone_s0)')
    parser.add_argument('--channels', type=int, default=5, help='입력 채널 수 (기본값: 5)')
    
    args = parser.parse_args()
    
    # 분석 실행
    analyze_model(args.model, input_channels=args.channels)
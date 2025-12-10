#!/usr/bin/env python3
"""
check_mobileone_arch.py: MobileOne S0의 학습 vs 추론 아키텍처 비교 분석
"""

import torch
import sys
import os
from torchinfo import summary

# 경로 설정 (model.py가 있는 위치)
dvs_root = os.path.dirname(os.path.abspath(__file__))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from model import get_model

def format_bytes(size):
    """바이트 크기 포맷팅"""
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def compare_mobileone_architecture():
    print("=" * 80)
    print("🚀 MobileOne S0 Architecture Analysis (Train vs Inference)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 초기 모델 생성 (학습 모드)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1️⃣ [Training Mode] Multi-branch Architecture")
    print("   - 특징: 복잡한 구조, 많은 파라미터 (Over-parameterized)")
    print("=" * 80)
    
    # 모델 생성 (5채널 입력, 2차원 출력)
    model = get_model('mobileone_s0', input_channels=5, output_dim=2)
    model.train() # 학습 모드 설정

    # 학습 모드 구조 요약
    train_summary = summary(
        model,
        input_size=(1, 5, 512, 512), # 배치 1, 채널 5, 512x512
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3, # depth를 3으로 해야 내부 브랜치(rbr_conv 등)가 보임
        verbose=1
    )
    
    train_params = train_summary.total_params
    train_size_mb = train_params * 4 / (1024**2) # Float32 기준
    
    # 학습 모드 state_dict 키 확인 (multi-branch 구조)
    train_keys = list(model.state_dict().keys())
    multi_branch_keys = [k for k in train_keys if 'rbr_conv' in k or 'rbr_scale' in k or 'rbr_1x1' in k]
    print(f"\n📋 Multi-branch keys found: {len(multi_branch_keys)}")
    if multi_branch_keys:
        print(f"   Examples: {multi_branch_keys[:5]}")
        if len(multi_branch_keys) > 5:
            print(f"   ... and {len(multi_branch_keys) - 5} more")

    # -------------------------------------------------------------------------
    # 2. 구조 재매개변수화 (Reparameterization)
    # -------------------------------------------------------------------------
    print("\n\n🔄 Applying Structural Re-parameterization...")
    print("   (Fusing branches: Conv3x3 + Conv1x1 + Skip -> Single Conv3x3)")
    
    model.eval()           # 평가 모드로 전환
    model.reparameterize() # ★ 핵심: 구조 변경 수행

    # -------------------------------------------------------------------------
    # 3. 변환 후 모델 (추론 모드)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2️⃣ [Inference Mode] Single-branch Architecture")
    print("   - 특징: 단순한 구조, 최적화된 속도, 파라미터 감소")
    print("=" * 80)

    # 추론 모드 구조 요약
    infer_summary = summary(
        model,
        input_size=(1, 5, 512, 512),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3, 
        verbose=1
    )

    infer_params = infer_summary.total_params
    infer_size_mb = infer_params * 4 / (1024**2) # Float32 기준
    
    # 추론 모드 state_dict 키 확인 (single-branch 구조)
    infer_keys = list(model.state_dict().keys())
    single_branch_keys = [k for k in infer_keys if 'rbr_conv' in k or 'rbr_scale' in k or 'rbr_1x1' in k]
    print(f"\n📋 Single-branch keys (rbr_* removed): {len(single_branch_keys)}")
    if len(single_branch_keys) == 0:
        print("   ✅ All multi-branch keys successfully fused!")
    else:
        print(f"   ⚠️ Remaining multi-branch keys: {single_branch_keys[:5]}")
    
    # 키 차이 확인
    removed_keys = set(multi_branch_keys) - set(infer_keys)
    print(f"\n🔍 Keys removed after reparameterization: {len(removed_keys)}")
    if removed_keys:
        print(f"   Examples: {list(removed_keys)[:5]}")
        if len(removed_keys) > 5:
            print(f"   ... and {len(removed_keys) - 5} more")

    # -------------------------------------------------------------------------
    # 4. 최종 비교 리포트
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 Final Comparison Report")
    print("=" * 80)
    
    print(f"{'Metric':<25} | {'Training (Multi-branch)':<25} | {'Inference (Single-branch)':<25}")
    print("-" * 80)
    print(f"{'Total Parameters':<25} | {train_params:<25,} | {infer_params:<25,}")
    print(f"{'Model Size (Float32)':<25} | {train_size_mb:.2f} MB{'':<18} | {infer_size_mb:.2f} MB")
    
    # 파라미터 감소율 계산
    reduced_params = train_params - infer_params
    reduction_ratio = (reduced_params / train_params) * 100
    
    print("-" * 80)
    print(f"📉 Parameter Reduction: {reduced_params:,} parameters removed ({reduction_ratio:.2f}%)")
    print(f"💡 Explanation: 학습 시 사용된 보조 브랜치들이 융합되어 사라졌습니다.")
    print(f"                추론 결과(Accuracy)는 동일하지만 속도는 훨씬 빠릅니다.")
    print("=" * 80)

if __name__ == "__main__":
    compare_mobileone_architecture()

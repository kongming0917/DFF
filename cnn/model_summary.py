#!/usr/bin/env python3
"""model_summary.py: CNN 모델 구조·파라미터·메모리 분석.

지원 모델: mobilenet_v2, mobileone_s0.
- 공통: torchinfo 레이어 요약 + 파라미터 수 + 이론 메모리(FP32 vs INT8)
- MobileOne: reparameterize 전/후 구조(multi-branch -> single-branch) 비교
  (param 감소량 + 융합된 rbr_* state_dict 키 수)
- INT8 체크포인트 경로를 주면 양자화 구조 로드까지 검증

  python model_summary.py --model mobileone_s0
  python model_summary.py --model mobilenet_v2 --channels 5
  python model_summary.py --model mobileone_s0 --int8 checkpoints_mobileone_s0_qat/mobileone_s0_int8.pth
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import get_model, count_parameters
from quantization import prepare_qat_model, convert_to_quantized


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def branch_keys(model) -> list:
    """multi-branch(rbr_conv/rbr_scale/rbr_1x1) state_dict 키 목록."""
    return [k for k in model.state_dict() if any(t in k for t in ("rbr_conv", "rbr_scale", "rbr_1x1"))]


def torchinfo_summary(model, channels: int) -> None:
    try:
        from torchinfo import summary
    except ImportError:
        print("  (torchinfo 미설치 — `pip install torchinfo` 하면 레이어별 요약 출력)")
        return
    summary(
        model,
        input_size=(1, channels, 512, 512),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        verbose=1,
    )


def verify_int8(model_name: str, channels: int, int8_path: str) -> None:
    print("\n[INT8 checkpoint load test]")
    if not os.path.exists(int8_path):
        print(f"  파일 없음: {int8_path}")
        return
    print(f"  file: {int8_path}  ({fmt_bytes(os.path.getsize(int8_path))})")
    try:
        # 저장 시점과 동일한 파이프라인으로 INT8 껍데기를 만든 뒤 가중치 로드
        m = get_model(model_name, input_channels=channels, output_dim=2)
        if hasattr(m, "reparameterize"):
            m.reparameterize()
        m = prepare_qat_model(m)
        m.eval()
        m.to("cpu")
        qm = convert_to_quantized(m)
        ckpt = torch.load(int8_path, map_location="cpu", weights_only=False)
        qm.load_state_dict(ckpt["model_state_dict"])
        print("  load OK — 저장 시점 구조와 일치")
    except Exception as e:
        print(f"  load 실패: {e}")
        print("  (모델 구조 / prepare_qat 설정이 저장 시점과 다를 수 있음)")


def analyze(model_name: str, channels: int = 5, int8_path: str = None) -> None:
    print("=" * 70)
    print(f"{model_name} 모델 구조 요약 (input = 1x{channels}x512x512)")
    print("=" * 70)

    model = get_model(model_name, input_channels=channels, output_dim=2, use_qat=False)

    # MobileOne: reparameterize 전/후 비교 (multi-branch -> single-branch)
    if "mobileone" in model_name:
        params_train = count_parameters(model)["total"]
        keys_before = branch_keys(model)
        print(f"\n[Training ] multi-branch  params: {params_train:,}  (rbr_* keys: {len(keys_before)})")

        model.eval()
        model.reparameterize()

        params_infer = count_parameters(model)["total"]
        keys_after = branch_keys(model)
        reduced = params_train - params_infer
        ratio = reduced / params_train * 100 if params_train else 0.0
        print(f"[Inference] single-branch params: {params_infer:,}  (rbr_* keys: {len(keys_after)})")
        print(f"  -> {reduced:,} params fused away ({ratio:.1f}%); 추론 결과는 동일, 속도만 향상")

    # torchinfo 레이어별 요약 (현재 구조 기준)
    print("\n[Layer summary]")
    torchinfo_summary(model, channels)

    # 파라미터 수 + 이론적 가중치 메모리
    total = sum(p.numel() for p in model.parameters())
    print("\n[Parameters & theoretical weight memory]")
    print(f"  total params : {total:,}")
    print(f"  FP32         : {fmt_bytes(total * 4)}")
    print(f"  INT8         : {fmt_bytes(total * 1)}  (~75% 감소)")

    # INT8 체크포인트 로드 검증 (선택)
    if int8_path:
        verify_int8(model_name, channels, int8_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN 모델 구조/파라미터/메모리 분석")
    ap.add_argument("--model", default="mobileone_s0", choices=["mobilenet_v2", "mobileone_s0"])
    ap.add_argument("--channels", type=int, default=5, help="입력 채널 수 (temporal window)")
    ap.add_argument("--int8", default=None, help="INT8 .pth 경로 (주면 양자화 구조 로드 검증)")
    args = ap.parse_args()
    analyze(args.model, channels=args.channels, int8_path=args.int8)

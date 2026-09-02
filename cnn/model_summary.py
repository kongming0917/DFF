#!/usr/bin/env python3
"""model_summary.py: CNN 모델 구조·파라미터·메모리 분석.

지원 모델: mobilenet_v2, mobileone_s0.
- 공통: torchinfo 레이어 요약 + 파라미터 수 + 이론 메모리(FP32 vs INT8)
- MobileOne: reparameterize 전/후 구조(multi-branch -> single-branch) 비교
  (param 감소량 + 융합된 rbr_* state_dict 키 수)
- INT8 체크포인트 경로를 주면 양자화 구조 로드까지 검증

  python model_summary.py --model mobileone_s0
  python model_summary.py --model mobilenet_v2 --channels 5
  python model_summary.py --model mobileone_s0 --int8 runs/qat_mobileone_s0/mobileone_s0_int8.pth
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import get_model, count_parameters
from dvslib.quant.pt2e import load_int8  # noqa: E402


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
    """PT2E INT8 체크포인트(dvslib.quant.pt2e.save_int8 산출물) 복원 검증 — 구조 재구성 후 strict 로드."""
    if not os.path.exists(int8_path):
        print(f"  파일 없음: {int8_path}")
        return
    print(f"  file: {int8_path}  ({fmt_bytes(os.path.getsize(int8_path))})")

    def build():
        m = get_model(model_name, input_channels=channels, output_dim=2)
        if hasattr(m, "reparameterize"):
            m.reparameterize()
        return m

    try:
        ck = torch.load(int8_path, map_location="cpu", weights_only=False)
        per_channel = bool(ck.get("config", {}).get("per_channel", True)) if isinstance(ck, dict) else True
        qm, cfg = load_int8(int8_path, build, (torch.randn(2, channels, 512, 512),), per_channel=per_channel)
        n_q = sum(1 for n in qm.graph.nodes if "quantize_per" in str(n.target))
        print(f"  ✅ PT2E INT8 그래프 복원 OK  (quant/dequant 노드 {n_q}개)")
        if cfg:
            keys = ("int8_pixel_error_mean", "int8_accuracy_5px", "per_channel", "qat_epochs")
            print("  config:", {k: cfg[k] for k in keys if k in cfg})
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 복원 실패: {type(e).__name__}: {e}")


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

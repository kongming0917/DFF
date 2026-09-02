#!/usr/bin/env python3
"""cnn experiment — PT2E QAT fine-tuning entrypoint.

학습된 FP32 체크포인트 → reparam → PT2E QAT fine-tune → INT8 정확도 측정 → INT8 그래프 저장.

PT2E exported 모델은 `.train()/.eval()`을 못 쓰므로, dvslib trainer/evaluator에
`set_mode=quantization.set_qat_mode` 훅을 주입해 **공유 loop를 그대로 재사용**한다 (A2).
정확도는 fine-tune된 그래프(fake-quant)로 측정 — INT8 시뮬레이션이라 변환 모델과 동등하고,
PT2E INT8 op의 CUDA 미지원 문제를 피한다. convert된 INT8 그래프는 export/배포용.

  python cnn/train_qat.py --checkpoint cnn/runs/baseline_mobileone_s0_pretrained/mobileone_s0_best.pth
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # dvslib
sys.path.insert(0, HERE)   # local model.py

import torch  # noqa: E402

from dvslib.data.dataset import load_frames_from_bin   # noqa: E402
from dvslib.data.split import make_train_val_loaders    # noqa: E402
from dvslib.eval.evaluate import evaluate_regression     # noqa: E402
from dvslib.training.loop import RegressionTrainer        # noqa: E402
from dvslib.training.seed import seed_everything          # noqa: E402
from model import get_model                               # noqa: E402
from quantization import prepare_qat, convert, set_qat_mode  # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description="cnn PT2E QAT fine-tuning")
    ap.add_argument("--checkpoint", required=True, help="학습된 FP32 체크포인트(.pth)")
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-frames", type=int, default=None, help="None이면 체크포인트 config 사용")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    # 체크포인트에서 학습 config 읽기 (model·tw·roi·max_frames)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    model_name = cfg.get("model", "mobileone_s0")
    tw = cfg.get("temporal_window", 5)
    roi_n = cfg.get("roi", 512)
    roi = (roi_n, roi_n)
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("max_frames", 3000)
    save_dir = args.save_dir or os.path.join(HERE, "runs", f"qat_{model_name}")

    frames = load_frames_from_bin(
        os.path.join(DATA, f"gaussian_brownian_{roi_n}x{roi_n}.bin"),
        max_frames=max_frames, height=roi_n, width=roi_n,
    )
    train_loader, val_loader = make_train_val_loaders(
        frames, os.path.join(DATA, f"gaussian_brownian_{roi_n}x{roi_n}_labels.csv"),
        batch_size=args.batch_size, temporal_window=tw, roi_size=roi,
    )

    def build_fp32():
        """FP32 가중치 로드 + (MobileOne) reparam → single-branch 배포 구조."""
        m = get_model(model_name, input_channels=tw, output_dim=2)
        m.load_state_dict(ck["model_state_dict"])
        if hasattr(m, "reparameterize"):
            m.reparameterize()
        return m

    # 1) FP32 baseline (양자화 손실의 기준점)
    fp32 = evaluate_regression(build_fp32(), val_loader, roi_size=roi, device=args.device)
    print(f"[FP32]        pixel_err={fp32['pixel_error_mean']:.4f}px  acc5={fp32['accuracy_5px']:.2f}%")

    # 2) PT2E QAT prepare (CPU 캡처) → 공유 trainer로 fine-tune (set_mode 훅 주입)
    example = (torch.randn(2, tw, roi_n, roi_n),)
    prepared = prepare_qat(build_fp32().cpu(), example, per_channel=args.per_channel)
    qcfg = {**cfg, "qat": True, "qat_lr": args.qat_lr, "qat_epochs": args.qat_epochs,
            "per_channel": args.per_channel}
    trainer = RegressionTrainer(
        prepared, train_loader, val_loader,
        model_name=model_name, roi_size=roi, lr=args.qat_lr, num_epochs=args.qat_epochs,
        save_dir=save_dir, device=args.device, monitor="loss",
        scheduler="cosine", warmup_epochs=0, grad_clip=1.0,
        config=qcfg, set_mode=set_qat_mode,
    )
    trainer.fit()

    # 3) convert → 진짜 INT8(배포본)로 정확도 측정.
    #    주의: fake-quant prepared 모델은 observer가 eval에서도 갱신돼 측정값이 비결정적·낙관적
    #    (검증됨: 같은 모델 두 번 eval에 값이 다름). 따라서 observer 없는 **변환 모델**(고정 scale)로
    #    측정한다 — 이게 실제 배포본. PT2E INT8 op은 CPU 실행이라 device="cpu".
    int8 = convert(prepared.cpu())
    qat = evaluate_regression(int8, val_loader, roi_size=roi, device="cpu", set_mode=set_qat_mode)
    d = qat["pixel_error_mean"] - fp32["pixel_error_mean"]
    print(f"[QAT INT8]    pixel_err={qat['pixel_error_mean']:.4f}px  acc5={qat['accuracy_5px']:.2f}%  "
          f"(Δ {d:+.4f}px vs FP32; converted, cpu)")

    # 4) INT8 그래프 저장 (export/배포용)
    int8_path = os.path.join(save_dir, f"{model_name}_int8.pth")
    torch.save(int8.state_dict(), int8_path)
    print(f"  INT8 그래프 저장 -> {os.path.relpath(int8_path, ROOT)}")


if __name__ == "__main__":
    main()

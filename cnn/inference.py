#!/usr/bin/env python3
"""cnn experiment — inference / evaluation on top of dvslib.

체크포인트를 로드해 blocked-split 검증셋에서 dvslib 지표(pixel error, Acc@Npx, FPS)를 보고한다.
데이터·split·metric은 모두 dvslib에서 온다.

  python cnn/inference.py                       # baseline best.pth 평가
  python cnn/inference.py --checkpoint <path>
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # dvslib
sys.path.insert(0, HERE)   # local model.py

import torch  # noqa: E402

from dvslib.data.dataset import load_frames_from_bin, parse_roi, brownian_paths  # noqa: E402
from dvslib.data.split import make_train_val_loaders    # noqa: E402
from dvslib.eval.evaluate import evaluate_regression    # noqa: E402
from model import get_model                             # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description="cnn inference / evaluation")
    ap.add_argument(
        "--checkpoint",
        default=os.path.join(HERE, "runs", "baseline_mobilenet_v2", "mobilenet_v2_best.pth"),
    )
    # None이면 체크포인트의 config에서 자동으로 채움 (CLI로 주면 override)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temporal-window", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--roi", default=None, help='"512" 또는 "720x960"(HxW). 기본: 체크포인트 config')
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 체크포인트를 먼저 로드해 학습 config를 읽는다 (config drift 방지)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}

    def pick(cli, key, fallback):
        return cli if cli is not None else cfg.get(key, fallback)

    model_name = pick(args.model, "model", "mobilenet_v2")
    temporal_window = pick(args.temporal_window, "temporal_window", 5)
    max_frames = pick(args.max_frames, "max_frames", 3000)
    roi = parse_roi(pick(args.roi, "roi", 512))     # (H, W); 옛 체크포인트는 int, 새 것은 "HxW" 문자열
    roi_n = f"{roi[0]}x{roi[1]}"

    bin_path, csv_path = brownian_paths(DATA, roi)

    frames = load_frames_from_bin(bin_path, max_frames=max_frames, height=roi[0], width=roi[1])
    _, val_loader = make_train_val_loaders(
        frames, csv_path,
        batch_size=args.batch_size, temporal_window=temporal_window, roi_size=roi,
    )

    model = get_model(model_name, input_channels=temporal_window, output_dim=2, use_qat=False)
    model.load_state_dict(ck["model_state_dict"])

    # MobileOne: multi-branch 가중치 로드 후 single-branch로 융합 (배포 구조 = 추론 최적화).
    # 수학적으로 정확(출력 동일), FPS만 향상 → 평가 FPS가 실제 배포 구조를 반영.
    if hasattr(model, "reparameterize"):
        model.reparameterize()

    m = evaluate_regression(model, val_loader, roi_size=roi, device=args.device)
    src = "from checkpoint config" if cfg else "defaults (config 없음 — 옛 체크포인트)"
    print(f"\n[cnn inference: {os.path.relpath(args.checkpoint, ROOT)}]")
    print(f"  config           : model={model_name} tw={temporal_window} roi={roi_n}  ({src})")
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"  checkpoint epoch : {ck['epoch']}")
    print(f"  val samples      : {m['num_samples']}")
    print(f"  pixel_error_mean : {m['pixel_error_mean']:.4f} px  (std {m['pixel_error_std']:.4f})")
    print(f"  Acc@5px / @10px  : {m['accuracy_5px']:.2f}% / {m['accuracy_10px']:.2f}%")
    print(f"  throughput       : {m['fps']:.1f} FPS  (bs={args.batch_size})")


if __name__ == "__main__":
    main()

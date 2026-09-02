#!/usr/bin/env python3
"""yolo experiment — inference / evaluation on top of dvslib.

체크포인트를 로드해 blocked-split 검증셋에서 dvslib 지표(pixel error, Acc@Npx, FPS)를 보고한다.
FPS에는 decode→center 후처리가 포함된다.

  python yolo/inference.py                       # baseline best.pth 평가
  python yolo/inference.py --checkpoint <path>   # 옛(pre-dvslib) 체크포인트도 로드 가능 (config 없음 → 기본값)
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

from dvslib.data.dataset import load_frames_from_bin, parse_roi, brownian_paths  # noqa: E402
from dvslib.data.split import make_train_val_loaders    # noqa: E402
from dvslib.eval.evaluate import evaluate_regression    # noqa: E402
from model import YOLOCenterDecoder, default_anchors, get_model  # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description="yolo inference / evaluation")
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "runs", "baseline_yolo_tiny", "yolo_tiny_best.pth"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--temporal-window", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--roi", default=None, help='"512" 또는 "720x960"(HxW). 기본: 체크포인트 config')
    ap.add_argument("--conf-threshold", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}

    def pick(cli, key, fallback):
        return cli if cli is not None else cfg.get(key, fallback)

    model_name = pick(args.model, "model", "yolo_tiny")
    temporal_window = pick(args.temporal_window, "temporal_window", 5)
    max_frames = pick(args.max_frames, "max_frames", 3000)
    roi = parse_roi(pick(args.roi, "roi", 512))
    conf_threshold = pick(args.conf_threshold, "conf_threshold", 0.6)

    bin_path, csv_path = brownian_paths(DATA, roi)
    frames = load_frames_from_bin(bin_path, max_frames=max_frames, height=roi[0], width=roi[1])
    _, val_loader = make_train_val_loaders(
        frames, csv_path, batch_size=args.batch_size, temporal_window=temporal_window, roi_size=roi)

    model = get_model(model_name, input_channels=temporal_window)
    model.load_state_dict(ck["model_state_dict"])
    decoder = YOLOCenterDecoder(default_anchors(min(roi)), conf_threshold=conf_threshold)

    m = evaluate_regression(model, val_loader, roi_size=roi, device=args.device, to_xy=decoder)
    src = "from checkpoint config" if cfg else "defaults (config 없음 — 옛 체크포인트)"
    print(f"\n[yolo inference: {os.path.relpath(args.checkpoint, ROOT)}]")
    print(f"  config           : model={model_name} tw={temporal_window} roi={roi[0]}x{roi[1]} conf={conf_threshold}  ({src})")
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"  checkpoint epoch : {ck['epoch']}")
    print(f"  val samples      : {m['num_samples']}")
    print(f"  pixel_error_mean : {m['pixel_error_mean']:.4f} px  (std {m['pixel_error_std']:.4f})")
    print(f"  Acc@5px / @10px  : {m['accuracy_5px']:.2f}% / {m['accuracy_10px']:.2f}%")
    print(f"  detection rate   : {decoder.detection_rate:.1f}%")
    print(f"  throughput       : {m['fps']:.1f} FPS  (bs={args.batch_size}, decode 포함)")


if __name__ == "__main__":
    main()

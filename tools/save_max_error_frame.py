#!/usr/bin/env python3
"""최대 오차 프레임에 GT(+)·예측(x)을 겹쳐 저장 — 어떤 프레임에서 실패하는지 본다 (방식 공용).

  python tools/save_max_error_frame.py --checkpoint cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
  python tools/save_max_error_frame.py --pred-csv filter/results/no_filter_kalman.csv
"""

import argparse

from _common import add_source_args, default_out, load_source  # noqa: E402  (tools/ 안에서 실행)
from dvslib.eval.visualize import save_max_error_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="save max-error frame overlay")
    add_source_args(ap)
    args = ap.parse_args()

    r = load_source(args, need_frames=True)
    out, fi, err = save_max_error_frame(
        r["frames"], r["frame_idx"], r["pixel_errors"], r["predictions_px"], r["targets_px"],
        default_out(args, "max_error_frame"), label=r["label"],
    )
    print(f"max error {err:.2f}px at frame {fi}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

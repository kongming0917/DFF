#!/usr/bin/env python3
"""프레임 인덱스에 따른 픽셀 오차 그래프 — 어느 구간에서 오차가 커지는지 본다 (방식 공용).

  python tools/plot_error_vs_frame.py --checkpoint cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
  python tools/plot_error_vs_frame.py --pred-csv filter/results/no_filter_kalman.csv
"""

import argparse

from _common import add_source_args, default_out, load_source  # noqa: E402  (tools/ 안에서 실행)
from dvslib.eval.visualize import plot_error_vs_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="pixel error vs frame index")
    add_source_args(ap)
    args = ap.parse_args()

    r = load_source(args, need_frames=False)
    out = plot_error_vs_frame(
        r["frame_idx"], r["pixel_errors"], default_out(args, "error_vs_frame"),
        title=f"[{r['label']}] mean {r['metrics']['pixel_error_mean']:.2f}px",
    )
    print(f"samples={len(r['pixel_errors'])}  mean={r['metrics']['pixel_error_mean']:.4f}px")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

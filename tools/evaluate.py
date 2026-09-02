#!/usr/bin/env python3
"""방식 공용 평가 — 어떤 소스든 dvslib 지표(pixel error, Acc@Npx, FPS)로 보고한다.

baseline 기록·이식 전후 회귀 검증에 쓴다. 지표 정의는 dvslib.eval.metrics 한 곳.

  python tools/evaluate.py --checkpoint cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
  python tools/evaluate.py --yolo-checkpoint yolo/runs/baseline_yolo_tiny/yolo_tiny_best.pth
  python tools/evaluate.py --pred-csv filter/results/no_filter_kalman.csv --split val --max-frames 3000
"""

import argparse

from _common import add_source_args, load_source  # noqa: E402  (tools/ 안에서 실행)


def main():
    ap = argparse.ArgumentParser(description="evaluate predictions with dvslib metrics")
    add_source_args(ap)
    args = ap.parse_args()

    r = load_source(args, need_frames=False)
    m = r["metrics"]
    print(f"\n[evaluate: {r['label']}]  split={args.split}  roi={r['roi'][0]}x{r['roi'][1]}")
    print(f"  samples          : {m['num_samples']}")
    print(f"  pixel_error_mean : {m['pixel_error_mean']:.4f} px  (std {m['pixel_error_std']:.4f})")
    if "rmse" in m:
        print(f"  rmse / mae       : {m['rmse']:.4f} / {m['mae']:.4f} px")
    if "accuracy_5px" in m:
        print(f"  Acc@5px / @10px  : {m['accuracy_5px']:.2f}% / {m['accuracy_10px']:.2f}%")
    if "fps" in m:
        print(f"  throughput       : {m['fps']:.1f} FPS")
    if "detection_rate" in m:
        print(f"  detection rate   : {m['detection_rate']:.1f}%  (conf>{m['conf_threshold']})")


if __name__ == "__main__":
    main()

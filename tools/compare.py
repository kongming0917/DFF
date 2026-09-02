#!/usr/bin/env python3
"""세 방식(CNN · YOLO · Filter)을 **같은 blocked val 프레임 집합**에서 dvslib 지표로 비교한다.

옛 compare_brownian.py(처음 100프레임 순차 = 학습 구간, PNG만)를 대체한다. 소스는 몇 개든 섞어 넣을 수 있다.
결과는 로컬(compare_result/<run>/)에 항상 남기고, --wandb면 같은 표를 wandb Table로도 올린다.

  python tools/compare.py \\
      --cnn mobilenet_v2=cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth \\
      --cnn mobileone_s0=cnn/runs/baseline_mobileone_s0_pretrained/mobileone_s0_best.pth \\
      --yolo yolo_tiny=yolo/runs/baseline_yolo_tiny/yolo_tiny_best.pth \\
      --csv filter_kalman=filter/results/no_filter_kalman.csv \\
      --csv filter_median=filter/results/no_filter_median.csv

출력: summary.csv / summary.md (방식별 지표), per_frame.csv (프레임별 오차, 방식별 열).
--plot 을 주면 comparison.png(CDF·박스플롯·프레임별 오차)도 저장 (기본은 생략 — 용량).
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _common import ROOT, cnn_predict, csv_predict, yolo_predict  # noqa: E402  (tools/ 안에서 실행)
from dvslib.eval.visualize import plot_method_comparison  # noqa: E402

COLUMNS = ["method", "samples", "mean_px", "std_px", "median_px", "rmse_px", "acc_5px", "acc_10px", "fps"]


def _parse(spec: str) -> Tuple[str, str]:
    """"name=path" 또는 "path" → (name, path)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    return os.path.splitext(os.path.basename(spec))[0], spec


def _row(name: str, r: Dict) -> List:
    e = np.asarray(r["pixel_errors"], dtype=np.float64)
    m = r["metrics"]
    return [name, int(len(e)), float(e.mean()), float(e.std()), float(np.median(e)),
            float(np.sqrt(np.mean(e ** 2))), float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100),
            float(m["fps"]) if "fps" in m else float("nan")]


def _md_table(df: pd.DataFrame) -> str:
    fmt = {"mean_px": "{:.2f}", "std_px": "{:.2f}", "median_px": "{:.2f}", "rmse_px": "{:.2f}",
           "acc_5px": "{:.1f}", "acc_10px": "{:.1f}", "fps": "{:.0f}"}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        cells = [(fmt[c].format(r[c]) if c in fmt and not pd.isna(r[c]) else ("-" if pd.isna(r[c]) else str(r[c])))
                 for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="cross-method comparison on blocked val (dvslib metrics)")
    ap.add_argument("--cnn", action="append", default=[], metavar="NAME=CKPT", help="cnn 체크포인트 (반복 가능)")
    ap.add_argument("--yolo", action="append", default=[], metavar="NAME=CKPT", help="yolo 체크포인트 (반복 가능)")
    ap.add_argument("--csv", action="append", default=[], metavar="NAME=CSV", help="픽셀 좌표 CSV, filter/run.py 출력 등")
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--conf-threshold", type=float, default=None, help="yolo 검출 임계값 (기본: ckpt config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None, help="run 이름 (기본: 날짜시각)")
    ap.add_argument("--out-dir", default=None, help="기본: compare_result/<name>/")
    ap.add_argument("--plot", action="store_true", help="comparison.png 저장 (기본 생략)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="dvs-laser")
    args = ap.parse_args()

    if not (args.cnn or args.yolo or args.csv):
        ap.error("at least one of --cnn / --yolo / --csv is required")
    name = args.name or time.strftime("%Y%m%d_%H%M")
    out_dir = args.out_dir or os.path.join(ROOT, "compare_result", name)
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, Dict] = {}
    for spec in args.cnn:
        n, p = _parse(spec)
        results[n] = cnn_predict(p, device=args.device, max_frames=args.max_frames, split="val")
    for spec in args.yolo:
        n, p = _parse(spec)
        results[n] = yolo_predict(p, device=args.device, max_frames=args.max_frames, split="val",
                                  conf_threshold=args.conf_threshold)
    for spec in args.csv:
        n, p = _parse(spec)
        results[n] = csv_predict(p, max_frames=args.max_frames, split="val")

    # 같은 프레임 집합으로 정렬 (교집합). 다르면 경고 — 비교의 전제.
    sets = {n: set(int(i) for i in r["frame_idx"]) for n, r in results.items()}
    common = set.intersection(*sets.values())
    for n, s in sets.items():
        if s != common:
            print(f"warning: {n} has {len(s)} frames, using intersection {len(common)}")
    order = sorted(common)
    for n, r in results.items():
        pos = {int(f): k for k, f in enumerate(r["frame_idx"])}
        sel = [pos[f] for f in order]
        r["frame_idx"] = np.asarray(order)
        r["pixel_errors"] = np.asarray(r["pixel_errors"])[sel]

    summary = pd.DataFrame([_row(n, r) for n, r in results.items()], columns=COLUMNS)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_frame = pd.DataFrame({"frame_idx": order, **{n: r["pixel_errors"] for n, r in results.items()}})
    per_frame.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False)
    md = _md_table(summary)
    header = (f"# comparison: {name}\n\nsplit=blocked val (center frames), max_frames={args.max_frames}, "
              f"samples={len(order)}\n\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(header + md + "\n")
    png = None
    if args.plot:
        png = plot_method_comparison(results, os.path.join(out_dir, "comparison.png"),
                                     title=f"blocked val, {len(order)} frames")

    print(header + md)
    files = "summary.csv, summary.md, per_frame.csv" + (", comparison.png" if png else "")
    print(f"\nsaved: {os.path.relpath(out_dir, ROOT)}/ ({files})")

    if args.wandb:
        from dvslib.tracking.wandb_logger import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=f"compare-{name}",
                             config=dict(max_frames=args.max_frames, split="blocked val", sources=vars(args)),
                             tags=["compare"], enabled=True)
        logger.table("comparison/summary", COLUMNS, summary.values.tolist())
        if png:
            logger.image("comparison/plot", png)
        logger.finish()


if __name__ == "__main__":
    main()

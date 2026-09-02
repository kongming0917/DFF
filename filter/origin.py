#!/usr/bin/env python3
"""정지 레이저 bin에서 원점(레이저 중심) 후보를 여러 추출기로 뽑아 보여준다 — 데이터셋 GT `initial_center` 결정용.

brownian 데이터셋의 GT 원점 (541, 361)은 원본 `gaussian_large.bin`(정지 레이저, 720x960)에서 이 방식의
여러 추출기 결과를 종합해 **수동으로** 정한 값이다. 새 원본 데이터를 찍으면 같은 절차로 후보를 뽑아 정한다.
자동으로 하나를 고르지 않는다 — 표를 보고 사람이 결정한다.

  python filter/origin.py data/gaussian_large.bin --max-frames 200
  python filter/origin.py <bin> --height 512 --width 512 --spatial   # ROI bin, SpatialClusterFilter 적용
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from dvs_filter import (  # noqa: E402
    FilterableBinProcessor, SpatialClusterFilter,
    MeanPointExtractor, MedianPointExtractor, KalmanPointExtractor,
)


def main():
    ap = argparse.ArgumentParser(description="static-laser origin candidates from several extractors")
    ap.add_argument("bin_file")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--spatial", action="store_true", help="SpatialClusterFilter(radius 5, min_neighbors 2) 적용")
    args = ap.parse_args()

    proc = FilterableBinProcessor(args.width, args.height, has_header=True)
    frames = proc.read_frames(args.bin_file, max_frames=args.max_frames)
    if args.spatial:
        f = SpatialClusterFilter(radius=5.0, min_neighbors=2, use_fast_mode=True)
        frames = [fr for fr in (x.copy() for x in frames) if f.apply(fr)]
    print(f"{os.path.relpath(args.bin_file, ROOT)}: {len(frames)} frames, {args.width}x{args.height}"
          f"{', spatial filter' if args.spatial else ''}")

    extractors = [MeanPointExtractor(), MedianPointExtractor(),
                  KalmanPointExtractor(process_noise=4.0, measurement_noise=8.0)]
    print(f"\n{'extractor':<28} {'mean x':>9} {'mean y':>9} {'median x':>9} {'median y':>9} {'std x':>7} {'std y':>7} {'valid':>6}")
    for ex in extractors:
        if hasattr(ex, "reset"):
            ex.reset()
        pts = [ex.extract(fr) for fr in frames]
        pts = np.array([p for p in pts if p is not None], dtype=float)
        if len(pts) == 0:
            print(f"{ex.get_name():<28} (no valid centers)")
            continue
        mx, my = pts.mean(0); qx, qy = np.median(pts, 0); sx, sy = pts.std(0)
        print(f"{ex.get_name():<28} {mx:9.2f} {my:9.2f} {qx:9.2f} {qy:9.2f} {sx:7.2f} {sy:7.2f} {len(pts):6d}")
    print("\n위 후보를 보고 initial_center를 사람이 정한다 (tools/generate_brownian_dataset.py의 BrownianConfig).")


if __name__ == "__main__":
    main()

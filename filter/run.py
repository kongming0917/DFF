#!/usr/bin/env python3
"""filter experiment — 필터 체인 + 중심점 추출기로 프레임별 중심 좌표 CSV를 만든다 (학습 없음).

옛 filter_brownian_sim/export_center_data.py 이관. 동작(필터·추출기·파라미터·처리 순서)은 그대로이고,
bin 읽기는 dvslib.data, 경로는 인자. 평가·시각화는 tools/ (evaluate.py / plot_error_vs_frame.py --pred-csv).

CSV 컬럼: frame_idx(0부터, GT의 frame_idx와 동일), frame_number(bin 헤더), center_x, center_y (ROI 픽셀 좌표).
※ 옛 evaluate_against_ground_truth.py는 frame_number(88부터)를 frame_idx에 병합해 88프레임 어긋난 비교였다.
   tools/_common.csv_predict는 행 순서(=frame_idx)로 GT와 맞춘다.

  python filter/run.py --max-frames 3000                 # median·kalman × {no_filter, spatial_filter} 4개 CSV
  python filter/run.py --extractors median --no-spatial  # 조합 선택
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from typing import List, Optional, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from dvslib.data.dataset import brownian_paths, parse_roi  # noqa: E402
from dvs_filter import (  # noqa: E402
    FilterableBinProcessor, SpatialClusterFilter,
    MeanPointExtractor, MedianPointExtractor, KalmanPointExtractor, TemporalAveragePointExtractor,
)

DATA = os.path.join(ROOT, "data")


def make_extractor(name: str, temporal_window: int = 3):
    if name == "median":
        return MedianPointExtractor()
    if name == "mean":
        return MeanPointExtractor()
    if name == "kalman":
        # Brownian motion: sigma = 2.0 → process_noise = sigma^2 = 4.0 (옛 설정 그대로)
        return KalmanPointExtractor(process_noise=4.0, measurement_noise=8.0)
    if name == "temporal":
        return TemporalAveragePointExtractor(window_size=temporal_window, base_extractor=MedianPointExtractor())
    raise ValueError(f"unknown extractor: {name}")


_WORKER_FILTERS = None


def _init_worker(filter_specs):
    """워커별로 필터 객체를 생성 (필터는 프레임 단위 독립 → 병렬화해도 결과 동일)."""
    global _WORKER_FILTERS
    _WORKER_FILTERS = [make_filter(**spec) for spec in filter_specs]


def _apply_filters(item):
    idx, frame = item
    cur = frame.copy()
    for f in _WORKER_FILTERS:
        if not f.apply(cur):
            return idx, None
    return idx, cur


def make_filter(kind: str, **kw):
    if kind == "spatial":
        return SpatialClusterFilter(radius=kw["radius"], min_neighbors=kw["min_neighbors"], use_fast_mode=True)
    raise ValueError(f"unknown filter: {kind}")


def filter_frames(frames, filter_specs, workers: int = 1):
    """필터 체인을 모든 프레임에 적용해 [(frame_idx, frame)] 반환 (걸러진 프레임 제외, 순서 유지).

    필터는 프레임마다 독립이므로 workers>1이면 multiprocessing으로 나눈다. 순서는 인덱스로 복원.
    추출기(Kalman·Temporal)는 순차 상태를 가지므로 여기서 병렬화하지 않고 호출자가 순서대로 돌린다.
    """
    if not filter_specs:
        return list(enumerate(frames))
    items = list(enumerate(frames))
    if workers <= 1:
        _init_worker(filter_specs)
        results = [_apply_filters(it) for it in items]
    else:
        chunk = max(1, len(items) // (workers * 4))
        with Pool(workers, initializer=_init_worker, initargs=(filter_specs,)) as pool:
            results = pool.map(_apply_filters, items, chunksize=chunk)
    return [(idx, fr) for idx, fr in results if fr is not None]


def extract_centers(kept, extractor) -> List[Tuple[int, int, Optional[float], Optional[float]]]:
    """필터 통과 프레임을 순서대로 추출기에 넣어 (frame_idx, frame_number, x, y). 상태 추출기는 시작 시 reset."""
    if hasattr(extractor, "reset"):
        extractor.reset()
    out = []
    for idx, frame in kept:
        c = extractor.extract(frame)
        out.append((idx, frame.header.frame_number,
                    float(c[0]) if c is not None else None, float(c[1]) if c is not None else None))
    return out


def main():
    ap = argparse.ArgumentParser(description="filter experiment — center extraction to CSV")
    ap.add_argument("--roi", default="512", help='"512" 또는 "720x960"(HxW) → data/gaussian_brownian_<tag>.bin')
    ap.add_argument("--bin", default=None, help="bin 경로 직접 지정 (--roi 대신)")
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--extractors", nargs="+", default=["median", "kalman"],
                    choices=["median", "mean", "kalman", "temporal"])
    ap.add_argument("--temporal-window", type=int, default=3)
    ap.add_argument("--no-spatial", action="store_true", help="SpatialClusterFilter 조건 생략")
    ap.add_argument("--no-plain", action="store_true", help="필터 없음 조건 생략")
    ap.add_argument("--spatial-radius", type=float, default=5.0)
    ap.add_argument("--spatial-min-neighbors", type=int, default=2)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--workers", type=int, default=16, help="필터 단계 병렬 프로세스 수 (1이면 순차)")
    args = ap.parse_args()

    roi = parse_roi(args.roi)
    bin_path = args.bin or brownian_paths(DATA, roi)[0]
    os.makedirs(args.out_dir, exist_ok=True)

    processor = FilterableBinProcessor(roi[1], roi[0], has_header=True)  # (width, height)
    t0 = time.time()
    frames = processor.read_frames(bin_path, max_frames=args.max_frames)
    print(f"read {len(frames)} frames from {os.path.relpath(bin_path, ROOT)} ({time.time() - t0:.1f}s)")

    conditions = []
    if not args.no_plain:
        conditions.append(("no_filter", []))
    if not args.no_spatial:
        conditions.append(("spatial_filter", [dict(kind="spatial", radius=args.spatial_radius,
                                                   min_neighbors=args.spatial_min_neighbors)]))

    # 필터는 조건당 한 번만 적용하고(비용 지배적), 그 결과를 모든 추출기가 공유한다.
    for cond_name, specs in conditions:
        t0 = time.time()
        kept = filter_frames(frames, specs, workers=args.workers if specs else 1)
        print(f"[{cond_name}] kept {len(kept)}/{len(frames)} frames ({time.time() - t0:.1f}s, workers={args.workers if specs else 1})")
        for ext_name in args.extractors:
            extractor = make_extractor(ext_name, args.temporal_window)
            t0 = time.time()
            rows = extract_centers(kept, extractor)
            df = pd.DataFrame(rows, columns=["frame_idx", "frame_number", "center_x", "center_y"])
            path = os.path.join(args.out_dir, f"{cond_name}_{ext_name}.csv")
            df.to_csv(path, index=False)
            valid = int(df["center_x"].notna().sum())
            print(f"  {extractor.get_name():<28} valid={valid:>5}/{len(df):<5} ({time.time() - t0:.1f}s) -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

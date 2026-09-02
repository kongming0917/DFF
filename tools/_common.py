"""tools/ 공용 도우미 — 예측 결과를 "프레임 인덱스·픽셀 오차·좌표" 배열로 통일한다.

시각화(dvslib.eval.visualize)는 방식을 모르므로, 방식별로 예측을 얻는 부분만 여기서 흡수한다.
  - cnn_predict : cnn 체크포인트 → dvslib 파이프라인으로 추론
  - csv_predict : (frame_number, center_x, center_y) CSV (Filter 결과 등) → GT와 매칭
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
DATA = os.path.join(ROOT, "data")
for _p in (ROOT, os.path.join(ROOT, "cnn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dvslib.data.dataset import (  # noqa: E402
    DVSBrownianDataset, brownian_paths, load_frames_from_bin, parse_roi,
)


def _load_cnn_get_model():
    """cnn/model.py의 get_model을 경로로 직접 로드 — sys.path에 yolo 등 다른 `model.py`가 있어도 안전."""
    import importlib.util
    path = os.path.join(ROOT, "cnn", "model.py")
    spec = importlib.util.spec_from_file_location("cnn_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model


def _to_px(norm_xy: np.ndarray, roi) -> np.ndarray:
    h, w = roi
    return np.asarray(norm_xy, dtype=np.float64) * np.array([w, h], dtype=np.float64)


def cnn_predict(
    checkpoint: str,
    device: str = "cuda",
    max_frames: Optional[int] = None,
    batch_size: int = 32,
    split: str = "val",
) -> Dict:
    """cnn 체크포인트로 추론해 샘플별 결과를 반환.

    split="val": blocked split 검증셋(baseline과 동일). split="all": 처음 max_frames 프레임의
    모든 sliding window (옛 compare_brownian과 동일한 순차 비교용).
    """
    import torch
    from torch.utils.data import DataLoader
    from dvslib.data.split import make_train_val_loaders
    from dvslib.eval.evaluate import evaluate_regression
    get_model = _load_cnn_get_model()

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    model_name = cfg.get("model", "mobilenet_v2")
    tw = int(cfg.get("temporal_window", 5))
    roi = parse_roi(cfg.get("roi", 512))
    if max_frames is None:
        max_frames = int(cfg.get("max_frames", 3000))

    bin_path, csv_path = brownian_paths(DATA, roi)
    frames = load_frames_from_bin(bin_path, max_frames=max_frames, height=roi[0], width=roi[1])

    if split == "val":
        _, loader = make_train_val_loaders(
            frames, csv_path, batch_size=batch_size, temporal_window=tw, roi_size=roi)
        sample_idx = np.asarray(loader.dataset.indices)
    elif split == "all":
        ds = DVSBrownianDataset(frames, csv_path, roi_size=roi, temporal_window=tw)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        sample_idx = np.arange(len(ds))
    else:
        raise ValueError(f"unknown split: {split}")

    model = get_model(model_name, input_channels=tw, output_dim=2, use_qat=False)
    model.load_state_dict(ck["model_state_dict"])
    if hasattr(model, "reparameterize"):
        model.reparameterize()

    m = evaluate_regression(model, loader, roi_size=roi, device=device, return_predictions=True)
    return dict(
        label=f"cnn/{model_name}",
        roi=roi,
        frames=frames,
        frame_idx=sample_idx + tw // 2,          # 각 window의 중심 프레임
        pixel_errors=m["pixel_errors"],
        predictions_px=_to_px(m["predictions"], roi),
        targets_px=_to_px(m["targets"], roi),
        metrics={k: v for k, v in m.items() if not isinstance(v, np.ndarray)},
    )


def csv_predict(
    pred_csv: str,
    roi="512",
    max_frames: Optional[int] = None,
    load_frames: bool = False,
) -> Dict:
    """(frame_number, center_x, center_y) 픽셀 좌표 CSV를 GT와 매칭.

    Filter 결과 CSV(filter_brownian_sim/csv_results)는 bin을 읽은 순서대로 기록되므로
    i번째 행 = 프레임 인덱스 i 로 본다 (옛 compare_brownian과 동일한 가정).
    """
    roi = parse_roi(roi)
    bin_path, labels_csv = brownian_paths(DATA, roi)
    pred = pd.read_csv(pred_csv).dropna().reset_index(drop=True)
    gt = pd.read_csv(labels_csv).set_index("frame_idx")
    if max_frames is not None:
        pred = pred.iloc[:max_frames]

    frame_idx, p_px, t_px = [], [], []
    for i, row in pred.iterrows():
        if i not in gt.index:
            continue
        g = gt.loc[i]
        if "roi_center_x" in gt.columns:
            t = (float(g["roi_center_x"]), float(g["roi_center_y"]))
        else:
            t = (float(g["cnn_rel_x"]) * roi[1], float(g["cnn_rel_y"]) * roi[0])
        frame_idx.append(i)
        p_px.append((float(row["center_x"]), float(row["center_y"])))
        t_px.append(t)

    p_px = np.asarray(p_px); t_px = np.asarray(t_px)
    errors = np.sqrt(np.sum((p_px - t_px) ** 2, axis=1))
    frames = None
    if load_frames:
        n = (int(max(frame_idx)) + 1) if frame_idx else 0
        frames = load_frames_from_bin(bin_path, max_frames=n, height=roi[0], width=roi[1])
    return dict(
        label=os.path.splitext(os.path.basename(pred_csv))[0],
        roi=roi, frames=frames,
        frame_idx=np.asarray(frame_idx), pixel_errors=errors,
        predictions_px=p_px, targets_px=t_px,
        metrics=dict(pixel_error_mean=float(errors.mean()), pixel_error_std=float(errors.std()),
                     num_samples=int(len(errors))),
    )


def add_source_args(ap):
    """--checkpoint(cnn) 또는 --pred-csv(픽셀 좌표 CSV) 중 하나를 받는 공통 인자."""
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="cnn 체크포인트(.pth) — dvslib 파이프라인으로 추론")
    g.add_argument("--pred-csv", help="(frame_number, center_x, center_y) 픽셀 좌표 CSV (Filter 결과 등)")
    ap.add_argument("--roi", default="512", help='--pred-csv일 때 ROI: "512" 또는 "720x960"')
    ap.add_argument("--split", default="val", choices=["val", "all"],
                    help="--checkpoint일 때 평가 구간: val(blocked 검증셋) / all(처음 max-frames 전부)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="출력 PNG 경로 (기본: 체크포인트/CSV 옆)")


def load_source(args, need_frames: bool) -> Dict:
    if args.checkpoint:
        return cnn_predict(args.checkpoint, device=args.device, max_frames=args.max_frames,
                           split=args.split)
    return csv_predict(args.pred_csv, roi=args.roi, max_frames=args.max_frames,
                       load_frames=need_frames)


def default_out(args, suffix: str) -> str:
    src = args.checkpoint or args.pred_csv
    base = os.path.splitext(os.path.basename(src))[0]
    return args.out or os.path.join(os.path.dirname(os.path.abspath(src)), f"{base}_{suffix}.png")

"""Accuracy metrics — computed in one place so they mean the same across methods.

cnn_brownian_sim의 검증 metric 정의와 동일 (train.py): 정규화 좌표를 픽셀로 환산한 뒤
유클리드 오차, Acc@Npx = (오차 <= N)의 비율 * 100.
"""

from typing import Dict, Sequence, Tuple

import numpy as np


def pixel_error_metrics(
    pred_norm: Sequence,
    target_norm: Sequence,
    roi_size: Tuple[int, int] = (512, 512),
    thresholds: Sequence[float] = (5.0, 10.0),
) -> Dict[str, float]:
    """Pixel-space metrics from normalized (x, y) predictions/targets.

    pred_norm, target_norm: shape (N, 2), [0,1] 정규화 좌표.
    roi_size: (H, W) — 정규화 좌표를 픽셀로 환산하는 스케일.
    """
    pred = np.asarray(pred_norm, dtype=np.float64)
    tgt = np.asarray(target_norm, dtype=np.float64)
    roi_h, roi_w = roi_size
    scale = np.array([roi_w, roi_h])  # x→W, y→H

    errors = np.sqrt(np.sum(((pred - tgt) * scale) ** 2, axis=1))
    out: Dict[str, float] = {
        "pixel_error_mean": float(np.mean(errors)),
        "pixel_error_std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs((pred - tgt) * scale))),  # 픽셀 좌표 절대오차 (per-coord)
    }
    for t in thresholds:
        out[f"accuracy_{int(t)}px"] = float(np.mean(errors <= t) * 100.0)
    return out

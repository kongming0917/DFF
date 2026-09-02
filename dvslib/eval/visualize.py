"""Method-agnostic result visualization — pixel error vs frame, worst-frame overlay.

cnn_brownian_sim의 plot_error_vs_frame.py / save_max_error_frame.py에서 이관.
입력은 (프레임 인덱스, 픽셀 오차, 좌표) 배열뿐이라 CNN·YOLO·Filter 어느 방식의 결과에도
그대로 쓴다. 예측을 얻는 일(모델 추론, CSV 로드)은 호출자(tools/) 몫이다.
"""

import os
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_error_vs_frame(
    frame_idx: Sequence[int],
    pixel_errors: Sequence[float],
    out_path: str,
    title: str = "",
    high_percentile: float = 90.0,
) -> str:
    """프레임 인덱스에 따른 픽셀 오차 산점도 + 상위 오차 샘플의 프레임 분포 히스토그램.

    어느 구간(시간대)에서 큰 오차가 몰리는지 보는 용도.
    """
    frame_idx = np.asarray(frame_idx)
    errors = np.asarray(pixel_errors, dtype=np.float64)
    if len(errors) == 0:
        raise ValueError("pixel_errors is empty")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.scatter(frame_idx, errors, s=2, alpha=0.5)
    ax1.set_xlabel("Center frame index")
    ax1.set_ylabel("Pixel error")
    ax1.set_title(f"Pixel error vs center frame index{('  ' + title) if title else ''}")
    ax1.grid(True, alpha=0.3)

    thresh = np.percentile(errors, high_percentile)
    high = errors >= thresh
    ax2.hist(frame_idx[high], bins=50, color="coral", alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Center frame index")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Frame index distribution (error >= {high_percentile:.0f}th percentile "
        f"= {thresh:.2f}px, {int(high.sum())} samples)"
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_frame_overlay(
    frame: np.ndarray,
    gt_px: Tuple[float, float],
    pred_px: Tuple[float, float],
    out_path: str,
    title: str = "",
) -> str:
    """단일 프레임 위에 GT(+)와 예측(x)을 겹쳐 그려 저장. 좌표는 픽셀 (x, y)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(np.asarray(frame), cmap="gray")
    ax.scatter([gt_px[0]], [gt_px[1]], c="lime", s=120, marker="+", linewidths=3, label="GT")
    ax.scatter([pred_px[0]], [pred_px[1]], c="red", s=120, marker="x", linewidths=3, label="Pred")
    ax.legend(loc="upper right", fontsize=12)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_max_error_frame(
    frames: Sequence[np.ndarray],
    frame_idx: Sequence[int],
    pixel_errors: Sequence[float],
    predictions_px: np.ndarray,
    targets_px: np.ndarray,
    out_path: str,
    label: str = "",
) -> Tuple[str, int, float]:
    """최대 오차 샘플을 찾아 그 중심 프레임에 GT·예측을 오버레이해 저장.

    frames: 프레임 인덱스로 접근 가능한 시퀀스 (frames[frame_idx[i]]가 i번째 샘플의 중심 프레임).
    반환: (저장 경로, 최대 오차 샘플의 프레임 인덱스, 최대 오차 px).
    """
    errors = np.asarray(pixel_errors, dtype=np.float64)
    i = int(np.argmax(errors))
    fi = int(frame_idx[i])
    title = f"Max error frame (frame={fi}, error={errors[i]:.2f} px)"
    if label:
        title = f"{label}: " + title
    save_frame_overlay(frames[fi], tuple(targets_px[i]), tuple(predictions_px[i]), out_path, title)
    return out_path, fi, float(errors[i])

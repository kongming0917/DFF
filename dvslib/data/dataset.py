"""DVS Brownian-motion dataset and frame loading.

cnn_brownian_sim에서 추출. 동작은 그대로 유지하되(baseline 재현 보장) 라이브러리답게
print 노이즈·vestigial 코드를 제거하고 타입을 명확히 했다.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dvslib.data.bin_processor import BinProcessor

# CSV ground-truth에 반드시 있어야 하는 컬럼
REQUIRED_LABEL_COLS = ["frame_idx", "shift_x", "shift_y", "cnn_rel_x", "cnn_rel_y"]


def parse_roi(spec) -> Tuple[int, int]:
    """ROI 지정을 (H, W)로 정규화. "512" / 512 → (512, 512), "720x960" → (720, 960), (H, W) 그대로."""
    if isinstance(spec, (tuple, list)):
        h, w = spec
        return int(h), int(w)
    text = str(spec).lower()
    if "x" in text:
        h, w = text.split("x")
        return int(h), int(w)
    n = int(text)
    return n, n


def roi_tag(roi: Tuple[int, int]) -> str:
    """(H, W) → 데이터 파일명 태그. 정사각은 "512x512", 비정사각은 "720x960"."""
    h, w = roi
    return f"{h}x{w}"


def brownian_paths(data_dir: str, roi) -> Tuple[str, str]:
    """ROI에 대응하는 brownian 데이터셋 (bin, labels csv) 경로."""
    import os
    tag = roi_tag(parse_roi(roi))
    return (
        os.path.join(data_dir, f"gaussian_brownian_{tag}.bin"),
        os.path.join(data_dir, f"gaussian_brownian_{tag}_labels.csv"),
    )


def load_frames_from_bin(
    bin_file_path: str,
    max_frames: Optional[int] = None,
    height: int = 512,
    width: int = 512,
) -> List[np.ndarray]:
    """Load individual DVS frames from a 2-bit packed .bin file.

    프레임은 raw 이산값(0/1/2)을 담은 **uint8 그대로** 반환한다 — 정규화하지 않으며
    (MobileOne 첫 BatchNorm이 스케일을 흡수), float 변환은 `__getitem__`에서 한 번만 한다.
    (옛 코드는 여기서 float32로 캐스팅했지만 그건 제거된 정규화 단계의 잔재였다.)
    max_frames 미지정 시 200.
    """
    processor = BinProcessor(frame_width=width, frame_height=height, has_header=True)
    n = 200 if max_frames is None else max_frames
    frames_data = processor.read_frames(bin_file_path, max_frames=n)
    return [f.raw_data for f in frames_data]  # uint8 (H, W)


class DVSBrownianDataset(Dataset):
    """Overlapping temporal-window dataset over pre-shifted 512x512 ROI frames.

    Brownian motion은 데이터 생성 시점에 이미 적용됨 → 런타임 ROI 추출/증강 없음.
    레이블은 temporal window의 중심 프레임 기준 (CSV의 cnn_rel_x/y, [0,1] 정규화 좌표).
    """

    def __init__(
        self,
        individual_frames: List[np.ndarray],
        csv_labels_path: str,
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,
    ):
        self.frames = individual_frames
        self.roi_height, self.roi_width = roi_size  # 선언적 ROI 크기 (프레임은 이미 이 크기)
        self.temporal_window = temporal_window

        labels_df = pd.read_csv(csv_labels_path)
        missing = [c for c in REQUIRED_LABEL_COLS if c not in labels_df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        # frame_idx -> (cnn_rel_x, cnn_rel_y) 사전 계산 (매 __getitem__ pandas .loc 조회 제거)
        self._labels = {
            int(fi): (float(x), float(y))
            for fi, x, y in zip(
                labels_df["frame_idx"], labels_df["cnn_rel_x"], labels_df["cnn_rel_y"]
            )
        }

        # overlapping sliding window 유효 샘플 수
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)

    def __len__(self) -> int:
        return self.valid_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.valid_samples:
            raise IndexError(f"Index {idx} out of range for {self.valid_samples} samples")

        # temporal window = [idx, idx+T); label은 중심 프레임 기준
        center_frame_idx = idx + self.temporal_window // 2
        rel_x, rel_y = self._labels.get(center_frame_idx, (0.5, 0.5))  # 경계: CSV에 없으면 중앙

        window = self.frames[idx : idx + self.temporal_window]            # uint8 (H, W) × T
        multi_frame_tensor = torch.from_numpy(np.stack(window)).float()   # (T, H, W), 0/1/2 → float32
        label_tensor = torch.tensor([rel_x, rel_y], dtype=torch.float32)
        return multi_frame_tensor, label_tensor

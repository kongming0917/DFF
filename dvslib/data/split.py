"""Time-series split — avoid temporal leakage by construction.

DVS 데이터는 시계열이라 랜덤 split이면 인접 프레임이 학습/검증에 동시에 들어간다.
blocked split은 연속 블록 단위로 나누고 각 블록 끝에서 gap 프레임을 버려 경계를 떼어놓는다.
모든 방식이 이 함수를 써서 split이 갈라지지 않게 한다.
"""

from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from dvslib.data.dataset import DVSBrownianDataset


def blocked_indices(
    n: int,
    num_blocks: int = 5,
    gap: int = 50,
    val_blocks: Iterable[int] = (1, 3),
) -> Tuple[List[int], List[int]]:
    """Split [0, n) into train/val index lists by contiguous blocks.

    각 블록은 [start, start + block_len - gap) 범위만 사용 (gap만큼 버림).
    baseline과 동일: num_blocks=5, gap=50, val_blocks={1,3}.
    """
    val_set = set(val_blocks)
    block_len = n // num_blocks
    train_idx: List[int] = []
    val_idx: List[int] = []
    for b in range(num_blocks):
        start = b * block_len
        end = start + block_len - gap
        block = list(range(start, end))
        (val_idx if b in val_set else train_idx).extend(block)
    return train_idx, val_idx


def make_train_val_loaders(
    individual_frames: Sequence,
    csv_labels_path: str,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    num_blocks: int = 5,
    gap: int = 50,
    val_blocks: Iterable[int] = (1, 3),
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Build blocked-split train/val DataLoaders from pre-loaded frames.

    train/val은 동일 dataset(증강 없음)에 대한 두 index Subset일 뿐이라 dataset은 하나만 만든다.
    """
    dataset = DVSBrownianDataset(individual_frames, csv_labels_path, **dataset_kwargs)

    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset: no valid samples from the given frames.")
    train_idx, val_idx = blocked_indices(n, num_blocks, gap, val_blocks)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader

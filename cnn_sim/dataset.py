#!/usr/bin/env python3
"""
DVS 데이터를 PyTorch Dataset으로 처리하는 모듈 (간소화 버전)
- ROI 추출 및 Random Shift 기반 데이터 증강
- 개별 timestamp 프레임 처리
- 오버래핑 슬라이딩 윈도우
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import os


class DVSFixedGTDataset(Dataset):
    """
    Fixed Ground Truth를 사용하는 DVS 데이터 증강 데이터셋
    
    핵심 기능:
    - ROI 추출 및 Random Shift 데이터 증강
    - 개별 timestamp 기반 프레임 처리
    - 오버래핑 슬라이딩 윈도우 (다중 프레임 입력)
    - 고정 ROI 크기로 구현 단순화
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        true_center_coord: Tuple[int, int] = (541, 361),
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,
        shift_range_x: int = 50,
        shift_range_y: int = 50,
        sensor_size: Tuple[int, int] = (720, 960)
    ):
        """
        DVS Fixed GT 데이터셋 초기화
        
        Args:
            individual_frames: 개별 timestamp 프레임 리스트
            true_center_coord: 레이저 광원의 실제 중심 좌표 (x, y)
            roi_size: ROI 크기 (height, width)
            temporal_window: 시간 윈도우 크기 (다중 채널 수)
            shift_range_x: X축 시프트 범위 (±픽셀)
            shift_range_y: Y축 시프트 범위 (±픽셀)
            sensor_size: 원본 센서 크기 (height, width)
        """
        # 기본 설정
        self.frames = individual_frames
        self.true_center_x, self.true_center_y = true_center_coord
        self.roi_height, self.roi_width = roi_size
        self.temporal_window = temporal_window
        self.shift_range_x = shift_range_x
        self.shift_range_y = shift_range_y
        self.sensor_height, self.sensor_width = sensor_size
        
        # 학습/검증 모드
        # 학습 시: 랜덤 shift (데이터 증강)
        # 검증 시: 랜덤 shift (다양한 평가, 단 시드 고정으로 재현성 확보)
        self.training = True
        self.validation_seed = 42  # 검증 시 재현성을 위한 시드
        
        # 오버래핑 슬라이딩 윈도우로 유효한 샘플 수 계산
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)
        
        print(f"   Individual frames: {len(self.frames)}")
        print(f"   Temporal window: {self.temporal_window}")
        print(f"   Valid samples: {self.valid_samples} (오버래핑 슬라이딩 윈도우)")
    
    def set_training_mode(self, training: bool):
        """
        학습/추론 모드 설정
        
        Args:
            training: True면 학습 모드 (랜덤 shift), False면 검증 모드 (랜덤 shift + 시드 고정)
        """
        self.training = training
    
    def _extract_roi_with_shift(self, frame: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
        """
        ROI 추출 (Shift 적용)
        
        Args:
            frame: 원본 프레임
            shift_x: X축 시프트 (픽셀)
            shift_y: Y축 시프트 (픽셀)
            
        Returns:
            정규화된 ROI (0-1 범위)
        """
        half_h, half_w = self.roi_height // 2, self.roi_width // 2
        
        # ROI 추출 중심 좌표 계산 (광원 중심에서 shift만큼 이동)
        # true_center: 레이저 광원의 실제 위치
        # center: ROI를 추출할 중심 위치 (shift 적용)
        center_x = self.true_center_x + shift_x
        center_y = self.true_center_y + shift_y
        
        # ROI 경계 계산 (센서 범위 내로 제한)
        y_start = max(0, center_y - half_h)
        y_end = min(self.sensor_height, center_y + half_h)
        x_start = max(0, center_x - half_w)
        x_end = min(self.sensor_width, center_x + half_w)
        
        # ROI 추출 및 정규화
        roi = np.zeros((self.roi_height, self.roi_width), dtype=np.float32)
        
        # 복사할 영역 계산 (경계 처리)
        src_h = y_end - y_start
        src_w = x_end - x_start
        dst_y = max(0, half_h - (center_y - y_start))
        dst_x = max(0, half_w - (center_x - x_start))
        
        copy_h = min(src_h, self.roi_height - dst_y)
        copy_w = min(src_w, self.roi_width - dst_x)
        
        if copy_h > 0 and copy_w > 0:
            roi[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                frame[y_start:y_start + copy_h, x_start:x_start + copy_w]
        
        # 정규화는 train.py에서 이미 적용됨 (이중 정규화 방지)
        return roi
    
    def _calculate_relative_label(self, shift_x: int, shift_y: int) -> Tuple[float, float]:
        """
        시프트에 따른 상대적 레이블 계산
        
        핵심 로직:
        - shift_x > 0: ROI가 오른쪽으로 이동 → 레이저는 ROI 내에서 왼쪽에 위치
        - shift_x < 0: ROI가 왼쪽으로 이동 → 레이저는 ROI 내에서 오른쪽에 위치
        - ROI 중심(0.5, 0.5)에서 shift 방향의 반대로 레이저 위치 계산
        
        Args:
            shift_x: X축 시프트 (픽셀)
            shift_y: Y축 시프트 (픽셀)
            
        Returns:
            ROI 내 상대적 위치 (0-1 범위)
        """
        rel_x = np.clip(0.5 - (shift_x / self.roi_width), 0.0, 1.0)
        rel_y = np.clip(0.5 - (shift_y / self.roi_height), 0.0, 1.0)
        return rel_x, rel_y
    
    def __len__(self) -> int:
        return self.valid_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        오버래핑 슬라이딩 윈도우로 다중 프레임 샘플 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            (multi_frame_tensor, label_tensor): 다중 프레임 텐서와 상대적 레이블
        """
        if idx >= self.valid_samples:
            raise IndexError(f"Index {idx} out of range for {self.valid_samples} samples")
        
        # 오버래핑 슬라이딩 윈도우로 프레임 선택
        frame_indices = list(range(idx, idx + self.temporal_window))
        
        # Shift 결정 (학습/검증 모두 랜덤 사용)
        if self.training:
            # 학습 시: 랜덤 shift (데이터 증강)
            shift_x = np.random.randint(-self.shift_range_x, self.shift_range_x + 1)
            shift_y = np.random.randint(-self.shift_range_y, self.shift_range_y + 1)
        else:
            # 검증 시: 랜덤 shift (재현성을 위해 idx 기반 시드 사용)
            # 매 에폭마다 동일한 shift 패턴 유지
            # rng = np.random.RandomState(self.validation_seed + idx)
            # shift_x = rng.randint(-self.shift_range_x, self.shift_range_x + 1)
            # shift_y = rng.randint(-self.shift_range_y, self.shift_range_y + 1)
            shift_x = 0
            shift_y = 0
        
        # 다중 프레임에서 ROI 추출 (동일한 shift 적용)
        multi_frame_rois = []
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            roi = self._extract_roi_with_shift(frame, shift_x, shift_y)
            multi_frame_rois.append(roi)
        
        # 다중 채널 텐서 생성 (temporal_window, H, W)
        multi_frame_tensor = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # 상대적 레이블 계산 (shift에 따른 ROI 내 위치)
        rel_x, rel_y = self._calculate_relative_label(shift_x, shift_y)
        label_tensor = torch.tensor([rel_x, rel_y], dtype=torch.float32)
        
        return multi_frame_tensor, label_tensor
    


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    훈련/검증 데이터로더 생성
    
    Args:
        individual_frames: 개별 프레임 리스트
        train_ratio: 훈련 데이터 비율
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        **kwargs: 데이터셋 생성 파라미터
        
    Returns:
        (train_loader, val_loader): 훈련/검증 데이터로더
    """
    
    print(f"📊 Creating dataset from {len(individual_frames)} individual frames...")
    
    # 전체 데이터셋 생성
    full_dataset = DVSFixedGTDataset(individual_frames, **kwargs)
    
    # 데이터셋 크기 확인
    dataset_size = len(full_dataset)
    print(f"   Dataset size: {dataset_size} samples")
    
    if dataset_size == 0:
        raise ValueError(f"❌ Dataset is empty! No valid samples from {len(individual_frames)} frames.")
    
    # 훈련/검증 분할
    if dataset_size < 2:
        print(f"   ⚠️ Very small dataset ({dataset_size} samples), using full dataset for both train/val")
        train_size = dataset_size
        val_size = 0
        train_dataset = full_dataset
        val_dataset = full_dataset
    else:
        train_size = max(1, int(train_ratio * dataset_size))
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # 훈련 모드 설정 (랜덤 shift 사용)
    if hasattr(train_dataset, 'dataset'):
        train_dataset.dataset.set_training_mode(True)
    else:
        train_dataset.set_training_mode(True)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 검증 모드 설정 (랜덤 shift + 시드 고정)
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.set_training_mode(False)
    else:
        val_dataset.set_training_mode(False)
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"   ✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_fixed_gt_dataloader(
    individual_frames: List[np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0,
    training: bool = True,
    **kwargs
) -> DataLoader:
    """Fixed GT 데이터로더 생성 (간소화)"""
    dataset = DVSFixedGTDataset(individual_frames, **kwargs)
    dataset.set_training_mode(training)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True
    )
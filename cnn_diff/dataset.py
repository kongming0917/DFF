#!/usr/bin/env python3
"""
dataset.py: DVS 데이터셋 로더
cnn_brownian_sim의 dataset.py를 참고하여 작성
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os
import sys

# 상위 디렉토리를 sys.path에 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)


class DVSDataset(Dataset):
    """
    DVS 이벤트 프레임 데이터셋
    - 개별 프레임 리스트와 CSV 레이블 파일 사용
    - Temporal window 처리 지원
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        csv_labels_path: str,
        roi_size: Tuple[int, int] = (128, 128),
        temporal_window: int = 1,
        sensor_size: Tuple[int, int] = (720, 960)
    ):
        """
        Args:
            individual_frames: 개별 timestamp 프레임 리스트
            csv_labels_path: CSV 레이블 파일 경로 (frame_idx, rel_x, rel_y 포함)
            roi_size: ROI 크기 (height, width)
            temporal_window: 시간 윈도우 크기 (다중 채널 수)
            sensor_size: 원본 센서 크기 (height, width)
        """
        self.frames = individual_frames
        self.roi_height, self.roi_width = roi_size
        self.temporal_window = temporal_window
        self.sensor_height, self.sensor_width = sensor_size
        
        # CSV 레이블 로드
        if not os.path.exists(csv_labels_path):
            raise FileNotFoundError(f"CSV labels file not found: {csv_labels_path}")
        
        self.labels_df = pd.read_csv(csv_labels_path)
        print(f"   Loaded {len(self.labels_df)} labels from {csv_labels_path}")
        
        # 레이블 검증
        required_cols = ['frame_idx', 'cnn_rel_x', 'cnn_rel_y']
        missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # frame_idx를 인덱스로 설정
        self.labels_df.set_index('frame_idx', inplace=True)
        
        # 오버래핑 슬라이딩 윈도우로 유효한 샘플 수 계산
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)
        
        print(f"   Individual frames: {len(self.frames)}")
        print(f"   Temporal window: {self.temporal_window}")
        print(f"   Valid samples: {self.valid_samples}")
    
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
        
        # 중심 프레임의 레이블 사용
        center_frame_idx = frame_indices[len(frame_indices) // 2]
        
        # CSV에서 해당 프레임의 레이블 가져오기
        if center_frame_idx in self.labels_df.index:
            rel_x = float(self.labels_df.loc[center_frame_idx, 'cnn_rel_x'])
            rel_y = float(self.labels_df.loc[center_frame_idx, 'cnn_rel_y'])
        else:
            # CSV에 없는 경우 (경계 처리)
            rel_x = 0.5
            rel_y = 0.5
        
        # 다중 프레임 사용
        multi_frame_rois = []
        for frame_idx in frame_indices:
            if frame_idx < len(self.frames):
                frame = self.frames[frame_idx]
                # 프레임 크기 조정 (필요 시)
                if frame.shape != (self.roi_height, self.roi_width):
                    # 간단한 리사이즈 (실제로는 더 정교한 방법 사용 가능)
                    from scipy.ndimage import zoom
                    zoom_factors = (self.roi_height / frame.shape[0], self.roi_width / frame.shape[1])
                    frame = zoom(frame, zoom_factors, order=1)
                roi = frame.astype(np.float32)
                multi_frame_rois.append(roi)
            else:
                # 경계 처리: 마지막 프레임 반복
                if len(multi_frame_rois) > 0:
                    multi_frame_rois.append(multi_frame_rois[-1])
                else:
                    # 첫 프레임이 없는 경우
                    multi_frame_rois.append(np.zeros((self.roi_height, self.roi_width), dtype=np.float32))
        
        # 다중 채널 텐서 생성 (temporal_window, H, W)
        multi_frame_tensor = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # 상대적 레이블 (CSV에서 가져온 값)
        label_tensor = torch.tensor([rel_x, rel_y], dtype=torch.float32)
        
        return multi_frame_tensor, label_tensor


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    csv_labels_path: str,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    훈련/검증 데이터로더 생성
    
    Args:
        individual_frames: 개별 프레임 리스트
        csv_labels_path: CSV 레이블 파일 경로
        train_ratio: 훈련 데이터 비율
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        **kwargs: 데이터셋 생성 파라미터
        
    Returns:
        (train_loader, val_loader): 훈련/검증 데이터로더
    """
    
    print(f"📊 Creating DVS dataset from {len(individual_frames)} individual frames...")
    
    # 전체 데이터셋 생성
    full_dataset = DVSDataset(individual_frames, csv_labels_path, **kwargs)
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"   ✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def load_individual_frames_from_bin(bin_file_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    DVS bin 파일에서 개별 프레임 로딩
    
    Args:
        bin_file_path: bin 파일 경로
        max_frames: 최대 프레임 수 (None이면 모두 로드)
        
    Returns:
        개별 프레임 리스트
    """
    print(f"📖 Loading individual frames from {bin_file_path}")
    
    if not os.path.exists(bin_file_path):
        print(f"❌ Bin file not found: {bin_file_path}")
        print("🔄 Using dummy frames for testing...")
        return _create_dummy_individual_frames(max_frames or 100)
    
    try:
        # lib.bin_processor 사용
        from lib.bin_processor import BinProcessor
        
        # BinProcessor 사용하여 프레임 로드
        processor = BinProcessor(512, 512, has_header=True)
        
        max_frames_limit = max_frames or 200
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames_limit)
        
        print(f"   Loaded {len(frames_data)} frames from bin file")
        
        # 프레임 데이터를 numpy 배열로 변환
        individual_frames = []
        for frame in frames_data:
            frame_array = frame.raw_data.astype(np.float32) / 2.0  # 고정 스케일링
            individual_frames.append(frame_array)
            
            if len(individual_frames) >= max_frames_limit:
                break
        
        print(f"   ✅ Converted {len(individual_frames)} individual frames")
        return individual_frames
        
    except Exception as e:
        print(f"⚠️ Error loading bin file: {e}")
        print("🔄 Falling back to dummy frames...")
        return _create_dummy_individual_frames(max_frames or 100)


def _create_dummy_individual_frames(num_frames: int, center: tuple = (240, 147)) -> List[np.ndarray]:
    """개별 timestamp 기반 더미 프레임 생성"""
    frames = []
    
    print(f"   Creating {num_frames} individual dummy frames centered at {center}")
    
    for i in range(num_frames):
        # 각 프레임은 하나의 timestamp를 나타냄
        frame = np.zeros((128, 128), dtype=np.float32)
        
        # 중심 주변에 가우시안 분포로 이벤트 생성
        events_per_frame = np.random.randint(50, 200)
        
        for _ in range(events_per_frame):
            x = int(np.random.normal(center[0], 10))
            y = int(np.random.normal(center[1], 10))
            
            # 센서 범위 내로 제한
            x = max(0, min(127, x))
            y = max(0, min(127, y))
            
            # 강도 누적
            frame[y, x] += 1.0
        
        # 정규화
        if np.max(frame) > 0:
            frame = frame / np.max(frame)
        
        frames.append(frame)
    
    print(f"   ✅ Generated {len(frames)} individual frames")
    return frames


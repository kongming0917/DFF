#!/usr/bin/env python3
"""
Brownian Motion DVS 데이터를 PyTorch Dataset으로 처리하는 모듈
- CSV 레이블 파일에서 ground truth 로드
- Brownian motion 시계열 데이터 처리
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os
import sys

# 상위 디렉토리를 sys.path에 추가 (lib 모듈 사용을 위해)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)


class DVSBrownianDataset(Dataset):
    """
    Brownian Motion DVS 데이터셋
    - CSV 파일에서 레이블 로드
    - 시계열 temporal window 처리
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        csv_labels_path: str,
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,
        sensor_size: Tuple[int, int] = (720, 960)
    ):
        """
        Args:
            individual_frames: 개별 timestamp 프레임 리스트
            csv_labels_path: CSV 레이블 파일 경로 (frame_idx, shift_x, shift_y, rel_x, rel_y 포함)
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
        required_cols = ['frame_idx', 'shift_x', 'shift_y', 'cnn_rel_x', 'cnn_rel_y']
        missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # frame_idx를 인덱스로 설정
        self.labels_df.set_index('frame_idx', inplace=True)
        
        # 학습/검증 모드
        self.training = True
        
        # 오버래핑 슬라이딩 윈도우로 유효한 샘플 수 계산
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)
        
        print(f"   Individual frames: {len(self.frames)}")
        print(f"   Temporal window: {self.temporal_window}")
        print(f"   Valid samples: {self.valid_samples}")
    
    def set_training_mode(self, training: bool):
        """
        학습/검증 모드 설정
        
        Args:
            training: True면 학습 모드, False면 검증 모드
        """
        self.training = training
    
    # _extract_roi_with_shift 메서드 제거
    # Brownian motion 데이터셋은 이미 augment된 512x512 ROI이므로
    # ROI extraction이 필요 없음
    
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
        
        # 첫 프레임의 shift 사용 (temporal window의 중심 프레임)
        center_frame_idx = frame_indices[len(frame_indices) // 2]
        
        # CSV에서 해당 프레임의 레이블 가져오기
        # Brownian motion 데이터셋은 이미 augment된 512x512 ROI이므로
        # 추가 ROI extraction 불필요, 프레임을 그대로 사용
        if center_frame_idx in self.labels_df.index:
            rel_x = float(self.labels_df.loc[center_frame_idx, 'cnn_rel_x'])
            rel_y = float(self.labels_df.loc[center_frame_idx, 'cnn_rel_y'])
        else:
            # CSV에 없는 경우 (경계 처리)
            rel_x = 0.5
            rel_y = 0.5
        
        # 다중 프레임 사용 (이미 512x512 ROI로 잘려있음)
        multi_frame_rois = []
        for frame_idx in frame_indices:
            if frame_idx < len(self.frames):
                frame = self.frames[frame_idx]
                # 이미 augment된 프레임이므로 그대로 사용 (정규화만 필요하면 수행)
                roi = frame.astype(np.float32)
                # 정규화는 train.py의 load_individual_frames_from_bin에서 이미 수행됨
                multi_frame_rois.append(roi)
        
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
    
    print(f"📊 Creating Brownian dataset from {len(individual_frames)} individual frames...")
    
    # 전체 데이터셋 생성
    train_full_dataset = DVSBrownianDataset(individual_frames, csv_labels_path, **kwargs)
    train_full_dataset.set_training_mode(True)
    
    val_full_dataset = DVSBrownianDataset(individual_frames, csv_labels_path, **kwargs)
    val_full_dataset.set_training_mode(False)
    
    # 데이터셋 크기 확인
    dataset_size = len(train_full_dataset)
    print(f"   Dataset size: {dataset_size} samples")
    
    if dataset_size == 0:
        raise ValueError(f"❌ Dataset is empty! No valid samples from {len(individual_frames)} frames.")
    
    # 분할할 블록 수
    num_blocks = 5
    gap = 50
    block_length = dataset_size // num_blocks
    
    # 검증 블록 인덱스
    val_block_indices = {1, 3}
    train_block_indices = {0, 2, 4}
   
    print(f"   ⚙️ Split Strategy: Blocked Split (Blocks: {num_blocks}, Gap: {gap})")
    print(f"   Validation Blocks: {val_block_indices}") 
    
    train_indices = []
    val_indices = []
    for i in range(num_blocks):
        start_idx = i * block_length
        end_idx = start_idx + block_length
        
        valid_end_idx = end_idx - gap
        
        block_indices = list(range(start_idx, valid_end_idx))
        
        if i in val_block_indices:
            val_indices.extend(block_indices)
        else:
            train_indices.extend(block_indices)
    
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)
        
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


def create_fixed_gt_dataloader(
    individual_frames: List[np.ndarray],
    csv_labels_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    training: bool = True,
    **kwargs
) -> DataLoader:
    """Fixed GT 데이터로더 생성"""
    dataset = DVSBrownianDataset(individual_frames, csv_labels_path, **kwargs)
    dataset.set_training_mode(training)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    # ... (기존 테스트 코드 위에 추가하거나 교체) ...
    
    # [데이터 값 검증용 코드]
    print("\n🔍 Inspecting Data Values...")
    
    # 테스트용 파일 경로 (train.py에 있던 경로 참고)
    BIN_PATH = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512.bin"
    CSV_PATH = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512_labels.csv"
    
    if os.path.exists(BIN_PATH) and os.path.exists(CSV_PATH):
        # 1. 프레임 로드 (train.py의 함수 필요, 여기서는 간단히 직접 로드 시뮬레이션 or import)
        from lib.bin_processor import BinProcessor
        
        try:
            print(f"   Loading first 100 frames from {BIN_PATH}...")
            processor = BinProcessor(512, 512, has_header=True)
            frames_data = processor.read_frames(BIN_PATH, max_frames=100)
            
            # numpy 변환
            individual_frames = [f.raw_data.astype(np.float32) for f in frames_data]
            
            # 2. 데이터셋 생성
            dataset = DVSBrownianDataset(
                individual_frames, 
                CSV_PATH, 
                temporal_window=5
            )
            
            # 3. 첫 번째 샘플 가져와서 값 확인
            sample_tensor, _ = dataset[0]
            
            print("\n📊 Sample Tensor Statistics:")
            print(f"   Shape: {sample_tensor.shape}")
            print(f"   Min Value: {sample_tensor.min().item()}")
            print(f"   Max Value: {sample_tensor.max().item()}")
            print(f"   Mean Value: {sample_tensor.mean().item():.4f}")
            
            # 고유값 확인 (가장 중요: 0, 1, 2 만 나오는지 확인)
            unique_vals = torch.unique(sample_tensor)
            print(f"   ℹ️ Unique Values in tensor: {unique_vals.tolist()}")
            
            if torch.all(torch.isin(unique_vals, torch.tensor([0., 1., 2.]))):
                print("   ✅ Data is correctly loaded as raw discrete values (0, 1, 2).")
            else:
                print("   ⚠️ Warning: Unexpected values found! (Normalization might be applied?)")
                
        except Exception as e:
            print(f"   ❌ Error during inspection: {e}")
    else:
        print("   ⚠️ Data files not found for inspection. Skipping.")
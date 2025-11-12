#!/usr/bin/env python3
"""
Brownian Motion YOLO용 DVS 데이터셋
레이저 스팟을 bounding box로 레이블링 (CSV 레이블 사용)
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


class LaserYOLOBrownianDataset(Dataset):
    """
    Brownian Motion 레이저 스팟 검출용 YOLO 데이터셋
    CSV 파일에서 레이블 로드
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        csv_labels_path: str,
        laser_diameter: int = 400,  # 레이저 스팟 직경 (픽셀)
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,
        sensor_size: Tuple[int, int] = (720, 960)
    ):
        self.frames = individual_frames
        self.laser_diameter = laser_diameter
        self.roi_height, self.roi_width = roi_size
        self.temporal_window = temporal_window
        self.sensor_height, self.sensor_width = sensor_size
        
        # CSV 레이블 로드
        if not os.path.exists(csv_labels_path):
            raise FileNotFoundError(f"CSV labels file not found: {csv_labels_path}")
        
        self.labels_df = pd.read_csv(csv_labels_path)
        print(f"   Loaded {len(self.labels_df)} labels from {csv_labels_path}")
        
        # 레이블 검증 (YOLO는 cnn_rel_x/y를 사용하거나, yolo_bbox_x/y를 사용)
        # 현재 CSV는 cnn_rel_x/y와 yolo_bbox_x/y 모두 포함
        required_cols = ['frame_idx', 'shift_x', 'shift_y']
        missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # 중심점 좌표는 cnn_rel_x/y 또는 yolo_bbox_x/y 사용
        if 'yolo_bbox_x' in self.labels_df.columns and 'yolo_bbox_y' in self.labels_df.columns:
            self.use_yolo_bbox = True
        elif 'cnn_rel_x' in self.labels_df.columns and 'cnn_rel_y' in self.labels_df.columns:
            self.use_yolo_bbox = False
        else:
            raise ValueError("CSV must contain either (cnn_rel_x, cnn_rel_y) or (yolo_bbox_x, yolo_bbox_y)")
        
        # frame_idx를 인덱스로 설정
        self.labels_df.set_index('frame_idx', inplace=True)
        
        self.training = True
        self.validation_seed = 42
        
        # 유효한 샘플 수 (슬라이딩 윈도우)
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)
        
        print(f"   Frames: {len(self.frames)}, Valid samples: {self.valid_samples}")
    
    def set_training_mode(self, training: bool):
        """학습/검증 모드 설정"""
        self.training = training
    
    # _extract_roi_with_shift 메서드 제거
    # Brownian motion 데이터셋은 이미 augment된 512x512 ROI이므로
    # ROI extraction이 필요 없음
    
    def _create_bbox_label(self, rel_x: float, rel_y: float) -> Tuple[float, float, float, float]:
        """
        Bounding box 레이블 생성 (YOLO 형식)
        
        Args:
            rel_x, rel_y: CSV에서 가져온 정규화된 중심점 좌표 (0-1)
        
        Returns:
            (x_center, y_center, width, height) - ROI 내 정규화된 좌표 (0-1)
        """
        # CSV에서 가져온 중심점 사용
        x_center = rel_x
        y_center = rel_y
        
        # Bbox 크기 (레이저 직경 기준)
        width = self.laser_diameter / self.roi_width
        height = self.laser_diameter / self.roi_height
        
        # 범위 제한
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        
        return x_center, y_center, width, height
    
    def __len__(self) -> int:
        return self.valid_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [temporal_window, H, W] 다중 프레임
            target: [5] (x_center, y_center, width, height, class_id)
        """
        # 프레임 선택
        frame_indices = list(range(idx, idx + self.temporal_window))
        
        # 첫 프레임의 shift 사용 (temporal window의 중심 프레임)
        center_frame_idx = frame_indices[len(frame_indices) // 2]
        
        # CSV에서 해당 프레임의 레이블 가져오기
        # Brownian motion 데이터셋은 이미 augment된 512x512 ROI이므로
        # 추가 ROI extraction 불필요, 프레임을 그대로 사용
        if center_frame_idx in self.labels_df.index:
            if self.use_yolo_bbox:
                rel_x = float(self.labels_df.loc[center_frame_idx, 'yolo_bbox_x'])
                rel_y = float(self.labels_df.loc[center_frame_idx, 'yolo_bbox_y'])
            else:
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
                # 정규화는 load_frames_from_bin에서 이미 수행됨
                multi_frame_rois.append(roi)
        
        # 텐서 생성
        image = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # Bbox 레이블 생성 (CSV에서 가져온 rel_x, rel_y 사용)
        x_center, y_center, width, height = self._create_bbox_label(rel_x, rel_y)
        target = torch.tensor([x_center, y_center, width, height, 0.0], dtype=torch.float32)  # class_id=0
        
        return image, target


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    csv_labels_path: str,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """훈련/검증 데이터로더 생성"""
    
    print(f"📊 Creating Brownian YOLO dataset from {len(individual_frames)} frames...")
    
    # 전체 데이터셋
    full_dataset = LaserYOLOBrownianDataset(individual_frames, csv_labels_path, **kwargs)
    dataset_size = len(full_dataset)
    
    if dataset_size == 0:
        raise ValueError(f"❌ Dataset is empty!")
    
    # Train/Val 분할
    train_size = max(1, int(train_ratio * dataset_size))
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 학습/검증 모드 설정
    if hasattr(train_dataset, 'dataset'):
        train_dataset.dataset.set_training_mode(True)
    
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.set_training_mode(False)
    
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
    
    print(f"   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


def load_frames_from_bin(bin_file_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """DVS bin 파일에서 프레임 로딩"""
    print(f"📖 Loading frames from {bin_file_path}")
    
    try:
        from lib.bin_processor import BinProcessor
        
        # Brownian motion 데이터셋: 512x512
        processor = BinProcessor(512, 512, has_header=True)
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames or 200)
        
        individual_frames = []
        for frame in frames_data:
            frame_array = frame.raw_data.astype(np.float32)
            if np.max(frame_array) > 0:
                frame_array = frame_array / np.max(frame_array)
            individual_frames.append(frame_array)
        
        print(f"   ✅ Loaded {len(individual_frames)} frames")
        return individual_frames
        
    except Exception as e:
        print(f"⚠️ Error loading bin file: {e}")
        return []


if __name__ == "__main__":
    print("🧪 Testing Brownian YOLO Dataset")
    print("=" * 50)
    
    # 더미 프레임 생성
    dummy_frames = [np.random.rand(720, 960).astype(np.float32) for _ in range(100)]
    
    # 더미 CSV 생성 (테스트용)
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'shift_x', 'shift_y', 'rel_x', 'rel_y'])
        for i in range(100):
            writer.writerow([i, 0, 0, 0.5, 0.5])
        csv_path = f.name
    
    # 데이터셋 생성
    dataset = LaserYOLOBrownianDataset(
        dummy_frames,
        csv_labels_path=csv_path,
        laser_diameter=400,
        roi_size=(512, 512),
        temporal_window=5,
        shift_range_x=50,
        shift_range_y=50
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # 샘플 확인
    image, target = dataset[0]
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Target: {target}")
    print(f"   - Center: ({target[0]:.3f}, {target[1]:.3f})")
    print(f"   - Size: ({target[2]:.3f}, {target[3]:.3f})")
    print(f"   - Class: {target[4]:.0f}")
    
    # 임시 파일 삭제
    os.unlink(csv_path)
    
    print("\n✅ Dataset test completed!")

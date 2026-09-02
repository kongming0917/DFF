#!/usr/bin/env python3
"""
YOLO용 DVS 데이터셋
레이저 스팟을 bounding box로 레이블링
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import sys

# yolo_sim/dataset.py 상단에 추가
import os

# 상위 디렉토리를 sys.path에 추가 (lib 모듈 사용을 위해)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)


class LaserYOLODataset(Dataset):
    """
    레이저 스팟 검출용 YOLO 데이터셋
    
    고정된 레이저 위치에서 데이터 증강을 통해 다양한 위치 생성
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        true_center_coord: Tuple[int, int] = (541, 361),
        laser_diameter: int = 400,  # 레이저 스팟 직경 (픽셀)
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,
        shift_range_x: int = 50,
        shift_range_y: int = 50,
        sensor_size: Tuple[int, int] = (720, 960)
    ):
        self.frames = individual_frames
        self.true_center_x, self.true_center_y = true_center_coord
        self.laser_diameter = laser_diameter
        self.roi_height, self.roi_width = roi_size
        self.temporal_window = temporal_window
        self.shift_range_x = shift_range_x
        self.shift_range_y = shift_range_y
        self.sensor_height, self.sensor_width = sensor_size
        
        self.training = True
        self.validation_seed = 42
        
        # 유효한 샘플 수 (슬라이딩 윈도우)
        self.valid_samples = max(0, len(self.frames) - self.temporal_window + 1)
        
        print(f"   Frames: {len(self.frames)}, Valid samples: {self.valid_samples}")
    
    def set_training_mode(self, training: bool):
        """학습/검증 모드 설정"""
        self.training = training
    
    def _extract_roi_with_shift(self, frame: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """ROI 추출"""
        half_h, half_w = self.roi_height // 2, self.roi_width // 2
        
        # ROI 중심 계산
        center_x = self.true_center_x + shift_x
        center_y = self.true_center_y + shift_y
        
        # ROI 경계
        y_start = max(0, center_y - half_h)
        y_end = min(self.sensor_height, center_y + half_h)
        x_start = max(0, center_x - half_w)
        x_end = min(self.sensor_width, center_x + half_w)
        
        # ROI 추출
        roi = np.zeros((self.roi_height, self.roi_width), dtype=np.float32)
        
        src_h = y_end - y_start
        src_w = x_end - x_start
        dst_y = max(0, half_h - (center_y - y_start))
        dst_x = max(0, half_w - (center_x - x_start))
        
        copy_h = min(src_h, self.roi_height - dst_y)
        copy_w = min(src_w, self.roi_width - dst_x)
        
        if copy_h > 0 and copy_w > 0:
            roi[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                frame[y_start:y_start + copy_h, x_start:x_start + copy_w]
        
        # 정규화는 load_frames_from_bin에서 이미 적용됨 (이중 정규화 방지)
        return roi
    
    def _create_bbox_label(self, shift_x: int, shift_y: int) -> Tuple[float, float, float, float]:
        """
        Bounding box 레이블 생성 (YOLO 형식)
        
        Returns:
            (x_center, y_center, width, height) - ROI 내 정규화된 좌표 (0-1)
        """
        # ROI 내에서 레이저 중심 위치
        x_center = 0.5 - (shift_x / self.roi_width)
        y_center = 0.5 - (shift_y / self.roi_height)
        
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
        
        # Shift 결정
        if self.training:
            # Training: 랜덤 shift
            shift_x = np.random.randint(-self.shift_range_x, self.shift_range_x + 1)
            shift_y = np.random.randint(-self.shift_range_y, self.shift_range_y + 1)
        else:
            shift_x = 0
            shift_y = 0
        
        # ROI 추출
        multi_frame_rois = []
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            roi = self._extract_roi_with_shift(frame, shift_x, shift_y)
            multi_frame_rois.append(roi)
        
        # 텐서 생성
        image = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # Bbox 레이블 생성
        x_center, y_center, width, height = self._create_bbox_label(shift_x, shift_y)
        target = torch.tensor([x_center, y_center, width, height, 0.0], dtype=torch.float32)  # class_id=0
        
        return image, target


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """훈련/검증 데이터로더 생성"""
    
    print(f"📊 Creating YOLO dataset from {len(individual_frames)} frames...")
    
    # 전체 데이터셋
    full_dataset = LaserYOLODataset(individual_frames, **kwargs)
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
        
        processor = BinProcessor(960, 720, has_header=True)
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
    print("🧪 Testing YOLO Dataset")
    print("=" * 50)
    
    # 더미 프레임 생성
    dummy_frames = [np.random.rand(720, 960).astype(np.float32) for _ in range(100)]
    
    # 데이터셋 생성
    dataset = LaserYOLODataset(
        dummy_frames,
        true_center_coord=(541, 361),
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
    
    print("\n✅ Dataset test completed!")

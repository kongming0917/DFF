#!/usr/bin/env python3
"""
YOLO Tracking 데이터셋
- ROI 고정, 물체가 움직이며 bounding box tracking
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import sys


class YOLOTrackingDataset(Dataset):
    """
    YOLO 기반 레이저 추적 데이터셋
    
    cnn_tracking과 유사하지만:
    - 레이블이 bounding box 형식 (x, y, w, h, class)
    - 물체 크기 정보 포함
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        roi_center: Tuple[int, int] = (541, 360),
        roi_size: Tuple[int, int] = (512, 512),
        laser_diameter: int = 400,
        motion_std: float = 2.0,
        num_temporal_frames: int = 5,
        sensor_size: Tuple[int, int] = (720, 960),
        motion_boundary_margin: int = 80,
        use_boundary_reflection: bool = True,
        random_seed: int = 42
    ):
        self.frames = individual_frames
        self.roi_center_x, self.roi_center_y = roi_center
        self.roi_height, self.roi_width = roi_size
        self.laser_diameter = laser_diameter
        self.motion_std = motion_std
        self.num_temporal_frames = num_temporal_frames
        self.sensor_height, self.sensor_width = sensor_size
        self.motion_boundary_margin = motion_boundary_margin
        self.use_boundary_reflection = use_boundary_reflection
        
        # ROI 경계
        self.roi_min_x = self.roi_center_x - self.roi_width // 2 + motion_boundary_margin
        self.roi_max_x = self.roi_center_x + self.roi_width // 2 - motion_boundary_margin
        self.roi_min_y = self.roi_center_y - self.roi_height // 2 + motion_boundary_margin
        self.roi_max_y = self.roi_center_y + self.roi_height // 2 - motion_boundary_margin
        
        # Trajectory 생성
        np.random.seed(random_seed)
        self.gt_trajectory = self._generate_bounded_trajectory(
            start_coord=roi_center,
            num_steps=len(self.frames)
        )
        
        self.valid_samples = max(0, len(self.frames) - self.num_temporal_frames + 1)
        
        print(f"   Frames: {len(self.frames)}")
        print(f"   ROI: {roi_center}, size: {roi_size}")
        print(f"   Laser diameter: {laser_diameter}px")
        print(f"   Valid samples: {self.valid_samples}")
    
    def _generate_bounded_trajectory(
        self, 
        start_coord: Tuple[int, int], 
        num_steps: int
    ) -> List[Tuple[float, float]]:
        """Brownian motion trajectory (cnn_tracking과 동일)"""
        trajectory = [start_coord]
        
        for _ in range(1, num_steps):
            prev_x, prev_y = trajectory[-1]
            delta_x = np.random.normal(0, self.motion_std)
            delta_y = np.random.normal(0, self.motion_std)
            next_x = prev_x + delta_x
            next_y = prev_y + delta_y
            
            if self.use_boundary_reflection:
                if next_x < self.roi_min_x:
                    next_x = 2 * self.roi_min_x - next_x
                    if next_x > self.roi_max_x:
                        next_x = self.roi_max_x
                elif next_x > self.roi_max_x:
                    next_x = 2 * self.roi_max_x - next_x
                    if next_x < self.roi_min_x:
                        next_x = self.roi_min_x
                
                if next_y < self.roi_min_y:
                    next_y = 2 * self.roi_min_y - next_y
                    if next_y > self.roi_max_y:
                        next_y = self.roi_max_y
                elif next_y > self.roi_max_y:
                    next_y = 2 * self.roi_max_y - next_y
                    if next_y < self.roi_min_y:
                        next_y = self.roi_min_y
            else:
                next_x = np.clip(next_x, self.roi_min_x, self.roi_max_x)
                next_y = np.clip(next_y, self.roi_min_y, self.roi_max_y)
            
            trajectory.append((next_x, next_y))
        
        return trajectory
    
    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """고정된 ROI 추출"""
        half_h = self.roi_height // 2
        half_w = self.roi_width // 2
        
        y_start = max(0, self.roi_center_y - half_h)
        y_end = min(self.sensor_height, self.roi_center_y + half_h)
        x_start = max(0, self.roi_center_x - half_w)
        x_end = min(self.sensor_width, self.roi_center_x + half_w)
        
        roi = np.zeros((self.roi_height, self.roi_width), dtype=np.float32)
        
        src_h = y_end - y_start
        src_w = x_end - x_start
        dst_y = max(0, half_h - (self.roi_center_y - y_start))
        dst_x = max(0, half_w - (self.roi_center_x - x_start))
        
        copy_h = min(src_h, self.roi_height - dst_y)
        copy_w = min(src_w, self.roi_width - dst_x)
        
        if copy_h > 0 and copy_w > 0:
            roi[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                frame[y_start:y_start + copy_h, x_start:x_start + copy_w]
        
        return roi
    
    def _gt_to_yolo_label(self, gt_x: float, gt_y: float) -> Tuple[float, float, float, float, float]:
        """
        GT 좌표를 YOLO bbox 형식으로 변환
        
        Returns:
            (x_center, y_center, width, height, class_id) - ROI 내 정규화 (0-1)
        """
        # 중심 좌표 (0-1 정규화)
        rel_x = (gt_x - self.roi_center_x) / self.roi_width + 0.5
        rel_y = (gt_y - self.roi_center_y) / self.roi_height + 0.5
        rel_x = np.clip(rel_x, 0.0, 1.0)
        rel_y = np.clip(rel_y, 0.0, 1.0)
        
        # Bbox 크기 (정규화)
        width = self.laser_diameter / self.roi_width
        height = self.laser_diameter / self.roi_height
        
        class_id = 0.0  # 레이저 스팟
        
        return rel_x, rel_y, width, height, class_id
    
    def __len__(self) -> int:
        return self.valid_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [num_temporal_frames, H, W]
            target: [5] (x_center, y_center, width, height, class_id)
        """
        if idx >= self.valid_samples:
            raise IndexError(f"Index {idx} out of range")
        
        frame_indices = list(range(idx, idx + self.num_temporal_frames))
        
        # 고정된 ROI에서 다중 프레임 추출
        multi_frame_rois = []
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            roi = self._extract_roi(frame)
            multi_frame_rois.append(roi)
        
        image_tensor = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # 마지막 프레임의 GT를 YOLO bbox로 변환
        last_gt_x, last_gt_y = self.gt_trajectory[frame_indices[-1]]
        x_center, y_center, width, height, class_id = self._gt_to_yolo_label(last_gt_x, last_gt_y)
        target_tensor = torch.tensor([x_center, y_center, width, height, class_id], dtype=torch.float32)
        
        return image_tensor, target_tensor


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    config,
    train_ratio: float = 0.8,
    batch_size: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """훈련/검증 데이터로더 생성"""
    
    print(f"📊 Creating YOLO tracking dataset from {len(individual_frames)} frames...")
    
    full_dataset = YOLOTrackingDataset(
        individual_frames=individual_frames,
        roi_center=config.roi_center,
        roi_size=config.roi_size,
        laser_diameter=config.laser_diameter,
        motion_std=config.motion_std,
        num_temporal_frames=config.num_temporal_frames,
        sensor_size=config.sensor_size,
        motion_boundary_margin=config.motion_boundary_margin,
        use_boundary_reflection=config.use_boundary_reflection,
        random_seed=config.random_seed
    )
    
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError("❌ Dataset is empty!")
    
    train_size = max(1, int(train_ratio * dataset_size))
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
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
    
    print(f"   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


def load_frames_from_bin(bin_file_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """DVS bin 파일에서 프레임 로딩 (cnn_tracking과 동일)"""
    print(f"📖 Loading frames from {bin_file_path}")
    
    try:
        from lib.bin_processor import BinProcessor
        
        processor = BinProcessor(960, 720, has_header=True)
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames or 500)
        
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
        print(f"🔄 Creating dummy frames...")
        return _create_dummy_frames(max_frames or 100)


def _create_dummy_frames(num_frames: int, center: Tuple[int, int] = (541, 360)) -> List[np.ndarray]:
    """테스트용 더미 프레임"""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((720, 960), dtype=np.float32)
        events_per_frame = np.random.randint(50, 200)
        for _ in range(events_per_frame):
            x = int(np.random.normal(center[0], 25))
            y = int(np.random.normal(center[1], 25))
            x = max(0, min(959, x))
            y = max(0, min(719, y))
            frame[y, x] += 1.0
        if np.max(frame) > 0:
            frame = frame / np.max(frame)
        frames.append(frame)
    print(f"   ✅ Created {len(frames)} dummy frames")
    return frames


if __name__ == "__main__":
    print("🧪 Testing YOLO Tracking Dataset")
    print("=" * 50)
    
    dummy_frames = _create_dummy_frames(100)
    
    dataset = YOLOTrackingDataset(
        individual_frames=dummy_frames,
        roi_center=(541, 360),
        roi_size=(512, 512),
        laser_diameter=400,
        motion_std=2.0,
        num_temporal_frames=5
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    image, target = dataset[0]
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Target: (x={target[0]:.3f}, y={target[1]:.3f}, w={target[2]:.3f}, h={target[3]:.3f}, class={target[4]:.0f})")
    
    print("\n✅ YOLO Tracking dataset test completed!")


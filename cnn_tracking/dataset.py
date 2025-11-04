#!/usr/bin/env python3
"""
DVS Laser Tracking 데이터셋
- ROI 고정, 물체가 ROI 내에서 움직임
- Brownian motion으로 궤적 생성
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import sys


class LaserTrackingDataset(Dataset):
    """
    레이저 추적 데이터셋
    
    핵심 특징:
    - ROI는 고정된 위치
    - 레이저가 Brownian motion으로 ROI 내에서 움직임
    - 모델은 시간적 패턴을 보고 현재 위치 예측
    """
    
    def __init__(
        self,
        individual_frames: List[np.ndarray],
        roi_center: Tuple[int, int] = (480, 294),
        roi_size: Tuple[int, int] = (384, 384),
        motion_std: float = 2.0,
        num_temporal_frames: int = 5,
        sensor_size: Tuple[int, int] = (720, 960),
        motion_boundary_margin: int = 80,
        use_boundary_reflection: bool = True,
        random_seed: int = 42
    ):
        """
        Args:
            individual_frames: 개별 프레임 리스트
            roi_center: ROI 중심 좌표 (x, y) - 고정됨!
            roi_size: ROI 크기 (height, width)
            motion_std: Brownian motion 표준편차
            num_temporal_frames: 시간 축 프레임 수
            sensor_size: 센서 크기 (height, width)
            motion_boundary_margin: ROI 경계로부터 여유 공간
            use_boundary_reflection: 경계에서 반사 여부
            random_seed: 재현성을 위한 시드
        """
        self.frames = individual_frames
        self.roi_center_x, self.roi_center_y = roi_center
        self.roi_height, self.roi_width = roi_size
        self.motion_std = motion_std
        self.num_temporal_frames = num_temporal_frames
        self.sensor_height, self.sensor_width = sensor_size
        self.motion_boundary_margin = motion_boundary_margin
        self.use_boundary_reflection = use_boundary_reflection
        
        # ROI 경계 계산 (물체가 움직일 수 있는 범위)
        self.roi_min_x = self.roi_center_x - self.roi_width // 2 + motion_boundary_margin
        self.roi_max_x = self.roi_center_x + self.roi_width // 2 - motion_boundary_margin
        self.roi_min_y = self.roi_center_y - self.roi_height // 2 + motion_boundary_margin
        self.roi_max_y = self.roi_center_y + self.roi_height // 2 - motion_boundary_margin
        
        # Brownian motion 궤적 생성 (ROI 내부로 제한)
        np.random.seed(random_seed)
        self.gt_trajectory = self._generate_bounded_trajectory(
            start_coord=roi_center,
            num_steps=len(self.frames)
        )
        
        # 유효한 샘플 수 (슬라이딩 윈도우)
        self.valid_samples = max(0, len(self.frames) - self.num_temporal_frames + 1)
        
        print(f"   Frames: {len(self.frames)}")
        print(f"   ROI center: {roi_center}, size: {roi_size}")
        print(f"   Motion range: X[{self.roi_min_x}, {self.roi_max_x}], Y[{self.roi_min_y}, {self.roi_max_y}]")
        print(f"   Valid samples: {self.valid_samples}")
    
    def _generate_bounded_trajectory(
        self, 
        start_coord: Tuple[int, int], 
        num_steps: int
    ) -> List[Tuple[float, float]]:
        """
        ROI 경계 내로 제한된 Brownian motion 궤적 생성
        
        경계 처리:
        - use_boundary_reflection=True: 경계에서 반사
        - use_boundary_reflection=False: 경계에 붙음
        """
        trajectory = [start_coord]
        
        for _ in range(1, num_steps):
            prev_x, prev_y = trajectory[-1]
            
            # Brownian motion 스텝
            delta_x = np.random.normal(0, self.motion_std)
            delta_y = np.random.normal(0, self.motion_std)
            
            next_x = prev_x + delta_x
            next_y = prev_y + delta_y
            
            # 경계 처리
            if self.use_boundary_reflection:
                # 경계에서 반사
                if next_x < self.roi_min_x:
                    next_x = 2 * self.roi_min_x - next_x
                    if next_x > self.roi_max_x:  # 너무 크게 반사되면 경계에
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
                # 경계에 붙음
                next_x = np.clip(next_x, self.roi_min_x, self.roi_max_x)
                next_y = np.clip(next_y, self.roi_min_y, self.roi_max_y)
            
            trajectory.append((next_x, next_y))
        
        return trajectory
    
    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        고정된 ROI 중심에서 ROI 추출
        """
        half_h = self.roi_height // 2
        half_w = self.roi_width // 2
        
        # ROI 경계
        y_start = max(0, self.roi_center_y - half_h)
        y_end = min(self.sensor_height, self.roi_center_y + half_h)
        x_start = max(0, self.roi_center_x - half_w)
        x_end = min(self.sensor_width, self.roi_center_x + half_w)
        
        # ROI 추출
        roi = np.zeros((self.roi_height, self.roi_width), dtype=np.float32)
        
        # 복사할 영역 계산
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
    
    def _gt_to_roi_relative(self, gt_x: float, gt_y: float) -> Tuple[float, float]:
        """
        GT 좌표를 ROI 내 상대 좌표로 변환 (0-1 정규화)
        
        Args:
            gt_x, gt_y: 센서 좌표계의 GT 위치
            
        Returns:
            (rel_x, rel_y): ROI 내 상대 위치 (0-1 범위)
        """
        # ROI 중심을 기준으로 상대 위치 계산
        rel_x = (gt_x - self.roi_center_x) / self.roi_width + 0.5
        rel_y = (gt_y - self.roi_center_y) / self.roi_height + 0.5
        
        # 0-1 범위로 제한
        rel_x = np.clip(rel_x, 0.0, 1.0)
        rel_y = np.clip(rel_y, 0.0, 1.0)
        
        return rel_x, rel_y
    
    def __len__(self) -> int:
        return self.valid_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [num_temporal_frames, H, W] 다중 프레임 텐서
            label: [2] (x, y) 마지막 프레임의 레이저 위치 (ROI 내 상대 좌표)
        """
        if idx >= self.valid_samples:
            raise IndexError(f"Index {idx} out of range")
        
        # 슬라이딩 윈도우로 프레임 선택
        frame_indices = list(range(idx, idx + self.num_temporal_frames))
        
        # 고정된 ROI에서 다중 프레임 추출
        multi_frame_rois = []
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            roi = self._extract_roi(frame)
            multi_frame_rois.append(roi)
        
        # 텐서로 변환 (num_temporal_frames, H, W)
        image_tensor = torch.from_numpy(np.stack(multi_frame_rois)).float()
        
        # 레이블: 마지막 프레임의 GT 위치를 ROI 내 상대 좌표로 변환
        last_gt_x, last_gt_y = self.gt_trajectory[frame_indices[-1]]
        rel_x, rel_y = self._gt_to_roi_relative(last_gt_x, last_gt_y)
        label_tensor = torch.tensor([rel_x, rel_y], dtype=torch.float32)
        
        return image_tensor, label_tensor


def create_train_val_loaders(
    individual_frames: List[np.ndarray],
    config,  # TrackingDataConfig
    train_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """훈련/검증 데이터로더 생성"""
    
    print(f"📊 Creating tracking dataset from {len(individual_frames)} frames...")
    
    # 전체 데이터셋 생성
    full_dataset = LaserTrackingDataset(
        individual_frames=individual_frames,
        roi_center=config.roi_center,
        roi_size=config.roi_size,
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
    
    # Train/Val 분할
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
    """DVS bin 파일에서 프레임 로딩"""
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
        print(f"🔄 Creating dummy frames for testing...")
        return _create_dummy_frames(max_frames or 100)


def _create_dummy_frames(num_frames: int, center: Tuple[int, int] = (480, 294)) -> List[np.ndarray]:
    """테스트용 더미 프레임 생성"""
    frames = []
    
    for i in range(num_frames):
        frame = np.zeros((720, 960), dtype=np.float32)
        
        # 가우시안 스팟 생성 (중심 주변)
        events_per_frame = np.random.randint(50, 200)
        for _ in range(events_per_frame):
            x = int(np.random.normal(center[0], 25))
            y = int(np.random.normal(center[1], 25))
            x = max(0, min(959, x))
            y = max(0, min(719, y))
            frame[y, x] += 1.0
        
        # 정규화
        if np.max(frame) > 0:
            frame = frame / np.max(frame)
        
        frames.append(frame)
    
    print(f"   ✅ Created {len(frames)} dummy frames")
    return frames


if __name__ == "__main__":
    print("🧪 Testing Laser Tracking Dataset")
    print("=" * 50)
    
    # 더미 프레임 생성
    dummy_frames = _create_dummy_frames(100)
    
    # 데이터셋 생성
    dataset = LaserTrackingDataset(
        individual_frames=dummy_frames,
        roi_center=(480, 294),
        roi_size=(384, 384),
        motion_std=2.0,
        num_temporal_frames=5,
        motion_boundary_margin=80
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # 샘플 확인
    image, label = dataset[0]
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Label: ({label[0]:.3f}, {label[1]:.3f})")
    
    # Trajectory 시각화 (처음 20개)
    print(f"\n🎯 First 20 trajectory points:")
    for i in range(min(20, len(dataset.gt_trajectory))):
        x, y = dataset.gt_trajectory[i]
        rel_x, rel_y = dataset._gt_to_roi_relative(x, y)
        print(f"   Frame {i}: GT=({x:.1f}, {y:.1f}) → ROI_rel=({rel_x:.3f}, {rel_y:.3f})")
    
    print("\n✅ Dataset test completed!")


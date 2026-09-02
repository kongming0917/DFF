#!/usr/bin/env python3
"""
Brownian Motion 기반 DVS 데이터셋 생성 스크립트

기존 gaussian_large.bin을 읽어서:
1. Gaussian 분포를 따르는 Brownian motion으로 레이저 중심점 이동
2. 프레임별로 temporal하게 shift 적용

모드:
- 기본 모드: ROI crop (512x512)
- -org 옵션: 원본 resolution 유지 (720x960, Zero padding)
"""

import numpy as np
import struct
import os
import json
import csv
from typing import Tuple, List, Optional
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 공통 라이브러리 사용
import sys
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sim/ (tools/의 상위)
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from dvslib.data import BinProcessor, DVSFrame, FrameHeader


@dataclass
class BrownianMotionConfig:
    """Brownian motion 파라미터 설정"""
    # 초기 중심점 (원본 GT)
    initial_center_x: int = 541
    initial_center_y: int = 361
    
    # ROI 크기
    roi_size: Tuple[int, int] = (512, 512)
    
    # Brownian motion 파라미터
    sigma_x: float = 2.0  # X축 shift의 표준편차 (픽셀)
    sigma_y: float = 2.0  # Y축 shift의 표준편차 (픽셀)
    
    # Shift 범위 제한
    max_shift_range: int = 50  # 최대 shift 범위 (픽셀)
    
    # 시드 (재현성)
    random_seed: Optional[int] = 42


class BrownianShiftGenerator:
    """Brownian motion 기반 shift 생성기"""
    
    def __init__(self, config: BrownianMotionConfig, keep_original_resolution: bool = False):
        self.config = config
        self.keep_original_resolution = keep_original_resolution
        
        # 시드 설정
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # 초기 shift (0, 0)
        self.current_shift_x = 0.0
        self.current_shift_y = 0.0
        
        # 궤적 저장 (디버깅/시각화용)
        if keep_original_resolution:
            # 원본 좌표계에서의 레이저 위치
            self.trajectory = [(config.initial_center_x, config.initial_center_y)]
        else:
            # ROI 내부에서의 레이저 위치
            roi_center_x = config.roi_size[0] // 2
            roi_center_y = config.roi_size[1] // 2
            self.trajectory = [(roi_center_x, roi_center_y)]
    
    def step(self) -> Tuple[int, int]:
        """
        Brownian motion 한 스텝 진행하여 shift 생성
        
        Returns:
            (shift_x, shift_y): ROI 추출 시 적용할 shift 값 (픽셀)
        """
        # Gaussian 분포로 이동량 생성
        dx = np.random.normal(0, self.config.sigma_x)
        dy = np.random.normal(0, self.config.sigma_y)
        
        # 새로운 shift 계산 (누적)
        new_shift_x = self.current_shift_x + dx
        new_shift_y = self.current_shift_y + dy
        
        # 최대 shift 범위로 클리핑
        new_shift_x = np.clip(new_shift_x, -self.config.max_shift_range, self.config.max_shift_range)
        new_shift_y = np.clip(new_shift_y, -self.config.max_shift_range, self.config.max_shift_range)
        
        # 상태 업데이트
        self.current_shift_x = new_shift_x
        self.current_shift_y = new_shift_y
        
        # 정수로 변환
        shift_x = int(round(new_shift_x))
        shift_y = int(round(new_shift_y))
        
        # 레이저 위치 계산 (시각화용)
        if self.keep_original_resolution:
            # 원본 좌표계에서의 레이저 위치
            laser_x = self.config.initial_center_x + shift_x
            laser_y = self.config.initial_center_y + shift_y
        else:
            # ROI 내부에서의 레이저 위치
            roi_center_x = self.config.roi_size[0] // 2
            roi_center_y = self.config.roi_size[1] // 2
            laser_x = roi_center_x + shift_x
            laser_y = roi_center_y + shift_y
        
        # 궤적 저장
        self.trajectory.append((laser_x, laser_y))
        
        return shift_x, shift_y
    
    def get_trajectory(self) -> np.ndarray:
        """원본 좌표계에서의 레이저 위치 궤적 반환"""
        return np.array(self.trajectory)
    
    def reset(self):
        """초기 상태로 리셋"""
        self.current_shift_x = 0.0
        self.current_shift_y = 0.0
        if self.keep_original_resolution:
            self.trajectory = [(self.config.initial_center_x, self.config.initial_center_y)]
        else:
            roi_center_x = self.config.roi_size[0] // 2
            roi_center_y = self.config.roi_size[1] // 2
        self.trajectory = [(roi_center_x, roi_center_y)]
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)


def extract_roi_from_frame_with_shift(
    frame: np.ndarray,
    true_center_x: int,
    true_center_y: int,
    roi_size: Tuple[int, int],
    shift_x: int = 0,
    shift_y: int = 0,
    sensor_size: Tuple[int, int] = (720, 960)
) -> np.ndarray:
    """
    원본 프레임에서 ROI 추출 (Shift 적용, CNN 방식)
    
    CNN의 random shift 방식과 동일:
    - shift_x > 0: ROI를 오른쪽으로 이동해서 추출 → 레이저는 ROI 내부에서 왼쪽에 위치
    - shift_x < 0: ROI를 왼쪽으로 이동해서 추출 → 레이저는 ROI 내부에서 오른쪽에 위치
    
    Args:
        frame: 원본 프레임 (sensor_height, sensor_width)
        true_center_x: 레이저 광원의 실제 중심 X 좌표 (원본 좌표계)
        true_center_y: 레이저 광원의 실제 중심 Y 좌표 (원본 좌표계)
        roi_size: ROI 크기 (width, height)
        shift_x: X축 시프트 (픽셀, 반대 방향으로 적용)
        shift_y: Y축 시프트 (픽셀, 반대 방향으로 적용)
        sensor_size: 센서 크기 (height, width)
    
    Returns:
        ROI 프레임 (roi_height, roi_width)
    """
    roi_width, roi_height = roi_size
    sensor_height, sensor_width = sensor_size
    
    # ROI 추출 중심 좌표 계산 (shift 반대 방향으로 적용)
    # shift가 +이면 ROI를 오른쪽으로 이동해서 추출 → 레이저는 ROI 내부에서 왼쪽에 위치
    center_x = true_center_x - shift_x
    center_y = true_center_y - shift_y
    
    # ROI 경계 계산
    half_w = roi_width // 2
    half_h = roi_height // 2
    
    x_start = max(0, center_x - half_w)
    x_end = min(sensor_width, center_x + half_w)
    y_start = max(0, center_y - half_h)
    y_end = min(sensor_height, center_y + half_h)
    
    # ROI 추출
    roi = np.zeros((roi_height, roi_width), dtype=frame.dtype)
    
    # 복사할 영역 계산 (경계 처리)
    src_h = y_end - y_start
    src_w = x_end - x_start
    dst_y = max(0, half_h - (center_y - y_start))
    dst_x = max(0, half_w - (center_x - x_start))
    
    copy_h = min(src_h, roi_height - dst_y)
    copy_w = min(src_w, roi_width - dst_x)
    
    if copy_h > 0 and copy_w > 0:
        roi[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
            frame[y_start:y_start + copy_h, x_start:x_start + copy_w]
    
    return roi


def apply_shift_with_padding(
    frame: np.ndarray,
    shift_x: int,
    shift_y: int,
    padding_mode: str = 'zero'
) -> np.ndarray:
    """
    원본 resolution을 유지하면서 shift 적용 (Zero padding)
    
    Args:
        frame: 원본 프레임 (sensor_height, sensor_width)
        shift_x: X축 shift (픽셀, 양수=레이저가 오른쪽으로 이동)
        shift_y: Y축 shift (픽셀, 양수=레이저가 아래로 이동)
        padding_mode: 'zero' (Zero padding 사용)
    
    Returns:
        Shift된 프레임 (원본과 동일한 크기)
    """
    sensor_height, sensor_width = frame.shape
    result = np.zeros_like(frame)
    
    # 원본에서 복사할 영역 계산
    # shift_x > 0: 레이저가 오른쪽으로 이동 → 프레임을 왼쪽으로 shift
    # shift_y > 0: 레이저가 아래로 이동 → 프레임을 위로 shift
    src_x_start = max(0, shift_x)
    src_x_end = min(sensor_width, sensor_width + shift_x)
    src_y_start = max(0, shift_y)
    src_y_end = min(sensor_height, sensor_height + shift_y)
    
    # 결과에 복사할 위치 계산
    dst_x_start = max(0, -shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_start = max(0, -shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    
    # 원본 데이터 복사
    if src_x_end > src_x_start and src_y_end > src_y_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            frame[src_y_start:src_y_end, src_x_start:src_x_end]
    
    # Zero padding은 이미 zeros로 초기화되어 있으므로 추가 작업 불필요
    
    return result


def create_brownian_motion_dataset(
    input_bin_path: str,
    output_bin_path: str,
    config: BrownianMotionConfig,
    max_frames: Optional[int] = None,
    visualize: bool = True,
    keep_original_resolution: bool = False
):
    """
    Brownian motion 기반 데이터셋 생성
    
    Args:
        input_bin_path: 입력 bin 파일 경로
        output_bin_path: 출력 bin 파일 경로
        config: Brownian motion 설정
        max_frames: 최대 처리 프레임 수 (None이면 전체)
        visualize: 시각화 생성 여부
        keep_original_resolution: True면 원본 resolution 유지, False면 ROI crop
    """
    print("=" * 80)
    if keep_original_resolution:
        print("Brownian Motion 기반 DVS 데이터셋 생성 (원본 Resolution 유지)")
    else:
        print("Brownian Motion 기반 DVS 데이터셋 생성 (ROI Crop)")
    print("=" * 80)
    print(f"📂 입력 파일: {input_bin_path}")
    print(f"📂 출력 파일: {output_bin_path}")
    print(f"⚙️  설정:")
    if keep_original_resolution:
        print(f"   - Resolution: 720x960 (원본 유지)")
        print(f"   - Padding: Zero padding (비어버리는 영역은 0으로 채움)")
    else:
        print(f"   - ROI 크기: {config.roi_size}")
    print(f"   - 초기 중심: ({config.initial_center_x}, {config.initial_center_y})")
    print(f"   - Brownian motion (σx={config.sigma_x}, σy={config.sigma_y})")
    print(f"   - 최대 shift 범위: ±{config.max_shift_range}px")
    print()
    
    # 파일 존재 확인
    if not os.path.exists(input_bin_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_bin_path}")
    
    # 1. 원본 bin 파일 읽기
    print("📖 원본 데이터 읽는 중...")
    processor_original = BinProcessor(
        frame_width=960,
        frame_height=720,
        has_header=True
    )
    
    original_frames = processor_original.read_frames(
        input_bin_path,
        max_frames=max_frames
    )
    
    print(f"   ✅ {len(original_frames)}개 프레임 읽음")
    
    # 2. Brownian shift 생성기 초기화
    shift_gen = BrownianShiftGenerator(config, keep_original_resolution=keep_original_resolution)
    
    # 3. 새로운 프레임 생성
    print("\n🔄 Brownian motion 적용 중...")
    new_frames = []
    shift_history = []  # shift 정보 저장
    
    for frame_idx, original_frame in enumerate(original_frames):
        if (frame_idx + 1) % 100 == 0 or frame_idx == 0:
            print(f"   처리 중: {frame_idx + 1}/{len(original_frames)}")
        
        # Brownian motion으로 shift 직접 생성
        shift_x, shift_y = shift_gen.step()
        
        # shift 정보 저장 (shift는 누적값)
        shift_history.append({
            'frame_idx': frame_idx,
            'shift_x': shift_x,  # 누적된 shift 값
            'shift_y': shift_y   # 누적된 shift 값
        })
        
        # 모드에 따라 처리
        if keep_original_resolution:
            # 원본 resolution 유지 모드
            shifted_frame = apply_shift_with_padding(
                frame=original_frame.raw_data,
                shift_x=shift_x,
                shift_y=shift_y,
                padding_mode='zero'
            )
            
            # 새로운 프레임 생성 (원본 resolution 유지)
            new_frame = DVSFrame(
                header=original_frame.header,
                raw_data=shifted_frame,
                width=original_frame.width,  # 원본 유지 (960)
                height=original_frame.height  # 원본 유지 (720)
            )
        else:
            # ROI crop 모드 (기존 방식)
            roi_frame = extract_roi_from_frame_with_shift(
                frame=original_frame.raw_data,
                true_center_x=config.initial_center_x,
                true_center_y=config.initial_center_y,
                roi_size=config.roi_size,
                shift_x=shift_x,
                shift_y=shift_y,
                sensor_size=(720, 960)
            )
        
            # 새로운 프레임 생성 (ROI 크기)
        new_frame = DVSFrame(
            header=original_frame.header,
            raw_data=roi_frame,
            width=config.roi_size[0],
            height=config.roi_size[1]
        )
        
        new_frames.append(new_frame)
    
    print(f"   ✅ {len(new_frames)}개 프레임 생성 완료")
    
    # 4. 새로운 bin 파일 저장
    print(f"\n💾 새로운 bin 파일 저장 중: {output_bin_path}")
    if keep_original_resolution:
        processor_new = BinProcessor(
            frame_width=960,  # 원본 resolution 유지
            frame_height=720,  # 원본 resolution 유지
            has_header=True
        )
    else:
        processor_new = BinProcessor(
            frame_width=config.roi_size[0],
            frame_height=config.roi_size[1],
            has_header=True
        )
    
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    processor_new.write_frames(new_frames, output_bin_path)
    
    # shift 정보 저장 (JSON)
    shift_info_path = output_bin_path.replace('.bin', '_shifts.json')
    with open(shift_info_path, 'w') as f:
        json.dump({
            'config': {
                'roi_size': config.roi_size,
                'initial_center': (config.initial_center_x, config.initial_center_y),
                'sigma_x': config.sigma_x,
                'sigma_y': config.sigma_y,
                'max_shift_range': config.max_shift_range,
                'random_seed': config.random_seed
            },
            'shifts': shift_history
        }, f, indent=2)
    print(f"   ✅ Shift 정보 저장: {shift_info_path}")
    
    # 학습용 레이블 CSV 저장 (CNN, YOLO, Filter 모두 지원)
    csv_path = output_bin_path.replace('.bin', '_labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if keep_original_resolution:
            # 원본 resolution 유지 모드
            sensor_width = 960
            sensor_height = 720
            laser_diameter = 400
            
            # 헤더
            writer.writerow([
                'frame_idx',
                'shift_x', 'shift_y',
                'original_center_x', 'original_center_y',
                'cnn_rel_x', 'cnn_rel_y',
                'yolo_bbox_x', 'yolo_bbox_y', 'yolo_bbox_w', 'yolo_bbox_h'
            ])
            
            # 각 프레임의 레이블 데이터
            for shift_data in shift_history:
                frame_idx = shift_data['frame_idx']
                shift_x = shift_data['shift_x']
                shift_y = shift_data['shift_y']
                
                # shift된 프레임에서의 레이저 중심 좌표
                # shift_x > 0: 프레임을 왼쪽으로 shift → 원본의 (x, y)가 결과의 (x-shift_x, y-shift_y)로 이동
                # 따라서 shift된 프레임에서 레이저 위치 = initial_center - shift
                laser_center_x = config.initial_center_x - shift_x
                laser_center_y = config.initial_center_y - shift_y
                
                # 경계 체크 및 클리핑
                laser_center_x = np.clip(laser_center_x, 0, sensor_width - 1)
                laser_center_y = np.clip(laser_center_y, 0, sensor_height - 1)
                
                # 정규화된 상대 좌표 (원본 resolution 기준)
                cnn_rel_x = laser_center_x / sensor_width
                cnn_rel_y = laser_center_y / sensor_height
                
                # 정규화된 bbox
                yolo_bbox_x = cnn_rel_x
                yolo_bbox_y = cnn_rel_y
                yolo_bbox_w = np.clip(laser_diameter / sensor_width, 0.0, 1.0)
                yolo_bbox_h = np.clip(laser_diameter / sensor_height, 0.0, 1.0)
                
                writer.writerow([
                    frame_idx,
                    shift_x, shift_y,
                    laser_center_x, laser_center_y,
                    cnn_rel_x, cnn_rel_y,
                    yolo_bbox_x, yolo_bbox_y, yolo_bbox_w, yolo_bbox_h
                ])
            
            print(f"   ✅ 학습용 레이블 CSV 저장: {csv_path}")
            print(f"      - CNN: cnn_rel_x, cnn_rel_y (정규화 좌표, 원본 resolution 기준)")
            print(f"      - YOLO: yolo_bbox_x/y/w/h (정규화 bbox)")
            print(f"      - Filter: original_center_x/y (원본 좌표계 레이저 중심)")
        else:
            # ROI crop 모드 (기존 방식)
            roi_width, roi_height = config.roi_size
            roi_center_x = roi_width // 2
            roi_center_y = roi_height // 2
            laser_diameter = 400
            
        # 헤더
        writer.writerow([
            'frame_idx',
            'shift_x', 'shift_y',
            'original_center_x', 'original_center_y',
            'roi_center_x', 'roi_center_y',
            'cnn_rel_x', 'cnn_rel_y',
            'yolo_bbox_x', 'yolo_bbox_y', 'yolo_bbox_w', 'yolo_bbox_h'
        ])
        
        # 각 프레임의 레이블 데이터
        for shift_data in shift_history:
            frame_idx = shift_data['frame_idx']
            shift_x = shift_data['shift_x']
            shift_y = shift_data['shift_y']
            
                # 원본 좌표계 절대 좌표
            original_center_x = config.initial_center_x - shift_x
            original_center_y = config.initial_center_y - shift_y
            
                # ROI 내부 절대 좌표
            roi_center_x_abs = roi_center_x + shift_x
            roi_center_y_abs = roi_center_y + shift_y
            
                # 정규화된 상대 좌표 (ROI 기준)
            cnn_rel_x = np.clip(0.5 - (shift_x / roi_width), 0.0, 1.0)
            cnn_rel_y = np.clip(0.5 - (shift_y / roi_height), 0.0, 1.0)
            
                # 정규화된 bbox
            yolo_bbox_x = np.clip(0.5 - (shift_x / roi_width), 0.0, 1.0)
            yolo_bbox_y = np.clip(0.5 - (shift_y / roi_height), 0.0, 1.0)
            yolo_bbox_w = np.clip(laser_diameter / roi_width, 0.0, 1.0)
            yolo_bbox_h = np.clip(laser_diameter / roi_height, 0.0, 1.0)
            
            writer.writerow([
                frame_idx,
                shift_x, shift_y,
                original_center_x, original_center_y,
                roi_center_x_abs, roi_center_y_abs,
                cnn_rel_x, cnn_rel_y,
                yolo_bbox_x, yolo_bbox_y, yolo_bbox_w, yolo_bbox_h
            ])
    
    print(f"   ✅ 학습용 레이블 CSV 저장: {csv_path}")
    print(f"      - CNN: cnn_rel_x, cnn_rel_y (정규화 좌표, ROI 기준)")
    print(f"      - YOLO: yolo_bbox_x/y/w/h (정규화 bbox)")
    print(f"      - Filter: original_center_x/y (원본 좌표계) 또는 roi_center_x/y (ROI 내부)")
    
    # 5. 시각화 (선택적)
    if visualize:
        print("\n📊 시각화 생성 중...")
        # 레이저 궤적 계산
        laser_trajectory = []
        for shift_data in shift_history:
            shift_x = shift_data['shift_x']
            shift_y = shift_data['shift_y']
            if keep_original_resolution:
                laser_x = config.initial_center_x + shift_x
                laser_y = config.initial_center_y + shift_y
            else:
                roi_center_x = config.roi_size[0] // 2
                roi_center_y = config.roi_size[1] // 2
                laser_x = roi_center_x + shift_x
                laser_y = roi_center_y + shift_y
            laser_trajectory.append([laser_x, laser_y])
        laser_trajectory = np.array(laser_trajectory)
        
        if keep_original_resolution:
            initial_pos = np.array([config.initial_center_x, config.initial_center_y])
        else:
            initial_pos = np.array([config.roi_size[0] // 2, config.roi_size[1] // 2])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 궤적 시각화 (원본 좌표계)
        ax = axes[0, 0]
        ax.plot(laser_trajectory[:, 0], laser_trajectory[:, 1], 'b-', alpha=0.6, linewidth=1)
        ax.scatter(laser_trajectory[0, 0], laser_trajectory[0, 1], c='green', s=100, 
                   marker='o', label='Start', zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter(laser_trajectory[-1, 0], laser_trajectory[-1, 1], c='red', s=100, 
                   marker='s', label='End', zorder=5, edgecolors='black', linewidths=1.5)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_title('Brownian Motion Trajectory (Original Coordinates)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # X 좌표 시간 변화
        ax = axes[0, 1]
        ax.plot(laser_trajectory[:, 0], 'b-', linewidth=1, alpha=0.7)
        ax.axhline(initial_pos[0], color='r', linestyle='--', 
                   label='Initial Center', alpha=0.5, linewidth=2)
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('X Coordinate', fontsize=11)
        ax.set_title('X Coordinate Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Y 좌표 시간 변화
        ax = axes[1, 0]
        ax.plot(laser_trajectory[:, 1], 'b-', linewidth=1, alpha=0.7)
        ax.axhline(initial_pos[1], color='r', linestyle='--', 
                   label='Initial Center', alpha=0.5, linewidth=2)
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title('Y Coordinate Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 거리 변화
        ax = axes[1, 1]
        distances = np.sqrt(np.sum((laser_trajectory - initial_pos) ** 2, axis=1))
        ax.plot(distances, 'b-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Distance from Start (pixels)', fontsize=11)
        ax.set_title('Distance from Initial Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        viz_path = output_bin_path.replace('.bin', '_trajectory.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ 시각화 저장: {viz_path}")
        plt.close()
    
    # 통계 출력
    # 궤적은 원본 좌표계 기준으로 계산
    laser_trajectory = []
    for shift_data in shift_history:
        shift_x = shift_data['shift_x']
        shift_y = shift_data['shift_y']
        laser_x = config.initial_center_x + shift_x
        laser_y = config.initial_center_y + shift_y
        laser_trajectory.append([laser_x, laser_y])
    
    laser_trajectory = np.array(laser_trajectory)
    initial_pos = np.array([config.initial_center_x, config.initial_center_y])
    distances = np.sqrt(np.sum((laser_trajectory - initial_pos) ** 2, axis=1))
    
    # 통계 출력
    laser_trajectory = []
    for shift_data in shift_history:
        shift_x = shift_data['shift_x']
        shift_y = shift_data['shift_y']
        if keep_original_resolution:
            laser_x = config.initial_center_x + shift_x
            laser_y = config.initial_center_y + shift_y
        else:
            roi_center_x = config.roi_size[0] // 2
            roi_center_y = config.roi_size[1] // 2
            laser_x = roi_center_x + shift_x
            laser_y = roi_center_y + shift_y
        laser_trajectory.append([laser_x, laser_y])
    
    laser_trajectory = np.array(laser_trajectory)
    if keep_original_resolution:
        initial_pos = np.array([config.initial_center_x, config.initial_center_y])
    else:
        initial_pos = np.array([config.roi_size[0] // 2, config.roi_size[1] // 2])
    distances = np.sqrt(np.sum((laser_trajectory - initial_pos) ** 2, axis=1))
    
    print("\n" + "=" * 80)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 80)
    print(f"📊 통계:")
    print(f"   - 총 프레임 수: {len(new_frames)}")
    if keep_original_resolution:
        print(f"   - Resolution: 720x960 (원본 유지)")
    else:
        print(f"   - ROI 크기: {config.roi_size}")
    print(f"   - 레이저 궤적 범위: X=[{laser_trajectory[:, 0].min():.1f}, {laser_trajectory[:, 0].max():.1f}], "
          f"Y=[{laser_trajectory[:, 1].min():.1f}, {laser_trajectory[:, 1].max():.1f}]")
    print(f"   - 최대 이동 거리: {np.max(distances):.1f} pixels")
    print(f"   - 평균 이동 거리: {np.mean(distances):.2f} pixels")
    print(f"   - 표준편차: {np.std(distances):.2f} pixels")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Brownian Motion 기반 DVS 데이터셋 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # ROI crop 모드 (기본)
  python generate_brownian_dataset.py
  
  # 원본 resolution 유지 모드
  python generate_brownian_dataset.py -org
        """
    )
    
    parser.add_argument(
        '-org', '--original-resolution',
        action='store_true',
        help='원본 resolution (720x960) 유지 모드 (기본: ROI crop 모드)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=os.path.join(dvs_root, "data", "gaussian_large.bin"),
        help='입력 bin 파일 경로'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 bin 파일 경로 (지정하지 않으면 자동 생성)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='최대 처리 프레임 수 (None이면 전체)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='시각화 생성 안 함'
    )
    
    args = parser.parse_args()
    
    # 출력 파일 경로 자동 생성
    if args.output is None:
        if args.original_resolution:
            args.output = args.input.replace('.bin', '_brownian_720x960.bin')
        else:
            args.output = args.input.replace('.bin', '_brownian_512x512.bin')
    
    # Brownian motion 설정
    config = BrownianMotionConfig(
        initial_center_x=541,
        initial_center_y=361,
        roi_size=(512, 512),
        sigma_x=2.0,  # X축 이동 표준편차
        sigma_y=2.0,  # Y축 이동 표준편차
        max_shift_range=50,  # 최대 shift 범위 (픽셀)
        random_seed=42  # 재현성을 위한 시드
    )
    
    # 데이터셋 생성
    try:
        create_brownian_motion_dataset(
            input_bin_path=args.input,
            output_bin_path=args.output,
            config=config,
            max_frames=args.max_frames,
            visualize=not args.no_visualize,
            keep_original_resolution=args.original_resolution
        )
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
"""
DVS bin 파일 프레임 단위 처리 및 필터링 프레임워크

이 모듈은 DVS 센서에서 생성된 bin 파일을 프레임 단위로 읽어서
2bit 데이터에 직접 다양한 필터를 적용할 수 있는 프레임워크를 제공합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from abc import ABC, abstractmethod
import time
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

# 상위 디렉토리를 sys.path에 추가 (lib 모듈 사용을 위해)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 공통 라이브러리에서 import
from lib.bin_processor import FrameHeader, ProcessingStats, DVSFrame, BinProcessor

class FrameFilter(ABC):
    """프레임 필터 추상 클래스"""
    
    @abstractmethod
    def apply(self, frame: DVSFrame) -> bool:
        """
        프레임에 필터 적용
        
        Args:
            frame: DVSFrame 객체
            
        Returns:
            bool: True면 프레임 유지, False면 프레임 제거
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """필터 이름 반환"""
        pass


class FilterableBinProcessor(BinProcessor):
    """필터 기능이 추가된 BinProcessor"""
    
    def __init__(self, frame_width: int, frame_height: int, has_header: bool = True):
        super().__init__(frame_width, frame_height, has_header)
        self.filters: List[FrameFilter] = []
        self.filter_stats = {
            'filtered_frames': 0,
            'events_before': 0,
            'events_after': 0
        }
    
    def add_filter(self, filter_obj: FrameFilter):
        """필터 추가"""
        self.filters.append(filter_obj)
        print(f"Filter added: {filter_obj.get_name()}")
    
    def clear_filters(self):
        """모든 필터 제거"""
        self.filters.clear()
        print("All filters cleared")
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> List[DVSFrame]:
        """파일 전체를 처리하고 필터링된 프레임 반환"""
        start_time = time.time()
        
        # 프레임 읽기
        frames = self.read_frames(input_path)
        self.stats.total_frames = len(frames)
        
        # 필터 적용
        filtered_frames = []
        total_events_before = 0
        total_events_after = 0
        
        for frame in frames:
            events_before = frame.count_events()['total_events']
            total_events_before += events_before
            
            # 모든 필터 적용
            keep_frame = True
            for filter_obj in self.filters:
                if not filter_obj.apply(frame):
                    keep_frame = False
                    break
            
            if keep_frame:
                filtered_frames.append(frame)
                events_after = frame.count_events()['total_events']
                total_events_after += events_after
                self.stats.processed_frames += 1
            else:
                self.filter_stats['filtered_frames'] += 1
        
        # 통계 업데이트
        self.stats.processing_time_sec = time.time() - start_time
        self.filter_stats['events_before'] = total_events_before
        self.filter_stats['events_after'] = total_events_after
        
        # 결과 저장
        if output_path:
            self.write_frames(filtered_frames, output_path)
        
        return filtered_frames
    
    def print_stats(self):
        """처리 통계 출력"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total frames read:     {self.stats.total_frames}")
        print(f"Frames processed:      {self.stats.processed_frames}")
        print(f"Frames filtered out:   {self.filter_stats['filtered_frames']}")
        print(f"Processing time:       {self.stats.processing_time_sec:.3f} sec")
        print(f"Events before:         {self.filter_stats['events_before']}")
        print(f"Events after:          {self.filter_stats['events_after']}")
        if self.filter_stats['events_before'] > 0:
            reduction = (1 - self.filter_stats['events_after'] / self.filter_stats['events_before']) * 100
            print(f"Event reduction:       {reduction:.1f}%")
        print("="*50)

# ============================================================================
# 기본 필터 구현들
# ============================================================================

class EventDensityFilter(FrameFilter):
    """이벤트 밀도 기반 필터 - 너무 많거나 적은 이벤트를 가진 프레임 제거"""
    
    def __init__(self, min_events: int = 10, max_events: Optional[int] = None):
        self.min_events = min_events
        self.max_events = max_events
    
    def apply(self, frame: DVSFrame) -> bool:
        total_events = frame.count_events()['total_events']
        
        if total_events < self.min_events:
            return False
        
        if self.max_events is not None and total_events > self.max_events:
            return False
        
        return True
    
    def get_name(self) -> str:
        return f"EventDensity(min={self.min_events}, max={self.max_events})"

class SpatialClusterFilter(FrameFilter):
    """최적화된 공간적 클러스터링 필터 - 정확도와 성능을 모두 고려"""
    
    def __init__(self, radius: float = 5.0, min_neighbors: int = 3, 
                 apply_to_on: bool = True, apply_to_off: bool = True,
                 use_fast_mode: bool = True):
        """
        Args:
            radius: 검색 반경
            min_neighbors: 최소 이웃 개수  
            apply_to_on: ON 이벤트에 적용 여부
            apply_to_off: OFF 이벤트에 적용 여부
            use_fast_mode: 대량 이벤트시 고속 모드 사용 (정확도 약간 감소)
        """
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.apply_to_on = apply_to_on
        self.apply_to_off = apply_to_off
        self.use_fast_mode = use_fast_mode
        self._fast_threshold = 20000  # 이 이상이면 고속 모드
    
    def apply(self, frame: DVSFrame) -> bool:
        """최적화된 필터 적용"""
        if self.apply_to_on:
            on_events = frame.get_on_events()
            if len(on_events) > 0:
                filtered_on = self._filter_events_adaptive(on_events)
                self._update_frame_vectorized(frame, on_events, filtered_on, 1)
        
        if self.apply_to_off:
            off_events = frame.get_off_events()
            if len(off_events) > 0:
                filtered_off = self._filter_events_adaptive(off_events)
                self._update_frame_vectorized(frame, off_events, filtered_off, 2)
        
        return True
    
    def _filter_events_adaptive(self, events: np.ndarray) -> np.ndarray:
        """적응적 필터링 - 이벤트 수에 따라 최적 알고리즘 선택"""
        if len(events) == 0:
            return events
        
        # 대량 이벤트시 고속 모드 사용 (정확도 유지하면서 성능 향상)
        if self.use_fast_mode and len(events) > self._fast_threshold:
            return self._filter_events_optimized(events)
        else:
            return self._filter_events_precise(events)
    
    def _filter_events_precise(self, events: np.ndarray) -> np.ndarray:
        """정확한 KDTree 기반 필터링 (소량 이벤트용)"""
        tree = cKDTree(events)
        neighbors_list = tree.query_ball_point(events, r=self.radius)
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbors_list])
        dense_mask = neighbor_counts >= (self.min_neighbors + 1)
        return events[dense_mask]
    
    def _filter_events_optimized(self, events: np.ndarray) -> np.ndarray:
        """개선된 픽셀 그리드 기반 필터링 (KDTree와 유사한 정확도)"""
        from scipy.ndimage import uniform_filter
        
        # DVS 해상도 (Brownian motion ROI: 512x512)
        height, width = 512, 512
        
        # 이벤트 범위 검증
        valid_mask = (events[:, 0] >= 0) & (events[:, 0] < width) & \
                     (events[:, 1] >= 0) & (events[:, 1] < height)
        valid_events = events[valid_mask]
        
        if len(valid_events) == 0:
            return np.array([]).reshape(0, 2)
        
        # 각 이벤트 주변의 정확한 이웃 카운트 (하이브리드 방식)
        keep_mask = np.zeros(len(valid_events), dtype=bool)
        
        # 픽셀별 이벤트 맵 생성
        event_map = np.zeros((height, width), dtype=bool)
        event_map[valid_events[:, 1], valid_events[:, 0]] = True
        
        # 각 이벤트별로 정확한 반경 내 이웃 계산
        for i, (x, y) in enumerate(valid_events):
            # 검색 영역 설정 (원형에 가깝게)
            r = int(self.radius) + 1
            x_min, x_max = max(0, x-r), min(width, x+r+1)
            y_min, y_max = max(0, y-r), min(height, y+r+1)
            
            # 해당 영역의 이벤트들 확인
            region = event_map[y_min:y_max, x_min:x_max]
            if not np.any(region):
                continue
                
            # 영역 내 이벤트 좌표 찾기
            y_coords, x_coords = np.where(region)
            y_coords += y_min
            x_coords += x_min
            
            # 실제 거리 계산하여 반경 내 이웃 카운트
            distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            neighbors_in_radius = np.sum(distances <= self.radius)
            
            # 최소 이웃 조건 확인 (자신 포함)
            if neighbors_in_radius >= (self.min_neighbors + 1):
                keep_mask[i] = True
        
        return valid_events[keep_mask]
    
    def _update_frame_vectorized(self, frame: DVSFrame, original_events: np.ndarray, 
                                filtered_events: np.ndarray, event_value: int):
        """벡터화된 고속 프레임 업데이트"""
        if len(original_events) == 0:
            return
        
        # 원본 이벤트들을 벡터화하여 0으로 설정
        frame.raw_data[original_events[:, 1], original_events[:, 0]] = 0
        
        # 필터링된 이벤트들을 벡터화하여 설정
        if len(filtered_events) > 0:
            frame.raw_data[filtered_events[:, 1], filtered_events[:, 0]] = event_value
    
    def get_name(self) -> str:
        mode = "adaptive" if self.use_fast_mode else "precise"
        return f"SpatialCluster(r={self.radius}, min_n={self.min_neighbors}, {mode})"

class TimeRangeFilter(FrameFilter):
    """시간 범위 필터"""
    
    def __init__(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time
    
    def apply(self, frame: DVSFrame) -> bool:
        return self.start_time <= frame.header.timestamp <= self.end_time
    
    def get_name(self) -> str:
        return f"TimeRange({self.start_time}-{self.end_time})"

class ROIFilter(FrameFilter):
    """자동 BBOX 기반 ROI 필터 - 이벤트 밀도에 따라 관심 영역 자동 설정"""
    
    def __init__(self, min_events: int = 50, margin: int = 30, update_frequency: int = 10):
        """
        Args:
            min_events: 최소 이벤트 수 (이보다 적으면 전체 프레임 사용)
            margin: BBOX 주변 여유 공간 (픽셀)
            update_frequency: ROI 업데이트 빈도 (프레임 단위)
        """
        self.min_events = min_events
        self.margin = margin
        self.update_frequency = update_frequency
        self.frame_count = 0
        self.current_bbox = None  # (x, y, width, height)
    
    def apply(self, frame: DVSFrame) -> bool:
        """자동 ROI 적용"""
        self.frame_count += 1
        
        # 지정된 빈도마다 BBOX 업데이트
        if self.frame_count % self.update_frequency == 1 or self.current_bbox is None:
            self._update_bbox(frame)
        
        # BBOX 적용 (관심 영역만 남기기)
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            new_frame = np.zeros_like(frame.raw_data)
            x_end, y_end = min(x + w, frame.width), min(y + h, frame.height)
            new_frame[y:y_end, x:x_end] = frame.raw_data[y:y_end, x:x_end]
            frame.raw_data = new_frame
        
        return True
    
    def _update_bbox(self, frame: DVSFrame):
        """노이즈 제거 후 BBOX 자동 계산"""
        # 모든 이벤트 좌표 수집
        on_events = frame.get_on_events()
        off_events = frame.get_off_events()
        all_events = np.vstack([on_events, off_events]) if len(on_events) > 0 and len(off_events) > 0 else \
                     on_events if len(on_events) > 0 else off_events
        
        if len(all_events) < self.min_events:
            self.current_bbox = None  # 전체 프레임 사용
            return
        
        # 노이즈 제거: 95% percentile 사용 (상위 5% 극값 제거)
        x_coords, y_coords = all_events[:, 0], all_events[:, 1]
        
        x_min = int(np.percentile(x_coords, 5))   # 하위 5% 제거
        x_max = int(np.percentile(x_coords, 95))  # 상위 5% 제거  
        y_min = int(np.percentile(y_coords, 5))   # 하위 5% 제거
        y_max = int(np.percentile(y_coords, 95))  # 상위 5% 제거
        
        # 마진 추가 및 경계 클리핑
        x_min = max(0, x_min - self.margin)
        y_min = max(0, y_min - self.margin)
        x_max = min(frame.width - 1, x_max + self.margin)
        y_max = min(frame.height - 1, y_max + self.margin)
        
        self.current_bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    def get_bbox_info(self):
        """현재 BBOX 정보 반환"""
        return self.current_bbox
    
    def get_name(self) -> str:
        return f"AutoROI(min_ev={self.min_events}, margin={self.margin})"

# ============================================================================
# 중심점 추출기 구현들 (Center Point Extractors)
# ============================================================================
class CenterPointExtractor(ABC):
    """중심점 추출기 추상 클래스"""

    def __init__(self):
        self.center_point: Optional[np.ndarray] = None

    @abstractmethod
    def extract(self, frame: DVSFrame) -> Optional[np.ndarray]:
        """
        프레임에서 중심점을 추출합니다.

        Args:
            frame: DVSFrame 객체

        Returns:
            np.ndarray: (x, y) 중심점 좌표. 추출 실패 시 None.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """추출기 이름 반환"""
        pass
    
    def get_center(self) -> Optional[np.ndarray]:
        """마지막으로 계산된 중심점 반환"""
        return self.center_point


class MeanPointExtractor(CenterPointExtractor):
    """
    ON과 OFF 전체 이벤트들의 산술 평균(mean)을 계산하여 중심점을 추출합니다.
    """
    def extract(self, frame: DVSFrame) -> Optional[np.ndarray]:
        on_events = frame.get_on_events()
        off_events = frame.get_off_events()

        # ON과 OFF 이벤트를 모두 결합
        if len(on_events) > 0 and len(off_events) > 0:
            # 둘 다 있는 경우: 결합
            all_events = np.vstack([on_events, off_events])
        elif len(on_events) > 0:
            # ON 이벤트만 있는 경우
            all_events = on_events
        elif len(off_events) > 0:
            # OFF 이벤트만 있는 경우
            all_events = off_events
        else:
            # 이벤트가 없는 경우
            self.center_point = None
            return None
        
        # 전체 이벤트의 평균으로 중심점 계산
        self.center_point = np.mean(all_events, axis=0)
        return self.center_point

    def get_name(self) -> str:
        return "MeanPointExtractor"

class MedianPointExtractor(CenterPointExtractor):
    """
    ON과 OFF 전체 이벤트들의 중앙값(median)을 계산하여 중심점을 추출합니다.
    아웃라이어에 덜 민감한 특징이 있습니다.
    """
    def extract(self, frame: DVSFrame) -> Optional[np.ndarray]:
        on_events = frame.get_on_events()
        off_events = frame.get_off_events()

        # ON과 OFF 이벤트를 모두 결합
        if len(on_events) > 0 and len(off_events) > 0:
            # 둘 다 있는 경우: 결합
            all_events = np.vstack([on_events, off_events])
        elif len(on_events) > 0:
            # ON 이벤트만 있는 경우
            all_events = on_events
        elif len(off_events) > 0:
            # OFF 이벤트만 있는 경우
            all_events = off_events
        else:
            # 이벤트가 없는 경우
            self.center_point = None
            return None

        # 전체 이벤트의 중앙값으로 중심점 계산
        self.center_point = np.median(all_events, axis=0)
        return self.center_point

    def get_name(self) -> str:
        return "MedianPointExtractor"

class KalmanPointExtractor(CenterPointExtractor):
    """
    칼만 필터를 사용하여 시간적으로 안정화된 중심점을 추정/추출합니다.
    Brownian motion 모델: process noise = sigma^2 (sigma_x=2.0, sigma_y=2.0)
    """
    def __init__(self, process_noise: float = 4.0, measurement_noise: float = 10.0):
        super().__init__()
        # filterpy 라이브러리 필요: pip install filterpy
        from filterpy.kalman import KalmanFilter
        from filterpy.common import Q_discrete_white_noise

        # State vector: [x, y] - position with Brownian motion
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.kf.x = np.array([0., 0.])  # Initial state [x, y]
        
        # State transition matrix (Brownian motion: position changes randomly)
        # Identity matrix - position changes through process noise
        self.kf.F = np.array([[1, 0], [0, 1]])  # Identity matrix
        
        # Observation matrix (position directly observed)
        self.kf.H = np.array([[1, 0], [0, 1]])  # Identity matrix
        
        # Initial covariance (high uncertainty)
        self.kf.P *= 1000.
        
        # Measurement noise covariance
        self.kf.R = np.eye(2) * measurement_noise
        
        # Process noise covariance (Brownian motion: sigma^2 per dimension)
        # Default: sigma = 2.0, so process_noise = sigma^2 = 4.0
        # This models the random walk behavior of Brownian motion
        self.kf.Q = np.eye(2) * process_noise
        
        self.is_initialized = False

    def extract(self, frame: DVSFrame) -> Optional[np.ndarray]:
        # Extract all events (both ON and OFF)
        on_events = frame.get_on_events()
        off_events = frame.get_off_events()
        
        # Combine ON and OFF events
        if len(on_events) > 0 and len(off_events) > 0:
            all_events = np.vstack([on_events, off_events])
        elif len(on_events) > 0:
            all_events = on_events
        elif len(off_events) > 0:
            all_events = off_events
        else:
            all_events = np.empty((0, 2))
        
        # Compute measurement (observed center) if enough events
        measured_center = None
        if len(all_events) > 5:  # Minimum event threshold
             measured_center = np.mean(all_events, axis=0)

        # Initialize Kalman filter with first valid measurement
        if not self.is_initialized and measured_center is not None:
            self.kf.x = measured_center  # Set initial position
            self.is_initialized = True
            
        # Kalman filter operation (after initialization)
        if self.is_initialized:
            # Prediction step (always performed)
            self.kf.predict()
            
            # Update step (only when measurement available)
            if measured_center is not None:
                self.kf.update(measured_center)
            
            # Return estimated position
            self.center_point = self.kf.x
            return self.center_point
        
        # Not initialized yet
        self.center_point = None
        return None

    def get_name(self) -> str:
        return "KalmanPointExtractor"

    def reset(self):
        """Reset Kalman filter state"""
        self.is_initialized = False
        self.center_point = None
        self.kf.x.fill(0)  # Reset state vector to [0, 0]
        self.kf.P *= 1000.  # Reset covariance to high uncertainty


class TemporalAveragePointExtractor(CenterPointExtractor):
    """
    여러 프레임에 걸쳐 중심점을 누적하여 템포럴 평균을 계산하는 추출기.
    기존의 CenterPointExtractor 구현체를 재활용하여 템포럴 평균을 계산합니다.
    """
    def __init__(self, window_size: int = 5, base_extractor: CenterPointExtractor = None):
        """
        Args:
            window_size: 평균을 계산할 프레임 윈도우 크기
            base_extractor: 각 프레임에서 기본 중심점을 계산할 추출기 객체
                           None이면 MeanPointExtractor 사용
        """
        super().__init__()
        self.window_size = window_size
        self.center_buffer = []  # 중심점들을 저장할 버퍼
        self.frame_count = 0
        
        # 기본 추출기 설정
        if base_extractor is None:
            self.base_extractor = MeanPointExtractor()
        else:
            self.base_extractor = base_extractor

    def extract(self, frame: DVSFrame) -> Optional[np.ndarray]:
        """
        프레임에서 중심점을 추출하고 템포럴 평균을 계산합니다.
        """
        self.frame_count += 1
        
        # 기존 추출기를 사용하여 현재 프레임에서 중심점 계산
        current_center = self.base_extractor.extract(frame)
        
        if current_center is not None:
            # 버퍼에 중심점 추가
            self.center_buffer.append(current_center.copy())
            
            # 윈도우 크기를 초과하면 오래된 데이터 제거
            if len(self.center_buffer) > self.window_size:
                self.center_buffer.pop(0)
        
        # 버퍼에 충분한 데이터가 있으면 템포럴 평균 계산
        if len(self.center_buffer) > 0:
            self.center_point = np.mean(self.center_buffer, axis=0)
            return self.center_point
        else:
            self.center_point = None
            return None
    
    def get_name(self) -> str:
        base_name = self.base_extractor.get_name()
        return f"TemporalAverage(window={self.window_size}, base={base_name})"
    
    def reset(self):
        """버퍼와 상태 초기화"""
        self.center_buffer.clear()
        self.frame_count = 0
        self.center_point = None
        # 기본 추출기가 reset 메서드를 가지고 있다면 호출
        if hasattr(self.base_extractor, 'reset'):
            self.base_extractor.reset()
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """현재 버퍼 상태 정보 반환"""
        return {
            'buffer_size': len(self.center_buffer),
            'window_size': self.window_size,
            'frame_count': self.frame_count,
            'buffer_full': len(self.center_buffer) >= self.window_size,
            'base_extractor': self.base_extractor.get_name()
        }


if __name__ == "__main__":
    # 테스트 코드
    print("DVS Bin Frame Processor - Test")
    
    # 테스트용 더미 데이터 생성 및 처리 예제
    # (실제 사용시에는 실제 bin 파일 경로를 지정)
    pass

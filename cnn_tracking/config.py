#!/usr/bin/env python3
"""
DVS Laser Tracking 설정 파일
- ROI 고정, 물체가 움직이는 tracking 시나리오
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json


@dataclass
class TrackingDataConfig:
    """Tracking 데이터 관련 설정"""
    
    # 파일 경로
    bin_file_path: str = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    
    # 센서 및 ROI 설정
    sensor_size: Tuple[int, int] = (720, 960)  # (height, width)
    roi_center: Tuple[int, int] = (480, 294)   # ROI 고정 중심 좌표
    roi_size: Tuple[int, int] = (384, 384)     # ROI 크기 (height, width)
    
    # Brownian Motion 설정
    motion_std: float = 2.0  # 브라우니언 모션 표준편차 (픽셀/프레임)
    motion_boundary_margin: int = 80  # ROI 경계로부터 여유 공간 (픽셀)
    use_boundary_reflection: bool = True  # 경계에서 반사 여부
    
    # Temporal 설정
    num_temporal_frames: int = 5  # 시간 축으로 쌓을 프레임 수
    
    # 데이터 처리
    max_frames: Optional[int] = None
    frame_duration: int = 10000  # microseconds
    
    # 데이터 분할
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class TrackingModelConfig:
    """Tracking 모델 관련 설정"""
    
    # 모델 아키텍처
    model_name: str = "mobilenet_v2_light"  # basic_tracking, mobilenet_v2, mobilenet_v2_light, lstm_tracking, transformer_tracking
    input_channels: int = 5  # temporal frames
    output_dim: int = 2  # (x, y) coordinates
    
    # 추가 모델 옵션
    use_lstm: bool = False  # LSTM 레이어 추가 여부
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    
    # 기본 훈련 설정
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    
    # 옵티마이저
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    
    # 스케줄러
    use_scheduler: bool = True
    scheduler_type: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    patience: int = 20
    min_delta: float = 0.0
    
    # 손실 함수
    loss_function: str = "mse"  # mse, smooth_l1


@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    
    # 하드웨어
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    
    # 디렉토리
    output_dir: str = "outputs_tracking"
    checkpoint_dir: str = "checkpoints_tracking"
    log_dir: str = "logs_tracking"
    
    # 저장 설정
    save_best: bool = True
    save_last: bool = True
    save_freq: int = 10
    
    # 로깅
    verbose: bool = True
    log_level: str = "INFO"


@dataclass
class TrackingExperimentConfig:
    """Tracking 전체 실험 설정"""
    
    # 실험 정보
    experiment_name: str = "laser_tracking_experiment"
    description: str = "DVS laser tracking with fixed ROI and moving target"
    
    # 설정 컴포넌트
    data: TrackingDataConfig = field(default_factory=TrackingDataConfig)
    model: TrackingModelConfig = field(default_factory=TrackingModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """설정 후처리"""
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__
        }
    
    def save_config(self, filepath: str):
        """설정을 파일로 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TrackingExperimentConfig':
        """파일에서 설정 로드"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.experiment_name = config_dict.get('experiment_name', config.experiment_name)
        config.description = config_dict.get('description', config.description)
        
        # 각 컴포넌트 업데이트
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    # Tuple 타입 처리
                    if isinstance(getattr(config.data, key), tuple):
                        setattr(config.data, key, tuple(value))
                    else:
                        setattr(config.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'system' in config_dict:
            for key, value in config_dict['system'].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        return config


# 사전 정의된 설정들
def get_quick_test_config() -> TrackingExperimentConfig:
    """빠른 테스트용 설정"""
    config = TrackingExperimentConfig()
    
    config.data.max_frames = 100
    config.data.motion_std = 1.5
    config.data.num_temporal_frames = 5
    
    config.model.model_name = "mobilenet_v2_light"
    
    config.training.num_epochs = 20
    config.training.batch_size = 8
    
    config.experiment_name = "quick_tracking_test"
    
    return config


def get_standard_config() -> TrackingExperimentConfig:
    """표준 tracking 설정"""
    config = TrackingExperimentConfig()
    
    config.data.max_frames = 500
    config.data.motion_std = 2.0
    config.data.num_temporal_frames = 5
    config.data.motion_boundary_margin = 80
    
    config.model.model_name = "mobilenet_v2_light"
    
    config.training.num_epochs = 100
    config.training.batch_size = 16
    config.training.learning_rate = 0.001
    
    config.experiment_name = "standard_tracking"
    
    return config


def get_lstm_tracking_config() -> TrackingExperimentConfig:
    """LSTM 기반 tracking 설정"""
    config = TrackingExperimentConfig()
    
    config.data.max_frames = 500
    config.data.motion_std = 2.5
    config.data.num_temporal_frames = 10  # LSTM은 더 긴 시퀀스
    
    config.model.model_name = "lstm_tracking"
    config.model.use_lstm = True
    config.model.lstm_hidden_size = 128
    config.model.lstm_num_layers = 2
    
    config.training.num_epochs = 150
    config.training.batch_size = 8
    config.training.learning_rate = 0.0005
    
    config.experiment_name = "lstm_tracking"
    
    return config


if __name__ == "__main__":
    print("⚙️ Tracking Configuration System Test")
    print("=" * 50)
    
    # 기본 설정
    config = TrackingExperimentConfig()
    print(f"\n📋 Default Configuration:")
    print(f"   ROI center: {config.data.roi_center}")
    print(f"   ROI size: {config.data.roi_size}")
    print(f"   Motion std: {config.data.motion_std}")
    print(f"   Temporal frames: {config.data.num_temporal_frames}")
    print(f"   Model: {config.model.model_name}")
    
    # 빠른 테스트 설정
    quick = get_quick_test_config()
    print(f"\n🚀 Quick Test Configuration:")
    print(f"   Max frames: {quick.data.max_frames}")
    print(f"   Epochs: {quick.training.num_epochs}")
    
    # 설정 저장/로드 테스트
    test_file = "test_tracking_config.json"
    try:
        quick.save_config(test_file)
        loaded = TrackingExperimentConfig.load_config(test_file)
        print(f"\n✅ Save/Load test successful!")
        os.remove(test_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print(f"\n✅ Configuration system ready!")


#!/usr/bin/env python3
"""
YOLO Tracking 설정 파일
- YOLO 기반 레이저 추적
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json


@dataclass
class YOLOTrackingDataConfig:
    """YOLO Tracking 데이터 관련 설정"""
    
    # 파일 경로
    bin_file_path: str = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    
    # 센서 및 ROI 설정
    sensor_size: Tuple[int, int] = (720, 960)  # (height, width)
    roi_center: Tuple[int, int] = (541, 360)   # ROI 고정 중심 좌표
    roi_size: Tuple[int, int] = (512, 512)     # ROI 크기 (height, width)
    
    # 레이저 스팟 크기
    laser_diameter: int = 400  # 픽셀
    
    # Brownian Motion 설정
    motion_std: float = 2.0
    motion_boundary_margin: int = 80
    use_boundary_reflection: bool = True
    
    # Temporal 설정
    num_temporal_frames: int = 5
    
    # 데이터 처리
    max_frames: Optional[int] = None
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class YOLOTrackingModelConfig:
    """YOLO Tracking 모델 관련 설정"""
    
    model_name: str = "yolo_tiny_tracking"
    input_channels: int = 5
    num_classes: int = 1  # 레이저 스팟
    num_anchors: int = 3
    
    # Anchor boxes (정규화된 크기)
    anchors: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.78, 0.78),  # 레이저 직경 400/512
        (0.5, 0.5),
        (1.0, 1.0)
    ])


@dataclass
class YOLOTrainingConfig:
    """YOLO 훈련 관련 설정"""
    
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.001
    
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    
    use_scheduler: bool = True
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    
    patience: int = 15
    
    # YOLO Loss weights
    lambda_coord: float = 5.0
    lambda_obj: float = 1.0
    lambda_noobj: float = 0.1


@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    
    output_dir: str = "outputs_yolo_tracking"
    checkpoint_dir: str = "checkpoints_yolo_tracking"
    log_dir: str = "logs_yolo_tracking"
    
    save_best: bool = True
    save_freq: int = 10
    verbose: bool = True


@dataclass
class YOLOTrackingExperimentConfig:
    """YOLO Tracking 전체 실험 설정"""
    
    experiment_name: str = "yolo_tracking_experiment"
    description: str = "YOLO-based laser tracking with fixed ROI"
    
    data: YOLOTrackingDataConfig = field(default_factory=YOLOTrackingDataConfig)
    model: YOLOTrackingModelConfig = field(default_factory=YOLOTrackingModelConfig)
    training: YOLOTrainingConfig = field(default_factory=YOLOTrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__
        }
    
    def save_config(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 YOLO Tracking config saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'YOLOTrackingExperimentConfig':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.experiment_name = config_dict.get('experiment_name', config.experiment_name)
        config.description = config_dict.get('description', config.description)
        
        for section in ['data', 'model', 'training', 'system']:
            if section in config_dict:
                section_obj = getattr(config, section)
                for key, value in config_dict[section].items():
                    if hasattr(section_obj, key):
                        if isinstance(getattr(section_obj, key), tuple):
                            setattr(section_obj, key, tuple(value))
                        else:
                            setattr(section_obj, key, value)
        
        return config


def get_quick_test_config() -> YOLOTrackingExperimentConfig:
    """빠른 테스트용"""
    config = YOLOTrackingExperimentConfig()
    config.data.max_frames = 100
    config.training.num_epochs = 20
    config.training.batch_size = 4
    config.experiment_name = "yolo_tracking_quick"
    return config


def get_standard_config() -> YOLOTrackingExperimentConfig:
    """표준 설정"""
    config = YOLOTrackingExperimentConfig()
    config.data.max_frames = 500
    config.training.num_epochs = 100
    config.training.batch_size = 4
    config.experiment_name = "yolo_tracking_standard"
    return config


if __name__ == "__main__":
    print("⚙️ YOLO Tracking Configuration Test")
    config = YOLOTrackingExperimentConfig()
    print(f"✅ Config: {config.experiment_name}")
    print(f"   ROI: {config.data.roi_center}, size: {config.data.roi_size}")
    print(f"   Laser diameter: {config.data.laser_diameter}")


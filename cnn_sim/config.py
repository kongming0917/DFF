#!/usr/bin/env python3
"""
DVS Fixed GT CNN 프로젝트 설정 파일 - Fixed GT 방식만 지원
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union


@dataclass 
class FixedGTDataConfig:
    """Fixed GT 데이터 관련 설정"""
    
    # 파일 경로
    bin_file_path: str = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    
    # Fixed GT 설정
    true_center_coord: Tuple[int, int] = (541, 360)  # 고정 중심 좌표
    roi_size: Tuple[int, int] = (512, 512)           # ROI 크기 (height, width)
    
    # 데이터 처리
    max_events: Optional[int] = None
    frame_duration: int = 10000  # microseconds
    sensor_size: Tuple[int, int] = (720, 960)  # (height, width)
    
    # 데이터 증강
    shift_range: Tuple[int, int] = (-20, 20)
    noise_injection_probability: float = 0.4
    intensity_jitter_probability: float = 0.3
    
    # 데이터 분할
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    
    # 모델 아키텍처
    model_name: str = "basic"  # basic, mobilenet_v2, mobilenet_v2_light
    input_channels: int = 1
    output_dim: int = 2


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    
    # 기본 훈련 설정
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    
    # 옵티마이저
    optimizer: str = "adam"  # adam, sgd, adamw
    weight_decay: float = 1e-4
    
    # 스케줄러
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # plateau, step, cosine
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.0
    
    # 손실 함수
    loss_function: str = "mse"  # mse, l1, smooth_l1, huber


@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    
    # 하드웨어
    device: str = "auto"  # auto, cpu, cuda, cuda:0
    num_workers: int = 0  # Fixed GT는 빠르므로 0으로 설정
    pin_memory: bool = True
    
    # 디렉토리
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 저장 설정
    save_best: bool = True
    save_last: bool = True
    save_freq: int = 10  # 몇 에폭마다 체크포인트 저장
    
    # 로깅
    verbose: bool = True
    log_level: str = "INFO"
    
    # 재현성
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class InferenceConfig:
    """추론 관련 설정"""
    
    # 모델
    checkpoint_path: str = ""
    
    # 데이터
    test_data_path: str = ""
    max_test_events: Optional[int] = None
    
    # 배치 처리
    batch_size: int = 64
    
    # 출력
    save_results: bool = True
    visualize_results: bool = True
    output_format: str = "csv"  # csv, json, npz
    
    # 성능 측정
    benchmark: bool = True
    benchmark_runs: int = 100


@dataclass
class FixedGTExperimentConfig:
    """Fixed GT 전체 실험 설정"""
    
    # 실험 정보
    experiment_name: str = "fixed_gt_experiment"
    description: str = "DVS laser center detection using Fixed GT CNN"
    
    # 설정 컴포넌트
    data: FixedGTDataConfig = field(default_factory=FixedGTDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        """설정 후처리"""
        # 디렉토리 생성
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
        
        # 체크포인트 경로 설정
        if not self.inference.checkpoint_path:
            self.inference.checkpoint_path = os.path.join(
                self.system.checkpoint_dir, 
                f"{self.model.model_name}_best.pth"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__,
            'inference': self.inference.__dict__
        }
    
    def save_config(self, filepath: str):
        """설정을 파일로 저장"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"💾 Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FixedGTExperimentConfig':
        """딕셔너리에서 설정 로드"""
        
        config = cls()
        config.experiment_name = config_dict.get('experiment_name', config.experiment_name)
        config.description = config_dict.get('description', config.description)
        
        # 각 컴포넌트 업데이트
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
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
        
        if 'inference' in config_dict:
            for key, value in config_dict['inference'].items():
                if hasattr(config.inference, key):
                    setattr(config.inference, key, value)
        
        return config
    
    @classmethod
    def load_config(cls, filepath: str) -> 'FixedGTExperimentConfig':
        """파일에서 설정 로드"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        print(f"📂 Configuration loaded from {filepath}")
        
        return config


# 학습 모드 설정 (train.py에서 사용)
def get_training_mode_configs() -> Dict[str, Dict[str, Any]]:
    """
    train.py에서 사용하는 학습 모드 설정들
    
    Returns:
        Dict[str, Dict]: 학습 모드별 설정 딕셔너리
    """
    return {
        "ultra_fast": {
            'model_name': 'basic',
            'max_frames': 80,         # 빠른 테스트 (80 프레임 → 76 샘플)
            'temporal_window': 5,
            'num_epochs': 20,
            'batch_size': 4,
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정 (레이저 직경 400px 고려)
            'shift_range_y': 50,
            'description': '빠른 테스트 (basic, 80 frames, 20 epochs)'
        },
        
        "single_frame": {
            'model_name': 'basic',
            'max_frames': 1000,        # 단일 프레임 테스트 (1000 프레임)
            'temporal_window': 1,     # 개별 프레임만 사용!
            'num_epochs': 30,
            'batch_size': 8,
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': '단일 프레임 (basic, 1000 frames, temporal=1)'
        },
        
        "standard": {
            'model_name': 'basic',
            'max_frames': 500,        
            'temporal_window': 5,
            'num_epochs': 50,
            'batch_size': 4,
            'patience': 15,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': '표준 학습 (basic, 500 frames, temporal=5, epochs=50)'
        },
        
        "mobilenet_v2": {
            'model_name': 'mobilenet_v2',
            'max_frames': 300,        
            'temporal_window': 5,     # MobileNetV2도 temporal 데이터 사용
            'num_epochs': 40,
            'batch_size': 6,          # temporal 채널로 인해 배치 크기 조정
            'patience': 12,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': 'MobileNetV2 기반 학습 (300 frames, temporal=5, epochs=40)'
        },
        
        "mobilenet_v2_light": {
            'model_name': 'mobilenet_v2_light',
            'max_frames': 400,        
            'temporal_window': 5,     # 경량 모델도 temporal 데이터 사용
            'num_epochs': 35,
            'batch_size': 8,          # temporal 채널로 인해 배치 크기 조정
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': '경량 MobileNetV2 학습 (400 frames, temporal=5, epochs=35)'
        },
        
        "mobilenet_v2_single": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,        
            'temporal_window': 1,     # 단일 프레임 버전
            'num_epochs': 30,
            'batch_size': 12,         # 단일 프레임이므로 더 큰 배치 크기
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': 'MobileNetV2 단일 프레임 학습 (500 frames, temporal=1, epochs=30)'
        }
    }


# 설정 검증 함수
def validate_fixed_gt_config(config: FixedGTExperimentConfig) -> List[str]:
    """Fixed GT 설정 유효성 검사"""
    
    warnings = []
    
    # 데이터 검증
    if not os.path.exists(config.data.bin_file_path):
        warnings.append(f"Data file not found: {config.data.bin_file_path}")
    
    if config.data.train_ratio <= 0 or config.data.train_ratio >= 1:
        warnings.append(f"Invalid train_ratio: {config.data.train_ratio}")
    
    # ROI 크기 검증
    roi_h, roi_w = config.data.roi_size
    if roi_h <= 0 or roi_w <= 0:
        warnings.append(f"Invalid ROI size: {config.data.roi_size}")
    
    # 시프트 범위 검증
    shift_min, shift_max = config.data.shift_range
    if shift_min > shift_max:
        warnings.append(f"Invalid shift range: {config.data.shift_range}")
    
    # 모델 검증
    valid_models = ['basic', 'mobilenet_v2', 'mobilenet_v2_light']
    if config.model.model_name not in valid_models:
        warnings.append(f"Unknown model: {config.model.model_name}")
    
    # 훈련 검증
    if config.training.batch_size <= 0:
        warnings.append(f"Invalid batch_size: {config.training.batch_size}")
    
    if config.training.learning_rate <= 0:
        warnings.append(f"Invalid learning_rate: {config.training.learning_rate}")
    
    return warnings


if __name__ == "__main__":
    # Fixed GT 설정 시스템 테스트
    print("⚙️ Fixed GT Configuration System Test")
    print("=" * 50)
    
    # 기본 설정 테스트
    print("\n📋 Default Fixed GT Configuration:")
    config = FixedGTExperimentConfig()
    print(f"   Experiment: {config.experiment_name}")
    print(f"   Model: {config.model.model_name}")
    print(f"   ROI size: {config.data.roi_size}")
    print(f"   True center: {config.data.true_center_coord}")
    print(f"   Shift range: {config.data.shift_range}")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    
    # 설정 검증
    warnings = validate_fixed_gt_config(config)
    if warnings:
        print(f"\n⚠️ Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print(f"\n✅ Configuration is valid")
    
    # 설정 저장/로드 테스트
    print(f"\n💾 Save/Load Test:")
    test_file = "test_fixed_gt_config.json"
    
    try:
        config.save_config(test_file)
        loaded_config = FixedGTExperimentConfig.load_config(test_file)
        
        print(f"   Loaded experiment: {loaded_config.experiment_name}")
        print(f"   Loaded model: {loaded_config.model.model_name}")
        
        # 테스트 파일 삭제
        os.remove(test_file)
        print(f"   ✅ Save/Load test successful")
        
    except Exception as e:
        print(f"   ❌ Save/Load test failed: {e}")
    
    # 사용 가능한 모델 확인
    print(f"\n📦 Available Models:")
    valid_models = ['basic', 'mobilenet_v2', 'mobilenet_v2_light']
    for i, model in enumerate(valid_models, 1):
        print(f"   {i}. {model}")
    
    print(f"\n✅ Fixed GT configuration system test completed!")
    print(f"\n🎯 Key Features:")
    print(f"   ✅ Fixed GT 중심 좌표 설정")
    print(f"   ✅ ROI 기반 처리 설정")
    print(f"   ✅ 데이터 증강 파라미터")
    print(f"   ✅ 간소화된 설정 구조")
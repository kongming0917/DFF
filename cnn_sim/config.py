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
    true_center_coord: Tuple[int, int] = (480, 294)  # 고정 중심 좌표
    roi_size: Tuple[int, int] = (384, 384)           # ROI 크기 (height, width)
    
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
    model_name: str = "lightweight"  # basic, resnet, unet, lightweight, multiscale
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


# 사전 정의된 Fixed GT 설정들
def get_quick_test_config() -> FixedGTExperimentConfig:
    """빠른 테스트용 설정"""
    config = FixedGTExperimentConfig()
    
    # 데이터 설정
    config.data.max_events = 5000
    config.data.train_ratio = 0.8
    config.data.roi_size = (384, 384)
    config.data.shift_range = (-8, 8)
    
    # 모델 설정  
    config.model.model_name = "lightweight"
    
    # 훈련 설정
    config.training.num_epochs = 20
    config.training.batch_size = 16
    config.training.patience = 8
    
    # 시스템 설정
    config.system.num_workers = 0
    config.experiment_name = "quick_test_fixed_gt"
    
    return config


def get_lightweight_config() -> FixedGTExperimentConfig:
    """FPGA 배포용 경량화 모델 설정"""
    config = FixedGTExperimentConfig()
    
    # 데이터 설정
    config.data.roi_size = (384, 384)
    config.data.shift_range = (-10, 10)
    config.data.noise_injection_probability = 0.3
    config.data.intensity_jitter_probability = 0.2
    
    # 모델 설정
    config.model.model_name = "lightweight"
    
    # 훈련 설정
    config.training.batch_size = 64
    config.training.learning_rate = 0.002
    config.training.num_epochs = 100
    
    # 추론 설정
    config.inference.batch_size = 1  # 실시간 처리
    config.inference.benchmark = True
    
    config.experiment_name = "lightweight_fpga_fixed_gt"
    
    return config


def get_high_accuracy_config() -> FixedGTExperimentConfig:
    """높은 정확도를 위한 설정"""
    config = FixedGTExperimentConfig()
    
    # 데이터 설정
    config.data.max_events = None  # 모든 이벤트 사용
    config.data.roi_size = (512, 512)  # 더 큰 ROI
    config.data.shift_range = (-20, 20)  # 더 넓은 증강
    config.data.noise_injection_probability = 0.5
    config.data.intensity_jitter_probability = 0.4
    
    # 모델 설정
    config.model.model_name = "resnet"
    
    # 훈련 설정
    config.training.num_epochs = 200
    config.training.batch_size = 16
    config.training.learning_rate = 0.0005
    config.training.patience = 25
    
    config.experiment_name = "high_accuracy_fixed_gt"
    
    return config


def get_comparison_configs() -> List[FixedGTExperimentConfig]:
    """모델 비교용 설정들"""
    
    base_config = get_quick_test_config()
    configs = []
    
    model_names = ['basic', 'lightweight', 'resnet', 'multiscale']
    
    for model_name in model_names:
        config = FixedGTExperimentConfig()
        
        # 기본 설정 복사
        config.data = base_config.data
        config.training = base_config.training
        config.system = base_config.system
        config.inference = base_config.inference
        
        # 모델별 설정
        config.model.model_name = model_name
        config.experiment_name = f"comparison_fixed_gt_{model_name}"
        
        # 모델별 특화 설정
        if model_name == "lightweight":
            config.training.batch_size = 64
            config.training.learning_rate = 0.002
        elif model_name == "resnet":
            config.training.batch_size = 16
            config.training.learning_rate = 0.0005
        elif model_name == "multiscale":
            config.training.batch_size = 16
            config.training.learning_rate = 0.001
        
        configs.append(config)
    
    return configs


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
    valid_models = ['basic', 'resnet', 'unet', 'lightweight', 'multiscale']
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
    
    # 빠른 테스트 설정
    print(f"\n🚀 Quick Test Fixed GT Configuration:")
    quick_config = get_quick_test_config()
    print(f"   Model: {quick_config.model.model_name}")
    print(f"   Max events: {quick_config.data.max_events}")
    print(f"   ROI size: {quick_config.data.roi_size}")
    print(f"   Epochs: {quick_config.training.num_epochs}")
    print(f"   Batch size: {quick_config.training.batch_size}")
    
    # 경량화 설정
    print(f"\n💡 Lightweight FPGA Configuration:")
    fpga_config = get_lightweight_config()
    print(f"   Model: {fpga_config.model.model_name}")
    print(f"   ROI size: {fpga_config.data.roi_size}")
    print(f"   Learning rate: {fpga_config.training.learning_rate}")
    print(f"   Inference batch: {fpga_config.inference.batch_size}")
    
    # 설정 저장/로드 테스트
    print(f"\n💾 Save/Load Test:")
    test_file = "test_fixed_gt_config.json"
    
    try:
        quick_config.save_config(test_file)
        loaded_config = FixedGTExperimentConfig.load_config(test_file)
        
        print(f"   Loaded experiment: {loaded_config.experiment_name}")
        print(f"   Loaded model: {loaded_config.model.model_name}")
        
        # 테스트 파일 삭제
        os.remove(test_file)
        print(f"   ✅ Save/Load test successful")
        
    except Exception as e:
        print(f"   ❌ Save/Load test failed: {e}")
    
    # 비교 설정들
    print(f"\n🔍 Model Comparison Configurations:")
    comparison_configs = get_comparison_configs()
    for i, comp_config in enumerate(comparison_configs):
        print(f"   {i+1}. {comp_config.model.model_name} - "
              f"batch_size={comp_config.training.batch_size}, "
              f"lr={comp_config.training.learning_rate}")
    
    print(f"\n✅ Fixed GT configuration system test completed!")
    print(f"\n🎯 Key Features:")
    print(f"   ✅ Fixed GT 중심 좌표 설정")
    print(f"   ✅ ROI 기반 처리 설정")
    print(f"   ✅ 데이터 증강 파라미터")
    print(f"   ✅ FPGA 배포 최적화")
    print(f"   ✅ 간소화된 설정 구조")
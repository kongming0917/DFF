#!/usr/bin/env python3
"""
DVS Fixed GT CNN 프로젝트 설정 파일 - 학습 모드 설정
"""

from typing import Dict, Any

# 학습 모드 설정 (train.py에서 사용)
def get_training_mode_configs() -> Dict[str, Dict[str, Any]]:
    """
    train.py에서 사용하는 학습 모드 설정들
    
    Returns:
        Dict[str, Dict]: 학습 모드별 설정 딕셔너리
    """
    configs = {
        "mobilenet_v2": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,
            'temporal_window': 5,
            'num_epochs': 50,
            'batch_size': 4,
            'patience': 15,
            'roi_size': (512, 512),
            'description': 'MobileNetV2 기반 학습 (300 frames, temporal=5, epochs=40)'
        },
        "mobilenet_v2_single": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,
            'temporal_window': 1,
            'num_epochs': 30,
            'batch_size': 12,
            'patience': 8,
            'roi_size': (512, 512),
            'description': 'MobileNetV2 단일 프레임 학습 (500 frames, temporal=1, epochs=30)'
        },
        "mobileone_s0": {
            'model_name': 'mobileone_s0',
            'max_frames': 1000,
            'temporal_window': 5,
            'num_epochs': 50,
            'batch_size': 6,
            'patience': 12,
            'roi_size': (512, 512),
            'description': 'MobileOneS0 기반 학습 (300 frames, temporal=5, epochs=40)'
        }
    }
    
    # QAT 버전 설정 추가 (int8 양자화)
    qat_configs = {}
    for name, config in configs.items():
        qat_config = config.copy()
        qat_config['use_qat'] = True
        qat_config['description'] = config.get('description', '') + ' (QAT int8)'
        qat_configs[f"{name}_qat"] = qat_config
    
    # 기존 설정과 QAT 설정 병합
    configs.update(qat_configs)
    
    return configs
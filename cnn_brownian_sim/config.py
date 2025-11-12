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
    return {
        "ultra_fast": {
            'model_name': 'basic',
            'max_frames': 80,
            'temporal_window': 5,
            'num_epochs': 20,
            'batch_size': 4,
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,
            'shift_range_y': 50,
            'description': '빠른 테스트 (basic, 80 frames, 20 epochs)'
        },
        "single_frame": {
            'model_name': 'basic',
            'max_frames': 1000,
            'temporal_window': 1,
            'num_epochs': 30,
            'batch_size': 8,
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,
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
            'shift_range_x': 50,
            'shift_range_y': 50,
            'description': '표준 학습 (basic, 500 frames, temporal=5, epochs=50)'
        },
        "mobilenet_v2": {
            'model_name': 'mobilenet_v2',
            'max_frames': 300,
            'temporal_window': 5,
            'num_epochs': 40,
            'batch_size': 6,
            'patience': 12,
            'roi_size': (512, 512),
            'shift_range_x': 50,
            'shift_range_y': 50,
            'description': 'MobileNetV2 기반 학습 (300 frames, temporal=5, epochs=40)'
        },
        "mobilenet_v2_light": {
            'model_name': 'mobilenet_v2_light',
            'max_frames': 400,
            'temporal_window': 5,
            'num_epochs': 35,
            'batch_size': 8,
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,
            'shift_range_y': 50,
            'description': '경량 MobileNetV2 학습 (400 frames, temporal=5, epochs=35)'
        },
        "mobilenet_v2_single": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,
            'temporal_window': 1,
            'num_epochs': 30,
            'batch_size': 12,
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,
            'shift_range_y': 50,
            'description': 'MobileNetV2 단일 프레임 학습 (500 frames, temporal=1, epochs=30)'
        }
    }

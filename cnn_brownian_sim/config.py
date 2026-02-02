#!/usr/bin/env python3
"""
DVS Fixed GT CNN 프로젝트 설정 파일 - 학습 모드 설정
"""

from typing import Dict, Any

def _make_description(c: Dict[str, Any]) -> str:
    """설정 값으로부터 description 문자열 자동 생성 (하드코딩 방지)."""
    name = c['model_name']
    mf = c['max_frames']
    tw = c['temporal_window']
    ne = c['num_epochs']

    return f"{name} (max_frames={mf}, temporal window={tw}, epochs={ne})"


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
            'max_frames': 8000,
            'temporal_window': 5,
            'num_epochs': 30,
            'batch_size': 16,
            'patience': 10,
        },
        "mobilenet_v2_single": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,
            'temporal_window': 1,
            'num_epochs': 30,
            'batch_size': 12,
            'patience': 8,
        },
        "mobileone_s0": {
            'model_name': 'mobileone_s0',
            'max_frames': 8000,
            'temporal_window': 5,
            'num_epochs': 30,
            'batch_size': 16,
            'patience': 12,
        }
    }
    for c in configs.values():
        c['description'] = _make_description(c)
    return configs
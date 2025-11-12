#!/usr/bin/env python3
"""
YOLO 학습/평가 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def calculate_pixel_error(
    predictions: List[Tuple[float, float]], 
    targets: List[Tuple[float, float]],
    roi_size: int = 512
) -> np.ndarray:
    """
    픽셀 단위 오차 계산
    
    Args:
        predictions: [(x, y), ...] 정규화된 좌표 (0-1)
        targets: [(x, y), ...] 정규화된 좌표 (0-1)
        roi_size: ROI 크기
    
    Returns:
        pixel_errors: [N] 픽셀 단위 오차
    """
    pred_array = np.array(predictions)
    target_array = np.array(targets)
    
    # 픽셀 좌표로 변환
    pred_pixels = pred_array * roi_size
    target_pixels = target_array * roi_size
    
    # 유클리드 거리
    errors = np.sqrt(np.sum((pred_pixels - target_pixels) ** 2, axis=1))
    
    return errors


def calculate_accuracy(pixel_errors: np.ndarray, threshold: float = 5.0) -> float:
    """
    임계값 이내 정확도 계산
    
    Args:
        pixel_errors: 픽셀 오차 배열
        threshold: 임계값 (픽셀)
    
    Returns:
        accuracy: 정확도 (%)
    """
    return np.mean(pixel_errors <= threshold) * 100


def visualize_detection(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    center: Tuple[float, float],
    confidence: float,
    save_path: str = "detection_result.png"
):
    """
    단일 검출 결과 시각화
    
    Args:
        image: [H, W] ROI 이미지
        bbox: (x_center, y_center, width, height) 정규화된 좌표
        center: (x, y) 중심점
        confidence: confidence 점수
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 이미지 표시
    ax.imshow(image, cmap='gray')
    
    H, W = image.shape
    x_c, y_c, w, h = bbox
    
    # Bbox 그리기
    x1 = (x_c - w/2) * W
    y1 = (y_c - h/2) * H
    width = w * W
    height = h * H
    
    rect = plt.Rectangle((x1, y1), width, height, 
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    # 중심점 표시
    cx = center[0] * W
    cy = center[1] * H
    ax.plot(cx, cy, 'b+', markersize=20, markeredgewidth=3, label='Center')
    
    ax.set_title(f'Detection (Confidence: {confidence:.3f})')
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


class EarlyStopping:
    """Early Stopping"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss: float) -> bool:
        """
        Returns:
            True if should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


if __name__ == "__main__":
    print("🧪 Testing YOLO utilities")
    
    # 더미 데이터
    predictions = [(0.5, 0.5), (0.6, 0.4), (0.45, 0.55)]
    targets = [(0.52, 0.48), (0.58, 0.42), (0.44, 0.54)]
    
    # 픽셀 오차 계산
    errors = calculate_pixel_error(predictions, targets)
    print(f"Pixel errors: {errors}")
    
    acc_5 = calculate_accuracy(errors, threshold=5.0)
    print(f"Acc@5px: {acc_5:.1f}%")
    
    # Early stopping 테스트
    early_stopping = EarlyStopping(patience=3)
    losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
    
    for i, loss in enumerate(losses):
        if early_stopping(loss):
            print(f"Early stopping at epoch {i}")
            break
    
    print("\n✅ Utils test completed!")

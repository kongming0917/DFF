#!/usr/bin/env python3
"""
utils.py: DVS CNN 프로젝트를 위한 유틸리티 함수들
cnn_brownian_sim의 utils.py를 참고하여 작성
"""

import torch
import numpy as np
import os
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time


class EarlyStopping:
    """Early stopping 클래스"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0
        
    def __call__(self, val_loss: float) -> bool:
        """
        Args:
            val_loss: 현재 validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.verbose:
                print(f"   📉 Validation loss improved to {val_loss:.6f}")
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                print(f"   ⏳ No improvement for {self.wait}/{self.patience} epochs")
                
        return self.wait >= self.patience


class ModelCheckpoint:
    """모델 체크포인트 저장 클래스"""
    
    def __init__(self, save_dir: str, save_best: bool = True, save_last: bool = True):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric = float('inf')
        
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, model, optimizer, epoch: int, metric: float, filename: str = None) -> bool:
        """
        모델 체크포인트 저장
        
        Returns:
            bool: 최고 성능 모델인지 여부
        """
        is_best = metric < self.best_metric
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'best_metric': self.best_metric
        }
        
        # 최신 모델 저장
        if self.save_last:
            if filename is None:
                filename = 'last_checkpoint.pth'
            torch.save(checkpoint, os.path.join(self.save_dir, filename))
        
        # 최고 성능 모델 저장
        if is_best and self.save_best:
            self.best_metric = metric
            checkpoint['best_metric'] = metric
            
            # 모델명 추출 (파일명에서)
            if filename and '_epoch_' in filename:
                model_name = filename.split('_epoch_')[0]
                best_filename = f'{model_name}_best.pth'
            else:
                best_filename = 'best_model.pth'
                
            torch.save(checkpoint, os.path.join(self.save_dir, best_filename))
        
        return is_best


class MetricsTracker:
    """훈련 메트릭 추적 클래스"""
    
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """메트릭 업데이트"""
        # 훈련 메트릭
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.history[f'train_{key}'].append(value)
        
        # 검증 메트릭
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.history[f'val_{key}'].append(value)
        
        self.history['epoch'].append(epoch)
    
    def get_history(self) -> Dict[str, List]:
        """히스토리 반환"""
        return dict(self.history)
    
    def save_history(self, filepath: str):
        """히스토리를 JSON 파일로 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.get_history(), f, indent=2)
    
    def load_history(self, filepath: str):
        """JSON 파일에서 히스토리 로드"""
        with open(filepath, 'r') as f:
            self.history = defaultdict(list, json.load(f))


def visualize_predictions(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    title: str = "Prediction vs Target",
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """예측 결과 시각화"""
    
    if len(predictions) == 0:
        print("No predictions to visualize")
        return
    
    # 오차 계산
    errors = predictions - targets
    pixel_errors = np.sqrt(np.sum(errors**2, axis=1))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 산점도 - X 좌표
    axes[0,0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20)
    axes[0,0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                   [targets[:, 0].min(), targets[:, 0].max()], 'r--', alpha=0.8)
    axes[0,0].set_xlabel('True X')
    axes[0,0].set_ylabel('Predicted X')
    axes[0,0].set_title('X Coordinate Prediction')
    axes[0,0].grid(True, alpha=0.3)
    
    # R² 계산
    var_x = np.sum((targets[:, 0] - np.mean(targets[:, 0]))**2)
    if var_x > 1e-10:
        r2_x = 1 - np.sum((targets[:, 0] - predictions[:, 0])**2) / var_x
    else:
        r2_x = 1.0
    axes[0,0].text(0.05, 0.95, f'R² = {r2_x:.3f}', transform=axes[0,0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 산점도 - Y 좌표
    axes[0,1].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20)
    axes[0,1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                   [targets[:, 1].min(), targets[:, 1].max()], 'r--', alpha=0.8)
    axes[0,1].set_xlabel('True Y')
    axes[0,1].set_ylabel('Predicted Y')
    axes[0,1].set_title('Y Coordinate Prediction')
    axes[0,1].grid(True, alpha=0.3)
    
    # R² 계산
    var_y = np.sum((targets[:, 1] - np.mean(targets[:, 1]))**2)
    if var_y > 1e-10:
        r2_y = 1 - np.sum((targets[:, 1] - predictions[:, 1])**2) / var_y
    else:
        r2_y = 1.0
    axes[0,1].text(0.05, 0.95, f'R² = {r2_y:.3f}', transform=axes[0,1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 오차 히스토그램
    axes[1,0].hist(pixel_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(np.mean(pixel_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(pixel_errors):.4f}')
    axes[1,0].axvline(np.median(pixel_errors), color='blue', linestyle='--', 
                      label=f'Median: {np.median(pixel_errors):.4f}')
    axes[1,0].set_xlabel('Pixel Error')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Pixel Error Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 오차 산점도 (X vs Y 오차)
    scatter = axes[1,1].scatter(errors[:, 0], errors[:, 1], c=pixel_errors, 
                               alpha=0.6, s=20, cmap='viridis')
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('X Error')
    axes[1,1].set_ylabel('Y Error')
    axes[1,1].set_title('Error Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    # 컬러바 추가
    plt.colorbar(scatter, ax=axes[1,1], label='Pixel Error')
    
    # 전체 제목 및 통계
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    plt.suptitle(f'{title}\n'
                f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, '
                f'Mean Pixel Error: {np.mean(pixel_errors):.4f}±{np.std(pixel_errors):.4f}',
                fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_model_checkpoint(checkpoint_path: str, model, optimizer=None, device='cpu'):
    """체크포인트에서 모델 로드"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 보안 경고 억제 및 안전한 로딩
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드 (선택적)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 메타데이터 반환
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'metric': checkpoint.get('metric', 0),
        'best_metric': checkpoint.get('best_metric', 0)
    }
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Epoch: {metadata['epoch']}, Metric: {metadata['metric']:.6f}")
    
    return metadata


def setup_reproducibility(seed: int = 42):
    """재현 가능한 결과를 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # CUDA deterministic 설정 (성능 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to {seed}")


if __name__ == "__main__":
    # 유틸리티 함수들 테스트
    print("🧪 Testing utility functions")
    print("=" * 50)
    
    # 재현성 설정 테스트
    setup_reproducibility(42)
    
    # 더미 데이터로 시각화 테스트
    num_samples = 50
    true_centers = np.random.rand(num_samples, 2)
    pred_centers = true_centers + np.random.normal(0, 0.01, (num_samples, 2))
    
    print("\n📊 Testing visualization...")
    visualize_predictions(pred_centers, true_centers, 
                         title="Test Visualization", show_plot=False)
    
    # Early stopping 테스트
    print("\n⏹️ Testing early stopping...")
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    test_losses = [1.0, 0.8, 0.9, 0.7, 0.75, 0.76, 0.77, 0.78]
    for i, loss in enumerate(test_losses):
        should_stop = early_stopping(loss)
        print(f"   Epoch {i+1}: Loss = {loss:.3f}, Stop = {should_stop}")
        if should_stop:
            break
    
    print("\n✅ Utility functions test completed!")


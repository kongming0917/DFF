#!/usr/bin/env python3
"""
train.py: DVS 레이저 중심점 탐지 모델 훈련 스크립트
difflogic 모델 학습에 필수적인 tau 스케줄링과 정규화 로직을 포함합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 경로 설정
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 모듈 import
from model import LogicDVSNet, get_model
from dataset import DVSDataset, create_train_val_loaders
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker, visualize_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {device}")


class LogicDVSTrainer:
    """DVS 레이저 중심점 탐지 모델 훈련 클래스"""
    
    def __init__(
        self,
        model: LogicDVSNet,
        train_loader,
        val_loader,
        lr: float = 0.01,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        weight_decay: float = 0.002,
        save_dir: str = 'checkpoints',
        num_epochs: int = 100,
        patience: int = 15
    ):
        """
        Args:
            model: LogicDVSNet 모델
            train_loader: 훈련 데이터로더
            val_loader: 검증 데이터로더
            lr: 학습률
            tau_start: 초기 tau 값
            tau_end: 최종 tau 값
            weight_decay: 가중치 감쇠 (정규화)
            save_dir: 체크포인트 저장 디렉토리
            num_epochs: 에폭 수
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.result_dir = 'result'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 손실 함수: MSE Loss (좌표 회귀용)
        self.criterion = nn.MSELoss()
        
        # Optimizer: AdamW with weight decay (정규화)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )
        
        # 유틸리티 클래스들
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.checkpoint = ModelCheckpoint(save_dir, save_best=True)
        self.metrics = MetricsTracker()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 훈련"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        # Tau 스케줄링 (에폭마다 감소)
        current_tau = self.tau_start * ((self.tau_end / self.tau_start) ** (epoch / self.num_epochs))
        self.model.set_tau(current_tau)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} (tau={current_tau:.4f})")
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # 이진화 전처리 (LogicNet은 0/1 입력을 선호)
            # DVS 데이터가 실수라면 threshold 처리
            if imgs.max() > 1.0 or imgs.min() < 0.0:
                # 정규화 또는 이진화
                imgs = (imgs > imgs.mean()).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            
            # Loss 계산
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
            
            # 진행률 업데이트
            pbar.set_postfix({'loss': loss.item()})
        
        # 에폭 통계 계산
        avg_loss = total_loss / total_samples
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        # MAE 계산 (픽셀 단위로 변환 필요 시)
        mae = np.mean(np.abs(predictions_array - targets_array))
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'tau': current_tau
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """한 에폭 검증"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # 이진화 전처리
                if imgs.max() > 1.0 or imgs.min() < 0.0:
                    imgs = (imgs > imgs.mean()).float()
                
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # 검증 통계 계산
        avg_loss = total_loss / total_samples
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        # MAE 및 정확도 계산
        mae = np.mean(np.abs(predictions_array - targets_array))
        pixel_errors = np.sqrt(np.sum((predictions_array - targets_array)**2, axis=1))
        accuracy_5px = np.mean(pixel_errors <= 0.05) * 100  # 0.05는 정규화된 좌표 기준
        accuracy_10px = np.mean(pixel_errors <= 0.10) * 100
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'accuracy_5px': accuracy_5px,
            'accuracy_10px': accuracy_10px,
            'pixel_error_mean': np.mean(pixel_errors),
            'pixel_error_std': np.std(pixel_errors),
            'predictions': predictions,
            'targets': targets
        }
    
    def train(self):
        """전체 훈련 루프"""
        print(f"\n🚀 Starting training for {self.num_epochs} epochs")
        print("=" * 70)
        print(f"   Initial tau: {self.tau_start}, Final tau: {self.tau_end}")
        print(f"   Learning rate: {self.lr}")
        print("=" * 70)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            print(f"\n📈 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # 훈련
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate_epoch()
            
            # Learning rate 스케줄러 업데이트
            self.scheduler.step(val_metrics['loss'])
            
            # 메트릭 기록
            self.metrics.update(epoch, train_metrics, val_metrics)
            
            # 경과 시간 계산
            epoch_time = time.time() - start_time
            
            # 결과 출력
            print(f"   Train Loss: {train_metrics['loss']:.6f}, MAE: {train_metrics['mae']:.6f}")
            print(f"   Val Loss: {val_metrics['loss']:.6f}, MAE: {val_metrics['mae']:.6f}")
            print(f"   Val Acc@5px: {val_metrics['accuracy_5px']:.1f}%, Acc@10px: {val_metrics['accuracy_10px']:.1f}%")
            print(f"   Pixel Error: {val_metrics['pixel_error_mean']:.4f}±{val_metrics['pixel_error_std']:.4f}")
            print(f"   Time: {epoch_time:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Tau: {train_metrics['tau']:.4f}")
            
            # 체크포인트 저장
            is_best = self.checkpoint.save(
                self.model, self.optimizer, epoch, val_metrics['loss'],
                filename=f"logic_dvs_epoch_{epoch+1}.pth"
            )
            
            if is_best:
                print("   ⭐ New best model saved!")
            
            # Early stopping 체크
            if self.early_stopping(val_metrics['loss']):
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # 훈련 완료
        print(f"\n✅ Training completed!")
        print(f"   Best validation loss: {self.checkpoint.best_metric:.6f}")
        
        # 최종 결과 시각화
        self._finalize_training()
    
    def _finalize_training(self):
        """훈련 완료 후 최종 처리"""
        # 메트릭 히스토리 저장
        metrics_file = os.path.join(self.save_dir, 'metrics_history.json')
        self.metrics.save_history(metrics_file)
        print(f"📊 Metrics history saved to {metrics_file}")
        
        # 훈련 곡선 그리기
        self.plot_training_curves()
        
        # 최고 모델 로드하여 최종 검증
        best_model_path = os.path.join(self.save_dir, 'logic_dvs_best.pth')
        if os.path.exists(best_model_path):
            print(f"\n🔄 Loading best model for final evaluation...")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            
            final_metrics = self.validate_epoch()
            print(f"🎯 Final Results:")
            print(f"   Val Loss: {final_metrics['loss']:.6f}")
            print(f"   Val MAE: {final_metrics['mae']:.6f}")
            print(f"   Val Acc@5px: {final_metrics['accuracy_5px']:.1f}%")
            print(f"   Val Acc@10px: {final_metrics['accuracy_10px']:.1f}%")
            
            # 예측 시각화
            self.visualize_final_predictions(final_metrics)
    
    def plot_training_curves(self):
        """훈련 곡선 그래프"""
        history = self.metrics.get_history()
        
        if not history or not history.get('train_loss'):
            print("   ⚠️ No training history available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0,0].plot(history['train_loss'], label='Train')
            axes[0,0].plot(history['val_loss'], label='Validation')
            axes[0,0].set_title('Loss')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('MSE Loss')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # MAE
        if 'train_mae' in history and 'val_mae' in history:
            axes[0,1].plot(history['train_mae'], label='Train')
            axes[0,1].plot(history['val_mae'], label='Validation')
            axes[0,1].set_title('Mean Absolute Error')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('MAE')
            axes[0,1].legend()
            axes[0,1].grid(True)
        
        # Accuracy
        if 'val_accuracy_5px' in history and 'val_accuracy_10px' in history:
            axes[1,0].plot(history['val_accuracy_5px'], label='5px threshold')
            axes[1,0].plot(history['val_accuracy_10px'], label='10px threshold')
            axes[1,0].set_title('Validation Accuracy')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Accuracy (%)')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # Tau 스케줄링
        if 'train_tau' in history:
            axes[1,1].plot(history['train_tau'], label='Tau')
            axes[1,1].set_title('Temperature (Tau) Schedule')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Tau')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.suptitle('LogicDVSNet Training Curves')
        plt.tight_layout()
        
        plot_path = os.path.join(self.result_dir, 'logic_dvs_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to {plot_path}")
        plt.close()
    
    def visualize_final_predictions(self, metrics: Dict):
        """최종 예측 결과 시각화"""
        try:
            print(f"\n🎨 Creating prediction visualization...")
            
            predictions = np.array(metrics['predictions'])
            targets = np.array(metrics['targets'])
            
            save_path = os.path.join(self.result_dir, 'logic_dvs_predictions.png')
            visualize_predictions(predictions, targets, 
                                 title="LogicDVSNet Predictions",
                                 save_path=save_path, show_plot=False)
            
            print(f"   ✅ Prediction visualization saved successfully!")
            
        except Exception as e:
            print(f"   ❌ Error in visualize_final_predictions: {e}")
            import traceback
            traceback.print_exc()


def train_model(
    train_loader,
    val_loader,
    input_channels: int = 1,
    num_neurons: int = 64,
    output_dim: int = 2,
    lr: float = 0.01,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    num_epochs: int = 100,
    save_dir: str = 'checkpoints',
    **kwargs
):
    """
    모델 훈련 함수
    
    Args:
        train_loader: 훈련 데이터로더
        val_loader: 검증 데이터로더
        input_channels: 입력 채널 수
        num_neurons: 기본 뉴런 수
        output_dim: 출력 차원
        lr: 학습률
        tau_start: 초기 tau 값
        tau_end: 최종 tau 값
        num_epochs: 에폭 수
        save_dir: 체크포인트 저장 디렉토리
        **kwargs: 추가 파라미터
    """
    # 모델 생성
    model = get_model(
        input_channels=input_channels,
        num_neurons=num_neurons,
        output_dim=output_dim,
        **kwargs
    )
    
    # 모델 정보 출력
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 트레이너 생성 및 훈련
    trainer = LogicDVSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        tau_start=tau_start,
        tau_end=tau_end,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # 예제 사용법
    print("🎯 LogicDVSNet Training Example")
    print("=" * 60)
    
    # 데이터셋 설정 (실제 데이터셋으로 교체 필요)
    # train_loader, val_loader = create_train_val_loaders(...)
    
    # 훈련 실행
    # trainer = train_model(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     input_channels=1,
    #     num_neurons=32,
    #     output_dim=2,
    #     lr=0.01,
    #     tau_start=1.0,
    #     tau_end=0.1,
    #     num_epochs=100
    # )
    
    print("\n✅ Training script ready!")
    print("   Please configure your dataset and run training.")


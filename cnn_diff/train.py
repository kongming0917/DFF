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

import matplotlib.pyplot as plt

# 경로 설정 (현재 파일 기준 상위 2단계 폴더를 path에 추가)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 모듈 import
try:
    from model import LogicDVSNet, get_model
    from dataset import DVSDataset, create_train_val_loaders, load_individual_frames_from_bin
    from utils import ModelCheckpoint, MetricsTracker, visualize_predictions
except ImportError as e:
    print(f"[Error] 모듈 import 실패: {e}")
    print("model.py, dataset.py, utils.py가 올바른 위치에 있는지 확인해주세요.")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {device}")

class DualLogger(object):
    def __init__(self, filename):
        self.filename = filename
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


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
        result_dir: str = 'result',
        num_epochs: int = 100,
        patience: int = 15,
        run_name: str = 'exp'
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
            result_dir: 결과 이미지 저장 디렉토리
            num_epochs: 에폭 수
            patience: Early stopping patience (여기서는 scheduler용)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.result_dir = result_dir
        
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
            self.optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # 유틸리티 클래스들
        self.checkpoint = ModelCheckpoint(save_dir, save_best=True)
        self.metrics = MetricsTracker()
        self.run_name = run_name
    
    def get_tau(self, epoch):
        """Exponential Decay Tau Scheduling"""
        if epoch >= self.num_epochs:
            return self.tau_end
        progress = epoch / self.num_epochs
        return self.tau_start * ((self.tau_end / self.tau_start) ** progress)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 훈련"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Tau 스케줄링 (에폭마다 감소)
        current_tau = self.get_tau(epoch)
        if hasattr(self.model, 'set_tau'):
            self.model.set_tau(current_tau)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} (tau={current_tau:.4f})", leave=False)
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            
            # Loss 계산
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 통계 업데이트
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 진행률 업데이트
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # 에폭 통계 계산
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss,
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
            for imgs, labels in self.val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
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
        
        # MAE 및 Pixel Error 계산
        # targets가 정규화된 값(0~1)이라고 가정하고 512x512 픽셀 기준으로 변환
        mae = np.mean(np.abs(predictions_array - targets_array))
        
        # 픽셀 에러 (H, W가 512이라고 가정, 실제 데이터셋에 맞게 수정 필요)
        H, W = 512, 512 
        pred_px = predictions_array * [W, H]
        target_px = targets_array * [W, H]
        pixel_errors = np.sqrt(np.sum((pred_px - target_px)**2, axis=1))
        
        accuracy_5px = np.mean(pixel_errors <= 5.0) * 100
        accuracy_10px = np.mean(pixel_errors <= 10.0) * 100
        
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
        
        start_total_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate_epoch()
            
            # Learning rate 스케줄러 업데이트
            self.scheduler.step(val_metrics['loss'])
            
            # 메트릭 기록 (utils.MetricTracker 사용)
            # 여기서는 편의상 dictionary 합쳐서 전달
            train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # 인자를 2개로 나누어서 전달
            self.metrics.update(epoch,train_metrics, val_metrics)
            
            # 경과 시간
            epoch_time = time.time() - epoch_start_time
            
            # 결과 출력
            print(f"Epoch {epoch+1:03d} | Time: {epoch_time:.1f}s | Tau: {train_metrics['tau']:.4f} | LR: {train_metrics['lr']:.1e}")
            print(f"   Train Loss: {train_metrics['loss']:.6f}")
            print(f"   Val Loss: {val_metrics['loss']:.6f} | MAE: {val_metrics['mae']:.6f} | Err: {val_metrics['pixel_error_mean']:.2f}px")
            print(f"   Acc@5px: {val_metrics['accuracy_5px']:.1f}% | Acc@10px: {val_metrics['accuracy_10px']:.1f}%")
            
            # 체크포인트 저장
            is_best = self.checkpoint.save(
                self.model, self.optimizer, epoch, val_metrics['loss'],
                filename=f"{self.run_name}_{epoch+1}.pth"
            )
            
            if is_best:
                print("   ⭐ New best model saved!")
                # Best model일 때만 예측 시각화 저장 (선택 사항)
                # self.visualize_final_predictions(val_metrics, suffix=f"_epoch_{epoch+1}")
        
        total_time = time.time() - start_total_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes!")
        print(f"   Best validation loss: {self.checkpoint.best_metric:.6f}")
        
        # 최종 결과 처리
        self._finalize_training()
    
    def _finalize_training(self):
        """훈련 완료 후 최종 처리"""
        # 메트릭 히스토리 저장
        metrics_file = os.path.join(self.result_dir, 'metrics_history.json')
        self.metrics.save_history(metrics_file)
        print(f"📊 Metrics history saved to {metrics_file}")
        
        # 훈련 곡선 그리기
        self.plot_training_curves()
        
        # 최고 모델 로드하여 최종 검증 및 시각화
        best_model_path = os.path.join(self.save_dir, 'best_model.pth') # ModelCheckpoint 기본 이름 가정
        if not os.path.exists(best_model_path):
             # 파일명이 다를 수 있으니 디렉토리에서 가장 최근 best 찾거나 고정 이름 사용
             best_model_path = os.path.join(self.save_dir, 'logic_dvs_best.pth') # 우리가 저장한 이름

        if os.path.exists(best_model_path):
            print(f"\n🔄 Loading best model for final visualization...")
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(device)
            
            final_metrics = self.validate_epoch()
            self.visualize_final_predictions(final_metrics)
    
    def plot_training_curves(self):
        """훈련 곡선 그래프"""
        history = self.metrics.get_history() # returns dict of list
        if not history: return

        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Loss
        axes[0,0].plot(epochs, history['train_loss'], label='Train')
        if 'val_loss' in history:
            axes[0,0].plot(epochs, history['val_loss'], label='Validation')
        axes[0,0].set_title('MSE Loss')
        axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
        axes[0,0].legend(); axes[0,0].grid(True)
        
        # 2. Pixel Error
        if 'val_pixel_error_mean' in history:
            axes[0,1].plot(epochs, history['val_pixel_error_mean'], label='Val Pixel Error', color='orange')
            axes[0,1].set_title('Mean Pixel Error')
            axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Error (px)')
            axes[0,1].legend(); axes[0,1].grid(True)
            
        # 3. Accuracy
        if 'val_accuracy_5px' in history:
            axes[1,0].plot(epochs, history['val_accuracy_5px'], label='Acc@5px')
        if 'val_accuracy_10px' in history:
            axes[1,0].plot(epochs, history['val_accuracy_10px'], label='Acc@10px')
        axes[1,0].set_title('Validation Accuracy')
        axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Accuracy (%)')
        axes[1,0].legend(); axes[1,0].grid(True)
        
        # 4. Tau & LR
        ax4 = axes[1,1]
        if 'train_tau' in history:
            lns1 = ax4.plot(epochs, history['train_tau'], label='Tau', color='green')
            ax4.set_ylabel('Tau', color='green')
            ax4.tick_params(axis='y', labelcolor='green')
        
        if 'lr' in history:
            ax4_r = ax4.twinx()
            lns2 = ax4_r.plot(epochs, history['lr'], label='LR', color='red', linestyle='--')
            ax4_r.set_ylabel('Learning Rate', color='red')
            ax4_r.tick_params(axis='y', labelcolor='red')
            ax4_r.set_yscale('log')
        
        axes[1,1].set_title('Parameters Schedule')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].grid(True)
        
        plt.suptitle('LogicDVSNet Training Results', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.result_dir, '{self.run_name}_curves.png')
        plt.savefig(plot_path, dpi=300)
        print(f"📈 Training curves saved to {plot_path}")
        plt.close()
    
    def visualize_final_predictions(self, metrics: Dict, suffix=''):
        """최종 예측 결과 시각화 (Scatter Plot & Sample Images)"""
        try:
            print(f"\n🎨 Creating prediction visualization...")
            
            predictions = np.array(metrics['predictions'])
            targets = np.array(metrics['targets'])
            
            # 1. Scatter Plot (GT vs Pred)
            plt.figure(figsize=(8, 8))
            plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.5, label='X coord', s=10)
            plt.scatter(targets[:, 1], predictions[:, 1], alpha=0.5, label='Y coord', s=10)
            plt.plot([0, 1], [0, 1], 'r--', label='Ideal')
            plt.xlabel('Ground Truth')
            plt.ylabel('Prediction')
            plt.title('Prediction vs Ground Truth')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            scatter_path = os.path.join(self.result_dir, f'{self.run_name}_scatter.png')
            plt.savefig(scatter_path)
            plt.close()
            
            # 2. utils.visualize_predictions 사용 (이미지 위에 점 찍기)
            # 만약 utils에 해당 함수가 이미지까지 처리한다면 호출
            # 여기서는 간단히 경로만 지정해서 호출
            save_path = os.path.join(self.result_dir, f'{self.run_name}_predictions.png')
            visualize_predictions(predictions, targets, title="LogicDVSNet Samples", save_path=save_path, show_plot=False)
            
            print(f"   ✅ Visualization saved: {scatter_path}, {save_path}")
            
        except Exception as e:
            print(f"   ❌ Error in visualization: {e}")


def main():
    # --- 설정 ---
    BATCH_SIZE = 4
    INPUT_CHANNELS = 5
    NUM_NEURONS = 32
    NUM_EPOCHS = 10
    LR = 0.01
    MAX_FRAMES = 100
    
    BIN_FILE = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512.bin"
    CSV_LABEL = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512_labels.csv"
    
    
    print("🎯 LogicDVSNet Training Start")
    print("=" * 60)
    
    # 1. 데이터셋 로드
    try:
        if not os.path.exists(BIN_FILE) or not os.path.exists(CSV_LABEL):
            raise FileNotFoundError(f"Dataset file not found: {BIN_FILE}, {CSV_LABEL}")
        
        individual_frames = load_individual_frames_from_bin(BIN_FILE, MAX_FRAMES)
        train_loader, val_loader = create_train_val_loaders(
            individual_frames=individual_frames,
            csv_labels_path=CSV_LABEL,
            train_ratio=0.8,
            batch_size=BATCH_SIZE,
            num_workers=4,
            temporal_window=INPUT_CHANNELS,
            roi_size=(512, 512)
        )

        
    except Exception as e:
        print(f"❌ Dataset load failed: {e}")
        return

    # 2. 모델 생성
    model = get_model(
        input_channels=INPUT_CHANNELS,
        num_neurons=NUM_NEURONS,
        output_dim=2,
        tau=1.0 # 초기값
    )
    
    # 3. Trainer 실행
    trainer = LogicDVSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LR,
        tau_start=1.0,
        tau_end=1.0,
        num_epochs=NUM_EPOCHS,
        save_dir='checkpoints',
        result_dir='result'
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
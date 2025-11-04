#!/usr/bin/env python3
"""
DVS Laser Tracking 모델 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from typing import Dict
from datetime import datetime
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import TrackingExperimentConfig, get_quick_test_config, get_standard_config
from dataset import create_train_val_loaders, load_frames_from_bin
from model import get_tracking_model, count_parameters


class TrackingTrainer:
    """Laser Tracking 모델 훈련 클래스"""
    
    def __init__(self, config: TrackingExperimentConfig):
        self.config = config
        
        # Device 설정
        if config.system.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.system.device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 생성 - 모델 타입에 따라 파라미터 선택
        model_kwargs = {
            'input_channels': config.data.num_temporal_frames,
            'output_dim': config.model.output_dim
        }
        
        # LSTM/Transformer 모델인 경우에만 추가 파라미터 전달
        if 'lstm' in config.model.model_name.lower():
            model_kwargs['lstm_hidden_size'] = config.model.lstm_hidden_size
            model_kwargs['lstm_num_layers'] = config.model.lstm_num_layers
        
        self.model = get_tracking_model(
            model_name=config.model.model_name,
            **model_kwargs
        )
        self.model.to(self.device)
        
        # 모델 정보
        params = count_parameters(self.model)
        print(f"📊 Model: {config.model.model_name}")
        print(f"   Parameters: {params['total']:,}")
        
        # 손실 함수
        if config.training.loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif config.training.loss_function == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # 옵티마이저
        if config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        elif config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        
        # 스케줄러
        if config.training.use_scheduler:
            if config.training.scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=config.training.scheduler_factor,
                    patience=config.training.scheduler_patience,
                    verbose=True
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pixel_error': [],
            'val_acc_5px': [],
            'val_acc_10px': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 데이터로더
        self.train_loader = None
        self.val_loader = None
    
    def setup_data(self, frames):
        """데이터로더 설정"""
        print("📊 Setting up data loaders...")
        
        self.train_loader, self.val_loader = create_train_val_loaders(
            individual_frames=frames,
            config=self.config.data,
            train_ratio=self.config.data.train_ratio,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.system.num_workers
        )
        
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> float:
        """한 에폭 훈련"""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 픽셀 오차 계산 (ROI 크기 기준)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        roi_h, roi_w = self.config.data.roi_size
        pred_pixels = predictions * np.array([roi_w, roi_h])
        target_pixels = targets * np.array([roi_w, roi_h])
        
        pixel_errors = np.sqrt(np.sum((pred_pixels - target_pixels)**2, axis=1))
        
        return {
            'loss': avg_loss,
            'pixel_error_mean': np.mean(pixel_errors),
            'pixel_error_std': np.std(pixel_errors),
            'acc_5px': np.mean(pixel_errors <= 5.0) * 100,
            'acc_10px': np.mean(pixel_errors <= 10.0) * 100,
            'predictions': predictions,
            'targets': targets
        }
    
    def train(self):
        """전체 훈련 루프"""
        print(f"\n🚀 Starting training for {self.config.training.num_epochs} epochs")
        print("=" * 70)
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            
            print(f"\n📈 Epoch {epoch+1}/{self.config.training.num_epochs}")
            print("-" * 50)
            
            # 훈련
            train_loss = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # 히스토리 기록
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_pixel_error'].append(val_metrics['pixel_error_mean'])
            self.history['val_acc_5px'].append(val_metrics['acc_5px'])
            self.history['val_acc_10px'].append(val_metrics['acc_10px'])
            
            # 결과 출력
            epoch_time = time.time() - start_time
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_metrics['loss']:.6f}")
            print(f"   Pixel Error: {val_metrics['pixel_error_mean']:.2f}±{val_metrics['pixel_error_std']:.2f}px")
            print(f"   Acc@5px: {val_metrics['acc_5px']:.1f}%, Acc@10px: {val_metrics['acc_10px']:.1f}%")
            print(f"   Time: {epoch_time:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Best model 저장
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best')
                print("   ⭐ New best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
            
            # 주기적 저장
            if (epoch + 1) % self.config.system.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
        
        print(f"\n✅ Training completed!")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        
        # 결과 시각화
        self.plot_training_curves()
    
    def save_checkpoint(self, name: str):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        path = os.path.join(
            self.config.system.checkpoint_dir,
            f"{self.config.experiment_name}_{name}.pth"
        )
        torch.save(checkpoint, path)
    
    def plot_training_curves(self):
        """훈련 곡선 그래프"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Pixel Error
        axes[0, 1].plot(self.history['val_pixel_error'])
        axes[0, 1].axhline(5.0, color='r', linestyle='--', label='5px')
        axes[0, 1].axhline(10.0, color='g', linestyle='--', label='10px')
        axes[0, 1].set_title('Pixel Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Pixel Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(self.history['val_acc_5px'], label='@5px')
        axes[1, 0].plot(self.history['val_acc_10px'], label='@10px')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss (Log scale)
        axes[1, 1].semilogy(self.history['train_loss'], label='Train')
        axes[1, 1].semilogy(self.history['val_loss'], label='Val')
        axes[1, 1].set_title('Loss (Log Scale)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(f'{self.config.experiment_name} Training Curves')
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.config.system.output_dir,
            f'{self.config.experiment_name}_training_curves.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to: {plot_path}")
        
        try:
            plt.show()
        except:
            plt.close()


def main():
    """메인 실행 함수"""
    print("🎯 DVS Laser Tracking Training")
    print("=" * 60)
    
    # 설정 선택
    print("\n📝 Select configuration:")
    print("1. Quick test (100 frames, 20 epochs)")
    print("2. Standard (500 frames, 100 epochs)")
    
    try:
        choice = input("Enter choice (1-2, default=1): ").strip() or "1"
        
        if choice == "1":
            config = get_quick_test_config()
        elif choice == "2":
            config = get_standard_config()
        else:
            config = get_quick_test_config()
    except:
        print("Using default quick test config")
        config = get_quick_test_config()
    
    print(f"\n✅ Using config: {config.experiment_name}")
    print(f"   ROI: {config.data.roi_center}, size: {config.data.roi_size}")
    print(f"   Motion std: {config.data.motion_std}")
    print(f"   Temporal frames: {config.data.num_temporal_frames}")
    print(f"   Model: {config.model.model_name}")
    
    # 데이터 로딩
    frames = load_frames_from_bin(
        config.data.bin_file_path,
        max_frames=config.data.max_frames
    )
    
    if len(frames) == 0:
        print("❌ No frames loaded!")
        return
    
    # 훈련
    trainer = TrackingTrainer(config)
    trainer.setup_data(frames)
    trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved to: {config.system.checkpoint_dir}/")


if __name__ == "__main__":
    main()


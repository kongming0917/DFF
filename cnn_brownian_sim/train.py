#!/usr/bin/env python3
"""
train.py: DVS 레이저 중심점 탐지 모델 훈련 스크립트
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
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')  # GUI 없는 환경에서만 Agg 사용
import matplotlib.pyplot as plt

# dvs_root 계산 및 sys.path 설정 (모듈 레벨에서 한 번만)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 공통 모듈 import
from lib.bin_processor import BinProcessor
from model import get_model, count_parameters, convert_to_quantized
from dataset import create_train_val_loaders, DVSBrownianDataset
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker, visualize_predictions
from config import get_training_mode_configs

torch.backends.quantized.engine = 'fbgemm'

class DVSBrownianTrainer:
    """DVS Brownian Motion 모델 훈련 클래스"""
    
    def __init__(
        self,
        model_name: str,
        individual_frames: List[np.ndarray],  # 개별 프레임
        csv_labels_path: str,  # CSV 레이블 파일 경로
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,  # 다중 채널 수
        device: str = 'auto',
        lr: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 100,
        patience: int = 15,
        save_dir: str = 'checkpoints',
        use_qat: bool = False,  # QAT 모드 (int8 양자화)
    ):
        self.model_name = model_name
        self.individual_frames = individual_frames
        self.csv_labels_path = csv_labels_path
        self.roi_size = roi_size
        self.temporal_window = temporal_window
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_qat = use_qat  # QAT 플래그 저장
        # 결과 저장 디렉토리 설정 (PNG 파일용)
        self.result_dir = 'result'
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 초기화 (다중 채널, QAT 옵션 추가)
        self.model = get_model(
            model_name, 
            input_channels=temporal_window, 
            output_dim=2,
            use_qat=use_qat,  # QAT 모드 전달
        )
        self.model.to(self.device)
        
        # 모델 정보 출력
        params = count_parameters(self.model)
        print(f"📊 Model: {model_name}")
        print(f"   Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
        if use_qat:
            print(f"   🔢 QAT Mode: Enabled (int8 quantization)")
        
        # 손실 함수 및 옵티마이저
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8)
        
        # 유틸리티 클래스들
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.checkpoint = ModelCheckpoint(save_dir, save_best=True)
        self.metrics = MetricsTracker()
        
        # 데이터로더
        self.train_loader = None
        self.val_loader = None
        
        # 데이터셋 참조 (finalize_training에서 사용)
        self.train_dataset = None
        self.val_dataset = None
        
    def setup_data(self, num_workers: int = 0, **dataset_kwargs):
        """Brownian Motion 데이터로더 설정"""
        print("📊 Setting up Brownian Motion data loaders...")
        
        # Brownian Motion 방식으로 데이터로더 생성 (CSV 레이블 사용)
        self.train_loader, self.val_loader = create_train_val_loaders(
            individual_frames=self.individual_frames,
            csv_labels_path=self.csv_labels_path,
            train_ratio=0.8,
            batch_size=self.batch_size,
            num_workers=num_workers,
            roi_size=self.roi_size,
            temporal_window=self.temporal_window,
            **dataset_kwargs
        )
        
        # 데이터셋 참조 저장 (finalize_training에서 사용)
        self.train_dataset = self.train_loader.dataset
        self.val_dataset = self.val_loader.dataset
        
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 훈련"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        for batch_idx, (roi_batch, label_batch) in enumerate(self.train_loader):
            roi_batch = roi_batch.to(self.device)      # (batch, 1, 64, 64)
            label_batch = label_batch.to(self.device)  # (batch, 2)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(roi_batch)
            loss = self.criterion(outputs, label_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item() * roi_batch.size(0)
            total_samples += roi_batch.size(0)
            
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(label_batch.detach().cpu().numpy())
            
            # 진행률 출력 (매 50 배치마다)
            if batch_idx % 50 == 0:
                current_loss = total_loss / total_samples
                print(f'   Batch {batch_idx}/{len(self.train_loader)}, Loss: {current_loss:.6f}')
        
        # 에폭 통계 계산
        avg_loss = total_loss / total_samples
        
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        roi_h, roi_w = self.roi_size
        pred_pixels = predictions_array * np.array([roi_w, roi_h])
        target_pixels = targets_array * np.array([roi_w, roi_h])
        
        mae = np.mean(np.abs(pred_pixels - target_pixels))
        
        return {
            'loss': avg_loss,
            'mae': mae
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """한 에폭 검증"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for roi_batch, label_batch in self.val_loader:
                roi_batch = roi_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                outputs = self.model(roi_batch)
                loss = self.criterion(outputs, label_batch)
                
                total_loss += loss.item() * roi_batch.size(0)
                total_samples += roi_batch.size(0)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(label_batch.cpu().numpy())
        
        # 검증 통계 계산
        avg_loss = total_loss / total_samples
        
        # 정확도 계산 (0-1 정규화된 좌표에서)
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        # ROI 크기로 다시 스케일링하여 픽셀 단위로 계산
        roi_h, roi_w = self.roi_size
        pred_pixels = predictions_array * np.array([roi_w, roi_h])
        target_pixels = targets_array * np.array([roi_w, roi_h])

        # pixel 단위로 계산
        mae = np.mean(np.abs(pred_pixels - target_pixels))
        
        pixel_errors = np.sqrt(np.sum((pred_pixels - target_pixels)**2, axis=1))
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
    
    def train(self, **dataset_kwargs):
        """전체 훈련 루프"""
        print(f"\n🚀 Start Traininging Pipeline for {self.num_epochs} epochs (QAT: {self.use_qat})")
        print("=" * 70)
        
        # ------------------------------------------------------------
        # [Phase 0] Data Setup
        # ------------------------------------------------------------
        # 데이터 설정
        self.setup_data(**dataset_kwargs)
        
        # Best model path
        if self.use_qat:
            standard_checkpoint_dir = self.save_dir.replace('_qat', '')
            best_fp32_path = os.path.join(standard_checkpoint_dir, f"{self.model_name}_best.pth")
        else:
            best_fp32_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
        
        skip_phase1 = False
        
        if self.use_qat and os.path.exists(best_fp32_path):
            print(f"   ℹ️ QAT mode: Best model already exists at {best_fp32_path}")
            print(f"   Skipping Phase 1 training...")
            
            # load best fp32 model
            print(f"   Loading best model from {best_fp32_path}...")
            checkpoint = torch.load(best_fp32_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ Best model loaded")
            skip_phase1 = True
        
        # ------------------------------------------------------------
        # [Phase 1] Standard Training (FP32)
        # ------------------------------------------------------------
        if not skip_phase1:
            print("\n" + "=" * 70)
            print("[Phase 1] Starting Phase 1 training")
            print("=" * 70)
            
            # 훈련 루프
            for epoch in range(self.num_epochs):
                start_time = time.time()
                
                print(f"\n📈 Epoch {epoch+1}/{self.num_epochs}")
                print("-" * 50)
                
                # 훈련
                train_metrics = self.train_epoch()
                # 검증
                val_metrics = self.validate_epoch()
                # 학습률 스케줄러 업데이트
                self.scheduler.step(val_metrics['loss'])
                # 메트릭 기록
                self.metrics.update(epoch, train_metrics, val_metrics)
                # 경과 시간 계산
                epoch_time = time.time() - start_time
                
                # 결과 출력
                print(f"   Train Loss: {train_metrics['loss']:.6f}, MAE: {train_metrics['mae']:.3f}")
                print(f"   Val Loss: {val_metrics['loss']:.6f}, MAE: {val_metrics['mae']:.3f}")
                print(f"   Val Acc@5px: {val_metrics['accuracy_5px']:.1f}%, Acc@10px: {val_metrics['accuracy_10px']:.1f}%")
                print(f"   Pixel Error: {val_metrics['pixel_error_mean']:.2f}±{val_metrics['pixel_error_std']:.2f}")
                print(f"   Time: {epoch_time:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # 체크포인트 저장
                is_best = self.checkpoint.save(
                    self.model, self.optimizer, epoch, val_metrics['loss'], 
                    filename=f"{self.model_name}_epoch_{epoch+1}.pth"
                )
                
                if is_best:
                    print("   ⭐ New best model saved!")
                
                # Early stopping 체크
                if self.early_stopping(val_metrics['loss']):
                    print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # ------------------------------------------------------------
        # [Phase 2] QAT Fine-tuning (conditional)
        # ------------------------------------------------------------
        if self.use_qat:
            print("\n" + "=" * 70)
            print("[Phase 2] Starting QAT Fine-tuning")
            print("=" * 70)
            
            # 1. Best FP32 model load
            # We already loaded the best FP32 model in Phase 0
            
            print("🚫 Disabling Dropout for QAT phase...")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                    print(f"   Layer '{name}': Dropout probability set to 0.0")
    
            
            # 2. MobileOne 구조 변환 (Multi -> Single)
            if hasattr(self.model, 'reparameterize'):
                print("🔄 Reparameterizing MobileOne model...")
                self.model.reparameterize()
                self.model.to(self.device)
                print("✅ Reparameterization complete")
                
            # 3. Prepare QAT model
            from model import prepare_qat_model
            self.model = prepare_qat_model(self.model)             
            self.model.to(self.device)
            
            # 4. Optimizer 및 Scheduler 재설정
            qat_lr = self.lr * 0.1
            self.optimizer =optim.Adam(self.model.parameters(), lr=qat_lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
            
            self.checkpoint.best_metric = float('inf')
            
            # 5. QAT Fine-tuning loop
            qat_epochs =30
            print(f" Fine-tuning QAT model for {qat_epochs} epochs ...")
        
            for epoch in range(qat_epochs):
                print(f"\n QAT Epoch {epoch+1}/{qat_epochs}")
                t_metrics = self.train_epoch()
                v_metrics = self.validate_epoch()              
                self.metrics.update(epoch, t_metrics, v_metrics)
                
                print(f"   QAT Train Loss: {t_metrics['loss']:.6f}, MAE: {t_metrics['mae']:.3f}")
                print(f"   QAT Val Loss: {v_metrics['loss']:.6f}, MAE: {v_metrics['mae']:.3f}")
                
                is_best = self.checkpoint.save(
                    self.model, self.optimizer, epoch, v_metrics['loss'], 
                    filename=f"{self.model_name}_qat_epoch_{epoch+1}.pth"
                )
                
                if is_best:
                    print("   ⭐ New best QAT model saved!")                
                print("-" * 50)
        
        
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
        # QAT 모드일 때는 qat_best.pth를 찾고, 없으면 일반 best.pth 사용
        if self.use_qat:
            target_model_name = f"{self.model_name}_qat_best.pth"
            best_model_path = os.path.join(self.save_dir, target_model_name)
        else:
            target_model_name = f"{self.model_name}_best.pth"
            best_model_path = os.path.join(self.save_dir, target_model_name)
            
        if os.path.exists(best_model_path):
            # 1. Load best model
            print(f"\n🔄 Loading best model for final evaluation...")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            
            # QAT 모드인 경우: 현재 모델이 이미 QAT 모델이므로 체크포인트에서 로드하지 않고 그대로 사용
            if self.use_qat:
                print("   ℹ️ QAT mode: Using current QAT model (checkpoint loading skipped)")
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if hasattr(self.model, 'reparameterize'):
                    print("🔄 Reparameterizing MobileOne model...")
                    self.model.reparameterize()
                    print("✅ Reparameterization complete")
            
            # QAT 모드인 경우 양자화된 모델로 변환하여 저장
            if self.use_qat:
                self.model.eval()
                self.model.to('cpu')
                quantized_model = convert_to_quantized(self.model)
                
                # 양자화된 모델 저장 (QAT 모델)
                quantized_path = os.path.join(self.save_dir, f"{self.model_name}_int8.pth")
                torch.save({
                    'model_state_dict': quantized_model.state_dict(),
                    'epoch': checkpoint.get('epoch', 0),
                    'val_loss': checkpoint.get('val_loss', float('inf')),
                    'quantized': True,
                    'model_name': self.model_name,
                    'input_channels': self.temporal_window,
                    'output_dim': 2
                }, quantized_path)
                print(f"✅ Quantized model (int8) saved to {quantized_path}")
                
                # 최종 검증은 QAT 모델로 수행 (양자화된 모델은 저장만)
                # 양자화된 모델은 일반 텐서로 실행할 수 없으므로 QAT 모델 사용
                self.model = quantized_model
                self.device = torch.device('cpu')
            
            self.model.to(self.device)

            # 최종 검증 (랜덤 shift + 시드 고정으로 재현성 확보)
            print(f"🔄 Running final validation with random shifts (seed fixed)...")
            # Subset 내부의 실제 데이터셋에 접근
            if hasattr(self.val_dataset, 'dataset'):
                self.val_dataset.dataset.set_training_mode(False)  # 검증 모드 (랜덤 shift + 시드)
            else:
                self.val_dataset.set_training_mode(False)  # 검증 모드 (랜덤 shift + 시드)
            final_metrics = self.validate_epoch()
            print(f"🎯 Final Results (random shifts with fixed seed):")
            print(f"   Val Loss: {final_metrics['loss']:.6f}")
            print(f"   Val MAE: {final_metrics['mae']:.3f}")
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
        
        # 사용 가능한 메트릭 확인
        available_metrics = list(history.keys())
        print(f"   📊 Available metrics: {available_metrics}")
        
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
        else:
            axes[0,0].text(0.5, 0.5, 'Loss data not available', ha='center', va='center')
            axes[0,0].set_title('Loss (No Data)')
        
        # MAE
        if 'train_mae' in history and 'val_mae' in history:
            axes[0,1].plot(history['train_mae'], label='Train')
            axes[0,1].plot(history['val_mae'], label='Validation')
            axes[0,1].set_title('Mean Absolute Error')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('MAE')
            axes[0,1].legend()
            axes[0,1].grid(True)
        else:
            axes[0,1].text(0.5, 0.5, 'MAE data not available', ha='center', va='center')
            axes[0,1].set_title('MAE (No Data)')
        
        # Accuracy
        if 'val_accuracy_5px' in history and 'val_accuracy_10px' in history:
            axes[1,0].plot(history['val_accuracy_5px'], label='5px threshold')
            axes[1,0].plot(history['val_accuracy_10px'], label='10px threshold')
            axes[1,0].set_title('Validation Accuracy')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Accuracy (%)')
            axes[1,0].legend()
            axes[1,0].grid(True)
        else:
            axes[1,0].text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')
            axes[1,0].set_title('Accuracy (No Data)')
        
        # Pixel Error
        if 'val_pixel_error_mean' in history:
            axes[1,1].plot(history['val_pixel_error_mean'], label='Mean pixel error')
            axes[1,1].set_title('Validation Pixel Error')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Pixel Error')
            axes[1,1].legend()
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Pixel Error data not available', ha='center', va='center')
            axes[1,1].set_title('Pixel Error (No Data)')
        
        plt.suptitle(f'{self.model_name} Fixed GT Training Curves')
        plt.tight_layout()
        
        # 저장 (result 디렉토리에 저장)
        if self.use_qat:
            plot_path = os.path.join(self.result_dir, f'{self.model_name}_qat_training_curves.png')
        else:
            plot_path = os.path.join(self.result_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to {plot_path}")
        plt.show()  # MobaXterm에서 GUI 창으로 표시
    
    def visualize_final_predictions(self, metrics: Dict):
        """최종 예측 결과 시각화"""
        try:
            print(f"\n🎨 Creating prediction visualization...")
            
            predictions = np.array(metrics['predictions'])
            targets = np.array(metrics['targets'])
            
            print(f"   📊 Predictions shape: {predictions.shape}")
            print(f"   📊 Targets shape: {targets.shape}")
            
            # 일부 샘플만 선택 (최대 20개)
            num_samples = min(20, len(predictions))
            indices = np.random.choice(len(predictions), num_samples, replace=False)
            
            sample_preds = predictions[indices]
            sample_targets = targets[indices]
            
            if self.use_qat:
                save_path = os.path.join(self.result_dir, f'{self.model_name}_qat_predictions.png')
            else:
                save_path = os.path.join(self.result_dir, f'{self.model_name}_predictions.png')
            print(f"   💾 Saving to: {save_path}")
            
            visualize_predictions(sample_preds, sample_targets, 
                                save_path=save_path)
            
            print(f"   ✅ Prediction visualization saved successfully!")
            
        except Exception as e:
            print(f"   ❌ Error in visualize_final_predictions: {e}")
            import traceback
            traceback.print_exc()


def load_individual_frames_from_bin(bin_file_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """DVS bin 파일에서 개별 프레임 로딩"""
    print(f"📖 Loading individual frames from {bin_file_path}")
    
    # DVS bin 파일이 존재하는지 확인
    if not os.path.exists(bin_file_path):
        print(f"❌ Bin file not found: {bin_file_path}")
        print("🔄 Using dummy frames for testing...")
        return _create_dummy_individual_frames(max_frames or 100)
    
    try:
        # 실제 DVS bin 파일 로딩
        # BinProcessor는 모듈 레벨에서 이미 import됨
        
        # BinProcessor 사용하여 프레임 로드 (Brownian motion: 512x512)
        processor = BinProcessor(512, 512, has_header=True)
        
        # 최대 프레임 수 제한 (메모리 보호)
        max_frames_limit = max_frames or 200  # 적절한 기본값
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames_limit)
        
        print(f"   Loaded {len(frames_data)} frames from bin file")
        
        # 프레임 데이터를 numpy 배열로 변환
        individual_frames = []
        for frame_idx, frame in enumerate(frames_data):
            # frame.raw_data를 float32로 변환하고 고정 스케일링 (2bit 데이터: 0,1,2 → 0.0,0.5,1.0)
            frame_array = frame.raw_data.astype(np.float32) / 2.0  # 고정 스케일링
            individual_frames.append(frame_array)
            
            if len(individual_frames) >= max_frames_limit:
                break
        
        print(f"   ✅ Converted {len(individual_frames)} individual frames")
        return individual_frames
        
    except Exception as e:
        print(f"⚠️ Error loading bin file: {e}")
        print("🔄 Falling back to dummy frames...")
        return _create_dummy_individual_frames(max_frames or 100)


def _create_dummy_individual_frames(num_frames: int, center: tuple = (480, 294)) -> List[np.ndarray]:
    """개별 timestamp 기반 더미 프레임 생성"""
    frames = []
    
    print(f"   Creating {num_frames} individual dummy frames centered at {center}")
    
    for i in range(num_frames):
        # 각 프레임은 하나의 timestamp를 나타냄
        frame = np.zeros((720, 960), dtype=np.float32)
        
        # 중심 주변에 가우시안 분포로 이벤트 생성 (프레임당 50-200개)
        events_per_frame = np.random.randint(50, 200)
        
        for _ in range(events_per_frame):
            x = int(np.random.normal(center[0], 25))  # 레이저 스팟 크기
            y = int(np.random.normal(center[1], 25))
            
            # 센서 범위 내로 제한
            x = max(0, min(959, x))
            y = max(0, min(719, y))
            
            # 강도 누적 (개별 timestamp에서의 이벤트 밀도)
            frame[y, x] += 1.0
        
        # 정규화
        if np.max(frame) > 0:
            frame = frame / np.max(frame)
        
        frames.append(frame)
    
    print(f"   ✅ Generated {len(frames)} individual frames")
    return frames


def train_brownian_model(
    config: Dict[str, Any],
    bin_file_path: str,
    csv_labels_path: str,
    lr: Optional[float] = None,
    use_qat: bool = False  # QAT 옵션 추가
):
    """Brownian Motion 방식으로 모델 훈련 (CSV 레이블 사용)
    
    Args:
        config: 훈련 설정 딕셔너리
        bin_file_path: bin 파일 경로
        csv_labels_path: CSV 레이블 파일 경로
        lr: 학습률 (옵션)
        use_qat: QAT 모드 사용 여부 (int8 양자화)
    """
    
    # 기본값 설정
    default_lr = 0.001
    
    # 전체 설정 병합
    full_config = {
        **config,
        'lr': lr if lr is not None else config.get('lr', default_lr),
        'save_dir': f"checkpoints_{config['model_name']}"
    }
    
    # QAT 모드인 경우 체크포인트 디렉토리 이름에 표시
    if use_qat:
        full_config['save_dir'] = f"checkpoints_{config['model_name']}_qat"
    
    # 개별 프레임 로드
    individual_frames = load_individual_frames_from_bin(
        bin_file_path, 
        max_frames=full_config.get('max_frames')
    )
    
    # 체크포인트 디렉토리 생성
    os.makedirs(full_config['save_dir'], exist_ok=True)
    
    # 훈련 설정 저장
    config_path = os.path.join(full_config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_name': config['model_name'],
            'bin_file_path': bin_file_path,
            'csv_labels_path': csv_labels_path,
            'training_config': full_config,
            'use_qat': use_qat,  # QAT 설정 저장
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # 트레이너 생성 및 훈련 (QAT 옵션 전달)
    trainer = DVSBrownianTrainer(
        model_name=config['model_name'],
        individual_frames=individual_frames,
        csv_labels_path=csv_labels_path,
        roi_size=full_config['roi_size'],
        lr=full_config['lr'],
        batch_size=full_config['batch_size'],
        num_epochs=full_config['num_epochs'],
        patience=full_config['patience'],
        save_dir=full_config['save_dir'],
        temporal_window=full_config['temporal_window'],
        use_qat=use_qat,  # QAT 옵션 추가
    )
    
    # 데이터셋 파라미터 (필요 없음, CSV에서 로드)
    dataset_params = {}
    
    trainer.train(**dataset_params)
    
    return trainer


if __name__ == "__main__":
    # Brownian Motion 데이터셋으로 학습
    print("🎯 Brownian Motion DVS 데이터로 학습")
    print("=" * 60)
    
    # Brownian Motion 데이터셋 파일 경로
    BIN_FILE_PATH = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512.bin"
    CSV_LABELS_PATH = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512_labels.csv"
    
    # 학습 하이퍼파라미터 설정
    LEARNING_RATE = 0.001  # 학습률
    # QAT 여부는 config.py의 설정에서 자동으로 결정됨
    
    # 파일 존재 확인
    if not os.path.exists(BIN_FILE_PATH):
        print(f"❌ DVS 데이터 파일을 찾을 수 없습니다: {BIN_FILE_PATH}")
        print("   Brownian motion 데이터셋을 먼저 생성해주세요: python generate_brownian_dataset.py")
        exit(1)
    else:
        print(f"✅ DVS 데이터 파일 확인: {BIN_FILE_PATH}")
    
    if not os.path.exists(CSV_LABELS_PATH):
        print(f"❌ CSV 레이블 파일을 찾을 수 없습니다: {CSV_LABELS_PATH}")
        print("   Brownian motion 데이터셋을 먼저 생성해주세요: python generate_brownian_dataset.py")
        exit(1)
    else:
        print(f"✅ CSV 레이블 파일 확인: {CSV_LABELS_PATH}")
    
    # 사용 가능한 학습 설정 목록 가져오기
    available_configs = get_training_mode_configs()
    
    # 설정 목록 표시 (모델 이름만)
    print(f"\n📋 사용 가능한 학습 설정:")
    print("=" * 60)
    config_list = list(available_configs.items())
    for idx, (config_name, config_data) in enumerate(config_list, 1):
        print(f"  {idx}. {config_name}")
    
    # 사용자 선택
    while True:
        try:
            choice = input(f"\n학습 설정을 선택하세요 (1-{len(config_list)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(config_list):
                selected_config_name, selected_config = config_list[choice_idx]
                break
            else:
                print(f"❌ 잘못된 선택입니다. 1-{len(config_list)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 취소했습니다.")
            exit(0)
    
    # 선택된 설정 사용
    config = selected_config.copy()
    # QAT 여부는 config에서 가져옴 (config.py의 설정 사용)
    use_qat = config.get('use_qat', False)
    
    print(f"\n✅ 선택된 설정: {selected_config_name}")
    print(f"   모델: {config['model_name']}")
    print(f"   최대 프레임: {config.get('max_frames', 'All') or 'All'}")
    print(f"   시간 윈도우: {config['temporal_window']}")
    print(f"   에폭 수: {config['num_epochs']}")
    print(f"   배치 크기: {config['batch_size']}")
    print(f"   ROI 크기: {config['roi_size'][0]}×{config['roi_size'][1]}")
    if use_qat:
        print(f"   🔢 QAT 모드: 활성화 (int8 양자화)")
    
    print(f"\n🚀 Brownian Motion 데이터셋으로 학습을 시작합니다...")
    
    try:
        # Brownian Motion 모델 학습 실행
        trainer = train_brownian_model(
            config=config,
            bin_file_path=BIN_FILE_PATH,
            csv_labels_path=CSV_LABELS_PATH,
            lr=LEARNING_RATE,
            use_qat=use_qat  # config.py의 설정 사용
        )
        
        print(f"\n✅ {config['model_name']} 모델 학습 완료!")
        if use_qat:
            print(f"📁 체크포인트 저장 위치: checkpoints_{config['model_name']}_qat/")
            print(f"🎯 최고 모델: checkpoints_{config['model_name']}_qat/{config['model_name']}_qat_best.pth")
            print(f"🔢 양자화된 모델 (int8): checkpoints_{config['model_name']}_qat/{config['model_name']}_qat.pth")
            print(f"📊 훈련 곡선: result/{config['model_name']}_qat_training_curves.png")
            print(f"🔍 예측 결과: result/{config['model_name']}_qat_predictions.png")
        else:
            print(f"📁 체크포인트 저장 위치: checkpoints_{config['model_name']}/")
            print(f"🎯 최고 모델: checkpoints_{config['model_name']}/{config['model_name']}_best.pth")
            print(f"📊 훈련 곡선: result/{config['model_name']}_training_curves.png")
            print(f"🔍 예측 결과: result/{config['model_name']}_predictions.png")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자가 학습을 중단했습니다.")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
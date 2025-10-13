#!/usr/bin/env python3
"""
DVS 레이저 중심점 탐지 모델 훈련 스크립트 - Fixed GT 방식만 지원
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')  # GUI 없는 환경에서만 Agg 사용
import matplotlib.pyplot as plt

from model import get_model, count_parameters
from dataset import create_train_val_loaders
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker, visualize_predictions


class DVSFixedGTTrainer:
    """DVS Fixed GT 모델 훈련 클래스"""
    
    def __init__(
        self,
        model_name: str,
        individual_frames: List[np.ndarray],  # 개별 프레임
        true_center_coord: Tuple[int, int] = (541, 360),  # 필터로 찾은 정확한 중심
        roi_size: Tuple[int, int] = (512, 512),
        temporal_window: int = 5,  # 다중 채널 수
        device: str = 'auto',
        lr: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 100,
        patience: int = 15,
        save_dir: str = 'checkpoints'
    ):
        self.model_name = model_name
        self.individual_frames = individual_frames
        self.true_center_coord = true_center_coord
        self.roi_size = roi_size
        self.temporal_window = temporal_window
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 초기화 (다중 채널)
        self.model = get_model(model_name, input_channels=temporal_window, output_dim=2)
        self.model.to(self.device)
        
        # 모델 정보 출력
        params = count_parameters(self.model)
        print(f"📊 Model: {model_name}")
        print(f"   Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
        
        # 손실 함수 및 옵티마이저
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )
        
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
        """Fixed GT 데이터로더 설정"""
        print("📊 Setting up Fixed GT data loaders...")
        
        # Fixed GT 방식으로 데이터로더 생성 (개별 프레임 기반)
        self.train_loader, self.val_loader = create_train_val_loaders(
            individual_frames=self.individual_frames,
            train_ratio=0.8,
            batch_size=self.batch_size,
            num_workers=num_workers,
            true_center_coord=self.true_center_coord,
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
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        
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
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        
        # 정확도 계산 (0-1 정규화된 좌표에서)
        predictions_array = np.array(predictions)
        targets_array = np.array(targets)
        
        # ROI 크기로 다시 스케일링하여 픽셀 단위로 계산
        roi_h, roi_w = self.roi_size
        pred_pixels = predictions_array * np.array([roi_w, roi_h])
        target_pixels = targets_array * np.array([roi_w, roi_h])
        
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
        print(f"\n🚀 Starting Fixed GT training for {self.num_epochs} epochs")
        print("=" * 70)
        
        # 데이터 설정
        self.setup_data(**dataset_kwargs)
        
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
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"\n🔄 Loading best model for final evaluation...")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])



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
        
        # 저장
        plot_path = os.path.join(self.save_dir, f'{self.model_name}_training_curves.png')
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
            
            save_path = os.path.join(self.save_dir, f'{self.model_name}_predictions.png')
            print(f"   💾 Saving to: {save_path}")
            
            visualize_predictions(sample_preds, sample_targets, 
                                save_path=save_path)
            
            print(f"   ✅ Prediction visualization saved successfully!")
            
        except Exception as e:
            print(f"   ❌ Error in visualize_final_predictions: {e}")
            import traceback
            traceback.print_exc()


def load_events_from_bin(bin_file_path: str, max_events: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """
    실제 DVS bin 파일에서 이벤트 리스트 로드
    """
    print(f"📖 Loading events from {bin_file_path}")
    
    # DVS bin 파일이 존재하는지 확인
    if not os.path.exists(bin_file_path):
        print(f"❌ Bin file not found: {bin_file_path}")
        print("🔄 Using dummy events for testing...")
        return _create_dummy_events(max_events or 10000)
    
    try:
        # 실제 DVS bin 파일 로딩
        import sys
        sys.path.append('/hai/home/jdj/dvs/filter_sim')
        from dvs_filter import BinProcessor
        
        # BinProcessor 사용하여 이벤트 로드 (메모리 최적화)
        processor = BinProcessor(960, 720, has_header=True)
        
        # 최대 프레임 수 제한 (메모리 보호)
        max_frames_limit = 1000 if max_events is None else min(1000, max_events // 50)
        frames = processor.read_frames(bin_file_path, max_frames=max_frames_limit)
        
        print(f"   Loaded {len(frames)} frames from bin file (limited for memory)")
        
        # 프레임에서 이벤트 추출 (효율적인 방식)
        events = []
        target_events = max_events or 50000  # 기본 최대값 설정
        
        for frame_idx, frame in enumerate(frames):
            if len(events) >= target_events:
                break
                
            # 프레임에서 이벤트 좌표 추출
            frame_data = frame.raw_data
            y_coords, x_coords = np.nonzero(frame_data)
            
            timestamp = frame.header.timestamp if hasattr(frame.header, 'timestamp') else frame_idx * 10000
            
            # 배치로 이벤트 추가 (더 효율적)
            batch_events = []
            for x, y in zip(x_coords, y_coords):
                if len(events) + len(batch_events) >= target_events:
                    break
                # 폴라리티는 실제 데이터에 따라 조정
                polarity = 1 if frame_data[y, x] == 1 else 0
                batch_events.append((int(x), int(y), int(timestamp), int(polarity)))
            
            events.extend(batch_events)
            
            # 메모리 사용량 모니터링
            if frame_idx % 100 == 0:
                print(f"   Progress: {frame_idx+1}/{len(frames)} frames, {len(events)} events extracted")
        
        print(f"   Extracted {len(events)} events from frames")
        return events
        
    except Exception as e:
        print(f"⚠️ Error loading bin file: {e}")
        print("🔄 Falling back to dummy events...")
        return _create_dummy_events(max_events or 10000)


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
        import sys
        sys.path.append('/hai/home/jdj/dvs/filter_sim')
        from dvs_filter import BinProcessor
        
        # BinProcessor 사용하여 프레임 로드
        processor = BinProcessor(960, 720, has_header=True)
        
        # 최대 프레임 수 제한 (메모리 보호)
        max_frames_limit = max_frames or 200  # 적절한 기본값
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames_limit)
        
        print(f"   Loaded {len(frames_data)} frames from bin file")
        
        # 프레임 데이터를 numpy 배열로 변환
        individual_frames = []
        for frame_idx, frame in enumerate(frames_data):
            # frame.raw_data를 float32로 변환하고 정규화
            frame_array = frame.raw_data.astype(np.float32)
            
            # 정규화 (0-1 범위)
            if np.max(frame_array) > 0:
                frame_array = frame_array / np.max(frame_array)
            
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


def train_fixed_gt_model(
    model_name: str,
    bin_file_path: str,
    config_overrides: Optional[Dict] = None
):
    """Fixed GT 방식으로 모델 훈련"""
    
    # 기본 설정 (간소화)
    default_config = {
        'lr': 0.001,
        'batch_size': 8,
        'num_epochs': 50,
        'patience': 10,
        'max_frames': 150,  # 개별 프레임 수
        'temporal_window': 5,  # 다중 채널 수
        'true_center_coord': (541, 360),  # 실제 빔 중심 좌표
        'roi_size': (512, 512),  # 빔 크기에 맞춘 ROI (512x512)
        'shift_range_x': 50,  # X축 시프트 범위 (±픽셀) - 안전한 범위로 조정
        'shift_range_y': 50,  # Y축 시프트 범위 (±픽셀) - 안전한 범위로 조정
        'save_dir': f'checkpoints_{model_name}'
    }
    
    # 설정 업데이트
    if config_overrides:
        default_config.update(config_overrides)
    
    # 개별 프레임 로드
    individual_frames = load_individual_frames_from_bin(
        bin_file_path, 
        max_frames=default_config['max_frames']
    )
    
    # 체크포인트 디렉토리 생성
    os.makedirs(default_config['save_dir'], exist_ok=True)
    
    # 훈련 설정 저장
    config_path = os.path.join(default_config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        config_to_save = {
            'model_name': model_name,
            'bin_file_path': bin_file_path,
            'training_config': default_config,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(config_to_save, f, indent=2)
    
    # 트레이너 생성 및 훈련
    trainer = DVSFixedGTTrainer(
        model_name=model_name,
        individual_frames=individual_frames,  # 개별 프레임 전달
        true_center_coord=(541, 360),  # 필터로 찾은 정확한 중심
        roi_size=default_config['roi_size'],
        lr=default_config['lr'],
        batch_size=default_config['batch_size'],
        num_epochs=default_config['num_epochs'],
        patience=default_config['patience'],
        save_dir=default_config['save_dir'],
        temporal_window=default_config['temporal_window']  # 다중 채널 수
    )
    
    # 데이터셋 파라미터 (X/Y별 시프트만 전달, 나머지는 트레이너에서 처리)
    dataset_params = {
        'shift_range_x': default_config['shift_range_x'],
        'shift_range_y': default_config['shift_range_y']
    }
    
    trainer.train(**dataset_params)
    
    return trainer


if __name__ == "__main__":
    # 실제 DVS Gaussian 데이터로 Fixed GT 학습
    print("🎯 실제 DVS Gaussian 데이터로 Fixed GT 학습")
    print("=" * 60)
    
    # 실제 DVS 데이터 파일 경로
    BIN_FILE_PATH = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    
    # 파일 존재 확인
    if not os.path.exists(BIN_FILE_PATH):
        print(f"❌ DVS 데이터 파일을 찾을 수 없습니다: {BIN_FILE_PATH}")
        print("   더미 데이터로 테스트를 진행합니다...")
    else:
        print(f"✅ DVS 데이터 파일 확인: {BIN_FILE_PATH}")
    
    # 학습 설정 선택 (basic 모델만)
    training_configs = {
        "ultra_fast": {
            'model_name': 'basic',
            'max_frames': 80,         # 빠른 테스트 (80 프레임 → 76 샘플)
            'temporal_window': 5,
            'num_epochs': 20,
            'batch_size': 4,
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정 (레이저 직경 400px 고려)
            'shift_range_y': 50,
            'description': '빠른 테스트 (basic, 80 frames, 20 epochs)'
        },
        
        "single_frame": {
            'model_name': 'basic',
            'max_frames': 1000,        # 단일 프레임 테스트 (1000 프레임)
            'temporal_window': 1,     # 개별 프레임만 사용!
            'num_epochs': 30,
            'batch_size': 8,
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
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
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': '표준 학습 (basic, 500 frames, temporal=5, epochs=50)'
        },
        
        "mobilenet_v2": {
            'model_name': 'mobilenet_v2',
            'max_frames': 300,        
            'temporal_window': 5,     # MobileNetV2도 temporal 데이터 사용
            'num_epochs': 40,
            'batch_size': 6,          # temporal 채널로 인해 배치 크기 조정
            'patience': 12,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': 'MobileNetV2 기반 학습 (300 frames, temporal=5, epochs=40)'
        },
        
        "mobilenet_v2_light": {
            'model_name': 'mobilenet_v2_light',
            'max_frames': 400,        
            'temporal_window': 5,     # 경량 모델도 temporal 데이터 사용
            'num_epochs': 35,
            'batch_size': 8,          # temporal 채널로 인해 배치 크기 조정
            'patience': 10,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': '경량 MobileNetV2 학습 (400 frames, temporal=5, epochs=35)'
        },
        
        "mobilenet_v2_single": {
            'model_name': 'mobilenet_v2',
            'max_frames': 500,        
            'temporal_window': 1,     # 단일 프레임 버전
            'num_epochs': 30,
            'batch_size': 12,         # 단일 프레임이므로 더 큰 배치 크기
            'patience': 8,
            'roi_size': (512, 512),
            'shift_range_x': 50,      # 안전한 범위로 조정
            'shift_range_y': 50,
            'description': 'MobileNetV2 단일 프레임 학습 (500 frames, temporal=1, epochs=30)'
        }
    }
    
    # 사용자 선택
    print("\n🔧 학습 모드를 선택하세요:")
    for i, (key, config) in enumerate(training_configs.items(), 1):
        print(f"{i}. {key} - {config['description']}")
    
    try:
        # 자동 실행을 위해 기본값 사용 (headless 환경 대응)
        import sys
        if not sys.stdin.isatty():  # 터미널이 아닌 환경
            choice = "2"
            print(f"\n자동 선택: 2 (single_frame)")
        else:
            choice = input(f"\n선택 (1-3, 기본값 2): ").strip() or "2"
        
        config_keys = list(training_configs.keys())
        selected_key = config_keys[int(choice) - 1]
        config = training_configs[selected_key]
    except (ValueError, IndexError, EOFError):
        print("기본값(ultra_fast)으로 진행합니다.")
        selected_key = "ultra_fast"
        config = training_configs[selected_key]
    
    print(f"\n🚀 {selected_key} 모드로 학습을 시작합니다...")
    print(f"   모델: {config['model_name']}")
    print(f"   최대 프레임: {config['max_frames'] or 'All'}")
    print(f"   시간 윈도우: {config['temporal_window']}")
    print(f"   에폭 수: {config['num_epochs']}")
    print(f"   배치 크기: {config['batch_size']}")
    print(f"   ROI 크기: {config['roi_size'][0]}×{config['roi_size'][1]}")
    print(f"   시프트 범위: X(±{config['shift_range_x']}), Y(±{config['shift_range_y']})")
    
    try:
        # Fixed GT 모델 학습 실행
        trainer = train_fixed_gt_model(
            model_name=config['model_name'],
            bin_file_path=BIN_FILE_PATH,
            config_overrides={
                'max_frames': config['max_frames'],
                'temporal_window': config['temporal_window'],
                'num_epochs': config['num_epochs'],
                'batch_size': config['batch_size'],
                'patience': config['patience'],
                'true_center_coord': (541, 360),  # 실제 빔 중심 좌표
                'roi_size': config['roi_size'],
                'shift_range_x': config['shift_range_x'],
                'shift_range_y': config['shift_range_y']
            }
        )
        
        print(f"\n✅ {config['model_name']} 모델 학습 완료!")
        print(f"📁 체크포인트 저장 위치: checkpoints_{config['model_name']}/")
        print(f"🎯 최고 모델: checkpoints_{config['model_name']}/{config['model_name']}_best.pth")
        print(f"📊 훈련 곡선: checkpoints_{config['model_name']}/{config['model_name']}_training_curves.png")
        print(f"🔍 예측 결과: checkpoints_{config['model_name']}/{config['model_name']}_predictions.png")
        
        # 다음 단계 안내
        print(f"\n💡 다음 단계:")
        print(f"   1. 추론 테스트: python inference.py")
        print(f"   2. 데모 실행: python example.py")
        print(f"   3. 다른 모델과 비교: 다시 실행하여 다른 모델 선택")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자가 학습을 중단했습니다.")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 문제 해결 방법:")
        print(f"   1. GPU 메모리 부족: 배치 크기를 더 줄여보세요 (batch_size=2)")
        print(f"   2. 데이터 로딩 오류: bin 파일 경로를 확인하세요")
        print(f"   3. 의존성 오류: filter_sim 모듈이 설치되었는지 확인하세요")
        print(f"   4. 더미 데이터로 테스트: max_events=1000으로 줄여서 테스트")
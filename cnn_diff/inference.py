#!/usr/bin/env python3
"""
inference.py: DVS 레이저 중심점 탐지 모델 추론 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Tuple, Optional, List
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 경로 설정
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from model import LogicDVSNet, get_model
from dataset import DVSDataset, load_individual_frames_from_bin

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: str, input_channels: int = 1, num_neurons: int = 64, output_dim: int = 2) -> LogicDVSNet:
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        input_channels: 입력 채널 수
        num_neurons: 기본 뉴런 수
        output_dim: 출력 차원
        
    Returns:
        로드된 모델
    """
    print(f"📖 Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 모델 생성
    model = get_model(
        input_channels=input_channels,
        num_neurons=num_neurons,
        output_dim=output_dim
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def predict_single_frame(
    model: LogicDVSNet,
    frame: np.ndarray,
    roi_size: Tuple[int, int] = (128, 128)
) -> Tuple[float, float]:
    """
    단일 프레임에 대한 예측
    
    Args:
        model: 학습된 모델
        frame: 입력 프레임 (H, W)
        roi_size: ROI 크기
        
    Returns:
        (x, y) 좌표 (정규화된 값)
    """
    model.eval()
    
    # 프레임 전처리
    if frame.shape != roi_size:
        from scipy.ndimage import zoom
        zoom_factors = (roi_size[0] / frame.shape[0], roi_size[1] / frame.shape[1])
        frame = zoom(frame, zoom_factors, order=1)
    
    # 텐서로 변환
    frame_tensor = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    frame_tensor = frame_tensor.to(device)
    
    # 이진화 전처리
    if frame_tensor.max() > 1.0 or frame_tensor.min() < 0.0:
        frame_tensor = (frame_tensor > frame_tensor.mean()).float()
    
    # 추론
    with torch.no_grad():
        output = model(frame_tensor)
        coords = output[0].cpu().numpy()
    
    return float(coords[0]), float(coords[1])


def predict_batch(
    model: LogicDVSNet,
    frames: List[np.ndarray],
    roi_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32
) -> np.ndarray:
    """
    배치 프레임에 대한 예측
    
    Args:
        model: 학습된 모델
        frames: 프레임 리스트
        roi_size: ROI 크기
        batch_size: 배치 크기
        
    Returns:
        예측 좌표 배열 (N, 2)
    """
    model.eval()
    predictions = []
    
    # 배치 처리
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        # 배치 텐서 생성
        batch_tensors = []
        for frame in batch_frames:
            if frame.shape != roi_size:
                from scipy.ndimage import zoom
                zoom_factors = (roi_size[0] / frame.shape[0], roi_size[1] / frame.shape[1])
                frame = zoom(frame, zoom_factors, order=1)
            
            frame_tensor = torch.from_numpy(frame.astype(np.float32))
            batch_tensors.append(frame_tensor)
        
        batch_tensor = torch.stack(batch_tensors).unsqueeze(1).to(device)  # (B, 1, H, W)
        
        # 이진화 전처리
        if batch_tensor.max() > 1.0 or batch_tensor.min() < 0.0:
            batch_tensor = (batch_tensor > batch_tensor.mean()).float()
        
        # 추론
        with torch.no_grad():
            outputs = model(batch_tensor)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


def visualize_predictions_on_frames(
    frames: List[np.ndarray],
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    num_samples: int = 10
):
    """
    프레임에 예측 결과 시각화
    
    Args:
        frames: 프레임 리스트
        predictions: 예측 좌표 배열 (N, 2)
        targets: 실제 좌표 배열 (N, 2) - 선택적
        save_path: 저장 경로
        num_samples: 표시할 샘플 수
    """
    num_samples = min(num_samples, len(frames))
    indices = np.random.choice(len(frames), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        frame = frames[ax_idx]
        pred = predictions[ax_idx]
        
        # 프레임 표시
        ax.imshow(frame, cmap='gray')
        
        # 예측 좌표 표시
        pred_pixel = (pred[0] * frame.shape[1], pred[1] * frame.shape[0])
        ax.plot(pred_pixel[0], pred_pixel[1], 'ro', markersize=8, label='Predicted')
        
        # 실제 좌표 표시 (있는 경우)
        if targets is not None:
            target = targets[ax_idx]
            target_pixel = (target[0] * frame.shape[1], target[1] * frame.shape[0])
            ax.plot(target_pixel[0], target_pixel[1], 'gx', markersize=8, label='Target')
        
        ax.set_title(f'Sample {ax_idx}')
        ax.axis('off')
    
    plt.suptitle('Prediction Visualization')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_inference(
    checkpoint_path: str,
    bin_file_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
    csv_labels_path: Optional[str] = None,
    input_channels: int = 1,
    num_neurons: int = 64,
    output_dim: int = 2,
    roi_size: Tuple[int, int] = (128, 128),
    save_dir: str = 'inference_results'
):
    """
    추론 실행
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        bin_file_path: bin 파일 경로 (선택적)
        frames: 프레임 리스트 (선택적)
        csv_labels_path: CSV 레이블 파일 경로 (선택적)
        input_channels: 입력 채널 수
        num_neurons: 기본 뉴런 수
        output_dim: 출력 차원
        roi_size: ROI 크기
        save_dir: 결과 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 로드
    model = load_model(checkpoint_path, input_channels, num_neurons, output_dim)
    
    # 데이터 로드
    if frames is None:
        if bin_file_path is None:
            raise ValueError("Either 'frames' or 'bin_file_path' must be provided")
        frames = load_individual_frames_from_bin(bin_file_path)
    
    print(f"\n🔍 Running inference on {len(frames)} frames...")
    
    # 예측 수행
    predictions = predict_batch(model, frames, roi_size=roi_size)
    
    print(f"✅ Inference completed!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # 실제 레이블 로드 (있는 경우)
    targets = None
    if csv_labels_path and os.path.exists(csv_labels_path):
        import pandas as pd
        labels_df = pd.read_csv(csv_labels_path)
        if 'cnn_rel_x' in labels_df.columns and 'cnn_rel_y' in labels_df.columns:
            targets = labels_df[['cnn_rel_x', 'cnn_rel_y']].values[:len(predictions)]
            print(f"   Loaded {len(targets)} target labels")
    
    # 시각화
    viz_path = os.path.join(save_dir, 'predictions_visualization.png')
    visualize_predictions_on_frames(
        frames, predictions, targets,
        save_path=viz_path, num_samples=10
    )
    
    # 결과 저장
    results_path = os.path.join(save_dir, 'predictions.npy')
    np.save(results_path, predictions)
    print(f"💾 Predictions saved to {results_path}")
    
    # 정확도 계산 (타겟이 있는 경우)
    if targets is not None:
        errors = np.sqrt(np.sum((predictions - targets)**2, axis=1))
        mae = np.mean(np.abs(predictions - targets))
        accuracy_5px = np.mean(errors <= 0.05) * 100
        accuracy_10px = np.mean(errors <= 0.10) * 100
        
        print(f"\n📊 Evaluation Results:")
        print(f"   MAE: {mae:.6f}")
        print(f"   Mean Error: {np.mean(errors):.6f} ± {np.std(errors):.6f}")
        print(f"   Accuracy @5px: {accuracy_5px:.1f}%")
        print(f"   Accuracy @10px: {accuracy_10px:.1f}%")
    
    return predictions, targets


if __name__ == "__main__":
    # 예제 사용법
    print("🎯 LogicDVSNet Inference Example")
    print("=" * 60)
    
    # 체크포인트 경로 설정
    # checkpoint_path = "checkpoints/logic_dvs_best.pth"
    # bin_file_path = "/path/to/data.bin"
    # csv_labels_path = "/path/to/labels.csv"
    
    # 추론 실행
    # predictions, targets = run_inference(
    #     checkpoint_path=checkpoint_path,
    #     bin_file_path=bin_file_path,
    #     csv_labels_path=csv_labels_path
    # )
    
    print("\n✅ Inference script ready!")
    print("   Please configure your checkpoint and data paths.")


#!/usr/bin/env python3
"""
YOLOv3-Tiny 레이저 검출 추론 및 시각화
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional

try:
    # yolo_brownian_sim 디렉토리 내에서 실행할 때
    from model import YOLOv3Tiny, decode_predictions, get_laser_center
    from dataset import LaserYOLOBrownianDataset, load_frames_from_bin
    from utils import calculate_event_center_from_roi
except ImportError:
    # 상위 디렉토리에서 실행할 때
    try:
        from yolo_brownian_sim.model import YOLOv3Tiny, decode_predictions, get_laser_center
        from yolo_brownian_sim.dataset import LaserYOLOBrownianDataset, load_frames_from_bin
    except ImportError:
        # fallback to original yolo_sim
        from yolo_sim.model import YOLOv3Tiny, decode_predictions, get_laser_center
        from yolo_sim.dataset import LaserYOLODataset as LaserYOLOBrownianDataset, load_frames_from_bin


class LaserYOLOInference:
    """YOLO 기반 레이저 중심점 추론"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            device: 'auto', 'cuda', 'cpu'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        laser_size = 400 / 512
        self.anchors = [(laser_size, laser_size), (0.5, 0.5), (1.0, 1.0)]
        
        # 이전 성공한 중심점 추적 (YOLO 실패 시 사용)
        self.last_successful_center = (0.5, 0.5)
        
        print(f"✅ Model loaded from {checkpoint_path}")
    
    def _load_model(self, checkpoint_path: str) -> YOLOv3Tiny:
        """체크포인트에서 모델 로드"""
        model = YOLOv3Tiny(input_channels=5, num_classes=1, num_anchors=3)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(
        self, 
        frames: List[np.ndarray],
        roi_params: dict,
        conf_threshold: float = 0.5,
        return_targets: bool = False
    ) -> tuple:
        """
        프레임들에서 레이저 중심점 예측
        
        Args:
            frames: 프레임 리스트
            roi_params: ROI 추출 파라미터
            conf_threshold: confidence 임계값
            return_targets: True일 경우 정답도 함께 반환
        
        Returns:
            predictions: [(x, y, confidence), ...] 정규화된 좌표
            targets: [(x, y), ...] 정답 좌표 (return_targets=True일 때만)
        """
        # 데이터셋 생성 (추론용)
        dataset = LaserYOLODataset(frames, **roi_params)
        dataset.set_training_mode(False)
        
        predictions = []
        targets = [] if return_targets else None
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, target = dataset[idx]
                image = image.unsqueeze(0).to(self.device)  # [1, C, H, W]
                
                # 정답 저장
                if return_targets:
                    targets.append((target[0].item(), target[1].item()))
                
                # 예측
                output = self.model(image)
                boxes_list, scores_list = decode_predictions(
                    output, self.anchors, conf_threshold=conf_threshold
                )
                
                # 중심점 추출
                # decode_predictions는 항상 batch_size만큼 반환 (빈 텐서 포함)
                if len(boxes_list[0]) > 0:
                    center = get_laser_center(boxes_list[0], scores_list[0])
                    if center:
                        best_score = torch.max(scores_list[0]).item()
                        predictions.append((center[0], center[1], best_score))
                        # 성공 시 마지막 위치 업데이트
                        self.last_successful_center = (center[0], center[1])
                    else:
                        # YOLO 감지 실패 시 이전 프레임 값 사용
                        predictions.append((self.last_successful_center[0], self.last_successful_center[1], 0.0))
                        # # YOLO 감지 실패 시 이벤트 중심 계산
                        # event_center = calculate_event_center_from_roi(image[0].cpu().numpy())
                        # predictions.append((event_center[0], event_center[1], 0.0))  # confidence=0.0
                else:
                    # YOLO 감지 실패 시 이전 프레임 값 사용
                    predictions.append((self.last_successful_center[0], self.last_successful_center[1], 0.0))
        
        if return_targets:
            return predictions, targets
        else:
            return predictions
    
    def visualize_results(
        self,
        predictions: List[Tuple[float, float, float]],
        targets: Optional[List[Tuple[float, float]]] = None,
        save_path: str = "yolo_inference_results.png"
    ):
        """
        예측 결과 시각화 (cnn_sim 방식 참고)
        
        Args:
            predictions: [(x, y, conf), ...]
            targets: [(x, y), ...] 정답 (있으면)
            save_path: 저장 경로
        """
        if len(predictions) == 0:
            print("⚠️ No predictions to visualize")
            return
        
        # 데이터 준비
        pred_coords = np.array([(p[0], p[1]) for p in predictions])
        confidences = np.array([p[2] for p in predictions])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if targets is not None:
            target_coords = np.array(targets)
            
            # 오차 계산
            errors = pred_coords - target_coords
            pixel_errors = np.sqrt(np.sum(errors**2, axis=1)) * 512  # ROI 크기 512 기준
            
            # 1. X 좌표 산점도 (True vs Predicted)
            ax = axes[0, 0]
            ax.scatter(target_coords[:, 0], pred_coords[:, 0], alpha=0.6, s=20)
            ax.plot([target_coords[:, 0].min(), target_coords[:, 0].max()], 
                   [target_coords[:, 0].min(), target_coords[:, 0].max()], 
                   'r--', alpha=0.8, label='Perfect prediction')
            ax.set_xlabel('True X')
            ax.set_ylabel('Predicted X')
            ax.set_title('X Coordinate Prediction')
            ax.grid(True, alpha=0.3)
            
            # R² 계산
            var_x = np.sum((target_coords[:, 0] - np.mean(target_coords[:, 0]))**2)
            if var_x > 1e-10:
                r2_x = 1 - np.sum((target_coords[:, 0] - pred_coords[:, 0])**2) / var_x
            else:
                r2_x = 1.0
            ax.text(0.05, 0.95, f'R² = {r2_x:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend()
            
            # 2. Y 좌표 산점도 (True vs Predicted)
            ax = axes[0, 1]
            ax.scatter(target_coords[:, 1], pred_coords[:, 1], alpha=0.6, s=20)
            ax.plot([target_coords[:, 1].min(), target_coords[:, 1].max()], 
                   [target_coords[:, 1].min(), target_coords[:, 1].max()], 
                   'r--', alpha=0.8, label='Perfect prediction')
            ax.set_xlabel('True Y')
            ax.set_ylabel('Predicted Y')
            ax.set_title('Y Coordinate Prediction')
            ax.grid(True, alpha=0.3)
            
            # R² 계산
            var_y = np.sum((target_coords[:, 1] - np.mean(target_coords[:, 1]))**2)
            if var_y > 1e-10:
                r2_y = 1 - np.sum((target_coords[:, 1] - pred_coords[:, 1])**2) / var_y
            else:
                r2_y = 1.0
            ax.text(0.05, 0.95, f'R² = {r2_y:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend()
            
            # 3. 픽셀 오차 히스토그램
            ax = axes[1, 0]
            ax.hist(pixel_errors, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(pixel_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(pixel_errors):.2f}px')
            ax.axvline(np.median(pixel_errors), color='blue', linestyle='--', 
                      label=f'Median: {np.median(pixel_errors):.2f}px')
            ax.set_xlabel('Pixel Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Pixel Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. 오차 분포 (X Error vs Y Error)
            ax = axes[1, 1]
            scatter = ax.scatter(errors[:, 0] * 512, errors[:, 1] * 512, 
                               c=pixel_errors, alpha=0.6, s=20, cmap='viridis')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('X Error (pixels)')
            ax.set_ylabel('Y Error (pixels)')
            ax.set_title('Error Distribution')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Pixel Error')
            
            # 메트릭 계산 및 출력
            mae = np.mean(np.abs(errors)) * 512
            mse = np.mean(errors**2) * 512**2
            rmse = np.sqrt(mse)
            acc_5px = np.mean(pixel_errors <= 5.0) * 100
            acc_10px = np.mean(pixel_errors <= 10.0) * 100
            
            print(f"\n📊 Evaluation Metrics:")
            print(f"   MAE: {mae:.2f}px")
            print(f"   RMSE: {rmse:.2f}px")
            print(f"   Mean Pixel Error: {np.mean(pixel_errors):.2f}±{np.std(pixel_errors):.2f}px")
            print(f"   Median Pixel Error: {np.median(pixel_errors):.2f}px")
            print(f"   Acc@5px: {acc_5px:.1f}%")
            print(f"   Acc@10px: {acc_10px:.1f}%")
            print(f"   Mean Confidence: {np.mean(confidences):.3f}")
            
            # 전체 제목
            plt.suptitle(f'YOLO Laser Detection Results\n'
                        f'MAE: {mae:.2f}px, RMSE: {rmse:.2f}px, '
                        f'Mean Error: {np.mean(pixel_errors):.2f}±{np.std(pixel_errors):.2f}px',
                        fontsize=14, fontweight='bold')
        else:
            # targets가 없을 때 (추론만)
            # 1. X 좌표 시계열
            ax = axes[0, 0]
            frame_indices = np.arange(len(pred_coords))
            ax.scatter(frame_indices, pred_coords[:, 0], alpha=0.6, s=20, c='blue')
            ax.plot(frame_indices, pred_coords[:, 0], alpha=0.3, linewidth=1, c='blue')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('X Coordinate')
            ax.set_title('Predicted X Coordinate')
            ax.grid(True, alpha=0.3)
            
            # 2. Y 좌표 시계열
            ax = axes[0, 1]
            ax.scatter(frame_indices, pred_coords[:, 1], alpha=0.6, s=20, c='orange')
            ax.plot(frame_indices, pred_coords[:, 1], alpha=0.3, linewidth=1, c='orange')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Predicted Y Coordinate')
            ax.grid(True, alpha=0.3)
            
            # 3. Confidence 분포
            ax = axes[1, 0]
            ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.3f}')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Detection Confidence Distribution')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 4. 2D 중심점 분포
            ax = axes[1, 1]
            scatter = ax.scatter(pred_coords[:, 0], pred_coords[:, 1], 
                      c=confidences, cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Predicted Centers (colored by confidence)')
            plt.colorbar(scatter, ax=ax, label='Confidence')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'YOLO Laser Detection Results\n'
                        f'Mean Confidence: {np.mean(confidences):.3f}',
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Results saved to: {save_path}")
        
        try:
            plt.show()
        except:
            plt.close()
    
    def visualize_worst_cases(
        self,
        predictions: List[Tuple[float, float, float]],
        targets: List[Tuple[float, float]],
        frames: List[np.ndarray],
        roi_params: dict,
        num_cases: int = 6,
        save_path: str = "worst_cases_analysis.png"
    ):
        """
        픽셀 오차가 큰 프레임들을 시각화하여 원인 분석
        
        Args:
            predictions: [(x, y, conf), ...]
            targets: [(x, y), ...]
            frames: 원본 프레임 리스트
            roi_params: ROI 추출 파라미터
            num_cases: 분석할 worst case 개수
            save_path: 저장 경로
        """
        print(f"\n🔍 Analyzing worst {num_cases} cases...")
        
        # 오차 계산 및 정렬
        pred_coords = np.array([(p[0], p[1]) for p in predictions])
        pixel_errors = np.sqrt(np.sum((pred_coords - np.array(targets))**2, axis=1)) * 512
        worst_indices = np.argsort(pixel_errors)[::-1][:num_cases]
        
        # ROI 추출용 데이터셋
        from dataset import LaserYOLODataset
        dataset = LaserYOLODataset(frames, **roi_params)
        dataset.set_training_mode(False)
        
        # 서브플롯 생성
        cols = (num_cases + 1) // 2
        fig, axes = plt.subplots(2, cols, figsize=(5*cols, 10))
        axes = axes.flatten() if num_cases > 1 else [axes]
        
        for plot_idx, sample_idx in enumerate(worst_indices):
            ax = axes[plot_idx]
            
            # ROI 이미지 추출
            roi_image, target = dataset[sample_idx]
            
            # Temporal window의 중간 프레임 사용
            mid_frame = roi_image[roi_image.shape[0] // 2].numpy()
            
            # 예측 및 정답 정보
            pred_x, pred_y, conf = predictions[sample_idx]
            true_x, true_y = targets[sample_idx]
            error = pixel_errors[sample_idx]
            
            # 이미지 및 중심점 표시
            ax.imshow(mid_frame, cmap='hot', interpolation='nearest')
            true_px, pred_px = true_x * 512, pred_x * 512
            true_py, pred_py = true_y * 512, pred_y * 512
            
            ax.plot(true_px, true_py, 'r+', markersize=20, markeredgewidth=3, label='True')
            ax.plot(pred_px, pred_py, 'bo', markersize=10, markerfacecolor='none', 
                   markeredgewidth=2, label='Predicted')
            ax.plot([true_px, pred_px], [true_py, pred_py], 'y--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f'Frame {sample_idx}\nError: {error:.1f}px, Conf: {conf:.3f}', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.axis('off')
        
        # 빈 서브플롯 숨기기
        for idx in range(len(worst_indices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Worst {num_cases} Cases Analysis\n'
            f'Red cross = True center, Blue circle = Predicted center',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Worst cases analysis saved to: {save_path}")
        
        # 통계 출력
        print(f"\n📊 Worst Cases Statistics:")
        for idx, sample_idx in enumerate(worst_indices):
            pred_x, pred_y, conf = predictions[sample_idx]
            error = pixel_errors[sample_idx]
            print(f"   {idx+1}. Frame {sample_idx}: Error={error:.1f}px, Conf={conf:.3f}")
        
        try:
            plt.show()
        except:
            plt.close()


def run_inference(
    checkpoint_path: str,
    bin_file_path: str,
    max_frames: int = 100,
    save_dir: str = "inference_results"
):
    """추론 실행 및 시각화"""
    
    print("🚀 Starting YOLO Inference")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 프레임 로딩
    frames = load_frames_from_bin(bin_file_path, max_frames=max_frames)
    if len(frames) == 0:
        print("❌ No frames loaded!")
        return
    
    # 추론기 생성
    inferencer = LaserYOLOInference(checkpoint_path)
    
    # ROI 파라미터
    roi_params = {
        'true_center_coord': (541, 361),
        'laser_diameter': 400,
        'roi_size': (512, 512),
        'temporal_window': 5,
        'shift_range_x': 50,
        'shift_range_y': 50
    }
    
    # 예측 (정답도 함께 반환)
    print("\n🔍 Running inference...")
    predictions, targets = inferencer.predict(frames, roi_params, conf_threshold=0.3, return_targets=True)
    
    print(f"✅ Predicted {len(predictions)} samples")
    print(f"   Detections: {sum(1 for p in predictions if p[2] > 0.3)}/{len(predictions)}")
    
    # 시각화 (정답과 함께)
    save_path = os.path.join(save_dir, "yolo_inference_results.png")
    inferencer.visualize_results(predictions, targets=targets, save_path=save_path)
    
    # 오차가 큰 프레임 시각화
    worst_cases_path = os.path.join(save_dir, "worst_cases_analysis.png")
    inferencer.visualize_worst_cases(
        predictions, targets, frames, roi_params, 
        num_cases=6, save_path=worst_cases_path
    )
    
    print("\n✅ Inference completed!")
    return predictions


if __name__ == "__main__":
    # 예시 실행
    checkpoint_path = "checkpoints_yolo_tiny_laser/yolo_tiny_laser_best.pth"
    bin_file_path = "/hai/home/jdj/dvs/sim/data/gaussian_large.bin"
    
    if os.path.exists(checkpoint_path):
        run_inference(
            checkpoint_path=checkpoint_path,
            bin_file_path=bin_file_path,
            max_frames=100
        )
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("   Train the model first: python train.py")

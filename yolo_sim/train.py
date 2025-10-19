#!/usr/bin/env python3
"""
YOLOv3-Tiny 레이저 검출 모델 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime

try:
    # yolo_sim 디렉토리 내에서 실행할 때
    from model import YOLOv3Tiny, decode_predictions, get_laser_center
    from dataset import create_train_val_loaders, load_frames_from_bin
    from utils import calculate_event_center_from_roi
except ImportError:
    # 상위 디렉토리에서 실행할 때
    from yolo_sim.model import YOLOv3Tiny, decode_predictions, get_laser_center
    from yolo_sim.dataset import create_train_val_loaders, load_frames_from_bin
    from yolo_sim.utils import calculate_event_center_from_roi


class YOLOLoss(nn.Module):
    """간소화된 YOLO Loss (단일 객체 감지용)"""
    
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.1):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj  # 255개 negative cell의 영향 최소화
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(self, predictions, targets, anchors):
        """
        Args:
            predictions: [batch, num_anchors * 6, H, W]
            targets: [batch, 5] (x, y, w, h, class)
            anchors: [(w, h), ...]
        """
        batch_size, _, grid_h, grid_w = predictions.shape
        num_anchors = len(anchors)
        
        # Reshape predictions
        pred = predictions.view(batch_size, num_anchors, 6, grid_h, grid_w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [batch, anchors, H, W, 6]
        
        # Grid 생성
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        grid_y = grid_y.float().to(predictions.device)
        grid_x = grid_x.float().to(predictions.device)
        
        # Target을 grid로 변환
        target_x = targets[:, 0] * grid_w  # 정규화 → grid 좌표
        target_y = targets[:, 1] * grid_h
        grid_i = target_x.long().clamp(0, grid_w - 1)
        grid_j = target_y.long().clamp(0, grid_h - 1)
        
        # Loss 계산 (첫 번째 anchor 사용 - 레이저는 단일 객체)
        total_loss = 0.0
        coord_loss = 0.0
        obj_loss = 0.0
        
        anchor_w, anchor_h = anchors[0]
        
        for b in range(batch_size):
            gi, gj = grid_i[b], grid_j[b]
            
            # 좌표 loss
            pred_xy = pred[b, 0, gj, gi, :2]
            target_xy = torch.tensor([target_x[b] - gi.float(), target_y[b] - gj.float()], 
                                     device=pred_xy.device)
            coord_loss += self.mse(torch.sigmoid(pred_xy), target_xy)
            
            # 크기 loss
            pred_wh = pred[b, 0, gj, gi, 2:4]
            target_wh_ratio = torch.tensor([targets[b, 2] / anchor_w, targets[b, 3] / anchor_h], 
                                          device=pred_wh.device)
            coord_loss += self.mse(pred_wh, torch.log(target_wh_ratio + 1e-6))
            
            # Objectness loss (positive와 negative 분리)
            pred_conf = pred[b, 0, :, :, 4]
            
            # Positive loss (객체 있는 cell)
            obj_loss += self.bce(pred_conf[gj, gi].unsqueeze(0), 
                                torch.ones(1, device=predictions.device))
            
            # Negative loss (객체 없는 cell) - 가중치 낮춤
            noobj_mask = torch.ones(grid_h, grid_w, device=predictions.device, dtype=torch.bool)
            noobj_mask[gj, gi] = False
            if noobj_mask.sum() > 0:
                obj_loss += self.lambda_noobj * self.bce(
                    pred_conf[noobj_mask], 
                    torch.zeros(noobj_mask.sum(), device=predictions.device)
                )
        
        total_loss = self.lambda_coord * coord_loss + obj_loss
        
        return total_loss / batch_size, coord_loss / batch_size, obj_loss / batch_size


def plot_training_curves(history: Dict[str, List], save_dir: str, model_name: str):
    """학습 곡선 시각화"""
    print(f"\n📈 Plotting training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Pixel Error
    ax = axes[0, 1]
    ax.plot(history['pixel_error'], label='Pixel Error', color='orange', linewidth=2)
    ax.axhline(5.0, color='red', linestyle='--', alpha=0.5, label='5px threshold')
    ax.axhline(10.0, color='green', linestyle='--', alpha=0.5, label='10px threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pixel Error (px)')
    ax.set_title('Validation Pixel Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Accuracy @ 5px
    ax = axes[1, 0]
    ax.plot(history['acc_5px'], label='Acc@5px', color='blue', linewidth=2)
    ax.plot(history['acc_10px'], label='Acc@10px', color='cyan', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # 4. Loss (Log Scale)
    ax = axes[1, 1]
    ax.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    ax.semilogy(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    plot_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {plot_path}")
    
    try:
        plt.show()
    except:
        plt.close()


def visualize_final_predictions(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    save_dir: str, 
    model_name: str
):
    """최종 예측 결과 시각화 (cnn_sim 방식)"""
    print(f"\n🎨 Creating final prediction visualization...")
    
    if len(predictions) == 0:
        print("⚠️ No predictions to visualize")
        return
    
    # 오차 계산
    errors = predictions - targets
    pixel_errors = np.sqrt(np.sum(errors**2, axis=1)) * 512  # ROI 크기 512 기준
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. X 좌표 산점도
    ax = axes[0, 0]
    ax.scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20)
    ax.plot([targets[:, 0].min(), targets[:, 0].max()], 
           [targets[:, 0].min(), targets[:, 0].max()], 
           'r--', alpha=0.8, label='Perfect prediction')
    ax.set_xlabel('True X')
    ax.set_ylabel('Predicted X')
    ax.set_title('X Coordinate Prediction')
    ax.grid(True, alpha=0.3)
    
    # R² 계산
    var_x = np.sum((targets[:, 0] - np.mean(targets[:, 0]))**2)
    if var_x > 1e-10:
        r2_x = 1 - np.sum((targets[:, 0] - predictions[:, 0])**2) / var_x
    else:
        r2_x = 1.0
    ax.text(0.05, 0.95, f'R² = {r2_x:.3f}', transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend()
    
    # 2. Y 좌표 산점도
    ax = axes[0, 1]
    ax.scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20)
    ax.plot([targets[:, 1].min(), targets[:, 1].max()], 
           [targets[:, 1].min(), targets[:, 1].max()], 
           'r--', alpha=0.8, label='Perfect prediction')
    ax.set_xlabel('True Y')
    ax.set_ylabel('Predicted Y')
    ax.set_title('Y Coordinate Prediction')
    ax.grid(True, alpha=0.3)
    
    # R² 계산
    var_y = np.sum((targets[:, 1] - np.mean(targets[:, 1]))**2)
    if var_y > 1e-10:
        r2_y = 1 - np.sum((targets[:, 1] - predictions[:, 1])**2) / var_y
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
    
    # 메트릭 계산
    mae = np.mean(np.abs(errors)) * 512
    mse = np.mean(errors**2) * 512**2
    rmse = np.sqrt(mse)
    acc_5px = np.mean(pixel_errors <= 5.0) * 100
    acc_10px = np.mean(pixel_errors <= 10.0) * 100
    
    # 전체 제목
    plt.suptitle(f'{model_name} Final Validation Results\n'
                f'MAE: {mae:.2f}px, RMSE: {rmse:.2f}px, '
                f'Mean Error: {np.mean(pixel_errors):.2f}±{np.std(pixel_errors):.2f}px',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(save_dir, f'{model_name}_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Final predictions saved to: {save_path}")
    
    try:
        plt.show()
    except:
        plt.close()


def visualize_worst_cases_training(
    predictions: np.ndarray,
    targets: np.ndarray,
    pixel_errors: List[float],
    save_dir: str,
    model_name: str,
    num_cases: int = 10
):
    """학습 시 worst cases 통계 시각화"""
    print(f"\n🔍 Analyzing worst {num_cases} cases from validation set...")
    
    pixel_errors_array = np.array(pixel_errors)
    worst_indices = np.argsort(pixel_errors_array)[::-1][:num_cases]
    
    # 2x1 레이아웃: 통계 테이블 + 오차 분포
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Worst cases 테이블
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    table_data = [['Rank', 'Sample #', 'Pixel Error', 'Pred X', 'Pred Y', 'True X', 'True Y', 'Error X', 'Error Y']]
    
    for rank, idx in enumerate(worst_indices, 1):
        pred_x, pred_y = predictions[idx]
        true_x, true_y = targets[idx]
        error = pixel_errors_array[idx]
        error_x = (pred_x - true_x) * 512
        error_y = (pred_y - true_y) * 512
        
        table_data.append([
            f'{rank}',
            f'{idx}',
            f'{error:.1f}px',
            f'{pred_x:.3f}',
            f'{pred_y:.3f}',
            f'{true_x:.3f}',
            f'{true_y:.3f}',
            f'{error_x:.1f}px',
            f'{error_y:.1f}px'
        ])
    
    table = ax1.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(9):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1.set_title('Top 10 Worst Predictions', fontsize=14, fontweight='bold', pad=20)
    
    # 2. 오차 크기별 분포
    ax2 = fig.add_subplot(gs[1, 0])
    error_bins = [0, 5, 10, 20, 50, 100, np.max(pixel_errors_array)+1]
    hist, _ = np.histogram(pixel_errors_array, bins=error_bins)
    
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    bars = ax2.bar(range(len(hist)), hist, color=colors[:len(hist)], alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(range(len(hist)))
    ax2.set_xticklabels(['0-5px', '5-10px', '10-20px', '20-50px', '50-100px', '>100px'])
    ax2.set_xlabel('Error Range')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution by Range')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 개수 표시
    for bar, count in zip(bars, hist):
        height = bar.get_height()
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9)
    
    # 3. Worst cases 위치 분포
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 전체 예측 (회색)
    ax3.scatter(predictions[:, 0], predictions[:, 1], c='lightgray', s=20, alpha=0.3, label='All predictions')
    
    # Worst cases (빨간색, 크게)
    worst_preds = predictions[worst_indices]
    worst_targets = targets[worst_indices]
    ax3.scatter(worst_preds[:, 0], worst_preds[:, 1], c='red', s=100, alpha=0.7, 
               marker='x', linewidths=2, label='Worst predictions')
    ax3.scatter(worst_targets[:, 0], worst_targets[:, 1], c='blue', s=100, alpha=0.7,
               marker='+', linewidths=2, label='Worst true centers')
    
    # 연결선
    for wp, wt in zip(worst_preds, worst_targets):
        ax3.plot([wp[0], wt[0]], [wp[1], wt[1]], 'r--', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Worst Cases Spatial Distribution')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    plt.suptitle(f'{model_name} - Worst Cases Analysis', fontsize=16, fontweight='bold')
    
    # 저장
    save_path = os.path.join(save_dir, f'{model_name}_worst_cases.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Worst cases analysis saved to: {save_path}")
    
    # 통계 출력
    print(f"\n📊 Worst {num_cases} Cases:")
    for rank, idx in enumerate(worst_indices, 1):
        error = pixel_errors_array[idx]
        print(f"   {rank}. Sample {idx}: {error:.1f}px")
    
    try:
        plt.show()
    except:
        plt.close()


def train_yolo(
    model_name: str = 'yolo_tiny',
    bin_file_path: str = "/hai/home/jdj/dvs/data/gaussian_large.bin",
    max_frames: int = 500,
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.001,
    device: str = 'auto'
):
    """YOLO모델 학습"""
    
    # Device 설정
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🔧 Using device: {device}")
    
    # 데이터 로딩
    frames = load_frames_from_bin(bin_file_path, max_frames=max_frames)
    if len(frames) == 0:
        print("❌ No frames loaded!")
        return
    
    # 데이터로더 생성
    train_loader, val_loader = create_train_val_loaders(
        frames,
        train_ratio=0.8,
        batch_size=batch_size,
        num_workers=0,
        true_center_coord=(541, 360),
        laser_diameter=400,
        roi_size=(512, 512),
        temporal_window=5,
        shift_range_x=50,
        shift_range_y=50
    )
    
    # 모델 생성
    model = YOLOv3Tiny(input_channels=5, num_classes=1, num_anchors=3).to(device)
    print(f"📊 Model created: {model_name}")
    
    # Loss, Optimizer
    criterion = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    # Anchors (레이저 직경 400px, ROI 512x512 기준)
    laser_size = 400 / 512
    anchors = [(laser_size, laser_size), (0.5, 0.5), (1.0, 1.0)]
    
    # 학습 히스토리 추적
    history = {
        'train_loss': [],
        'val_loss': [],
        'pixel_error': [],
        'acc_5px': [],
        'acc_10px': []
    }
    
    # 학습 루프
    print(f"\n🚀 Starting training for {num_epochs} epochs")
    print("=" * 70)
    
    best_val_loss = float('inf')
    save_dir = f'checkpoints_{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, coord_loss, obj_loss = criterion(outputs, targets, anchors)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0.0
        pixel_errors = []
        val_predictions = []
        val_targets = []
        
        last_successful_center = (0.5, 0.5)
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss, _, _ = criterion(outputs, targets, anchors)
                val_loss += loss.item()
                
                # 중심점 정확도 계산
                boxes_list, scores_list = decode_predictions(outputs, anchors, conf_threshold=0.4)
                for i, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
                    target_center = (targets[i, 0].item(), targets[i, 1].item())
                    val_targets.append(target_center)
                    
                    pred_center = get_laser_center(boxes, scores) if len(boxes) > 0 else None
                    
                    if pred_center is None:
                        # YOLO 감지 실패 시 이벤트 중심 계산
                        event_center = calculate_event_center_from_roi(images[i].cpu().numpy())
                        final_center = event_center
                    else:
                        # 성공: 현재 좌표 사용 및 마지막 위치 업데이트
                        final_center = pred_center
                        last_successful_center = pred_center
                    
                    val_predictions.append(final_center)
                    pixel_error = np.sqrt(((np.array(final_center) - np.array(target_center)) * 512) ** 2).sum()
                    pixel_errors.append(pixel_error)
                    
        
        val_loss /= len(val_loader)
        avg_pixel_error = np.mean(pixel_errors) if pixel_errors else 999.0
        acc_5px = np.mean([e <= 5.0 for e in pixel_errors]) * 100 if pixel_errors else 0.0
        acc_10px = np.mean([e <= 10.0 for e in pixel_errors]) * 100 if pixel_errors else 0.0
        
        # 히스토리 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pixel_error'].append(avg_pixel_error)
        history['acc_5px'].append(acc_5px)
        history['acc_10px'].append(acc_10px)
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Pixel Error: {avg_pixel_error:.2f}px")
        print(f"  Acc@5px: {acc_5px:.1f}%, Acc@10px: {acc_10px:.1f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
            print("  ⭐ Best model saved!")
    
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"📁 Checkpoints saved to: {save_dir}/")
    
    # 학습 곡선 시각화
    plot_training_curves(history, save_dir, model_name)
    
    # 최종 예측 결과 시각화
    if val_predictions and val_targets:
        visualize_final_predictions(
            np.array(val_predictions), 
            np.array(val_targets), 
            save_dir, 
            model_name
        )
        
        # Worst cases 시각화 (학습 완료 후)
        visualize_worst_cases_training(
            np.array(val_predictions),
            np.array(val_targets),
            pixel_errors,
            save_dir,
            model_name
        )


if __name__ == "__main__":
    train_yolo(
        model_name='yolo_tiny_laser',
        max_frames=500,
        num_epochs=50,
        batch_size=4,
        lr=0.001
    )

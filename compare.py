#!/usr/bin/env python3
"""
CNN (MobileNet) vs YOLO vs Filter (Kalman) 중심값 추정 결과 비교 스크립트

cnn_sim의 MobileNet 모델 추론 결과, 
yolo_sim의 YOLOv3-Tiny 모델 추론 결과, 그리고
filter_sim의 spatial_filter_kalman 중심점 추정 결과를 비교 분석합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import sys
from typing import Dict, Tuple, Optional

# 경로 추가
sys.path.append('/hai/home/jdj/dvs')  # lib 모듈 사용을 위해 루트 추가
sys.path.append('/hai/home/jdj/dvs/cnn_sim')
sys.path.append('/hai/home/jdj/dvs/filter_sim')
sys.path.append('/hai/home/jdj/dvs/yolo_sim')

from cnn_sim.inference import DVSInference
from lib.bin_processor import BinProcessor  # lib에서 가져오기
from yolo_sim.inference import LaserYOLOInference
from yolo_sim.dataset import load_frames_from_bin as yolo_load_frames


def load_cnn_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    max_frames: int = 50,
    test_augmentation: bool = True
) -> Dict:
    """CNN 모델의 추론 결과 로드 (상대 좌표를 절대 좌표로 변환)"""
    print("🤖 Loading CNN (MobileNet) predictions...")
    
    # Inference 객체 생성
    inferencer = DVSInference(checkpoint_path)
    
    # 프레임 로드
    individual_frames = inferencer.load_frames_from_bin(bin_file_path, max_frames=max_frames)
    
    # 추론 실행
    results = inferencer.predict_from_frames(
        individual_frames, 
        test_augmentation=test_augmentation
    )
    
    # 상대 좌표를 절대 좌표로 변환
    roi_size = 512
    true_center_coord = (541, 360)
    
    predictions_rel = np.array(results['predictions'])
    targets_rel = np.array(results['targets'])
    
    predictions_abs = []
    targets_abs = []
    
    for pred_rel, target_rel in zip(predictions_rel, targets_rel):
        # 예측값 변환: 상대 좌표 → 절대 좌표
        pred_x_abs = true_center_coord[0] + (pred_rel[0] - 0.5) * roi_size
        pred_y_abs = true_center_coord[1] + (pred_rel[1] - 0.5) * roi_size
        
        # 타겟값 변환: 상대 좌표 → 절대 좌표
        target_x_abs = true_center_coord[0] + (target_rel[0] - 0.5) * roi_size
        target_y_abs = true_center_coord[1] + (target_rel[1] - 0.5) * roi_size
        
        predictions_abs.append((pred_x_abs, pred_y_abs))
        targets_abs.append((target_x_abs, target_y_abs))
    
    # 오차 계산 (절대 좌표 기준)
    errors = np.sqrt(np.sum((np.array(predictions_abs) - np.array(targets_abs))**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"✅ CNN predictions loaded: {len(predictions_abs)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f}")
    
    return {
        'predictions': np.array(predictions_abs),
        'targets': np.array(targets_abs),
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'CNN (MobileNet)'
    }


def load_filter_predictions(csv_file_path: str) -> Dict:
    """Filter (Kalman) 결과를 CSV에서 로드"""
    print("🔍 Loading Filter (Kalman) predictions...")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # CSV 로드
    df = pd.read_csv(csv_file_path)
    df = df.dropna()  # 유효한 데이터만
    
    # predictions 형식으로 변환
    predictions = df[['center_x', 'center_y']].values
    
    # filter의 경우 고정된 ground truth와 비교
    # cnn_sim과 동일한 true_center_coord 사용
    true_center = np.array([541, 360])
    targets = np.tile(true_center, (len(predictions), 1))
    
    # 오차 계산
    errors = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"✅ Filter predictions loaded: {len(predictions)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'Filter (Kalman)'
    }


def load_yolo_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    max_frames: int = 50,
    conf_threshold: float = 0.5
) -> Dict:
    """YOLO 모델의 추론 결과 로드 (상대 좌표를 절대 좌표로 변환)"""
    print("🎯 Loading YOLO predictions...")
    
    # YOLO Inference 객체 생성
    inferencer = LaserYOLOInference(checkpoint_path)
    
    # 프레임 로드
    individual_frames = yolo_load_frames(bin_file_path, max_frames=max_frames)
    
    # ROI 파라미터 설정
    roi_params = {
        'true_center_coord': (541, 360),
        'laser_diameter': 400,
        'roi_size': (512, 512),
        'temporal_window': 5,
        'shift_range_x': 0,  # 공정한 비교를 위해 증강 비활성화
        'shift_range_y': 0
    }
    
    # 추론 실행
    predictions_rel, targets_rel = inferencer.predict(
        frames=individual_frames,
        roi_params=roi_params,
        conf_threshold=conf_threshold,
        return_targets=True
    )
    
    # 상대 좌표를 절대 좌표로 변환
    roi_size = 512
    true_center_coord = (541, 360)
    
    predictions_abs = []
    targets_abs = []
    
    for pred_rel, target_rel in zip(predictions_rel, targets_rel):
        # 예측값 변환: 상대 좌표 → 절대 좌표
        pred_x_abs = true_center_coord[0] + (pred_rel[0] - 0.5) * roi_size
        pred_y_abs = true_center_coord[1] + (pred_rel[1] - 0.5) * roi_size
        
        # 타겟값 변환: 상대 좌표 → 절대 좌표
        target_x_abs = true_center_coord[0] + (target_rel[0] - 0.5) * roi_size
        target_y_abs = true_center_coord[1] + (target_rel[1] - 0.5) * roi_size
        
        predictions_abs.append((pred_x_abs, pred_y_abs))
        targets_abs.append((target_x_abs, target_y_abs))
    
    # 오차 계산 (절대 좌표 기준)
    errors = np.sqrt(np.sum((np.array(predictions_abs) - np.array(targets_abs))**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"✅ YOLO predictions loaded: {len(predictions_abs)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f}")
    
    return {
        'predictions': np.array(predictions_abs),
        'targets': np.array(targets_abs),
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'YOLO (Tiny)'
    }


def compare_methods(
    cnn_results: Dict,
    yolo_results: Dict,
    filter_results: Dict,
    save_path: Optional[str] = None
):
    """세 방법의 결과를 한 그래프에 비교"""
    
    print("\n📊 Creating 3-way comparison visualization...")
    
    # 데이터 추출
    cnn_pred = cnn_results['predictions']
    cnn_target = cnn_results['targets']
    yolo_pred = yolo_results['predictions']
    yolo_target = yolo_results['targets']
    filter_pred = filter_results['predictions']
    filter_target = filter_results['targets']
    
    # 오차 계산
    cnn_errors = np.sqrt(np.sum((cnn_pred - cnn_target)**2, axis=1))
    yolo_errors = np.sqrt(np.sum((yolo_pred - yolo_target)**2, axis=1))
    filter_errors = np.sqrt(np.sum((filter_pred - filter_target)**2, axis=1))
    
    # 그래프 생성
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. X 좌표 분포도 비교 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 각 방법별 X 좌표 오차 분포 (True - Predicted)
    cnn_x_errors = cnn_target[:, 0] - cnn_pred[:, 0]
    yolo_x_errors = yolo_target[:, 0] - yolo_pred[:, 0]
    filter_x_errors = filter_target[:, 0] - filter_pred[:, 0]
    
    # KDE 분포도 그리기
    sns.kdeplot(data=cnn_x_errors, ax=ax1, label='CNN', color='blue', alpha=0.7, linewidth=2)
    sns.kdeplot(data=yolo_x_errors, ax=ax1, label='YOLO', color='green', alpha=0.7, linewidth=2)
    sns.kdeplot(data=filter_x_errors, ax=ax1, label='Filter', color='red', alpha=0.7, linewidth=2)
    
    # 완벽한 예측선 (오차 = 0)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    
    ax1.set_xlabel('X Coordinate Error (True - Predicted)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('X Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Y 좌표 분포도 비교 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 각 방법별 Y 좌표 오차 분포 (True - Predicted)
    cnn_y_errors = cnn_target[:, 1] - cnn_pred[:, 1]
    yolo_y_errors = yolo_target[:, 1] - yolo_pred[:, 1]
    filter_y_errors = filter_target[:, 1] - filter_pred[:, 1]
    
    # KDE 분포도 그리기
    sns.kdeplot(data=cnn_y_errors, ax=ax2, label='CNN', color='blue', alpha=0.7, linewidth=2)
    sns.kdeplot(data=yolo_y_errors, ax=ax2, label='YOLO', color='green', alpha=0.7, linewidth=2)
    sns.kdeplot(data=filter_y_errors, ax=ax2, label='Filter', color='red', alpha=0.7, linewidth=2)
    
    # 완벽한 예측선 (오차 = 0)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    
    ax2.set_xlabel('Y Coordinate Error (True - Predicted)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Y Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. 오차 분포 히스토그램 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(cnn_errors, bins=30, alpha=0.6, label='CNN', color='blue', edgecolor='navy')
    ax3.hist(yolo_errors, bins=30, alpha=0.6, label='YOLO', color='green', edgecolor='darkgreen')
    ax3.hist(filter_errors, bins=30, alpha=0.6, label='Filter', color='red', edgecolor='darkred')
    ax3.axvline(np.mean(cnn_errors), color='blue', linestyle='--', linewidth=2, 
               label=f'CNN Mean: {np.mean(cnn_errors):.2f}')
    ax3.axvline(np.mean(yolo_errors), color='green', linestyle='--', linewidth=2, 
               label=f'YOLO Mean: {np.mean(yolo_errors):.2f}')
    ax3.axvline(np.mean(filter_errors), color='red', linestyle='--', linewidth=2, 
               label=f'Filter Mean: {np.mean(filter_errors):.2f}')
    ax3.set_xlabel('Pixel Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 2D 궤적 비교 (X-Y 평면) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(cnn_pred[:, 0], cnn_pred[:, 1], 
               alpha=0.6, s=40, label='CNN', c='blue', edgecolors='navy')
    ax4.scatter(yolo_pred[:, 0], yolo_pred[:, 1], 
               alpha=0.6, s=40, label='YOLO', c='green', edgecolors='darkgreen')
    ax4.scatter(filter_pred[:, 0], filter_pred[:, 1], 
               alpha=0.6, s=40, label='Filter', c='red', edgecolors='darkred')
    
    # Ground truth 중심
    true_center_x = np.mean(cnn_target[:, 0])
    true_center_y = np.mean(cnn_target[:, 1])
    ax4.scatter(true_center_x, true_center_y, 
               s=200, marker='*', c='gold', edgecolors='black', 
               linewidths=2, label='True Center', zorder=10)
    
    ax4.set_xlabel('X Coordinate', fontsize=11)
    ax4.set_ylabel('Y Coordinate', fontsize=11)
    ax4.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. 오차 박스플롯 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [cnn_errors, yolo_errors, filter_errors]
    box = ax5.boxplot(box_data, labels=['CNN', 'YOLO', 'Filter'], patch_artist=True)
    
    # 색상 설정
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('Pixel Error', fontsize=11)
    ax5.set_title('Error Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== 6. 누적 분포 함수 (CDF) ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    # CNN CDF
    cnn_sorted = np.sort(cnn_errors)
    cnn_cdf = np.arange(1, len(cnn_sorted) + 1) / len(cnn_sorted)
    ax6.plot(cnn_sorted, cnn_cdf, label='CNN', color='blue', linewidth=2)
    
    # YOLO CDF
    yolo_sorted = np.sort(yolo_errors)
    yolo_cdf = np.arange(1, len(yolo_sorted) + 1) / len(yolo_sorted)
    ax6.plot(yolo_sorted, yolo_cdf, label='YOLO', color='green', linewidth=2)
    
    # Filter CDF
    filter_sorted = np.sort(filter_errors)
    filter_cdf = np.arange(1, len(filter_sorted) + 1) / len(filter_sorted)
    ax6.plot(filter_sorted, filter_cdf, label='Filter', color='red', linewidth=2)
    
    ax6.set_xlabel('Pixel Error', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ========== 7. 시간에 따른 오차 변화 ==========
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(cnn_errors, alpha=0.7, linewidth=1.5, label='CNN', color='blue')
    ax7.plot(yolo_errors, alpha=0.7, linewidth=1.5, label='YOLO', color='green')
    ax7.plot(filter_errors, alpha=0.7, linewidth=1.5, label='Filter', color='red')
    ax7.axhline(np.mean(cnn_errors), color='blue', linestyle='--', alpha=0.5)
    ax7.axhline(np.mean(yolo_errors), color='green', linestyle='--', alpha=0.5)
    ax7.axhline(np.mean(filter_errors), color='red', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Sample Index', fontsize=11)
    ax7.set_ylabel('Pixel Error', fontsize=11)
    ax7.set_title('Error Over Time', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ========== 8. 통계 요약 표 ==========
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # 통계 계산
    stats_data = [
        ['Method', 'Mean Error', 'Std Error', 'Median Error', 'Max Error', 'Min Error'],
        ['CNN', 
         f'{np.mean(cnn_errors):.2f}', 
         f'{np.std(cnn_errors):.2f}',
         f'{np.median(cnn_errors):.2f}',
         f'{np.max(cnn_errors):.2f}',
         f'{np.min(cnn_errors):.2f}'],
        ['YOLO', 
         f'{np.mean(yolo_errors):.2f}', 
         f'{np.std(yolo_errors):.2f}',
         f'{np.median(yolo_errors):.2f}',
         f'{np.max(yolo_errors):.2f}',
         f'{np.min(yolo_errors):.2f}'],
        ['Filter', 
         f'{np.mean(filter_errors):.2f}', 
         f'{np.std(filter_errors):.2f}',
         f'{np.median(filter_errors):.2f}',
         f'{np.max(filter_errors):.2f}',
         f'{np.min(filter_errors):.2f}']
    ]
    
    table = ax8.table(cellText=stats_data, loc='center', cellLoc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 헤더 스타일링
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행 색상
    table[(1, 0)].set_facecolor('#E3F2FD')  # CNN - blue
    table[(2, 0)].set_facecolor('#E8F5E8')  # YOLO - green
    table[(3, 0)].set_facecolor('#FFEBEE')  # Filter - red
    
    ax8.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    # ========== 전체 제목 ==========
    fig.suptitle(
        f'CNN vs YOLO vs Filter Comparison\n'
        f'CNN: {np.mean(cnn_errors):.2f}±{np.std(cnn_errors):.2f} px | '
        f'YOLO: {np.mean(yolo_errors):.2f}±{np.std(yolo_errors):.2f} px | '
        f'Filter: {np.mean(filter_errors):.2f}±{np.std(filter_errors):.2f} px',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 저장 (전체 버전)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Full comparison plot saved to: {save_path}")
    
    plt.show()
    
    # ========================================
    # 두 번째 이미지: 간소화 버전 (6개 그래프만)
    # ========================================
    print("\n📊 Creating compact comparison visualization (6 plots)...")
    
    fig2 = plt.figure(figsize=(18, 10))
    gs2 = fig2.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. X 좌표 분포도 비교 ==========
    ax1 = fig2.add_subplot(gs2[0, 0])
    sns.kdeplot(data=cnn_x_errors, ax=ax1, label='CNN', color='blue', alpha=0.7, linewidth=2)
    sns.kdeplot(data=yolo_x_errors, ax=ax1, label='YOLO', color='green', alpha=0.7, linewidth=2)
    sns.kdeplot(data=filter_x_errors, ax=ax1, label='Filter', color='red', alpha=0.7, linewidth=2)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax1.set_xlabel('X Coordinate Error (True - Predicted)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('X Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Y 좌표 분포도 비교 ==========
    ax2 = fig2.add_subplot(gs2[0, 1])
    sns.kdeplot(data=cnn_y_errors, ax=ax2, label='CNN', color='blue', alpha=0.7, linewidth=2)
    sns.kdeplot(data=yolo_y_errors, ax=ax2, label='YOLO', color='green', alpha=0.7, linewidth=2)
    sns.kdeplot(data=filter_y_errors, ax=ax2, label='Filter', color='red', alpha=0.7, linewidth=2)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax2.set_xlabel('Y Coordinate Error (True - Predicted)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Y Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. 오차 분포 히스토그램 ==========
    ax3 = fig2.add_subplot(gs2[0, 2])
    ax3.hist(cnn_errors, bins=30, alpha=0.6, label='CNN', color='blue', edgecolor='navy')
    ax3.hist(yolo_errors, bins=30, alpha=0.6, label='YOLO', color='green', edgecolor='darkgreen')
    ax3.hist(filter_errors, bins=30, alpha=0.6, label='Filter', color='red', edgecolor='darkred')
    ax3.axvline(np.mean(cnn_errors), color='blue', linestyle='--', linewidth=2, 
               label=f'CNN Mean: {np.mean(cnn_errors):.2f}')
    ax3.axvline(np.mean(yolo_errors), color='green', linestyle='--', linewidth=2, 
               label=f'YOLO Mean: {np.mean(yolo_errors):.2f}')
    ax3.axvline(np.mean(filter_errors), color='red', linestyle='--', linewidth=2, 
               label=f'Filter Mean: {np.mean(filter_errors):.2f}')
    ax3.set_xlabel('Pixel Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 2D 궤적 비교 (X-Y 평면) ==========
    ax4 = fig2.add_subplot(gs2[1, 0])
    ax4.scatter(cnn_pred[:, 0], cnn_pred[:, 1], 
               alpha=0.6, s=40, label='CNN', c='blue', edgecolors='navy')
    ax4.scatter(yolo_pred[:, 0], yolo_pred[:, 1], 
               alpha=0.6, s=40, label='YOLO', c='green', edgecolors='darkgreen')
    ax4.scatter(filter_pred[:, 0], filter_pred[:, 1], 
               alpha=0.6, s=40, label='Filter', c='red', edgecolors='darkred')
    ax4.scatter(true_center_x, true_center_y, 
               s=200, marker='*', c='gold', edgecolors='black', 
               linewidths=2, label='True Center', zorder=10)
    ax4.set_xlabel('X Coordinate', fontsize=11)
    ax4.set_ylabel('Y Coordinate', fontsize=11)
    ax4.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. 오차 박스플롯 ==========
    ax5 = fig2.add_subplot(gs2[1, 1])
    box_data = [cnn_errors, yolo_errors, filter_errors]
    box = ax5.boxplot(box_data, labels=['CNN', 'YOLO', 'Filter'], patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax5.set_ylabel('Pixel Error', fontsize=11)
    ax5.set_title('Error Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== 6. 누적 분포 함수 (CDF) ==========
    ax6 = fig2.add_subplot(gs2[1, 2])
    ax6.plot(cnn_sorted, cnn_cdf, label='CNN', color='blue', linewidth=2)
    ax6.plot(yolo_sorted, yolo_cdf, label='YOLO', color='green', linewidth=2)
    ax6.plot(filter_sorted, filter_cdf, label='Filter', color='red', linewidth=2)
    ax6.set_xlabel('Pixel Error', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ========== 전체 제목 (간소화 버전) ==========
    fig2.suptitle(
        f'CNN vs YOLO vs Filter Comparison (Compact)\n'
        f'CNN: {np.mean(cnn_errors):.2f}±{np.std(cnn_errors):.2f} px | '
        f'YOLO: {np.mean(yolo_errors):.2f}±{np.std(yolo_errors):.2f} px | '
        f'Filter: {np.mean(filter_errors):.2f}±{np.std(filter_errors):.2f} px',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 저장 (간소화 버전)
    if save_path:
        compact_save_path = save_path.replace('.png', '_compact.png')
        plt.savefig(compact_save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Compact comparison plot saved to: {compact_save_path}")
    
    plt.show()
    
    # 추가 통계 출력
    print("\n" + "="*100)
    print("📊 3-WAY COMPARISON SUMMARY")
    print("="*100)
    print(f"\n{'Metric':<20} {'CNN (MobileNet)':<18} {'YOLO (Tiny)':<18} {'Filter (Kalman)':<18} {'Winner':<15}")
    print("-"*100)
    
    # 평균 오차
    cnn_mean = np.mean(cnn_errors)
    yolo_mean = np.mean(yolo_errors)
    filter_mean = np.mean(filter_errors)
    winner_mean = min([('CNN', cnn_mean), ('YOLO', yolo_mean), ('Filter', filter_mean)], key=lambda x: x[1])[0]
    print(f"{'Mean Error (px)':<20} {cnn_mean:<18.3f} {yolo_mean:<18.3f} {filter_mean:<18.3f} {winner_mean:<15}")
    
    # 표준편차
    cnn_std = np.std(cnn_errors)
    yolo_std = np.std(yolo_errors)
    filter_std = np.std(filter_errors)
    winner_std = min([('CNN', cnn_std), ('YOLO', yolo_std), ('Filter', filter_std)], key=lambda x: x[1])[0]
    print(f"{'Std Error (px)':<20} {cnn_std:<18.3f} {yolo_std:<18.3f} {filter_std:<18.3f} {winner_std:<15}")
    
    # 중앙값
    cnn_median = np.median(cnn_errors)
    yolo_median = np.median(yolo_errors)
    filter_median = np.median(filter_errors)
    winner_median = min([('CNN', cnn_median), ('YOLO', yolo_median), ('Filter', filter_median)], key=lambda x: x[1])[0]
    print(f"{'Median Error (px)':<20} {cnn_median:<18.3f} {yolo_median:<18.3f} {filter_median:<18.3f} {winner_median:<15}")
    
    # 최대 오차
    cnn_max = np.max(cnn_errors)
    yolo_max = np.max(yolo_errors)
    filter_max = np.max(filter_errors)
    winner_max = min([('CNN', cnn_max), ('YOLO', yolo_max), ('Filter', filter_max)], key=lambda x: x[1])[0]
    print(f"{'Max Error (px)':<20} {cnn_max:<18.3f} {yolo_max:<18.3f} {filter_max:<18.3f} {winner_max:<15}")
    
    print("-"*100)


def main():
    """메인 함수"""
    print("🎯 CNN vs YOLO vs Filter Comparison")
    print("="*80)
    
    # 경로 설정
    cnn_checkpoint = "/hai/home/jdj/dvs/cnn_sim/checkpoints_mobilenet_v2/mobilenet_best.pth"
    yolo_checkpoint = "/hai/home/jdj/dvs/yolo_sim/checkpoints_yolo_tiny_laser/yolo_tiny_laser_best.pth"
    filter_csv = "/hai/home/jdj/dvs/filter_sim/csv_results/spatial_filter_kalman.csv"
    bin_file = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    output_path = "/hai/home/jdj/dvs/cnn_yolo_filter_comparison.png"
    
    # 파일 존재 확인
    if not os.path.exists(cnn_checkpoint):
        print(f"❌ CNN checkpoint not found: {cnn_checkpoint}")
        return
    
    if not os.path.exists(yolo_checkpoint):
        print(f"❌ YOLO checkpoint not found: {yolo_checkpoint}")
        return
    
    if not os.path.exists(filter_csv):
        print(f"❌ Filter CSV not found: {filter_csv}")
        return
    
    if not os.path.exists(bin_file):
        print(f"❌ Bin file not found: {bin_file}")
        return
    
    # CNN 결과 로드
    cnn_results = load_cnn_predictions(
        checkpoint_path=cnn_checkpoint,
        bin_file_path=bin_file,
        max_frames=50,
        test_augmentation=False  # 공정한 비교를 위해 증강 비활성화
    )
    
    # YOLO 결과 로드
    yolo_results = load_yolo_predictions(
        checkpoint_path=yolo_checkpoint,
        bin_file_path=bin_file,
        max_frames=50,
        conf_threshold=0.3
    )
    
    # Filter 결과 로드
    filter_results = load_filter_predictions(filter_csv)
    
    # 샘플 수 맞추기 및 마지막 3개 제외 (temporal window 경계 문제 방지)
    min_samples = min(len(cnn_results['predictions']), 
                     len(yolo_results['predictions']), 
                     len(filter_results['predictions']))
    safe_samples = min_samples - 3  # 마지막 3개 제외 (안전 마진)
    
    cnn_results['predictions'] = cnn_results['predictions'][:safe_samples]
    cnn_results['targets'] = cnn_results['targets'][:safe_samples]
    yolo_results['predictions'] = yolo_results['predictions'][:safe_samples]
    yolo_results['targets'] = yolo_results['targets'][:safe_samples]
    filter_results['predictions'] = filter_results['predictions'][:safe_samples]
    filter_results['targets'] = filter_results['targets'][:safe_samples]
    
    print(f"\n📊 Using {safe_samples} samples for comparison (excluded last 3 for stability)")
    
    # 비교 시각화
    compare_methods(cnn_results, yolo_results, filter_results, save_path=output_path)
    
    print(f"\n✅ Comparison completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()


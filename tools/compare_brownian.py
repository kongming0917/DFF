#!/usr/bin/env python3
"""
CNN vs YOLO vs Filter 비교 스크립트 (Brownian Motion 데이터셋)

- CNN   : cnn/ 체크포인트를 dvslib 파이프라인으로 추론 (tools/_common.cnn_predict)
- YOLO  : yolo/ 체크포인트를 dvslib 파이프라인으로 추론 (tools/_common.yolo_predict)
- Filter: filter/results CSV (filter/run.py 출력)
세 방식 모두 처음 --max-frames 프레임의 순차 sliding window를 사용한다 (옛 동작 유지).
blocked split 기반 정량 비교는 Phase 2(wandb)에서 대체 예정.

  python tools/compare_brownian.py                # 기본 경로 (cnn baseline, yolo/filter 산출물)
  python tools/compare_brownian.py --max-frames 300 --out compare_result/xxx.png
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from _common import ROOT, DATA, cnn_predict, yolo_predict  # noqa: E402  (tools/ 안에서 실행)


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """정답 CSV 파일 로드"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} ground truth records")
    return df


def load_cnn_predictions(checkpoint_path: str, max_frames: int = 100, device: str = "cuda") -> Dict:
    """CNN 추론 결과 (ROI 내부 픽셀 좌표). dvslib 파이프라인 사용, 처음 max_frames의 순차 window."""
    print("🤖 Loading CNN predictions...")
    r = cnn_predict(checkpoint_path, device=device, max_frames=max_frames, split="all")
    errors = r["pixel_errors"]
    print(f"✅ CNN predictions loaded: {len(errors)} samples")
    print(f"   Mean error: {errors.mean():.2f}±{errors.std():.2f} pixels")
    return {
        'predictions': r["predictions_px"],
        'targets': r["targets_px"],
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'method': f"CNN ({r['label'].split('/')[-1]})",
    }


def load_yolo_predictions(checkpoint_path: str, max_frames: int = 100, conf_threshold: float = 0.6,
                          device: str = "cuda") -> Dict:
    """YOLO 추론 결과 (ROI 내부 픽셀 좌표). dvslib 파이프라인 사용, 처음 max_frames의 순차 window."""
    print("🎯 Loading YOLO predictions...")
    r = yolo_predict(checkpoint_path, device=device, max_frames=max_frames, split="all",
                     conf_threshold=conf_threshold)
    errors = r["pixel_errors"]
    print(f"✅ YOLO predictions loaded: {len(errors)} samples "
          f"(detection rate {r['metrics']['detection_rate']:.1f}%)")
    print(f"   Mean error: {errors.mean():.2f}±{errors.std():.2f} pixels")
    return {
        'predictions': r["predictions_px"],
        'targets': r["targets_px"],
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'method': 'YOLO (Tiny)',
    }


def load_filter_predictions(
    csv_file_path: str,
    ground_truth_csv: str,
    temporal_window: int = 5,
    max_frames: int = 100
) -> Dict:
    """Filter 결과를 CSV에서 로드하고 정답과 비교
    Filter CSV의 각 행이 bin 파일을 읽는 순서대로 생성되었으므로,
    Filter CSV의 i번째 행 = 프레임 인덱스 i (0부터 시작)
    Filter CSV의 frame_number를 프레임 인덱스로 변환하여 GT CSV와 매칭
    """
    print("🔍 Loading Filter predictions...")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Filter CSV file not found: {csv_file_path}")
    
    # Filter 예측 CSV 로드 (원래 순서 유지)
    pred_df = pd.read_csv(csv_file_path)
    pred_df = pred_df.dropna()  # 유효한 데이터만
    
    # 정답 CSV 로드
    gt_df = load_ground_truth(ground_truth_csv)
    gt_df.set_index('frame_idx', inplace=True)  # frame_idx를 인덱스로 설정
    
    # Filter CSV의 각 행이 bin 파일을 읽는 순서대로 생성되었으므로
    # Filter CSV의 i번째 행 = 프레임 인덱스 i (0부터 시작)
    # frame_number는 무시하고 순서만 사용
    
    # CNN/YOLO와 동일한 temporal window 샘플 생성
    # CNN 샘플 i: frame 인덱스 [i, i+1, i+2, i+3, i+4] 사용, center_frame 인덱스 = i+2
    num_samples = max_frames - temporal_window + 1
    predictions = []
    targets = []
    
    for sample_idx in range(num_samples):
        center_frame_idx = sample_idx + temporal_window // 2  # center_frame 인덱스 = sample_idx + 2
        
        # Filter CSV에서 center_frame_idx번째 행 찾기 (순서대로)
        # Filter CSV의 i번째 행 = 프레임 인덱스 i
        if center_frame_idx < len(pred_df):
            row = pred_df.iloc[center_frame_idx]
            
            # predictions 추출
            pred = np.array([row['center_x'], row['center_y']])
            
            # GT에서 프레임 인덱스로 target 찾기
            if center_frame_idx in gt_df.index:
                gt_row = gt_df.loc[center_frame_idx]
                # Filter는 roi_center_x/y를 target으로 사용 (evaluate_against_ground_truth.py와 동일)
                target = np.array([gt_row['roi_center_x'], gt_row['roi_center_y']])
            else:
                # GT에 없으면 스킵
                continue
            
            predictions.append(pred)
            targets.append(target)
        else:
            # Filter CSV에 해당 인덱스가 없으면 스킵
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 오차 계산
    errors = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"✅ Filter predictions loaded: {len(predictions)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f} pixels")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'Filter (Kalman)'
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
    
    # 샘플 수 맞추기 (최소값 기준)
    min_samples = min(len(cnn_pred), len(yolo_pred), len(filter_pred))
    cnn_pred = cnn_pred[:min_samples]
    cnn_target = cnn_target[:min_samples]
    yolo_pred = yolo_pred[:min_samples]
    yolo_target = yolo_target[:min_samples]
    filter_pred = filter_pred[:min_samples]
    filter_target = filter_target[:min_samples]
    
    print(f"   Using {min_samples} samples for comparison")
    
    # 오차 계산
    cnn_errors = np.sqrt(np.sum((cnn_pred - cnn_target)**2, axis=1))
    yolo_errors = np.sqrt(np.sum((yolo_pred - yolo_target)**2, axis=1))
    filter_errors = np.sqrt(np.sum((filter_pred - filter_target)**2, axis=1))
    
    # 그래프 생성
    fig = fig = plt.figure(figsize=(20, 12))
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
    # Brownian motion은 시간에 따라 움직이므로 궤적을 선으로 표시
    ax4.plot(cnn_target[:, 0], cnn_target[:, 1], 'k-', alpha=0.6, linewidth=2, label='GT Trajectory', marker='o', markersize=3)
    ax4.plot(cnn_pred[:, 0], cnn_pred[:, 1], 'b--', alpha=0.6, linewidth=1, label='CNN Pred', marker='x', markersize=2)
    ax4.plot(yolo_pred[:, 0], yolo_pred[:, 1], 'g--', alpha=0.6, linewidth=1, label='YOLO Pred', marker='x', markersize=2)
    ax4.plot(filter_pred[:, 0], filter_pred[:, 1], 'r--', alpha=0.6, linewidth=1, label='Filter Pred', marker='x', markersize=2)
    
    ax4.set_xlabel('X Coordinate', fontsize=11)
    ax4.set_ylabel('Y Coordinate', fontsize=11)
    ax4.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
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
        f'CNN vs YOLO vs Filter Comparison (Brownian Motion)\n'
        f'CNN: {np.mean(cnn_errors):.2f}±{np.std(cnn_errors):.2f} px | '
        f'YOLO: {np.mean(yolo_errors):.2f}±{np.std(yolo_errors):.2f} px | '
        f'Filter: {np.mean(filter_errors):.2f}±{np.std(filter_errors):.2f} px',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 저장 (전체 버전)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Full comparison plot saved to: {save_path}")
    
    plt.close()
    
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
    
    # ========== 4. 2D 궤적 비교 ==========
    ax4 = fig2.add_subplot(gs2[1, 0])
    ax4.plot(cnn_target[:, 0], cnn_target[:, 1], 'b-', alpha=0.6, linewidth=2, label='GT', marker='o', markersize=2)
    ax4.plot(cnn_pred[:, 0], cnn_pred[:, 1], 'b--', alpha=0.6, linewidth=1, label='CNN', marker='x', markersize=1)
    ax4.plot(yolo_pred[:, 0], yolo_pred[:, 1], 'g--', alpha=0.6, linewidth=1, label='YOLO', marker='x', markersize=1)
    ax4.plot(filter_pred[:, 0], filter_pred[:, 1], 'r--', alpha=0.6, linewidth=1, label='Filter', marker='x', markersize=1)
    ax4.set_xlabel('X Coordinate', fontsize=11)
    ax4.set_ylabel('Y Coordinate', fontsize=11)
    ax4.set_title('Trajectory Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # ========== 5. 오차 박스플롯 ==========
    ax5 = fig2.add_subplot(gs2[1, 1])
    box = ax5.boxplot(box_data, labels=['CNN', 'YOLO', 'Filter'], patch_artist=True)
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
        f'CNN vs YOLO vs Filter Comparison (Brownian Motion - Compact)\n'
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
    
    plt.close()
    
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
    ap = argparse.ArgumentParser(description="CNN vs YOLO vs Filter comparison (brownian)")
    ap.add_argument("--cnn-checkpoint",
                    default=os.path.join(ROOT, "cnn", "runs", "baseline_mobilenet_v2", "mobilenet_v2_best.pth"))
    ap.add_argument("--yolo-checkpoint",
                    default=os.path.join(ROOT, "yolo", "runs", "baseline_yolo_tiny", "yolo_tiny_best.pth"))
    ap.add_argument("--filter-csv",
                    default=os.path.join(ROOT, "filter", "results", "no_filter_kalman.csv"))
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(ROOT, "compare_result", "cnn_yolo_filter_comparison_brownian.png"))
    args = ap.parse_args()

    print("🎯 CNN vs YOLO vs Filter Comparison (Brownian Motion)")
    print("=" * 80)
    bin_file = os.path.join(DATA, "gaussian_brownian_512x512.bin")
    ground_truth_csv = os.path.join(DATA, "gaussian_brownian_512x512_labels.csv")
    for path, hint in [
        (args.cnn_checkpoint, "python cnn/train.py --model mobilenet_v2"),
        (args.yolo_checkpoint, "python yolo/train.py"),
        (args.filter_csv, "python filter/run.py --max-frames 3000"),
        (ground_truth_csv, "python tools/generate_brownian_dataset.py"),
        (bin_file, "python tools/generate_brownian_dataset.py"),
    ]:
        if not os.path.exists(path):
            print(f"❌ Not found: {path}\n   → {hint}")
            return 1

    cnn_results = load_cnn_predictions(args.cnn_checkpoint, max_frames=args.max_frames, device=args.device)
    yolo_results = load_yolo_predictions(
        args.yolo_checkpoint, max_frames=args.max_frames, conf_threshold=args.conf_threshold, device=args.device)
    filter_results = load_filter_predictions(
        csv_file_path=args.filter_csv, ground_truth_csv=ground_truth_csv,
        temporal_window=5, max_frames=args.max_frames)

    min_samples = min(len(cnn_results['predictions']), len(yolo_results['predictions']),
                      len(filter_results['predictions']))
    for r in (cnn_results, yolo_results, filter_results):
        r['predictions'] = r['predictions'][:min_samples]
        r['targets'] = r['targets'][:min_samples]
    print(f"\n📊 Using {min_samples} samples for comparison")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    compare_methods(cnn_results, yolo_results, filter_results, save_path=args.out)
    print(f"\n✅ Comparison completed! Results saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CNN (MobileNet) vs Filter (Kalman) 중심값 추정 결과 비교 스크립트

cnn_sim의 MobileNet 모델 추론 결과와 
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
sys.path.append('/hai/home/jdj/dvs/cnn_sim')
sys.path.append('/hai/home/jdj/dvs/filter_sim')

from cnn_sim.inference import DVSInference
from filter_sim.dvs_filter import BinProcessor


def load_cnn_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    max_frames: int = 50,
    test_augmentation: bool = True
) -> Dict:
    """CNN 모델의 추론 결과 로드"""
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
    
    # numpy 배열로 변환
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    print(f"✅ CNN predictions loaded: {len(predictions)} samples")
    print(f"   Mean error: {results['mean_error']:.2f}±{results['std_error']:.2f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'mean_error': results['mean_error'],
        'std_error': results['std_error'],
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


def compare_methods(
    cnn_results: Dict,
    filter_results: Dict,
    save_path: Optional[str] = None
):
    """두 방법의 결과를 한 그래프에 비교"""
    
    print("\n📊 Creating comparison visualization...")
    
    # 데이터 추출
    cnn_pred = cnn_results['predictions']
    cnn_target = cnn_results['targets']
    filter_pred = filter_results['predictions']
    filter_target = filter_results['targets']
    
    # 오차 계산
    cnn_errors = np.sqrt(np.sum((cnn_pred - cnn_target)**2, axis=1))
    filter_errors = np.sqrt(np.sum((filter_pred - filter_target)**2, axis=1))
    
    # 그래프 생성
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== 1. X 좌표 산점도 비교 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(cnn_target[:, 0], cnn_pred[:, 0], 
               alpha=0.6, s=30, label='CNN', c='blue', edgecolors='navy')
    ax1.scatter(filter_target[:, 0], filter_pred[:, 0], 
               alpha=0.6, s=30, label='Filter', c='red', edgecolors='darkred')
    
    # 완벽한 예측 선
    all_x = np.concatenate([cnn_target[:, 0], filter_target[:, 0]])
    ax1.plot([all_x.min(), all_x.max()], [all_x.min(), all_x.max()], 
            'k--', alpha=0.5, linewidth=2, label='Perfect')
    
    ax1.set_xlabel('True X', fontsize=11)
    ax1.set_ylabel('Predicted X', fontsize=11)
    ax1.set_title('X Coordinate Prediction Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Y 좌표 산점도 비교 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(cnn_target[:, 1], cnn_pred[:, 1], 
               alpha=0.6, s=30, label='CNN', c='blue', edgecolors='navy')
    ax2.scatter(filter_target[:, 1], filter_pred[:, 1], 
               alpha=0.6, s=30, label='Filter', c='red', edgecolors='darkred')
    
    # 완벽한 예측 선
    all_y = np.concatenate([cnn_target[:, 1], filter_target[:, 1]])
    ax2.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 
            'k--', alpha=0.5, linewidth=2, label='Perfect')
    
    ax2.set_xlabel('True Y', fontsize=11)
    ax2.set_ylabel('Predicted Y', fontsize=11)
    ax2.set_title('Y Coordinate Prediction Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. 오차 분포 히스토그램 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(cnn_errors, bins=30, alpha=0.6, label='CNN', color='blue', edgecolor='navy')
    ax3.hist(filter_errors, bins=30, alpha=0.6, label='Filter', color='red', edgecolor='darkred')
    ax3.axvline(np.mean(cnn_errors), color='blue', linestyle='--', linewidth=2, 
               label=f'CNN Mean: {np.mean(cnn_errors):.2f}')
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
    box_data = [cnn_errors, filter_errors]
    box = ax5.boxplot(box_data, labels=['CNN', 'Filter'], patch_artist=True)
    
    # 색상 설정
    colors = ['lightblue', 'lightcoral']
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
    ax7.plot(filter_errors, alpha=0.7, linewidth=1.5, label='Filter', color='red')
    ax7.axhline(np.mean(cnn_errors), color='blue', linestyle='--', alpha=0.5)
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
    table[(1, 0)].set_facecolor('#E3F2FD')
    table[(2, 0)].set_facecolor('#FFEBEE')
    
    ax8.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    # ========== 전체 제목 ==========
    fig.suptitle(
        f'CNN (MobileNet) vs Filter (Kalman) Comparison\n'
        f'CNN: {np.mean(cnn_errors):.2f}±{np.std(cnn_errors):.2f} px | '
        f'Filter: {np.mean(filter_errors):.2f}±{np.std(filter_errors):.2f} px',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # 추가 통계 출력
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<20} {'CNN (MobileNet)':<20} {'Filter (Kalman)':<20} {'Winner':<15}")
    print("-"*80)
    
    # 평균 오차
    cnn_mean = np.mean(cnn_errors)
    filter_mean = np.mean(filter_errors)
    winner_mean = 'CNN' if cnn_mean < filter_mean else 'Filter'
    print(f"{'Mean Error (px)':<20} {cnn_mean:<20.3f} {filter_mean:<20.3f} {winner_mean:<15}")
    
    # 표준편차
    cnn_std = np.std(cnn_errors)
    filter_std = np.std(filter_errors)
    winner_std = 'CNN' if cnn_std < filter_std else 'Filter'
    print(f"{'Std Error (px)':<20} {cnn_std:<20.3f} {filter_std:<20.3f} {winner_std:<15}")
    
    # 중앙값
    cnn_median = np.median(cnn_errors)
    filter_median = np.median(filter_errors)
    winner_median = 'CNN' if cnn_median < filter_median else 'Filter'
    print(f"{'Median Error (px)':<20} {cnn_median:<20.3f} {filter_median:<20.3f} {winner_median:<15}")
    
    # 최대 오차
    cnn_max = np.max(cnn_errors)
    filter_max = np.max(filter_errors)
    winner_max = 'CNN' if cnn_max < filter_max else 'Filter'
    print(f"{'Max Error (px)':<20} {cnn_max:<20.3f} {filter_max:<20.3f} {winner_max:<15}")
    
    print("-"*80)


def main():
    """메인 함수"""
    print("🎯 CNN (MobileNet) vs Filter (Kalman) Comparison")
    print("="*80)
    
    # 경로 설정
    cnn_checkpoint = "/hai/home/jdj/dvs/cnn_sim/checkpoints_mobilenet_v2/mobilenet_best.pth"
    filter_csv = "/hai/home/jdj/dvs/filter_sim/csv_results/spatial_filter_kalman.csv"
    bin_file = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    output_path = "/hai/home/jdj/dvs/cnn_vs_filter_comparison.png"
    
    # 파일 존재 확인
    if not os.path.exists(cnn_checkpoint):
        print(f"❌ CNN checkpoint not found: {cnn_checkpoint}")
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
    
    # Filter 결과 로드
    filter_results = load_filter_predictions(filter_csv)
    
    # 샘플 수 맞추기 (더 적은 쪽에 맞춤)
    min_samples = min(len(cnn_results['predictions']), len(filter_results['predictions']))
    cnn_results['predictions'] = cnn_results['predictions'][:min_samples]
    cnn_results['targets'] = cnn_results['targets'][:min_samples]
    filter_results['predictions'] = filter_results['predictions'][:min_samples]
    filter_results['targets'] = filter_results['targets'][:min_samples]
    
    print(f"\n📊 Using {min_samples} samples for comparison")
    
    # 비교 시각화
    compare_methods(cnn_results, filter_results, save_path=output_path)
    
    print(f"\n✅ Comparison completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()


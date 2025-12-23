#!/usr/bin/env python3
"""
Brownian Motion 데이터셋에 대한 Filter 성능 평가

CSV 정답 데이터와 추출된 중심점을 비교하여 정확도를 평가합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Optional

def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """정답 CSV 파일 로드"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} ground truth records from {os.path.basename(csv_path)}")
    return df

def load_predictions(csv_path: str) -> pd.DataFrame:
    """추출된 중심점 CSV 파일 로드"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} prediction records from {os.path.basename(csv_path)}")
    return df

def calculate_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
    """
    예측값과 정답값을 비교하여 메트릭 계산
    
    Args:
        predictions_df: 추출된 중심점 (frame_number, center_x, center_y)
        ground_truth_df: 정답 데이터 (frame_idx, roi_center_x, roi_center_y)
    
    Returns:
        메트릭 딕셔너리
    """
    # 데이터 병합 (frame_number 기준)
    merged = pd.merge(
        predictions_df,
        ground_truth_df,
        left_on='frame_number',
        right_on='frame_idx',
        how='inner'
    )
    
    # 유효한 데이터만 필터링 (둘 다 None이 아닌 경우)
    valid = merged.dropna(subset=['center_x', 'center_y', 'roi_center_x', 'roi_center_y'])
    
    if len(valid) == 0:
        return {"error": "No valid data points for comparison"}
    
    # 오차 계산
    error_x = valid['center_x'] - valid['roi_center_x']
    error_y = valid['center_y'] - valid['roi_center_y']
    pixel_errors = np.sqrt(error_x**2 + error_y**2)
    
    # 메트릭 계산
    metrics = {
        'total_frames': len(merged),
        'valid_frames': len(valid),
        'invalid_frames': len(merged) - len(valid),
        
        # X/Y 오차
        'mae_x': np.mean(np.abs(error_x)),
        'mae_y': np.mean(np.abs(error_y)),
        'rmse_x': np.sqrt(np.mean(error_x**2)),
        'rmse_y': np.sqrt(np.mean(error_y**2)),
        
        # 전체 오차
        'mae': np.mean(pixel_errors),
        'rmse': np.sqrt(np.mean(pixel_errors**2)),
        'median_error': np.median(pixel_errors),
        'max_error': np.max(pixel_errors),
        'std_error': np.std(pixel_errors),
        
        # 정확도 (임계값별)
        'acc_5px': np.mean(pixel_errors <= 5.0) * 100,   # 5픽셀 이내
        'acc_10px': np.mean(pixel_errors <= 10.0) * 100, # 10픽셀 이내
        'acc_20px': np.mean(pixel_errors <= 20.0) * 100,  # 20픽셀 이내
        
        # 상관관계
        'corr_x': valid['center_x'].corr(valid['roi_center_x']),
        'corr_y': valid['center_y'].corr(valid['roi_center_y']),
    }
    
    return metrics

def print_metrics(metrics: Dict, method_name: str = ""):
    """메트릭 출력"""
    if "error" in metrics:
        print(f"❌ {metrics['error']}")
        return
    
    print(f"\n📊 Metrics for {method_name}")
    print("=" * 80)
    print(f"Valid frames: {metrics['valid_frames']}/{metrics['total_frames']} ({metrics['valid_frames']/metrics['total_frames']*100:.1f}%)")
    print(f"\n📍 Error Metrics:")
    print(f"   MAE:  {metrics['mae']:.2f} pixels")
    print(f"   RMSE: {metrics['rmse']:.2f} pixels")
    print(f"   Median: {metrics['median_error']:.2f} pixels")
    print(f"   Max:    {metrics['max_error']:.2f} pixels")
    print(f"   Std:    {metrics['std_error']:.2f} pixels")
    
    print(f"\n📐 Component Errors:")
    print(f"   X - MAE: {metrics['mae_x']:.2f}, RMSE: {metrics['rmse_x']:.2f}")
    print(f"   Y - MAE: {metrics['mae_y']:.2f}, RMSE: {metrics['rmse_y']:.2f}")
    
    print(f"\n🎯 Accuracy (threshold-based):")
    print(f"   @5px:  {metrics['acc_5px']:.1f}%")
    print(f"   @10px: {metrics['acc_10px']:.1f}%")
    print(f"   @20px: {metrics['acc_20px']:.1f}%")
    
    print(f"\n📈 Correlation:")
    print(f"   X: {metrics['corr_x']:.4f}")
    print(f"   Y: {metrics['corr_y']:.4f}")

def visualize_comparison(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    method_name: str = "",
    save_path: Optional[str] = None
):
    """예측값과 정답값 비교 시각화"""
    # 데이터 병합
    merged = pd.merge(
        predictions_df,
        ground_truth_df,
        left_on='frame_number',
        right_on='frame_idx',
        how='inner'
    )
    
    valid = merged.dropna(subset=['center_x', 'center_y', 'roi_center_x', 'roi_center_y'])
    
    if len(valid) == 0:
        print("❌ No valid data for visualization")
        return
    
    # 오차 계산
    error_x = valid['center_x'] - valid['roi_center_x']
    error_y = valid['center_y'] - valid['roi_center_y']
    pixel_errors = np.sqrt(error_x**2 + error_y**2)
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Filter Performance Evaluation: {method_name}', fontsize=16, fontweight='bold')
    
    # 1. X 좌표 산점도 (예측 vs 정답)
    axes[0, 0].scatter(valid['roi_center_x'], valid['center_x'], alpha=0.6, s=20)
    axes[0, 0].plot([valid['roi_center_x'].min(), valid['roi_center_x'].max()],
                    [valid['roi_center_x'].min(), valid['roi_center_x'].max()],
                    'r--', linewidth=2, label='Perfect')
    axes[0, 0].set_xlabel('Ground Truth X')
    axes[0, 0].set_ylabel('Predicted X')
    axes[0, 0].set_title('X Coordinate: Predicted vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² 계산
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(valid['roi_center_x'], valid['center_x'])
    r2_x = r_value**2
    axes[0, 0].text(0.05, 0.95, f'R² = {r2_x:.4f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Y 좌표 산점도 (예측 vs 정답)
    axes[0, 1].scatter(valid['roi_center_y'], valid['center_y'], alpha=0.6, s=20)
    axes[0, 1].plot([valid['roi_center_y'].min(), valid['roi_center_y'].max()],
                    [valid['roi_center_y'].min(), valid['roi_center_y'].max()],
                    'r--', linewidth=2, label='Perfect')
    axes[0, 1].set_xlabel('Ground Truth Y')
    axes[0, 1].set_ylabel('Predicted Y')
    axes[0, 1].set_title('Y Coordinate: Predicted vs Ground Truth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    slope, intercept, r_value, p_value, std_err = linregress(valid['roi_center_y'], valid['center_y'])
    r2_y = r_value**2
    axes[0, 1].text(0.05, 0.95, f'R² = {r2_y:.4f}', transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 픽셀 오차 히스토그램
    axes[0, 2].hist(pixel_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.mean(pixel_errors), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(pixel_errors):.2f}px')
    axes[0, 2].axvline(np.median(pixel_errors), color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(pixel_errors):.2f}px')
    axes[0, 2].set_xlabel('Pixel Error')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Pixel Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 시간에 따른 오차
    valid_sorted = valid.sort_values('frame_number')
    pixel_errors_sorted = np.sqrt(
        (valid_sorted['center_x'] - valid_sorted['roi_center_x'])**2 +
        (valid_sorted['center_y'] - valid_sorted['roi_center_y'])**2
    )
    axes[1, 0].plot(valid_sorted['frame_number'], pixel_errors_sorted, alpha=0.7, linewidth=1)
    axes[1, 0].axhline(np.mean(pixel_errors), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(pixel_errors):.2f}px')
    axes[1, 0].set_xlabel('Frame Number')
    axes[1, 0].set_ylabel('Pixel Error')
    axes[1, 0].set_title('Pixel Error Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 오차 벡터 (X vs Y 오차)
    scatter = axes[1, 1].scatter(error_x, error_y, c=pixel_errors, alpha=0.6, s=20, cmap='viridis')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('X Error (pixels)')
    axes[1, 1].set_ylabel('Y Error (pixels)')
    axes[1, 1].set_title('Error Vector Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Pixel Error')
    
    # 6. 궤적 비교
    axes[1, 2].plot(valid_sorted['roi_center_x'], valid_sorted['roi_center_y'],
                   'b-', alpha=0.6, linewidth=2, label='Ground Truth', marker='o', markersize=3)
    axes[1, 2].plot(valid_sorted['center_x'], valid_sorted['center_y'],
                   'r--', alpha=0.6, linewidth=2, label='Predicted', marker='x', markersize=3)
    axes[1, 2].set_xlabel('X Coordinate')
    axes[1, 2].set_ylabel('Y Coordinate')
    axes[1, 2].set_title('Trajectory Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.close()

def compare_methods(
    prediction_files: Dict[str, str],
    ground_truth_file: str,
    output_dir: str = "evaluation_results"
):
    """여러 방법들의 성능 비교"""
    print("🚀 Filter Performance Evaluation")
    print("=" * 80)
    
    # 정답 데이터 로드
    gt_df = load_ground_truth(ground_truth_file)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 방법별로 평가
    all_metrics = {}
    for method_name, pred_file in prediction_files.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print('='*80)
        
        # 예측 데이터 로드
        pred_df = load_predictions(pred_file)
        
        # 메트릭 계산
        metrics = calculate_metrics(pred_df, gt_df)
        all_metrics[method_name] = metrics
        
        # 메트릭 출력
        print_metrics(metrics, method_name)
        
        # 시각화
        viz_path = os.path.join(output_dir, f"{method_name}_evaluation.png")
        visualize_comparison(pred_df, gt_df, method_name, viz_path)
    
    # 비교 요약
    print(f"\n{'='*80}")
    print("📋 PERFORMANCE COMPARISON SUMMARY")
    print('='*80)
    print(f"{'Method':<30} {'MAE':<10} {'RMSE':<10} {'Median':<10} {'@5px':<10} {'@10px':<10} {'@20px':<10}")
    print("-" * 100)
    
    for method_name, metrics in all_metrics.items():
        if "error" not in metrics:
            print(f"{method_name:<30} "
                  f"{metrics['mae']:<10.2f} "
                  f"{metrics['rmse']:<10.2f} "
                  f"{metrics['median_error']:<10.2f} "
                  f"{metrics['acc_5px']:<10.1f}% "
                  f"{metrics['acc_10px']:<10.1f}% "
                  f"{metrics['acc_20px']:<10.1f}%")
    
    # 비교 시각화
    if len(all_metrics) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Method Comparison', fontsize=16, fontweight='bold')
        
        methods = [m for m in all_metrics.keys() if "error" not in all_metrics[m]]
        mae_values = [all_metrics[m]['mae'] for m in methods]
        rmse_values = [all_metrics[m]['rmse'] for m in methods]
        acc_5px = [all_metrics[m]['acc_5px'] for m in methods]
        acc_10px = [all_metrics[m]['acc_10px'] for m in methods]
        
        x_pos = np.arange(len(methods))
        
        # MAE 비교
        axes[0, 0].bar(x_pos, mae_values, alpha=0.7)
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('MAE (pixels)')
        axes[0, 0].set_title('Mean Absolute Error')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE 비교
        axes[0, 1].bar(x_pos, rmse_values, alpha=0.7)
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('RMSE (pixels)')
        axes[0, 1].set_title('Root Mean Squared Error')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy @5px 비교
        axes[1, 0].bar(x_pos, acc_5px, alpha=0.7)
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Accuracy @5px')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy @10px 비교
        axes[1, 1].bar(x_pos, acc_10px, alpha=0.7)
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Accuracy @10px')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "method_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison visualization saved to: {comparison_path}")
        plt.close()
    
    print(f"\n📁 All results saved in: {output_dir}/")

if __name__ == "__main__":
    # 설정
    GROUND_TRUTH_FILE = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512_labels.csv"
    
    # 평가할 방법들 (CSV 파일 경로)
    PREDICTION_FILES = {
        "median_no_filter": "csv_results/no_filter_median.csv",
        "median_spatial_filter": "csv_results/spatial_filter_median.csv",
        "kalman_no_filter": "csv_results/no_filter_kalman.csv",
        "kalman_spatial_filter": "csv_results/spatial_filter_kalman.csv",
    }
    
    # 평가 실행
    compare_methods(
        prediction_files=PREDICTION_FILES,
        ground_truth_file=GROUND_TRUTH_FILE,
        output_dir="evaluation_results"
    )


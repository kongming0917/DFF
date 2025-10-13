#!/usr/bin/env python3
"""
DVS 중심점 데이터 분석 및 시각화 유틸리티

CSV 파일로 저장된 중심점 데이터를 읽어서:
1. 히스토그램으로 분포 시각화
2. 통계적 분석 (평균, 분산, 표준편차 등)
3. 필터별 비교 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from typing import Dict, List, Tuple, Optional

def load_center_data(csv_file_path: str) -> pd.DataFrame:
    """CSV 파일에서 중심점 데이터 로드"""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    print(f"📂 Loaded {len(df)} records from {os.path.basename(csv_file_path)}")
    return df

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """중심점 데이터의 통계 정보 계산"""
    # 유효한 데이터만 필터링
    valid_data = df.dropna()
    
    if len(valid_data) == 0:
        return {"error": "No valid data points"}
    
    stats_dict = {
        # 기본 통계
        "total_frames": len(df),
        "valid_frames": len(valid_data),
        "invalid_frames": len(df) - len(valid_data),
        
        # X 좌표 통계
        "x_mean": valid_data['center_x'].mean(),
        "x_std": valid_data['center_x'].std(),
        "x_var": valid_data['center_x'].var(),
        "x_min": valid_data['center_x'].min(),
        "x_max": valid_data['center_x'].max(),
        "x_median": valid_data['center_x'].median(),
        "x_range": valid_data['center_x'].max() - valid_data['center_x'].min(),
        
        # Y 좌표 통계
        "y_mean": valid_data['center_y'].mean(),
        "y_std": valid_data['center_y'].std(),
        "y_var": valid_data['center_y'].var(),
        "y_min": valid_data['center_y'].min(),
        "y_max": valid_data['center_y'].max(),
        "y_median": valid_data['center_y'].median(),
        "y_range": valid_data['center_y'].max() - valid_data['center_y'].min(),
        
        # 중심점간 거리 분석
        "displacement_mean": 0,
        "displacement_std": 0,
        "displacement_max": 0,
        
        # 추가 성능 지표
        "tracking_consistency": 0,  # 추적 일관성 (displacement_std의 역수)
        "outlier_ratio": 0,         # 아웃라이어 비율
        "convergence_rate": 0       # 수렴률 (후반부 vs 전반부 안정성)
    }
    
    # 연속 프레임간 중심점 이동거리 계산
    if len(valid_data) > 1:
        valid_data_sorted = valid_data.sort_values('frame_number')
        dx = valid_data_sorted['center_x'].diff().dropna()
        dy = valid_data_sorted['center_y'].diff().dropna()
        displacements = np.sqrt(dx**2 + dy**2)
        
        stats_dict["displacement_mean"] = displacements.mean()
        stats_dict["displacement_std"] = displacements.std()
        stats_dict["displacement_max"] = displacements.max()
        
        # 추적 일관성: displacement 표준편차가 낮을수록 일관성이 높음
        if displacements.std() > 0:
            stats_dict["tracking_consistency"] = 1.0 / displacements.std()
        else:
            stats_dict["tracking_consistency"] = float('inf')
        
        # 아웃라이어 비율: IQR 기준으로 아웃라이어 계산
        q1_x, q3_x = valid_data['center_x'].quantile([0.25, 0.75])
        q1_y, q3_y = valid_data['center_y'].quantile([0.25, 0.75])
        iqr_x, iqr_y = q3_x - q1_x, q3_y - q1_y
        
        outliers_x = (valid_data['center_x'] < (q1_x - 1.5 * iqr_x)) | (valid_data['center_x'] > (q3_x + 1.5 * iqr_x))
        outliers_y = (valid_data['center_y'] < (q1_y - 1.5 * iqr_y)) | (valid_data['center_y'] > (q3_y + 1.5 * iqr_y))
        outliers = outliers_x | outliers_y
        stats_dict["outlier_ratio"] = outliers.sum() / len(valid_data) * 100
        
        # 수렴률: 후반부와 전반부의 안정성 비교
        if len(valid_data) >= 20:  # 충분한 데이터가 있을 때만
            mid_point = len(valid_data) // 2
            first_half = valid_data.iloc[:mid_point]
            second_half = valid_data.iloc[mid_point:]
            
            first_half_std = np.sqrt(first_half['center_x'].std()**2 + first_half['center_y'].std()**2)
            second_half_std = np.sqrt(second_half['center_x'].std()**2 + second_half['center_y'].std()**2)
            
            if first_half_std > 0:
                # 개선률: (초기 - 후기) / 초기 * 100 (양수면 개선)
                stats_dict["convergence_rate"] = (first_half_std - second_half_std) / first_half_std * 100
            else:
                stats_dict["convergence_rate"] = 0
    
    return stats_dict

def print_statistics(stats: Dict, title: str):
    """통계 정보를 보기 좋게 출력"""
    print(f"\n📊 {title}")
    print("=" * 60)
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return
    
    # 기본 정보
    print(f"📋 기본 정보:")
    print(f"   전체 프레임: {stats['total_frames']}")
    print(f"   유효 프레임: {stats['valid_frames']}")
    print(f"   무효 프레임: {stats['invalid_frames']}")
    
    # X 좌표 통계
    print(f"\n📍 X 좌표 통계:")
    print(f"   평균: {stats['x_mean']:.2f}")
    print(f"   표준편차: {stats['x_std']:.2f}")
    print(f"   분산: {stats['x_var']:.2f}")
    print(f"   중앙값: {stats['x_median']:.2f}")
    print(f"   범위: {stats['x_min']:.1f} ~ {stats['x_max']:.1f} (폭: {stats['x_range']:.1f})")
    
    # Y 좌표 통계
    print(f"\n📍 Y 좌표 통계:")
    print(f"   평균: {stats['y_mean']:.2f}")
    print(f"   표준편차: {stats['y_std']:.2f}")
    print(f"   분산: {stats['y_var']:.2f}")
    print(f"   중앙값: {stats['y_median']:.2f}")
    print(f"   범위: {stats['y_min']:.1f} ~ {stats['y_max']:.1f} (폭: {stats['y_range']:.1f})")
    
    # 이동거리 통계
    print(f"\n🔄 프레임간 이동거리 통계:")
    print(f"   평균 이동거리: {stats['displacement_mean']:.2f}")
    print(f"   이동거리 표준편차: {stats['displacement_std']:.2f}")
    print(f"   최대 이동거리: {stats['displacement_max']:.2f}")
    
    # 성능 지표
    print(f"\n⚡ 성능 지표:")
    tracking_consistency = stats['tracking_consistency']
    if tracking_consistency == float('inf'):
        print(f"   추적 일관성: 완벽 (이동거리 변화 없음)")
    else:
        print(f"   추적 일관성: {tracking_consistency:.2f} (높을수록 좋음)")
    print(f"   아웃라이어 비율: {stats['outlier_ratio']:.1f}% (낮을수록 좋음)")
    print(f"   수렴 개선률: {stats['convergence_rate']:.1f}% (양수면 시간에 따라 개선)")

def plot_center_distribution(df: pd.DataFrame, title: str, save_path: Optional[str] = None):
    """중심점 분포 히스토그램 및 산점도 그리기"""
    valid_data = df.dropna()
    
    if len(valid_data) == 0:
        print(f"⚠️ No valid data to plot for {title}")
        return
    
    # 서브플롯 설정
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Center Point Analysis', fontsize=16, fontweight='bold')
    
    # 1. X 좌표 히스토그램
    axes[0, 0].hist(valid_data['center_x'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(valid_data['center_x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_data["center_x"].mean():.1f}')
    axes[0, 0].axvline(valid_data['center_x'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {valid_data["center_x"].median():.1f}')
    axes[0, 0].set_xlabel('X Coordinate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('X Coordinate Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Y 좌표 히스토그램
    axes[0, 1].hist(valid_data['center_y'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(valid_data['center_y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_data["center_y"].mean():.1f}')
    axes[0, 1].axvline(valid_data['center_y'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {valid_data["center_y"].median():.1f}')
    axes[0, 1].set_xlabel('Y Coordinate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Y Coordinate Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 중심점 산점도 (궤적)
    axes[1, 0].scatter(valid_data['center_x'], valid_data['center_y'], alpha=0.6, s=20, c=valid_data['frame_number'], cmap='viridis')
    axes[1, 0].plot(valid_data['center_x'], valid_data['center_y'], alpha=0.3, linewidth=1, color='gray')
    axes[1, 0].scatter(valid_data['center_x'].mean(), valid_data['center_y'].mean(), color='red', s=100, marker='*', label='Mean Center')
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    axes[1, 0].set_title('Center Point Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 시간에 따른 이동거리
    valid_data_sorted = valid_data.sort_values('frame_number')
    if len(valid_data_sorted) > 1:
        dx = valid_data_sorted['center_x'].diff().fillna(0)
        dy = valid_data_sorted['center_y'].diff().fillna(0)
        displacements = np.sqrt(dx**2 + dy**2)
        
        axes[1, 1].plot(valid_data_sorted['frame_number'].iloc[1:], displacements.iloc[1:], alpha=0.7, linewidth=1)
        axes[1, 1].axhline(displacements.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {displacements.mean():.2f}')
        axes[1, 1].set_xlabel('Frame Number')
        axes[1, 1].set_ylabel('Displacement (pixels)')
        axes[1, 1].set_title('Frame-to-Frame Displacement')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor displacement analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()

def compare_datasets(datasets: Dict[str, pd.DataFrame], save_dir: Optional[str] = None):
    """여러 데이터셋 비교 분석"""
    print(f"\n🔍 COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # 통계 계산 및 출력
    all_stats = {}
    for name, df in datasets.items():
        stats = calculate_statistics(df)
        all_stats[name] = stats
        print_statistics(stats, name)
    
    # 성능 요약 표 출력
    print(f"\n📋 PERFORMANCE SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Method':<25} {'Stability':<12} {'Tracking':<12} {'Outliers':<10} {'Convergence':<12}")
    print("-" * 100)
    
    for name, stats in all_stats.items():
        if "error" not in stats:
            stability = np.sqrt(stats['x_std']**2 + stats['y_std']**2)
            tracking = stats['tracking_consistency']
            tracking_str = "Perfect" if tracking == float('inf') else f"{tracking:.2f}"
            
            print(f"{name:<25} {stability:<12.2f} {tracking_str:<12} {stats['outlier_ratio']:<10.1f} {stats['convergence_rate']:<12.1f}")
    
    print("-" * 100)
    print("Lower is better: Stability, Outliers")
    print("Higher is better: Tracking, Convergence (positive)")
    
    # 색상 설정
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    # ==================== 1. 겹쳐진 분포도 ====================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Overlayed Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1-1. X 좌표 겹쳐진 히스토그램
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna()
        if len(valid_data) > 0:
            axes1[0, 0].hist(valid_data['center_x'], bins=30, alpha=0.6, 
                           label=name, color=colors[i], edgecolor='black', linewidth=0.5)
    axes1[0, 0].set_xlabel('X Coordinate')
    axes1[0, 0].set_ylabel('Frequency')
    axes1[0, 0].set_title('X Coordinate Distribution Overlay')
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)
    
    # 1-2. Y 좌표 겹쳐진 히스토그램  
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna()
        if len(valid_data) > 0:
            axes1[0, 1].hist(valid_data['center_y'], bins=30, alpha=0.6, 
                           label=name, color=colors[i], edgecolor='black', linewidth=0.5)
    axes1[0, 1].set_xlabel('Y Coordinate')
    axes1[0, 1].set_ylabel('Frequency')
    axes1[0, 1].set_title('Y Coordinate Distribution Overlay')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)
    
    # 1-3. 겹쳐진 산점도 (중심점 궤적)
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna()
        if len(valid_data) > 0:
            axes1[1, 0].scatter(valid_data['center_x'], valid_data['center_y'], 
                              alpha=0.6, s=20, label=name, color=colors[i])
    axes1[1, 0].set_xlabel('X Coordinate')
    axes1[1, 0].set_ylabel('Y Coordinate')
    axes1[1, 0].set_title('Center Point Trajectories Overlay')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)
    
    # 1-4. 이동거리 비교
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna().sort_values('frame_number')
        if len(valid_data) > 1:
            dx = valid_data['center_x'].diff().fillna(0)
            dy = valid_data['center_y'].diff().fillna(0)
            displacements = np.sqrt(dx**2 + dy**2)
            axes1[1, 1].plot(valid_data['frame_number'].iloc[1:], displacements.iloc[1:], 
                           alpha=0.7, linewidth=2, label=name, color=colors[i])
    axes1[1, 1].set_xlabel('Frame Number')
    axes1[1, 1].set_ylabel('Displacement (pixels)')
    axes1[1, 1].set_title('Frame-to-Frame Displacement Comparison')
    axes1[1, 1].legend()
    axes1[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        overlay_path = os.path.join(save_dir, "overlay_comparison.png")
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        print(f"📊 Overlay comparison saved to: {overlay_path}")
    plt.show()
    
    # ==================== 2. 박스플롯 비교 ====================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Box Plot Comparison', fontsize=16, fontweight='bold')
    
    # 데이터 준비
    x_data = []
    y_data = []
    displacement_data = []
    labels = []
    
    for name, df in datasets.items():
        valid_data = df.dropna()
        if len(valid_data) > 0:
            x_data.append(valid_data['center_x'].values)
            y_data.append(valid_data['center_y'].values)
            labels.append(name)
            
            # 이동거리 계산
            if len(valid_data) > 1:
                valid_data_sorted = valid_data.sort_values('frame_number')
                dx = valid_data_sorted['center_x'].diff().dropna()
                dy = valid_data_sorted['center_y'].diff().dropna()
                displacements = np.sqrt(dx**2 + dy**2)
                displacement_data.append(displacements.values)
            else:
                displacement_data.append([0])
    
    # 2-1. X 좌표 박스플롯
    box1 = axes2[0].boxplot(x_data, labels=labels, patch_artist=True)
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[0].set_xlabel('Method')
    axes2[0].set_ylabel('X Coordinate')
    axes2[0].set_title('X Coordinate Distribution')
    axes2[0].grid(True, alpha=0.3)
    plt.setp(axes2[0].get_xticklabels(), rotation=45, ha='right')
    
    # 2-2. Y 좌표 박스플롯
    box2 = axes2[1].boxplot(y_data, labels=labels, patch_artist=True)
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[1].set_xlabel('Method')
    axes2[1].set_ylabel('Y Coordinate')
    axes2[1].set_title('Y Coordinate Distribution')
    axes2[1].grid(True, alpha=0.3)
    plt.setp(axes2[1].get_xticklabels(), rotation=45, ha='right')
    
    # 2-3. 이동거리 박스플롯
    box3 = axes2[2].boxplot(displacement_data, labels=labels, patch_artist=True)
    for patch, color in zip(box3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes2[2].set_xlabel('Method')
    axes2[2].set_ylabel('Displacement (pixels)')
    axes2[2].set_title('Frame-to-Frame Displacement')
    axes2[2].grid(True, alpha=0.3)
    plt.setp(axes2[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    if save_dir:
        boxplot_path = os.path.join(save_dir, "boxplot_comparison.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Boxplot comparison saved to: {boxplot_path}")
    plt.show()
    
    # ==================== 3. 밀도 분포 (KDE) 비교 ====================
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle('Kernel Density Estimation Comparison', fontsize=16, fontweight='bold')
    
    # 3-1. X 좌표 KDE
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna()
        if len(valid_data) > 0:
            sns.kdeplot(data=valid_data, x='center_x', ax=axes3[0], 
                       label=name, color=colors[i], linewidth=2)
    axes3[0].set_xlabel('X Coordinate')
    axes3[0].set_ylabel('Density')
    axes3[0].set_title('X Coordinate Density Distribution')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    # 3-2. Y 좌표 KDE
    for i, (name, df) in enumerate(datasets.items()):
        valid_data = df.dropna()
        if len(valid_data) > 0:
            sns.kdeplot(data=valid_data, x='center_y', ax=axes3[1], 
                       label=name, color=colors[i], linewidth=2)
    axes3[1].set_xlabel('Y Coordinate')
    axes3[1].set_ylabel('Density')
    axes3[1].set_title('Y Coordinate Density Distribution')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        kde_path = os.path.join(save_dir, "kde_comparison.png")
        plt.savefig(kde_path, dpi=300, bbox_inches='tight')
        print(f"📊 KDE comparison saved to: {kde_path}")
    plt.show()
    
    # ==================== 4. 성능 지표 바차트 ====================
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    fig4.suptitle('Performance Metrics Bar Chart', fontsize=16, fontweight='bold')
    
    # 데이터 준비
    names = []
    stability_scores = []
    tracking_scores = []
    outlier_ratios = []
    convergence_rates = []
    
    for name, stats in all_stats.items():
        if "error" not in stats:
            names.append(name)
            stability_scores.append(np.sqrt(stats['x_std']**2 + stats['y_std']**2))
            
            # tracking consistency 처리 (무한대 값 처리)
            tracking = stats['tracking_consistency']
            if tracking == float('inf'):
                tracking_scores.append(100)  # 무한대는 100으로 표시
            else:
                tracking_scores.append(min(tracking, 100))  # 100으로 상한 제한
            
            outlier_ratios.append(stats['outlier_ratio'])
            convergence_rates.append(stats['convergence_rate'])
    
    x_pos = np.arange(len(names))
    
    # 4-1. 전체 안정성 점수 (낮을수록 좋음)
    bars1 = axes4[0, 0].bar(x_pos, stability_scores, alpha=0.7, color=colors[:len(names)])
    axes4[0, 0].set_xlabel('Method')
    axes4[0, 0].set_ylabel('Stability Score (pixels)')
    axes4[0, 0].set_title('Overall Stability (Lower = Better)')
    axes4[0, 0].set_xticks(x_pos)
    axes4[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes4[0, 0].grid(True, alpha=0.3)
    
    # 4-2. 추적 일관성 (높을수록 좋음)
    bars2 = axes4[0, 1].bar(x_pos, tracking_scores, alpha=0.7, color=colors[:len(names)])
    axes4[0, 1].set_xlabel('Method')
    axes4[0, 1].set_ylabel('Tracking Consistency')
    axes4[0, 1].set_title('Tracking Consistency (Higher = Better)')
    axes4[0, 1].set_xticks(x_pos)
    axes4[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes4[0, 1].grid(True, alpha=0.3)
    
    # 4-3. 아웃라이어 비율 (낮을수록 좋음)
    bars3 = axes4[1, 0].bar(x_pos, outlier_ratios, alpha=0.7, color=colors[:len(names)])
    axes4[1, 0].set_xlabel('Method')
    axes4[1, 0].set_ylabel('Outlier Ratio (%)')
    axes4[1, 0].set_title('Outlier Ratio (Lower = Better)')
    axes4[1, 0].set_xticks(x_pos)
    axes4[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes4[1, 0].grid(True, alpha=0.3)
    
    # 4-4. 수렴 개선률 (양수가 좋음)
    bars4 = axes4[1, 1].bar(x_pos, convergence_rates, alpha=0.7, color=colors[:len(names)])
    axes4[1, 1].set_xlabel('Method')
    axes4[1, 1].set_ylabel('Convergence Rate (%)')
    axes4[1, 1].set_title('Convergence Improvement (Positive = Better)')
    axes4[1, 1].set_xticks(x_pos)
    axes4[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes4[1, 1].grid(True, alpha=0.3)
    axes4[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_dir:
        barplot_path = os.path.join(save_dir, "performance_barplot.png")
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance barplot saved to: {barplot_path}")
    plt.show()

def analyze_center_data(csv_files: List[str], output_dir: str = "analysis_results"):
    """메인 분석 함수"""
    print("🎯 DVS Center Point Data Analysis")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    datasets = {}
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            name = os.path.splitext(os.path.basename(csv_file))[0]
            datasets[name] = load_center_data(csv_file)
        else:
            print(f"⚠️ File not found: {csv_file}")
    
    if not datasets:
        print("❌ No valid CSV files found!")
        return
    
    # 개별 분석 및 시각화
    for name, df in datasets.items():
        print_statistics(calculate_statistics(df), name)
        plot_path = os.path.join(output_dir, f"{name}_analysis.png")
        plot_center_distribution(df, name, plot_path)
    
    # 비교 분석
    if len(datasets) > 1:
        compare_datasets(datasets, output_dir)

if __name__ == "__main__":
    # 분석할 CSV 파일들 설정
    csv_files = [
        # 기본 추출기들
        "csv_results/no_filter_median.csv",
        "csv_results/spatial_filter_median.csv",
        "csv_results/no_filter_kalman.csv",
        "csv_results/spatial_filter_kalman.csv",
        
        # 템포럴 평균 추출기들
        "csv_results_temporal/no_filter_temporal_avg_w3.csv", 
        "csv_results_temporal/spatial_filter_temporal_avg_w3.csv",
        "csv_results_temporal/no_filter_kalman.csv",
        "csv_results_temporal/spatial_filter_kalman.csv"
    ]
    
    # 분석 실행
    analyze_center_data(csv_files, output_dir="analysis_results")

#!/usr/bin/env python3
"""
MobileNetV2 일반 모델 vs QAT 양자화 모델 비교 스크립트

cnn_brownian_sim의 일반 MobileNetV2 모델과 
QAT(int8 양자화)를 적용한 MobileNetV2 모델의 추론 결과를 비교 분석합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import sys
from typing import Dict, Tuple, Optional, List
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')  # GUI 없는 환경에서만 Agg 사용

# 경로 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from cnn_brownian_sim.inference import DVSInference
from lib.bin_processor import BinProcessor


def load_model_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    csv_labels_path: str,
    max_frames: int = 50,
    use_quantized: bool = False,
    model_name: str = "Model"
) -> Dict:
    """모델의 추론 결과 로드 (상대 좌표)
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        bin_file_path: bin 파일 경로
        csv_labels_path: CSV 레이블 파일 경로
        max_frames: 최대 프레임 수
        use_quantized: 양자화된 모델 사용 여부
        model_name: 모델 이름 (표시용)
    """
    print(f"🤖 Loading {model_name} predictions...")
    
    # Inference 객체 생성
    try:
        inferencer = DVSInference(
            checkpoint_path=checkpoint_path,
            use_quantized=use_quantized
        )
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        raise e
    
    # 프레임 로드
    individual_frames = inferencer.load_frames_from_bin(
        bin_file_path, 
        max_frames=max_frames,
        roi_size=(512, 512)
    )
    
    # 추론 실행
    results = inferencer.predict_from_frames(
        individual_frames=individual_frames,
        csv_labels_path=csv_labels_path,
        roi_size=(512, 512),
        test_augmentation=False  # 공정한 비교를 위해 증강 비활성화
    )
    
    # 예측값과 타겟값 추출
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # 오차 계산 (상대 좌표 기준, 픽셀 단위로 변환)
    roi_size = 512
    pixel_errors = []
    for pred, target in zip(predictions, targets):
        # 상대 좌표를 픽셀 단위로 변환
        pred_px = pred * roi_size
        target_px = target * roi_size
        # 유클리드 거리 계산
        error = np.sqrt(np.sum((pred_px - target_px)**2))
        pixel_errors.append(error)
    
    pixel_errors = np.array(pixel_errors)
    mean_error = np.mean(pixel_errors)
    std_error = np.std(pixel_errors)
    
    print(f"✅ {model_name} predictions loaded: {len(predictions)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f} px")
    print(f"   Mean inference time: {results.get('mean_time', 0):.2f} ms")
    print(f"   FPS: {results.get('fps', 0):.1f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'pixel_errors': pixel_errors,
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_time': results.get('mean_time', 0),
        'fps': results.get('fps', 0),
        'method': model_name
    }


def compare_multiple_models(results: Dict[str, Dict], save_path: Optional[str] = None):
    """여러 모델의 결과를 비교 시각화"""
    
    model_names = list(results.keys())
    print(f"\n📊 Comparing {len(model_names)} models: {model_names}")
    
    if not model_names:
        print("⚠️ No models to compare.")
        return

    # 그래프 생성 (통계 표 제외, 3x3 그리드로 변경)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 색상 팔레트 (모델 개수에 맞춰 자동 할당)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = {name: colors[i] for i, name in enumerate(model_names)}

    # 공통 데이터 추출 (첫 번째 모델 기준 Targets 사용 - 모든 모델이 같은 데이터셋을 쓴다고 가정)
    first_model = model_names[0]
    targets = results[first_model]['targets']

    # ========== 1. X 좌표 분포도 비교 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    for name in model_names:
        res = results[name]
        # X 좌표 오차 (True - Predicted, px 단위)
        x_errors = (res['targets'][:, 0] - res['predictions'][:, 0]) * 512
        sns.kdeplot(data=x_errors, ax=ax1, label=name, color=color_map[name], alpha=0.7, linewidth=2)
    
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax1.set_xlabel('X Coordinate Error (True - Predicted, px)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('X Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Y 좌표 분포도 비교 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    for name in model_names:
        res = results[name]
        # Y 좌표 오차
        y_errors = (res['targets'][:, 1] - res['predictions'][:, 1]) * 512
        sns.kdeplot(data=y_errors, ax=ax2, label=name, color=color_map[name], alpha=0.7, linewidth=2)
    
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax2.set_xlabel('Y Coordinate Error (True - Predicted, px)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Y Coordinate Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== 3. 오차 분포 히스토그램 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    for name in model_names:
        res = results[name]
        ax3.hist(res['pixel_errors'], bins=30, alpha=0.4, label=name, color=color_map[name], edgecolor=color_map[name])
        ax3.axvline(np.mean(res['pixel_errors']), color=color_map[name], linestyle='--', linewidth=2,
                   label=f'{name} Mean: {np.mean(res["pixel_errors"]):.2f} px')
    
    ax3.set_xlabel('Pixel Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ========== 4. 2D 궤적 비교 (X-Y 평면) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    # True trajectory (Ground truth 궤적)
    true_px = targets * 512
    ax4.plot(true_px[:, 0], true_px[:, 1], 'o-', color='gold', linewidth=2, markersize=6, 
             label='True Trajectory', zorder=10, alpha=0.8)
    
    # 각 모델의 예측 궤적
    for name in model_names:
        res = results[name]
        pred_px = res['predictions'] * 512
        ax4.scatter(pred_px[:, 0], pred_px[:, 1], alpha=0.6, s=40, label=name, c=[color_map[name]], edgecolors='none')
    
    ax4.set_xlabel('X Coordinate (px)', fontsize=11)
    ax4.set_ylabel('Y Coordinate (px)', fontsize=11)
    ax4.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ========== 5. 오차 박스플롯 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [results[name]['pixel_errors'] for name in model_names]
    box = ax5.boxplot(box_data, labels=model_names, patch_artist=True)
    
    for patch, name in zip(box['boxes'], model_names):
        patch.set_facecolor(color_map[name])
        patch.set_alpha(0.6)
        
    ax5.set_ylabel('Pixel Error', fontsize=11)
    ax5.set_title('Error Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== 6. 누적 분포 함수 (CDF) ==========
    ax6 = fig.add_subplot(gs[1, 2])
    for name in model_names:
        res = results[name]
        sorted_errors = np.sort(res['pixel_errors'])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax6.plot(sorted_errors, cdf, label=name, color=color_map[name], linewidth=2)
        
    ax6.set_xlabel('Pixel Error', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ========== 7. 시간에 따른 오차 변화 ==========
    ax7 = fig.add_subplot(gs[2, 0])
    for name in model_names:
        res = results[name]
        ax7.plot(res['pixel_errors'], alpha=0.7, linewidth=1.5, label=name, color=color_map[name])
        ax7.axhline(np.mean(res['pixel_errors']), color=color_map[name], linestyle='--', alpha=0.5)
        
    ax7.set_xlabel('Sample Index', fontsize=11)
    ax7.set_ylabel('Pixel Error', fontsize=11)
    ax7.set_title('Error Over Time', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # ========== 8. 추론 시간 비교 ==========
    ax8 = fig.add_subplot(gs[2, 1])
    times = [results[name].get('mean_time', 0) for name in model_names]
    fps_list = [results[name].get('fps', 0) for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax8_twin = ax8.twinx()
    ax8.bar(x - width/2, times, width, label='Inference Time (ms)', color='blue', alpha=0.5)
    ax8_twin.bar(x + width/2, fps_list, width, label='FPS', color='green', alpha=0.5)
    
    ax8.set_xlabel('Model', fontsize=11)
    ax8.set_ylabel('Inference Time (ms)', fontsize=11, color='blue')
    ax8_twin.set_ylabel('FPS', fontsize=11, color='green')
    ax8.set_xticks(x)
    ax8.set_xticklabels(model_names, rotation=15)
    ax8.set_title('Performance Comparison', fontsize=12, fontweight='bold')

    # 전체 제목
    fig.suptitle(f'Model Comparison: {", ".join(model_names)}', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # 추가 통계 출력 (콘솔)
    print("\n" + "="*100)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*100)
    header = f"{'Model':<25} {'Mean Error':<15} {'Std Error':<15} {'Median':<15} {'Max Error':<15} {'Time(ms)':<15} {'FPS':<10}"
    print(header)
    print("-" * len(header))
    
    for name in model_names:
        res = results[name]
        errors = res['pixel_errors']
        print(f"{name:<25} {np.mean(errors):<15.3f} {np.std(errors):<15.3f} {np.median(errors):<15.3f} {np.max(errors):<15.3f} {res.get('mean_time', 0):<15.2f} {res.get('fps', 0):<10.1f}")
    print("="*100)


def create_statistics_table(results: Dict[str, Dict], save_path: Optional[str] = None):
    """통계 요약표를 별도 PNG 파일로 생성"""
    
    model_names = list(results.keys())
    
    if not model_names:
        return
    
    # 통계 표 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 통계 데이터 생성
    metrics = ['Mean Error (px)', 'Std Error (px)', 'Median Error (px)', 'Max Error (px)', 'Time (ms)', 'FPS']
    stats_data = [['Metric'] + model_names]
    
    for metric in metrics:
        row = [metric]
        for name in model_names:
            res = results[name]
            if metric == 'Mean Error (px)': val = np.mean(res['pixel_errors'])
            elif metric == 'Std Error (px)': val = np.std(res['pixel_errors'])
            elif metric == 'Median Error (px)': val = np.median(res['pixel_errors'])
            elif metric == 'Max Error (px)': val = np.max(res['pixel_errors'])
            elif metric == 'Time (ms)': val = res.get('mean_time', 0)
            elif metric == 'FPS': val = res.get('fps', 0)
            row.append(f"{val:.2f}")
        stats_data.append(row)
        
    table = ax.table(cellText=stats_data, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # 헤더 스타일링
    for i in range(len(model_names) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행별 색상 (교대로)
    for i in range(1, len(stats_data)):
        for j in range(len(model_names) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Statistical Summary', fontsize=16, fontweight='bold', pad=20)
    fig.suptitle(f'Model Comparison Statistics: {", ".join(model_names)}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Statistics table saved to: {save_path}")
    
    plt.close()


def main():
    """메인 함수"""
    print("🎯 Model Comparison Tool")
    print("="*80)
    
    # ============================================================
    # [사용자 설정 영역] 비교할 모델 정의
    # ============================================================
    models_to_compare = [
        {
            'name': 'MobileNetV2 (Standard)',
            'path': '/hai/home/jdj/dvs/cnn_brownian_sim/checkpoints_mobilenet_v2/mobilenet_v2_best.pth',
            'quantized': False
        },
        # {
        #     'name': 'MobileNetV2 (QAT)',
        #     'path': '/hai/home/jdj/dvs/cnn_brownian_sim/checkpoints_mobilenet_v2_qat/mobilenet_v2_qat_int8.pth',
        #     'quantized': True
        # },
        # 예시: 추가 모델 비교 (필요 시 주석 해제)
        {
            'name': 'MobileOne-S0 (Standard)',
            'path': '/hai/home/jdj/dvs/cnn_brownian_sim/checkpoints_mobileone_s0/mobileone_s0_best.pth',
            'quantized': False
        },
        {
            'name': 'MobileOne-S0 (QAT)',
            'path': '/hai/home/jdj/dvs/cnn_brownian_sim/checkpoints_mobileone_s0_qat/mobileone_s0_int8.pth',
            'quantized': True
        }
    ]
    # ============================================================
    
    bin_file = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512.bin"
    csv_labels = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512_labels.csv"
    
    # 결과 저장 디렉토리 설정
    output_dir = "compare_qnt"
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "compare_graph.png")
    table_path = os.path.join(output_dir, "compare_table.png")
    
    # 필수 데이터 파일 존재 확인
    if not os.path.exists(bin_file):
        print(f"❌ Bin file not found: {bin_file}")
        return
    if not os.path.exists(csv_labels):
        print(f"❌ CSV labels file not found: {csv_labels}")
        return

    # 결과 수집
    results = {}
    
    for model_cfg in models_to_compare:
        if not os.path.exists(model_cfg['path']):
            print(f"⚠️ Checkpoint not found: {model_cfg['path']} (Skipping)")
            continue
            
        try:
            res = load_model_predictions(
                checkpoint_path=model_cfg['path'],
                bin_file_path=bin_file,
                csv_labels_path=csv_labels,
                max_frames=100,
                use_quantized=model_cfg['quantized'],
                model_name=model_cfg['name']
            )
            results[model_cfg['name']] = res
        except Exception as e:
            print(f"⚠️ Failed to process {model_cfg['name']}: {e}")
            continue
    
    if not results:
        print("❌ No results to compare.")
        return
        
    # 샘플 수 맞추기 (모든 모델 중 최소 샘플 수로 통일)
    min_samples = min(len(res['predictions']) for res in results.values())
    safe_samples = min_samples - 3  # 안전 마진
    
    print(f"\n📊 Using {safe_samples} samples for fair comparison (excluded last 3)")
    
    for name in results:
        for key in ['predictions', 'targets', 'pixel_errors']:
            results[name][key] = results[name][key][:safe_samples]
            
    # 비교 그래프 생성
    compare_multiple_models(results, save_path=graph_path)
    
    # 통계 표 생성
    create_statistics_table(results, save_path=table_path)
    
    print(f"\n✅ Comparison completed!")
    print(f"   Graph saved to: {graph_path}")
    print(f"   Table saved to: {table_path}")


if __name__ == "__main__":
    main()

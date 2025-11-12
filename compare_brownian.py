#!/usr/bin/env python3
"""
CNN vs YOLO vs Filter 비교 스크립트 (Brownian Motion 데이터셋)

Brownian motion 데이터셋에 대해:
- cnn_brownian_sim의 MobileNet 모델 추론 결과
- yolo_brownian_sim의 YOLOv3-Tiny 모델 추론 결과
- filter_brownian_sim의 중심점 추정 결과
를 비교 분석합니다.

정답 데이터는 CSV 파일(gaussian_brownian_512x512_labels.csv)에서 로드합니다.
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
sys.path.append('/hai/home/jdj/dvs')
sys.path.append('/hai/home/jdj/dvs/cnn_brownian_sim')
sys.path.append('/hai/home/jdj/dvs/filter_brownian_sim')
sys.path.append('/hai/home/jdj/dvs/yolo_brownian_sim')

from cnn_brownian_sim.inference import DVSInference
from lib.bin_processor import BinProcessor
from yolo_brownian_sim.inference import LaserYOLOInference
from yolo_brownian_sim.dataset import load_frames_from_bin as yolo_load_frames


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """정답 CSV 파일 로드"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} ground truth records")
    return df


def load_cnn_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    csv_labels_path: str,
    max_frames: int = 50
) -> Dict:
    """CNN 모델의 추론 결과 로드 (정규화 좌표를 ROI 내부 절대 좌표로 변환)"""
    print("🤖 Loading CNN (MobileNet) predictions...")
    
    # Inference 객체 생성
    inferencer = DVSInference(checkpoint_path)
    
    # 프레임 로드 (512x512 ROI) - BinProcessor는 512x512로 설정
    processor = BinProcessor(frame_width=512, frame_height=512, has_header=True)
    frames = processor.read_frames(bin_file_path, max_frames=max_frames)
    
    # numpy 배열로 변환 및 정규화
    individual_frames = []
    for frame in frames:
        frame_array = np.array(frame.raw_data, dtype=np.float32)
        if np.max(frame_array) > 0:
            frame_array = frame_array / np.max(frame_array)  # 0-1 정규화
        individual_frames.append(frame_array)
    
    print(f"   Loaded {len(individual_frames)} frames")
    
    # 정답 데이터 로드
    gt_df = load_ground_truth(csv_labels_path)
    
    # 데이터셋 생성 (Brownian motion용)
    from cnn_brownian_sim.dataset import DVSBrownianDataset
    dataset = DVSBrownianDataset(
        individual_frames=individual_frames,
        csv_labels_path=csv_labels_path,
        roi_size=(512, 512),
        temporal_window=5
    )
    dataset.set_training_mode(False)
    
    # 추론 실행
    predictions_rel = []
    targets_rel = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample_input, sample_label = dataset[i]
            input_tensor = sample_input.unsqueeze(0).to(inferencer.device)
            
            output = inferencer.model(input_tensor)
            pred_rel = output[0].cpu().numpy()
            target_rel = sample_label.numpy()
            
            predictions_rel.append(pred_rel)
            targets_rel.append(target_rel)
    
    # 정규화 좌표(0-1)를 ROI 내부 절대 좌표로 변환
    roi_size = 512
    predictions_abs = np.array(predictions_rel) * roi_size
    targets_abs = np.array(targets_rel) * roi_size
    
    # 오차 계산
    errors = np.sqrt(np.sum((predictions_abs - targets_abs)**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"✅ CNN predictions loaded: {len(predictions_abs)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f} pixels")
    
    return {
        'predictions': predictions_abs,
        'targets': targets_abs,
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'CNN (MobileNet)'
    }


def load_yolo_predictions(
    checkpoint_path: str,
    bin_file_path: str,
    csv_labels_path: str,
    max_frames: int = 50,
    conf_threshold: float = 0.5
) -> Dict:
    """YOLO 모델의 추론 결과 로드"""
    print("🎯 Loading YOLO predictions...")
    
    # YOLO Inference 객체 생성
    inferencer = LaserYOLOInference(checkpoint_path)
    
    # 프레임 로드 (512x512 ROI)
    individual_frames = yolo_load_frames(bin_file_path, max_frames=max_frames)
    
    # 데이터셋 생성 (Brownian motion용)
    from yolo_brownian_sim.dataset import LaserYOLOBrownianDataset
    from yolo_brownian_sim.model import decode_predictions, get_laser_center
    
    dataset = LaserYOLOBrownianDataset(
        individual_frames=individual_frames,
        csv_labels_path=csv_labels_path,
        laser_diameter=400,
        roi_size=(512, 512),
        temporal_window=5
    )
    dataset.set_training_mode(False)
    
    # 추론 실행
    predictions_rel = []
    targets_rel = []
    detection_status = []  # YOLO 감지 성공/실패 기록
    frame_info = []  # 각 샘플의 프레임 인덱스 정보
    
    print(f"   Dataset length: {len(dataset)} samples")
    print(f"   Temporal window: 5")
    print(f"   Last sample idx: {len(dataset)-1}")
    print(f"   Last sample uses frames: {len(dataset)-1} to {len(dataset)-1 + 4}")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target = dataset[idx]
            image = image.unsqueeze(0).to(inferencer.device)
            
            # 정답 저장 (bbox 중심점)
            targets_rel.append((target[0].item(), target[1].item()))
            
            # Temporal window 프레임 정보 확인
            # 실제 dataset의 __getitem__에서 사용하는 frame_indices 계산
            frame_indices = list(range(idx, idx + 5))  # temporal_window=5
            center_frame_idx = frame_indices[2]  # 중앙 프레임 (idx + temporal_window//2)
            frame_info.append({
                'sample_idx': idx,
                'center_frame_idx': center_frame_idx,
                'frame_range': (frame_indices[0], frame_indices[-1]),
                'all_frames_valid': all(f < len(individual_frames) for f in frame_indices)
            })
            
            # 예측
            output = inferencer.model(image)
            boxes_list, scores_list = decode_predictions(
                output, inferencer.anchors, conf_threshold=conf_threshold
            )
            
            # 중심점 추출
            detection_success = False
            detected_boxes_info = None
            if len(boxes_list[0]) > 0:
                center = get_laser_center(boxes_list[0], scores_list[0])
                if center:
                    predictions_rel.append((center[0], center[1]))
                    detection_success = True
                    # 디버그: 감지된 모든 박스 정보 저장 (특히 70-85 범위)
                    if 70 <= idx <= 85:
                        detected_boxes_info = {
                            'num_boxes': len(boxes_list[0]),
                            'all_boxes': boxes_list[0].cpu().numpy(),
                            'all_scores': scores_list[0].cpu().numpy(),
                            'selected_center': center,
                            'selected_idx': torch.argmax(scores_list[0]).item()
                        }
                else:
                    # 실패 시 이전 프레임 값 사용
                    if len(predictions_rel) > 0:
                        predictions_rel.append(predictions_rel[-1])
                    else:
                        predictions_rel.append((0.5, 0.5))  # 첫 프레임 실패 시 ROI 중심
                    detection_success = False
            else:
                # 실패 시 이전 프레임 값 사용
                if len(predictions_rel) > 0:
                    predictions_rel.append(predictions_rel[-1])
                else:
                    predictions_rel.append((0.5, 0.5))  # 첫 프레임 실패 시 ROI 중심
                detection_success = False
            
            detection_status.append({
                'sample_idx': idx,
                'detection_success': detection_success,
                'used_previous': not detection_success and len(predictions_rel) > 1,
                'boxes_info': detected_boxes_info
            })
    
    # 정규화 좌표(0-1)를 ROI 내부 절대 좌표로 변환
    roi_size = 512
    predictions_abs = np.array(predictions_rel) * roi_size
    targets_abs = np.array(targets_rel) * roi_size
    
    # 오차 계산
    errors = np.sqrt(np.sum((predictions_abs - targets_abs)**2, axis=1))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 마지막 10개 샘플 상세 분석
    print(f"\n🔍 Analyzing last 10 samples:")
    last_n = min(10, len(errors))
    for i in range(len(errors) - last_n, len(errors)):
        status = detection_status[i]
        info = frame_info[i]
        error = errors[i]
        pred = predictions_abs[i]
        target = targets_abs[i]
        print(f"   Sample {i:3d}: Error={error:6.2f}px, "
              f"CenterFrame={info['center_frame_idx']:3d}, "
              f"Detected={status['detection_success']}, "
              f"Pred=({pred[0]:6.1f},{pred[1]:6.1f}), "
              f"GT=({target[0]:6.1f},{target[1]:6.1f})")
    
    # 오류 급증 지점 찾기
    error_threshold = np.mean(errors) + 3 * np.std(errors)  # 평균 + 3*표준편차
    spike_indices = np.where(errors > error_threshold)[0]
    if len(spike_indices) > 0:
        print(f"\n⚠️  Error spikes detected (>{error_threshold:.1f}px):")
        for spike_idx in spike_indices:
            status = detection_status[spike_idx]
            info = frame_info[spike_idx]
            error = errors[spike_idx]
            pred = predictions_abs[spike_idx]
            target = targets_abs[spike_idx]
            print(f"   Sample {spike_idx:3d}: Error={error:6.2f}px, "
                  f"CenterFrame={info['center_frame_idx']:3d}, "
                  f"Detected={status['detection_success']}, "
                  f"UsedPrevious={status.get('used_previous', False)}")
            print(f"      Pred=({pred[0]:6.1f},{pred[1]:6.1f}), "
                  f"GT=({target[0]:6.1f},{target[1]:6.1f}), "
                  f"Diff=({target[0]-pred[0]:6.1f},{target[1]-pred[1]:6.1f})")
    
    # 70-85 범위 샘플들 상세 분석 (오류 급증 구간)
    print(f"\n🔍 Detailed analysis of samples 70-85 (error spike region):")
    for i in range(70, min(86, len(errors))):
        status = detection_status[i]
        info = frame_info[i]
        error = errors[i]
        pred = predictions_abs[i]
        target = targets_abs[i]
        print(f"   Sample {i:3d}: Error={error:6.2f}px, "
              f"CenterFrame={info['center_frame_idx']:3d}, "
              f"Frames={info['frame_range'][0]}-{info['frame_range'][1]}, "
              f"AllValid={info['all_frames_valid']}, "
              f"Detected={status['detection_success']}, "
              f"Pred=({pred[0]:6.1f},{pred[1]:6.1f}), "
              f"GT=({target[0]:6.1f},{target[1]:6.1f})")
        
        # 오류가 큰 샘플의 박스 정보 출력
        if status.get('boxes_info') is not None and error > 50:
            boxes_info = status['boxes_info']
            print(f"      🔍 Detected {boxes_info['num_boxes']} boxes:")
            for box_idx, (box, score) in enumerate(zip(boxes_info['all_boxes'], boxes_info['all_scores'])):
                box_abs = box * 512  # 절대 좌표로 변환
                marker = "★" if box_idx == boxes_info['selected_idx'] else " "
                print(f"         {marker} Box {box_idx}: center=({box_abs[0]:6.1f},{box_abs[1]:6.1f}), "
                      f"size=({box_abs[2]:6.1f},{box_abs[3]:6.1f}), conf={score:.3f}")
    
    print(f"\n✅ YOLO predictions loaded: {len(predictions_abs)} samples")
    print(f"   Mean error: {mean_error:.2f}±{std_error:.2f} pixels")
    print(f"   Detection success rate: {sum(s['detection_success'] for s in detection_status) / len(detection_status) * 100:.1f}%")
    
    return {
        'predictions': predictions_abs,
        'targets': targets_abs,
        'mean_error': mean_error,
        'std_error': std_error,
        'method': 'YOLO (Tiny)'
    }


def load_filter_predictions(
    csv_file_path: str,
    ground_truth_csv: str
) -> Dict:
    """Filter 결과를 CSV에서 로드하고 정답과 비교"""
    print("🔍 Loading Filter predictions...")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Filter CSV file not found: {csv_file_path}")
    
    # Filter 예측 CSV 로드
    pred_df = pd.read_csv(csv_file_path)
    pred_df = pred_df.dropna()  # 유효한 데이터만
    
    # 정답 CSV 로드
    gt_df = load_ground_truth(ground_truth_csv)
    
    # 데이터 병합 (frame_number와 frame_idx 기준)
    merged = pd.merge(
        pred_df,
        gt_df,
        left_on='frame_number',
        right_on='frame_idx',
        how='inner'
    )
    
    # 유효한 데이터만
    valid = merged.dropna(subset=['center_x', 'center_y', 'roi_center_x', 'roi_center_y'])
    
    if len(valid) == 0:
        raise ValueError("No valid merged data found. Check frame_number/frame_idx matching.")
    
    # predictions와 targets 추출
    predictions = valid[['center_x', 'center_y']].values
    targets = valid[['roi_center_x', 'roi_center_y']].values
    
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
    ax4.plot(cnn_target[:, 0], cnn_target[:, 1], 'b-', alpha=0.6, linewidth=2, label='GT Trajectory', marker='o', markersize=3)
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
    """메인 함수"""
    print("🎯 CNN vs YOLO vs Filter Comparison (Brownian Motion)")
    print("="*80)
    
    # 경로 설정
    cnn_checkpoint = "/hai/home/jdj/dvs/cnn_brownian_sim/checkpoints_mobilenet_v2/mobilenet_best.pth"
    yolo_checkpoint = "/hai/home/jdj/dvs/yolo_brownian_sim/checkpoints_yolo_tiny_laser_brownian/yolo_tiny_laser_brownian_best.pth"
    filter_csv = "/hai/home/jdj/dvs/filter_brownian_sim/csv_results/spatial_filter_kalman.csv"
    ground_truth_csv = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512_labels.csv"
    bin_file = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512.bin"
    output_path = "/hai/home/jdj/dvs/cnn_yolo_filter_comparison_brownian.png"
    
    # 파일 존재 확인
    if not os.path.exists(cnn_checkpoint):
        print(f"❌ CNN checkpoint not found: {cnn_checkpoint}")
        print("   Please train CNN model first")
        return
    
    if not os.path.exists(yolo_checkpoint):
        print(f"❌ YOLO checkpoint not found: {yolo_checkpoint}")
        print("   Please train YOLO model first")
        return
    
    if not os.path.exists(filter_csv):
        print(f"❌ Filter CSV not found: {filter_csv}")
        print("   Please run: cd filter_brownian_sim && python export_center_data.py")
        return
    
    if not os.path.exists(ground_truth_csv):
        print(f"❌ Ground truth CSV not found: {ground_truth_csv}")
        print("   Please generate Brownian motion dataset first")
        return
    
    if not os.path.exists(bin_file):
        print(f"❌ Bin file not found: {bin_file}")
        print("   Please generate Brownian motion dataset first")
        return
    
    # CNN 결과 로드
    cnn_results = load_cnn_predictions(
        checkpoint_path=cnn_checkpoint,
        bin_file_path=bin_file,
        csv_labels_path=ground_truth_csv,
        max_frames=100
    )
    
    # YOLO 결과 로드
    yolo_results = load_yolo_predictions(
        checkpoint_path=yolo_checkpoint,
        bin_file_path=bin_file,
        csv_labels_path=ground_truth_csv,
        max_frames=100,
        conf_threshold=0.6
    )
    
    # Filter 결과 로드
    filter_results = load_filter_predictions(
        csv_file_path=filter_csv,
        ground_truth_csv=ground_truth_csv
    )
    
    # 샘플 수 맞추기
    min_samples = min(len(cnn_results['predictions']), 
                     len(yolo_results['predictions']), 
                     len(filter_results['predictions']))
    
    cnn_results['predictions'] = cnn_results['predictions'][:min_samples]
    cnn_results['targets'] = cnn_results['targets'][:min_samples]
    yolo_results['predictions'] = yolo_results['predictions'][:min_samples]
    yolo_results['targets'] = yolo_results['targets'][:min_samples]
    filter_results['predictions'] = filter_results['predictions'][:min_samples]
    filter_results['targets'] = filter_results['targets'][:min_samples]
    
    print(f"\n📊 Using {min_samples} samples for comparison")
    
    # 비교 시각화
    compare_methods(cnn_results, yolo_results, filter_results, save_path=output_path)
    
    print(f"\n✅ Comparison completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()


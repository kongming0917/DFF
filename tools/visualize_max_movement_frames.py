#!/usr/bin/env python3
"""
Brownian Motion 데이터셋에서 가장 많이 움직인 프레임들을 시각화

각 프레임 간의 이동 거리를 계산하고, 가장 큰 이동이 발생한 순간들을 시각화합니다.
"""

import numpy as np
import os
import json
import matplotlib
# GUI 환경 확인 후 백엔드 설정
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# 공통 라이브러리 사용
import sys
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sim/ (tools/의 상위)
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from dvslib.data import BinProcessor


def load_shift_info(bin_file_path: str) -> Optional[dict]:
    """shift 정보 JSON 파일 로드"""
    shift_info_path = bin_file_path.replace('.bin', '_shifts.json')
    if os.path.exists(shift_info_path):
        with open(shift_info_path, 'r') as f:
            return json.load(f)
    return None


def calculate_movement_distances(bin_file_path: str, max_frames: int = None) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    각 프레임 간의 이동 거리 계산
    
    Returns:
        distances: 각 프레임에서의 이동 거리 배열 (프레임 0은 0)
        frames: 프레임 데이터 리스트
        centers: 각 프레임의 레이저 중심점 좌표 배열
    """
    print(f"📖 데이터 읽는 중: {bin_file_path}")
    
    processor = BinProcessor(
        frame_width=512,
        frame_height=512,
        has_header=True
    )
    
    frames = processor.read_frames(bin_file_path, max_frames=max_frames)
    print(f"   ✅ {len(frames)}개 프레임 읽음")
    
    # shift 정보 로드 시도
    shift_info = load_shift_info(bin_file_path)
    
    if shift_info is not None:
        print(f"   ✅ Shift 정보 로드: {len(shift_info['shifts'])}개 프레임")
        # shift 정보에서 중심점 계산 (shift는 누적값이므로 그대로 사용)
        roi_center = 256
        centers = []
        for shift_data in shift_info['shifts']:
            if shift_data['frame_idx'] < len(frames):
                # shift가 누적값이므로 ROI 중앙에서 shift만큼 이동
                laser_x = roi_center + shift_data['shift_x']
                laser_y = roi_center + shift_data['shift_y']
                centers.append((laser_x, laser_y))
        
        # 부족한 프레임은 마지막 값으로 채움
        while len(centers) < len(frames):
            if len(centers) > 0:
                centers.append(centers[-1])
            else:
                centers.append((roi_center, roi_center))
        
        centers = np.array(centers[:len(frames)])
    else:
        print(f"   ⚠️  Shift 정보 없음. 이벤트에서 중심점 계산...")
        # shift 정보가 없으면 이벤트에서 계산 (fallback)
        centers = []
        for frame_idx, frame in enumerate(frames):
            on_events_y, on_events_x = np.where(frame.raw_data == 1)
            
            if len(on_events_x) > 0:
                if frame_idx > 0 and len(centers) > 0:
                    prev_center = centers[-1]
                    search_center_x, search_center_y = prev_center[0], prev_center[1]
                else:
                    search_center_x, search_center_y = 256, 256
                
                if len(on_events_x) > 20:
                    distances = np.sqrt((on_events_x - search_center_x)**2 + (on_events_y - search_center_y)**2)
                    threshold = np.percentile(distances, 40)
                    mask = distances <= threshold
                    
                    if np.sum(mask) >= 3:
                        center_x = np.mean(on_events_x[mask])
                        center_y = np.mean(on_events_y[mask])
                    else:
                        center_x = np.mean(on_events_x)
                        center_y = np.mean(on_events_y)
                else:
                    center_x = np.mean(on_events_x)
                    center_y = np.mean(on_events_y)
                
                centers.append((center_x, center_y))
            else:
                if len(centers) > 0:
                    centers.append(centers[-1])
                else:
                    centers.append((256, 256))
        
        centers = np.array(centers)
    
    # 프레임 간 이동 거리 계산
    distances = np.zeros(len(frames))
    for i in range(1, len(frames)):
        dx = centers[i, 0] - centers[i-1, 0]
        dy = centers[i, 1] - centers[i-1, 1]
        distances[i] = np.sqrt(dx**2 + dy**2)
    
    return distances, frames, centers


def visualize_max_movement_frames(
    bin_file_path: str,
    num_top_frames: int = 10,
    max_frames: int = None,
    output_path: str = None,
    show_sequence: bool = True
):
    """
    가장 많이 움직인 프레임들을 시각화
    
    Args:
        bin_file_path: 입력 bin 파일 경로
        num_top_frames: 시각화할 상위 프레임 개수
        max_frames: 최대 처리 프레임 수
        output_path: 출력 이미지 경로
        show_sequence: 연속 프레임 시퀀스도 표시할지 여부
    """
    print("=" * 80)
    print("가장 많이 움직인 프레임 시각화")
    print("=" * 80)
    
    # 이동 거리 및 중심점 계산
    distances, frames, centers = calculate_movement_distances(bin_file_path, max_frames)
    
    # 상위 N개 프레임 인덱스 찾기
    top_indices = np.argsort(distances)[-num_top_frames:][::-1]
    top_distances = distances[top_indices]
    
    print(f"\n📊 상위 {num_top_frames}개 이동 거리:")
    for i, (idx, dist) in enumerate(zip(top_indices, top_distances)):
        print(f"   {i+1}. Frame {idx}: {dist:.2f} pixels")
    
    # 1. 상위 프레임들 개별 시각화 (이전/현재 비교)
    num_cols = 5
    num_rows = (num_top_frames + num_cols - 1) // num_cols
    
    # 더 큰 크기로 조정 (각 subplot이 충분히 크도록)
    fig = plt.figure(figsize=(24, 6 * num_rows))
    fig.suptitle(f'Top {num_top_frames} Maximum Movement Frames', fontsize=16, fontweight='bold', y=0.995)
    
    for plot_idx, (frame_idx, distance) in enumerate(zip(top_indices, top_distances)):
        ax = fig.add_subplot(num_rows, num_cols, plot_idx + 1)
        
        # 프레임 데이터 시각화
        frame_data = frames[frame_idx].raw_data
        
        # ON 이벤트와 OFF 이벤트를 다르게 표시
        im = ax.imshow(frame_data, cmap='gray', vmin=0, vmax=2, aspect='equal', interpolation='nearest')
        
        # 현재 중심점
        center_x = centers[frame_idx, 0]
        center_y = centers[frame_idx, 1]
        ax.scatter(center_x, center_y, c='red', s=300, marker='x', 
                  linewidths=5, label='Current', zorder=10)
        
        # 이전 프레임 중심점
        if frame_idx > 0:
            prev_center_x = centers[frame_idx-1, 0]
            prev_center_y = centers[frame_idx-1, 1]
            ax.scatter(prev_center_x, prev_center_y, c='blue', s=300, 
                      marker='+', linewidths=5, label='Previous', zorder=9)
            
            # 화살표로 이동 방향 표시
            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            if abs(dx) > 1 or abs(dy) > 1:  # 이동이 있을 때만 화살표 표시
                ax.annotate('', xy=(center_x, center_y), xytext=(prev_center_x, prev_center_y),
                          arrowprops=dict(arrowstyle='->', color='lime', lw=4, alpha=0.9, shrinkA=5, shrinkB=5))
            
            # 이동 거리 텍스트
            mid_x = (prev_center_x + center_x) / 2
            mid_y = (prev_center_y + center_y) / 2
            ax.text(mid_x + 15, mid_y + 15, f'{distance:.1f}px', 
                   color='lime', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # ROI 중앙 표시
        roi_center = 256
        ax.scatter(roi_center, roi_center, c='yellow', s=200, marker='o', 
                  linewidths=3, label='ROI Center', zorder=8, edgecolors='black', alpha=0.6)
        
        # ROI 중앙으로부터의 거리
        dist_from_center = np.sqrt((center_x - roi_center)**2 + (center_y - roi_center)**2)
        ax.text(0.02, 0.98, f'From center: {dist_from_center:.1f}px', 
               color='white', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f'Frame {frame_idx}\nMove: {distance:.2f}px', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax.set_xlim(-10, 522)
        ax.set_ylim(522, -10)  # Y축 반전 (이미지 좌표계)
        ax.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
    
    # 저장
    if output_path is None:
        output_path = bin_file_path.replace('.bin', '_max_movement_frames.png')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 시각화 저장: {output_path}")
    plt.close()
    
    # 2. 연속 프레임 시퀀스 시각화 (선택적)
    if show_sequence and len(top_indices) > 0:
        print("\n📽️  연속 프레임 시퀀스 생성 중...")
        seq_fig, seq_axes = plt.subplots(2, 3, figsize=(20, 14))
        seq_fig.suptitle(f'Frame Sequence Around Maximum Movement (Frame {top_indices[0]})', 
                        fontsize=16, fontweight='bold', y=0.995)
        seq_axes = seq_axes.flatten()
        
        # 첫 번째 상위 프레임 주변의 시퀀스
        first_top_idx = top_indices[0]
        start_idx = max(0, first_top_idx - 2)
        end_idx = min(len(frames), first_top_idx + 3)
        
        for plot_idx, frame_idx in enumerate(range(start_idx, end_idx)):
            if plot_idx >= 6:
                break
            ax = seq_axes[plot_idx]
            
            frame_data = frames[frame_idx].raw_data
            ax.imshow(frame_data, cmap='gray', vmin=0, vmax=2, aspect='equal', interpolation='nearest')
            
            # 중심점
            center_x = centers[frame_idx, 0]
            center_y = centers[frame_idx, 1]
            ax.scatter(center_x, center_y, c='red', s=400, marker='x', 
                      linewidths=5, zorder=10)
            
            # 이전 프레임 중심점
            if frame_idx > start_idx:
                prev_center_x = centers[frame_idx-1, 0]
                prev_center_y = centers[frame_idx-1, 1]
                ax.scatter(prev_center_x, prev_center_y, c='blue', s=400, 
                          marker='+', linewidths=5, zorder=9)
                if abs(center_x - prev_center_x) > 1 or abs(center_y - prev_center_y) > 1:
                    ax.annotate('', xy=(center_x, center_y), xytext=(prev_center_x, prev_center_y),
                              arrowprops=dict(arrowstyle='->', color='lime', lw=4, alpha=0.9, shrinkA=5, shrinkB=5))
            
            # ROI 중앙
            ax.scatter(256, 256, c='yellow', s=250, marker='o', 
                     linewidths=3, zorder=8, edgecolors='black', alpha=0.6)
            
            move_dist = distances[frame_idx] if frame_idx > 0 else 0
            ax.set_title(f'Frame {frame_idx}\nMove: {move_dist:.2f}px', 
                        fontsize=13, fontweight='bold', pad=10)
            ax.set_xlim(-10, 522)
            ax.set_ylim(522, -10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
        seq_path = output_path.replace('.png', '_sequence.png')
        plt.savefig(seq_path, dpi=150, bbox_inches='tight')
        print(f"✅ 시퀀스 시각화 저장: {seq_path}")
        plt.close()
    
    # 추가: 이동 거리 분포 히스토그램
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 전체 이동 거리 분포
    ax = axes[0]
    ax.hist(distances[1:], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(distances[1:]), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(distances[1:]):.2f}px')
    ax.axvline(np.median(distances[1:]), color='blue', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(distances[1:]):.2f}px')
    ax.set_xlabel('Movement Distance (pixels)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Frame-to-Frame Movement', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 시간에 따른 이동 거리
    ax = axes[1]
    ax.plot(distances[1:], 'b-', alpha=0.6, linewidth=0.5)
    # 상위 N개 프레임 강조
    ax.scatter(top_indices, top_distances, c='red', s=50, 
              zorder=5, label=f'Top {num_top_frames} movements')
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Movement Distance (pixels)', fontsize=11)
    ax.set_title('Movement Distance Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    hist_path = output_path.replace('.png', '_distribution.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✅ 분포 시각화 저장: {hist_path}")
    plt.close()
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("📊 통계:")
    print(f"   - 총 프레임 수: {len(frames)}")
    print(f"   - 평균 이동 거리: {np.mean(distances[1:]):.2f} pixels")
    print(f"   - 중앙값 이동 거리: {np.median(distances[1:]):.2f} pixels")
    print(f"   - 최대 이동 거리: {np.max(distances):.2f} pixels")
    print(f"   - 최소 이동 거리: {np.min(distances[1:]):.2f} pixels")
    print(f"   - 표준편차: {np.std(distances[1:]):.2f} pixels")
    print("=" * 80)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Brownian 데이터셋에서 이동이 큰 프레임 시각화")
    ap.add_argument("bin_file", nargs="?",
                    default=os.path.join(dvs_root, "data", "gaussian_brownian_512x512.bin"))
    ap.add_argument("--num-top-frames", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None, help="None이면 전체")
    args = ap.parse_args()
    visualize_max_movement_frames(
        bin_file_path=args.bin_file,
        num_top_frames=args.num_top_frames,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

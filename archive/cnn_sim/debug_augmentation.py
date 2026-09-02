#!/usr/bin/env python3
"""
데이터 증강 파이프라인 디버깅 및 시각화 스크립트
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 상위 디렉토리를 sys.path에 추가 (lib 모듈 사용을 위해)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 상위 디렉토리 모듈 import
from dataset import DVSFixedGTDataset
from lib.bin_processor import BinProcessor

def visualize_augmentation_samples(dataset: DVSFixedGTDataset, num_samples: int = 6, save_path: str = "augmentation/augmentation_debug.png", show_original: bool = False):
    """데이터 증강 파이프라인 검증을 위한 시각화"""
    print(f"🔍 데이터 증강 시각화 중... ({num_samples} 샘플)")
    
    # 임시로 training 모드로 설정 (증강 적용)
    original_training = dataset.training
    dataset.set_training_mode(True)
    
    # 서브플롯 설정
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        if show_original:
            # 원본 프레임 가져오기 (ROI로 잘리지 않은 전체 프레임)
            original_frame = dataset.frames[i]
            image_to_show = original_frame
            
            # GT 좌표 (원본 프레임에서의 실제 레이저 중심)
            gt_x = dataset.true_center_x
            gt_y = dataset.true_center_y
        else:
            # 증강된 샘플 가져오기 (ROI로 잘린 데이터 + shift 적용)
            sample_tensor, label_tensor = dataset[i]
            
            # 첫 번째 채널 사용 (temporal_window가 1보다 큰 경우)
            if sample_tensor.dim() == 3:  # (C, H, W)
                roi_image = sample_tensor[0].numpy()
            else:  # (H, W)
                roi_image = sample_tensor.numpy()
            
            image_to_show = roi_image
            
            # GT 좌표 (normalized 좌표를 픽셀 좌표로 변환)
            gt_x = label_tensor[0].item() * dataset.roi_width
            gt_y = label_tensor[1].item() * dataset.roi_height
        
        # 시각화
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 이미지 표시
        im = ax.imshow(image_to_show, cmap='hot', interpolation='nearest')
        
        # GT 점 표시 (빨간 십자)
        ax.plot(gt_x, gt_y, 'r+', markersize=15, markeredgewidth=3, label='Ground Truth')
        
        if show_original:
            # ROI 영역 표시 (파란 사각형)
            roi_half_w = dataset.roi_width // 2
            roi_half_h = dataset.roi_height // 2
            roi_rect = plt.Rectangle((gt_x - roi_half_w, gt_y - roi_half_h), 
                                   dataset.roi_width, dataset.roi_height,
                                   linewidth=2, edgecolor='blue', facecolor='none', label='ROI Area')
            ax.add_patch(roi_rect)
            
            # 제목 설정
            ax.set_title(f'Sample {i}\nGT: ({gt_x}, {gt_y})\nROI: {dataset.roi_width}x{dataset.roi_height}')
        else:
            # ROI 중심점 표시 (파란 점)
            center_x, center_y = dataset.roi_width / 2, dataset.roi_height / 2
            ax.plot(center_x, center_y, 'bo', markersize=8, alpha=0.7, label='ROI Center')
            
            # 제목 설정
            ax.set_title(f'Sample {i}\nGT: ({gt_x:.1f}, {gt_y:.1f})\nNormalized: ({label_tensor[0]:.3f}, {label_tensor[1]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 컬러바 추가
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 빈 서브플롯 제거
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 증강 시각화 저장: {save_path}")
    
    # training 모드 복원
    dataset.set_training_mode(original_training)
    
    # 환경에 따라 표시
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        plt.show()
    else:
        plt.close()


def debug_augmentation_pipeline(dataset: DVSFixedGTDataset, sample_idx: int = 0, num_variations: int = 8):
    """특정 샘플에 대해 다양한 증강 결과 확인"""
    print(f"🔬 샘플 {sample_idx}에 대한 증강 파이프라인 디버깅...")
    
    if sample_idx >= len(dataset):
        print(f"❌ 샘플 인덱스 {sample_idx}가 범위를 벗어남 (최대: {len(dataset)-1})")
        return
    
    # 원본 프레임 가져오기
    frame_indices = list(range(sample_idx, sample_idx + dataset.temporal_window))
    original_frame = dataset.frames[frame_indices[0]]  # 첫 번째 프레임 사용
    
    # 임시로 training 모드로 설정
    original_training = dataset.training
    dataset.set_training_mode(True)
    
    # 서브플롯 설정
    cols = 4
    rows = (num_variations + cols - 1) // cols + 1  # +1 for original
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 원본 프레임 표시
    ax = axes[0, 0]
    ax.imshow(original_frame, cmap='hot', interpolation='nearest')
    ax.plot(dataset.true_center_x, dataset.true_center_y, 'r+', markersize=15, markeredgewidth=3)
    ax.set_title(f'Original Frame\nTrue Center: ({dataset.true_center_x}, {dataset.true_center_y})')
    ax.grid(True, alpha=0.3)
    
    # 증강 변형들 생성
    for i in range(1, num_variations + 1):
        # 랜덤 시프트 생성
        shift_x = np.random.randint(-dataset.shift_range_x, dataset.shift_range_x + 1)
        shift_y = np.random.randint(-dataset.shift_range_y, dataset.shift_range_y + 1)
        
        # ROI 추출
        roi = dataset._extract_roi_with_shift(original_frame, shift_x, shift_y)
        
        # 상대적 레이블 계산
        rel_x, rel_y = dataset._calculate_relative_label(shift_x, shift_y)
        gt_x = rel_x * dataset.roi_width
        gt_y = rel_y * dataset.roi_height
        
        # 시각화
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        im = ax.imshow(roi, cmap='hot', interpolation='nearest')
        ax.plot(gt_x, gt_y, 'r+', markersize=12, markeredgewidth=2, label='GT')
        ax.plot(dataset.roi_width/2, dataset.roi_height/2, 'bo', markersize=6, alpha=0.7, label='Center')
        
        ax.set_title(f'Shift: ({shift_x:+3d}, {shift_y:+3d})\nGT: ({gt_x:.1f}, {gt_y:.1f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 빈 서브플롯 제거
    total_plots = num_variations + 1
    for i in range(total_plots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = f"augmentation/augmentation_debug_sample_{sample_idx}.png"
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 증강 디버깅 저장: {save_path}")
    
    # training 모드 복원
    dataset.set_training_mode(original_training)
    
    # 환경에 따라 표시
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        plt.show()
    else:
        plt.close()

def load_real_frames(bin_file_path: str, max_frames: int = 20) -> list:
    """실제 DVS 데이터에서 프레임 로드"""
    print(f"📖 실제 프레임 로드 중... ({bin_file_path})")
    
    if not os.path.exists(bin_file_path):
        print(f"⚠️ 파일을 찾을 수 없음: {bin_file_path}")
        return None
    
    try:
        # 올바른 프레임 크기로 BinProcessor 초기화
        processor = BinProcessor(frame_width=960, frame_height=720)
        frames_data = processor.read_frames(bin_file_path, max_frames=max_frames)
        
        individual_frames = []
        for frame in frames_data:
            if hasattr(frame, 'raw_data'):
                # DVS 데이터 특성 확인
                data = frame.raw_data.astype(np.float32)
                individual_frames.append(data)
            else:
                print(f"⚠️ 프레임 데이터 형식 오류")
                return None
        
        print(f"✅ {len(individual_frames)}개 프레임 로드 완료")
        print(f"   프레임 크기: {individual_frames[0].shape}")
        print(f"   데이터 범위: {individual_frames[0].min()} ~ {individual_frames[0].max()}")
        return individual_frames
    
    except Exception as e:
        print(f"❌ 프레임 로드 실패: {e}")
        return None

def main():
    """메인 디버깅 함수"""
    print("🔬 데이터 증강 파이프라인 디버깅 시작")
    print("=" * 50)
    
    # 1. 실제 DVS 데이터 로드
    bin_file_path = "/hai/home/jdj/dvs/sim/data/gaussian_large.bin"
    
    if not os.path.exists(bin_file_path):
        print(f"❌ DVS 데이터 파일을 찾을 수 없음: {bin_file_path}")
        print("💡 올바른 경로를 확인하거나 train.py를 통해 학습용 데이터를 먼저 준비하세요.")
        return
    
    print("📖 실제 DVS 데이터 사용")
    individual_frames = load_real_frames(bin_file_path, max_frames=30)
    
    if individual_frames is None:
        print("❌ DVS 데이터 로드 실패")
        return
    
    # 2. 데이터셋 생성
    print("\n🎯 데이터셋 생성...")
    dataset = DVSFixedGTDataset(
        individual_frames=individual_frames,
        true_center_coord=(541, 361),  # 실제 빔 중심
        roi_size=(512, 512),
        temporal_window=1,  # 단일 프레임
        shift_range_x=80,
        shift_range_y=60
    )
    
    print(f"✅ 데이터셋 크기: {len(dataset)} 샘플")
    
    # 3. 일반적인 증강 샘플 시각화
    print("\n🔍 일반 증강 샘플 시각화...")
    visualize_augmentation_samples(
        dataset,
        num_samples=9, 
        save_path="augmentation/augmentation_samples.png"
    )
    
    # 4. 특정 샘플의 다양한 증강 시각화
    print("\n🔬 특정 샘플의 증강 변형 디버깅...")
    debug_augmentation_pipeline(
        dataset,
        sample_idx=0, 
        num_variations=15
    )
    
    # 5. 다른 샘플로도 테스트
    if len(dataset) > 5:
        print("\n🔬 다른 샘플로 추가 테스트...")
        debug_augmentation_pipeline(
            dataset,
            sample_idx=5, 
            num_variations=11
        )
    
    # 6. 증강 없이도 테스트 (비교용)
    print("\n🎯 증강 없는 버전 (비교용)...")
    dataset.set_training_mode(False)
    visualize_augmentation_samples(
        dataset,
        num_samples=6, 
        save_path="augmentation/no_augmentation_samples.png",
        show_original=True  # 원본 프레임 전체 표시
    )
    
    print("\n✅ 데이터 증강 디버깅 완료!")
    print("📁 생성된 파일들 (augmentation/ 디렉토리):")
    print("   - augmentation_samples.png")
    print("   - augmentation_debug_sample_0.png")
    print("   - augmentation_debug_sample_5.png") 
    print("   - no_augmentation_samples.png")
    
    print("\n💡 확인 사항:")
    print("   1. ROI 이미지에서 빨간 십자(+)가 레이저 중심에 정확히 위치하는가?")
    print("   2. 시프트가 적용될 때 GT 좌표가 올바르게 변경되는가?")
    print("   3. 정규화된 좌표(0-1)가 실제 픽셀 좌표와 일치하는가?")
    print("   4. 증강된 ROI가 원본 데이터의 올바른 영역을 포함하는가?")

if __name__ == "__main__":
    main()

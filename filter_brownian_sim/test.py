#!/usr/bin/env python3
"""
통일된 DVS 필터링 함수
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from dvs_filter import (
    FilterableBinProcessor, FrameFilter, EventDensityFilter, SpatialClusterFilter, 
    TimeRangeFilter, ROIFilter, DVSFrame,
    KalmanPointExtractor, MeanPointExtractor, MedianPointExtractor, CenterPointExtractor,
    TemporalAveragePointExtractor
)
from bin_utils import create_test_bin_file

@dataclass
class FilterConfig:
    """필터 설정을 담는 데이터 클래스"""
    filters: List[FrameFilter]
    name: str = "Custom"
    description: str = ""

@dataclass
class FilteringConfig:
    """필터링 전체 설정"""
    # 필터 설정
    filter_config: FilterConfig
    
    # 파일 관련
    bin_file_path: Optional[str] = None  # None이면 테스트 파일 생성
    has_header: bool = True  # 고정: 헤더 있음
    
    # 프레임 선택
    max_frames: Optional[int] = None  # 최대 처리 프레임 수
    
    # 출력 관련
    save_filtered: bool = False
    output_path: Optional[str] = None
    visualize: bool = True
    
    # 로깅/통계
    show_stats: bool = True
    verbose: bool = False
    show_step_by_step: bool = False

def unified_filter(config: FilteringConfig) -> Tuple[List[DVSFrame], List[DVSFrame]]:
    """
    통일된 DVS 필터링 함수
    
    Args:
        config: 필터링 설정
        
    Returns:
        (original_frames, filtered_frames) 튜플
    """
    
    if config.verbose:
        print("\n" + "="*60)
        print(f"DVS FILTERING: {config.filter_config.name}")
        print(f"Description: {config.filter_config.description}")
        print("="*60)
    
    # 1. 파일 및 해상도 설정
    if config.bin_file_path is None:
        # 테스트 파일 생성
        width, height = 300, 200
        num_frames = config.max_frames or 8
        bin_file_path = "bin/unified_test.bin"
        create_test_bin_file(bin_file_path, width, height, num_frames)
        if config.verbose:
            print(f"📝 Created test file: {bin_file_path}")
            print(f"   Resolution: {width}×{height}, Frames: {num_frames}")
    else:
        # 실제 파일 사용 (Brownian motion dataset: 512x512)
        width, height = 512, 512
        bin_file_path = config.bin_file_path
        if config.verbose:
            print(f"📂 Loading file: {bin_file_path}")
            print(f"   Resolution: {width}×{height} (Brownian motion ROI)")
    
    # 2. FilterableBinProcessor 생성
    processor = FilterableBinProcessor(width, height, has_header=config.has_header)
    
    # 3. Read frames
    if config.verbose:
        if config.max_frames:
            print(f"\n📖 Reading frames (max: {config.max_frames})...")
        else:
            print(f"\n📖 Reading all frames...")
    
    # Read all frames from file
    all_frames = processor.read_frames(bin_file_path, max_frames=config.max_frames)
    
    # 4. Apply frame limit
    if config.max_frames:
        original_frames = all_frames[:config.max_frames]
        if config.verbose:
            print(f"   Selected: {len(original_frames)} out of {len(all_frames)} frames")
    else:
        original_frames = all_frames
    
    if config.verbose:
        print(f"   Processing: {len(original_frames)} frames")
    
    # 5. Setup filters
    processor.clear_filters()
    for filter_obj in config.filter_config.filters:
        processor.add_filter(filter_obj)
    
    if config.verbose:
        print(f"\n🔧 Configured {len(config.filter_config.filters)} filters:")
        for i, filter_obj in enumerate(config.filter_config.filters, 1):
            print(f"   {i}. {filter_obj.get_name()}")
    
    # 6. Step-by-step analysis (optional)
    if config.show_step_by_step:
        show_step_by_step_filtering(original_frames, config.filter_config.filters)
    
    # 7. Apply filtering
    if config.verbose:
        print(f"\n⚡ Applying filters to {len(original_frames)} frames...")
    
    output_path = config.output_path if config.save_filtered else None
    
    # 선택된 프레임에 대해서만 필터링 수행하기 위해 임시 파일 생성 대신 직접 처리
    filtered_frames = []
    for frame in original_frames:
        # 모든 필터 적용
        keep_frame = True
        current_frame = frame.copy()
        
        for filter_obj in config.filter_config.filters:
            if not filter_obj.apply(current_frame):
                keep_frame = False
                break
        
        if keep_frame:
            filtered_frames.append(current_frame)
    
    # 8. Show statistics
    if config.show_stats:
        total_original = len(original_frames)
        total_filtered = len(filtered_frames)
        filtered_out = total_original - total_filtered
        
        events_before = sum(f.count_events()['total_events'] for f in original_frames)
        events_after = sum(f.count_events()['total_events'] for f in filtered_frames)
        reduction = (1 - events_after / events_before) * 100 if events_before > 0 else 0
        
        print(f"\n📊 PROCESSING RESULTS")
        print("─" * 60)
        print(f"📥 Input frames:       {total_original}")
        print(f"📤 Output frames:      {total_filtered}")
        print(f"🗑️  Filtered out:       {filtered_out}")
        print(f"📈 Events before:      {events_before:,}")
        print(f"📉 Events after:       {events_after:,}")
        print(f"📊 Event reduction:    {reduction:.1f}%")
        print("─" * 60)
    
    # 9. Visualization
    if config.visualize:
        if config.verbose:
            print(f"\n🎨 Generating visualization...")
        
        if len(filtered_frames) == 0:
            print("⚠️  No frames remaining after filtering - skipping visualization")
        else:
            visualize_results(original_frames, filtered_frames, frame_width=width, frame_height=height)
    
    # 10. 정리
    if config.bin_file_path is None:
        os.remove(bin_file_path)
    
    return original_frames, filtered_frames

def show_step_by_step_filtering(frames: List[DVSFrame], filters: List[FrameFilter]):
    """Show step-by-step effect of each filter"""
    
    print(f"\n🔍 STEP-BY-STEP FILTER ANALYSIS")
    print("─" * 60)
    
    if not frames:
        print("No frames to analyze")
        return
    
    # Analyze first frame only
    test_frame = frames[0].copy()
    print(f"📋 Test frame: timestamp={test_frame.header.timestamp}")
    
    original_count = test_frame.count_events()['total_events']
    print(f"🎯 Original events: {original_count}")
    print()
    
    current_frame = test_frame
    for i, filter_obj in enumerate(filters):
        frame_copy = current_frame.copy()
        
        events_before = frame_copy.count_events()['total_events']
        passed = filter_obj.apply(frame_copy)
        
        if passed:
            events_after = frame_copy.count_events()['total_events']
            change = events_after - events_before
            status = "✅ PASS" if change >= 0 else "🔄 MODIFIED"
            print(f"   {i+1}. {filter_obj.get_name()}")
            print(f"      {events_before} → {events_after} ({change:+d}) {status}")
            current_frame = frame_copy
        else:
            print(f"   {i+1}. {filter_obj.get_name()}")
            print(f"      {events_before} → ❌ FILTERED OUT")
            break
    
    print("─" * 60)

def process_and_extract_center_points(
    filtering_config: FilteringConfig,
    point_extractor: CenterPointExtractor
) -> Tuple[List[DVSFrame], List[DVSFrame], List[Optional[np.ndarray]]]:
    """
    노이즈 필터링 후 중심점을 추출하는 통합 프로세스.

    Args:
        filtering_config: 노이즈 필터링을 위한 설정.
        point_extractor: 사용할 중심점 추출기 객체.

    Returns:
        (original_frames, filtered_frames, center_points) 튜플.
        center_points는 각 필터링된 프레임에 대한 중심점 좌표 리스트.
    """
    print("\n" + "="*60)
    print(f"PROCESS: Filtering with '{filtering_config.filter_config.name}'")
    print(f"EXTRACT: Center points with '{point_extractor.get_name()}'")
    print("="*60)
    
    # 1. 노이즈 필터링 수행 (기존 unified_filter 로직 재활용)
    # 시각화는 여기서 하지 않고, 나중에 중심점과 함께 한번에 처리
    filtering_config.visualize = False 
    original_frames, filtered_frames = unified_filter(filtering_config)
    
    # 2. 중심점 추출
    print(f"\n✨ Extracting center points from {len(filtered_frames)} filtered frames...")
    center_points = []
    
    # 칼만 필터의 경우 상태를 초기화
    if hasattr(point_extractor, 'reset'):
        point_extractor.reset()

    for frame in filtered_frames:
        center = point_extractor.extract(frame)
        center_points.append(center)
    
    # 결과 요약
    valid_points = sum(1 for p in center_points if p is not None)
    print(f"   Extraction complete. Found {valid_points} valid center points.")

    return original_frames, filtered_frames, center_points

def visualize_results(original_frames, filtered_frames, center_points=None, frame_width=None, frame_height=None, title_suffix="", point_extractor=None):
    """Visualize filtering results"""
    # Auto-detect frame dimensions if not provided
    if frame_width is None or frame_height is None:
        if len(original_frames) > 0:
            frame_height, frame_width = original_frames[0].raw_data.shape
        else:
            frame_width, frame_height = 512, 512  # Brownian motion ROI
    
    # Original frames에서도 중심점 계산 (기본 추출기 사용)
    original_center_points = []
    if point_extractor is not None:
        # 기본 추출기를 사용하여 원본 프레임의 중심점 계산
        base_extractor = MedianPointExtractor()  # 기본으로 MedianPointExtractor 사용
        for frame in original_frames:
            center = base_extractor.extract(frame)
            original_center_points.append(center)
    
    num_frames_to_show = len(original_frames)
    fig, axes = plt.subplots(2, num_frames_to_show, 
                            figsize=(3*num_frames_to_show, 8), squeeze=False)
    
    for i in range(num_frames_to_show):
        # Original frame
        ax_orig = axes[0, i]
        orig_frame = original_frames[i].raw_data
        on_events_orig = np.where(orig_frame == 1)
        off_events_orig = np.where(orig_frame == 2)
        
        # Empty plots for legend (always have labels)
        ax_orig.scatter([], [], c='red', s=2, alpha=0.7, label='ON events')
        ax_orig.scatter([], [], c='blue', s=2, alpha=0.7, label='OFF events')
        ax_orig.scatter([], [], c='orange', s=50, marker='*', label='Original center')
        
        if len(on_events_orig[0]) > 0:
            ax_orig.scatter(on_events_orig[1], on_events_orig[0], 
                           c='red', s=2, alpha=0.7)
        if len(off_events_orig[0]) > 0:
            ax_orig.scatter(off_events_orig[1], off_events_orig[0], 
                           c='blue', s=2, alpha=0.7)
        
        # 원본 프레임의 중심점 표시
        center_text_orig = ""
        if i < len(original_center_points) and original_center_points[i] is not None:
            orig_center = original_center_points[i]
            ax_orig.scatter(orig_center[0], orig_center[1], c='orange', s=50, marker='*', edgecolors='black')
            center_text_orig = f"\nOriginal Center: ({orig_center[0]:.1f}, {orig_center[1]:.1f})"
        else:
            center_text_orig = "\nOriginal Center: N/A"
        
        ax_orig.set_title(f'Original Frame {i}\nEvents: {original_frames[i].count_events()["total_events"]}{center_text_orig}')
        ax_orig.set_xlim(0, frame_width)
        ax_orig.set_ylim(frame_height, 0)
        ax_orig.legend(fontsize=8)
        ax_orig.grid(True, alpha=0.3)
        
        # Filtered frame - proper frame matching
        orig_timestamp = original_frames[i].header.timestamp
        matching_filtered_frame = None
        
        for filt_frame in filtered_frames:
            if filt_frame.header.timestamp == orig_timestamp:
                matching_filtered_frame = filt_frame
                break
        
        if matching_filtered_frame is not None:
            ax_filt = axes[1, i]
            filt_frame_data = matching_filtered_frame.raw_data
            on_events_filt = np.where(filt_frame_data == 1)
            off_events_filt = np.where(filt_frame_data == 2)
            
            # Empty plots for legend
            ax_filt.scatter([], [], c='red', s=2, alpha=0.7, label='ON events')
            ax_filt.scatter([], [], c='blue', s=2, alpha=0.7, label='OFF events')
            
            # Add center point legend if provided
            if center_points is not None:
                # 중심점 추출기 이름 기반으로 라벨 결정
                if point_extractor is not None:
                    center_label = f'{point_extractor.get_name()} center'
                else:
                    center_label = 'Center point'
                ax_filt.scatter([], [], c='yellow', s=50, marker='*', label=center_label)
            
            if len(on_events_filt[0]) > 0:
                ax_filt.scatter(on_events_filt[1], on_events_filt[0], 
                               c='red', s=2, alpha=0.7)
            if len(off_events_filt[0]) > 0:
                ax_filt.scatter(off_events_filt[1], off_events_filt[0], 
                               c='blue', s=2, alpha=0.7)
            
            # Add center point if available
            center_text = ""
            if center_points is not None:
                # Find the corresponding center point for this filtered frame
                frame_idx = None
                for idx, filt_frame in enumerate(filtered_frames):
                    if filt_frame.header.timestamp == matching_filtered_frame.header.timestamp:
                        frame_idx = idx
                        break
                
                if frame_idx is not None and frame_idx < len(center_points):
                    center = center_points[frame_idx]
                    if center is not None:
                        ax_filt.scatter(center[0], center[1], c='yellow', s=50, marker='*', edgecolors='black')
                        # 중심점 추출기 이름 기반으로 라벨 결정
                        if point_extractor is not None:
                            center_label = point_extractor.get_name()
                        else:
                            center_label = "Center"
                        center_text = f"\n{center_label}: ({center[0]:.1f}, {center[1]:.1f})"
                    else:
                        if point_extractor is not None:
                            center_label = point_extractor.get_name()
                        else:
                            center_label = "Center"
                        center_text = f"\n{center_label}: N/A"
            
            ax_filt.set_title(f'Filtered Frame {i}\nEvents: {matching_filtered_frame.count_events()["total_events"]}{center_text}')
            ax_filt.set_xlim(0, frame_width)
            ax_filt.set_ylim(frame_height, 0)
            ax_filt.legend(fontsize=8)
            ax_filt.grid(True, alpha=0.3)
        else:
            # Frame was filtered out
            axes[1, i].text(0.5, 0.5, 'FILTERED OUT', 
                            transform=axes[1, i].transAxes, 
                            ha='center', va='center', fontsize=12, color='red')
            axes[1, i].set_xlim(0, frame_width)
            axes[1, i].set_ylim(frame_height, 0)
    
    # Enhanced title with filter information
    main_title = 'DVS Frame Filtering Results'
    if center_points is not None:
        main_title += ' with Center Points'
    if title_suffix:
        main_title += f'\n{title_suffix}'
    
    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92 if title_suffix else 0.96])
    plt.show()

def compare_filters(filters_list, bin_file_path=None, max_frames=3, visualize=True, verbose=True, point_extractor=None):
    """
    Compare multiple filters or filter chains using unified_filter
    
    Args:
        filters_list: 다음 중 하나:
            - List[FrameFilter]: 개별 필터들 (기존 방식)
            - List[FilterConfig]: 필터 체인들
            - List[List[FrameFilter]]: 필터 체인들 (자동으로 FilterConfig 생성)
    """
    print(f"\n🔍 FILTER COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # 입력 타입에 따라 처리
    processed_configs = []
    for i, item in enumerate(filters_list):
        if isinstance(item, FilterConfig):
            # 이미 FilterConfig인 경우
            processed_configs.append(item)
        elif isinstance(item, list):
            # 필터 리스트인 경우 → FilterConfig로 변환
            filter_names = [f.get_name() for f in item]
            chain_name = " → ".join(filter_names)
            config = FilterConfig(filters=item, name=f"Chain: {chain_name}")
            processed_configs.append(config)
        else:
            # 단일 필터인 경우 → FilterConfig로 변환
            config = FilterConfig(filters=[item], name=item.get_name())
            processed_configs.append(config)
    
    # Test each filter configuration
    for i, filter_config in enumerate(processed_configs):
        print(f"\n📊 Testing {i+1}: {filter_config.name}")
        print("-" * 60)
        
        # Use unified_filter directly
        config = FilteringConfig(
            filter_config=filter_config,
            bin_file_path=bin_file_path,
            max_frames=max_frames,
            verbose=verbose,
            show_step_by_step=verbose,
            visualize=False,  # Handle visualization separately
            show_stats=True
        )
        
        original, filtered = unified_filter(config)
        
        # Extract center points if extractor provided
        centers = None
        if point_extractor is not None:
            print(f"   🎯 Extracting center points with {point_extractor.get_name()}...")
            if hasattr(point_extractor, 'reset'):
                point_extractor.reset()
            centers = []
            for frame in filtered:
                center = point_extractor.extract(frame)
                centers.append(center)
            valid_centers = sum(1 for c in centers if c is not None)
            print(f"   Found {valid_centers} valid center points.")
        
        # Calculate statistics
        events_before = sum(f.count_events()['total_events'] for f in original)
        events_after = sum(f.count_events()['total_events'] for f in filtered)
        reduction = (1 - events_after / events_before) * 100 if events_before > 0 else 0
        
        results[filter_config.name] = {
            'original': original,
            'filtered': filtered,
            'centers': centers,
            'events_before': events_before,
            'events_after': events_after,
            'reduction': reduction
        }
        
        print(f"✅ Result: {events_before} → {events_after} events ({reduction:.1f}% reduction)")
        
        # Show individual visualization if requested
        if visualize and len(filtered) > 0:
            filter_info = f"Filter #{i+1}: {filter_config.name} | " + \
                         f"Event Reduction: {reduction:.1f}% | Frames: {len(original)}"
            if centers is not None:
                filter_info += f" | Valid Centers: {valid_centers}"
            visualize_results(original, filtered, center_points=centers, title_suffix=filter_info, point_extractor=point_extractor)
    
    # Show comparison table
    print(f"\n📋 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Filter Name':<30} {'Before':<8} {'After':<8} {'Reduction':<10}")
    print("-" * 80)
    
    for name, data in results.items():
        print(f"{name:<30} {data['events_before']:<8} {data['events_after']:<8} {data['reduction']:.1f}%")
    
    return results

if __name__ == "__main__":
    print("🚀 DVS Filtering System with Center Point Extraction")
    print("=" * 60)
    
    # 중심점 추출기 선택
    point_extractor = MeanPointExtractor()
    #point_extractor = MedianPointExtractor()
    #point_extractor = KalmanPointExtractor(process_noise=4.0)  # Brownian motion: sigma^2 = 4.0
    
        # 템포럴 평균 추출기 (5프레임 윈도우, 기존 추출기 재활용)
    # point_extractor = TemporalAveragePointExtractor(window_size=5)  # 기본: MeanPointExtractor
    # point_extractor = TemporalAveragePointExtractor(window_size=5, base_extractor=MedianPointExtractor())
    # point_extractor = TemporalAveragePointExtractor(window_size=3, base_extractor=MedianPointExtractor())
    
    # 개별 필터 vs 필터 체인 비교
    compare_filters([
        # 개별 필터들
        SpatialClusterFilter(radius=5.0, min_neighbors=1),
        
        
        
        # # 필터 체인
        # FilterConfig(
        #     filters=[
        #         EventDensityFilter(min_events=500),
        #         SpatialClusterFilter(radius=5.0, min_neighbors=1)
        #     ],
        #     name="Custom 2-Stage Pipeline",
        #     description="Density + Spatial"
        # )
    ], 
    # bin_file_path="/hai/home/jdj/dvs/data/low_freq_large.bin",  # 실제 파일 사용
    # bin_file_path="/hai/home/jdj/dvs/data/1kHz_large.bin",  # 실제 파일 사용
    bin_file_path="/hai/home/jdj/dvs/data/gaussian_brownian_512x512.bin",  # Brownian motion dataset
    
    max_frames=10,
    visualize=True,
    verbose=True,  # 간단한 로그로
    point_extractor=point_extractor
    )
    
    

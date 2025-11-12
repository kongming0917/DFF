#!/usr/bin/env python3
"""
DVS 필터링 결과를 CSV 파일로 내보내는 유틸리티

이 스크립트는 다음 데이터를 CSV로 저장합니다:
1. 필터 없이 모든 프레임의 중심점 좌표
2. SpatialClusterFilter 적용 후 모든 프레임의 중심점 좌표
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import List, Optional, Tuple
import time
from dvs_filter import (
    FilterableBinProcessor, SpatialClusterFilter, DVSFrame, 
    MedianPointExtractor, TemporalAveragePointExtractor, KalmanPointExtractor
)

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    터미널에 진행률 바를 출력하는 함수 (한 줄 덮어쓰기)
    
    Args:
        iteration: 현재 진행 단계
        total: 전체 단계 수
        prefix: 바 앞에 표시할 텍스트
        suffix: 바 뒤에 표시할 텍스트
        length: 바의 길이
        fill: 채움 문자
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # \r로 커서를 줄 처음으로 이동하여 덮어쓰기
    progress_line = f'\r{prefix} |{bar}| {percent}% {suffix}'
    
    # 이전 출력을 완전히 덮어쓰기 위해 충분한 공백 추가
    progress_line = progress_line.ljust(80)
    
    sys.stdout.write(progress_line)
    sys.stdout.flush()
    
    # 완료 시에만 새 줄
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def extract_all_center_points(
    bin_file_path: str, 
    filters: List = None, 
    point_extractor = None,
    max_frames: Optional[int] = None
) -> List[Tuple[int, Optional[float], Optional[float]]]:
    """
    모든 프레임에서 중심점을 추출하여 리스트로 반환
    
    Args:
        bin_file_path: DVS bin 파일 경로
        filters: 적용할 필터 리스트 (None이면 필터 없음)
        point_extractor: 중심점 추출기 (None이면 MedianPointExtractor 사용)
        max_frames: 최대 처리 프레임 수
        
    Returns:
        List[(frame_number, center_x, center_y)]
    """
    # 기본 설정 (Brownian motion dataset: 512x512)
    width, height = 512, 512
    processor = FilterableBinProcessor(width, height, has_header=True)
    
    if point_extractor is None:
        point_extractor = MedianPointExtractor()
    
    # 필터 설정
    processor.clear_filters()
    if filters:
        for filter_obj in filters:
            processor.add_filter(filter_obj)
    
    # 프레임 읽기
    print(f"📖 Reading frames from {bin_file_path}...")
    start_time = time.time()
    all_frames = processor.read_frames(bin_file_path, max_frames=max_frames)
    read_time = time.time() - start_time
    print(f"   ⏱️ Reading took {read_time:.2f} seconds")
    
    # 필터 적용 (있는 경우)
    if filters:
        print(f"🔧 Applying {len(filters)} filters...")
        filter_start_time = time.time()
        filtered_frames = []
        total_frames = len(all_frames)
        
        for i, frame in enumerate(all_frames):
            # 진행률 바 업데이트 (매 10프레임마다 또는 마지막)
            if i % 10 == 0 or i == total_frames - 1:
                elapsed = time.time() - filter_start_time
                eta = elapsed / (i + 1) * (total_frames - i - 1) if i > 0 else 0
                print_progress_bar(i + 1, total_frames, 
                                 prefix='   🔧 Filtering', 
                                 suffix=f'({i+1}/{total_frames}) ETA: {eta:.0f}s')
            
            keep_frame = True
            current_frame = frame.copy()
            
            for filter_obj in filters:
                if not filter_obj.apply(current_frame):
                    keep_frame = False
                    break
            
            if keep_frame:
                filtered_frames.append(current_frame)
        
        filter_time = time.time() - filter_start_time
        frames_to_process = filtered_frames
        print(f"   ✅ Filtering complete: {len(filtered_frames)}/{total_frames} frames kept ({len(filtered_frames)/total_frames*100:.1f}%)")
        print(f"   ⏱️ Filtering took {filter_time:.2f} seconds")
    else:
        frames_to_process = all_frames
        print(f"   Processing all frames: {len(frames_to_process)}")
    
    # 중심점 추출
    print(f"✨ Extracting center points...")
    extraction_start_time = time.time()
    if hasattr(point_extractor, 'reset'):
        point_extractor.reset()
    
    results = []
    total_frames = len(frames_to_process)
    valid_centers = 0
    
    for i, frame in enumerate(frames_to_process):
        # 진행률 바 업데이트 (매 5프레임마다 또는 마지막)
        if i % 5 == 0 or i == total_frames - 1:
            elapsed = time.time() - extraction_start_time
            eta = elapsed / (i + 1) * (total_frames - i - 1) if i > 0 else 0
            print_progress_bar(i + 1, total_frames, 
                             prefix='   ✨ Extracting', 
                             suffix=f'({i+1}/{total_frames}) Valid: {valid_centers} ETA: {eta:.0f}s')
        
        center = point_extractor.extract(frame)
        
        if center is not None:
            results.append((frame.header.frame_number, float(center[0]), float(center[1])))
            valid_centers += 1
        else:
            results.append((frame.header.frame_number, None, None))
    
    extraction_time = time.time() - extraction_start_time
    print(f"   ✅ Extraction complete: {valid_centers}/{len(results)} valid center points ({valid_centers/len(results)*100:.1f}%)")
    print(f"   ⏱️ Extraction took {extraction_time:.2f} seconds")
    return results

def save_to_csv(data: List[Tuple], filename: str, description: str = ""):
    """
    중심점 데이터를 CSV 파일로 저장
    
    Args:
        data: [(frame_number, center_x, center_y)] 형태의 데이터
        filename: 저장할 파일명
        description: 파일 설명 (헤더에 추가)
    """
    # DataFrame 생성
    df = pd.DataFrame(data, columns=['frame_number', 'center_x', 'center_y'])
    
    # 통계 정보 계산
    valid_points = df.dropna()
    total_frames = len(df)
    valid_count = len(valid_points)
    invalid_count = total_frames - valid_count
    
    # CSV 파일로 저장
    df.to_csv(filename, index=False)
    
    # 결과 출력
    print(f"\n📊 {description}")
    print(f"   Saved to: {filename}")
    print(f"   Total frames: {total_frames}")
    print(f"   Valid centers: {valid_count}")
    print(f"   Invalid centers: {invalid_count}")
    
    if valid_count > 0:
        print(f"   Center X range: {valid_points['center_x'].min():.1f} ~ {valid_points['center_x'].max():.1f}")
        print(f"   Center Y range: {valid_points['center_y'].min():.1f} ~ {valid_points['center_y'].max():.1f}")
        print(f"   Mean center: ({valid_points['center_x'].mean():.1f}, {valid_points['center_y'].mean():.1f})")

def export_center_comparison(
    bin_file_path: str, 
    output_dir: str = "csv_results",
    max_frames: Optional[int] = None,
    use_temporal_average: bool = False,
    temporal_window: int = 3,
    include_kalman: bool = True
):
    """
    필터 적용 전후의 중심점 데이터를 CSV로 내보내기
    
    Args:
        bin_file_path: DVS bin 파일 경로
        output_dir: 출력 디렉토리
        max_frames: 최대 처리 프레임 수
        use_temporal_average: 템포럴 평균 사용 여부
        temporal_window: 템포럴 윈도우 크기
        include_kalman: 칼만 필터 포함 여부
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 DVS Center Point Data Export")
    print("=" * 60)
    
    # 중심점 추출기 설정
    extractors = {}
    
    if use_temporal_average:
        extractor_name = f"temporal_avg_w{temporal_window}"
        extractors["temporal"] = TemporalAveragePointExtractor(
            window_size=temporal_window, 
            base_extractor=MedianPointExtractor()
        )
        print(f"📍 Using TemporalAveragePointExtractor (window={temporal_window})")
    else:
        extractor_name = "median"
        extractors["median"] = MedianPointExtractor()
        print(f"📍 Using MedianPointExtractor")
    
    # 칼만 필터 추가
    if include_kalman:
        # Brownian motion: sigma = 2.0, so process_noise = sigma^2 = 4.0
        extractors["kalman"] = KalmanPointExtractor(process_noise=4.0, measurement_noise=8.0)
        print(f"📍 Adding KalmanPointExtractor (Brownian motion: process_noise=4.0)")
    
    # 각 추출기별로 처리
    all_results = {}
    
    for ext_name, extractor in extractors.items():
        print(f"\n🔍 Processing with {extractor.get_name()}")
        
        # 1. 필터 없음 - 원본 데이터
        print(f"   📋 No Filter")
        no_filter_data = extract_all_center_points(
            bin_file_path=bin_file_path,
            filters=None,
            point_extractor=extractor,
            max_frames=max_frames
        )
        
        no_filter_file = os.path.join(output_dir, f"no_filter_{ext_name}.csv")
        save_to_csv(no_filter_data, no_filter_file, f"No Filter - {extractor.get_name()}")
        
        # 2. 적응적 SpatialClusterFilter 적용
        print(f"   📋 SpatialClusterFilter (adaptive)")
        # 이벤트 수에 따라 자동으로 최적 알고리즘 선택
        spatial_filter = SpatialClusterFilter(radius=5.0, min_neighbors=2, use_fast_mode=True)
        
        # 추출기 리셋 (상태가 있는 경우)
        if hasattr(extractor, 'reset'):
            extractor.reset()
        
        spatial_filter_data = extract_all_center_points(
            bin_file_path=bin_file_path,
            filters=[spatial_filter],
            point_extractor=extractor,
            max_frames=max_frames
        )
        
        spatial_filter_file = os.path.join(output_dir, f"spatial_filter_{ext_name}.csv")
        save_to_csv(spatial_filter_data, spatial_filter_file, f"SpatialClusterFilter - {extractor.get_name()}")
        
        all_results[ext_name] = {
            'no_filter': no_filter_data,
            'spatial_filter': spatial_filter_data,
            'extractor_name': extractor.get_name()
        }
    
    # 비교 요약
    print(f"\n📋 COMPARISON SUMMARY")
    print("=" * 80)
    
    for ext_name, results in all_results.items():
        no_filter_data = results['no_filter']
        spatial_filter_data = results['spatial_filter']
        extractor_name = results['extractor_name']
        
        valid_no_filter = len([d for d in no_filter_data if d[1] is not None])
        valid_spatial = len([d for d in spatial_filter_data if d[1] is not None])
        
        print(f"\n{extractor_name}:")
        print(f"   No Filter - Valid centers: {valid_no_filter}/{len(no_filter_data)}")
        print(f"   Spatial Filter - Valid centers: {valid_spatial}/{len(spatial_filter_data)}")
        print(f"   Frame reduction: {len(no_filter_data) - len(spatial_filter_data)}")
    
    print(f"\n📁 Output files saved in: {output_dir}/")
    for ext_name in all_results.keys():
        print(f"   - no_filter_{ext_name}.csv")
        print(f"   - spatial_filter_{ext_name}.csv")

if __name__ == "__main__":
    # 설정
    #BIN_FILE_PATH = "/hai/home/jdj/dvs/data/1kHz_large.bin"
    BIN_FILE_PATH = "/hai/home/jdj/dvs/data/gaussian_brownian_512x512.bin"
    OUTPUT_DIR = "csv_results"
    #MAX_FRAMES = None  # None이면 모든 프레임 처리
    MAX_FRAMES = 100   # gaussian 데이터는 매우 느려서 50프레임만 테스트
    
    # 기본 중심점 추출기들로 내보내기 (Median + Kalman)
    print("🎯 Exporting with MedianPointExtractor + KalmanPointExtractor...")
    export_center_comparison(
        bin_file_path=BIN_FILE_PATH,
        output_dir=OUTPUT_DIR,
        max_frames=MAX_FRAMES,
        use_temporal_average=False,
        include_kalman=True
    )
    
    print("\n" + "="*60)
    
    # 템포럴 평균 추출기로 내보내기 (Temporal + Kalman)
    print("🎯 Exporting with TemporalAveragePointExtractor + KalmanPointExtractor...")
    export_center_comparison(
        bin_file_path=BIN_FILE_PATH,
        output_dir=OUTPUT_DIR + "_temporal",
        max_frames=MAX_FRAMES,
        use_temporal_average=True,
        temporal_window=3,
        include_kalman=True
    )


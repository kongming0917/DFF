#!/usr/bin/env python3
"""
gaussian_large.bin 파일 구조 확인 스크립트
"""
import struct
import numpy as np
import os


def analyze_bin_file(bin_file, num_frames=3):
    """
    DVS bin 파일을 분석하는 함수
    
    Args:
        bin_file (str): 분석할 bin 파일 경로
        num_frames (int): 분석할 프레임 개수 (기본값: 3)
    """
    print("=" * 70)
    print(f"DVS {os.path.basename(bin_file)} 파일 구조 분석")
    print("=" * 70)
    
    # 파일 크기 확인
    file_size = os.path.getsize(bin_file)
    print(f"\n📁 파일 크기: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # 프레임 크기 계산
    frame_width = 960
    frame_height = 720
    header_size = 8  # timestamp(4) + frame_number(4)
    image_data_size = (frame_width * frame_height) // 4  # 2bit 압축
    frame_size = header_size + image_data_size
    
    print(f"📐 예상 프레임 구조:")
    print(f"   - 해상도: {frame_width} × {frame_height}")
    print(f"   - 헤더 크기: {header_size} bytes")
    print(f"   - 이미지 데이터 크기: {image_data_size:,} bytes")
    print(f"   - 총 프레임 크기: {frame_size:,} bytes")
    print(f"   - 예상 프레임 수: {file_size // frame_size}")
    
    # 실제 데이터 읽기
    print("\n" + "=" * 70)
    print(f"처음 {num_frames}개 프레임 상세 분석")
    print("=" * 70)
    
    with open(bin_file, 'rb') as f:
        for frame_idx in range(num_frames):
            print(f"\n📊 Frame {frame_idx}:")
            print("-" * 50)
            
            # 헤더 읽기
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                print(f"   ⚠️  파일 끝 도달 (읽은 바이트: {len(header_bytes)})")
                break
            
            timestamp, frame_number = struct.unpack('<II', header_bytes)
            
            print(f"   📋 헤더:")
            print(f"      - Raw bytes: {header_bytes.hex()}")
            print(f"      - Timestamp: {timestamp} (0x{timestamp:08x})")
            print(f"      - Frame number: {frame_number}")
            
            # 이미지 데이터 읽기
            image_bytes = f.read(image_data_size)
            if len(image_bytes) < image_data_size:
                print(f"   ⚠️  이미지 데이터 부족 (읽은 바이트: {len(image_bytes)})")
                break
            
            print(f"\n   🖼️  이미지 데이터:")
            print(f"      - 크기: {len(image_bytes):,} bytes")
            print(f"      - 처음 20 bytes (hex): {image_bytes[:20].hex()}")
            print(f"      - 처음 20 bytes (dec): {[b for b in image_bytes[:20]]}")
            
            # 2bit 디코딩 샘플 (처음 4 bytes = 16 pixels)
            print(f"\n   🔍 2bit 디코딩 샘플 (첫 4 bytes = 16 pixels):")
            for i in range(min(4, len(image_bytes))):
                byte = image_bytes[i]
                p1 = (byte >> 6) & 0x03
                p2 = (byte >> 4) & 0x03
                p3 = (byte >> 2) & 0x03
                p4 = byte & 0x03
                print(f"      Byte {i} (0x{byte:02x} = {byte:3d}): [{p1}, {p2}, {p3}, {p4}]")
            
            # 전체 프레임 디코딩하여 이벤트 통계
            raw_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
            p1 = (raw_bytes >> 6) & 0x03
            p2 = (raw_bytes >> 4) & 0x03
            p3 = (raw_bytes >> 2) & 0x03
            p4 = raw_bytes & 0x03
            flat_pixels = np.stack([p1, p2, p3, p4], axis=-1).flatten()
            frame_array = flat_pixels[:frame_height * frame_width].reshape((frame_height, frame_width))
            
            # 이벤트 통계
            unique, counts = np.unique(frame_array, return_counts=True)
            event_stats = dict(zip(unique, counts))
            
            print(f"\n   📈 이벤트 통계 (전체 프레임):")
            print(f"      - 0 (no event): {event_stats.get(0, 0):,} pixels")
            print(f"      - 1 (ON event): {event_stats.get(1, 0):,} pixels")
            print(f"      - 2 (OFF event): {event_stats.get(2, 0):,} pixels")
            print(f"      - 3 (reserved): {event_stats.get(3, 0):,} pixels")
            print(f"      - 총 이벤트: {event_stats.get(1, 0) + event_stats.get(2, 0):,}")
            
            # ON 이벤트 위치 샘플
            on_events = np.argwhere(frame_array == 1)
            if len(on_events) > 0:
                print(f"\n   📍 ON 이벤트 위치 샘플 (최대 10개):")
                for i, (y, x) in enumerate(on_events[:10]):
                    print(f"      {i+1}. (x={x}, y={y})")
                
                # 중심 계산
                center_x = np.mean(on_events[:, 1])
                center_y = np.mean(on_events[:, 0])
                print(f"\n   🎯 ON 이벤트 중심 (평균):")
                print(f"      - Center: (x={center_x:.1f}, y={center_y:.1f})")
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)


def main():
    """메인 함수 - 여기서 설정을 변경하세요"""
    
    # ========== 설정 (여기를 수정하세요) ==========
    data_dir = '/hai/home/jdj/dvs/data'      # 데이터 디렉토리 경로
    file_name = 'gaussian_large.bin'          # 분석할 bin 파일명
    num_frames = 3                            # 분석할 프레임 개수
    # ============================================
    
    # 전체 파일 경로 생성
    bin_file = os.path.join(data_dir, file_name)
    
    # 파일 존재 여부 확인
    if not os.path.exists(bin_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {bin_file}")
        return
    
    # 분석 실행
    analyze_bin_file(bin_file, num_frames)


if __name__ == '__main__':
    main()


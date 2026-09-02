#!/usr/bin/env python3
"""
DVS bin 파일 프레임 단위 처리 (공통 라이브러리)

DVS 센서에서 생성된 bin 파일을 프레임 단위로 읽고 쓰는 기본 기능을 제공합니다.
필터링 기능은 제공하지 않으며, 순수한 bin 파일 I/O만 담당합니다.
"""

import numpy as np
import struct
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FrameHeader:
    """프레임 헤더 정보"""
    timestamp: int
    frame_number: int


@dataclass
class ProcessingStats:
    """처리 통계 정보"""
    total_frames: int = 0
    processed_frames: int = 0
    processing_time_sec: float = 0.0


class DVSFrame:
    """DVS 프레임 데이터 클래스"""
    
    def __init__(self, header: FrameHeader, raw_data: np.ndarray, 
                 width: int, height: int):
        self.header = header
        self.raw_data = raw_data  # 2bit 언패킹된 (height, width) 배열
        self.width = width
        self.height = height
        self._modified = False
    
    def get_pixel(self, x: int, y: int) -> int:
        """특정 위치의 픽셀 값 반환 (0: no event, 1: ON, 2: OFF)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.raw_data[y, x]
        return 0
    
    def set_pixel(self, x: int, y: int, value: int):
        """특정 위치의 픽셀 값 설정"""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= value <= 3:
            self.raw_data[y, x] = value
            self._modified = True
    
    def get_on_events(self) -> np.ndarray:
        """ON 이벤트(값=1) 좌표 추출 - (N, 2) 배열 반환"""
        y_coords, x_coords = np.where(self.raw_data == 1)
        return np.column_stack((x_coords, y_coords))
    
    def get_off_events(self) -> np.ndarray:
        """OFF 이벤트(값=2) 좌표 추출 - (N, 2) 배열 반환"""
        y_coords, x_coords = np.where(self.raw_data == 2)
        return np.column_stack((x_coords, y_coords))
    
    def get_all_events(self) -> Tuple[np.ndarray, np.ndarray]:
        """모든 이벤트 좌표 반환 - (ON_events, OFF_events)"""
        return self.get_on_events(), self.get_off_events()
    
    def count_events(self) -> Dict[str, int]:
        """이벤트 개수 통계"""
        unique, counts = np.unique(self.raw_data, return_counts=True)
        event_counts = dict(zip(unique, counts))
        return {
            'no_event': event_counts.get(0, 0),
            'on_event': event_counts.get(1, 0),
            'off_event': event_counts.get(2, 0),
            'total_events': event_counts.get(1, 0) + event_counts.get(2, 0)
        }
    
    def copy(self):
        """프레임 복사본 생성"""
        return DVSFrame(self.header, self.raw_data.copy(), self.width, self.height)


class BinProcessor:
    """bin 파일 프레임 단위 처리기 (베이스 클래스)"""
    
    def __init__(self, frame_width: int, frame_height: int, has_header: bool = True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.has_header = has_header
        self.header_size = 8 if has_header else 0  # timestamp(4) + frame_number(4)
        self.image_data_size = (frame_height * frame_width) // 4
        self.stats = ProcessingStats()
    
    def _decode_header(self, header_data: bytes) -> FrameHeader:
        """헤더 데이터 디코딩"""
        if len(header_data) >= 8:
            timestamp, frame_number = struct.unpack('<II', header_data)
            return FrameHeader(timestamp, frame_number)
        return FrameHeader(0, 0)
    
    def _unpack_2bit_data(self, packed_data: bytes) -> np.ndarray:
        """2bit 패킹된 데이터를 프레임 배열로 변환"""
        raw_bytes = np.frombuffer(packed_data, dtype=np.uint8)
        
        # 각 바이트에서 4개의 2bit 픽셀 추출
        p1 = (raw_bytes >> 6) & 0x03
        p2 = (raw_bytes >> 4) & 0x03
        p3 = (raw_bytes >> 2) & 0x03
        p4 = raw_bytes & 0x03
        
        # 순서대로 쌓아서 1D 배열 생성
        flat_pixels = np.stack([p1, p2, p3, p4], axis=-1).flatten()
        
        # 2D 프레임으로 reshape
        frame_array = flat_pixels[:self.frame_height * self.frame_width].reshape(
            (self.frame_height, self.frame_width))
        
        return frame_array
    
    def _pack_2bit_data(self, frame_array: np.ndarray) -> bytes:
        """프레임 배열을 2bit 패킹된 데이터로 변환"""
        flat_pixels = frame_array.flatten()
        packed_data = bytearray()
        
        for i in range(0, len(flat_pixels), 4):
            p1 = flat_pixels[i] if i < len(flat_pixels) else 0
            p2 = flat_pixels[i+1] if i+1 < len(flat_pixels) else 0
            p3 = flat_pixels[i+2] if i+2 < len(flat_pixels) else 0
            p4 = flat_pixels[i+3] if i+3 < len(flat_pixels) else 0
            
            # 2bit씩 패킹
            byte = (p1 << 6) | (p2 << 4) | (p3 << 2) | p4
            packed_data.append(byte)
        
        return bytes(packed_data)
    
    def read_frames(self, bin_file_path: str, max_frames: Optional[int] = None) -> List[DVSFrame]:
        """bin 파일에서 모든 프레임 읽기"""
        if not os.path.exists(bin_file_path):
            raise FileNotFoundError(f"File not found: {bin_file_path}")
        
        frames = []
        
        with open(bin_file_path, 'rb') as f:
            frame_count = 0
            while True:
                # max_frames에 도달하면 루프 종료
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # 헤더 읽기
                header_data = f.read(self.header_size)
                if len(header_data) < self.header_size:
                    break
                header = self._decode_header(header_data)
                
                # 이미지 데이터 읽기
                image_data = f.read(self.image_data_size)
                if len(image_data) < self.image_data_size:
                    break
                
                # 데이터 처리
                frame_array = self._unpack_2bit_data(image_data)
                frame = DVSFrame(header, frame_array, self.frame_width, self.frame_height)
                frames.append(frame)
                frame_count += 1

        return frames
    
    def write_frames(self, frames: List[DVSFrame], output_path: str):
        """프레임들을 bin 파일로 저장"""
        with open(output_path, 'wb') as f:
            for frame in frames:
                # 헤더 쓰기
                if self.has_header:
                    header_data = struct.pack('<II', 
                                            frame.header.timestamp, 
                                            frame.header.frame_number)
                    f.write(header_data)
                
                # 이미지 데이터 쓰기
                packed_data = self._pack_2bit_data(frame.raw_data)
                f.write(packed_data)
        
        print(f"Wrote {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    print("DVS Bin Processor Library")
    print("=" * 50)
    print("공통 bin 파일 처리 라이브러리")
    print("필터링 기능은 filter_sim/dvs_filter.py의 FilterableBinProcessor를 사용하세요.")


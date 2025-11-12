#!/usr/bin/env python3
"""
DVS bin file utilities

Utility functions for creating and managing DVS bin files.
"""

import numpy as np
import struct
import os

def create_test_bin_file(file_path: str, frame_width: int, frame_height: int, num_frames: int = 5):
    """
    Create a test dummy .bin file for demonstration purposes.
    The file contains noise and dense clusters (laser pulses).
    
    Args:
        file_path (str): Output file path
        frame_width (int): Frame width in pixels
        frame_height (int): Frame height in pixels
        num_frames (int): Number of frames to generate
    """
    # Ensure bin directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    print(f"Creating test dummy binary file at '{file_path}'...")
    with open(file_path, 'wb') as f:
        for i in range(num_frames):
            # 1. Generate header data
            timestamp = 1000 * i
            frame_number = i
            # <II: Little-endian, unsigned int, unsigned int
            header = struct.pack('<II', timestamp, frame_number)
            f.write(header)

            # 2. Generate image data (noise + laser pulse)
            # Create frame filled with 'no event' (0) by default
            frame = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Create dense 'ON' event cluster using Gaussian distribution in center
            if i % 2 == 0:  # Laser pulse only in even frames
                center_x = frame_width // 2 + np.random.randint(-20, 21)
                center_y = frame_height // 2 + np.random.randint(-20, 21)
                
                num_pulse_events = 150
                pulse_coords_x = np.random.normal(loc=center_x, scale=4, size=num_pulse_events).astype(int)
                pulse_coords_y = np.random.normal(loc=center_y, scale=4, size=num_pulse_events).astype(int)
                
                # Clip coordinates to be within frame boundaries
                pulse_coords_x = np.clip(pulse_coords_x, 0, frame_width - 1)
                pulse_coords_y = np.clip(pulse_coords_y, 0, frame_height - 1)
                
                frame[pulse_coords_y, pulse_coords_x] = 1  # ON events

            # Add random 'ON' event noise throughout the frame
            num_noise_events = 30
            noise_x = np.random.randint(0, frame_width, size=num_noise_events)
            noise_y = np.random.randint(0, frame_height, size=num_noise_events)
            frame[noise_y, noise_x] = np.random.choice([1, 2], num_noise_events)  # ON/OFF noise

            # 3. Convert 2D frame to 2bit-packed binary
            flat_pixels = frame.flatten()
            packed_data = bytearray()
            
            for j in range(0, len(flat_pixels), 4):
                p1 = flat_pixels[j] if j < len(flat_pixels) else 0
                p2 = flat_pixels[j+1] if j+1 < len(flat_pixels) else 0
                p3 = flat_pixels[j+2] if j+2 < len(flat_pixels) else 0
                p4 = flat_pixels[j+3] if j+3 < len(flat_pixels) else 0
                
                # Pack pixels according to specification with high bits first
                # (p1 << 6) | (p2 << 4) | (p3 << 2) | p4
                byte = (p1 << 6) | (p2 << 4) | (p3 << 2) | p4
                packed_data.append(byte)
            
            f.write(packed_data)
    print("Dummy file creation completed.")

def create_laser_pattern_bin_file(file_path: str, frame_width: int, frame_height: int, 
                                 num_frames: int = 10, laser_intensity: int = 200,
                                 laser_radius: float = 8.0):
    """
    Create a bin file with more realistic laser patterns.
    
    Args:
        file_path (str): Output file path
        frame_width (int): Frame width
        frame_height (int): Frame height
        num_frames (int): Number of frames
        laser_intensity (int): Number of events in laser pulse
        laser_radius (float): Laser pulse radius (standard deviation)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    print(f"Creating laser pattern bin file at '{file_path}'...")
    with open(file_path, 'wb') as f:
        for i in range(num_frames):
            # Header
            timestamp = 1000 * i + np.random.randint(0, 50)  # Add some jitter
            frame_number = i
            header = struct.pack('<II', timestamp, frame_number)
            f.write(header)

            # Frame data
            frame = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Laser pulse (every 3rd frame)
            if i % 3 == 0:
                # Moving laser spot
                center_x = int(frame_width * 0.3 + (frame_width * 0.4) * (i / num_frames))
                center_y = int(frame_height * 0.3 + (frame_height * 0.4) * (i / num_frames))
                
                # Dense ON events for laser
                pulse_x = np.random.normal(center_x, laser_radius, laser_intensity).astype(int)
                pulse_y = np.random.normal(center_y, laser_radius, laser_intensity).astype(int)
                
                valid_mask = (pulse_x >= 0) & (pulse_x < frame_width) & \
                            (pulse_y >= 0) & (pulse_y < frame_height)
                pulse_x = pulse_x[valid_mask]
                pulse_y = pulse_y[valid_mask]
                
                frame[pulse_y, pulse_x] = 1

            # Background noise
            num_bg_noise = np.random.randint(10, 40)
            if num_bg_noise > 0:
                noise_x = np.random.randint(0, frame_width, num_bg_noise)
                noise_y = np.random.randint(0, frame_height, num_bg_noise)
                frame[noise_y, noise_x] = np.random.choice([1, 2], num_bg_noise)

            # Pack and write
            flat_pixels = frame.flatten()
            packed_data = bytearray()
            
            for j in range(0, len(flat_pixels), 4):
                p1 = flat_pixels[j] if j < len(flat_pixels) else 0
                p2 = flat_pixels[j+1] if j+1 < len(flat_pixels) else 0
                p3 = flat_pixels[j+2] if j+2 < len(flat_pixels) else 0
                p4 = flat_pixels[j+3] if j+3 < len(flat_pixels) else 0
                
                byte = (p1 << 6) | (p2 << 4) | (p3 << 2) | p4
                packed_data.append(byte)
            
            f.write(packed_data)
    
    print(f"Laser pattern file created with {num_frames} frames.")

def create_empty_bin_directory():
    """Create bin directory if it doesn't exist"""
    bin_dir = "bin"
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir, exist_ok=True)
        print(f"Created bin directory: {bin_dir}")
    else:
        print(f"Bin directory already exists: {bin_dir}")
    return bin_dir

if __name__ == "__main__":
    # Example usage
    create_empty_bin_directory()
    
    # Create test files
    create_test_bin_file("bin/sample_test.bin", 300, 200, 5)
    create_laser_pattern_bin_file("bin/sample_laser.bin", 400, 300, 8)
    
    print("Sample bin files created in bin/ directory")

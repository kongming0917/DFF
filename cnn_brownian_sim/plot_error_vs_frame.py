#!/usr/bin/env python3
"""Run inference and plot pixel error vs center frame index (where large errors occur)."""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
dvs_root = os.path.dirname(script_dir)
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from inference import find_available_models, select_model, DVSInference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-frames', type=int, default=2000, help='Max frames to load')
    args = parser.parse_args()

    selected = select_model(script_dir)
    if not selected:
        return
    if selected['roi_str'] == '720x960':
        bin_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_720x960.bin"
        csv_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_720x960_labels.csv"
        roi_load, roi_dataset = (960, 720), (720, 960)
    else:
        bin_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512.bin"
        csv_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512_labels.csv"
        roi_load = roi_dataset = (512, 512)

    inferencer = DVSInference(
        selected['path'],
        use_quantized=selected['use_qat'],
        model_name=selected.get('model_name'),
    )
    if not os.path.exists(bin_path) or not os.path.exists(csv_path):
        print("Data not found.")
        return

    individual_frames = inferencer.load_frames_from_bin(bin_path, max_frames=args.num_frames, roi_size=roi_load)
    results = inferencer.predict_from_frames(individual_frames, csv_path, roi_size=roi_dataset)
    pixel_errors = np.array(results.get('pixel_errors', []))
    if len(pixel_errors) == 0:
        print("No results.")
        return

    tw = inferencer.input_channels
    center_frame_idx = np.arange(len(pixel_errors)) + (tw // 2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax1, ax2 = axes

    ax1.scatter(center_frame_idx, pixel_errors, s=2, alpha=0.5)
    ax1.set_xlabel('Center frame index')
    ax1.set_ylabel('Pixel error')
    ax1.set_title('Pixel error vs center frame index')
    ax1.grid(True, alpha=0.3)

    thresh = np.percentile(pixel_errors, 90)
    high_mask = pixel_errors >= thresh
    ax2.hist(center_frame_idx[high_mask], bins=50, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Center frame index')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Frame index distribution (samples with error >= 90th percentile, {high_mask.sum()} samples)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(script_dir, selected['dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{selected['name']}_error_vs_frame.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run inference and save the frame where max pixel error occurs (GT vs Pred overlay)."""
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

from inference import (
    find_available_models,
    select_model,
    DVSInference,
)
from dataset import DVSBrownianDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-frames', type=int, default=200, help='Max frames to load')
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
    if not os.path.exists(bin_path):
        print(f"Bin not found: {bin_path}")
        return
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    individual_frames = inferencer.load_frames_from_bin(bin_path, max_frames=args.num_frames, roi_size=roi_load)
    results = inferencer.predict_from_frames(individual_frames, csv_path, roi_size=roi_dataset)
    pixel_errors = results.get('pixel_errors', [])
    if not pixel_errors:
        print("No pixel errors.")
        return

    idx_max = int(np.argmax(pixel_errors))
    max_err = results['max_pixel_error']
    dataset = DVSBrownianDataset(
        individual_frames=individual_frames,
        csv_labels_path=csv_path,
        roi_size=roi_dataset,
        temporal_window=inferencer.input_channels,
    )
    dataset.set_training_mode(False)
    sample_input, _ = dataset[idx_max]
    center_ch = inferencer.input_channels // 2
    center_frame = sample_input[center_ch].numpy()
    roi_h, roi_w = roi_dataset[0], roi_dataset[1]
    true_x, true_y = results['targets'][idx_max]
    pred_x, pred_y = results['predictions'][idx_max]
    gt_px = (true_x * roi_w, true_y * roi_h)
    pred_px = (pred_x * roi_w, pred_y * roi_h)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(center_frame, cmap='gray')
    ax.scatter([gt_px[0]], [gt_px[1]], c='lime', s=120, marker='+', linewidths=3, label='GT')
    ax.scatter([pred_px[0]], [pred_px[1]], c='red', s=120, marker='x', linewidths=3, label='Pred')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(f'Max error frame (sample idx={idx_max}, error={max_err:.2f} px)')
    ax.axis('off')
    out_path = os.path.join(script_dir, selected['dir'], f"{selected['name']}_max_error_frame.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# compare_training_curves_gui.py
"""두 auto-encoder 실험의 train / val MSE 곡선 비교 + config 표시"""

import argparse, json, textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────── helpers ────────────────────
def latest_json(exp_dir: Path) -> Path:
    js = sorted(exp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not js:
        raise FileNotFoundError(f"{exp_dir} 안에 *.json summary 가 없습니다")
    return js[-1]

def load_summary(json_path: Path, trial_idx: int = 0):
    with open(json_path) as fp:
        data = json.load(fp)

    conf = data["config"]
    trials = data["trials"]
    if trial_idx >= len(trials):
        raise IndexError(f"{json_path}: trial #{trial_idx} 없음")

    trial = trials[trial_idx]

    print(data['config'])
    print(trial['best_mse'])
    train_curve = np.asarray(trial["loss_curve"], float)
    val_curve   = np.asarray(trial.get("val_mse_curve", []), float)
    if len(val_curve) == 0:                          # 호환성
        val_curve = np.zeros_like(train_curve)

    epochs = np.arange(1, len(train_curve) + 1)

    # val == 0 → 측정 안 된 지점 → np.nan 으로 바꿔서 플롯에서 건너뜀
    val_curve[val_curve == 0] = np.nan
    return epochs, train_curve, val_curve, conf

def cfg_short(cfg: dict) -> str:
    """plot 안에 넣기 좋은 한 덩어리 문자열 요약"""
    keys = ["model", "latent", "enc_width", "dec_width", "lr",
            "epochs", "batch", "scheduler"]
    picked = {k: cfg[k] for k in keys if k in cfg}
    txt = "\n".join(f"{k}: {picked[k]}" for k in picked)
    return textwrap.fill(txt, width=50)

# ──────────────────── main ────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compare train/val MSE curves of two experiments")
    ap.add_argument("exp_a", type=Path, help="실험 경로(폴더) 또는 summary.json #1")
    ap.add_argument("--exp_b", type=Path, help="실험 경로(폴더) 또는 summary.json #2", default=None)
    ap.add_argument("--trial-a", type=int, default=0,
                    help="trial index to plot (default 0)")
    ap.add_argument("--trial-b", type=int, default=None,
                    help="trial index to plot (default 0)")
    args = ap.parse_args()
    if args.exp_b is None:
        args.exp_b = args.exp_a
    if args.trial_b is None:
        args.trial_b = args.trial_a

    js_a = args.exp_a if args.exp_a.suffix == ".json" else latest_json(args.exp_a)
    js_b = args.exp_b if args.exp_b.suffix == ".json" else latest_json(args.exp_b)

    ep_a, tr_a, val_a, cfg_a = load_summary(js_a, args.trial_a)
    ep_b, tr_b, val_b, cfg_b = load_summary(js_b, args.trial_b)

    plt.figure(figsize=(10, 6))

    # ─ Train curves
    plt.plot(ep_a, tr_a, label=f"Train A  ({js_a.parent.name})", lw=1.8)
    plt.plot(ep_b, tr_b, label=f"Train B  ({js_b.parent.name})", lw=1.8)

    # ─ Val curves (점으로)
    plt.plot(ep_a, val_a, "o", ms=4, label="Val A")
    plt.plot(ep_b, val_b, "o", ms=4, label="Val B")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log-scale)")
    plt.title("Training / Validation MSE comparison")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()

    # ─ Config 텍스트 박스
    txt_a = cfg_short(cfg_a)
    txt_b = cfg_short(cfg_b)
    plt.gcf().text(0.02, 0.98, f"[A]\n{txt_a}", va="top",
                   bbox=dict(facecolor="#eef", alpha=0.7, boxstyle="round"))
    plt.gcf().text(0.75, 0.98, f"[B]\n{txt_b}", va="top",
                   bbox=dict(facecolor="#fee", alpha=0.7, boxstyle="round"))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

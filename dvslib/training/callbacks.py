"""Training callbacks — early stopping, checkpointing, metric history.

cnn_brownian_sim에서 정리해 가져왔다. 핵심 수정: 한 run은 자기 save_dir만 사용하고
(여러 run 산출물 혼입 방지 — 기존 baseline 붕괴 원인), 체크포인트에 epoch·metrics를
함께 저장해 자기기술적으로 만든다.
"""

import json
import os
from collections import defaultdict
from typing import Dict

import torch


class EarlyStopping:
    """Stop when the monitored value stops improving for `patience` epochs."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def __call__(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


class Checkpoint:
    """Save best (lower monitored metric) and last checkpoint into one run dir.

    payload에 epoch·metrics·monitor를 함께 저장 → 어떤 run의 어떤 성능인지 파일만 봐도 안다.
    """

    def __init__(self, save_dir: str, model_name: str, monitor: str = "pixel_error_mean", config: Dict = None):
        self.save_dir = save_dir
        self.model_name = model_name
        self.monitor = monitor
        self.config = config or {}  # 학습 config (model·temporal_window·roi 등) — inference가 자동 설정에 사용
        self.best = float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def step(self, model, optimizer, epoch: int, metrics: Dict[str, float]) -> bool:
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "monitor": self.monitor,
            "config": self.config,
        }
        torch.save(payload, os.path.join(self.save_dir, f"{self.model_name}_last.pth"))
        is_best = metrics[self.monitor] < self.best
        if is_best:
            self.best = metrics[self.monitor]
            torch.save(payload, os.path.join(self.save_dir, f"{self.model_name}_best.pth"))
        return is_best


class MetricsTracker:
    """Accumulate per-epoch train/val metrics and dump to JSON."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, epoch: int, train: Dict, val: Dict) -> None:
        for k, v in train.items():
            if isinstance(v, (int, float)):
                self.history[f"train_{k}"].append(v)
        for k, v in val.items():
            if isinstance(v, (int, float)):
                self.history[f"val_{k}"].append(v)
        self.history["epoch"].append(epoch)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dict(self.history), f, indent=2)

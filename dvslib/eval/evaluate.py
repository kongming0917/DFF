"""Run a regression model over a loader and report accuracy + throughput.

리팩토링 전후가 같은 코드로 평가되도록, baseline 재현 검증과 방식 비교 모두 여기서 한다.
"""

import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from dvslib.eval.metrics import pixel_error_metrics


@torch.no_grad()
def evaluate_regression(
    model: torch.nn.Module,
    loader,
    roi_size: Tuple[int, int] = (512, 512),
    device: str = "cuda",
    set_mode=None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate a (x, y) regression model: pixel-error metrics + mean latency/FPS.

    모델 출력은 정규화 (x, y) 가정. 반환에 accuracy_*px / pixel_error_* / fps 포함.
    set_mode: eval 모드 전환 방법 주입 (기본=표준 .eval()). PT2E exported 모델은
    quantization.set_qat_mode를 넘긴다.
    return_predictions: True면 샘플별 결과도 반환 — "predictions"/"targets"(정규화 (N,2)),
    "pixel_errors"(px, (N,)). 시각화(dvslib.eval.visualize)·방식 간 비교용.
    """
    _set_mode = set_mode or (lambda m, training: m.train(training))
    model.to(device)
    _set_mode(model, False)

    # warmup: 첫 배치의 cudnn autotune / CUDA lazy init을 타이밍에서 제외 (throughput 정확도)
    first = next(iter(loader), None)
    if first is not None:
        wx = first[0].to(device)
        for _ in range(2):
            model(wx)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    preds, targets, times = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / inputs.size(0))  # per-sample
        preds.append(out.cpu().numpy())
        targets.append(labels.numpy())

    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(targets, axis=0)

    metrics = pixel_error_metrics(pred, tgt, roi_size=roi_size)
    mean_time = float(np.mean(times))
    metrics["mean_time_ms"] = mean_time * 1000.0
    metrics["fps"] = (1.0 / mean_time) if mean_time > 0 else 0.0
    metrics["num_samples"] = int(len(pred))
    if return_predictions:
        roi_h, roi_w = roi_size
        scale = np.array([roi_w, roi_h], dtype=np.float64)
        metrics["predictions"] = pred
        metrics["targets"] = tgt
        metrics["pixel_errors"] = np.sqrt(np.sum(((pred - tgt) * scale) ** 2, axis=1))
    return metrics

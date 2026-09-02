"""Regression training loop for (x, y) center prediction.

Recipe는 cnn_brownian_sim과 동일: MSELoss + Adam + ReduceLROnPlateau + EarlyStopping.
best checkpoint는 보고 지표인 pixel_error_mean 기준으로 저장한다 (val_loss가 아니라).
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from dvslib.eval.metrics import pixel_error_metrics
from dvslib.training.callbacks import Checkpoint, EarlyStopping, MetricsTracker


class RegressionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        *,
        model_name: str,
        roi_size: Tuple[int, int] = (512, 512),
        lr: float = 1e-3,
        num_epochs: int = 50,
        patience: int = 15,
        save_dir: str = "checkpoints",
        device: str = "cuda",
        logger=None,
        monitor: str = "loss",
        scheduler: str = "plateau",
        warmup_epochs: int = 0,
        grad_clip: Optional[float] = None,
        config: Optional[dict] = None,
        set_mode=None,
    ):
        self.model = model.to(device)
        # train/eval 전환 방법 주입 (기본=표준 .train()). PT2E exported 모델은 .train()이 막혀
        # 있어 quantization.set_qat_mode 같은 전용 함수를 넘긴다 (FP32는 기본값으로 무변화).
        self._set_mode = set_mode or (lambda m, training: m.train(training))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.roi_size = roi_size
        self.num_epochs = num_epochs
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.grad_clip = grad_clip

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # scheduler: "plateau"(현행, val_loss 기반) 또는 "cosine"(warmup → cosine decay, 안정화용)
        if scheduler == "cosine":
            sched = []
            if warmup_epochs > 0:
                sched.append(optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=warmup_epochs))
            sched.append(optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, num_epochs - warmup_epochs)))
            self.scheduler = (
                optim.lr_scheduler.SequentialLR(self.optimizer, sched, milestones=[warmup_epochs])
                if len(sched) > 1 else sched[0]
            )
            self._sched_needs_metric = False
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=8)
            self._sched_needs_metric = True

        self.early = EarlyStopping(patience=patience)
        # loss·scheduler·early-stop·best 모두 동일 지표(monitor, 기본 val_loss)로 통일
        self.ckpt = Checkpoint(save_dir, model_name, monitor=monitor, config=config)
        self.metrics = MetricsTracker()

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        self._set_mode(self.model, train)
        total_loss, n = 0.0, 0
        preds, targets = [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                if train:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
                preds.append(out.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        m = pixel_error_metrics(preds, targets, self.roi_size)  # mae 포함
        m["loss"] = total_loss / n
        return m

    def fit(self) -> Dict:
        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_m = self._run_epoch(self.train_loader, train=True)
            val_m = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_m["loss"]) if self._sched_needs_metric else self.scheduler.step()
            self.metrics.update(epoch, train_m, val_m)
            is_best = self.ckpt.step(self.model, self.optimizer, epoch, val_m)

            if self.logger:
                self.logger.log(
                    {
                        **{f"train/{k}": v for k, v in train_m.items()},
                        **{f"val/{k}": v for k, v in val_m.items()},
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )
            print(
                f"[{epoch + 1}/{self.num_epochs}] "
                f"val pixel_err={val_m['pixel_error_mean']:.3f} "
                f"acc5={val_m['accuracy_5px']:.1f} loss={val_m['loss']:.6f} "
                f"{'*best' if is_best else ''} ({time.time() - t0:.1f}s)"
            )
            # early stopping은 적응형(plateau)에서만 — cosine은 고정 horizon이라 full schedule 실행
            if self._sched_needs_metric and self.early(val_m["loss"]):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.metrics.save(os.path.join(self.save_dir, f"{self.model_name}_metrics_history.json"))
        if self.logger:
            self.logger.summary({f"best/{self.ckpt.monitor}": self.ckpt.best})
        return self.metrics.history

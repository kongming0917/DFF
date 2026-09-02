#!/usr/bin/env python3
"""cnn experiment — thin training entrypoint on top of dvslib.

데이터·split·metric·학습 루프·wandb는 모두 dvslib에서 온다.
이 디렉토리는 모델(model.py, mobileone_official.py)과 학습 설정만 가진다.

  python cnn/train.py --model mobilenet_v2 --epochs 50 --wandb
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # dvslib
sys.path.insert(0, HERE)   # local model.py

from dvslib.data.dataset import load_frames_from_bin, parse_roi, brownian_paths  # noqa: E402
from dvslib.data.split import make_train_val_loaders          # noqa: E402
from dvslib.training.loop import RegressionTrainer            # noqa: E402
from dvslib.training.seed import seed_everything              # noqa: E402
from dvslib.tracking.wandb_logger import WandbLogger          # noqa: E402
from model import get_model                                   # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description="cnn experiment training")
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temporal-window", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--roi", default="512", help='ROI 크기: "512"(정사각) 또는 "720x960"(HxW)')
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="dvs-laser")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", default="cosine", choices=["plateau", "cosine"])
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    seed_everything(args.seed)  # 결정적 학습 (run 간 재현)

    roi = parse_roi(args.roi)                       # (H, W)
    bin_path, csv_path = brownian_paths(DATA, roi)
    # 새 run은 항상 자기 디렉토리에 저장 (run 산출물 혼입 방지)
    save_dir = args.save_dir or os.path.join(HERE, "runs", args.model)

    frames = load_frames_from_bin(bin_path, max_frames=args.max_frames, height=roi[0], width=roi[1])
    train_loader, val_loader = make_train_val_loaders(
        frames, csv_path,
        batch_size=args.batch_size, temporal_window=args.temporal_window, roi_size=roi,
    )

    # 학습은 항상 ImageNet pretrained 백본 사용 (mobilenet은 원래 default, mobileone도 적용).
    model = get_model(args.model, input_channels=args.temporal_window, output_dim=2, use_qat=False, pretrained=True)

    config = dict(
        model=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        temporal_window=args.temporal_window, max_frames=args.max_frames, roi=args.roi,
        seed=args.seed, monitor="loss",
        scheduler=args.scheduler, warmup_epochs=args.warmup_epochs, grad_clip=args.grad_clip,
        split="blocked(num_blocks=5, gap=50, val={1,3})",
    )
    logger = WandbLogger(
        project=args.wandb_project, name=f"cnn-{args.model}",
        config=config, tags=["cnn", args.model], enabled=args.wandb,
    )
    trainer = RegressionTrainer(
        model, train_loader, val_loader,
        model_name=args.model, roi_size=roi, lr=args.lr, num_epochs=args.epochs,
        patience=args.patience, save_dir=save_dir, device=args.device, logger=logger,
        scheduler=args.scheduler, warmup_epochs=args.warmup_epochs, grad_clip=args.grad_clip,
        config=config,
    )
    trainer.fit()
    logger.finish()
    print(f"\nDone. best val {trainer.ckpt.monitor} = {trainer.ckpt.best:.6e}  ->  {save_dir}")


if __name__ == "__main__":
    main()

"""tracking layer — thin wandb wrapper so every method logs the same way.

Responsibilities:
    - run creation (config·tags), per-epoch metric logging, model artifact upload
    - keep baseline and post-refactor runs comparable in one project (dvs-laser)

사용: `from dvslib.tracking.wandb_logger import WandbLogger`.
"""

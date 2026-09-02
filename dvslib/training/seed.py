"""Reproducibility — 모든 RNG seed 고정 + cudnn 결정적 모드.

학습이 run마다 동일하게 재현되도록 한다 (신뢰 가능한 baseline의 전제).
seed_everything(seed)을 학습 스크립트 맨 앞에서 한 번 호출한다.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

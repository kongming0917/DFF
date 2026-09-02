"""data layer — every method loads the same data the same way.

Responsibilities:
    - 2-bit packed binary (.bin) I/O — single source (moved from lib/bin_processor.py)
    - DVS Dataset (temporal window handling)
    - train/val split: blocked / K-fold — defined here ONCE

공정 비교의 핵심. split 경계가 방식마다 달라지면 비교가 무효가 되므로 split은 여기 한 곳에만 둔다.

bin_processor만 numpy 의존이라 여기서 re-export한다. dataset/split은 torch 의존이므로
필요할 때 submodule(`dvslib.data.dataset`, `dvslib.data.split`)에서 직접 import한다.
"""

from dvslib.data.bin_processor import (  # noqa: F401
    BinProcessor,
    DVSFrame,
    FrameHeader,
    ProcessingStats,
)

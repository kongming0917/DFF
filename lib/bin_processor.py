"""Backward-compat shim.

실제 구현은 dvslib/data/bin_processor.py로 이전됨 (단일 소스).
아직 dvslib으로 마이그레이션하지 않은 디렉토리(yolo/filter brownian)가
기존 `from lib.bin_processor import ...`로 계속 동작하도록 유지한다.
마이그레이션 완료 후 lib/ 전체를 제거할 예정.
"""

from dvslib.data.bin_processor import (  # noqa: F401
    BinProcessor,
    DVSFrame,
    FrameHeader,
    ProcessingStats,
)

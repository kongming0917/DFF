"""training layer — write the training boilerplate once.

Responsibilities:
    - train loop (FP32 / QAT)
    - EarlyStopping, Checkpoint, MetricsTracker

연구 질문과 무관한 학습 배관이다. 방식별 구조가 다르면(예: filter는 학습 없음)
억지로 통일하지 않고, 진짜로 동일한 부분만 공유한다.

heavy(torch) 의존이라 submodule에서 import: `dvslib.training.loop`, `dvslib.training.callbacks`.
"""

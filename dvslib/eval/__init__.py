"""eval layer — metrics computed in one place so they mean the same across methods.

Responsibilities:
    - accuracy: mean pixel error, RMSE, Acc@Npx (N = 3/5/10)
    - throughput: inference FPS / latency
    - cross-method comparison tables (wandb, no PNG dependency)

"2.11px"가 CNN·YOLO·Filter에서 모두 같은 의미가 되도록 metric 계산은 여기서만 한다.

metrics는 numpy 전용이라 re-export. evaluate_regression은 torch 의존이므로
`dvslib.eval.evaluate`에서 직접 import한다.
"""

from dvslib.eval.metrics import pixel_error_metrics  # noqa: F401

"""dvslib — shared library for DVS laser-center detection.

필터 / CNN / YOLO (그리고 향후 EventTransformer)가 공유하는 코드를 한 곳에 모아,
모든 방식이 동일한 data·split·metric으로 평가되도록 보장한다.

Subpackages:
    data      bin I/O, Dataset, train/val split (avoid temporal leakage)
    models    shared model components, model registry
    training  train loop, EarlyStopping, Checkpoint, MetricsTracker
    tracking  wandb logging wrapper
    eval      metrics (pixel error, Acc@Npx, FPS), visualize, cross-method comparison
    quant     PT2E QAT (prepare/convert/set_qat_mode, INT8 save/load) — FPGA INT8 one path
"""

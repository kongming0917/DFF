# cnn

DVS 센서 데이터에서 레이저 빔 중심 좌표 (x, y)를 직접 예측하는 CNN 회귀 experiment.
`dvslib` 위에 올라간 얇은 experiment로, 이 디렉토리는 모델(`model.py`, `mobileone_official.py`)과
진입점(`train.py`, `inference.py`, `train_qat.py`)만 가진다. 학습 설정은 `train.py`의 argparse 기본값이다. 데이터·split·metric·학습 루프·wandb는 모두 `dvslib`에서 온다.

지원 모델: MobileNetV2(baseline), MobileOneS0(QAT → INT8, FPGA 배포 목표).

## Usage

```bash
python cnn/train.py --model mobilenet_v2 --epochs 50 --wandb
python cnn/inference.py                       # baseline best.pth 평가
```

## Baseline

MobileNetV2, blocked split, seeded cosine recipe(warmup+cosine+grad_clip). 5-seed sweep
**3.07 ± 0.31 px / Acc@5px 85.7 ± 1.3 %** (canonical seed 42 = 2.79px / 87.5%). 재현: `python train.py --seed 42`.
상세는 루트 [BASELINE.md](../BASELINE.md).

QAT(FPGA INT8)는 PT2E 기반 `quantization.py` + `train_qat.py`, 분석 도구는 `model_summary.py` / `export_mobileone_info.py`.

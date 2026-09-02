# yolo

YOLOv3-Tiny로 레이저 스팟을 검출하고 bbox 중심을 좌표로 쓰는 experiment. `dvslib` 기반이며
데이터셋·split·지표·학습 루프는 CNN(`cnn/`)과 같다. 이 디렉토리는 `model.py`(모델 + loss + decode hook)와
진입점(`train.py`, `inference.py`)만 가진다.

```bash
python yolo/train.py --epochs 50 --seed 42
python yolo/inference.py
```

옛 `yolo_brownian_sim` 체크포인트도 `--checkpoint`로 로드된다. baseline 수치는 루트 [BASELINE.md](../BASELINE.md).

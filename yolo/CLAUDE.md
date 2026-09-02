# CLAUDE.md — yolo experiment

레이저 스팟을 단일 객체로 검출(YOLOv3-Tiny)한 뒤 bbox 중심을 레이저 중심으로 쓰는 experiment.
**dvslib 위에 올라간 얇은 experiment**로, 데이터셋·split·metric·학습 루프·wandb는 CNN과 완전히 같다.
이 디렉토리는 모델과 dvslib 루프용 hook 두 개만 가진다.

## Structure

```
yolo/
├── train.py        # 학습 진입점 (dvslib.training.RegressionTrainer + criterion/to_xy hook)
├── inference.py    # 체크포인트 평가 (dvslib.eval.evaluate_regression + to_xy hook)
├── model.py        # YOLOv3Tiny, decode/NMS/center, YOLOLoss, YOLOCriterion, YOLOCenterDecoder
├── runs/           # 학습 산출물 — gitignore
└── checkpoints_yolo_tiny_laser_brownian/   # pre-dvslib 옛 체크포인트 보존분 (epoch 5) — gitignore
```

## How it fits dvslib

- **데이터셋은 CNN과 동일** (`DVSBrownianDataset`, 타깃 = 정규화 중심 (x, y)). bbox 타깃은 `YOLOCriterion`이
  중심 + 고정 크기(`LASER_DIAMETER`=400px)로 만든다. YOLO 전용 dataset 클래스를 만들지 말 것.
- **입력 스케일**은 모델 안(`input_scale=0.5`, raw 0/1/2 → 0/0.5/1). 옛 코드의 `/max` 정규화와 동일 결과.
  데이터 파이프라인에서 정규화하지 말 것 (CNN은 raw 그대로 씀).
- **`YOLOCenterDecoder`**(to_xy): decode → NMS → ROI 중심 우선 선택. 검출 실패 시 직전 성공 좌표 유지
  (초기 (0.5, 0.5)). 상태를 가지므로 루프가 epoch/평가 시작 시 `reset()`을 호출한다. FPS에 decode 포함.
- 지표(pixel error·Acc@Npx)는 decoder 출력 좌표로 dvslib이 계산 → CNN과 같은 의미.

## Usage

```bash
python yolo/train.py --epochs 50 --seed 42          # CNN과 동일 recipe (warmup+cosine, grad_clip, blocked split)
python yolo/inference.py                             # runs/baseline_yolo_tiny/yolo_tiny_best.pth 평가
python yolo/inference.py --checkpoint <path> [--conf-threshold 0.3]
python tools/evaluate.py --yolo-checkpoint <path>    # 방식 공용 평가/시각화 도구도 사용 가능
```

## Baseline

옛 체크포인트(500프레임·5 epoch·random split)는 blocked val에서 **93.69px / Acc@5px 1.37%** — 학습 구간 밖
일반화 실패. 이식 검증: 새 `inference.py`로 같은 값 정확 재현. 재학습 baseline은 루트 `BASELINE.md`.

## Key Rules

- 학습 run은 fresh 디렉토리(`runs/<name>`)에. 한 dir에 여러 run을 섞지 말 것.
- 재현성: `--seed`(기본 42) 결정적. 기본 레시피 cosine.
- 손실은 `YOLOLoss`(CIoU + objectness BCE)이며 batch를 python loop로 돈다 — 느리면 vectorize 대상.
- conf_threshold는 지표에만 영향(검출률 vs 오차). 체크포인트 config에 저장되어 inference가 자동으로 읽는다.

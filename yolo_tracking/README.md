# 🎯 YOLO-based DVS Laser Tracking

YOLOv3-Tiny 기반 레이저 스팟 추적

## 📋 개요

`cnn_tracking`과 동일한 개념이지만 YOLO 방식:
- **ROI 고정**: 카메라 시야 고정
- **Bounding Box**: 물체 위치 + 크기 예측
- **YOLO Loss**: Objectness + Localization

## 🚀 빠른 시작

```bash
cd /hai/home/jdj/dvs/yolo_tracking
python train.py
# 선택: 1 (Quick test)
```

## 📊 CNN vs YOLO Tracking

| | **CNN Tracking** | **YOLO Tracking** |
|---|---|---|
| **출력** | (x, y) 좌표 | (x, y, w, h) bbox |
| **손실** | MSE | YOLO Loss |
| **장점** | 간단, 빠름 | 크기 정보, Confidence |
| **단점** | 크기 모름 | 복잡, 느림 |

## ⚙️ 설정

```python
from config import YOLOTrackingExperimentConfig

config = YOLOTrackingExperimentConfig()
config.data.roi_center = (541, 360)
config.data.laser_diameter = 400
config.training.lambda_coord = 5.0
```

## 📈 학습

```python
from train import train_yolo_tracking
from config import get_standard_config

config = get_standard_config()
train_yolo_tracking(config)
```

결과:
```
checkpoints_yolo_tracking/
└── yolo_tracking_standard_best.pth
```

## 🔍 평가

YOLO는 추가로:
- **Objectness**: 물체 존재 확률
- **IOU**: Bbox 겹침 정도
- **Detection Rate**: 검출 성공률

## 💡 사용 시나리오

### CNN Tracking 추천:
- 빠른 속도 필요
- 위치만 중요
- 실시간 추적

### YOLO Tracking 추천:
- 물체 크기 중요
- Confidence 필요
- 복잡한 환경

## 📚 참고

- `cnn_tracking/`: CNN 기반 tracking
- `yolo_sim/`: YOLO detection (기존)
- YOLO 논문: YOLOv3 architecture

## 🎓 주의사항

1. **메모리**: YOLO가 더 많이 사용
2. **속도**: CNN이 더 빠름
3. **정확도**: 비슷하거나 YOLO가 약간 우세
4. **복잡도**: YOLO가 더 복잡

대부분의 경우 `cnn_tracking`으로 충분합니다!


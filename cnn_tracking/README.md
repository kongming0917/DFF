# 🎯 DVS Laser Tracking

레이저 스팟의 실시간 추적을 위한 딥러닝 모델

## 📋 개요

### Detection vs Tracking

| | **Detection** (cnn_sim) | **Tracking** (cnn_tracking) |
|---|---|---|
| **목표** | 레이저 위치 검출 | 레이저 움직임 추적 |
| **ROI** | 랜덤하게 이동 | **고정** |
| **물체** | 항상 같은 위치 | **시간에 따라 움직임** |
| **학습** | 단일 프레임 예측 | 시간적 패턴 학습 |
| **레이블** | Shift에 따라 변화 | 실제 궤적 따라 변화 |

### 핵심 특징

✅ **ROI 고정**: 카메라 시야가 고정되어 있고, 물체가 움직이는 실제 시나리오  
✅ **Brownian Motion**: 물체의 움직임을 브라우니언 모션으로 시뮬레이션  
✅ **경계 제어**: 물체가 ROI 밖으로 나가지 않도록 경계 반사  
✅ **시간적 학습**: 여러 프레임(5-10개)을 보고 현재 위치 예측  
✅ **다양한 모델**: CNN, LSTM, Transformer 지원  

## 🚀 빠른 시작

### 1. 빠른 테스트

```bash
cd /hai/home/jdj/dvs/cnn_tracking
python train.py
# 선택: 1 (Quick test)
```

### 2. 표준 학습

```bash
python train.py
# 선택: 2 (Standard)
```

### 3. Python 스크립트

```python
from config import get_quick_test_config
from dataset import load_frames_from_bin
from train import TrackingTrainer

# 설정
config = get_quick_test_config()
config.data.motion_std = 2.5  # 더 빠른 움직임
config.data.num_temporal_frames = 7  # 더 긴 시퀀스

# 데이터 로딩
frames = load_frames_from_bin(
    "/hai/home/jdj/dvs/data/gaussian_large.bin",
    max_frames=200
)

# 훈련
trainer = TrackingTrainer(config)
trainer.setup_data(frames)
trainer.train()
```

## 📊 모델 종류

### 1. Basic Tracking (CNN)
```python
config.model.model_name = "basic_tracking"
config.data.num_temporal_frames = 5
```
- 가장 빠름
- 파라미터: ~500K
- 적합: 단순한 움직임

### 2. LSTM Tracking
```python
config.model.model_name = "lstm_tracking"
config.model.use_lstm = True
config.model.lstm_hidden_size = 128
config.model.lstm_num_layers = 2
config.data.num_temporal_frames = 10
```
- 시간적 의존성 학습
- 파라미터: ~1M
- 적합: 복잡한 궤적

### 3. Transformer Tracking
```python
config.model.model_name = "transformer_tracking"
config.data.num_temporal_frames = 8
```
- Self-attention 메커니즘
- 파라미터: ~1.5M
- 적합: 장기 의존성

## ⚙️ 주요 설정

### 데이터 설정

```python
config.data.roi_center = (480, 294)        # ROI 중심 (고정)
config.data.roi_size = (384, 384)          # ROI 크기
config.data.motion_std = 2.0               # 움직임 속도
config.data.motion_boundary_margin = 80    # ROI 경계 여유
config.data.use_boundary_reflection = True # 경계 반사
config.data.num_temporal_frames = 5        # 시간 축 프레임 수
```

### 움직임 속도 조절

| motion_std | 움직임 속도 | 설명 |
|-----------|---------|------|
| 0.5 | 매우 느림 | 거의 정지 |
| 1.0 | 느림 | 천천히 이동 |
| 2.0 | 보통 | **추천** |
| 5.0 | 빠름 | 빠른 움직임 |
| 10.0 | 매우 빠름 | 추적 어려움 |

## 📈 학습 결과

학습 후 다음 파일들이 생성됩니다:

```
checkpoints_tracking/
├── experiment_name_best.pth          # 최고 모델
├── experiment_name_epoch_10.pth      # 주기적 저장
└── ...

outputs_tracking/
└── experiment_name_training_curves.png  # 학습 곡선

logs_tracking/
└── training.log                         # 로그
```

## 🔍 평가 지표

- **Pixel Error**: 예측 위치와 실제 위치의 픽셀 거리
- **Acc@5px**: 5픽셀 이내 정확도
- **Acc@10px**: 10픽셀 이내 정확도

목표:
- Pixel Error < 5px
- Acc@5px > 90%
- Acc@10px > 95%

## 🛠️ 고급 사용법

### 1. 커스텀 설정

```python
from config import TrackingExperimentConfig

config = TrackingExperimentConfig()
config.experiment_name = "my_tracking"

# 데이터
config.data.motion_std = 3.0
config.data.num_temporal_frames = 8
config.data.motion_boundary_margin = 100

# 모델
config.model.model_name = "lstm_tracking"
config.model.use_lstm = True

# 훈련
config.training.num_epochs = 150
config.training.batch_size = 8
config.training.learning_rate = 0.0005
```

### 2. 설정 저장/로드

```python
# 저장
config.save_config("my_config.json")

# 로드
from config import TrackingExperimentConfig
config = TrackingExperimentConfig.load_config("my_config.json")
```

### 3. 체크포인트에서 재개

```python
checkpoint = torch.load("checkpoints_tracking/experiment_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 🐛 문제 해결

### Q1: ROI 밖으로 물체가 나갑니다
```python
# 경계 여유를 늘림
config.data.motion_boundary_margin = 120

# 또는 움직임 속도를 줄임
config.data.motion_std = 1.0
```

### Q2: 학습이 수렴하지 않습니다
```python
# Learning rate를 낮춤
config.training.learning_rate = 0.0001

# Batch size를 늘림
config.training.batch_size = 32

# 더 간단한 모델 사용
config.model.model_name = "basic_tracking"
```

### Q3: GPU 메모리 부족
```python
# Batch size 줄임
config.training.batch_size = 4

# Temporal frames 줄임
config.data.num_temporal_frames = 3

# 더 작은 ROI
config.data.roi_size = (256, 256)
```

## 📚 비교: Detection vs Tracking

### cnn_sim (Detection)
```python
# ROI가 이동, 물체는 고정
for frame in frames:
    shift = random_shift()
    roi = extract_roi(frame, center + shift)
    label = 0.5, 0.5  # 항상 중심
```

### cnn_tracking (Tracking)
```python
# ROI 고정, 물체가 이동
roi_center = fixed_center
for t in trajectory:
    roi = extract_roi(frame, roi_center)  # 고정!
    label = object_position[t]  # 시간에 따라 변화
```

## 🎓 추가 개선 아이디어

1. **Adaptive ROI**: 물체가 경계 근처면 ROI를 천천히 이동
2. **Multi-step Prediction**: 다음 위치뿐만 아니라 다음 N개 위치 예측
3. **Velocity Prediction**: 위치 + 속도를 함께 예측
4. **Attention Visualization**: Transformer의 attention map 시각화
5. **Uncertainty Estimation**: 예측의 신뢰도 추정

## 📞 문의

문제가 있으면 issue를 남겨주세요!


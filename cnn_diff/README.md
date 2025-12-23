# LogicDVSNet: DVS 레이저 중심점 탐지를 위한 Differentiable Logic Network

졸업연구용 DVS 레이저 중심점 탐지 프로젝트를 위한 독립적인 학습 환경입니다.

## 📁 폴더 구조

```
cnn_diff/
├── model.py          # LogicDVSNet 모델 정의 (CIFAR-10 구조 기반, 회귀 헤드)
├── train.py          # 훈련 스크립트 (tau 스케줄링 및 정규화 포함)
├── inference.py      # 추론 스크립트
├── dataset.py        # DVS 데이터셋 로더
├── utils.py          # 유틸리티 함수들 (EarlyStopping, MetricsTracker 등)
└── README.md         # 이 파일
```

## 🏗️ 모델 아키텍처

### LogicDVSNet

- **Backbone**: CIFAR-10용 4-stage FusedLogicTreeBlock 구조
  - Stage 1: Input → k
  - Stage 2: k → 4k
  - Stage 3: 4k → 16k
  - Stage 4: 16k → 32k

- **Head**: 회귀 헤드 (RegressionLayer 또는 MultiOutputRegressionLayer)
  - 입력: DVS 이벤트 프레임
  - 출력: (x, y) 좌표 (정규화된 값, 0-1 범위)

## 🚀 사용법

### 1. 환경 설정

```bash
# birel 패키지 설치 (필요한 경우)
cd /hai/home/jdj/dvs/sim/birel
pip install -e .
```

### 2. 데이터 준비

- DVS bin 파일과 CSV 레이블 파일이 필요합니다.
- CSV 파일 형식: `frame_idx, cnn_rel_x, cnn_rel_y`

### 3. 훈련

```python
from train import train_model
from dataset import create_train_val_loaders, load_individual_frames_from_bin

# 데이터 로드
bin_file_path = "/path/to/data.bin"
csv_labels_path = "/path/to/labels.csv"
individual_frames = load_individual_frames_from_bin(bin_file_path)

# 데이터로더 생성
train_loader, val_loader = create_train_val_loaders(
    individual_frames=individual_frames,
    csv_labels_path=csv_labels_path,
    batch_size=32,
    temporal_window=1,
    roi_size=(128, 128)
)

# 훈련 실행
trainer = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    input_channels=1,
    num_neurons=64,      # 기본 뉴런 수 (k)
    output_dim=2,        # (x, y) 좌표
    lr=0.01,
    tau_start=1.0,       # 초기 temperature
    tau_end=0.1,         # 최종 temperature
    num_epochs=100,
    save_dir='checkpoints'
)
```

### 4. 추론

```python
from inference import run_inference

# 추론 실행
predictions, targets = run_inference(
    checkpoint_path="checkpoints/logic_dvs_best.pth",
    bin_file_path="/path/to/data.bin",
    csv_labels_path="/path/to/labels.csv",
    input_channels=1,
    num_neurons=64,
    output_dim=2,
    save_dir='inference_results'
)
```

## ⚙️ 주요 기능

### Tau 스케줄링

difflogic 모델 학습에 필수적인 temperature 파라미터 스케줄링:
- 초기: `tau_start` (기본값: 1.0) - 부드러운 학습
- 최종: `tau_end` (기본값: 0.1) - hard logic에 가까운 학습
- 스케줄: 지수 감쇠 (`tau = tau_start * (tau_end/tau_start)^(epoch/num_epochs)`)

### 정규화

- **Weight Decay**: AdamW 옵티마이저에 `weight_decay=0.002` 적용
- **Learning Rate Scheduling**: ReduceLROnPlateau 사용

### 손실 함수

- **MSE Loss**: 좌표 회귀를 위한 Mean Squared Error 사용

## 📊 출력 파일

훈련 후 생성되는 파일들:

- `checkpoints/logic_dvs_best.pth`: 최고 성능 모델
- `checkpoints/logic_dvs_epoch_*.pth`: 각 에폭별 체크포인트
- `checkpoints/metrics_history.json`: 훈련 메트릭 히스토리
- `result/logic_dvs_training_curves.png`: 훈련 곡선 그래프
- `result/logic_dvs_predictions.png`: 예측 결과 시각화

## 🔧 하이퍼파라미터

주요 하이퍼파라미터:

- `num_neurons` (k): 기본 뉴런 수 (기본값: 64)
  - 모델 크기 조절: 작을수록 경량, 클수록 정확도 향상 가능
- `tau_start`: 초기 temperature (기본값: 1.0)
- `tau_end`: 최종 temperature (기본값: 0.1)
- `lr`: 학습률 (기본값: 0.01)
- `tree_depth`: Logic tree 깊이 (기본값: 3)

## 📝 참고사항

- **입력 전처리**: DVS 데이터는 이진화 전처리가 필요할 수 있습니다 (0/1 입력 선호)
- **출력 범위**: 모델 출력은 정규화된 좌표 (0-1 범위)입니다
- **데이터 형식**: CSV 레이블 파일은 `cnn_rel_x`, `cnn_rel_y` 컬럼을 포함해야 합니다

## 🐛 문제 해결

### Import 오류

```python
# birel 패키지 경로가 올바르게 설정되었는지 확인
import sys
sys.path.insert(0, '/hai/home/jdj/dvs/sim/birel')
```

### CUDA 메모리 부족

- `batch_size`를 줄이거나
- `num_neurons`를 줄여 모델 크기를 축소

## 📚 참고 자료

- `birel/experiments/conv_difflogic.py`: 원본 CIFAR-10 모델 구조
- `cnn_brownian_sim/`: 유사한 프로젝트 구조 참고


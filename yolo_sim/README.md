# YOLOv3-Tiny 레이저 중심점 검출 - 객체 감지 방식

DVS(Dynamic Vision Sensor) 카메라 데이터에서 레이저 스팟을 객체로 간주하여 Bounding Box 감지 후 중심점을 추출합니다.

## 🎯 개요

이 프로젝트는 **YOLO 기반 객체 감지 방식**으로 DVS 센서 데이터에서 레이저 빔을 감지하고 중심점을 추출합니다.
레이저 스팟을 하나의 객체로 간주하여 YOLOv3-Tiny 아키텍처로 Bounding Box를 예측한 후 중심을 계산합니다.

**장점**:
- ✅ 레이저 크기(w, h) 정보 추가 획득
- ✅ 여러 레이저 동시 감지 가능 (확장성)
- ✅ 신뢰도(confidence) 점수 제공
- ✅ 객체 감지 프레임워크 활용

**한계**:
- ⚠️ CNN 회귀보다 복잡한 구조
- ⚠️ 단일 레이저 감지에는 과도한 복잡도
- ⚠️ FPGA 구현이 가장 어려움
- ⚠️ 학습 시간이 가장 오래 걸림

이 방식은 전체 DVS 프로젝트의 **Phase 3: 객체 감지 확장** 단계에 해당합니다.

---

# YOLOv3-Tiny 레이저 중심점 검출

Object Detection 방식으로 레이저 스팟을 감지하고 중심점을 추출합니다.

## 📁 구조

```
yolo_detection/
├── model.py          # YOLOv3-Tiny 모델
├── dataset.py        # YOLO용 데이터셋 (bbox 형식)
├── train.py          # 학습 스크립트
├── inference.py      # 추론 및 시각화
├── utils.py          # 유틸리티 함수
└── README.md         # 이 파일
```

## 🚀 사용법

### 학습
```bash
cd /hai/home/jdj/dvs/yolo_detection
python train.py
```

### 추론
```bash
python inference.py
```

### 모델/데이터셋 테스트
```bash
python model.py      # 모델 구조 확인
python dataset.py    # 데이터셋 확인
python utils.py      # 유틸리티 테스트
```

## 🎯 특징

- **YOLOv3-Tiny 기반**: 경량화된 구조
- **단일 객체 감지**: 레이저 스팟 하나만 감지
- **Bbox → Center**: Bounding box 중심점 추출
- **완전 독립**: cnn_sim과 독립적으로 동작

## 📊 vs CNN Regression

| 방식 | 접근법 | 출력 |
|------|--------|------|
| CNN Regression | 좌표 직접 예측 | (x, y) |
| YOLO Detection | 물체 감지 후 중심 추출 | (x, y, w, h, conf) → (x, y) |

## ⚙️ 주요 파라미터

- `max_frames`: 500 (학습 프레임 수)
- `num_epochs`: 30
- `batch_size`: 4
- `roi_size`: 512×512
- `shift_range`: ±50px
- `laser_diameter`: 400px

---

## 🔗 관련 프로젝트

이 YOLO 방식은 전체 DVS 레이저 중심점 감지 프로젝트의 일부입니다:

- **상위 프로젝트**: [dvs/README.md](../README.md) - 전체 연구 개요 및 방법론 비교
- **Filter 방식**: [filter_sim/README.md](../filter_sim/README.md) - 휴리스틱 기반 접근
- **CNN 방식**: [cnn_sim/README.md](../cnn_sim/README.md) - 딥러닝 회귀 접근

### 성능 비교

| 방법 | 평균 오차 | Acc@5px | 처리 속도 | 크기 정보 | 다중 객체 |
|------|-----------|---------|-----------|-----------|-----------|
| Filter | 10-20px | 40-60% | ⚡⚡⚡⚡⚡ | ❌ | ❌ |
| CNN | 3-5px | 85-95% | ⚡⚡⚡ | ❌ | ❌ |
| **YOLO (이 프로젝트)** | **4-7px** | **80-90%** | **⚡⚡** | **✅** | **✅** |

**YOLO 방식의 특장점**: 레이저 크기 정보와 여러 객체 동시 감지 가능 (확장성)

자세한 비교는 [dvs/README.md](../README.md)를 참고하세요.

---

## 💡 언제 YOLO를 사용해야 하나?

### YOLO가 적합한 경우
- 🎯 **여러 레이저**를 동시에 추적해야 하는 경우
- 📏 **레이저 크기(직경)** 정보가 필요한 경우
- 📊 **신뢰도 점수**를 활용한 필터링이 필요한 경우
- 🔄 객체 감지 프레임워크를 **재사용**하고 싶은 경우

### CNN 회귀가 더 나은 경우
- ⚡ **빠른 추론 속도**가 중요한 경우
- 🎯 **단일 레이저만** 추적하는 경우
- 💻 **FPGA 구현**을 고려하는 경우
- 📐 레이저 크기 정보가 **불필요**한 경우

### Filter가 더 나은 경우
- ⚡⚡⚡ **초고속 실시간** 처리가 필수인 경우
- 🔧 **FPGA 직접 구현**이 목표인 경우
- 📚 **학습 데이터가 없는** 경우
- 🎓 **간단한 프로토타이핑**이 목적인 경우

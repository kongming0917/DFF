# DVS 레이저 빔 중심점 검출 시스템

DVS(Dynamic Vision Sensor) 카메라 데이터에서 레이저 빔의 중심 좌표 (x, y)를 실시간으로 검출하고, 최종적으로 FPGA에 배포하는 것을 목표로 하는 졸업 연구 프로젝트입니다. 동일한 문제를 필터 기반 휴리스틱, CNN 회귀, YOLO 검출 세 가지 방식으로 구현하고 정량적으로 비교합니다.

향후 연구 발전 계획은 [research_plan.md](research_plan.md)를 참고하세요.

## Project Structure

공통 로직(data·split·metric·training loop·wandb)을 **`dvslib` 패키지**로 모으고, 각 방식은 그 위의 **thin experiment**(모델·설정·entrypoint만)로 두는 구조로 리팩토링 중입니다. 이렇게 하면 방식 간 split·지표가 갈라지지 않아 비교가 공정합니다.

데이터는 두 종류입니다 — fixed GT(1세대, `archive/`로 이동)와 brownian motion(현행). 세 방식 모두 dvslib 이식이 끝났습니다.

```
dvs/sim/
├── dvslib/                  # 공통 패키지 — data·eval·training·tracking (thin experiment가 import)
├── cnn/                     # CNN 회귀 (주력) — dvslib 기반, MobileNetV2 / MobileOne+QAT
├── yolo/                    # YOLOv3-Tiny 검출 — dvslib 기반 (CNN과 동일 데이터셋·split·recipe)
├── filter/                  # 필터 휴리스틱 (학습 없음) — dvslib 기반, GT 원점 측정 도구 겸용
├── eventrans/               # EventTransformer (Phase 3, 예정)
├── tools/                   # 방식 공용 CLI — 데이터셋 생성·bin 검사·오차 시각화·3방식 비교
├── archive/                 # fixed-GT 1세대 — {filter,cnn,yolo}_sim
└── data/                    # DVS 센서 데이터 (.bin, 레이블 CSV)
```

`cnn`·`yolo`·`filter`는 데이터·split·지표를 `dvslib`에서 import하고 모델(또는 필터 알고리즘)과 진입점만 가집니다. 학습 루프와 평가 함수는 `criterion`·`to_xy` hook으로 출력 형식 차이(좌표 회귀 vs detection grid)를 흡수합니다. 리팩토링 전 원본(`cnn_brownian_sim`)은 baseline 재현을 검증한 뒤 삭제했습니다.

## Data Format

원시 데이터는 `data/`에 2-bit packed binary(`.bin`) 형식으로 저장됩니다.

- 프레임 구조: 8-byte 헤더(`<II`: timestamp + frame_number) + `(H*W)/4` 바이트의 패킹된 픽셀
- 픽셀 값: `0` = no event, `1` = ON event, `2` = OFF event (바이트당 2bit씩 4픽셀)
- 해상도: 원본 720×960, ROI crop 512×512. 학습·추론의 `--roi`는 `512`(정사각) 또는 `720x960`(HxW) 형식

bin 파일 입출력의 single source는 **`dvslib/data/bin_processor.py`**의 `BinProcessor` / `DVSFrame`입니다. 필터링이 필요한 모듈은 이를 상속한 `filter/dvs_filter.py`의 `FilterableBinProcessor`를 사용합니다. bin 헤더의 `frame_number`는 원본 센서 기록값(88부터)이라 GT의 `frame_idx`(0부터)와 다릅니다. 프레임 매칭은 항상 인덱스로 합니다.

### Brownian Motion Dataset

`*_brownian_*` 데이터셋은 학습 시점에 증강하지 않습니다. `tools/generate_brownian_dataset.py`가 사전에 `gaussian_large.bin`을 읽어 brownian motion shift를 적용한 `gaussian_brownian_512x512.bin`과 `..._labels.csv`를 생성하며, 데이터셋 클래스는 CSV의 ground truth(`frame_idx, shift_x, shift_y, cnn_rel_x, cnn_rel_y`)를 로드하기만 합니다. 이 생성 스크립트는 일반적으로 재실행할 필요가 없습니다.

## Approaches

### Filter (`filter`, dvslib 기반, 1세대 `archive/filter_sim`)

신호 처리 기법으로 중심점을 추출하는 초기 접근법입니다. `filter/run.py`가 필터 조건과 추출기 조합별로 프레임당 중심 좌표 CSV를 만들고, 평가는 `tools/evaluate.py --pred-csv`로 합니다. 이 방식은 비교 대상이면서 데이터셋 GT 원점 (541, 361)을 정할 때 쓴 측정 도구이기도 합니다 (`filter/origin.py`).

- Event Density Filter: 이벤트 밀도 기반 노이즈 제거
- Spatial Cluster Filter: KDTree 기반 공간 클러스터링으로 레이저 영역 추출
- Kalman Filter: 시간적 일관성을 위한 중심점 추적
- Median / Mean Point Extractor: 통계적 중심점 계산

학습이 필요 없고 연산량이 낮아 FPGA 구현에 적합하지만, 노이즈에 민감하고 수동 파라미터 튜닝이 필요합니다. brownian 데이터에서 SpatialClusterFilter는 효과가 없었고(±0.2px), Kalman이 평균 오차를 절반으로 줄이는 대신 지연으로 5px 정확도는 낮습니다 (`BASELINE.md`).

### CNN Regression (`cnn`, dvslib 기반)

좌표를 직접 예측하는 회귀 모델입니다. 실시간 필터링을 제거하고 ROI 기반 처리(960×720 → 512×512)로 메모리 사용량을 줄였습니다. 현행 코드는 `cnn`(주력, `cnn_brownian_v2`에서 rename)이며 데이터·split·metric·training loop는 `dvslib`에서 옵니다. fixed-GT 1세대는 `archive/cnn_sim`에 있습니다.

지원 모델은 MobileNetV2(baseline)와 MobileOneS0이며, MobileOneS0은 PT2E QAT를 거쳐 INT8로 양자화하여 FPGA 배포를 목표로 합니다 (`cnn/quantization.py`, `cnn/train_qat.py`).

### YOLO Detection (`yolo`, dvslib 기반, 1세대 `archive/yolo_sim`)

레이저 스팟을 객체로 보고 YOLOv3-Tiny로 단일 객체를 검출한 뒤 bounding box 중심을 레이저 중심으로 사용합니다. 데이터셋은 CNN과 같은 `DVSBrownianDataset`이고, bbox 타깃(중심 + 고정 직경 400px)은 손실 함수 안에서 만듭니다. 검출 실패 시 직전 성공 좌표를 유지합니다. 크기(w, h) 정보와 신뢰도 점수를 추가로 얻을 수 있고 다중 레이저로 확장 가능하지만, 단일 레이저 검출에는 구조가 과도하게 복잡합니다.

## Model Architecture

회귀 방식의 공통 사항입니다.

- 입력: `(batch, temporal_window, H, W)`. 연속 `temporal_window`개(기본 5) 프레임을 다채널로 쌓아 시간 정보를 인코딩합니다. `temporal_window=1`이면 단일 프레임.
- 출력: `(batch, 2)` = 정규화된 (x, y) ∈ [0, 1] (CNN 회귀는 Hardsigmoid로 클램프).
- 입력 정규화 없음: DVS 값이 이산값 {0, 1, 2}이고 MobileOne 첫 레이어의 BatchNorm이 스케일을 흡수하므로 별도 정규화를 적용하지 않습니다.

### Reparameterization & QAT

학습 시에는 multi-branch 구조를 사용하고, 추론 전 `model.reparameterize()`로 single-branch로 변환합니다. INT8 양자화는 PT2E(`torch.export` 기반) QAT로 수행하며 순서는 다음과 같습니다.

1. FP32 학습 완료 (`cnn/train.py`)
2. `reparameterize()` — multi-branch → single-branch
3. `quantization.prepare_qat()` — 그래프 캡처 후 Quantizer가 observer/fake-quant 자동 삽입 (Conv-BN fusion 포함)
4. QAT 파인튜닝 (`cnn/train_qat.py`, dvslib 학습 루프 재사용)
5. `quantization.convert()` — INT8 추론 그래프로 변환·저장

`mobileone_official.py`는 Apple 공식 구현이므로 수정하지 않습니다.

### Data Split

DVS 데이터는 시계열이므로 랜덤 split을 사용하면 시간적으로 인접한 프레임이 학습/검증에 동시에 포함되어 temporal leakage가 발생합니다. 이를 막기 위해 블록 단위로 나누는 blocked split(예: 블록 0,2,4 → 학습 / 1,3 → 검증, 블록 사이 갭 50프레임)과 K-fold 분할을 사용합니다.

## Setup

```bash
conda env create -f environment.yml
conda activate dvs_project        # python 3.11, pytorch>=2.0, cuda 11.8
```

## Usage

CNN(`cnn`)은 argparse-based, non-interactive입니다. 전체 명령 모음은 [`cnn/COMMANDS.md`](cnn/COMMANDS.md) 참고.

```bash
# CNN 학습 및 추론 (dvslib 기반)
python cnn/train.py --model mobilenet_v2 --epochs 50 --wandb   # 학습 + wandb 기록
python cnn/inference.py                                         # baseline checkpoint 평가
python cnn/model_summary.py --model mobileone_s0               # 구조·파라미터·메모리 분석

# YOLO (dvslib 기반, CNN과 동일 recipe)
python yolo/train.py --epochs 50 --seed 42
python yolo/inference.py

# Filter (dvslib 기반, 학습 없음)
python filter/run.py --max-frames 3000                                   # filter/results/*.csv
python tools/evaluate.py --pred-csv filter/results/no_filter_kalman.csv --split val --max-frames 3000
python filter/origin.py data/gaussian_large.bin --max-frames 200          # 정지 레이저 원점 후보 (GT initial_center 결정용)
```

방식 공용 분석 도구는 `tools/`에 있습니다. 예측을 얻는 부분만 `--checkpoint`(cnn) 또는 `--pred-csv`(픽셀 좌표 CSV, Filter 결과 등)로 나뉘고, 그림은 `dvslib/eval/visualize.py`가 그립니다.

```bash
python tools/evaluate.py --checkpoint cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth   # dvslib 지표 보고
python tools/evaluate.py --yolo-checkpoint yolo/runs/baseline_yolo_tiny/yolo_tiny_best.pth
python tools/plot_error_vs_frame.py --checkpoint cnn/runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
python tools/save_max_error_frame.py --pred-csv filter/results/no_filter_kalman.csv
python tools/compare_brownian.py --max-frames 100        # CNN vs YOLO vs Filter (순차 window, PNG)
python tools/inspect_bin.py data/gaussian_large.bin       # bin 구조·이벤트 통계
python tools/generate_brownian_dataset.py --help          # 데이터셋 생성 (보통 재실행 불필요)
```

별도의 테스트 프레임워크는 없습니다. `test.py`는 pytest가 아니라 필터 파이프라인 실행 스크립트입니다.

## Performance

세 방식은 정확도-연산량-FPGA 구현 난이도 사이의 trade-off 관계에 있습니다.

| 방식 | 상대 정확도 | 연산량 | FPGA 구현 | 노이즈 강건성 |
|---|---|---|---|---|
| Filter | 낮음 | 매우 낮음 | 용이 | 낮음 |
| CNN | 높음 | 중간 | 보통 | 높음 |
| YOLO | 중간 | 높음 | 어려움 | 높음 |

검증된 정량 baseline(재현 확인된 CNN 2.62px 등)은 [BASELINE.md](BASELINE.md)에 정리돼 있습니다. 세 방식을 동일 split·지표로 묶는 정량 비교(wandb)는 Phase 2 작업입니다 ([research_plan.md](research_plan.md)).

## Dependencies

핵심 라이브러리: numpy, pytorch(+torchvision), scipy, pandas, matplotlib, scikit-learn, filterpy(Kalman filter용). 전체 목록은 `environment.yml`을 참고하세요.

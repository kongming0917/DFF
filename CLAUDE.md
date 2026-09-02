# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DVS(Dynamic Vision Sensor) 카메라 데이터에서 레이저 빔 중심 좌표 (x, y)를 실시간 탐지하고 FPGA에 배포하는 졸업 연구. 같은 문제를 필터 / CNN / YOLO 세 가지로 풀어 비교합니다. 문서·주석은 한국어를 기본으로 하되, 챕터 제목·기술 용어·구조 레이블 등 영어가 더 읽기 편한 부분은 영어로 둡니다 (설명/근거는 한국어).

상세한 데이터 형식·모델 아키텍처·방식별 설명은 [README.md](README.md), 향후 연구 계획은 [research_plan.md](research_plan.md)를 참고하세요.

## Usage

```bash
conda env create -f environment.yml && conda activate dvs_project

# CNN(주력): dvslib 기반 thin experiment. argparse-based, non-interactive (명령 모음: cnn/COMMANDS.md)
python cnn/train.py --model mobilenet_v2 --epochs 50 --wandb
python cnn/inference.py                    # baseline checkpoint 평가
python tools/evaluate.py --checkpoint <best.pth>   # 방식 공용 평가 (--yolo-checkpoint / --pred-csv 도 가능)
python tools/compare.py --cnn a=<ckpt> --yolo b=<ckpt> --csv c=<csv>   # 방식 간 비교 → compare_result/<run>/
python tools/plot_error_vs_frame.py --checkpoint <best.pth>   # 방식 공용 분석 도구 (tools/)
python cnn/train_qat.py --checkpoint cnn/runs/baseline_mobileone_s0_pretrained/mobileone_s0_best.pth   # PT2E QAT → INT8

# YOLO(dvslib 기반, CNN과 동일 recipe·split)
python yolo/train.py --epochs 50 --seed 42
python yolo/inference.py

# Filter(dvslib 기반, 학습 없음): CSV 생성 → tools/evaluate.py --pred-csv 로 평가
python filter/run.py --max-frames 3000
python tools/evaluate.py --pred-csv filter/results/no_filter_kalman.csv --split val --max-frames 3000
```

테스트 프레임워크 없음. 각 방식의 `model.py`·`dvs_filter.py`는 `python <file>`로 self-test만 제공합니다.

## Directory Map (가장 헷갈리는 부분)

공통 로직은 **`dvslib` 패키지**에 있고, 각 방식(`cnn`·`yolo`·`filter`)은 그 위의 **thin experiment**입니다. **세 방식 모두 이식 완료** (Phase 1). 옛 self-contained 디렉토리와 `lib/` shim은 삭제됨.

| 디렉토리 | 역할 |
|---|---|
| `dvslib/` | 공통 패키지 — `data`(bin I/O·Dataset·blocked split)·`eval`(metric)·`training`(loop·callback)·`tracking`(wandb). 이식된 방식이 import |
| `cnn/` | **CNN 회귀 (주력)** — dvslib 기반 thin experiment. MobileNetV2 / MobileOne S0 + PT2E QAT(`train_qat.py`). `cnn_brownian_v2`에서 rename. 자체 `CLAUDE.md` |
| `yolo/` | **YOLOv3-Tiny 검출** — dvslib 기반 thin experiment. 데이터셋은 CNN과 동일, bbox 타깃·decode는 `model.py`의 hook(`YOLOCriterion`·`YOLOCenterDecoder`). 자체 `CLAUDE.md` |
| `filter/` | **필터 휴리스틱** (학습 없음) — `dvs_filter.py`(필터·추출기 본체, GT 원점 측정 도구이기도 함)·`run.py`(CSV 생성)·`origin.py`(정지 레이저 원점 후보). 자체 `CLAUDE.md` |
| `tools/` | 방식 공용 명령줄 도구 — `evaluate.py`(단일 소스 지표), `compare.py`(여러 소스를 blocked val 동일 프레임에서 비교, 로컬 CSV/MD/PNG + 선택적 wandb Table), 오차 시각화, 데이터셋 생성, bin 검사. 예측 얻기는 `_common.py`(cnn/yolo/csv), 계산은 dvslib |
| `archive/` | fixed-GT 1세대 — `cnn_sim`·`filter_sim`·`yolo_sim` |
| `eventrans/` | EventTransformer (Phase 3, 예정) |

리팩토링 전 원본 `cnn_brownian_sim`은 **삭제됨**(재현성 검증 완료 후). 옛 체크포인트 보존분은 `cnn/checkpoints_*/`, 기록값은 `BASELINE.md`.

## Key Rules

- 원시 데이터는 `data/`의 2-bit packed binary(`.bin`). bin I/O single source는 **`dvslib/data/bin_processor.py`** 하나뿐 (`lib/` shim은 삭제됨). 포맷 변경은 dvslib 한 곳만.
- **Filter 알고리즘(`filter/dvs_filter.py`)은 동작을 바꾸지 말 것.** 데이터셋 GT 원점 (541, 361)은 정지 레이저에서 여러 추출기 결과를 보고 수동으로 정한 값이라, 알고리즘이 바뀌면 GT 근거가 달라진다. Filter 결과 CSV의 프레임 매칭은 `frame_idx`(0부터)로 — bin 헤더 `frame_number`(88부터)를 GT에 직접 병합하지 말 것 (옛 평가 스크립트의 88프레임 어긋남 원인).
- 학습 루프(`dvslib/training/loop.py`)와 평가(`dvslib/eval/evaluate.py`)는 `criterion`·`to_xy` hook으로 방식 차이를 흡수한다. 새 방식은 hook만 제공하고 루프를 복제하지 말 것.
- `*/mobileone_official.py`는 Apple 공식 구현이므로 수정 금지.
- 데이터가 시계열이라 random split은 temporal leakage 발생 → blocked / K-fold split 사용 (split 로직은 `dvslib/data/split.py`에 일원화). 이 제약 유지.
- QAT는 **PT2E(`torch.export`) 기반** `cnn/quantization.py` + `cnn/train_qat.py`. eager QAT(QuantStub 수동 배치)는 폐기됐으므로 되살리지 말 것. FPGA 제약(대칭 INT8)은 `get_fpga_quantizer` 한 곳에만 기술.
- differentiable logic network 분기(`birel`, `cnn_diff`/LogicDVSNet)는 **삭제됨**. 관련 코드·의존성을 다시 추가하지 말 것.
- ROI는 `--roi 512`(정사각) 또는 `--roi 720x960`(HxW) — 파싱은 `dvslib.data.dataset.parse_roi` 한 곳. 데이터 경로는 `brownian_paths`로 결정.
- 분석·시각화 로직은 `dvslib/eval/visualize.py`에 함수로 두고, 방식별 진입점은 `tools/`에 얇게(예측 얻기만 `tools/_common.py`). 방식 디렉토리 안에 시각화 스크립트를 새로 만들지 말 것.
- 방식 간 성능 비교는 **항상 blocked val 동일 프레임 집합**(`tools/compare.py`)에서. 처음 N프레임 순차 비교는 학습 구간이라 무효 (옛 `compare_brownian.py`는 이 이유로 삭제됨).
- `.bin`·체크포인트·로그·이미지는 gitignore 대상이라 커밋되지 않습니다.

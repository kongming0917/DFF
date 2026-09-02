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
python tools/plot_error_vs_frame.py --checkpoint <best.pth>   # 방식 공용 분석 도구 (tools/)
python cnn/train_qat.py --checkpoint cnn/runs/baseline_mobileone_s0_pretrained/mobileone_s0_best.pth   # PT2E QAT → INT8

# YOLO/Filter: 아직 옛 self-contained layout — 해당 디렉토리 안에서 실행 (assumes cwd-relative paths)
cd yolo_brownian_sim && python train.py
```

테스트 프레임워크 없음. `test.py`는 pytest가 아니라 필터 파이프라인 실행 스크립트입니다.

## Directory Map (가장 헷갈리는 부분)

리팩토링이 진행 중입니다. 공통 로직은 **`dvslib` 패키지**로 모이고, 각 방식은 그 위의 **thin experiment**가 됩니다. **CNN은 이식 완료**, YOLO/Filter는 아직 옛 self-contained layout(`train.py`/`inference.py`/`dataset.py`/`model.py`/`utils.py` 반복)입니다.

| 디렉토리 | 역할 |
|---|---|
| `dvslib/` | 공통 패키지 — `data`(bin I/O·Dataset·blocked split)·`eval`(metric)·`training`(loop·callback)·`tracking`(wandb). 이식된 방식이 import |
| `cnn/` | **CNN 회귀 (주력)** — dvslib 기반 thin experiment. MobileNetV2 / MobileOne S0 + PT2E QAT(`train_qat.py`). `cnn_brownian_v2`에서 rename. 자체 `CLAUDE.md` |
| `yolo_brownian_sim/` | YOLOv3-Tiny 검출 — 옛 layout (dvslib migration 예정) |
| `filter_brownian_sim/` | 필터 휴리스틱 (학습 불필요) — 옛 layout |
| `tools/` | 방식 공용 명령줄 도구 — 데이터셋 생성·bin 검사·오차 시각화·3방식 비교. 계산은 dvslib에 위임, 경로는 인자 |
| `archive/` | fixed-GT 1세대 — `cnn_sim`·`filter_sim`·`yolo_sim` |
| `eventrans/` | EventTransformer (Phase 3, 예정) |
| `lib/` | `bin_processor` re-export shim → dvslib (미이식 디렉토리용) |

리팩토링 전 원본 `cnn_brownian_sim`은 **삭제됨**(재현성 검증 완료 후). 옛 체크포인트 보존분은 `cnn/checkpoints_*/`, 기록값은 `BASELINE.md`.

## Key Rules

- 원시 데이터는 `data/`의 2-bit packed binary(`.bin`). bin I/O single source는 **`dvslib/data/bin_processor.py`** (`lib/bin_processor.py`는 backward-compat shim — 미이식 디렉토리가 `from lib.bin_processor import ...`로 계속 동작). 포맷 변경은 dvslib 한 곳만.
- `*/mobileone_official.py`는 Apple 공식 구현이므로 수정 금지.
- 데이터가 시계열이라 random split은 temporal leakage 발생 → blocked / K-fold split 사용 (split 로직은 `dvslib/data/split.py`에 일원화). 이 제약 유지.
- QAT는 **PT2E(`torch.export`) 기반** `cnn/quantization.py` + `cnn/train_qat.py`. eager QAT(QuantStub 수동 배치)는 폐기됐으므로 되살리지 말 것. FPGA 제약(대칭 INT8)은 `get_fpga_quantizer` 한 곳에만 기술.
- differentiable logic network 분기(`birel`, `cnn_diff`/LogicDVSNet)는 **삭제됨**. 관련 코드·의존성을 다시 추가하지 말 것.
- ROI는 `--roi 512`(정사각) 또는 `--roi 720x960`(HxW) — 파싱은 `dvslib.data.dataset.parse_roi` 한 곳. 데이터 경로는 `brownian_paths`로 결정.
- 분석·시각화 로직은 `dvslib/eval/visualize.py`에 함수로 두고, 방식별 진입점은 `tools/`에 얇게(예측 얻기만 `tools/_common.py`). 방식 디렉토리 안에 시각화 스크립트를 새로 만들지 말 것.
- `.bin`·체크포인트·로그·이미지는 gitignore 대상이라 커밋되지 않습니다.

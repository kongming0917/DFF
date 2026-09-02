# BASELINE.md

리팩토링(공통 패키지 통합) 전 회귀 기준선. 리팩토링된 코드는 아래 지표를 오차 범위 내에서 재현해야 한다. 측정 스크립트: `baseline_cnn.py` (원본 코드 미수정, `cnn_brownian_sim`의 best 체크포인트 사용).

## CNN — MobileNetV2

### ⚠️ 기존 기록(2.11px)은 무효 — 재현 불가한 유령 수치

`cnn_brownian_sim/checkpoints_mobilenet_v2/metrics_history.json`은 2.11px / Acc@5px 98%를
기록하지만, **저장된 어떤 체크포인트로도 재현되지 않는다** (검증 결과):

- `mobilenet_v2_best.pth`에는 epoch-1 가중치가 저장돼 있음 (→ 11.9px) — 한 `save_dir`에
  여러 run의 산출물이 뒤섞임. metrics_history·best.pth·epoch_*.pth가 서로 다른 run.
- 재현 가능한 최고 저장 체크포인트는 epoch_30의 8.9px / 68.7%.
- 즉 2.11px는 blocked split 이전(temporal leakage) 또는 train 평가에서 나온 값으로 추정.

원본 loader와 dvslib loader가 **동일한 결과**를 내므로(11.878px) dvslib 추출은 정확함.

### 중간 단계: unseeded 2.62px도 단발(운)이었음

dvslib 재학습 첫 결과는 2.62px/88%였으나 **seed 미고정**이었다. seed를 고정해 재학습하니
**학습 분산이 크다는 게 드러났다** — 같은 코드·데이터로 seed에 따라 3.16~7.22px, 그리고
**5개 중 1개(seed 2024)는 epoch 3에서 붕괴**(7.22px/39%). 즉 단일 run 수치는 baseline으로 부적합.

레시피 진단: `LR=1e-3` + `ReduceLROnPlateau`만으로는 val이 noisy하고 일부 init이 발산.
→ **warmup(3ep) + cosine annealing + grad clip(1.0)** 도입. 추가로 seed 고정(`seed_everything`),
monitor를 `val_loss`로 통일.

### ✅ 안정화된 baseline (seeded, cosine recipe)

recipe = MSE / Adam / **LinearWarmup(3) → CosineAnnealing** / **grad_clip 1.0**, batch 32, lr 1e-3,
blocked split, **seed 고정(결정적: run1==run2 bit-identical 검증)**. monitor=`val_loss`.

5-seed sweep (42·123·2024·7·1234):

| recipe | pixel error (mean±std) | Acc@5px | Acc@10px | range | 붕괴(>5px) |
|---|---|---|---|---|---|
| plateau (옛) | 4.11 ± 1.57 | 75.4 ± 18.3 | 90.1 ± 3.6 | 3.16~7.22 | 1/5 |
| **cosine (현)** | **3.07 ± 0.31** | **85.7 ± 1.3** | **94.0 ± 2.1** | 2.68~3.54 | **0/5** |

→ 붕괴 제거, 분산 5× 축소, 평균 개선. **baseline = 3.07 ± 0.31 px / Acc@5px 85.7 ± 1.3 %** (5 seeds).

**Canonical 체크포인트** (seed 42, 결정적·재검증 완료):

| 항목 | 값 |
|---|---|
| val pixel error (mean ± std) | **2.79 ± 2.83 px** |
| val Acc@5px / @10px | **87.52 % / 94.17 %** |
| 추론 처리량 | ~1090 FPS (batched bs=32, warmup 후 측정) |
| 측정 GPU | NVIDIA RTX 4090 |

- 체크포인트: `cnn/runs/baseline_mobilenet_v2/` (= seed 42 cosine, epoch 34, monitor=val_loss)
- 재현: `python cnn/train.py --model mobilenet_v2 --seed 42` (기본 레시피가 cosine — 동일 결과)
- best.pth 재로드 평가 = 학습 기록값과 정확히 일치 (2.7933px / 87.52%)

> 참고: phantom 2.11px(acc5 98%)와 unseeded 2.62px는 모두 재현 불가/단발. 신뢰 baseline은 위 seeded 수치.

## CNN — MobileOne S0

mobilenet_v2와 **동일 recipe**(seed 42, MSE/Adam, LinearWarmup(3)→CosineAnnealing, grad_clip 1.0,
batch 32, lr 1e-3, blocked split, max_frames 3000)로 학습한 refactored baseline. 평가는
**reparameterize(single-branch) 후** — MobileOne 배포 구조 = 추론 구조. 백본은 **Apple 공식
ImageNet pretrained**(`mobileone_s0_unfused`, stage0(5ch)·head는 random) — 학습은 항상 pretrained.

| 항목 | 값 |
|---|---|
| val pixel error (mean ± std) | **2.82 ± 3.22 px** |
| val Acc@5px / @10px | **85.70 % / 93.17 %** |
| 추론 처리량 | **~2218 FPS** (batched bs=32, reparam 후, warmup 후 측정) |
| 측정 GPU | NVIDIA RTX 4090 |

- 체크포인트: `cnn/runs/baseline_mobileone_s0_pretrained/` (seed 42 cosine, epoch 41, monitor=val_loss)
- 재현: `python cnn/train.py --model mobileone_s0 --seed 42` (train은 pretrained 항상 적용)
- best.pth 재로드(reparam) 평가 = 학습 기록값과 일치 (2.8224px / 85.70%)

### MobileOne vs MobileNetV2 — pretrained가 결정적

| 모델 | pixel error | Acc@5px | Acc@10px | FPS | init |
|---|---|---|---|---|---|
| mobilenet_v2 | 2.79px | 87.5 % | 94.2 % | 1032 | ImageNet pretrained |
| **mobileone_s0 (pretrained)** | **2.82px** | **85.7 %** | 93.2 % | **2218** | ImageNet pretrained |
| mobileone_s0 (from-scratch, ablation) | 4.59px | 71.4 % | 87.2 % | 1934 | random init |

- **pretrained mobileone = mobilenet급 정확도(2.82 vs 2.79px)에 ~2.1× 속도(2218 vs 1032 FPS)** → FPGA 배포 1순위.
- from-scratch(4.59px) → pretrained(2.82px) **−1.77px, acc5 71→86%**. 격차의 원인은 전적으로 **init(pretrain 유무)**.
  작은 데이터(3000 frame)·50 epoch에선 ImageNet pretrain이 결정적.
- from-scratch ablation 재현: `python cnn/train.py --model mobileone_s0 --seed 42` 에서
  모델 생성 시 `pretrained=False` (옛 체크포인트 `cnn/runs/baseline_mobileone_s0/`, 4.59px).
- 옛 기록 **2.66px/92.5%(아래 표)는 blocked split 이전 수치**라 mobilenet phantom 2.11px처럼
  **temporal leakage로 부풀려졌을 가능성**이 큼. 공정 blocked split + pretrained에선 2.82px가 신뢰 baseline.

## 보존된 옛 체크포인트 (기록값 — best.pth로 재검증 필요)

리팩토링 전 학습된 체크포인트들. 디스크 절약을 위해 `epoch_*.pth`(수 GB)는 삭제하고
각 dir에 **`*_best.pth`·`*_int8.pth`·`metrics_history.json`·`config.json`·결과 PNG만 보존**했다.
아래 수치는 `metrics_history.json`의 best epoch 기록값으로, **저장된 best.pth와 일치하지 않을 수
있다** (아래 mobilenet_v2 512의 2.11px가 그 사례). 신뢰하려면 best.pth를 dvslib로 재평가할 것.

| 체크포인트 dir | model | data | best px err (기록) | Acc@5px | epochs | INT8 |
|---|---|---|---|---|---|---|
| `checkpoints_mobilenet_v2` | mobilenet_v2 | 512 | 2.11 ⚠️ phantom | 98.0 | 50 | — |
| `checkpoints_mobilenet_v2_720x960` | mobilenet_v2 | 720×960 | 3.09 | 86.4 | 30 | — |
| `checkpoints_mobilenet_v2_qat` | mobilenet_v2 | 512 | 2.70 | 88.0 | 10 | `*_int8.pth` |
| `checkpoints_mobileone_s0` | mobileone_s0 | 512 | 2.66 ⚠️ pre-blocked | 92.5 | 33 | — |
| `checkpoints_mobileone_s0_qat` | mobileone_s0 | 512 | 2.94 | 87.3 | 30 | `*_int8.pth` (per-channel) |
| `checkpoints_mobileone_s0_qat_tensor` | mobileone_s0 | 512 | 2.98 | 87.3 | 30 | `*_int8.pth` (per-tensor) |

- 512 FP32 모델은 `python cnn/inference.py --checkpoint <best.pth> --model <name>`으로 재평가 가능.
- 720×960은 비정사각 ROI라 현재 `inference.py`(정사각 `--roi`만 가정)로는 직접 평가 불가 — 별도 처리 필요.
- INT8 로드 검증: `python cnn/model_summary.py --model <name> --int8 <int8.pth>`.

## YOLO — YOLOv3-Tiny (`yolo_brownian_sim`, dvslib 이식 전)

측정: `python tools/evaluate.py --yolo-checkpoint yolo_brownian_sim/checkpoints_yolo_tiny_laser_brownian/yolo_tiny_laser_brownian_best.pth`
(`tools/_common.yolo_predict` — 옛 inference 의미 그대로: 입력 /max 정규화, decode → NMS → ROI 중심 우선, 검출 실패 시 직전 성공 좌표).
구현 검증: 처음 100프레임 순차 평가가 옛 `compare_brownian.py` 기록값 9.934px와 일치 (9.9347px).

### 저장된 체크포인트의 실체

- 체크포인트는 `*_best.pth` 하나뿐 (**epoch 5**, val_loss 1.383). metrics_history·config 없음 (PNG만).
- 학습 조건: **max_frames=500**, `random_split` 80/20 (**seed 없음 → 검증셋 재현 불가**, temporal leakage), batch 4, lr 1e-3, conf 0.6.
- 즉 학습 시 기록된 pixel error는 재현할 수 없고, 아래 재평가 수치가 유일한 baseline.

### 재평가 (동일 체크포인트, 현재 데이터)

| 평가 구간 | conf | samples | pixel error (mean ± std) | Acc@5px | Acc@10px | 검출률 | FPS |
|---|---|---|---|---|---|---|---|
| **blocked val (3000f, CNN과 동일)** | **0.6** | 1098 | **93.69 ± 52.52 px** | **1.37 %** | **3.46 %** | 71.6 % | ~1109 |
| blocked val (3000f) | 0.3 | 1098 | 84.87 ± 51.61 px | 0.91 % | 3.55 % | 100 % | ~1267 |
| 처음 500f 전체 (= 학습 데이터 포함) | 0.6 | 496 | 12.57 ± 18.97 px | 21.57 % | 57.86 % | 98.2 % | ~1204 |
| 처음 100f 전체 (옛 compare 조건) | 0.6 | 96 | 9.93 ± 10.69 px | 32.29 % | 68.75 % | 97.9 % | — |

- **회귀 검증 기준 = blocked val, conf 0.6: 93.69px / 1.37%** (dvslib 이식 후 같은 체크포인트로 이 값을 재현해야 함). FPS는 RTX 4090, bs=32, 공유 GPU 상태에서 측정.
- **해석**: 이 체크포인트는 학습 구간(프레임 0~499, 모두 train block 0에 속함) 밖에서는 **사실상 동작하지 않는다**
  (val block은 599~1147, 1797~2345). 500프레임·5 epoch 학습 + random split의 낙관적 검증이 만든 결과로,
  옛 비교표의 YOLO 9.9px는 학습 데이터 위 수치였다. **이식 후에는 CNN과 동일 recipe(3000f, blocked, seed)로 재학습해야 공정 비교 가능.**
- conf 0.3은 검출률 100%지만 오차는 그대로 큼 → 임계값 문제가 아니라 일반화 실패.

## Filter — 휴리스틱 (`filter/`, dvslib 이식 후 측정)

측정: `python filter/run.py --max-frames 3000` → `filter/results/*.csv`,
`python tools/evaluate.py --pred-csv filter/results/<cond>.csv --split val --max-frames 3000`.
학습이 없으므로 "이식 전후 동일성"은 CSV로 검증했다: `filter/run.py`(dvslib bin I/O, 필터 단계 16-worker 병렬,
필터 결과를 추출기 간 공유)가 옛 `filter_brownian_sim/csv_results/`(100프레임)와 **bit-identical**
(spatial_filter_kalman만 3.8e-5px 차이 — 옛 코드가 Kalman 객체를 재사용하며 `reset()`이 공분산을 완전히 초기화하지 않던 잔여 상태).
평가 프레임 = CNN·YOLO blocked val 샘플의 **중심 프레임**(1098개) → 세 방식이 같은 프레임 집합에서 비교됨.

### 옛 기록은 무효

`filter_brownian_sim/evaluate_against_ground_truth.py`는 bin 헤더 `frame_number`(**88부터**)를 GT `frame_idx`(0부터)에
그대로 병합해 **88프레임 어긋난 비교**였다 → `evaluation_results/*.png`의 수치는 신뢰 불가. 또 옛 CSV는 100프레임(train block 0)뿐.

### 재평가 (3000프레임, blocked val 중심 프레임 1098개)

| 조건 | pixel error (mean ± std) | Acc@5px | Acc@10px |
|---|---|---|---|
| no_filter + Median | 20.31 ± 33.67 px | 60.47 % | 69.31 % |
| spatial_filter + Median | 20.53 ± 34.21 px | 60.56 % | 69.22 % |
| **no_filter + Kalman** | **10.62 ± 10.94 px** | 44.17 % | 64.75 % |
| spatial_filter + Kalman | 10.82 ± 11.19 px | 43.72 % | 63.93 % |

- **회귀 검증 기준 = 위 4개 CSV** (`filter/results/`, 재생성 시 동일해야 함). 처리 시간: SpatialClusterFilter 3000f ≈ 7분(16 workers), 추출기 ≈ 15초.
- 해석: Median은 절반 이상 프레임에서 5px 안이지만 실패 프레임의 오차가 매우 커서(std 34px) 평균이 나쁘다.
  Kalman은 이상치를 눌러 평균 오차는 절반이지만 지연 때문에 5px 정확도는 낮다. SpatialClusterFilter는 이 데이터에서 효과 없음(±0.2px).
- 세 방식 비교(blocked val, 동일 프레임): CNN 2.79px / YOLO(옛 ckpt) 93.7px / Filter(Kalman) 10.6px.
- Filter는 GT 원점(541, 361)을 정할 때 쓴 측정 도구이기도 하다 — 알고리즘 변경 시 GT 근거가 달라지므로 주의.

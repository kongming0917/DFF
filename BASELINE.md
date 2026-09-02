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

## YOLO — (예정)

`yolo_brownian_sim` 마이그레이션 직전에 동일 방식으로 기록.

## Filter — (예정)

`filter_brownian_sim` 마이그레이션 직전에 동일 방식으로 기록.

# research_plan.md

DVS 카메라 데이터에서 레이저 빔 중심 좌표 (x, y)를 실시간 탐지하는 연구의 발전 계획. 기존 3가지 방식(필터 / CNN / YOLO)을 전문가 수준 코드로 재정비하고, 정량적 실험 추적 체계를 세운 뒤, EventTransformer 모델로 확장한다.

## Assessment

**강점**
- 3가지 접근법 + brownian/fixed 두 데이터셋으로 실험 폭이 넓다.
- bin I/O가 `lib/bin_processor.py` 한 곳에 이미 분리돼 있어 통합의 출발점이 좋다.

**약점**
- 거의 동일한 코드가 6개+ 디렉토리에 중복 → 한 곳 수정이 다른 곳에 반영되지 않고, 버그가 전파된다.
- 비교 결과가 PNG 위주 → 재현·정량 비교가 불가능하다.
- split/평가 지표가 디렉토리마다 미세하게 다를 수 있어 공정 비교가 위험하다.

**계획 평가**
- 1·2번 동시 진행, 3번 후행은 타당하다. 깨끗한 파이프라인·평가 체계 없이 새 모델을 비교하면 신뢰할 수 없기 때문이다.
- **보완점:** 원안에는 *리팩토링 전 회귀 기준선(baseline)* 이 빠져 있다. 통합 후 정확도가 동일하게 재현되는지 검증할 기준이 없으면 "완성도 향상"을 증명할 수 없다 → **Phase 0**로 추가했다.

## Decisions
- 리팩토링: **공통 패키지로 통합** (중복 제거)
- 리팩토링 대상(활성 방식): **CNN · YOLO · Filter의 brownian 버전 전부** (CNN을 파일럿으로 먼저 진행)
- fixed-GT 구버전(`cnn_sim`, `filter_sim`, `yolo_sim` 등): **`archive/`로 이동**
- 실험 추적: **wandb** (이미 사용 중)

## Open Questions
- [x] 공통 패키지 이름 확정 → **`dvslib`** (구조는 Phase 1-1)
- [ ] EventTransformer 출발 repo URL
- [ ] 정량 비교 표준 지표 확정 (제안: 평균 픽셀 오차, RMSE, Acc@3/5/10px, 추론 FPS)

---

## Phase 0 — Baseline (리팩토링 전 필수)

리팩토링이 정확도를 깨지 않았음을 증명할 비교 기준 마련. 각 방식은 해당 마이그레이션 직전에 baseline을 기록한다.

- [x] CNN baseline 확보 — 기존 기록(2.11px)은 재현 불가 유령 수치, 초기 unseeded 2.62px도 단발(운)로 판명 → seed 고정 + cosine recipe로 안정화. **재현 검증된 baseline: 5-seed sweep 3.07±0.31px / Acc@5px 85.7±1.3% (붕괴 0/5), canonical seed42 = 2.79px / 87.5% / 94.2% / ~1090 FPS** (best.pth 재로드 = 기록값 일치). 상세: `BASELINE.md`
- [ ] YOLO(`yolo_brownian_sim`) 현 지표 baseline 기록
- [ ] Filter(`filter_brownian_sim`) 현 지표 baseline 기록
- [ ] 공통: split·seed 고정, 수치 캡처 (mean px error, Acc@5px, 추론 FPS)

**완료 기준:** 리팩토링할 각 방식에 대해 wandb에 `baseline` run이 존재하고 수치가 문서화됨.

## Phase 1 — Refactoring (공통 패키지 통합)

### 1-1. Package structure
제안 구조 (확정 필요):
```
dvslib/
  data/      # bin I/O(lib 이전), Dataset 베이스, split(blocked/K-fold)
  models/    # 공유 컴포넌트, model registry
  training/  # Trainer 루프, EarlyStopping, Checkpoint, MetricsTracker
  tracking/  # wandb 래퍼
  eval/      # 지표(px error, Acc@Npx), 비교 로직
# 방식별 디렉토리는 top-level flat 유지, dvslib을 import
cnn/   yolo_brownian_sim/   filter_brownian_sim/    # 얇은 config + entrypoint
archive/                                            # fixed-GT 구버전 (이동 완료)
```
- [x] 이름·구조 확정 → `dvslib` (data/models/training/tracking/eval) + 방식 디렉토리 top-level flat + `archive/`

### 1-2. Archive 구버전 ✓
- [x] `cnn_sim`, `filter_sim`, `yolo_sim` + `compare.py`를 `archive/`로 이동 (git rename 인식, `archive/README.md` 추가)
- [x] 이동 후 활성 코드 구버전 참조 없음 확인 (스모크: dvslib import OK, baseline 2.11px 재현)

### 1-3. 공통 코드 추출
- [ ] bin I/O: `lib/bin_processor.py` → `dvslib/data/`로 이전, 단일 소스 유지
- [ ] 중복 utils(EarlyStopping / Checkpoint / MetricsTracker / 시각화) 통합
- [ ] Dataset 베이스 + split 로직 통합 (temporal leakage 방지 규칙 일원화)
- [ ] Trainer 루프 공통화, 모델별 차이는 `config` / `model.py`에만 남김

### 1-4. 활성 방식 마이그레이션 (CNN → YOLO → Filter)
- [x] **CNN**: `cnn_brownian_v2` → `cnn`, `dvslib` 기반 thin experiment로 재작성 완료. `train.py`·`inference.py` 모두 dvslib 사용. QAT는 `cnn/quantization.py`로 분리, 옛 스크립트·중복 체크포인트 정리. README·CLAUDE 갱신. baseline 재현 검증됨 (2.62px)
- [ ] **YOLO**: `yolo_brownian_sim`을 `dvslib` 기반으로 재작성 (CNN 마이그레이션 패턴 그대로 적용)
- [ ] **Filter**: `filter_brownian_sim`을 `dvslib` 기반으로 정리 (학습 없는 방식이라 data/eval만 공유)

**부속 스크립트 이관** ✓ — `cnn_brownian_sim` 삭제 완료 (재현성 검증: baseline 체크포인트 2.7933px/2.8224px 재현, seed 42 학습 run 간 bit-identical, 기록된 run의 초반 epoch과 일치).
부수 발견: `generate_brownian_dataset.py`는 `-org` 옵션 추가 시 `else:` 본문 들여쓰기가 빠져 **import 자체가 불가능한 상태**였음 → 수정. 방식 무관한 분석 로직은 `dvslib/eval`에 함수로, 명령줄 진입점은 루트 `tools/`에 얇게(경로는 인자로, 계산은 dvslib에 위임).

- [x] `dvslib/eval/visualize.py` 신설: 프레임별 pixel error 그래프, 최대 오차 프레임 GT·예측 오버레이 (← `plot_error_vs_frame.py`, `save_max_error_frame.py`). `evaluate_regression`이 프레임 인덱스·예측 좌표를 함께 반환하도록 확장
- [x] `tools/` 신설 후 루트 산재 스크립트 이동: `compare_brownian.py`, `inspect_bin.py`, `visualize_max_movement_frames.py`, `generate_brownian_dataset.py`. 절대 경로 제거, argparse로 통일
- [x] `tools/plot_error_vs_frame.py`, `tools/save_max_error_frame.py` 진입점 작성 (CNN·YOLO·Filter 공용)
- [x] 폐기(이관 안 함): `debug_augmentation.py`(brownian은 학습 시 증강 없음), `check.py`·`check_one.py`(`cnn/model_summary.py`·`python cnn/model.py`가 대체), `compare.py`(eager INT8 전제·절대 경로 — FP32 vs INT8 비교는 Phase 2 wandb 비교로 대체)
- [x] 720×960 비정사각 ROI: **지원 추가** — `--roi 720x960`(HxW) 파싱을 `dvslib.data.dataset.parse_roi`로 일원화, train/inference 모두 적용. 옛 720×960 체크포인트 로드·평가 동작 확인(단, 그 체크포인트는 pre-blocked 학습이라 수치는 참고용)
- [x] 위 완료 후 `cnn_brownian_sim` 삭제, `CLAUDE.md`·`README.md` Directory Map 갱신(`tools/` 추가, `cnn_brownian_sim` 제거)

**완료 기준:** 세 방식 모두 통합 구조에서 동작하고, 각자 Phase 0 baseline 지표를 오차 범위 내 재현. 중복 코드 대폭 감소.

### 1-5. CNN 파이프라인 품질 감사 (code review)

마이그레이션이 "동작"을 넘어 "잘 구성"됐는지 점검 — 불필요(vestigial)·비효율 코드 제거. 순서대로:

- [x] **Data**: 구성·증강·로딩 검토 완료 (`dvslib/data`). 프레임 **uint8 보관**으로 메모리 4× 절감(3GB→750MB), 삼중 변환→단일 `.float()`, label `frame_idx→(x,y)` dict 사전계산, split **1 dataset + 2 Subset**, 死코드 제거(`training` 플래그·`sensor_size`·死 가드), `read_frames` print 제거. baseline 2.62px bit-identical 재현. (float32 캐스팅은 제거된 정규화의 잔재였음)
- [x] **Training**: `dvslib/training` 검토·개선 완료. **seed 고정**(`seed_everything`, 결정적: run1==run2 bit-identical), monitor를 `val_loss`로 통일, mae를 metrics로 dedup, non_blocking 전송. **레시피 안정화**: seed sweep으로 plateau 불안정(1/5 붕괴, 3.16~7.22px) 발견 → **warmup+cosine+grad_clip** 도입 → 붕괴 0/5, 분산 5×↓, baseline **3.07±0.31px**(5 seeds). canonical seed 42 = 2.79px. cosine을 기본 레시피로 채택
- [x] **Inference**: `cnn/inference.py` + `dvslib/eval` 검토·개선 완료. **체크포인트가 학습 config(model·tw·roi 등)를 저장**하고 inference가 자동으로 읽음(config drift 방지, CLI override 가능, 옛 체크포인트는 defaults fallback). `evaluate_regression`에 **warmup** 추가(첫 배치 cudnn autotune 제외 → FPS 정확: ~1090). 검증 완료
- [x] **2모델 적용**: `mobilenet_v2` / `mobileone_s0` 둘 다 data→train→infer 전 구간 동작 검증. mobileone은 **reparameterize 경로 포함** — `inference.py`가 reparam을 안 하던 이슈 발견·수정(load 후 single-branch 융합). 정확도 동일(28.3558→28.3555px, float noise)·**FPS 5×↑(379→1894)**. (smoke 3ep 기준 — 정확도 아닌 동작/구조 검증)
- [x] **baseline 재현 유지**: 변경(reparam 추가) 후 mobilenet_v2 baseline **2.7933px / 87.52% / 94.17% 정확 재현**(reparam은 `hasattr` 가드로 스킵 → 영향 없음). mobileone_s0(seed42, 동일 recipe)는 from-scratch 4.59px였으나, **Apple 공식 ImageNet pretrained 적용 → 2.82px / 85.7% / ~2218 FPS** (`runs/baseline_mobileone_s0_pretrained`). 즉 **mobilenet급 정확도(2.82 vs 2.79px)에 ~2.1× 속도** → FPGA 1순위. 격차 원인은 전적으로 pretrain 유무(−1.77px). 상세: `BASELINE.md`

**완료 기준:** CNN 파이프라인에 불필요/비효율 코드 없이 깔끔하고, 두 모델 모두 동작하며 baseline 재현.

## Phase 2 — Experiment Tracking (wandb) + Quantization 통일

- [ ] `dvslib/tracking`에 wandb 래퍼 (config·metric·모델 아티팩트 로깅)
- [ ] 학습 시 epoch별 loss/지표, 검증 예측 시각화 자동 로깅
- [ ] `compare.py` / `compare_brownian.py`를 wandb Table/Report 기반 정량 비교로 교체 (PNG 의존 제거)
- [ ] 3가지 방식을 동일 split·지표로 한 번에 비교하는 스크립트
- [~] **QAT를 PT2E 기반으로 통일 (eager 대체).** `cnn/quantization.py`·`cnn/train_qat.py`가 PT2E로 재작성됨(진행 중 — `dvslib/quant`로의 이동, `export_mobileone_info.py` PT2E 대응은 남음). 옛 eager 코드는 **MobileOne 전용**(MobileNetV2는 QuantStub 부재·fusion 스킵으로 QAT 불가)이고 `reduce_range` 등 x86 잔재가 박혀 있음. PT2E(`capture_pre_autograd_graph` + Quantizer)는 **모델 무관**으로 observer/fake-quant·Conv-BN fusion을 그래프에서 자동 처리 → CNN(mobilenet_v2/mobileone_s0)·YOLO·EventTransformer에 동일 적용. FPGA 제약(대칭 INT8, 향후 PoT·bit-width)은 Quantizer 한 곳에 기술. **부수 작업**: FPGA weight 추출(`export_mobileone_info.py`)을 PT2E 그래프 기준으로 재작성, 옛 eager INT8 체크포인트는 호환 불가(재학습 필요).

**완료 기준:** 새 모델 추가만으로 wandb에서 비교 가능(PNG 수작업 불필요), 모든 방식이 동일 PT2E 경로로 INT8 양자화됨.

## Phase 3 — EventTransformer (Phase 1·2 완료 후)

- [ ] 대상 repo 확정 및 clone (`eventrans/` 활용)
- [ ] 입력 어댑터: DVS bin/이벤트 → EventTransformer 입력 형식
- [ ] `dvslib` 평가·추적 체계에 통합 (동일 split·지표·wandb)
- [ ] 기존 3방식과 정량 비교

**완료 기준:** EventTransformer가 동일 파이프라인에서 학습·평가되고 비교표에 포함됨.

---

## Risks
- 통합 중 미세한 동작 변화로 정확도 저하 → Phase 0 baseline으로 상시 검증.
- split/seed 불일치로 불공정 비교 → split 로직을 `dvslib`에 일원화.
- EventTransformer 입력 형식 차이가 클 수 있음 → 어댑터 계층으로 분리.

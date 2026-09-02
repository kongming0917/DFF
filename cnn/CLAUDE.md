# CLAUDE.md — cnn experiment

DVS 레이저 빔 중심 좌표 (x, y)를 예측하는 CNN 회귀 experiment. `cnn_brownian_v2`에서 이름이 바뀌었고,
이제 **dvslib 위에 올라간 얇은 experiment**다. 데이터·split·metric·학습 루프·wandb는 모두 dvslib에서 오고,
이 디렉토리는 **모델·진입점·양자화**만 가진다. 최종 목표는 FPGA 배포 (MobileOne S0 + PT2E QAT → INT8).

## Structure

```
cnn/
├── train.py               # FP32 학습 진입점 (dvslib.training 사용). argparse 기본값이 곧 config (별도 config 파일 없음)
├── inference.py           # 체크포인트 평가 (dvslib.eval 사용). 체크포인트에 저장된 학습 config를 자동 로드
├── train_qat.py           # PT2E QAT fine-tune 진입점: FP32 ckpt → reparam → QAT → INT8 그래프 저장
├── quantization.py        # PT2E 유틸 — get_fpga_quantizer / prepare_qat / convert / set_qat_mode
├── model.py               # 모델 정의 (mobilenet_v2, mobileone_s0). `python model.py`로 self-test
├── mobileone_official.py  # Apple 공식 구현 — 수정 금지
├── model_summary.py       # 구조·파라미터·메모리 분석 (reparam 전/후, --int8 로드 검증)
├── export_mobileone_info.py  # FPGA용 weight·구조 추출 → mobileone_info/ (gitignore)
├── COMMANDS.md            # 복붙용 실행 명령 모음
├── runs/                  # 학습 산출물 (체크포인트·metrics·로그) — gitignore
└── checkpoints_*/         # 리팩토링 전 옛 체크포인트 보존분 (best/int8/metrics만) — gitignore, BASELINE.md 참고
```

dvslib에서 오는 것: `data`(bin I/O·Dataset·blocked split), `eval`(metrics·evaluate), `training`(Trainer·callbacks·seed), `tracking`(wandb).

## Usage

```bash
python cnn/train.py --model mobilenet_v2 --epochs 50 --wandb   # 학습 + wandb 기록
python cnn/train.py --model mobileone_s0 --seed 42              # FPGA 후보 (ImageNet pretrained 항상 적용)
python cnn/inference.py                                         # baseline best.pth 평가
python cnn/inference.py --checkpoint <path>
python cnn/train_qat.py --checkpoint cnn/runs/baseline_mobileone_s0_pretrained/mobileone_s0_best.pth
```

비대화형(인자 기반). 데이터 경로는 `--roi`로 결정 (`512` → `data/gaussian_brownian_512x512.*`, `720x960` → `..._720x960.*`; HxW 문자열, `dvslib.data.dataset.parse_roi`).
train recipe 기본값: MSE / Adam / lr 1e-3 / batch 32 / LinearWarmup(3) → CosineAnnealing / grad_clip 1.0 / seed 42.

## Baseline (seeded, 재현 검증됨)

동일 recipe(blocked split, cosine, seed 42). 상세와 옛 체크포인트 기록값은 루트 `BASELINE.md`.

| 모델 | val px err | Acc@5px | FPS(bs=32) | 체크포인트 |
|---|---|---|---|---|
| mobilenet_v2 | 2.79px | 87.5 % | ~1090 | `runs/baseline_mobilenet_v2/` |
| mobileone_s0 (pretrained, reparam 후) | 2.82px | 85.7 % | ~2218 | `runs/baseline_mobileone_s0_pretrained/` |

mobilenet_v2 5-seed sweep: **3.07 ± 0.31 px / Acc@5px 85.7 ± 1.3 %** (붕괴 0/5). mobileone은 mobilenet급 정확도에 ~2.1× 속도라 FPGA 1순위.
QAT 산출물: `runs/qat_mobileone_s0_pretrained/` (`*_int8.pth`).

## Key Rules

- `mobileone_official.py`는 Apple 공식 구현이므로 수정 금지.
- 데이터·split·metric을 여기서 재구현하지 말 것 — dvslib을 import (split이 갈라지면 방식 간 비교가 무효).
- 학습 run은 항상 fresh 디렉토리(`runs/<name>`)에 저장 — 한 `save_dir`에 여러 run을 섞으면 best 체크포인트가 오염됨 (옛 baseline 붕괴 원인).
- 재현성: `--seed`로 결정적 학습 (기본 42). 기본 레시피는 cosine(warmup 3 + cosine decay + grad_clip 1.0) — plateau보다 seed-robust.
- MobileOne 평가·배포는 항상 **reparameterize 후**(single-branch). `inference.py`가 load 후 자동 수행.
- QAT는 **PT2E(`torch.export`) 기반**만 사용. eager QAT(QuantStub/fusion 수동 배치)는 폐기됐으므로 되살리지 말 것. PT2E 그래프는 `.train()/.eval()`을 못 쓰므로 dvslib 루프에 `set_qat_mode` 훅을 주입해 재사용한다. FPGA 제약(대칭 INT8)은 `get_fpga_quantizer` 한 곳에만 기술.
- `checkpoints_*/`의 옛 INT8 체크포인트는 eager 기반이라 PT2E 경로와 호환되지 않음 (재학습 필요). 옛 FP32 체크포인트(720×960 포함)는 `inference.py --checkpoint <path> --model <name> --roi <HxW>`로 재평가 가능.
- 오차 시각화·방식 비교는 여기 두지 않는다 — `tools/plot_error_vs_frame.py`, `tools/save_max_error_frame.py`, `tools/compare.py` (계산은 `dvslib/eval`).
- 향후 작업: `quantization.py`를 `dvslib/quant`로 이동, `export_mobileone_info.py` PT2E 그래프 대응 (루트 `research_plan.md` Phase 2).

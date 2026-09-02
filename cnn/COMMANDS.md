# cnn — 실행 명령어 모음 (복붙용)

환경: `conda activate dvs_project`. 아래 명령은 모두 `cnn/` 안에서 실행한다고 가정.

```bash
cd cnn
```

## Train

train config는 `train.py`의 argparse 기본값이 곧 config다 (별도 config 파일 없음).

```bash
# baseline (MobileNetV2) + wandb 기록
python train.py --model mobilenet_v2 --epochs 50 --wandb

# MobileOne S0 (FPGA 배포 후보)
python train.py --model mobileone_s0 --epochs 50 --wandb

# 전체 인자 명시 (기본값과 동일 — 바꿀 것만 수정)
python train.py --model mobilenet_v2 \
  --epochs 50 --batch-size 32 --lr 1e-3 \
  --temporal-window 5 --max-frames 3000 --patience 15 --roi 512
```

## Inference / Eval

```bash
# baseline best.pth 평가 (val pixel error / Acc@Npx / FPS)
python inference.py

# 특정 체크포인트 평가
python inference.py --checkpoint runs/mobilenet_v2/mobilenet_v2_best.pth
```

## Visualization (방식 공용 — tools/)

```bash
# 프레임별 오차 그래프 / 최대 오차 프레임 오버레이 (루트에서 실행)
python ../tools/plot_error_vs_frame.py --checkpoint runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
python ../tools/save_max_error_frame.py --checkpoint runs/baseline_mobilenet_v2/mobilenet_v2_best.pth
```

## Analysis (model / FPGA·QAT)

```bash
# 모델 self-test (5채널·512 forward, MobileOne reparam 일치 확인)
python model.py

# 구조·파라미터·메모리 요약 (MobileOne은 reparam 전/후 비교)
python model_summary.py --model mobileone_s0
python model_summary.py --model mobilenet_v2

# INT8 체크포인트 로드 검증 (있을 때)
python model_summary.py --model mobileone_s0 --int8 checkpoints_mobileone_s0_qat/mobileone_s0_int8.pth

# FPGA용 INT8 weight·구조 추출 (mobileone_info/ 에 저장)
python export_mobileone_info.py -all     # npz 레이어 weight 포함 전체
python export_mobileone_info.py -short   # txt 요약만
```

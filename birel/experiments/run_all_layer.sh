#!/usr/bin/env bash
# run_all.sh – Execute autoencoder.py across multiple data‑types

set -e  # exit on first error
NUM_LAYERS=12
# ───────────────────────────────────────────────────────────────
# ① 공통 인자: 공백·개행 유지에 주의하세요
#    (쉘 변수로 묶어서 재사용)
# ───────────────────────────────────────────────────────────────
BASE_ARGS="\

--enc-width [16,16,16] \
--dec-width [16,16,16] \
--trials 5 \
--epochs 400 \
--noise-sched linear \
--noise-start 0.05 \
--noise-end 0.00 \
--dataset wikitext2_2048_opt-125m \
--tau-start 1.0 \
--tau-end 0.01 \
--clip-grad 1.0 \
--scheduler step \
--lr 0.1 \
--step-size 150 \
--warmup-epochs 5 \
--root-dir ../calibration_sets/opt-125m"

# ───────────────────────────────────────────────────────────────
# ② 실행할 data‑type 목록 (띄어쓰기로 구분)
# ───────────────────────────────────────────────────────────────
DATA_TYPES=(key query)


# ───────────────────────────────────────────────────────────────
# ③ 반복 실행 루프 – 각 실행 로그를 별도 파일로 저장
#    * layer/head 값은 필요에 따라 수정
# ───────────────────────────────────────────────────────────────

# ─ 실행 함수 ---------------------------------------------------
exec_cmd () {
  dtype=$1
  layer=$2
  logf="log_${dtype}_L${layer}.txt"

  echo -e "\n===== ${dtype}  layer=${layer} =====" | tee -a "$logf"
  python -u ./autoencoder.py \
      $BASE_ARGS \
      --data-type "$dtype" \
      --layer "$layer"  2>&1 | tee -a "$logf"

  status=${PIPESTATUS[0]}        # python exit code
  if [[ $status -ne 0 ]]; then
      echo "❌  ${dtype} L${layer} failed (exit $status)" | tee -a "$logf"
      exit $status               # 중단하려면 keep, 계속하려면 주석
  fi
}

# ─ 중첩 루프 ---------------------------------------------------
for dtype in "${DATA_TYPES[@]}"; do
  for ((L=0; L<NUM_LAYERS; L++)); do
      exec_cmd "$dtype" "$L"
  done
done

echo "✅  모든 layer × data-type 실행 완료"

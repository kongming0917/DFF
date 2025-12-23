#!/bin/bash

# 실행할 커맨드 목록
#20250510/231729

#./brlv3.py --enc-width [2048, 1024, 512] --dec-width [2048, 1024, 512] --trials 3 # 20250511/105736
# (μ=2.667e-04, σ=2.813e-05)

# 
commands=(
    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400"
    "python3 ./autoencoder.py --enc-width '[2048, 2048, 2048]' --dec-width '[2048, 2048, 2048]' --trials 5 --encoder-noise-prob 0.05 --decoder-noise-prob 0.05  --epochs 400"
    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.01 --decoder-noise-prob 0.01  --epochs 400"
    "python3 ./autoencoder.py --enc-width '[2048, 2048, 2048]' --dec-width '[2048, 2048, 2048]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400"
    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --k-history 1"
    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --connections 'random'"
)

# 사용 가능한 GPU ID들
gpus=(0 1)
num_gpus=${#gpus[@]}

# 현재 실행 중인 PID 및 로그 파일 경로 배열
pids=()
log_files=()

# 로그 디렉토리 준비
log_dir="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# main loop
for cmd in "${commands[@]}"; do
    while true; do
        for i in "${!gpus[@]}"; do
            gpu_id="${gpus[$i]}"
            
            # 해당 GPU가 비어있는 경우
            if [ -z "${pids[$i]}" ] || ! kill -0 "${pids[$i]}" 2>/dev/null; then
                log_file="$log_dir/gpu${gpu_id}_$(date +%s).log"
                echo "[GPU $gpu_id] Launching: $cmd"
                echo "Logging to $log_file"
                
                CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "$cmd" > "$log_file" 2>&1 &
                pids[$i]=$!
                log_files+=("$log_file")
                sleep 1
                break 2
            fi
        done
        sleep 1
    done
done

# 모든 프로세스가 끝날 때까지 대기
for pid in "${pids[@]}"; do
    if [ -n "$pid" ]; then
        wait "$pid"
    fi
done

# 모든 로그 합치기
final_log="${log_dir}/combined_output.log"
echo "🔗 Merging logs into $final_log"
cat "${log_files[@]}" > "$final_log"

echo "✅ All jobs completed. Logs saved in $log_dir"


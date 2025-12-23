# Binary Representation Learning Using Logic Autoencoder

## Update After 6/4/2024
* Found that BIREL is pretty bad for wide distributions
* Added normal2 data set with std of 2
* New strategy: Use Crossbar layer only once at the top for speed. It doesn't seem hurt accuracy.
* Now cuda option is available for LogicLayer. It had a bug before.
* Now accuracy of BIREL is almost similar to best fp8 or slight worse.
* The size of the first crossbar affected accuracy 
* Turned off ternary in voting. It was not intended.
* Turned on tenary in voting and regression again.
* Hidden layer size change: the bottleneck structure
* TreeCrossbar is off-by-default now (It affects accuracy)



## TODO
* Try vector compression (16 X 4 -> 8 X 4 -> 16 X 4)
* Think about vector version of posit
* Model Ensemble to boost last accuracy

### Best Results (commit: 6daf1ab)
```bash 
python3 ./autoencoder.py --enc-width '[16384, 8192, 4096]' --dec-width '[4096, 8192, 16384]' --trials 1 --epochs 1200 --noise-start 0.20 --noise-end 0.00 --tau-start 1.0 --tau-end 0.01 --noise-sched linear --dataset normal --scheduler step --lr 0.1 --step-size 500 --warmup-epochs 5 
```
```
float16_float8_e3m4_float32 MSE : 2.049e-03
float16_float8_e4m3_float32 MSE : 2.594e-03
float16_float8_e5m2_float32 MSE : 9.401e-03
float16_posit_float32 MSE : 1.724e-04
LOGIC 16_8_16 final MSEs : 3.686e-04  (_=3.686e-04, _=0.000e+00)
```

### Typical Results:
```bash
python ./autoencoder.py --enc-width '[16384, 8192, 4096]' --dec-width '[16384, 32768, 65536]' --trials 1 --epochs 1200 --noise-start 0.20 --noise-end 0.00 --tau-start 1.0 --tau-end 0.01 --noise-sched linear --dataset normal --scheduler step --lr 0.1 --step-size 500 --warmup-epochs 5 # 20250617/115415
```
```
float16→float8_e3m4→float32 MSE : 2.049e-03
float16→float8_e4m3→float32 MSE : 2.594e-03
float16→float8_e5m2→float32 MSE : 9.401e-03
float16→posit→float32 MSE : 1.724e-04
LOGIC 16→8→16 final MSEs : 6.513e-04, 2.893e-04, 7.373e-04  (μ=5.593e-04, σ=1.941e-04)
```

#  Improving DiffLogic

## Quick Experiments 
```bash
python main.py -bs 100 -t  30 --dataset mnist -ni 1000 -ef 1_000 -k 16_000 -l 6
```
```python
{'train_acc_eval_mode': 0.9421666484077772, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.9432999789714813, 'test_acc_train_mode': 0.9432999789714813, 'best_acc_test': 0.9432999789714813} 
```
```bash
python main.py -bs 100 -t  30 --dataset mnist -ni 1000 -ef 1_000 -k 16_000 -l 6 -a learned_routing 
```
```python
{'train_acc_eval_mode': 0.9772166341543198, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.9719999778270721, 'test_acc_train_mode': 0.9719999784231186, 'best_acc_test': 0.9719999778270721}
```

### CIFAR-10
```bash
python main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 40_000 -ef 1_000 -k   12_000 -l 4 
```
```python
{'train_acc_eval_mode': 0.4735199869275093, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 't
est_acc_eval_mode': 0.4475999885797501, 'test_acc_train_mode': 0.4475999885797501, 'best_acc_test': 0.44799998909235}
```
(17min)
```
python main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 20_000 -ef 1_000 -k   12_000 -l 4 -a learned_routing
```
(36min)
```python
{'train_acc_eval_mode': 0.5764799864888192, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.4963999897241592, 'test_acc_train_mode': 0.49449998766183856, 'best_acc_test': 0.49929998964071276}
```
```
python main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 20_000 -ef 1_000 -k   12_000 -l 4 -a learned_routing  --use_crossbar_tree
```
(24min)
```python
{'train_acc_eval_mode': 0.553979986846447, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.489099987745285, 'test_acc_train_mode': 0.48849998742341993, 'best_acc_test': 0.4898999872803688}
```


## Quick Full Experiments (<22min on A100)

### Adults Dataset
```bash
python main.py  -bs 100 -t 20 --dataset adult         -ni 100_000 -ef 1_000 -k 256 -l 5 
```
The following results may be a bit better than the paper owing to STE.
```python 
{'train_acc_eval_mode': 0.8460553075304095, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.8434263467788696, 'test_acc_train_mode': 0.8434263467788696, 'best_acc_test': 0.8482072353363037}
```
```bash
python main.py  -bs 100 -t 20 --dataset adult         -ni 100_000 -ef 1_000 -k 256 -l 5  -a learned_routing
```
```python
{'train_acc_eval_mode': 0.8524001076916196, 'train_acc_train_mode': -1, 'valid_acc_eval_mode': -1, 'valid_acc_train_mode': -1, 'test_acc_eval_mode': 0.8466135859489441, 'test_acc_train_mode': 0.8471447825431824, 'best_acc_test': 0.8503320217132568}
```

## TODO
* Ternary voting
* Compare Dropout, BitFlip Dropout, Sweep probabilities


# Pruning
First, train a model with eid (experiment ID) specified.
### Training 
```bash
python main.py -bs 100 -t  30 --dataset mnist -ni 1000 -ef 1_000 -k 16_000 -l 6 -eid 520001 --load_model 
```
Second, load a model with eid you specified when you train, and choose a pruning method and pruning hyperparameters.

```bash
python main.py -bs 100 -t  30 --dataset mnist -ni 1000 -ef 1_000 -k 16_000 -l 6 -eid 520001 --load_model --prune_method retrain --prune_lam_reg 0.00002
```

# TODO
* Re-implement random pruning
* Compare retrain with random pruning in full setting
* More baselines




# Experimental Setup
You can run the following experiments on dolphin, whale, and octopus.

```bash
conda activate difflogic # (or conda activate difflogic_ if you are using octopus) 
cd experiments
```


# Autoencoder
## Baselines

```bash
./autoencoder.py --model mlp --enc-width [16, 16, 16] --dec-width [16, 16, 16] --trials 5 --epochs 400 --tau-start 0.2 --tau-end 0.01 --dataset uniform --scheduler step --lr 0.01 --step-size 150 --clip-grad 1.0 --noise-start 0.0 --noise-end 0.0 --tau-sched linear # 20250524/222905
```

### Results
```
float16→float8_e3m4→float32 MSE : 1.905e-03
float16→float8_e4m3→float32 MSE : 7.590e-04
float16→float8_e5m2→float32 MSE : 3.007e-03
MLP 16→8→16 final MSEs : 1.802e-03, 2.721e-03, 1.723e-03, 1.676e-03, 1.352e-03  (μ=1.855e-03, σ=4.594e-04)
```


```bash
./autoencoder.py --model mlp --enc-width [1024, 1024, 1024] --dec-width [1024, 1024, 1024] --trials 5 --epochs 400 --tau-start 0.2 --tau-end 0.01 --dataset uniform --scheduler step --lr 0.01 --step-size 150 --clip-grad 1.0 --noise-start 0.0 --noise-end 0.0 --tau-sched linear # 20250524/232226
```

### Results
```
float16→float8_e3m4→float32 MSE : 1.905e-03
float16→float8_e4m3→float32 MSE : 7.590e-04
float16→float8_e5m2→float32 MSE : 3.007e-03
MLP 16→8→16 final MSEs : 2.415e-03, 1.834e-03, 1.925e-03, 2.099e-03, 1.143e-03  (μ=1.883e-03, σ=4.199e-04
```

## Our Results

```bash
./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --epochs 400 --noise-start 0.1 --noise-end 0.01 --noise-sched linear --dataset uniform --scheduler manual --lr 0.1  # 20250519/215627
```

The results below are obtained from old manual LR schedule. See the commit cd0daaee72bd382da8d946feeabcc478ed24242b.

### Results
```
#float16→float8_e3m4→float32 MSE : 1.905e-03
#float16→float8_e4m3→float32 MSE : 7.590e-04
#float16→float8_e5m2→float32 MSE : 3.007e-03
#LOGIC 16→8→16 final MSEs : 2.925e-04, 2.701e-04, 1.724e-04, 2.461e-04, 3.071e-04  (μ=2.577e-04, σ=4.735e-05)
```

The following is a similar result to the above using non-manual lr schedule.
```bash
./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --epochs 400 --noise-start 0.1 --noise-end 0.01 --noise-sched linear --dataset uniform --scheduler step --lr 0.1 --step-size 150 --warmup-epochs 5 # 20250519/215627
```

```
LOGIC 16→8→16 final MSEs : 1.783e-04, 5.730e-04, 9.855e-05, 1.745e-04, 3.164e-04  (μ=2.681e-04, σ=1.678e-04)
```

 Without noise injection, a litte worse result can be obtained using tau scheduling. 
Tau anneling was applied to voter only (No logic layer). This was enabled by gradient clipping.
```bash
./autoencoder.py --enc-width [1024, 1024, 1024] --dec-width [1024, 1024, 1024] --trials 5 --noise-sched linear --noise-start 0.0 --noise-end 0.0 --dataset uniform --tau-start 1.0 --tau-end 0.01 --clip-grad 1.0 # 20250525/055457
```

### Results
```
LOGIC 16→8→16 final MSEs : 2.554e-04, 3.971e-04, 4.121e-04, 1.656e-04, 5.242e-04  (μ=3.509e-04, σ=1.260e-04)
```

With noise inject and tau schedule, here is the current best result.
```bash
./autoencoder.py --enc-width [1024, 1024, 1024] --dec-width [1024, 1024, 1024] --trials 5 --noise-sched linear --noise-start 0.05 --noise-end 0.00 --dataset uniform --tau-start 1.0 --tau-end 0.01 --clip-grad 1.0 --scheduler step --lr 0.1 --step-size 150 --warmup-epochs 5 # 20250525/121624
```
### Results
```
LOGIC 16→8→16 final MSEs : 2.106e-04, 3.651e-04, 1.593e-04, 1.818e-04, 2.215e-04  (μ=2.276e-04, σ=7.212e-05)
```

Compared to the uniform distribution, the results for normal are not optimized throughly, but this is the best results at the momenet.
```bash
python3 ./autoencoder.py --enc-width '[2048, 2048, 2048]' --dec-width '[2048, 2048, 2048]' --trials 1 --epochs 400 --noise-start 0.1 --noise-end 0.01 --noise-sched linear --dataset normal # 20250515/065022
```

### Results
```
float16→float8_e3m4→float32 MSE : 2.049e-03
float16→float8_e4m3→float32 MSE : 2.594e-03
float16→float8_e5m2→float32 MSE : 9.401e-03
LOGIC 16→8→16 final MSEs : 1.505e-03  (μ=1.505e-03, σ=0.000e+00)
```

# Differentiable Logic Circuit Search

```bash
python3 diffsyn.py --width "[100,100,200]" --adder_size 4   --steps 500 --lr 0.1
```

To reproduce diffsyn results, log into turtle server, where EDA tools are installed.
```bash
cd circuits/build
../scripts/run_all.sh 4
```

| 4bit Adder | Baseline (DC)   | diffsyn        |
|-----------|------------------|-----------------|
|   Area    |   257            |   221           |
|   WNS     |   0.60           |   0.49          |


```bash
python3 diffsyn.py --width "[1000, 1000, 600]" --adder_size 5   --steps 600
```


To reproduce diffsyn results,
```bash
cd circuits/build
../scripts/run_all.sh 5
```

| 5bit Adder | Baseline (DC)   | diffsyn        |
|-----------|------------------|------------------|
|   Area    |   316            |   308            |
|   WNS     |   0.71           |   0.58           |


It's just one bit difference, but there is a big jump at the moment.

| 6bit Adder | Baseline (DC)   | diffsyn        |
|-----------|------------------|------------------|
|   Area    |   320            |   529            |
|   WNS     |   0.82           |   1.03           |


# Convolutional Differentiable Logic Networks (conv_difflogic.py)

## 일반 학습

기본 모델 학습을 수행합니다.

### CIFAR-10 예시
```bash
python conv_difflogic.py \
  --dataset cifar-10-3-thresholds \
  --model-size M \
  --implementation im2col \
  --eid 700001 \
  --num-iterations 200_000 \
  --eval-freq 2000 \
  --learning-rate 0.02 \  
```

### 주요 옵션
- `--dataset`: 데이터셋 선택 (`cifar-10-3-thresholds`, `mnist`)
- `--model-size`: 모델 크기 (`S`, `M`, `B`, `L`, `G`)
- `--implementation`: 구현 방식 (`triton`, `im2col`)
- `--eid`: 실험 ID (결과 저장용)
- `--num-iterations`: 학습 반복 횟수
- `--batch-size`: 배치 크기
- `--learning-rate`: 학습률
- `--scheduler`: 학습률 스케줄러 (`cosine`, `step`, `none`)



## Retraining

### Setup
```bash
  cd ../difflogic 
  pip install -e .
  cd ../experiments
  cp /hai/home/lsh/birel/experiments/results_conv/baseline.pt results_conv/baseline.pt
```


## Pruning Experiments

### Quick Start

```bash
python pruning_simple.py \
  --retrain-eid results_conv/baseline.pt \
  --dataset cifar-10-3-thresholds \
  --model-size M \
  --prune-method 2nd_MSE \
  --prune-pct 50 \
  --prune-eval-batches 40 \
  --num-iterations 0 \
  --eval-freq 0 \
  --batch-size 4
```

### Prune Methods

| Method | Description | Notes |
|--------|-------------|-------|
| `loss` | Actual loss-based pruning | 각 채널을 0-tie/1-tie로 교체했을 때의 실제 loss 변화 계산 (매우 느림) |
| `2nd_CE` | 2nd order CE approximation | Gradient² 기반 p-space 2차 근사 (CE loss) |
| `2nd_MSE` | 2nd order MSE approximation (no Hutchinson) | Hutchinson 없이 직접 계산하는 Gauss-Newton 근사 (MSE)) |
| `2nd_MSE_hutch` | 2nd order MSE approximation (with Hutchinson) | Hutchinson trick을 사용한 2차 근사 |
| `1st_absolute` | 1st order absolute approximation | 1차 Taylor 근사 (Absolute Loss) |
| `1st_relative` | 1st order relative approximation | 1차 Taylor 근사 (Relative Loss) |
| `weight` | Weight-based pruning | LogicLayer의 weight 확률 기반 |
| `random` | Random pruning | 랜덤 선택 |

### Conv1 Pruning Results

각 method별로 conv1 layer를 50% pruning한 직후의 결과입니다. 40 batch, 

| Method | After Conv1 Pruning |
|--------|---------------------|
| `loss_CE` | 56.66 |
| `loss_MSE` | 60.43 |
| `2nd_CE` | 57.90 |
| `2nd_MSE` | 55.40 |
| `1st_absolute` | 39.54 |
| `1st_relative` | 39.54 |
| `weight` | 53.02 |
| `random` | 39.58 |

### 주요 옵션

- `--prune-method`: Pruning 방법 선택 (위 표 참조)
- `--prune-pct`: Pruning할 채널 비율 (퍼센트, 예: 50 = 50%)
- `--prune-eval-batches`: Pruning 평가에 사용할 배치 수 (기본값: 40)
- `--prune-eval-probes`: Hutchinson trick에 사용할 probe 수 (기본값: 100)
- `--loss-prune-type`: `loss` method에서 사용할 loss 타입 (`mse`, `l1`, `ce`)








### Iterative 모드 (순차적으로 mask 삽입 및 학습)
```bash
python conv_difflogic.py \
  --retrain-eid results_conv/baseline.pt \
  --pruned-eid 700020 \
  --mask-channel-prune \
  --iterative \
  --implementation im2col \
  --dataset cifar-10-3-thresholds \
  --num-iterations 100_000 \
  --eval-freq 2000 \
  --learning-rate 0.02 \
  --prune-pct 50.0 \
  --mask-channel-prune-method weight 
```

### One-shot 모드 (모든 mask를 한번에 삽입 후 학습)
```bash
python conv_difflogic.py \
  --retrain-eid results_conv/baseline.pt \
  --pruned-eid 700021 \
  --mask-channel-prune \
  --implementation im2col \
  --dataset cifar-10-3-thresholds \
  --num-iterations 50000 \
  --eval-freq 1000 \
  --learning-rate 0.02 \
  --prune-pct 50.0 \
  --mask-channel-prune-method random
```


### 주요 옵션
- `--iterative`: Iterative 모드 활성화 (없으면 one-shot 모드). 기본적으로 GS 직전 layer는 포함하지 않습니다.
- `--prune-pct`: 프루닝할 채널 비율 (퍼센트, 예: 50.0 = 50%)
- `--prune-method`: 초기화 방법 (`random`, `weight`, `loss`, 또는 None=학습)



### Current results (weight based pruning)

Baseline accuracy: 0.7157

| Prune % | Pruned Acc | Ops Reduction | Gates Reduction |
|---------|------------|---------------|-----------------|
| 70%     | 0.6705     | 78.16%        | -               | 




## WGS (WeightedGroupSum) Pruning

WeightedGroupSum을 사용한 classifier 프루닝을 수행합니다.

### 기본 WGS Pruning
```bash
python conv_difflogic.py \
  --retrain-eid results_conv/700001.pt \
  --pruned-eid 700030 \
  --implementation im2col \
  --dataset cifar-10-3-thresholds \
  --num-iterations 100000 \
  --eval-freq 2000 \
  --learning-rate 0.02 \
  --wgs-lam-reg 1e-5
```

### 주요 옵션
- `--retrain-eid`: 재학습할 모델 경로 (정수 ID 또는 전체 경로, `.pt` 또는 `.pth` 파일)
- `--pruned-eid`: 재학습된 모델을 저장할 새로운 실험 ID
- `--wgs-lam-reg`: WeightedGroupSum의 L1 정규화 람다 (기본값: 1e-5)
- `--num-iterations`: 각 phase의 학습 반복 횟수 (정규화 과정과 finetune 모두 동일하게 적용)
- `--learning-rate`: Phase 1의 학습률 (Phase 2는 자동으로 `learning_rate * 0.1` 사용)
- `--eval-freq`: 평가 주기

### 참고
  1. **Phase 1 (Pruning)**: WGS 정규화를 사용한 프루닝 단계
     - `--wgs-lam-reg`로 L1 정규화 강도 조절
     - `--learning-rate`로 학습률 설정
     - 중간 모델은 `{pruned_eid}_phase1_wgs_pruned.pt`로 저장
  2. **Phase 2 (Fine-tuning)**: WGS를 freeze하고 최종 fine-tuning 단계
     - WGS 레이어는 freeze되고, 나머지 레이어만 학습
     - 학습률은 자동으로 `learning_rate * 0.1`로 설정
     - 최종 모델은 `{pruned_eid}_final.pt`로 저장

## 모델 압축 (Model Compression)

학습된 모델을 물리적으로 압축하여 dead node를 제거합니다.

### 기본 사용법
```bash
CUDA_VISIBLE_DEVICES=1 python conv_compression.py \
  --implementation im2col \
  --model-size M \
  --dataset cifar-10-3-thresholds \
  --input-model results_conv/10_mask_prune_phase7.pt \
  --output-model results_conv/compressed/10_mask_prune.pt \
  --compression
```

### 주요 옵션
- `--input-model`: 입력 모델 파일 경로 (`.pt` 파일)
- `--output-model`: 출력 모델 파일 경로 (`.pt` 파일)
- `--compression`: 압축 활성화 플래그
- `--model-size`: 모델 크기 (`S`, `M`, `B`, `L`, `G`)
- `--dataset`: 데이터셋 (`cifar-10-3-thresholds`, `mnist`)
- `--implementation`: 구현 방식 (`im2col`, `triton`)
- `--experiment-id`: 실험 ID (선택사항, `--input-model`이 제공되면 불필요)

### 참고
- 압축된 모델은 dead node가 제거되어 더 작고 빠릅니다
- 압축 후 모델의 정확도는 원본과 동일해야 합니다



## Convert to Verilog
압축된 모델을 기반으로 verilog file을 생성합니다.

### 기본 사용법
Example for 2nd order MSE, 20% pruning
```bash
cd birel/experiments
python3 diffsyn_conv_op_pruning_optimized.py \
--model-path birel_data/cifar-10-3-thresholds/M/2nd_ce/cifar-10-3-thresholds_M_p20_20251217_101652_mask_prune_final_compressed.pt \ --module_name 2ndce_20
```

### 옵션
- `--model-path`: 압축된 모델 pt 파일 경로 지정 (/hai/home/birel_data/)
- `--module_name`: verilog 파일명 (Ex. 2ndmse_10, 2ndmse_20, 2ndce_10 / 10, 20은 pruning pct)


## Synthesis 

### setup (use turtle server)
```bash
source /etc/bashrc
cd birel/conv_syn
mkdir build
cd build
```

### 기본 사용법
../run_conv_vivado_syn.sh <합성할 파일명> <합성하고자 하는 Module list> 형태로 실행

```bash
../run_conv_vivado_syn.sh 2ndmse_20 conv_block2 conv_block3 conv_block4 Classifier_block_layer01
```

### 옵션
- `파일명`: Convert to Verilog 단계에서 만든 파일명 입력
- `Module List`: conv_block1, conv_block2, conv_block3, conv_block4, Classifier_block_layer01, Classifier_block




## 모델 벤치마킹 (Model Benchmarking)

학습된 모델의 inference 시간을 측정합니다.

### GPU Inference Time 측정
```bash
python benchmark.py \
  --model-path results_conv/compressed/13_mask_prune.pt \
  --dataset cifar-10-3-thresholds \
  --gpu \
  --model-size M \
  --profile-layers
```

### 주요 옵션
- `--model-path`: 벤치마킹할 모델 파일 경로 (`.pt` 파일)
- `--dataset`: 데이터셋 (`cifar-10-3-thresholds`, `mnist`)
- `--gpu`: GPU inference time 측정 활성화
- `--model-size`: 모델 크기 (`S`, `M`, `B`, `L`, `G`)
- `--profile-layers`: 레이어별 inference time 측정 활성화
- `--batch-size`: 배치 크기 (기본값: 128)
- `--num-iterations`: 측정 반복 횟수 (기본값: 1000)

### 출력
- 전체 모델의 평균 inference time
- `--profile-layers` 옵션 사용 시 각 레이어별 inference time도 출력






# CLAUDE.md — filter experiment

신호 처리 휴리스틱으로 프레임당 레이저 중심 (x, y)를 뽑는 방식. 학습이 없으므로 dvslib에서 **data(bin I/O)와
eval(지표, tools/ 경유)** 만 쓴다. 이 방식은 비교 대상이면서 **GT 원점을 재는 측정 도구**이기도 하다.

## Structure

```
filter/
├── dvs_filter.py   # 필터(EventDensity/SpatialCluster/TimeRange/ROI)·추출기(Mean/Median/Kalman/TemporalAverage) 본체
├── run.py          # 조건(no_filter / spatial_filter) × 추출기 → results/<cond>_<ext>.csv (frame_idx, frame_number, x, y)
├── origin.py       # 정지 레이저 bin에서 원점 후보 표 출력 (GT initial_center 결정용, 사람이 고름)
└── results/        # run.py 출력 CSV — gitignore (baseline 4개는 BASELINE.md에 수치 기록)
```

## Usage

```bash
python filter/run.py --max-frames 3000                        # 4개 CSV (median·kalman × no/spatial), 약 8분 (16 workers)
python filter/run.py --extractors kalman --no-spatial          # 조합 선택
python tools/evaluate.py --pred-csv filter/results/no_filter_kalman.csv --split val --max-frames 3000
python tools/plot_error_vs_frame.py --pred-csv filter/results/no_filter_kalman.csv
python filter/origin.py data/gaussian_large.bin --max-frames 200
```

## Key Rules

- **`dvs_filter.py`의 알고리즘 동작을 바꾸지 말 것.** 데이터셋 GT 원점 (541, 361)은 정지 레이저에서 이 추출기들의
  결과를 보고 수동으로 정한 값 → 알고리즘이 바뀌면 GT 근거가 달라진다. 새 필터/추출기는 **추가**로만.
- 프레임 매칭은 `frame_idx`(0부터). bin 헤더 `frame_number`(88부터)를 GT `frame_idx`에 직접 병합하면 88프레임
  어긋난다 (옛 `evaluate_against_ground_truth.py`의 버그 — 옛 Filter 지표 기록은 무효).
- `run.py`는 필터를 조건당 한 번만 적용해 추출기 간 공유하고, 필터 단계만 multiprocessing으로 나눈다.
  Kalman·TemporalAverage는 순차 상태가 있으므로 추출 단계는 병렬화하지 말 것. 조건마다 추출기 객체를 새로 만든다
  (옛 코드의 `reset()` 잔여 상태 문제 회피).
- 평가·시각화는 여기 두지 않는다 — `tools/` (`--pred-csv`). 지표 정의는 `dvslib/eval/metrics.py`.
- SpatialClusterFilter는 brownian 데이터에서 효과 없음(±0.2px)이면서 비용 지배적(프레임당 ~1.9s 순차). 비교에는 no_filter로 충분.

## Baseline (3000f, blocked val 중심 프레임 1098개)

Kalman **10.62px / Acc@5px 44.2%**, Median 20.31px / 60.5% (SpatialCluster 유무 무관). 상세는 루트 `BASELINE.md`.

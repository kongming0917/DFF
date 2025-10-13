# DVS Filter Framework - 휴리스틱 방식

DVS(Dynamic Vision Sensor) 카메라 데이터를 필터링하고 레이저 빔의 중심점을 추출하는 휴리스틱 기반 프레임워크입니다.

## 🎯 개요

이 프로젝트는 **필터 기반 휴리스틱 방식**으로 DVS 센서 데이터에서 레이저 빔의 중심점을 추출합니다. 
이벤트 밀도 필터링, 공간 클러스터링, 칼만 필터 등 전통적인 신호 처리 기법을 활용합니다.

**장점**:
- ✅ 구현이 간단하고 이해하기 쉬움
- ✅ FPGA 구현에 적합한 낮은 연산량
- ✅ 실시간 처리 가능
- ✅ 학습 데이터 불필요

**한계**:
- ⚠️ 노이즈에 상대적으로 민감
- ⚠️ 복잡한 패턴 인식 어려움
- ⚠️ 수동 파라미터 튜닝 필요

이 방식은 전체 DVS 프로젝트의 **Phase 1: 기초 연구** 단계에 해당합니다.

---

# DVS Filter Framework

A unified framework for filtering Dynamic Vision Sensor (DVS) binary data files with customizable filter chains and real-time visualization.

## 📁 Project Structure

```
dvs/filter_sim/
├── test.py           # Main execution file with unified filtering
├── dvs_filter.py     # Core classes (BinProcessor, Filters)
├── bin_utils.py      # Utility functions for test file creation
├── bin/              # Data directory
│   ├── .gitkeep      # Keep directory in git
│   └── *.bin         # Binary data files (git ignored)
└── README.md         # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib scipy
```

### Basic Usage

1. **Run built-in demos:**
```bash
python test.py
```

2. **Use in your code:**
```python
from test import FilteringConfig, unified_filter, create_predefined_filters

# Quick filtering with predefined filters
config = FilteringConfig(
    filter_config=create_predefined_filters()['basic'],
    bin_file_path='your_data.bin',  # or None for test data
    max_frames=10,
    verbose=True
)

original, filtered = unified_filter(config)
```

## 🔧 Configuration

### Filter Types

- **`basic`**: Event density + spatial clustering
- **`advanced`**: Multi-stage pipeline (time/space/density/laser)
- **`laser`**: Laser pulse detection
- **`denoise`**: Noise reduction focused

### Key Parameters

```python
FilteringConfig(
    filter_config=FilterConfig(...),    # Filter configuration
    bin_file_path='data.bin',            # Input file (None = test data)
    max_frames=100,                      # Limit frames processed
    verbose=True,                        # Show progress logs
    show_step_by_step=True,             # Detailed filter analysis
    visualize=True,                      # Generate plots
    show_stats=True                      # Processing statistics
)
```

## 📊 Supported Data Format

- **File Format**: 2-bit packed binary files
- **Default Resolution**: 960×720 (real data) / 300×200 (test data)
- **Header**: 8-byte frame headers (timestamp + frame_number)
- **Events**: 0=no_event, 1=ON_event, 2=OFF_event

## 🎯 Example Output

```
============================================================
DVS FILTERING: Basic
Description: Event density + spatial filtering
============================================================
📂 Loading file: your_data.bin
   Resolution: 960×720

📖 Reading frames (max: 10)...
   Selected: 10 out of 9446 frames
   Processing: 10 frames

🔧 Configured 2 filters:
   1. EventDensity(min=20, max=1000)
   2. SpatialCluster(r=5.0, min_n=2)

⚡ Applying filters to 10 frames...

📊 PROCESSING RESULTS
────────────────────────────────────────────────────────────
📥 Input frames:       10
📤 Output frames:      8
🗑️  Filtered out:       2
📈 Events before:      15,247
📉 Events after:       10,891
📊 Event reduction:    28.6%
────────────────────────────────────────────────────────────
```

## 🛠️ Custom Filters

Create your own filters by extending the `FrameFilter` class:

```python
from dvs_filter import FrameFilter

class MyCustomFilter(FrameFilter):
    def apply(self, frame) -> bool:
        # Your filtering logic here
        return True  # Keep frame or False to filter out
    
    def get_name(self) -> str:
        return "MyCustomFilter"

# Use in config
custom_config = FilterConfig(
    filters=[MyCustomFilter(), EventDensityFilter(min_events=50)],
    name="Custom Pipeline"
)
```

## 📁 Data Files

Place your DVS binary files in the `bin/` directory. The framework automatically:
- Detects file format and resolution
- Applies configured filters
- Generates visualization and statistics

For testing without data files, the framework generates synthetic test data automatically.

---

## 🔗 관련 프로젝트

이 필터 방식은 전체 DVS 레이저 중심점 감지 프로젝트의 일부입니다:

- **상위 프로젝트**: [dvs/README.md](../README.md) - 전체 연구 개요 및 방법론 비교
- **CNN 방식**: [cnn_sim/README.md](../cnn_sim/README.md) - 딥러닝 기반 접근
- **YOLO 방식**: [yolo_sim/README.md](../yolo_sim/README.md) - 객체 감지 기반 접근

### 성능 비교

- **Filter**: 평균 오차 ~10-20px, 처리 속도 ⚡⚡⚡⚡⚡
- **CNN**: 평균 오차 ~3-5px, 처리 속도 ⚡⚡⚡
- **YOLO**: 평균 오차 ~4-7px, 처리 속도 ⚡⚡

자세한 비교는 [dvs/README.md](../README.md)를 참고하세요.
#!/usr/bin/env python3
"""
DVS CNN 추론 시스템 (간소화 버전)
"""

import torch
import numpy as np
import os
import sys
import time
from typing import List, Tuple, Dict, Any

# 상위 디렉토리를 sys.path에 추가 (lib 모듈 사용을 위해)
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

# 상위 디렉토리 모듈 import
from model import get_model, convert_to_quantized, prepare_qat_model
from lib.bin_processor import BinProcessor
from utils import visualize_predictions
from dataset import DVSBrownianDataset  # Brownian motion 전용

class DVSInference:
    """간소화된 DVS 추론 클래스"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto', use_quantized: bool = False):
        """추론 시스템 초기화
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 사용할 디바이스 ('auto', 'cuda', 'cpu')
            use_quantized: 양자화된 모델 사용 여부 (자동 감지도 가능)
        """
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint not found: {self.checkpoint_path}")
        
        self.use_quantized = use_quantized
        
        if self.use_quantized:
            self.device = torch.device('cpu')
            print("🔧 Using CPU for quantized inference")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
        
        self.model_name = "unknown"
        # 모델 로드
        self.model, self.input_channels = self._load_model()
        
    def _load_model(self):
        """모델 로드"""      
        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        input_channels = checkpoint.get('input_channels', 5)
        self.model_name = checkpoint.get('model_name', None)
        
        print(f"🔍 Input channels: {input_channels}, Model name: {self.model_name}")
        
        # 모델 생성 및 로드 (체크포인트 또는 파일명에서 모델 타입 읽기)
        file_name = os.path.basename(self.checkpoint_path)
        
        # 체크포인트에서 model_name 읽기, 없으면 파일명에서 추론
        self.model_name = checkpoint.get('model_name', None)
        if self.model_name is None:
            # 파일명에서 모델 이름 추론
            if 'mobileone' in file_name.lower() or 's0' in file_name.lower():
                self.model_name = 'mobileone_s0'
                print(f"⚠️ [Warning] 'model_name' not found in checkpoint. Inferred from filename: {self.model_name}")
            else:
                self.model_name = 'mobilenet_v2'
                print(f"⚠️ [Warning] 'model_name' not found in checkpoint. Using default: {self.model_name}")
        else:
            print(f"🔍 Model name: {self.model_name} (from checkpoint)")
        
        # 1. skeleton model (FP32)
        model = get_model(self.model_name, input_channels=input_channels, output_dim=2, use_qat=False)
        
        # 2. Check qat mode
        if self.use_quantized:
            print(f"🎭 Mode: Quantized Inference")
            
            # MobileOne 모델의 경우 reparameterize 수행 (multi-branch -> single-branch)
            if hasattr(model, 'reparameterize'):
                print("🔄 Reparameterizing MobileOne model for inference...")
                model.reparameterize()
                print("✅ Reparameterization complete")

            model.eval()
            model = prepare_qat_model(model)             # QAT 모델 준비
            model = convert_to_quantized(model)          # INT8 모델 변환
        else:
            print(f"🚀 Mode: Standard FP32 Inference")
        
        # 가중치 로드
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ State dict loaded")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not load state_dict: {e}")
        
        model.to(self.device)
        model.eval()
        
        return model, input_channels
    
    def load_frames_from_bin(self, bin_file_path: str, max_frames: int = 50, roi_size: Tuple[int, int] = (512, 512)) -> List[np.ndarray]:
        """bin 파일에서 프레임 로드
        
        Args:
            bin_file_path: bin 파일 경로
            max_frames: 최대 프레임 수
            roi_size: 프레임 크기 (width, height) - Brownian motion은 (512, 512)
        """
        print(f"📖 Loading frames from {bin_file_path}")
        
        # BinProcessor 초기화 (Brownian motion 데이터는 512x512)
        frame_width, frame_height = roi_size
        processor = BinProcessor(frame_width=frame_width, frame_height=frame_height, has_header=True)
        frames = processor.read_frames(bin_file_path, max_frames=max_frames)
        
        # numpy 배열로 변환 및 고정 스케일링 (2bit 데이터: 0,1,2 → 0.0,0.5,1.0)
        individual_frames = []
        
        for frame in frames:
            frame_array = np.array(frame.raw_data, dtype=np.float32)
            
            if not ('mobileone' in self.model_name):
                frame_array = frame_array / 2.0  # 고정 스케일링
            individual_frames.append(frame_array)
        
        print(f"   Loaded {len(individual_frames)} frames")
        return individual_frames
    
    def predict_single(self, input_tensor: torch.Tensor, measure_time: bool = True) -> Tuple[np.ndarray, float]:
        """단일 텐서 추론"""
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)

            inference_time = 0.0
            # 추론 시간 측정
            if measure_time and self.device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = self.model(input_tensor)
                end.record()
                
                torch.cuda.synchronize()
                inference_time = start.elapsed_time(end)
            elif measure_time:
                start = time.time()
                output = self.model(input_tensor)
                inference_time = (time.time() - start) * 1000 # ms 단위
            else:
                output = self.model(input_tensor)
            
            return output.cpu().numpy(), inference_time
    
    def predict_from_frames(self, individual_frames: List[np.ndarray], 
                          csv_labels_path: str,
                          roi_size: Tuple[int, int] = (512, 512),
                          test_augmentation: bool = False) -> Dict[str, Any]:
        """프레임 리스트에서 추론 (Brownian motion 전용)
        
        Args:
            individual_frames: 개별 프레임 리스트
            csv_labels_path: CSV 레이블 파일 경로 (필수)
            roi_size: ROI 크기
            test_augmentation: 테스트 증강 사용 여부 (현재는 미사용)
        """
        print(f"🔮 Creating dataset for inference...")
        
        # Brownian motion 데이터셋 사용
        dataset = DVSBrownianDataset(
            individual_frames=individual_frames,
            csv_labels_path=csv_labels_path,
            roi_size=roi_size,
            temporal_window=self.input_channels
        )
        dataset.set_training_mode(False)  # 추론 모드
        
        print(f"📊 Dataset size: {len(dataset)} samples")
        
        if len(dataset) == 0:
            return {'predictions': [], 'times': [], 'errors': []}
        
        # 추론 실행
        print("🔮 Running inference...")
        predictions = []
        times = []
        errors = []
    
        
        for i in range(len(dataset)):
            sample_input, sample_label = dataset[i]
            
            # 배치 차원 추가
            input_tensor = sample_input.unsqueeze(0)
            
            # 추론
            output, inference_time = self.predict_single(input_tensor)
            
            # 결과 처리
            pred_x, pred_y = output[0]
            true_x, true_y = sample_label.numpy()
            
            predictions.append((pred_x, pred_y))
            times.append(inference_time)
            
            # 오차 계산
            error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            errors.append(error)
            
            if i < 5:  # 처음 5개만 출력
                print(f"   Sample {i}: pred=({pred_x:.3f}, {pred_y:.3f}), true=({true_x:.3f}, {true_y:.3f}), time={inference_time:.2f}ms")
        
        # 실제 타겟 값들 수집
        targets = []
        for i in range(len(dataset)):
            _, sample_label = dataset[i]
            true_x, true_y = sample_label.numpy()
            targets.append((true_x, true_y))
        
        return {
            'predictions': predictions,
            'times': times,
            'errors': errors,
            'mean_time': np.mean(times),
            'fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'targets': targets  # 실제 타겟 값들
        }
    
    def benchmark(self, num_iterations: int = 100, roi_size: Tuple = (512, 512)) -> Dict[str, float]:
        """모델 성능 벤치마크"""
        print(f"🏃 Benchmarking model performance ({num_iterations} iterations)...")
        
        # 더미 입력 생성
        input_shape = (1, self.input_channels, roi_size[0], roi_size[1])
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.predict_single(dummy_input, measure_time=False)
        
        # 실제 측정
        times = []
        for _ in range(num_iterations):
            _, inference_time = self.predict_single(dummy_input, measure_time=True)
            times.append(inference_time)
        
        results = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1000 / np.mean(times)
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   Mean time: {results['mean_time']:.2f}ms")
        print(f"   Std time: {results['std_time']:.2f}ms")
        print(f"   Min time: {results['min_time']:.2f}ms")
        print(f"   Max time: {results['max_time']:.2f}ms")
        print(f"   FPS: {results['fps']:.1f}")
        
        return results

    def visualize_inference_results(self, results: Dict[str, Any], save_path: str = None):
        """추론 결과 시각화"""
        if len(results['predictions']) == 0:
            print("⚠️ No predictions to visualize")
            return
        
        print(f"📊 Creating inference visualization...")
        
        # 데이터 변환
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        # 시각화 생성
        if save_path is None:
            save_path = f"inference_results_{len(predictions)}_samples.png"
        
        visualize_predictions(
            predictions=predictions,
            targets=targets,
            title=f"Inference Results ({len(predictions)} samples)",
            save_path=save_path,
            show_plot=False  # headless 환경 대응
        )
        
        print(f"📊 Inference visualization saved to: {save_path}")

def find_available_models():
    """사용 가능한 모델들 찾기"""
    models = []
    for d in os.listdir('.'):
        if d.startswith('checkpoints_') and os.path.isdir(d):
            config_path = os.path.join(d, 'config.json')
            model_info = {'name': 'Unknown', 'temporal_window': 1, 'max_frames': 0, 'epochs': 0, 'roi_size': [512, 512], 'use_qat': False}
            
            # 디렉토리 이름에서 QAT 여부 확인
            is_qat_from_dir = '_qat' in d
            
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        tc = config.get('training_config', {})
                        model_info = {
                            'name': config.get('model_name', 'Unknown'),
                            'temporal_window': tc.get('temporal_window', 1),
                            'max_frames': tc.get('max_frames', 0),
                            'epochs': tc.get('num_epochs', 0),
                            'roi_size': tc.get('roi_size', [512, 512]),
                            'use_qat': config.get('use_qat', is_qat_from_dir)
                        }
                except: pass
                
            best_file = next((f for f in os.listdir(d) if f.endswith('_best.pth')), None)
            int8_file = next((f for f in os.listdir(d) if f.endswith('_int8.pth')), None)
            
            target_file = None
            
            if int8_file:
                target_file = int8_file
            elif best_file:
                target_file = best_file

            if target_file:
                models.append({
                    'path': os.path.join(d, target_file),
                    'dir': d,
                    **model_info
                })
    return models

def select_model():
    """모델 선택 메뉴"""
    print("🔮 DVS CNN Inference System")
    print("=" * 50)
    
    models = find_available_models()
    if not models:
        print("❌ 사용 가능한 학습된 모델이 없습니다.")
        return None
    
    print(f"\n📋 사용 가능한 모델들 ({len(models)}개):")
    for i, model in enumerate(models, 1):
        # use_qat 값에 따라 표시
        if model['use_qat']:
            mode_str = "INT8/CPU" # QAT면 INT8로 추론
        else:
            mode_str = "FP32/GPU"
            
        print(f" {i:<4} {model['name']:<20} {mode_str:<10} {model['temporal_window']:<8} {model['epochs']:<8}")

    # 사용자 선택
    import sys
    if not sys.stdin.isatty():
        print(f"\n✅ {models[0]['name']} 모델을 사용합니다.")
        return models[0]
    else:
        try:
            choice = input(f"\n모델을 선택하세요 (1-{len(models)}, 기본값 1): ").strip() or "1"
        except (KeyboardInterrupt, EOFError):
            choice = "1"
    
    try:
        idx = int(choice) - 1
        selected = models[idx] if 0 <= idx < len(models) else models[0]
        return selected
    except (ValueError, IndexError):
        return models[0]

def main():
    """메인 추론 테스트"""
    selected_model = select_model()
    if not selected_model:
        return
    
    print(f"\n✅ Selected model: {selected_model['name']} (QAT = {selected_model['use_qat']})")
    try:
        inferencer = DVSInference(
            selected_model['path'],
            use_quantized=selected_model['use_qat']
        )
    except FileNotFoundError:
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {selected_model['path']}")
        return
    
    # 벤치마크
    print(f"\n🏃 {selected_model['name']} 모델 벤치마크...")
    benchmark_results = inferencer.benchmark(num_iterations=50)
    
    # 데이터 로드
    bin_file_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512.bin"
    csv_labels_path = "/hai/home/jdj/dvs/sim/data/gaussian_brownian_512x512_labels.csv"
    
    if os.path.exists(bin_file_path):
        individual_frames = inferencer.load_frames_from_bin(bin_file_path, max_frames=50)
    else:
        print("⚠️ 더미 데이터 사용")
        individual_frames = [np.random.rand(512, 512).astype(np.float32) for _ in range(50)]
    
    if not os.path.exists(csv_labels_path):
        print(f"⚠️ CSV 레이블 파일을 찾을 수 없습니다: {csv_labels_path}")
        print("   추론을 계속하지만 정확한 오차 계산이 불가능합니다.")
        csv_labels_path = None
    
    # 추론 실행
    print(f"\n🔮 {selected_model['name']} 모델 추론...")
    if csv_labels_path and os.path.exists(csv_labels_path):
        results = inferencer.predict_from_frames(individual_frames, csv_labels_path)
    else:
        print("⚠️ CSV 레이블이 없어 추론만 수행합니다 (오차 계산 불가)")
        # CSV 없이도 추론 가능하도록 수정 필요하지만, 일단 경고만 출력
        results = {'predictions': [], 'times': [], 'errors': [], 'mean_time': 0, 'fps': 0, 'mean_error': 0, 'std_error': 0, 'targets': []}
    
    # 결과 출력
    print(f"\n📈 결과: {len(results['predictions'])}개, {results['mean_time']:.2f}ms, {results['fps']:.1f} FPS")
    print(f"   오차: {results['mean_error']:.3f}±{results['std_error']:.3f}")
    
    # 시각화 저장
    output_path = os.path.join(selected_model['dir'], f"{selected_model['name']}_inference_results.png")
    inferencer.visualize_inference_results(results, output_path)
    print(f"\n✅ 완료! 결과: {output_path}")

if __name__ == "__main__":
    main()
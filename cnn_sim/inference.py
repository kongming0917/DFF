#!/usr/bin/env python3
"""
DVS CNN 추론 시스템 (간소화 버전)
"""

import torch
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Any

# 상위 디렉토리 모듈 import
sys.path.append('/hai/home/jdj/dvs/filter_sim')

from model import get_model
from dataset import DVSFixedGTDataset
from dvs_filter import BinProcessor
from utils import visualize_predictions

class DVSInference:
    """간소화된 DVS 추론 클래스"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """추론 시스템 초기화"""
        self.checkpoint_path = checkpoint_path
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 로드
        self.model, self.input_channels = self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint not found: {self.checkpoint_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 첫 번째 conv 레이어에서 입력 채널 수 추출 (다양한 모델 지원)
        input_channels = None
        conv_patterns = ['conv1.weight', 'features.0.weight', 'conv.weight', 'conv1.conv.weight']
        
        for pattern in conv_patterns:
            for key in checkpoint['model_state_dict'].keys():
                if pattern in key:
                    input_channels = checkpoint['model_state_dict'][key].shape[1]
                    print(f"🔍 Found input channels: {input_channels} (from {key})")
                    break
            if input_channels:
                break
        
        if input_channels is None:
            # 기본값 사용 (temporal_window=5로 추정)
            input_channels = 5
            print(f"⚠️ Could not detect input channels, using default: {input_channels}")
        
        # 모델 생성 및 로드 (config.json에서 모델 타입 읽기)
        model_name = 'basic'  # 기본값
        config_path = os.path.join(os.path.dirname(self.checkpoint_path), 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('model_name', 'basic')
            except:
                pass
        
        print(f"🔧 Using model type: {model_name}")
        model = get_model(model_name, input_channels=input_channels, output_dim=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded from {self.checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Best Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        return model, input_channels
    
    def load_frames_from_bin(self, bin_file_path: str, max_frames: int = 50) -> List[np.ndarray]:
        """bin 파일에서 프레임 로드"""
        print(f"📖 Loading frames from {bin_file_path}")
        
        # BinProcessor 초기화
        processor = BinProcessor(frame_width=960, frame_height=720, has_header=True)
        frames = processor.read_frames(bin_file_path, max_frames=max_frames)
        
        # numpy 배열로 변환 및 정규화
        individual_frames = []
        for frame in frames:
            frame_array = np.array(frame.raw_data, dtype=np.float32)
            if np.max(frame_array) > 0:
                frame_array = frame_array / np.max(frame_array)  # 0-1 정규화
            individual_frames.append(frame_array)
        
        print(f"   Loaded {len(individual_frames)} frames")
        return individual_frames
    
    def predict_single(self, input_tensor: torch.Tensor, measure_time: bool = True) -> Tuple[np.ndarray, float]:
        """단일 텐서 추론"""
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            
            # 추론 시간 측정
            if measure_time and self.device.type == 'cuda':
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                output = self.model(input_tensor)
                
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
            else:
                output = self.model(input_tensor)
                inference_time = 0.0
            
            return output.cpu().numpy(), inference_time
    
    def predict_from_frames(self, individual_frames: List[np.ndarray], 
                          roi_size: Tuple[int, int] = (512, 512),
                          test_augmentation: bool = False) -> Dict[str, Any]:
        """프레임 리스트에서 추론"""
        print(f"🔮 Creating dataset for inference...")
        
        # 증강 설정 결정
        if test_augmentation:
            shift_x, shift_y = 80, 60  # 학습 시와 동일한 증강 범위
            print(f"🔄 Test augmentation enabled: shift_x=±{shift_x}, shift_y=±{shift_y}")
        else:
            shift_x, shift_y = 0, 0  # 증강 없음 (성능 측정용)
            print(f"🎯 No augmentation: fixed center prediction")
        
        # 데이터셋 생성 (추론 모드)
        dataset = DVSFixedGTDataset(
            individual_frames=individual_frames,
            true_center_coord=(541, 360),  # 실제 빔 중심
            roi_size=roi_size,
            temporal_window=self.input_channels,  # 모델의 입력 채널 수 사용
            shift_range_x=shift_x,
            shift_range_y=shift_y
        )
        dataset.set_training_mode(test_augmentation)  # 증강 여부에 따라 모드 설정
        
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
            best_file = next((f for f in os.listdir(d) if f.endswith('_best.pth')), None)
            if best_file:
                config_path = os.path.join(d, 'config.json')
                model_info = {'name': 'Unknown', 'temporal_window': 1, 'max_frames': 0, 'epochs': 0, 'roi_size': [512, 512]}
                
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
                                'roi_size': tc.get('roi_size', [512, 512])
                            }
                    except:
                        pass
                
                models.append({
                    'path': os.path.join(d, best_file),
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
        print(f"{i}. {model['name']} 모델 (윈도우:{model['temporal_window']}, 프레임:{model['max_frames']}, 에폭:{model['epochs']})")
    
    # 사용자 선택
    import sys
    if not sys.stdin.isatty():
        choice = "1"
        print(f"\n자동 선택: 1")
    else:
        try:
            choice = input(f"\n모델을 선택하세요 (1-{len(models)}, 기본값 1): ").strip() or "1"
        except (KeyboardInterrupt, EOFError):
            choice = "1"
    
    try:
        idx = int(choice) - 1
        selected = models[idx] if 0 <= idx < len(models) else models[0]
        print(f"\n✅ {selected['name']} 모델을 선택했습니다.")
        return selected
    except (ValueError, IndexError):
        print(f"\n✅ {models[0]['name']} 모델을 사용합니다.")
        return models[0]

def main():
    """메인 추론 테스트"""
    selected_model = select_model()
    if not selected_model:
        return
    
    try:
        inferencer = DVSInference(selected_model['path'])
    except FileNotFoundError:
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {selected_model['path']}")
        return
    
    # 벤치마크
    print(f"\n🏃 {selected_model['name']} 모델 벤치마크...")
    benchmark_results = inferencer.benchmark(num_iterations=50)
    
    # 데이터 로드
    bin_file_path = "/hai/home/jdj/dvs/data/gaussian_large.bin"
    if os.path.exists(bin_file_path):
        individual_frames = inferencer.load_frames_from_bin(bin_file_path, max_frames=50)
    else:
        print("⚠️ 더미 데이터 사용")
        individual_frames = [np.random.rand(720, 960).astype(np.float32) for _ in range(50)]
    
    # 추론 실행
    print(f"\n🔮 {selected_model['name']} 모델 추론...")
    results = inferencer.predict_from_frames(individual_frames)
    
    # 결과 출력
    print(f"\n📈 결과: {len(results['predictions'])}개, {results['mean_time']:.2f}ms, {results['fps']:.1f} FPS")
    print(f"   오차: {results['mean_error']:.3f}±{results['std_error']:.3f}")
    
    # 시각화 저장
    output_path = os.path.join(selected_model['dir'], f"{selected_model['name']}_inference_results.png")
    inferencer.visualize_inference_results(results, output_path)
    print(f"\n✅ 완료! 결과: {output_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DVS 프로젝트 환경 테스트 스크립트
"""
import os
import sys

def test_basic_imports():
    """기본 라이브러리 import 테스트"""
    print("🔍 기본 라이브러리 테스트...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🚀 CUDA: ✅ (GPU: {torch.cuda.device_count()}개)")
        else:
            print("   🚀 CUDA: ❌ (CPU 모드)")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
        
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    return True

def test_dvs_modules():
    """DVS 프로젝트 모듈 테스트"""
    print("\n🎯 DVS 프로젝트 모듈 테스트...")
    
    import sys
    import os
    
    # 경로 추가
    cnn_sim_path = os.path.join(os.path.dirname(__file__), 'cnn_sim')
    filter_sim_path = os.path.join(os.path.dirname(__file__), 'filter_sim')
    
    sys.path.insert(0, cnn_sim_path)
    sys.path.insert(0, filter_sim_path)
    
    # cnn_sim 테스트
    try:
        from model import get_model, BasicCNN
        print("✅ cnn_sim.model: 정상")
    except Exception as e:
        print(f"❌ cnn_sim.model: {e}")
        return False
        
    try:
        from dataset import DVSFixedGTDataset
        print("✅ cnn_sim.dataset: 정상")
    except Exception as e:
        print(f"❌ cnn_sim.dataset: {e}")
        return False
    
    # filter_sim 테스트
    try:
        from dvs_filter import BinProcessor
        print("✅ filter_sim.dvs_filter: 정상")
    except Exception as e:
        print(f"❌ filter_sim.dvs_filter: {e}")
        return False
    
    return True

def test_model_creation():
    """모델 생성 테스트"""
    print("\n🧠 모델 생성 테스트...")
    
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cnn_sim'))
        
        from model import get_model, count_parameters
        import torch
        
        # 모델 생성
        model = get_model('basic', input_channels=1, output_dim=2)
        params = count_parameters(model)
        
        print(f"✅ BasicCNN 모델 생성 성공")
        print(f"   파라미터 수: {params['total']:,}")
        
        # 테스트 입력
        test_input = torch.randn(1, 1, 400, 400)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   입력 크기: {test_input.shape}")
        print(f"   출력 크기: {output.shape}")
        print(f"   출력 범위: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🐍 Python 버전:", __import__('sys').version)
    print("📂 작업 디렉토리:", os.getcwd())
    print("=" * 60)
    
    # 테스트 실행
    step1 = test_basic_imports()
    step2 = test_dvs_modules() if step1 else False
    step3 = test_model_creation() if step2 else False
    
    print("\n" + "=" * 60)
    if step1 and step2 and step3:
        print("🎉 모든 테스트 통과! DVS 프로젝트 실행 준비 완료!")
        print("\n📋 다음 단계:")
        print("   1. cd cnn_sim")
        print("   2. python train.py")
    else:
        print("⚠️ 일부 테스트 실패. 환경을 다시 확인해주세요.")

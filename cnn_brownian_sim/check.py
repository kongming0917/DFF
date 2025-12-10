# 검증용 코드 (model.py나 별도 스크립트에서 실행)
import torch

# 1. 저장된 파일 로드
checkpoint = torch.load("checkpoints_mobileone_s0_qat/mobileone_s0_int8.pth", weights_only=False)
state_dict = checkpoint['model_state_dict']
    
for key, value in state_dict.items():
    # 가중치(weight)이면서 양자화된 텐서인지 확인
    if 'weight' in key and value.is_quantized:
        print(f"\n🔍 Layer: {key}")
        print(f"   Type: {value.dtype}")
        print(f"   Scheme: {value.qscheme()}")  # 양자화 방식 확인
        
        # 실제 정수값(int8) 일부 확인
        int_values = value.int_repr().flatten()
        print(f"   Int8 Values (first 5): {int_values[:5].tolist()}")
        
        # 양자화 방식(Scheme)에 따라 다르게 스케일 출력
        if value.qscheme() == torch.per_channel_symmetric or value.qscheme() == torch.per_channel_affine:
            # Per-Channel인 경우 (주로 Conv 레이어)
            scales = value.q_per_channel_scales()
            print(f"   Scales (Per-Channel): {scales[:5].tolist()} ... (Total {len(scales)} channels)")
        else:
            # Per-Tensor인 경우 (주로 Linear 레이어)
            print(f"   Scale (Per-Tensor): {value.q_scale()}")
            print(f"   Zero Point: {value.q_zero_point()}")
        
        break

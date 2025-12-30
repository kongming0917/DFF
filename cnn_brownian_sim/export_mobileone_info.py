#!/usr/bin/env python3
"""
FPGA 구현을 위한 INT8 모델 weight 및 구조 추출 스크립트 (수정됨)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import math
import re
import argparse
# 경로 추가
dvs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dvs_root not in sys.path:
    sys.path.insert(0, dvs_root)

from model import get_model, convert_to_quantized, prepare_qat_model

def load_int8_model(checkpoint_path: str):
    """
    _int8.pth 파일을 올바른 순서로 로드하여 PyTorch 모델 객체로 반환
    """
    print(f"📦 Loading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # 1. 메타데이터 로드 (CPU)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"❌ File not found: {checkpoint_path}")
        sys.exit(1)

    model_name = checkpoint.get('model_name', 'mobileone_s0')
    input_channels = checkpoint.get('input_channels', 5)
    output_dim = checkpoint.get('output_dim', 2)
    
    print(f"   Model: {model_name}")
    print(f"   Input: {input_channels} channels")
    
    # 2. 뼈대 모델 생성 (FP32)
    model = get_model(model_name, input_channels=input_channels, output_dim=output_dim)
    model.to('cpu') # 변환을 위해 CPU 대기

    # 3. 변환 파이프라인 (Single-branch 구조 만들기)
    print("🔄 Pipeline: [Reparameterize] -> [QAT Prepare] -> [Convert to INT8] -> [Load Weights]")
    
    # 3-1. 구조 통합 (MobileOne의 핵심: Multi-branch -> Single-branch)
    if hasattr(model, 'reparameterize'):
        model.reparameterize()
        print("   ✅ Reparameterized (Single-branch structure created)")
    
    # 3-2. QAT 준비
    model.eval()
    model = prepare_qat_model(model)
    
    # 3-3. INT8 구조로 변환
    quantized_model = convert_to_quantized(model)
    print("   ✅ Converted to QuantizedConv2d structure")

    # 4. 가중치 로드
    # strict=True로 해서 구조가 완벽히 일치하는지 확인
    quantized_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("   ✅ INT8 Weights loaded successfully")
    
    return quantized_model, input_channels, output_dim

def generate_model_info(model, input_size=(1, 5, 512, 512), output_dir="fpga_export"):
    """
    FPGA 구현을 위한 상세 모델 구조 요약표 생성
    - 각 레이어의 입출력 Shape 계산
    - Depthwise/Pointwise 구분
    - Scale/ZeroPoint 흐름 추적
    """
    print("\n📐 Generating FPGA Architecture Summary...")
    
    summary_path = os.path.join(output_dir, "model_info_2.txt")
    
    # PyTorch Quantized Modules
    import torch.nn.quantized.modules as qmodules
    
    # 입력 초기화
    current_c = input_size[1]
    current_h = input_size[2]
    current_w = input_size[3]
    
    # 리포트 헤더
    header = (
        f"{'ID':<4} | {'Layer Name':<30} | {'Type':<15} | "
        f"{'Input Shape':<15} | {'Output Shape':<15} | "
        f"{'K/S/P':<10} | {'Out Scale (Shift)':<18} | {'Params':<8}"
    )
    separator = "-" * len(header)
    
    lines = []
    lines.append(f"Model Architecture for FPGA Implementation")
    lines.append(f"Input Image: {input_size}")
    lines.append(separator)
    lines.append(header)
    lines.append(separator)
    
    layer_idx = 0
    total_params = 0
    
    # FPGA 데이터 흐름 추적을 위한 리스트
    fpga_layers = []

    for name, module in model.named_modules():
        is_conv = isinstance(module, (qmodules.Conv2d, torch.nn.quantized.modules.Conv2d))
        is_linear = isinstance(module, (qmodules.Linear, torch.nn.quantized.modules.Linear))
        
        if not (is_conv or is_linear):
            continue
            
        layer_idx += 1
        
        # 1. 파라미터 수 및 타입 분석
        weight, bias = module._weight_bias()
        num_params = weight.numel() + (bias.numel() if bias is not None else 0)
        total_params += num_params
        
        # 2. Shape 계산 및 타입 상세화
        type_str = "Unknown"
        k_s_p = "-"
        in_shape_str = "-"
        out_shape_str = "-"
        
        if is_conv:
            k = module.kernel_size
            s = module.stride
            p = module.padding
            g = module.groups
            
            # Conv 타입 구분 (FPGA 구현 시 중요)
            if g == module.in_channels and g == module.out_channels:
                type_str = "DW Conv2d" # Depthwise
            elif k == (1, 1):
                type_str = "PW Conv2d" # Pointwise
            else:
                type_str = "Std Conv2d" # Standard
                
            k_s_p = f"{k[0]}/{s[0]}/{p[0]}"
            
            in_shape_str = f"({current_c}, {current_h}, {current_w})"
            
            # 출력 Shape 계산
            current_h = math.floor((current_h + 2 * p[0] - k[0]) / s[0] + 1)
            current_w = math.floor((current_w + 2 * p[1] - k[1]) / s[1] + 1)
            current_c = module.out_channels
            
            out_shape_str = f"({current_c}, {current_h}, {current_w})"
            
        elif is_linear:
            type_str = "Linear (FC)"
            in_shape_str = f"({module.in_features})"
            out_shape_str = f"({module.out_features})"
            current_c = module.out_features # Flattened assumed
            
        # 3. Quantization Params
        scale_val = module.scale
        # shift 계산: scale = 2^(-shift) -> shift = -log2(scale)
        shift = round(-math.log2(scale_val)) if scale_val > 0 else 0
        out_scale = f"{scale_val:.6f} ({shift})"
        
        # 라인 생성
        line = (
            f"{layer_idx:<4} | {name:<30} | {type_str:<15} | "
            f"{in_shape_str:<15} | {out_shape_str:<15} | "
            f"{k_s_p:<10} | {out_scale:<18} | {num_params:<8}"
        )
        lines.append(line)
        
        # FPGA export용 구조체에 저장
        fpga_layers.append({
            'id': layer_idx,
            'name': name,
            'type': type_str,
            'in_shape': [module.in_channels, int(input_size[2]), int(input_size[3])] if is_conv else [module.in_features], # Note: H/W is tricky to track perfectly in dict without full graph pass
            'out_shape': [current_c, current_h, current_w] if is_conv else [module.out_features],
            'scale': float(module.scale),
            'zp': int(module.zero_point)
        })

    lines.append(separator)
    lines.append(f"Total Parameters: {total_params:,}")
    lines.append(f"Note: 'DW' = Depthwise (spatial), 'PW' = Pointwise (1x1 channel mix)")
    
    # 파일 저장
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f"\n📝 Detailed architecture saved to: {summary_path}")
    return fpga_layers

def save_compact_json(data, file_path):
    """
    JSON을 저장하되, 숫자 리스트([1, 2, 3])는 한 줄로 표현하여 길이를 줄임
    """
    # 1. 일단 기본 들여쓰기로 문자열 변환
    json_str = json.dumps(data, indent=2)
    
    # 2. 정규표현식으로 숫자 리스트 패턴을 찾아서 한 줄로 압축
    # 패턴 설명: '[' 뒤에 숫자, 콤마, 공백, 줄바꿈만 있다가 ']'로 끝나는 구간
    def collapse_list(match):
        # 매칭된 문자열에서 숫자만 추출하여 한 줄 리스트로 재구성
        content = match.group(1)
        # 줄바꿈과 공백을 제거하고 콤마로 다시 연결
        compact_content = ' '.join(content.split())
        return f"[{compact_content}]"

    # 정수/실수 리스트 압축 (예: shape, stride, padding)
    json_str = re.sub(r'\[\s*([\d\s,\.-]+)\s*\]', collapse_list, json_str)
    
    # 3. 파일 저장
    with open(file_path, 'w') as f:
        f.write(json_str)

def extract_weights_and_structure(model, model_name, output_dir="fpga_export"):
    """
    양자화된 모델에서 FPGA 구현에 필요한 정보 추출
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_structure = []
    weights_data = {}
    
    print("\n📊 Extracting layer information for FPGA...")
    
    # PyTorch의 Quantized 모듈 클래스들
    import torch.nn.quantized.modules as qmodules
    
    layer_idx = 0
    current_input_scale = 1.0
    
    # 모델의 named_modules를 순회하며 정보 추출
    for name, module in model.named_modules():
        # 우리가 관심 있는 연산 레이어만 필터링
        is_conv = isinstance(module, (qmodules.Conv2d, torch.nn.quantized.modules.Conv2d))
        is_linear = isinstance(module, (qmodules.Linear, torch.nn.quantized.modules.Linear))
        
        if not (is_conv or is_linear):
            continue
            
        try:
            layer_idx += 1
            print(f"   Layer {layer_idx}: {name} ({type(module).__name__})")
            
            # 1. Weight 추출 (PackedParams에서 언패킹)
            # scale, zero_point도 함께 들어있음
            if is_conv:
                # Conv2d는 (weight, bias) 튜플 혹은 PackedObj 반환
                weight, bias = module._weight_bias() 
            else:
                # Linear도 비슷함
                weight, bias = module._weight_bias()

            # 2. Weight 세부 정보 추출
            # int_repr(): 정수값(int8) 그대로 추출
            weight_int8 = weight.int_repr().numpy().astype(np.int8)
            
            # Scale & Zero Point 추출
            if weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine):
                # Per-channel
                w_scales = weight.q_per_channel_scales().numpy().astype(np.float32)
                w_zps = weight.q_per_channel_zero_points().numpy().astype(np.int32)
                is_per_channel = True
            else:
                # Per-tensor
                w_scales = np.array([weight.q_scale()], dtype=np.float32)
                w_zps = np.array([weight.q_zero_point()], dtype=np.int32)
                is_per_channel = False
                
            # Bias (FP32 -> INT32)
            bias_int32 = None
            if bias is not None:
                bias_fp32 = bias.detach().numpy().astype(np.float32)
                
                # Bias Scale = Input Scale * Weight Scale
                # (Broadcasting: scalar * array)
                bias_scale = current_input_scale * w_scales
                
                # Bias Quantization: Round(Bias_fp32 / Bias_scale)
                # 0으로 나누기 방지 (safety eps)
                bias_scale = np.maximum(bias_scale, 1e-9)
                bias_int32 = np.round(bias_fp32 / bias_scale).astype(np.int32)
            
            # 3. Activation Scale/ZeroPoint (Input/Output scale)
            # 보통 Conv/Linear 자체에는 output scale 정보가 scale 속성으로 저장됨
            output_scale = np.float32(module.scale)
            output_zp = np.int32(module.zero_point)

            # 4. 구조 정보 저장
            layer_info = {
                'id': layer_idx,
                'name': name,
                'type': 'Conv2d' if is_conv else 'Linear',
                'weight': {
                    'shape': list(weight_int8.shape),
                    'dtype': 'int8',
                    'is_per_channel': is_per_channel
                },
                'quantization': {
                    'output_scale': float(output_scale),
                    'output_zero_point': int(output_zp)
                }
            }
            
            if is_conv:
                layer_info.update({
                    'kernel_size': list(module.kernel_size),
                    'stride': list(module.stride),
                    'padding': list(module.padding),
                    'groups': module.groups,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels
                })
            elif is_linear:
                layer_info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features
                })
                
            model_structure.append(layer_info)
            
            # 5. 데이터 딕셔너리 저장 (numpy 포맷)
            # FPGA에서는 Weight(int8)와 Bias(fp32), 그리고 Re-quantization을 위한 Scale(fp32)이 필요함
            layer_data = {
                'weight': weight_int8,      # [Out, In, K, K] (int8)
                'w_scale': w_scales,        # [Out] (float32)
                'w_zp': w_zps,              # [Out] (int32)
                'out_scale': output_scale,  # Scalar (float32)
                'out_zp': output_zp         # Scalar (int32)
            }
            
            if bias_int32 is not None:
                layer_data['bias'] = bias_int32 # [Out] (int32)
                
            weights_data[name] = layer_data
            current_input_scale = output_scale
            
        except Exception as e:
            print(f"⚠️ Error extracting {name}: {e}")
            import traceback
            traceback.print_exc()

    # 파일 저장
    # 1. 구조 JSON
    json_path = os.path.join(output_dir, f"{model_name}_structure_2.json")
    save_compact_json(model_structure, json_path)
    
    # 2. 개별 레이어 데이터 (npz)
    weights_dir = os.path.join(output_dir, "layers")
    os.makedirs(weights_dir, exist_ok=True)
    
    for name, data in weights_data.items():
        # 파일명 안전하게 변경 (특수문자 제거)
        safe_name = name.replace('.', '_')
        file_path = os.path.join(weights_dir, f"{safe_name}.npz")
        np.savez(file_path, **data)
        
    print(f"\n💾 Saved structure to: {json_path}")
    print(f"💾 Saved layer weights to: {weights_dir}/")
    
    return model_structure

def save_npz_structure_info(weights_dir, layer_names=None, output_file=None):
    """
    레이어들의 npz 파일 구조를 설명하는 txt 파일 생성
    layer_names가 None이면 weights_dir의 모든 npz 파일을 처리
    """
    lines = ["NPZ File Structure Information", "=" * 60, ""]
    
    if layer_names is None:
        # 모든 npz 파일 찾기
        if not os.path.exists(weights_dir):
            lines.append("⚠️ layers 디렉토리가 없습니다.")
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(lines))
            return
        
        npz_files = sorted([f for f in os.listdir(weights_dir) if f.endswith('.npz')])
        if not npz_files:
            lines.append("⚠️ npz 파일이 없습니다.")
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(lines))
            return
        
        # 파일명에서 레이어 이름 추출 (backbone_linear_1.npz -> backbone_linear_1)
        layer_names = [f.replace('.npz', '') for f in npz_files]
    
    for layer_name in layer_names:
        safe_name = layer_name.replace('.', '_')
        npz_path = os.path.join(weights_dir, f"{safe_name}.npz")
        
        if not os.path.exists(npz_path):
            lines.extend([f"⚠️ {safe_name}.npz: File not found", ""])
            continue
        
        data = np.load(npz_path)
        lines.extend([f"File: {safe_name}.npz", f"  Number of .npy files: {len(data.keys())}", ""])
        
        for key in sorted(data.keys()):
            arr = data[key]
            shape_str = ' x '.join(map(str, arr.shape)) if arr.shape else 'scalar'
            precision = str(arr.dtype)
            lines.extend([f"  {key}.npy:", f"    Dimension: {shape_str}", f"    Precision: {precision}"])
        
        lines.append("")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"💾 NPZ structure info saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='FPGA 구현을 위한 INT8 모델 정보 추출')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-all', action='store_true', help='모든 파일 저장 (layer npz 포함)')
    group.add_argument('-short', action='store_true', help='txt 파일만 저장 (model_summary, npz_structure_info)')
    args = parser.parse_args()
    
    # [설정] 추출할 int8 모델 경로
    int8_model_path = "checkpoints_mobileone_s0_qat/mobileone_s0_int8.pth"
    
    if not os.path.exists(int8_model_path):
        print(f"❌ Model file not found: {int8_model_path}")
        print("   Please train the model in QAT mode first.")
        sys.exit(1)
        
    # 1. 모델 로드 (Single-branch, INT8 구조)
    model, _, _ = load_int8_model(int8_model_path)

    output_dir = "mobileone_info"
    
    # 2. 모델 정보 txt 파일 생성 (항상 실행)
    generate_model_info(model, (1, 5, 720, 960), output_dir)
    
    if args.all:
        # 모든 파일 저장 (npz 포함)
        extract_weights_and_structure(model, "mobileone_s0", output_dir)
        
        # NPZ 구조 정보 저장 (모든 레이어)
        weights_dir = os.path.join(output_dir, "layers")
        npz_info_path = os.path.join(output_dir, "npz_structure_info.txt")
        save_npz_structure_info(weights_dir, layer_names=None, output_file=npz_info_path)
    elif args.short:
        # txt 파일만 저장
        # 기존 npz 파일이 있다면 그 구조 정보만 저장
        weights_dir = os.path.join(output_dir, "layers")
        if os.path.exists(weights_dir):
            npz_info_path = os.path.join(output_dir, "npz_structure_info.txt")
            save_npz_structure_info(weights_dir, layer_names=None, output_file=npz_info_path)
        else:
            print("⚠️ layers 디렉토리가 없습니다. npz_structure_info.txt를 생성할 수 없습니다.")
    
    print("\n✅ Extraction Finished!")
    print("   Now you can use these files for FPGA implementation.")

if __name__ == "__main__":
    main()
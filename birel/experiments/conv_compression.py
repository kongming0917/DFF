#!/usr/bin/env python3
import argparse
import torch
import sys
import json
from conv_difflogic import eval as eval_fn
# WeightedGroupSum, MaskedGroupSumを difflogic에서 가져옵니다.
from difflogic import GroupSum, LogicLayer, FusedLogicTreeBlock, WeightedGroupSum, PrunedGroupSum, PrunedWeightedGroupSum, MaskedGroupSum
from difflogic.functional import bin_op_s
from birel.model import *
from birel.pruning import *
from birel.conv import *
from birel.utils import *
from birel.pruning import BinaryMask, CopyMask, GroupScale

import random
import numpy as np
import torch.nn as nn
from torch.nn import Flatten
import copy
import time
import math
import uci_datasets
import mnist_dataset
import torchvision
from conv_difflogic import ORPool2d, get_model
import torch.nn.functional as F
import os
from birel.conv import *
from birel.model import *
from collections import OrderedDict

# ───────── utils ──────────
def _get(cfg: dict, key: str, default=None):
    """json 최상위 또는 cfg['args'] 에서 키를 찾는다."""
    return cfg.get(key, cfg.get("args", {}).get(key, default))

def parse_int_list(txt: str | list[int] | int, default: list[int]):
    """입력을 정수 리스트로 변환한다."""
    if isinstance(txt, list):
        return txt
    if isinstance(txt, int):
        return [txt]
    try:
        return [int(x) for x in str(txt).replace(' ', '').split(',') if x]
    except (ValueError, AttributeError):
        return default      # fallback





################################## Load all dataset. No batching!!!! ############################################
def load_dataset(args):
    if 'cifar-10' in args.dataset:
        if args.model_size in ['S', 'M']: # 2-bit
            def custom_transform(x):
                outputs = [(x > (i + 1) / 4.0).float() for i in range(3)]
                return torch.cat([(x > (i + 1) / 4.0).float() for i in range(3)], dim=0)
            final_channels = 3 * 3
        else: # 5-bit
            def custom_transform(x):
                return torch.cat([(x > (i + 1) / 32.0).float() for i in range(31)], dim=0)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ]) 

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=False, download=True, transform=test_transforms)

    elif 'mnist' in args.dataset:
        def custom_transform(x):
            return (x > 0.5).float()
        final_channels = 1
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        test_set = torchvision.datasets.MNIST(root='./data-mnist', train=False, download=True, transform=test_transforms)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported for evaluation.")
        
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return test_loader, final_channels


def build_model(ds, ks):
    """
    num_neurons와 num_layers를 받아 모델을 생성합니다.
    모든 LogicLayer는 num_neurons 크기의 출력 차원을 가집니다.
    """
    idim = input_dim_of_dataset(ds)
    
    # num_neurons 값을 num_layers 개수만큼 복사하여 ks 리스트 생성
    
    # ks 리스트를 사용하여 모델 레이어 구성
    layers = [torch.nn.Flatten(), LogicLayer(idim, ks[0], connections='unique')]
    for in_dim, out_dim in zip(ks[:-1], ks[1:]):
        layers.append(LogicLayer(in_dim, out_dim, connections='unique'))
        
    layers.append(GroupSum(num_classes_of_dataset(ds), tau=10))
    
    return torch.nn.Sequential(*layers)



# ───────── Pruned Model Builder (ks 사용) ────────
def build_pruned_model(ds, ks, prune_method):
    """
    prune_method에 따라 달라진 아키텍처를 재구성
    ks: LogicLayer 출력 차원 리스트
    """
    n_classes = num_classes_of_dataset(ds)
    idim      = input_dim_of_dataset(ds)

    # ① LogicLayer 스택
    base_layers = [torch.nn.Flatten(), LogicLayer(idim, ks[0], connections='unique')]
    for in_dim, out_dim in zip(ks[:-1], ks[1:]):
        base_layers.append(LogicLayer(in_dim, out_dim, connections='unique'))

    # ② pruning 방식별 꼬리 구성
    if prune_method in ['retrain', 'retrain_all', 'WGS']:
        final_layers = [
            *base_layers,
            WeightedGroupSum(k=n_classes, in_dim=ks[-1], tau=10)
        ]

    elif prune_method == 'masked_gs':
        final_layers = [
            *base_layers,
            MaskedGroupSum(k=n_classes, in_dim=ks[-1], tau=10)
        ]

    else:
        raise ValueError(f"지원되지 않는 prune_method: {prune_method}")

    return torch.nn.Sequential(*final_layers)



# ───────── 모델 압축(Physical Pruning) 관련 함수 ───────────
def apply_mask_to_logic_weights(logic_layer, mask_layer, device="cuda"):
    """
    Helper 함수: MaskLayer의 설정을 LogicLayer의 가중치에 덮어씌움
    """
    # Get mask_weights safely (handle backward compatibility)
    if hasattr(mask_layer, 'mask_weights'):
        mask_weights = mask_layer.mask_weights
    elif hasattr(mask_layer, 'mask_layer') and hasattr(mask_layer.mask_layer, 'mask_weights'):
        # Old version: mask_layer wrapped in mask_layer
        mask_weights = mask_layer.mask_layer.mask_weights
    else:
        raise AttributeError(f"ChannelMaskLayer has no mask_weights attribute")
    
    # 0: 0-tie, 1: 1-tie, 2: Bypass
    mask_vals = mask_weights.argmax(dim=-1).to(device)
    
    with torch.no_grad():
        # 0-tie 처리 (0번 연산: Always 0)
        zero_tie_indices = torch.where(mask_vals == 0)[0]
        if len(zero_tie_indices) > 0:
            logic_layer.weights.data[zero_tie_indices] = -100.0
            logic_layer.weights.data[zero_tie_indices, 0] = 100.0 
        
        # 1-tie 처리 (15번 연산: Always 1)
        one_tie_indices = torch.where(mask_vals == 1)[0]
        if len(one_tie_indices) > 0:
            logic_layer.weights.data[one_tie_indices] = -100.0
            logic_layer.weights.data[one_tie_indices, 15] = 100.0
            
    print(f"    -> Applied mask to {len(zero_tie_indices)} zero-ties and {len(one_tie_indices)} one-ties.")


def fuse_and_remove_masks(module, device="cuda"):
    """
    [Advanced Fusion] 
    1. LogicLayer (or TreeConv) + ChannelMaskLayer 조합을 찾아 병합.
    2. 병합된 ChannelMaskLayer와 불필요한 Identity Layer를 모델 구조에서 '삭제'.
    """
    # 자식 모듈부터 재귀적으로 처리 (Bottom-up or traverse children first)
    for name, child in module.named_children():
        fuse_and_remove_masks(child, device)

    # 현재 모듈이 Sequential인 경우에만 구조 변경(삭제) 가능
    if isinstance(module, nn.Sequential):
        new_modules = OrderedDict()
        skip_next = False
        
        # 이름과 모듈을 리스트로 변환하여 인덱싱 접근
        layer_names = list(module._modules.keys())
        
        for i, name in enumerate(layer_names):
            if skip_next:
                skip_next = False
                continue
            
            current_mod = module._modules[name]
            
            # 다음 모듈 확인 (범위 체크)
            next_mod = None
            if i + 1 < len(layer_names):
                next_name = layer_names[i+1]
                next_mod = module._modules[next_name]
            
            # [삭제 대상 1] 이미 Identity인 레이어 (이전 퓨전 잔재 등)
            if isinstance(current_mod, nn.Identity):
                # print(f"  - Removing Identity layer: {name}")
                continue # 새 리스트에 추가 안 함 -> 삭제됨

            # [Fusion 검사] 현재=Logic/TreeConv AND 다음=Mask
            is_fused = False
            if isinstance(next_mod, ChannelMaskLayer):
                
                # Case A: LogicLayer + Mask
                if isinstance(current_mod, LogicLayer):
                    print(f"  - Fusing {name}(Logic) + Next(Mask)...")
                    apply_mask_to_logic_weights(current_mod, next_mod, device)
                    is_fused = True
                    
                # Case B: TreeConvLayer + Mask
                # TreeConvLayer는 내부에 cascade(Sequential)를 가짐. 그 중 '마지막' LogicLayer에 적용해야 함.
                elif isinstance(current_mod, (TreeConvLayer, FusedTreeConvLayer)):
                    print(f"  - Fusing {name}(TreeConv) + Next(Mask)...")
                    # cascade의 마지막 레이어 가져오기
                    last_logic = current_mod.cascade[-1]
                    apply_mask_to_logic_weights(last_logic, next_mod, device)
                    is_fused = True
            
            # 퓨전되었다면 현재 모듈만 추가하고, 다음 모듈(Mask)은 스킵(삭제)
            new_modules[name] = current_mod
            if is_fused:
                skip_next = True # 다음 레이어(Mask)는 건너뜀 (삭제 효과)
        
        # 재구성된 모듈로 Sequential 덮어쓰기
        # _modules 속성을 직접 수정하여 내부 구조 변경
        module._modules = new_modules





def _analyze_logic_cascade_and_get_masks(modules_in_block, start_mask, live_masks, name_prefix, verbose):
    """
    [수정된 보조 함수]
    블록 내부 로직 레이어를 역순으로 분석하며 각 레이어의 '살아있는 출력 마스크'를 기록합니다.

    Args:
        modules_in_block (iterable): 분석할 (이름, 모듈) 쌍.
        start_mask (torch.Tensor): 추적을 시작할 출력 마스크.
        live_masks (dict): 계층별 '살아있는 출력 마스크'를 저장할 딕셔너리.
        name_prefix (str): 결과 딕셔너리의 키로 사용될 계층의 접두사.
        verbose (bool): 상세 정보 출력 여부.
    
    Returns:
        torch.Tensor: 해당 블록 전체에 대한 '입력 마스크'.
    """
    alive_mask = start_mask
    # reversed list를 사용하여 nn.Sequential의 역순으로 순회
    for name, m in reversed(list(modules_in_block)):
        full_name = f"{name_prefix}.{name}"
        
        # 현재 alive_mask는 이 레이어(m)의 '출력'에 대한 마스크입니다.
        # 따라서 이 시점에서 m의 살아있는 '출력 마스크'를 저장합니다.
        live_masks[full_name] = alive_mask.clone()
        
        if verbose:
            total = alive_mask.numel()
            alive = alive_mask.sum().item()
            print(f"  [Mask Recorded] {full_name:<55s}: Output Alive: {alive} / {total}")

        # --- 원본 로직을 사용하여 이 레이어의 '입력' 마스크를 계산 ---
        if isinstance(m, LogicLayer): 
            if m.out_dim != alive_mask.numel():
                if verbose: print(f"  [!] Skipping logic layer {name} due to mask size mismatch ({m.out_dim} vs {alive_mask.numel()})")
                new_alive_mask = torch.ones(m.in_dim, dtype=torch.bool, device=alive_mask.device) # Fallback
            else:
                new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=alive_mask.device)
                alive_output_indices = torch.where(alive_mask)[0]
                op_indices = m.weights.argmax(-1)
                for j in alive_output_indices:
                    op_id = op_indices[j].item()
                    if op_id not in {0, 5, 10, 15}: new_alive_mask[m.indices[0][j].item()] = True
                    if op_id not in {0, 3, 12, 15}: new_alive_mask[m.indices[1][j].item()] = True
            alive_mask = new_alive_mask.bool()

        elif isinstance(m, (CrossbarLayer, BlockEfficientCrossbarLayer)):
            if m.out_dim != alive_mask.numel():
                if verbose: print(f"  [!] Skipping crossbar layer {name} due to mask size mismatch ({m.out_dim} vs {alive_mask.numel()})")
                new_alive_mask = torch.ones(m.in_dim, dtype=torch.bool, device=alive_mask.device) # Fallback
            else:
                new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=alive_mask.device)
                alive_output_indices = torch.where(alive_mask)[0]
                if alive_output_indices.numel() > 0:
                    required_input_indices = None
                    if isinstance(m, BlockEfficientCrossbarLayer):
                        connection_indices_relative = m.weights.argmax(dim=-1)
                        chosen_relative_indices = connection_indices_relative[alive_output_indices]
                        block_indices = alive_output_indices // m.out_per_block
                        required_input_indices = block_indices * m.block_size + chosen_relative_indices
                    elif isinstance(m, EfficientCrossbarLayer):
                        relative_indices = m.weights.argmax(dim=-1).unsqueeze(1)
                        absolute_indices_map = torch.gather(m.active_input_indices, 1, relative_indices).squeeze(1)
                        required_input_indices = absolute_indices_map[alive_output_indices]
                    else: # 기본 CrossbarLayer
                        connection_indices = m.weights.argmax(dim=-1)
                        required_input_indices = connection_indices[alive_output_indices]
                    new_alive_mask[required_input_indices.unique()] = True
            alive_mask = new_alive_mask.bool()
        
    return alive_mask


def get_live_masks_per_layer(model, in_channels, args, device="cuda", verbose: bool = False):
    """
    [수정된 메인 함수]
    모델의 각 계층을 역방향으로 추적하여, 각 계층의 '살아있는 출력 마스크'를 반환합니다.

    Args:
        model (nn.Module): 분석할 PyTorch 모델 (Features + Classifier 구조).
        in_channels (int): 모델의 입력 채널 수.
        args: 데이터셋 정보 등을 포함하는 설정 객체.
        device (str): 연산에 사용할 장치.
        verbose (bool): 분석 과정의 상세 정보를 출력할지 여부.

    Returns:
        dict: 계층의 전체 이름을 키로, 살아있는 출력 노드를 나타내는 
              불리언 마스크(Tensor)를 값으로 갖는 딕셔너리.
    """
    
    # --- 1. Get intermediate shapes of all relevant modules ---
    model.eval()
    module_shapes = {}
    hooks = []
    def get_shape_hook(name):
        def hook(module, input, output):
            # output이 튜플/리스트인 경우 첫 번째 요소만 사용 (일부 모델 구조 대응)
            out = output[0] if isinstance(output, (tuple, list)) else output
            inp = input[0] if isinstance(input, (tuple, list)) else input
            module_shapes[name] = {'input_shape': inp.shape, 'output_shape': out.shape}
        return hook
    
    # 분석에 필요한 모든 레이어 타입을 여기에 포함시킵니다.
    analyzed_types = (ORPool2d, nn.Flatten,
                      CrossbarLayer,
                      BlockEfficientCrossbarLayer, LogicLayer, 
                      WeightedGroupSum, PrunedGroupSum, MaskedGroupSum, GroupSum,
                      Crossbar1x1Conv, FusedTreeConvLayer, TreeConvLayer,
                      ChannelMaskLayer)

    for name, module in model.named_modules():
        if isinstance(module, analyzed_types):
            hooks.append(module.register_forward_hook(get_shape_hook(name)))
            
    dummy_input_shape = (1, in_channels, 28, 28) if 'mnist' in args.dataset else (1, in_channels, 32, 32)
    with torch.no_grad(): model(torch.randn(dummy_input_shape).to(device))
    for h in hooks: h.remove()

    # --- 2. Initialize dictionary for per-layer masks ---
    live_masks = {}

    # --- 3. Stage 1: Analyze Classifier ---
    if verbose: print("\n--- Stage 1: Analyzing Classifier ---")

    # 모델 구조에서 Flatten 레이어를 동적으로 찾아 분류기 시작점을 식별
    flatten_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Flatten):
            flatten_layer_name = name
            break

    # 이름 구조('1.0')를 기반으로 분류기 모듈 식별
    classifier_prefix = flatten_layer_name.split('.')[0] # '1'
    classifier_module = model[int(classifier_prefix)]
    
    classifier_layers = list(classifier_module.named_children())
    final_layer_name_local, final_layer = classifier_layers[-1]
    # Flatten 이후의 로직 레이어들
    logic_layers = classifier_layers[1:-1]
    
    final_layer_full_name = f"{classifier_prefix}.{final_layer_name_local}"

    # 역추적을 시작할 마스크 결정
    if isinstance(final_layer, (WeightedGroupSum, PrunedGroupSum)):
        weights = final_layer.weight_raw.data.round()
        alive_mask = (weights.flatten() != 0)
    else: # GroupSum, nn.Linear 등 다른 최종 레이어의 경우
        # 최종 레이어의 입력 차원과 동일한 크기의 마스크 생성
        last_logic_layer_name_local, last_logic_layer = logic_layers[-1]
        last_logic_layer_full_name = f"{classifier_prefix}.{last_logic_layer_name_local}"
        output_shape = module_shapes[last_logic_layer_full_name]['output_shape']
        
        # [핵심 수정] 마지막 LogicLayer의 0-tie 정보를 확인하여 마스크 설정
        # 0-tie된 노드들은 dead로, 나머지는 live로 설정
        alive_mask = torch.ones(output_shape[1], dtype=torch.bool, device=device)
        
        if isinstance(last_logic_layer, LogicLayer):
            # LogicLayer의 weights에서 0-tie 확인
            # weights shape: [out_dim, 16]
            # 0-tie: argmax == 0, 1-tie: argmax == 15
            with torch.no_grad():
                weights_argmax = last_logic_layer.weights.argmax(dim=-1)  # [out_dim]
                is_0tie = (weights_argmax == 0)
                is_1tie = (weights_argmax == 15)
                is_bypass = ~(is_0tie | is_1tie)  # 0-tie도 1-tie도 아닌 경우
            
            # 0-tie된 노드들은 dead로 설정 (GroupSum에 전달되지 않음)
            # 1-tie와 bypass는 live로 유지
            if is_0tie.numel() == alive_mask.numel():
                alive_mask = ~is_0tie  # 0-tie가 아닌 것만 live
                if verbose:
                    print(f"Last LogicLayer tie info: 0-tie={is_0tie.sum().item()}, 1-tie={is_1tie.sum().item()}, bypass={is_bypass.sum().item()}")
                    print(f"Alive mask for GroupSum input: {alive_mask.sum().item()}/{alive_mask.numel()}")
            else:
                if verbose:
                    print(f"Warning: Size mismatch. LogicLayer out_dim: {is_0tie.numel()}, alive_mask: {alive_mask.numel()}")

    # 최종 레이어의 출력은 모델의 최종 출력이므로, 모든 노드가 살아있다고 가정 (10개 클래스)
    # 이는 재구성 시에는 사용되지 않지만, 분석의 완전성을 위해 기록
    if final_layer_full_name in module_shapes:
         final_output_shape = module_shapes[final_layer_full_name]['output_shape']
         live_masks[final_layer_full_name] = torch.ones(final_output_shape[1], dtype=torch.bool, device=device)
    
    # 상세 분석 함수를 사용하여 Classifier 로직을 역추적
    # alive_mask는 final_layer의 입력이자, 마지막 logic_layer의 출력이 됨
    classifier_input_alive_mask = _analyze_logic_cascade_and_get_masks(
        logic_layers, alive_mask, live_masks, name_prefix=classifier_prefix, verbose=verbose
    )

    # --- 4. Stage 2: Analyze Features ---
    if verbose: print("\n--- Stage 2: Analyzing Feature Channels ---")
    
    # Flatten 레이어의 출력 마스크 기록
    live_masks[flatten_layer_name] = classifier_input_alive_mask.clone()

    feature_output_shape = module_shapes[flatten_layer_name]['input_shape']
    C, H, W = feature_output_shape[1], feature_output_shape[2], feature_output_shape[3]
    
    # 1D 마스크를 채널 차원의 3D 마스크로 변환
    alive_channels_mask = classifier_input_alive_mask.view(C, H * W).any(dim=1)
    
    feature_prefix = str(int(classifier_prefix) - 1) # '0'
    feature_module = model[int(feature_prefix)]
    
    # Feature Extractor의 모듈들을 역순으로 순회
    for name, m in reversed(list(feature_module.named_children())):
        full_name = f"{feature_prefix}.{name}"
        if full_name not in module_shapes: continue

        # 현재 alive_channels_mask는 모듈 m의 '출력 채널'에 대한 마스크입니다.
        live_masks[full_name] = alive_channels_mask.clone()
        if verbose:
            total = alive_channels_mask.numel()
            alive = alive_channels_mask.sum().item()
            print(f"[Mask Recorded] {full_name:<55s}: Output Channels Alive: {alive} / {total}")
        
        # Sanity check
        output_shape = module_shapes[full_name]['output_shape']
        if alive_channels_mask.numel() != output_shape[1]:
            print(f"[ERROR] Mismatch at {full_name}: Mask size {alive_channels_mask.numel()} != Module output channels {output_shape[1]}. Skipping.")
            continue
            
        # --- 원본 로직을 사용하여 이 모듈의 '입력 채널' 마스크를 계산 ---
        if isinstance(m, ORPool2d):
            # ORPool2d는 채널 수 변경 없음, 마스크 그대로 통과
            pass
        elif isinstance(m, Crossbar1x1Conv):
            # Crossbar1x1Conv는 BlockEfficientCrossbarLayer를 내부에 가지고 있음
            # Crossbar의 연결을 역추적하여 입력 채널 마스크 계산
            if hasattr(m, 'crossbar') and isinstance(m.crossbar, BlockEfficientCrossbarLayer):
                # Crossbar의 출력 마스크를 입력 마스크로 역추적
                crossbar_output_mask = alive_channels_mask
                crossbar_input_mask = torch.zeros(m.in_channels, dtype=torch.bool, device=device)
                
                # Crossbar의 연결 관계를 역추적
                if crossbar_output_mask.numel() == m.out_channels:
                    alive_output_indices = torch.where(crossbar_output_mask)[0]
                    if alive_output_indices.numel() > 0:
                        # BlockEfficientCrossbarLayer의 연결 추적
                        # connections == 'unique'일 때는 weights가 None이고 connection_indices를 사용
                        if m.crossbar.weights is None:
                            # 'unique' 또는 'unique_contiguous' 모드
                            if hasattr(m.crossbar, 'connection_indices'):
                                # connection_indices는 [out_dim] 또는 [2, out_dim] 형태
                                if m.crossbar.connection_indices.dim() == 1:
                                    # 'unique' 모드: connection_indices는 [out_dim] 형태
                                    connection_indices_relative = m.crossbar.connection_indices
                                else:
                                    # 'unique_contiguous' 모드: connection_indices는 [2, out_dim] 형태
                                    # 첫 번째 후보를 사용 (또는 argmax로 선택)
                                    connection_indices_relative = m.crossbar.connection_indices[0]
                            else:
                                # connection_indices도 없으면 모든 입력을 살아있다고 가정
                                alive_channels_mask = torch.ones(m.in_channels, dtype=torch.bool, device=device)
                                continue
                        else:
                            # 'ste' 또는 'softmax' 모드: weights 사용
                            # weights shape: [out_dim, block_size]
                            connection_indices_relative = m.crossbar.weights.argmax(dim=-1)
                        
                        chosen_relative_indices = connection_indices_relative[alive_output_indices]
                        block_indices = alive_output_indices // m.crossbar.out_per_block
                        required_input_indices = block_indices * m.crossbar.block_size + chosen_relative_indices
                        crossbar_input_mask[required_input_indices.unique()] = True
                
                alive_channels_mask = crossbar_input_mask
        elif isinstance(m, (FusedTreeConvLayer, TreeConvLayer)):
            # TreeConvLayer/FusedTreeConvLayer 내부의 로직 캐스케이드 분석
            # cascade는 nn.Sequential로 LogicLayer들을 포함
            if hasattr(m, 'cascade'):
                logic_input_mask = _analyze_logic_cascade_and_get_masks(
                    m.cascade.named_children(), alive_channels_mask, live_masks, f"{full_name}.cascade", verbose
                )
                in_ch = m.in_dim
                kernel_flat_dim = m.kernel_size ** 2
                if logic_input_mask.numel() != in_ch * kernel_flat_dim:
                    if verbose:
                        print(f"  [!] Shape mismatch in {full_name} logic. Expected {in_ch*kernel_flat_dim}, got {logic_input_mask.numel()}")
                    # Fallback: 모든 입력 채널을 살아있다고 가정
                    alive_channels_mask = torch.ones(in_ch, dtype=torch.bool, device=device)
                else:
                    alive_channels_mask = logic_input_mask.view(in_ch, kernel_flat_dim).any(dim=1)
            else:
                if verbose:
                    print(f"  [!] {full_name} has no cascade attribute, skipping")


        elif isinstance(m, ChannelMaskLayer):
            # [추가된 로직] Channel Mask Layer 처리
            # 이 레이어는 채널 단위로 0(0-tie), 1(1-tie), 2(bypass) 중 하나를 수행합니다.
            # 역방향 분석 시:
            # "이 레이어의 입력이 살아있으려면?"
            # = "이 레이어의 출력이 뒤에서 필요하고(alive_channels_mask)" 
            #   AND "이 레이어가 그 신호를 막지 않고 통과시켜야 함(Bypass)"
            
            with torch.no_grad():
                # argmax가 2인 경우만 Bypass (살아있음)
                # mask_weights shape: [num_channels, 3]
                selection = m.mask_weights.argmax(dim=-1)
                is_bypass = (selection == 2).to(device)
            
            # 크기 검증 (안전장치)
            if is_bypass.shape[0] != alive_channels_mask.shape[0]:
                 print(f" [!] WARNING: Channel count mismatch in {full_name}. "
                       f"MaskLayer: {is_bypass.shape[0]}, CurrentMask: {alive_channels_mask.shape[0]}")
                 # 크기가 다르면 일단 안전하게 기존 마스크 유지하거나 에러 처리
            else:
                # [핵심] 기존에 살아있더라도, 여기서 Tie(0 or 1)되면 죽은 것으로 간주 (AND 연산)
                alive_channels_mask = alive_channels_mask & is_bypass
                

    # --- 5. 최종 결과 반환 ---
    # 모델의 최종 입력에 필요한 채널 마스크
    live_masks['model_input'] = alive_channels_mask.clone()

    return live_masks





def rebuild_physically_pruned_model_conv(
    original_model: nn.Sequential,
    live_masks: dict,
    args: argparse.Namespace,
    device: str = "cuda"
) -> nn.Sequential:
    """
    [FINAL HYBRID VERSION]
    - 전체적인 구조는 안정적인 문자열 기반 마스크 조회 및 상태 전파 방식을 사용합니다.
    - PatchLogicBlock 내부의 LogicLayer 재구성 로직은 사용자가 제공한 검증된 코드를 그대로 적용합니다.
    """
    print("Rebuilding physically pruned model...")
    # --- 1. Feature Extractor 재구성 (model[0]) ---
    new_features = nn.Sequential()
    # 모델의 첫 입력 마스크로 재구성 시작 (상태 변수)
    current_input_channel_mask = live_masks['model_input']

    for name, module in original_model[0].named_children():
        full_name = f"0.{name}"
        if full_name not in live_masks: continue

        # 이 블록의 최종 출력 채널 마스크를 미리 가져옴
        output_channel_mask = live_masks[full_name]
        new_in_channels = current_input_channel_mask.sum().item()
        
        
        new_layer = None
        # ----------------------------------------------------------------------
        # [Case 1] Crossbar1x1Conv 재구성 (BlockEfficientCrossbarLayer 활용)
        # ----------------------------------------------------------------------
        if isinstance(module, Crossbar1x1Conv):
            new_out_channels = output_channel_mask.sum().item()
            
            # [핵심 변경] num_blocks=1, connections='unique'로 설정
            # 이렇게 하면 내부적으로 BlockEfficientCrossbarLayer(num_blocks=1)이 생성됩니다.
            # -> 'unique' 모드이므로 weights 대신 connection_indices를 사용합니다.
            new_layer = type(module)(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                num_blocks=1,             # [Key] 전체를 하나의 블록으로 취급 (절대 인덱스 사용)
                connections='unique',     # [Key] 인덱스 기반 매핑 사용
                shuffle=module.shuffle,
                input_height=getattr(module, 'input_height', None),
                input_width=getattr(module, 'input_width', None),
                init='normal'             # 초기화 방식은 덮어쓸 것이라 중요하지 않음
            ).to(device)
            
            # ------------------------------------------------------------------
            # 연결 인덱스(Connection Indices) 계산 로직
            # ------------------------------------------------------------------
            live_out_indices = torch.where(output_channel_mask)[0]
            
            # 1. 현재 레이어의 살아있는 입력 인덱스 (절대 위치) 파악
 
            live_in_indices = torch.where(current_input_channel_mask)[0]

            # 2. 원본 모델의 연결 정보(절대 인덱스) 추출
            # (기존 복잡한 로직을 그대로 사용하여 원본이 가리키던 절대 주소를 가져옴)
            if isinstance(module.crossbar, BlockEfficientCrossbarLayer):
                if module.crossbar.weights is None: # unique mode
                    if hasattr(module.crossbar, 'connection_indices'):
                         # 1D or 2D indices handling
                        if module.crossbar.connection_indices.dim() == 1:
                            orig_rel = module.crossbar.connection_indices
                        else:
                            orig_rel = module.crossbar.connection_indices[0]
                    else:
                        raise ValueError(f"Crossbar {full_name} missing indices")
                else: # ste/softmax mode
                    orig_rel = module.crossbar.weights.argmax(dim=-1)
                
                # 상대 주소 -> 절대 주소 변환
                orig_block_starts = (torch.arange(module.crossbar.out_dim, device=device) // module.crossbar.out_per_block) * module.crossbar.block_size
                orig_abs_input_indices = orig_block_starts + orig_rel
            
            elif isinstance(module.crossbar, CrossbarLayer):
                orig_abs_input_indices = module.crossbar.weights.argmax(dim=-1)
            
            # 3. 살아남은 출력들이 가리키던 원본 입력 인덱스 선택
            selected_orig_abs_indices = orig_abs_input_indices[live_out_indices]
            
            # ------------------------------------------------------------------
            # [핵심] Remap Table을 통한 새 인덱스 매핑
            # ------------------------------------------------------------------
            # Shuffle 모드 고려: Shuffle이 켜져 있으면 Feature Map이 Flatten된 상태로 인덱싱됨
            # 하지만 여기서는 '채널' 단위의 Remapping을 수행한 뒤, Crossbar 내부 로직에 맡김
            
            # 원본 입력 차원 크기의 테이블 생성
            original_in_dim = module.crossbar.in_dim # ex: 9 or Previous_Out
            in_remap_table = torch.full((original_in_dim,), -1, dtype=torch.long, device=device)
            

            # 내부 레이어: 압축된 인덱스로 매핑
            in_remap_table[live_in_indices] = torch.arange(new_in_channels, device=device)
            
            # 최종 매핑된 인덱스 계산
            remapped_indices = in_remap_table[selected_orig_abs_indices]
            
            # [Crash 방지] 매핑 실패(-1)한 연결은 0번 입력으로 강제 연결
            if (remapped_indices < 0).any():
                # print(f"Warning: {full_name} has dead inputs. Mapping to 0.")
                remapped_indices[remapped_indices < 0] = 0
            
            # ------------------------------------------------------------------
            # [Final Step] 새 레이어에 인덱스 주입
            # ------------------------------------------------------------------
            # BlockEfficientCrossbarLayer(num_blocks=1)은 connection_indices가 [out_dim] 크기여야 함
            # remapped_indices의 크기는 [new_out_channels] 이므로 정확히 일치함.
            
            # connection_indices를 올바르게 설정
            new_layer.crossbar.register_buffer('connection_indices', remapped_indices.clone())
            new_layer.crossbar.weights = None # unique 모드 확정
            
            current_input_channel_mask = output_channel_mask
            
        elif isinstance(module, (FusedTreeConvLayer, TreeConvLayer)):
            # TreeConvLayer/FusedTreeConvLayer 재구성
            new_out_channels = output_channel_mask.sum().item()
            
            # 입력 마스크를 kernel_size^2로 확장하여 로직 레이어 입력 마스크로 사용
            kernel_size_sq = module.kernel_size ** 2
            live_in_mask_for_logic = current_input_channel_mask.repeat_interleave(kernel_size_sq)
            
            # cascade 내부 LogicLayer들을 순방향으로 재구성
            rebuilt_cascade_layers = []
            cascade_prefix = f"{full_name}.cascade"
            
            # -----------------------------------------------------------
            # [수정] Cascade 내부 마스크 추적용 변수 초기화
            # -----------------------------------------------------------
            current_cascade_mask = current_input_channel_mask

            for i, (logic_name, logic_layer) in enumerate(module.cascade.named_children()):
                logic_full_name = f"{cascade_prefix}.{logic_name}"
                
                # --- [A] 현재 레이어의 '실제 입력 마스크' 결정 ---
                if i == 0:
                    # [중요] 첫 번째 로직 레이어는 (Channel * Kernel^2) 입력을 받습니다.
                    # 따라서 채널 마스크를 커널 픽셀 수만큼 확장해야 합니다.
                    kernel_size_sq = module.kernel_size ** 2
                    layer_input_mask = current_cascade_mask.repeat_interleave(kernel_size_sq)
                else:
                    # 두 번째부터는 이전 레이어의 출력이 곧 입력이므로 그대로 사용
                    layer_input_mask = current_cascade_mask

                # --- [B] 현재 레이어의 '출력 마스크' 가져오기 ---
                live_out_mask_for_logic = live_masks.get(logic_full_name, None)


                if live_out_mask_for_logic is None:
                    # 마스크가 없으면(마지막 레이어 등) 처리
                    if logic_name == list(module.cascade.named_children())[-1][0]:
                        live_out_mask_for_logic = recorded_output_mask # TreeConv 전체 출력 마스크 사용
                    else:
                        # Fallback
                        live_out_mask_for_logic = torch.ones(logic_layer.out_dim, dtype=torch.bool, device=device)

                # --- [C] 차원 계산 및 레이어 생성 ---        
                new_in_dim = layer_input_mask.sum().item()
                new_out_dim = live_out_mask_for_logic.sum().item()
                
                
                # LogicLayer 재구성
                # group 기반 연결을 사용하지 않고 'unique'로 생성한 후 원본 indices를 재매핑
                new_logic_layer = LogicLayer(new_in_dim, new_out_dim, connections='unique', device=device)
                remap_table = torch.full((logic_layer.in_dim,), -1, dtype=torch.long, device=device)
                remap_table[layer_input_mask] = torch.arange(new_in_dim, device=device)
                live_out_indices = torch.where(live_out_mask_for_logic)[0]
                
                new_logic_layer.weights.data = logic_layer.weights.data[live_out_indices].clone()
                remapped_a = remap_table[logic_layer.indices[0][live_out_indices]]
                remapped_b = remap_table[logic_layer.indices[1][live_out_indices]]
                
                if (remapped_a == -1).any() or (remapped_b == -1).any():
                    print(f"  [!] Replacing dead input index with 0.")
                    remapped_a[remapped_a == -1] = 0
                    remapped_b[remapped_b == -1] = 0
                
                # 원본 연결을 절대 인덱스로 재매핑하여 유지
                new_logic_layer.indices = (remapped_a, remapped_b)
                rebuilt_cascade_layers.append(new_logic_layer)
                

                current_cascade_mask = live_out_mask_for_logic
            
            # 새로운 TreeConvLayer/FusedTreeConvLayer 생성
            # base_logic_layer_kw 생성
            base_logic_layer_kw = dict(
                ste=args.ste,
                implementation='cuda',
                init=args.init,
                tau=1.0,
                device=device
            )
            
            if isinstance(module, FusedTreeConvLayer):
                # FusedTreeConvLayer 재생성
                new_layer = FusedTreeConvLayer(
                    in_channels=new_in_channels,
                    out_channels=new_out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    k=module.k,
                    k_in=module.k_in,
                    logic_layer_kwargs=base_logic_layer_kw,
                    connections='unique'
                ).to(device)
                
                # cascade 교체
                new_layer.cascade = nn.Sequential(*rebuilt_cascade_layers)
                new_layer.l1 = rebuilt_cascade_layers[0] if len(rebuilt_cascade_layers) > 0 else None
                new_layer.l2 = rebuilt_cascade_layers[1] if len(rebuilt_cascade_layers) > 1 else None
                new_layer.l3 = rebuilt_cascade_layers[2] if len(rebuilt_cascade_layers) > 2 else None
            else: # TreeConvLayer
                # TreeConvLayer 재생성
                # 압축된 입력 차원이 k_in으로 나누어떨어지지 않으면 k_in을 None으로 설정
                new_logic_in_features = new_in_channels * (module.kernel_size ** 2)
                adjusted_k_in = module.k_in
                if module.k_in is not None and new_logic_in_features % module.k_in != 0:
                    adjusted_k_in = None
                    print(f"  [!] WARNING: new_logic_in_features ({new_logic_in_features}) is not divisible by k_in ({module.k_in}). Setting k_in to None.")
                
                new_layer = TreeConvLayer(
                    in_channels=new_in_channels,
                    out_channels=new_out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    k=module.k,
                    k_in=adjusted_k_in,
                    logic_layer_kwargs=base_logic_layer_kw,
                    compress=True
                ).to(device)
                
                # cascade 교체
                new_layer.cascade = nn.Sequential(*rebuilt_cascade_layers)

                current_input_channel_mask = current_cascade_mask

        elif isinstance(module, ORPool2d):
            new_layer = copy.deepcopy(module)

        if new_layer is not None:
            new_features.add_module(name, new_layer)

        new_in_channels = current_input_channel_mask.sum().item()
    # -----------------------------------------------------------    
    # --- 2. Classifier 재구성 (나머지 로직은 이전과 동일하게 유지) ---
    # -----------------------------------------------------------

    new_classifier = nn.Sequential()

    # [수정됨] 1. Flatten 레이어 마스크 생성
    # Features의 출력: (B, C, H, W) -> Flatten -> (B, C*H*W)
    # LogicLayer는 (N, in_dim) 형태를 기대하므로, in_dim = C*H*W (살아있는 채널 수 * feature map 크기)
    
    # Feature map 크기: MNIST는 3x3, CIFAR-10은 2x2
    if 'mnist' in args.dataset:
        feature_map_size = 3 * 3  # 9
    else:  # cifar-10
        feature_map_size = 2 * 2  # 4
    
    # channel_mask를 feature_map_size만큼 확장
    # channel_mask가 [1, 0, ...] 이면 -> [1,1,...,1 (feature_map_size개), 0,0,...,0 (feature_map_size개), ...]으로 확장
    current_input_mask = current_input_channel_mask.repeat_interleave(feature_map_size)
    print(f"current_input_mask.shape: {current_input_mask.shape} (channels={current_input_channel_mask.sum().item()}, feature_map_size={feature_map_size})")

    # 1-2. [핵심 수정] Flatten 직후의 실제 입력 차원(new_in_dim_actual)을 여기서 계산해야 합니다.
    new_in_dim_actual = current_input_mask.sum().item()
    print(f"Flatten Output Dim (Channels x Feature Map Size): {new_in_dim_actual} = {current_input_channel_mask.sum().item()} x {feature_map_size}")


    # Classifier 내부 모듈들을 순방향으로 순회하며 재구성합니다.
    for name, module in original_model[1].named_children():
        full_name = f"1.{name}"
        if full_name not in live_masks: continue

        # 이 모듈의 '출력' 마스크
        output_mask = live_masks[full_name]
        
        # [중요] 현재 레이어의 '실제 입력 차원'은 직전 단계에서 계산된 new_in_dim_actual을 사용
        new_layer = None
        
        if isinstance(module, nn.Flatten):
            # Flatten은 이미 위에서 마스크 처리를 했으므로 레이어만 복사하고 넘어갑니다.
            new_layer = copy.deepcopy(module)
            new_classifier.add_module(name, new_layer)

            continue

        elif isinstance(module, LogicLayer):
            new_out_dim = output_mask.sum().item()
            new_layer = LogicLayer(new_in_dim_actual, new_out_dim, connections='unique', device=device)
            
            remap_table = torch.full((module.in_dim,), -1, dtype=torch.long, device=device)
            limit = min(current_input_mask.shape[0], module.in_dim)
            remap_table[:limit][current_input_mask[:limit]] = torch.arange(current_input_mask[:limit].sum().item(), device=device)

            live_out_indices = torch.where(output_mask)[0]
            new_layer.weights.data = module.weights.data[live_out_indices].clone()
            
            remapped_a = remap_table[module.indices[0][live_out_indices]]
            remapped_b = remap_table[module.indices[1][live_out_indices]]
            remapped_a[remapped_a < 0] = 0
            remapped_b[remapped_b < 0] = 0
            
            new_layer.indices = (remapped_a, remapped_b)

        elif isinstance(module, (GroupSum, WeightedGroupSum)):
            # GroupSum/WeightedGroupSum은 입력 마스크를 k개로 쪼개서 그룹 사이즈를 다시 계산해야 함
            # current_input_mask는 마지막 LogicLayer의 실제 출력 마스크 (pruning된 상태)
            
            # 원본 모듈의 입력 차원 확인
            original_in_dim = module.in_dim if hasattr(module, 'in_dim') else current_input_mask.numel()
            
            # 원본 모듈에서 각 그룹의 크기 계산
            original_group_size = original_in_dim // module.k
            
            # current_input_mask를 원본 그룹 구조에 맞춰서 각 그룹별로 살아있는 노드 수 계산
            # current_input_mask는 원본 크기이므로, 각 그룹별로 나눠서 계산
            new_group_sizes = []
            for group_idx in range(module.k):
                group_start = group_idx * original_group_size
                group_end = min(group_start + original_group_size, current_input_mask.numel())
                if group_start < current_input_mask.numel():
                    group_mask = current_input_mask[group_start:group_end]
                    new_group_sizes.append(group_mask.sum().item())
                else:
                    new_group_sizes.append(0)
            
            new_group_sizes_tensor = torch.tensor(new_group_sizes, dtype=torch.long, device=device)
            
            if isinstance(module, WeightedGroupSum):
                # WeightedGroupSum -> PrunedWeightedGroupSum
                # 가중치도 마스크에 맞춰서 잘라냄
                # WeightedGroupSum의 weight_raw는 [k, g] 형태이므로 flatten하여 [k*g]로 만든 후 마스크 적용
                pruned_weights = module.weight_raw.data.view(-1)[current_input_mask]
                
                new_layer = PrunedWeightedGroupSum(
                    k=module.k, 
                    group_sizes=new_group_sizes_tensor, 
                    weights=pruned_weights, 
                    tau=module.tau,
                    beta=getattr(module, 'beta', 0.0),
                    noise_prob=getattr(module, 'noise_prob', 0.0),
                    reg_type=getattr(module, 'reg_type', 'l1'),
                    w_max=getattr(module, 'w_max', None)
                ).to(device)
            else:
                # GroupSum -> PrunedGroupSum
                new_layer = PrunedGroupSum(
                    k=module.k, 
                    group_sizes=new_group_sizes_tensor, 
                    tau=module.tau,
                    beta=getattr(module, 'beta', 0.0),
                    noise_prob=getattr(module, 'noise_prob', 0.0)
                ).to(device)


        if new_layer is not None:
            new_classifier.add_module(name, new_layer)
        
        # [중요] 다음 레이어를 위해 입력 마스크 업데이트
        current_input_mask = output_mask
        # 다음 레이어의 in_dim은 현재 레이어의 out_dim (마스크의 sum)이 됩니다.
        new_in_dim_actual = current_input_mask.sum().item()

    # --- 3. 최종 압축 모델 조립 ---
    final_pruned_model = nn.Sequential(OrderedDict([('0', new_features), ('1', new_classifier)]))
    #if verbose: print("\n[Model Rebuilding Complete]")
    return final_pruned_model



# ==============================================================================
# ▲ 평가 및 데이터 로딩 함수 끝 ▲
# ==============================================================================

def process_single_model(input_path, args, dev, in_channels, test_loader, log_file=None):
    """
    단일 모델을 압축하는 함수
    
    Args:
        input_path: 입력 모델 파일 경로
        args: argparse 인자
        dev: 디바이스
        in_channels: 입력 채널 수
        test_loader: 테스트 데이터 로더
        log_file: 로그 파일 객체 (선택)
    
    Returns:
        output_path: 저장된 모델 경로
    """
    import io
    import sys
    
    # 로그 파일이 있으면 출력을 리다이렉트
    original_stdout = sys.stdout
    if log_file:
        sys.stdout = log_file
    
    try:
        print(f"\n{'='*80}")
        print(f"Processing: {input_path}")
        print(f"{'='*80}")
        
        # 모델 로드
        if not os.path.exists(input_path):
            print(f"ERROR: Model file not found at {input_path}")
            return None
        
        loaded_data = torch.load(input_path, map_location=dev)
        
        if isinstance(loaded_data, nn.Module):
            model = loaded_data
            print(f"Successfully loaded full model from {input_path}")
        elif isinstance(loaded_data, dict):
            model, _, _ = get_model(args, in_channels)
            model.to(dev)
            if 'model' in loaded_data:
                model.load_state_dict(loaded_data['model'])
            else:
                model.load_state_dict(loaded_data)
            print(f"Successfully loaded model structure and weights from {input_path}")
        else:
            print(f"ERROR: Unknown format in {input_path}. Expected nn.Module or dict (state_dict).")
            return None
        
        model.to(dev)
        print("\n[압축 전 모델 구조]")
        print(model)
        model.eval()
        
        # Fusion Step
        print("\n[Fusion Step]")
        fuse_and_remove_masks(model, device=dev)
        
        print("\n[Fusion 후 모델 구조]")
        print(model)
        
        # 정확도 평가
        print("\nEvaluating the loaded model...")
        accuracy = eval_fn(model, test_loader, False)
        print(f"  - Test Accuracy: {accuracy * 100:.2f}%")
        
        # 압축
        if args.compression:
            with torch.no_grad():
                live_masks = get_live_masks_per_layer(model, in_channels, args, device="cuda")
                
                print("\n--- Generated Live Masks (Layer Output) ---")
                for name, mask in live_masks.items():
                    total = mask.numel()
                    alive = mask.sum().item()
                    pruned_ratio = (1 - alive / total) * 100 if total > 0 else 0
                    print(f"{name:<60s} | Alive: {alive:>5d} / {total:<5d} | Pruned: {pruned_ratio:6.2f}%")
                
                model = rebuild_physically_pruned_model_conv(model, live_masks, args, device="cuda")
            
            print("\n[압축 후 모델 구조]")
            print(model)
            model.to(dev).eval()
            
            print("\n[압축 후 모델 정확도 측정]")
            compressed_acc = eval_fn(model, test_loader, mode=False)
            print(f"  - 정확도: {compressed_acc:.4f}")
            
            print("\n[정확도 비교]")
            if abs(accuracy - compressed_acc) < 1e-5:
                print("  ✅ 정확도가 일치합니다. 압축이 성공적으로 완료되었습니다.")
            else:
                print(f"  ❌ 경고: 정확도가 일치하지 않습니다! 초기: {accuracy:.4f}, 압축 후: {compressed_acc:.4f}")
        
        # 출력 경로 결정
        if args.output_model and args.input_model and len(args.input_model) == 1 and args.input_model[0] == input_path:
            # 단일 모델이고 output_model이 지정된 경우
            output_path = args.output_model
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        else:
            # 여러 모델이거나 output_dir 사용
            os.makedirs(args.output_dir, exist_ok=True)
            # 입력 파일명 기반으로 출력 파일명 생성
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            if args.compression:
                output_filename = f"{base_name}_compressed.pt"
            else:
                output_filename = f"{base_name}.pt"
            output_path = os.path.join(args.output_dir, output_filename)
        
        # 모델 저장
        print(f"\n[모델 저장]")
        torch.save(model, output_path)
        print(f"✅ Full model saved to: {output_path}")
        
        # 검증
        try:
            loaded_model = torch.load(output_path)
            loaded_model.to(dev).eval()
            print("Verification: Successfully loaded the saved .pt file.")
        except Exception as e:
            print(f"Verification failed: Could not load the saved model. Error: {e}")
        
        return output_path
        
    finally:
        if log_file:
            sys.stdout = original_stdout


def main():
    # --- 3. argparse: 기존 fast_inference.py와 유사하게, conv_difflogic용 인자 추가 ---
    parser = argparse.ArgumentParser(description="Replicates the process of fast_inference.py for Conv-DiffLogic models (without compression).")
    
    # 필수 인자
    parser.add_argument('--experiment_id', '-eid', type=int, default=None, help="Experiment ID of the trained model from conv_difflogic.py. Required if --input-model is not provided.")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar-10-3-thresholds', 'mnist'], help="Dataset used for training the model.")
    
    # 모델 아키텍처 인자 (conv_difflogic.py와 동일하게)
    parser.add_argument('--model-size', type=str, default='S', choices=['S','M','B','L','G'])
    parser.add_argument('--num-neurons', '-k', type=int, default=None)
    parser.add_argument('--eval-freq', '-ef', type=int, default=1000)
    parser.add_argument('--num-iterations', '-ni', type=int, default=50000)
    parser.add_argument('--tree-depth', '-td', type=int, default=3)
    parser.add_argument('--init', type=str, default='residual', choices=['random', 'residual'])
    parser.add_argument('--groups', type=int, default=None, help="Number of groups for grouped convolutions.")
    parser.add_argument('--implementation', type=str, default='im2col', choices=['python', 'cuda', 'triton', 'im2col', 'im2col_separable'])
    parser.add_argument('--classifier-arch', type=str, default='groupsum', choices=['groupsum', 'wgs', 'soft_wgs', 'im2col_pruned'],
                        help='Architecture for the final classifier layer.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None)

    # 기타 실행 옵션
    parser.add_argument('--no-ste', dest='ste', action='store_false', help="Disable Straight-Through Estimator.")
    parser.add_argument('--tau', '-t', type=float, default=1.0, help="Softmax temperature tau for the classifier.")
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument('--results-dir', type=str, default='./results/', help="Directory where .pth weights are stored.")
    parser.add_argument('--output-dir', type=str, default='compressed_models/', help="Directory to save the compressed .pt models.")
    parser.add_argument('--input-model', type=str, nargs='+', default=None, help="Path(s) to input model file(s) (.pt). Can specify multiple models. If not provided, will use --results-dir and --experiment_id to construct path.")
    parser.add_argument('--output-model', type=str, default=None, help="Path to output model file (.pt). Only used when processing a single model. For multiple models, files are saved to --output-dir with auto-generated names.")
    parser.add_argument('--log-dir', type=str, default='compression_logs/', help="Directory to save compression logs with model structures.")
    parser.add_argument('--seed', '-s', type=int, default=0, help='Seed for reproducibility.')

    parser.add_argument('--compression', action='store_true', help='Compress the model.')
    parser.add_argument('--block-size', type=int, default=128,
                        help='block size for crossbar.')

    args = parser.parse_args()

    # --- 기존 하이퍼파라미터 자동 설정 로직 (k, groups, tau, lr, bs) ---
    if args.num_neurons is None:
        if 'cifar-10' in args.dataset:
            k_map = {'S': 32, 'M': 256, 'B': 512, 'L': 1024, 'G': 2048}
        else: # mnist
            k_map = {'S': 16, 'M': 64, 'L': 1024}
        args.num_neurons = k_map.get(args.model_size)
            
    if args.groups is None:
        args.groups = max(1, args.num_neurons // 8)

    dataset_key = 'cifar-10' if 'cifar-10' in args.dataset else 'mnist'
    tau_map = {
        'cifar-10': {'S': 20, 'M': 40, 'B': 280, 'L': 340, 'G': 450},
        'mnist':    {'S': 6.5, 'M': 28, 'L': 35}
    }

    if args.model_size in tau_map[dataset_key]:
        args.tau = tau_map[dataset_key][args.model_size]
        print(f"AUTO: Set tau to {args.tau} for {dataset_key} (size: {args.model_size}).")
    

    if args.batch_size is None:
        bs_map = {
            'cifar-10': {'S': 128, 'M': 128, 'B': 128, 'L': 128, 'G': 128},
            'mnist': {'S': 512, 'M': 256, 'L': 128}
        }
        if dataset_key in bs_map and args.model_size in bs_map[dataset_key]:
            args.batch_size = bs_map[dataset_key][args.model_size]
            print(f"AUTO: Set batch_size to {args.batch_size} for {dataset_key} (size: {args.model_size})")
        else:
            print(f"WARNING: No batch size in table for {dataset_key} (size: {args.model_size}). Batch size remains None.")

    # 시드 고정
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 4. 인자 기반 모델 생성 ---
    print("="*60)
    print("Building Model from Command-Line Arguments")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")

    # `get_model`이 필요로 하는 `in_channels` 계산
    if 'cifar-10' in args.dataset:
        in_channels = 9 if args.model_size in ['S', 'M'] else 93
    elif 'mnist' in args.dataset:
        in_channels = 1
    
    dataset_key = 'cifar-10' if 'cifar-10' in args.dataset else 'mnist'

    if args.learning_rate is None:
        lr_map = {'cifar-10': 0.02, 'mnist': 0.01}
        args.learning_rate = lr_map[dataset_key]
        print(f"AUTO: Set learning_rate to {args.learning_rate}")

    

    if args.batch_size is None:
        bs_map = {
            'cifar-10': {'S': 128, 'M': 128, 'B': 128, 'L': 128, 'G': 128},
            'mnist': {'S': 512, 'M': 256, 'L': 128}
        }
        if dataset_key in bs_map and args.model_size in bs_map[dataset_key]:
            args.batch_size = bs_map[dataset_key][args.model_size]
            print(f"AUTO: Set batch_size to {args.batch_size} for {dataset_key} (size: {args.model_size})")
        else:
            print(f"WARNING: No batch size in table for {dataset_key} (size: {args.model_size}). Batch size remains None.")

    # --- 5. 입력 모델 경로 리스트 생성 ---
    input_paths = []
    if args.input_model is not None:
        input_paths = args.input_model
    else:
        # 기존 방식: results_dir와 experiment_id로 경로 구성
        if args.experiment_id is None:
            sys.exit("\nError: Either --input-model or --experiment_id must be provided.")
        pt_path = os.path.join(args.results_dir, f"{args.experiment_id}_mask_prune_phase6.pt")
        input_paths = [pt_path]
    
    # --- 6. 테스트 데이터 로더 준비 ---
    test_loader, in_channels = load_dataset(args)
    
    # --- 7. 로그 디렉토리 생성 ---
    os.makedirs(args.log_dir, exist_ok=True)
    
    # --- 8. 각 모델 처리 ---
    print(f"\n{'='*80}")
    print(f"Processing {len(input_paths)} model(s)")
    print(f"{'='*80}\n")
    
    log_file_paths = []
    
    for idx, input_path in enumerate(input_paths, 1):
        print(f"\n[{idx}/{len(input_paths)}] Processing: {input_path}")
        
        # 입력 파일명 기반으로 로그 파일명 생성 (pt 파일명과 동일한 형식)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if args.compression:
            log_filename = f"{base_name}_compressed.log"
        else:
            log_filename = f"{base_name}.log"
        log_file_path = os.path.join(args.log_dir, log_filename)
        log_file_paths.append(log_file_path)
        
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Compression Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Input Model: {input_path}\n")
            log_file.write(f"Dataset: {args.dataset}\n")
            log_file.write(f"Model Size: {args.model_size}\n")
            log_file.write(f"Compression: {args.compression}\n")
            log_file.write(f"{'='*80}\n\n")
            
            output_path = process_single_model(
                input_path, args, dev, in_channels, test_loader, log_file
            )
            
            if output_path:
                print(f"✅ Completed: {input_path} -> {output_path}")
                log_file.write(f"\n✅ Saved to: {output_path}\n\n")
            else:
                print(f"❌ Failed: {input_path}")
                log_file.write(f"\n❌ Failed to process: {input_path}\n\n")
    
    print(f"\n{'='*80}")
    print(f"All models processed!")
    print(f"Log files saved to: {args.log_dir}")
    for log_path in log_file_paths:
        print(f"  - {log_path}")
    print(f"Compressed models saved to: {args.output_dir}")
    print(f"{'='*80}")



if __name__ == '__main__':
    main()






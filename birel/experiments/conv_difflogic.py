"""
conv_difflogic.py
Train a Convolutional Differentiable Logic Gate Network (LogicTreeNet)
based on the paper "Convolutional Differentiable Logic Gate Networks".

Added features:
- Support for WeightedGroupSum classifier architecture.
- Retraining/pruning functionality for existing models.
"""
import argparse
import math
import random
import os
import sys

# Add project root to Python path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

# difflogic.py에 필요한 클래스들이 있다고 가정합니다.
# WeightedGroupSum을 새로 import 합니다.
from difflogic import GroupSum, LogicLayer, FusedLogicTreeBlock, WeightedGroupSum, PrunedGroupSum, PrunedWeightedGroupSum, MaskedGroupSum, PackBitsTensor
from difflogic.functional import bin_op_s
from birel.model import *
from birel.pruning import *
from birel.conv import *
from birel.utils import (
    analyze_crossbar_connections,
    _analyze_logic_cascade,
    finding_live_nodes_by_channel,
    summarize_and_print_analysis,
    summarize_concise_analysis,
    generate_feature_map_saliency
)
from torch.cuda.amp import GradScaler, autocast
from collections import Counter, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import copy
from typing import List, Tuple
from torch.profiler import profile, record_function, ProfilerActivity

scaler = GradScaler()


# results_json이 없을 경우를 대비한 더미 클래스
try:
    from results_json import ResultsJSON
except ImportError:
    print("Warning: 'results_json' library not found. Experiment results will not be saved.")
    class ResultsJSON:
        def __init__(self, *args, **kwargs): pass
        def store_args(self, *args, **kwargs): pass
        def store_results(self, *args, **kwargs): pass
        def store_final_results(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass

# wandb 로깅 지원
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: 'wandb' library not found. Wandb logging will be disabled.")
    WANDB_AVAILABLE = False
    wandb = None

torch.set_num_threads(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def remove_residual_mask_hooks(model):
    """
    Remove all forward hooks from ResidualChannelMaskLayer instances in the model.
    This is necessary before saving the model to avoid pickle errors.
    """
    from birel.conv import ResidualChannelMaskLayer
    for module in model.modules():
        if isinstance(module, ResidualChannelMaskLayer):
            module._remove_input_hook()


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


def apply_residual_mask_to_logic_weights(prev_logic_layer, residual_mask_layer, device="cuda"):
    """
    Helper 함수: ResidualChannelMaskLayer의 설정을 이전 LogicLayer의 가중치에 덮어씌움
    include_tie=False: 0=bypass-a, 1=bypass-b, 2=neg bypass-a, 3=neg bypass-b, 4=bypass
    include_tie=True: 0=0-tie, 1=1-tie, 2=bypass-a, 3=bypass-b, 4=neg bypass-a, 5=neg bypass-b, 6=bypass
    """
    from birel.conv import ResidualChannelMaskLayer
    
    if not isinstance(residual_mask_layer, ResidualChannelMaskLayer):
        raise TypeError(f"Expected ResidualChannelMaskLayer, got {type(residual_mask_layer)}")
    
    if prev_logic_layer is None or not hasattr(prev_logic_layer, 'weights'):
        raise ValueError("prev_logic_layer must have weights attribute")
    
    # Get mask_weights safely
    if hasattr(residual_mask_layer, 'mask_weights'):
        mask_weights = residual_mask_layer.mask_weights
    else:
        raise AttributeError(f"ResidualChannelMaskLayer has no mask_weights attribute")
    
    include_tie = getattr(residual_mask_layer, 'include_tie', False)
    mask_vals = mask_weights.argmax(dim=-1).to(device)
    
    # 차원 확인
    prev_out_dim = prev_logic_layer.out_dim
    mask_num_channels = mask_weights.shape[0]
    
    if prev_out_dim != mask_num_channels:
        print(f"    WARNING: Dimension mismatch - prev_logic_layer.out_dim={prev_out_dim}, mask_weights.shape[0]={mask_num_channels}")
        # 작은 차원에 맞춰서 처리
        min_dim = min(prev_out_dim, mask_num_channels)
        mask_vals = mask_vals[:min_dim]
    else:
        min_dim = prev_out_dim
    
    with torch.no_grad():
        if include_tie:
            # include_tie=True: 0=0-tie, 1=1-tie, 2=bypass-a, 3=bypass-b, 4=neg bypass-a, 5=neg bypass-b, 6=bypass
            # 0-tie 처리 (operation 0)
            tie_0_indices = torch.where(mask_vals == 0)[0]
            if len(tie_0_indices) > 0:
                valid_indices = tie_0_indices[tie_0_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 0] = 100.0  # operation 0 (0-tie)
            
            # 1-tie 처리 (operation 15)
            tie_1_indices = torch.where(mask_vals == 1)[0]
            if len(tie_1_indices) > 0:
                valid_indices = tie_1_indices[tie_1_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 15] = 100.0  # operation 15 (1-tie)
            
            # bypass-a 처리 (operation 3: a)
            bypass_a_indices = torch.where(mask_vals == 2)[0]
            if len(bypass_a_indices) > 0:
                valid_indices = bypass_a_indices[bypass_a_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 3] = 100.0  # operation 3 (a)
            
            # bypass-b 처리 (operation 5: b)
            bypass_b_indices = torch.where(mask_vals == 3)[0]
            if len(bypass_b_indices) > 0:
                valid_indices = bypass_b_indices[bypass_b_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 5] = 100.0  # operation 5 (b)
            
            # neg bypass-a 처리 (operation 12: NOT a)
            neg_bypass_a_indices = torch.where(mask_vals == 4)[0]
            if len(neg_bypass_a_indices) > 0:
                valid_indices = neg_bypass_a_indices[neg_bypass_a_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 12] = 100.0  # operation 12 (NOT a)
            
            # neg bypass-b 처리 (operation 10: NOT b)
            neg_bypass_b_indices = torch.where(mask_vals == 5)[0]
            if len(neg_bypass_b_indices) > 0:
                valid_indices = neg_bypass_b_indices[neg_bypass_b_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 10] = 100.0  # operation 10 (NOT b)
            
            # bypass (6)는 원본 유지하므로 처리하지 않음
            print(f"    -> Applied residual mask to {len(tie_0_indices)} 0-tie, {len(tie_1_indices)} 1-tie, {len(bypass_a_indices)} bypass-a, {len(bypass_b_indices)} bypass-b, {len(neg_bypass_a_indices)} neg bypass-a, {len(neg_bypass_b_indices)} neg bypass-b.")
        else:
            # include_tie=False: 0=bypass-a, 1=bypass-b, 2=neg bypass-a, 3=neg bypass-b, 4=bypass
            # bypass-a 처리 (operation 3: a)
            bypass_a_indices = torch.where(mask_vals == 0)[0]
            if len(bypass_a_indices) > 0:
                valid_indices = bypass_a_indices[bypass_a_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 3] = 100.0  # operation 3 (a)
            
            # bypass-b 처리 (operation 5: b)
            bypass_b_indices = torch.where(mask_vals == 1)[0]
            if len(bypass_b_indices) > 0:
                valid_indices = bypass_b_indices[bypass_b_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 5] = 100.0  # operation 5 (b)
            
            # neg bypass-a 처리 (operation 12: NOT a)
            neg_bypass_a_indices = torch.where(mask_vals == 2)[0]
            if len(neg_bypass_a_indices) > 0:
                valid_indices = neg_bypass_a_indices[neg_bypass_a_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 12] = 100.0  # operation 12 (NOT a)
            
            # neg bypass-b 처리 (operation 10: NOT b)
            neg_bypass_b_indices = torch.where(mask_vals == 3)[0]
            if len(neg_bypass_b_indices) > 0:
                valid_indices = neg_bypass_b_indices[neg_bypass_b_indices < prev_out_dim]
                if len(valid_indices) > 0:
                    prev_logic_layer.weights.data[valid_indices] = -100.0
                    prev_logic_layer.weights.data[valid_indices, 10] = 100.0  # operation 10 (NOT b)
            
            # bypass (4)는 원본 유지하므로 처리하지 않음
            print(f"    -> Applied residual mask to {len(bypass_a_indices)} bypass-a, {len(bypass_b_indices)} bypass-b, {len(neg_bypass_a_indices)} neg bypass-a, {len(neg_bypass_b_indices)} neg bypass-b.")


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
            from birel.conv import ResidualChannelMaskLayer
            
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
            
            elif isinstance(next_mod, ResidualChannelMaskLayer):
                # ResidualChannelMaskLayer는 이전 LogicLayer를 참조하여 weights를 치환
                prev_logic_layer = getattr(next_mod, 'prev_logic_layer', None)
                
                if prev_logic_layer is not None and hasattr(prev_logic_layer, 'weights'):
                    print(f"  - Fusing ResidualMask (prev LogicLayer) + Next(ResidualMask)...")
                    apply_residual_mask_to_logic_weights(prev_logic_layer, next_mod, device)
                    is_fused = True
                else:
                    print(f"  - Warning: ResidualChannelMaskLayer {next_name} has no prev_logic_layer, skipping fusion.")
            
            # 퓨전되었다면 현재 모듈만 추가하고, 다음 모듈(Mask)은 스킵(삭제)
            new_modules[name] = current_mod
            if is_fused:
                skip_next = True # 다음 레이어(Mask)는 건너뜀 (삭제 효과)
        
        # 재구성된 모듈로 Sequential 덮어쓰기
        # _modules 속성을 직접 수정하여 내부 구조 변경
        module._modules = new_modules


def finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device="cuda", verbose=False):
    """
    finding_live_nodes_by_channel을 호출하기 전에 fuse를 수행하는 헬퍼 함수.
    
    중요: 원본 모델을 보존하기 위해 반드시 복사본에서 fuse를 수행합니다.
    fuse_and_remove_masks는 모델 구조와 weights를 직접 수정하므로,
    원본 모델에 직접 적용하면 phase 간 모델 상태가 변경될 수 있습니다.
    
    Args:
        model: 원본 모델 (수정되지 않음)
        in_channels: 입력 채널 수
        args: 인자
        device: 디바이스
        verbose: 상세 출력 여부
    
    Returns:
        analysis_results: 분석 결과
    """
    # 원본 모델을 복사하여 fuse 수행 (원본 모델은 절대 수정하지 않음)
    # copy.deepcopy는 모델의 모든 파라미터와 구조를 완전히 복사합니다.
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    model_copy.eval()
    
    # 복사본에서만 fuse 수행 (원본 모델은 영향받지 않음)
    # fuse_and_remove_masks는 모델 구조를 변경하고 LogicLayer weights를 수정합니다.
    fuse_and_remove_masks(model_copy, device=device)
    
    # Fuse된 복사본에서 분석 수행
    analysis_results = finding_live_nodes_by_channel(
        model_copy, in_channels, args, device=device, verbose=verbose
    )
    
    # 복사본은 자동으로 메모리에서 해제됩니다.
    # 원본 모델은 그대로 유지되어 다음 phase에서 사용할 수 있습니다.
    return analysis_results




def load_dataset(args):
    # 이 함수는 이제 모든 전처리를 담당합니다.
    if 'cifar-10' in args.dataset:
        if args.model_size in ['S', 'M']: # 2-bit precision
            def custom_transform(x):
                outputs = [(x > (i + 1) / 4.0).float() for i in range(3)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 3
        else: # B, L, G: 5-bit precision
            def custom_transform(x):
                outputs = [(x > (i + 1) / 32.0).float() for i in range(31)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 31
        
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])  
        
        # 훈련(Train) 데이터에만 적용할 증강 기법들
        train_transforms = torchvision.transforms.Compose([
            #torchvision.transforms.RandomCrop(32, padding=4), # 32x32 이미지 주변에 4픽셀 패딩 후 무작위로 32x32 잘라내기
            #torchvision.transforms.RandomHorizontalFlip(),    # 50% 확률로 좌우 반전
            torchvision.transforms.ToTensor(),                # 텐서로 변환
            torchvision.transforms.Lambda(custom_transform)   # 우리의 커스텀 이진화 적용
        ])
        
        # 검증/테스트(Validation/Test) 데이터에는 증강을 적용하지 않음
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])


        train_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=True, download=True, transform=train_transforms)
        test_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=False, transform=test_transforms)

    elif 'mnist' in args.dataset:
        def custom_transform(x):
            return (x > 0.5).float()
        final_channels = 1
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])

        # 훈련(Train) 데이터에만 적용할 증강 기법들
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),                # 텐서로 변환
            torchvision.transforms.Lambda(custom_transform)   # 우리의 커스텀 이진화 적용
        ])
        
        # 검증/테스트(Validation/Test) 데이터에는 증강을 적용하지 않음
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])

        train_set = torchvision.datasets.MNIST(root='./data-mnist', train=True, download=True, transform=train_transforms)
        test_set = torchvision.datasets.MNIST(root='./data-mnist', train=False, transform=test_transforms)
        
    else:
        raise NotImplementedError

    if 'cifar-10' in args.dataset:
        args.valid_set_size = 5000 / 50000.0
    elif 'mnist' in args.dataset:
        args.valid_set_size = 10000 / 60000.0
    else:
        args.valid_set_size = 0.0

    print(f"valid_set_size: {args.valid_set_size}")

    train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
    valid_set_size = len(train_set) - train_set_size
    if valid_set_size > 0:
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
    else:
        validation_set = test_set

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    return train_loader, validation_loader, test_loader, final_channels




def approximate_channel_loss_scores(
    model,
    target_module,          # TreeConvLayer or LogicLayer
    eval_batches,           # 리스트나 iterable of (x, y) 미니배치
    device='cuda',
    num_batches=5,
    num_probes=1,
    tie_to_zero=True,
    tie_to_one=True,
):

    """
    각 채널을 0-tie / 1-tie 했을 때의 MSE 기반 Δloss를
    Hessian(Gauss-Newton) 2차 근사 + Hutchinson trick으로 계산.

    Returns:
        approx_loss0: [C]  # 0-tie 근사 ΔL (없으면 None)
        approx_loss1: [C]  # 1-tie 근사 ΔL (없으면 None)
    """
    model.train()

    # 1) dummy_x 하나 뽑아서 채널 수/shape 확인
    it = iter(eval_batches)
    try:
        dummy_x, _ = next(it)
    except StopIteration:
        raise ValueError("empty eval_batches in approximate_channel_loss_scores")

    dummy_x = dummy_x.to(device)

    act_cache = {}

    def save_act(_m, _inp, out):
        act_cache['act'] = out
       # out.retain_grad()

    h = target_module.register_forward_hook(save_act)
    with torch.no_grad():
        _ = model(dummy_x)
    h.remove()

    act = act_cache['act']      # [B, C, ...]
    C = act.shape[1]

    h0 = torch.zeros(C, device=device) if tie_to_zero else None
    h1 = torch.zeros(C, device=device) if tie_to_one else None

    # 2) 실제 근사 loop
    batch_iter = iter(eval_batches)
    total_used = 0

    for b in range(num_batches):
        try:
            x, y = next(batch_iter)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)

        for _ in range(num_probes):
            model.zero_grad(set_to_none=True)
            act_cache.clear()

            handle = target_module.register_forward_hook(save_act)

            logits = model(x)   # [B, num_classes]
            handle.remove()

            act = act_cache['act']             # [B, C, ...]
            act.retain_grad()
            B = act.shape[0]
            act_view = act.view(B, C, -1)      # [B, C, S]

            v = torch.randn_like(logits)       # [B, num_classes]
            s = (logits * v).sum()
            s.backward()

            gradA = act.grad                   # [B, C, ...]
            gradA_view = gradA.view(B, C, -1)  # [B, C, S]

            if tie_to_zero:
                delta0 = act_view              # [B, C, S]
                inner0 = (gradA_view * delta0).sum(dim=(0, 2))  # [C]
                h0 += inner0.pow(2).detach()

            if tie_to_one:
                ones = torch.ones_like(act_view)
                delta1 = act_view - ones
                inner1 = (gradA_view * delta1).sum(dim=(0, 2))  # [C]
                h1 += inner1.pow(2).detach()

        total_used += x.size(0)

    approx_loss0 = 0.5 * h0 if h0 is not None else None
    approx_loss1 = 0.5 * h1 if h1 is not None else None
    return approx_loss0, approx_loss1




def measure_treeconv_activation_correlation(model, data_loader, num_samples=1000, save_path=None, method='channel_correlation'):
    """
    TreeConvLayer의 output activation들 사이의 상관관계를 측정하고 2D plot으로 시각화합니다.
    평균하지 않고 전체 activation map을 사용하여 비교합니다.
    
    Args:
        model: 학습된 모델
        data_loader: 데이터 로더
        num_samples: 수집할 샘플 수
        save_path: 저장할 경로 (None이면 표시만)
        method: 비교 방법
            - 'channel_correlation': 채널별 상관관계 (각 채널을 독립적으로 비교)
            - 'spatial_correlation': Spatial pattern 상관관계 (전체 activation map 비교)
            - 'cross_entropy': Cross entropy 기반 비교 (기존 방법)
    
    Returns:
        correlation_matrix: 상관관계 행렬
    """
    model.eval()
    
    # 모든 TreeConvLayer 찾기
    tree_conv_layers = []
    tree_conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, TreeConvLayer):
            tree_conv_layers.append(module)
            tree_conv_names.append(name)
    
    if len(tree_conv_layers) == 0:
        print("Warning: No TreeConvLayer found in the model.")
        return None
    
    print(f"Found {len(tree_conv_layers)} TreeConvLayer(s): {tree_conv_names}")
    print(f"Using comparison method: {method}")
    
    # Activation 저장을 위한 딕셔너리 (전체 activation map 저장)
    activations = {name: [] for name in tree_conv_names}
    
    # Forward hook 등록
    hooks = []
    def get_activation_hook(name):
        def hook(module, input, output):
            # output shape: (B, C, H, W)
            # 평균하지 않고 전체 activation map 저장
            with torch.no_grad():
                # (B, C, H, W) -> (B, C, H*W) flatten spatial
                B, C, H, W = output.shape
                output_flat = output.view(B, C, -1).cpu().numpy()  # (B, C, H*W)
                activations[name].append(output_flat)
        return hook
    
    for name, module in zip(tree_conv_names, tree_conv_layers):
        hooks.append(module.register_forward_hook(get_activation_hook(name)))
    
    # 데이터 수집
    collected_samples = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if collected_samples >= num_samples:
                break
            x = x.to(device)
            _ = model(x)
            collected_samples += x.size(0)
    
    # Hook 제거
    for hook in hooks:
        hook.remove()
    
    # Activation들을 numpy 배열로 변환
    activation_arrays = {}
    for name in tree_conv_names:
        if len(activations[name]) > 0:
            # Stack all collected activations: (total_batches, B, C, H*W) -> (total_B, C, H*W)
            # 모든 배치를 하나로 합침
            all_batches = np.concatenate(activations[name], axis=0)  # (total_B, C, H*W)
            activation_arrays[name] = all_batches
            print(f"  {name}: collected {all_batches.shape[0]} samples, {all_batches.shape[1]} channels, {all_batches.shape[2]} spatial locations")
    
    if len(activation_arrays) == 0:
        print("Warning: No activations collected.")
        return None
    
    # 상관관계 계산 - 같은 TreeConvLayer 내에서 샘플들 간의 상관관계
    # 각 TreeConvLayer별로 독립적인 correlation matrix 생성
    all_correlation_matrices = {}
    
    for layer_idx, layer_name in enumerate(tree_conv_names):
        if layer_name not in activation_arrays:
            continue
        
        act = activation_arrays[layer_name]  # (N, C, H*W)
        N, C, S = act.shape
        print(f"\n  Processing {layer_name}: {N} samples, {C} channels, {S} spatial locations")
        
        if method == 'channel_correlation':
            # 방법 1: 같은 TreeConvLayer 내에서 채널들 간의 상관관계
            # 각 채널을 (N*S) 벡터로 flatten하여 채널 간 correlation 계산
            act_flat = act.reshape(N * S, C).T  # (C, N*S) - 각 행이 하나의 채널
            
            # 채널 간 correlation matrix 계산
            correlation_matrix = np.corrcoef(act_flat)  # (C, C)
            
            # NaN 처리
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            all_correlation_matrices[layer_name] = correlation_matrix
            print(f"    Channel correlation matrix shape: {correlation_matrix.shape}")
            
        elif method == 'spatial_correlation':
            # 방법 2: 같은 TreeConvLayer 내에서 샘플들 간의 spatial pattern 상관관계
            # 각 샘플을 (C*S) 벡터로 flatten하여 샘플 간 correlation 계산
            act_flat = act.reshape(N, C * S)  # (N, C*S) - 각 행이 하나의 샘플
            
            # 샘플 간 correlation matrix 계산
            correlation_matrix = np.corrcoef(act_flat)  # (N, N)
            
            # NaN 처리
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            all_correlation_matrices[layer_name] = correlation_matrix
            print(f"    Sample correlation matrix shape: {correlation_matrix.shape}")
            
        else:  # method == 'cross_entropy' or default
            # 방법 3: 같은 TreeConvLayer 내에서 샘플들 간의 cross entropy 기반 비교
            # Spatial dimension 평균
            act_avg = act.mean(axis=2)  # (N, C)
            
            # Activation을 확률 분포로 변환 (softmax over channels)
            act_probs = F.softmax(torch.from_numpy(act_avg), dim=1).numpy()  # (N, C)
            
            # 샘플 간 cross entropy 계산
            N_samples = act_probs.shape[0]
            correlation_matrix = np.zeros((N_samples, N_samples))
            
            eps = 1e-10
            for i in range(N_samples):
                for j in range(N_samples):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        continue
                    
                    p = act_probs[i]
                    q = act_probs[j]
                    ce = -np.sum(p * np.log(q + eps))
                    
                    # Cross entropy를 상관관계로 변환
                    max_ce = np.log(C)
                    normalized_ce = ce / max_ce if max_ce > 0 else 0
                    correlation = 1.0 - normalized_ce
                    
                    correlation_matrix[i, j] = correlation
            
            all_correlation_matrices[layer_name] = correlation_matrix
            print(f"    Sample correlation matrix (cross entropy) shape: {correlation_matrix.shape}")
    
    # 각 TreeConvLayer별로 시각화
    for layer_name, correlation_matrix in all_correlation_matrices.items():
        plt.figure(figsize=(12, 10))
        
        method_label = {
            'channel_correlation': 'Channel-wise Correlation',
            'spatial_correlation': 'Sample-wise Spatial Pattern Correlation',
            'cross_entropy': 'Sample-wise Correlation (1 - Normalized Cross Entropy)'
        }.get(method, 'Correlation')
        
        if method == 'channel_correlation':
            # 채널 간 correlation - 채널 인덱스를 레이블로
            tick_labels = [f'Ch{i}' for i in range(correlation_matrix.shape[0])]
            title_suffix = f'Channel Correlation'
        else:
            # 샘플 간 correlation - 샘플 인덱스를 레이블로 (너무 많으면 일부만 표시)
            if correlation_matrix.shape[0] > 50:
                tick_labels = [f'S{i}' if i % 10 == 0 else '' for i in range(correlation_matrix.shape[0])]
            else:
                tick_labels = [f'S{i}' for i in range(correlation_matrix.shape[0])]
            title_suffix = f'Sample Correlation'
        
        sns.heatmap(correlation_matrix, 
                    xticklabels=tick_labels if len(tick_labels) <= 50 else False,
                    yticklabels=tick_labels if len(tick_labels) <= 50 else False,
                    annot=False,  # 너무 크면 annot=False
                    fmt='.3f',
                    cmap='coolwarm',
                    center=0.0 if method != 'cross_entropy' else 0.5,
                    vmin=-1 if method != 'cross_entropy' else 0, 
                    vmax=1,
                    cbar_kws={'label': method_label},
                    square=True)
        plt.title(f'TreeConvLayer "{layer_name}" - {title_suffix}\n(Method: {method})')
        plt.xlabel('Channel' if method == 'channel_correlation' else 'Sample')
        plt.ylabel('Channel' if method == 'channel_correlation' else 'Sample')
        plt.tight_layout()
        
        if save_path:
            # Layer name을 파일명에 포함
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            layer_save_path = f"{base_path}_{layer_name.replace('.', '_')}_{method}.{ext}"
            plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
            print(f"    Correlation matrix saved to: {layer_save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # 첫 번째 레이어의 correlation matrix를 반환 (호환성을 위해)
    if all_correlation_matrices:
        first_layer = list(all_correlation_matrices.keys())[0]
        return all_correlation_matrices[first_layer]
    else:
        return None


def collect_logic_outputs(model, x):
    """
    Collect outputs from all LogicLayers during forward pass.
    Returns a dictionary mapping layer names to their outputs.
    """
    caches, hooks = {}, []
    def save(name): 
        def _hook(_m, _i, o): 
            caches[name] = o.detach()
        return _hook
    
    for name, m in model.named_modules():
        if isinstance(m, LogicLayer):
            hooks.append(m.register_forward_hook(save(name)))
    
    _ = model(x)  # forward pass
    for h in hooks: 
        h.remove()
    return caches


def detect_stuck_by_entropy_single_layer(model, loader, target_layer_name=None, target_module=None, prune_pct=0.1, max_batches=10, device='cuda'):
    """
    Detect neurons with lowest entropy in a specific layer for replacement with tie operations.
    
    Args:
        model: The neural network model
        loader: Data loader for analysis
        target_layer_name: Name of the specific layer to analyze (for LogicLayer)
        target_module: Direct module reference (for TreeConvLayer or LogicLayer)
        prune_pct: Percentage of neurons to prune (default: 0.1 = 10%)
        max_batches: Number of batches to analyze (default: 5)
        device: Device to run on
    
    Returns:
        List of tuples: (layer_name, channel_idx, stuck_type, avg_prob, entropy)
        where stuck_type is 'sa0' (stuck-at-0) or 'sa1' (stuck-at-1)
    """
    # Training mode로 설정하여 실제 학습 시와 동일한 activation 패턴 측정
    model.train()
    stats = []  # List of activation tensors for the target layer
    it = iter(loader)
    
    # target_module이 제공된 경우 직접 hook 사용
    use_direct_hook = (target_module is not None)
    hook = None
    
    if use_direct_hook:
        def save_output(module, input, output):
            stats.append(output.detach().cpu())
        hook = target_module.register_forward_hook(save_output)
    
    with torch.no_grad():
        for batch_idx in range(min(max_batches, len(loader))):
            try:
                x, _ = next(it)
                x = x.to(device)
                
                if use_direct_hook:
                    # 직접 hook 사용 (TreeConvLayer 또는 LogicLayer)
                    _ = model(x)
                else:
                    # collect_logic_outputs 사용 (LogicLayer만)
                    caches = collect_logic_outputs(model, x)
                    
                    # Only process the target layer
                    if target_layer_name in caches:
                        y = caches[target_layer_name]
                        # y: [B, C, H, W] or [B, C] depending on layer position
                        if y.dim() == 4:  # Conv layer output [B, C, H, W]
                            # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
                            activations = y.float().permute(1, 0, 2, 3).contiguous().view(y.size(1), -1)  # [C, B*H*W]
                        elif y.dim() == 2:  # FC layer output [B, C]
                            # Transpose to [C, B]
                            activations = y.float().t()  # [C, B]
                        else:
                            continue  # Skip unsupported dimensions
                        
                        stats.append(activations.cpu())
                    
            except StopIteration:
                break
    
    # Hook 제거
    if hook is not None:
        hook.remove()
    
    if not stats:
        layer_info = target_layer_name if target_layer_name else "target module"
        print(f"Warning: No activations found for layer {layer_info}")
        return []
    
    # 직접 hook을 사용한 경우, 출력 형태를 변환
    if use_direct_hook:
        processed_stats = []
        for y in stats:
            # y: [B, C, H, W] or [B, C] depending on layer position
            if y.dim() == 4:  # Conv layer output [B, C, H, W]
                # Flatten spatial dimensions: [B, C, H, W] -> [C, B*H*W]
                activations = y.float().permute(1, 0, 2, 3).contiguous().view(y.size(1), -1)  # [C, B*H*W]
            elif y.dim() == 2:  # FC layer output [B, C]
                # Transpose to [C, B]
                activations = y.float().t()  # [C, B]
            else:
                continue  # Skip unsupported dimensions
            processed_stats.append(activations)
        stats = processed_stats
    
    # Combine all batches: [C, total_samples]
    all_activations = torch.cat(stats, dim=1)  # [C, total_samples]
    
    # Analyze all neurons in this layer
    # 각 채널에 대해 0-tie와 1-tie 둘 다 고려해서 더 나은 것을 선택
    layer_neurons = []  # (channel_idx, avg_prob, entropy, best_stuck_type, distance_to_tie)
    
    for channel_idx in range(all_activations.size(0)):
        channel_activations = all_activations[channel_idx]  # [total_samples]
        
        # Compute average probability
        avg_prob = channel_activations.mean().item()
        
        # 원본 entropy 계산
        unique_vals, counts = torch.unique(channel_activations, return_counts=True)
        if len(unique_vals) > 1:
            probs = counts.float() / counts.sum()
            probs = torch.clamp(probs, min=1e-8)
            empirical_entropy = -(probs * torch.log2(probs)).sum().item()
        else:
            empirical_entropy = 0.0
        
        # 0-tie와 1-tie 중 더 나은 것 선택
        # 방법: 원본 activation이 0 또는 1에 얼마나 가까운지 측정
        # avg_prob가 0.5에 가까우면 entropy가 높은 것 (정보가 많음)
        # avg_prob가 0 또는 1에 가까우면 entropy가 낮은 것 (정보가 적음)
        
        # 0-tie로 교체했을 때의 "거리" = avg_prob (0에 가까울수록 좋음)
        # 1-tie로 교체했을 때의 "거리" = 1 - avg_prob (1에 가까울수록 좋음)
        distance_to_0tie = avg_prob  # 0에 가까울수록 작음
        distance_to_1tie = 1.0 - avg_prob  # 1에 가까울수록 작음
        
        # 거리가 작을수록 해당 tie 타입으로 교체하기 좋음
        if distance_to_0tie <= distance_to_1tie:
            best_stuck_type = 'sa0'
            best_distance = distance_to_0tie
        else:
            best_stuck_type = 'sa1'
            best_distance = distance_to_1tie
        
        # 원본 entropy를 score로 사용 (낮을수록 prune할 가치가 높음)
        layer_neurons.append((channel_idx, avg_prob, empirical_entropy, best_stuck_type, best_distance))
    
    # Sort by entropy (ascending - lowest entropy first)
    layer_neurons.sort(key=lambda x: x[2])  # Sort by entropy (index 2)
    
    # Select bottom prune_pct percentage
    num_to_prune = max(1, int(len(layer_neurons) * prune_pct))
    neurons_to_prune = layer_neurons[:num_to_prune]
    
    layer_info = target_layer_name if target_layer_name else "target module"
    print(f"Layer {layer_info}: {len(layer_neurons)} total neurons, pruning {num_to_prune} ({prune_pct*100:.1f}%)")
    
    # Convert to stuck format with stuck_type determination
    stuck = []
    for channel_idx, avg_prob, entropy, best_stuck_type, distance in neurons_to_prune:
        stuck.append((target_layer_name or "unknown", channel_idx, best_stuck_type, avg_prob, entropy))
    
    return stuck


# ────────────────────────────────────────────────────────────────
# Crossbar 삽입 관련 헬퍼 함수들
# ────────────────────────────────────────────────────────────────

def find_tree_conv_layers(features_module: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    """
    features 모듈에서 모든 TreeConvLayer, FusedTreeConvLayer, FusedTreeConvORPoolLayer 찾기
    
    Args:
        features_module: features Sequential 모듈
    
    Returns:
        (인덱스, 이름, 모듈) 튜플 리스트
    """
    tree_conv_layers = []
    
    if isinstance(features_module, nn.Sequential):
        for idx, module in enumerate(features_module):
            if isinstance(module, (TreeConvLayer, FusedTreeConvLayer)):
                tree_conv_layers.append((idx, f"{idx}", module))
    else:
        # Sequential이 아닌 경우 named_modules 사용
        for name, module in features_module.named_modules():
            if isinstance(module, (TreeConvLayer, FusedTreeConvLayer)):
                depth = name.count('.')
                tree_conv_layers.append((depth, name, module))
    
    # 순서대로 정렬
    tree_conv_layers.sort(key=lambda x: x[0])
    return tree_conv_layers



def insert_crossbar(
    model,
    target_module,  # 어떤 모듈이든 가능 (TreeConvLayer, LogicLayer 등)
    args,
    device: str = 'cuda',
    block_size: int = None  # LogicLayer crossbar의 block_size (기본값: args.block_size)
):
    """
    특정 모듈 뒤에 crossbar 삽입
    
    Args:
        model: 전체 모델 (nn.Module)
        target_module: 삽입할 타겟 모듈 (TreeConvLayer, LogicLayer 등)
        args: argparse 객체
        device: 디바이스
        block_size: LogicLayer crossbar의 block_size (기본값: args.block_size)
    
    Returns:
        (수정된 모델, 삽입된 crossbar의 인덱스)
    """
    # target_module이 속한 Sequential 찾기 (model[0] 또는 model[1])
    modules_list = None
    sequential_idx = None
    
    # model[0] (features)에서 찾기
    if isinstance(model[0], nn.Sequential):
        for idx, module in enumerate(model[0]):
            if module is target_module:
                modules_list = list(model[0])
                sequential_idx = 0
                break
    
    # model[1] (classifier)에서 찾기
    if modules_list is None and isinstance(model[1], nn.Sequential):
        for idx, module in enumerate(model[1]):
            if module is target_module:
                modules_list = list(model[1])
                sequential_idx = 1
                break
    
    if modules_list is None:
        raise ValueError(f"Could not find target_module in model[0] or model[1]")
    
    # target_module의 실제 위치 찾기
    target_idx = None
    for idx, module in enumerate(modules_list):
        if module is target_module:
            target_idx = idx
            break
    
    if target_idx is None:
        raise ValueError(f"Could not find target_module in modules_list")
    
    # 출력 차원 자동 감지 (out_dim으로 통일)
    if hasattr(target_module, 'out_dim'):
        out_dim = target_module.out_dim
    else:
        raise ValueError(f"target_module must have 'out_dim' attribute")
    
    # target_module 타입에 따라 crossbar 타입 자동 결정
    is_logic_layer = isinstance(target_module, LogicLayer)
    
    if is_logic_layer:
        if block_size is None:
            block_size = getattr(args, 'block_size', 128)
        
        num_blocks = out_dim // block_size
        if num_blocks == 0:
            num_blocks = 1
        
        bottleneck = getattr(args, 'crossbar_bottleneck', False)
        
        if bottleneck:
            bottleneck_factor = getattr(args, 'bottleneck_factor', 0.125)  # 기본값: 0.125 (1/8)
            bottleneck_mid_dim = int(out_dim * bottleneck_factor)
            num_blocks_mid = bottleneck_mid_dim // block_size
            if num_blocks_mid == 0:
                num_blocks_mid = 1
            
            new_crossbar1 = BlockEfficientCrossbarLayer(
                in_dim=out_dim,
                out_dim=bottleneck_mid_dim,
                num_blocks=num_blocks,
                connections='ste',
                device=device,
                init=getattr(args, 'crossbar_init', 'normal')
            ).to(device)
            new_crossbar2 = BlockEfficientCrossbarLayer(
                in_dim=bottleneck_mid_dim,
                out_dim=out_dim,
                num_blocks=num_blocks,
                connections='ste',
                device=device,
                init=getattr(args, 'crossbar_init', 'normal')
            ).to(device)
            new_crossbar_module = nn.Sequential(new_crossbar1, new_crossbar2)
        else:
            new_crossbar_module = BlockEfficientCrossbarLayer(
                in_dim=out_dim,
                out_dim=out_dim,
                device=device,
                connections='ste',
                num_blocks=num_blocks
            ).to(device)
    else:
        # TreeConv용: Crossbar1x1Conv 생성
        connections = 'ste'
        bottleneck = getattr(args, 'crossbar_bottleneck', False)
        num_blocks = getattr(args, 'num_blocks', None)
        block_size = getattr(args, 'block_size', 128)
        
        # num_blocks 계산
        if num_blocks is None:
            num_blocks = out_dim // block_size
            if num_blocks == 0:
                num_blocks = 1
        
        if bottleneck:
            # Bottleneck 구조: 두 개의 crossbar를 Sequential로 묶음
            bottleneck_factor = getattr(args, 'bottleneck_factor', 0.125)  # 기본값: 0.125 (1/8)
            bottleneck_mid_dim = int(out_dim * bottleneck_factor)
            if getattr(args, 'num_blocks', None) is None:
                num_blocks_mid = bottleneck_mid_dim // block_size
                if num_blocks_mid == 0:
                    num_blocks_mid = 1
            else:
                num_blocks_mid = num_blocks
            
            new_crossbar1 = BlockEfficientCrossbarLayer(
                in_dim=out_dim,
                out_dim=bottleneck_mid_dim,
                num_blocks=num_blocks,
                connections=connections,
                device=device,
                init='normal'
            )
            new_crossbar2 = BlockEfficientCrossbarLayer(
                in_dim=bottleneck_mid_dim,
                out_dim=out_dim,
                num_blocks=num_blocks,
                connections=connections,
                device=device,
                init='normal'
            )
            new_crossbar_seq = nn.Sequential(new_crossbar1, new_crossbar2)
            
            new_crossbar_module = Crossbar1x1Conv(
                in_channels=out_dim,
                out_channels=out_dim,
                num_blocks=num_blocks,
                connections=connections
            )
            new_crossbar_module.crossbar = new_crossbar_seq
        else:
            # 단일 crossbar
            new_crossbar_module = Crossbar1x1Conv(
                in_channels=out_dim,
                out_channels=out_dim,
                num_blocks=num_blocks,
                connections=connections
            )
    
    # target_module 바로 다음에 삽입
    insert_pos = target_idx + 1
    modules_list.insert(insert_pos, new_crossbar_module)
    
    # 새로 삽입된 crossbar만 requires_grad = True로 설정
    if is_logic_layer:
        # LogicLayer crossbar: 직접 BlockEfficientCrossbarLayer 또는 Sequential
        if isinstance(new_crossbar_module, nn.Sequential):
            for sub_crossbar in new_crossbar_module:
                for param in sub_crossbar.parameters():
                    param.requires_grad = True
        else:
            for param in new_crossbar_module.parameters():
                param.requires_grad = True
    else:
        # TreeConv crossbar: Crossbar1x1Conv 내부의 crossbar
        if isinstance(new_crossbar_module.crossbar, nn.Sequential):
            for sub_crossbar in new_crossbar_module.crossbar:
                for param in sub_crossbar.parameters():
                    param.requires_grad = True
        else:
            for param in new_crossbar_module.crossbar.parameters():
                param.requires_grad = True
    
    # Sequential로 재구성
    model[sequential_idx] = nn.Sequential(*modules_list)
    model[sequential_idx] = model[sequential_idx].to(device)
    return model, insert_pos

def get_model(args, in_channels):
    class_count = 10
    k = args.num_neurons

    # 1. 역할에 따라 인자 딕셔너리를 명확하게 분리합니다.

    # LogicLayer를 위한 기본(base) 인자 딕셔너리 (원본을 유지하며 사용)
    base_logic_layer_kw = dict(
        ste=False, 
        implementation=args.implementation if args.implementation != 'im2col' else 'cuda', 
        init=args.init, 
        tau=1.0
    )

    torch.manual_seed(0)

    if 'cifar-10' in args.dataset:
        print(f"Building LogicTreeNet for CIFAR-10 (k={k})")
        
        # --- CIFAR-10용 컨볼루션 인자 정의 ---
        
        # for 'python', 'triton' implementations
        logic_tree_conv_kw = dict(
            kernel_size=3, padding=1, tree_depth=args.tree_depth, 
            groups=args.groups, **base_logic_layer_kw
        )
        first_logic_tree_conv_kw = logic_tree_conv_kw.copy()
        first_logic_tree_conv_kw['groups'] = 1
        
        # for 'im2col' implementation
        patch_logic_block_kw = dict(
            kernel_size=3, padding=1,  stride=1, 
            logic_layer_kwargs=base_logic_layer_kw
        )

        pooling_kw = base_logic_layer_kw.copy()
        pooling_kw['init'] = 'residual'

        # --- features 부분 구성 ---
        
        if args.implementation in ["python", "triton"]:
            # LogicTreeConv2d는 더 이상 존재하지 않으므로 triton만 사용
            if args.implementation == "python":
                raise NotImplementedError("LogicTreeConv2d is no longer available. Please use 'triton' implementation.")
            LayerClass = FusedLogicTreeBlock
            PoolClass = nn.Identity
            print(f"Using {LayerClass.__name__} implementation.")
            
            features = nn.Sequential(
                FusedLogicTreeBlock(in_channels, k, **first_logic_tree_conv_kw),
                FusedLogicTreeBlock(k, 4*k, **logic_tree_conv_kw),
                FusedLogicTreeBlock(4*k, 16*k, **logic_tree_conv_kw),
                FusedLogicTreeBlock(16*k, 32*k, **logic_tree_conv_kw)
            )
            # FusedLogicTreeBlock은 Pooling을 포함하므로 PoolClass를 조건부로 적용
            if args.implementation == "triton":
                features = nn.Sequential(
                    FusedLogicTreeBlock(in_channels, k, **first_logic_tree_conv_kw),
                    FusedLogicTreeBlock(k, 4*k, **logic_tree_conv_kw),
                    FusedLogicTreeBlock(4*k, 16*k, **logic_tree_conv_kw),
                    FusedLogicTreeBlock(16*k, 32*k, **logic_tree_conv_kw)
                )
            
            final_feature_dim = 32 * k *2 *2
        elif args.implementation == "im2col":
            base_logic_layer_kw['implementation'] = 'cuda'
            min_cascade_depth = 3
            
            # FusedTreeConvORPoolLayer 사용 여부 확인
            use_fused = getattr(args, 'use_fused_tree_conv', False)
            
            if use_fused:
                # FusedTreeConvLayer 사용 (TreeConvLayer의 LogicLayer만 fused, ORPool은 별도)
                print("Using FusedTreeConvLayer (fused TreeConv logic layers, ORPool separate)")
                base_features = [
                # --- Stage 1 ---
                    Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
                    FusedTreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, 
                                      k=k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 2 ---
                    Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks = k//8, connections='unique'),
                    FusedTreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1,
                                      k=4*k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 3 ---
                    Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks = k//8, connections='unique'),
                    FusedTreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1,
                                      k=16*k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 4 ---
                    Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks = k//8, connections='unique'),
                    #FusedTreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1,
                    #                  k=32*k, k_in=base_logic_layer_kw.get('k_in'),
                    #                  logic_layer_kwargs=base_logic_layer_kw),
                    TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, k=32*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                ]
            else:
                # 기본 features 구성 (TreeConvLayer + ORPool2d 분리)
                base_features = [
                # --- Stage 1 ---
                Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
                TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, k=k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 2 ---
                Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1, k=4*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2), 
                # --- Stage 3 ---
                Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1, k=16*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 4 ---   
                Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, k=32*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                ]
            features = nn.Sequential(*base_features)
            final_feature_dim = 32 * k *2 *2

        elif args.implementation == "im2col_sparse":
            base_logic_layer_kw['implementation'] = 'cuda'
            min_cascade_depth = 3
            
            # FusedTreeConvORPoolLayer 사용 여부 확인
            use_fused = getattr(args, 'use_fused_tree_conv', False)
            
            if use_fused:
                # FusedTreeConvLayer 사용 (TreeConvLayer의 LogicLayer만 fused, ORPool은 별도)
                print("Using FusedTreeConvLayer (fused TreeConv logic layers, ORPool separate)")
                base_features = [
                # --- Stage 1 ---
                    Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
                    FusedTreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, 
                                      k=k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                    ChannelMaskLayer(num_channels=k, device='cuda', frozen=True),
                    ORPool2d(2, 2),
                # --- Stage 2 ---
                    Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks = k//8, connections='unique'),
                    FusedTreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1,
                                      k=4*k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                    ChannelMaskLayer(num_channels=4*k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                # --- Stage 3 ---
                    Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks = k//8, connections='unique'),
                    FusedTreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1,
                                      k=16*k, k_in=base_logic_layer_kw.get('k_in'),
                                      logic_layer_kwargs=base_logic_layer_kw),
                    ChannelMaskLayer(num_channels=16*k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                # --- Stage 4 ---
                    Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks = k//8, connections='unique'),
                    #FusedTreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1,
                    #                  k=32*k, k_in=base_logic_layer_kw.get('k_in'),
                    #                  logic_layer_kwargs=base_logic_layer_kw),
                    TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, k=32*k, logic_layer_kwargs=base_logic_layer_kw),
                    ChannelMaskLayer(num_channels=32*k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                ]
            else:
                # 기본 features 구성 (TreeConvLayer + ORPool2d 분리)
                base_features = [
                # --- Stage 1 ---
                Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
                TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, k=k, logic_layer_kwargs=base_logic_layer_kw),
                ChannelMaskLayer(num_channels=k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                # --- Stage 2 ---
                Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1, k=4*k, logic_layer_kwargs=base_logic_layer_kw),
                ChannelMaskLayer(num_channels=4*k, device='cuda', frozen=True),
                ORPool2d(2, 2), 
                # --- Stage 3 ---
                Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1, k=16*k, logic_layer_kwargs=base_logic_layer_kw),
                ChannelMaskLayer(num_channels=16*k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                # --- Stage 4 ---   
                Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks = k//16, connections='unique'),
                TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, k=32*k, logic_layer_kwargs=base_logic_layer_kw),
                ChannelMaskLayer(num_channels=32*k, device='cuda', frozen=True),
                ORPool2d(2, 2),
                ]
            features = nn.Sequential(*base_features)
            final_feature_dim = 32 * k *2 *2

        else:
            raise NotImplementedError(f"Implementation '{args.implementation}' is not implemented.")
        
        # --- Classifier 부분 구성 ---
        #final_feature_dim = 32 * k * 2 * 2
        layer_dims = [1280*k, 640*k, 320*k]
        if args.model_size in ['B', 'L']:
            layer_dims = [d * 2 for d in layer_dims]
        
        classifier_layers = [
            nn.Flatten(),
            #BlockEfficientCrossbarLayer(in_dim=final_feature_dim, out_dim=final_feature_dim, device=device, connections='ste', block_size=args.block_size),           
            LogicLayer(final_feature_dim, layer_dims[0], **base_logic_layer_kw),
            LogicLayer(layer_dims[0], layer_dims[1], **base_logic_layer_kw),
            LogicLayer(layer_dims[1], layer_dims[2], **base_logic_layer_kw),
            GroupSum(k=class_count, tau=args.tau),
        ]

    elif 'mnist' in args.dataset:
        print(f"Building LogicTreeNet for MNIST (k={k})")
        
        # --- MNIST용 컨볼루션 인자 정의 ---
        
        # for 'python', 'triton' implementations
        logic_tree_conv_kw = dict(
            kernel_size=3, padding=1, tree_depth=args.tree_depth, 
            groups=args.groups, **base_logic_layer_kw
        )
        first_logic_tree_conv_kw = logic_tree_conv_kw.copy()
        first_logic_tree_conv_kw['groups'] = 1
        first_logic_tree_conv_kw['kernel_size'] = 5
        first_logic_tree_conv_kw['padding'] = 0


        # for 'cuda' implementation
        # for 'im2col' implementation
        patch_logic_block_kw = dict(
            kernel_size=3, padding=1,  stride=1,
            logic_layer_kwargs=base_logic_layer_kw
        )
        first_patch_logic_block_kw = patch_logic_block_kw.copy()
        first_patch_logic_block_kw['kernel_size'] = 5
        first_patch_logic_block_kw['padding'] = 0
        
        pooling_kw = dict(
            kernel_size=2, padding=0, stride=2,
            # 입력 4C, 출력 C이므로 비율 4.0으로 설정해 캐스케이드 방지
            
            #max_input_output_ratio=2.0, 
            logic_layer_kwargs=base_logic_layer_kw
        )
        
        # --- features 부분 구성 ---
        if args.implementation in ["python", "triton"]:
            # LogicTreeConv2d는 더 이상 존재하지 않으므로 triton만 사용
            if args.implementation == "python":
                raise NotImplementedError("LogicTreeConv2d is no longer available. Please use 'triton' implementation.")
            LayerClass = FusedLogicTreeBlock
            PoolClass = nn.Identity
            print(f"Using {LayerClass.__name__} implementation.")
            
            # FusedLogicTreeBlock은 Pooling을 포함함
            features = nn.Sequential(
                FusedLogicTreeBlock(in_channels, k, **first_logic_tree_conv_kw),
                FusedLogicTreeBlock(k, 3*k, **logic_tree_conv_kw),
                FusedLogicTreeBlock(3*k, 9*k, **logic_tree_conv_kw)
            )
        
        elif args.implementation == "im2col":
            base_features = [
                # --- Stage 1 ---
                Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
                TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=5, padding=0, stride=1, k=k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 2 ---
                Crossbar1x1Conv(in_channels=k, out_channels=3*k*2, num_blocks = k//8, connections='unique'),
                TreeConvLayer(in_channels=3*k*2, out_channels=3*k, kernel_size=3, padding=1, stride=1, k=3*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
                # --- Stage 3 ---
                Crossbar1x1Conv(in_channels=3*k, out_channels=9*k*2, num_blocks = k//8, connections='unique'),
                TreeConvLayer(in_channels=9*k*2, out_channels=9*k, kernel_size=3, padding=1, stride=1, k=9*k, logic_layer_kwargs=base_logic_layer_kw),
                ORPool2d(2, 2),
            ]
            print(f"base_features: {base_features}")
            features = nn.Sequential(*base_features)

        else:
            raise NotImplementedError(f"Implementation '{args.implementation}' is not implemented.")
            
        # --- Classifier 부분 구성 ---
        final_feature_dim = 9 * k * 3 * 3
        layer_dims = [1280*k, 640*k, 320*k]
        if args.model_size in ['S', 'M']:
            layer_dims = [d * 2 for d in layer_dims]

        classifier_layers = [
            nn.Flatten(),
            LogicLayer(final_feature_dim, layer_dims[0], **base_logic_layer_kw),
            LogicLayer(layer_dims[0], layer_dims[1], **base_logic_layer_kw),
            LogicLayer(layer_dims[1], layer_dims[2], **base_logic_layer_kw),
            GroupSum(k=class_count, tau=args.tau),
        ]
    else:
        raise NotImplementedError(f"Architecture for dataset '{args.dataset}' is not implemented.")


    classifier = nn.Sequential(*classifier_layers)
    model = nn.Sequential(features, classifier)

        
    model = model.to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()

    param_groups = [
        {'params': model[0].parameters(), 'weight_decay': 0.002},  # Features
        {'params': model[1].parameters(), 'weight_decay': 0.002} # Classifier
    ]



    if args.dataset == 'cifar-10-3-thresholds':
        weight_decay = 0.001 if args.model_size == 'G' else 0.002
        print(f"Using AdamW optimizer for CIFAR-10 with weight_decay={weight_decay}")
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
    else:
        print("Using default Adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    
    return model, loss_fn, optimizer


# -----------------------------------------------------------------------------
# 3. UPDATED TRAINING AND EVALUATION LOOP (기능 추가)
# -----------------------------------------------------------------------------

def load_n(loader, n):
    i=0
    while i < n:
        for d in loader: yield d; i+=1;
        if i >= n: return



def train(model, x, y, loss_fn, optimizer, clip_grad_norm):
    model.train()
    output = model(x)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()

    # --- 그래디언트 클리핑 추가 ---
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

    optimizer.step()
    return loss.item()


def freeze_channel_masks(model: nn.Module):
    """
    Freeze all ChannelMaskLayer modules by setting frozen=True and requires_grad=False.
    """
    for m in model.modules():
        if isinstance(m, ChannelMaskLayer):
            m.frozen = True
            if hasattr(m, 'mask_weights'):
                # Set requires_grad to False to freeze the mask
                m.mask_weights.requires_grad = False


def mask_channel_bypass_penalty(model: nn.Module, args, target_layers=None) -> torch.Tensor:
    """
    Simple bypass probability penalty for ChannelMaskLayer.
    Penalizes when bypass probability is lower than target, encouraging more bypass channels.
    
    Args:
        model: The neural network model
        args: Arguments containing mask_reg_tau (temperature for softmax) and target_bypass_prob
        target_layers: Optional list of ChannelMaskLayer modules to process
    
    Returns:
        Scalar tensor with total penalty loss
    """
    total = None
    
    # Target bypass probability (default: 0.8 = 80%)
    target_bypass_prob = getattr(args, 'target_bypass_prob', 0.8)
    
    # Get temperature from args
    reg_tau = getattr(args, 'mask_reg_tau', 1.0)
    
    # If target_layers is specified, only apply regularization to those layers
    if target_layers is not None:
        modules_to_process = target_layers
    else:
        modules_to_process = [m for m in model.modules() if isinstance(m, ChannelMaskLayer)]
    
    for m in modules_to_process:
        if not isinstance(m, ChannelMaskLayer):
            continue
        
        # Skip frozen layers
        if getattr(m, 'frozen', False):
            continue
        
        # Get mask_weights: [num_channels, 3]
        if not hasattr(m, 'mask_weights'):
            continue
        
        W = m.mask_weights  # [num_channels, 3]
        dev = W.device
        
        # Compute softmax probabilities
        probs = torch.softmax(W / reg_tau, dim=-1)  # [num_channels, 3]
        
        # Get bypass probability (index 2)
        bypass_probs = probs[:, 2]  # [num_channels]
        avg_bypass_prob = bypass_probs.mean()  # scalar
        
        # Penalty: encourage bypass prob to be at least target_bypass_prob
        # If bypass_prob < target, penalty = (target - bypass_prob)
        # If bypass_prob >= target, penalty = 0
        penalty = torch.clamp(target_bypass_prob - avg_bypass_prob, min=0.0)
        
        total = penalty if total is None else total + penalty
    
    if total is None:
        p = next(model.parameters(), None)
        dev = p.device if p is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        total = torch.tensor(0.0, device=dev)
    return total


def mask_channel_kl_regularization(model: nn.Module, args, target_layers=None) -> torch.Tensor:
    """
    Compute KL divergence regularization for ChannelMaskLayer to pull mask_weights 
    towards target distribution [0.1, 0.1, 0.8] (10% 0-tie, 10% 1-tie, 80% bypass).
    
    KL(P_current || P_target) = sum(P_current * log(P_current / P_target))
    where P_current is the current softmax probability and P_target is [0.1, 0.1, 0.8].
    
    Args:
        model: The neural network model
        args: Arguments containing kl_reg_tau (temperature for softmax)
        target_layers: Optional list of ChannelMaskLayer modules to process
    
    Returns:
        Scalar tensor with total KL divergence loss
    """
    total = None
    
    # Target probabilities: [0-tie, 1-tie, bypass] = [0.1, 0.1, 0.8]
    target_probs = torch.tensor([0.49, 0.49, 0.02], device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get temperature from args
    reg_tau = getattr(args, 'mask_reg_tau', 1.0)
    
    # If target_layers is specified, only apply regularization to those layers
    if target_layers is not None:
        modules_to_process = target_layers
    else:
        modules_to_process = [m for m in model.modules() if isinstance(m, ChannelMaskLayer)]
    
    for m in modules_to_process:
        if not isinstance(m, ChannelMaskLayer):
            continue
        
        # Skip frozen layers
        if getattr(m, 'frozen', False):
            continue
        
        # Get mask_weights: [num_channels, 3]
        if not hasattr(m, 'mask_weights'):
            continue
        
        W = m.mask_weights  # [num_channels, 3]
        dev, dtype = W.device, W.dtype
        num_channels = W.shape[0]
        
        # Compute softmax probabilities
        probs = torch.softmax(W / reg_tau, dim=-1)  # [num_channels, 3]
        
        # Expand target distribution to match batch dimension
        target_dist = target_probs.to(dev).unsqueeze(0).expand(num_channels, -1)  # [num_channels, 3]
        
        # Compute KL divergence: KL(P_current || P_target)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs = torch.clamp(probs, min=eps)
        target_dist = torch.clamp(target_dist, min=eps)
        
        # KL(P || Q) = sum(P * log(P / Q))
        kl_div = probs * torch.log(probs / target_dist)
        reg = kl_div.sum(dim=-1).mean()  # Sum over operations, mean over channels
        
        total = reg if total is None else total + reg
    
    if total is None:
        p = next(model.parameters(), None)
        dev = p.device if p is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        total = torch.tensor(0.0, device=dev)
    return total


def train_step(model, x, y, loss_fn, optimizer, lam_reg_wgs, lam_reg_crossbar, clip_grad_norm, args=None, current_iter=0, total_iter=0):
    """
    [UNIFIED TRAINING FUNCTION]
    모든 학습 시나리오를 처리하는 단일 함수.
    - 일반 학습: lam_reg_wgs=0, lam_reg_exp=0, non_simple_reg=False
    - WGS 프루닝: lam_reg_wgs > 0, lam_reg_exp=0
    - Expansion 프루닝: lam_reg_wgs=0, lam_reg_exp > 0
    - Non-simple ops 정규화: non_simple_reg=True
    - 동시 프루닝: 여러 정규화 동시 적용 가능
    - mask_reg: iteration의 절반까지는 KL divergence regularization, 이후는 finetune (mask freeze)
    """
    model.train()
    output = model(x)
    
    # 1. 기본 분류 손실 (Cross-Entropy)
    task_loss = loss_fn(output, y)
    
    # device를 x에서 가져오기
    device = x.device
    
    # 2. WGS 정규화 손실 계산
    reg_loss_wgs = torch.tensor(0.0, device=device) 
    if lam_reg_wgs > 0:
        final_layer = model[1][-1]
        if isinstance(final_layer, (WeightedGroupSum)):
            reg_loss_wgs = final_layer.reg_loss()
    
    # 3. Crossbar Layer 정규화 손실 계산
    reg_loss_crossbar = torch.tensor(0.0, device=device)
    if lam_reg_crossbar > 0:
        for module in model.modules():
            # PruningBlock은 더 이상 존재하지 않음
            if isinstance(module, (CrossbarLayer, BlockEfficientCrossbarLayer)):
                if hasattr(module, 'reg_loss'):
                    reg_loss_crossbar += module.reg_loss()
    
    # 4. Mask channel pruning 정규화 손실 계산
    reg_loss_mask_channel = torch.tensor(0.0, device=device)
    mask_channel_lambda = 0.0
    if args is not None and getattr(args, 'mask_channel_prune', False):
        prune_method = getattr(args, 'prune_method', None)
        
        if prune_method == 'mask_reg':
            # mask_reg 방식: iteration의 절반까지는 regularization, 이후는 finetune
            half_iter = total_iter // 2 if total_iter > 0 else 0
            
            if current_iter < half_iter:
                # 첫 번째 절반: 일반 training (mask 학습, regularization 적용)
                mask_channel_lambda = getattr(args, 'mask_reg_lambda', 1e-2)
                mask_reg_type = getattr(args, 'mask_reg_type', 'bypass')  # 'kl' or 'bypass'
                
                if mask_reg_type == 'kl':
                    # KL divergence regularization
                    reg_loss_mask_channel = mask_channel_kl_regularization(model, args)
                else:  # 'bypass' (default)
                    # Simple bypass probability penalty
                    reg_loss_mask_channel = mask_channel_bypass_penalty(model, args)
            else:
                # 두 번째 절반: finetune (mask freeze, lam_reg=0)
                # 첫 번째로 절반 지점에 도달했을 때 mask를 freeze
                if current_iter == half_iter:
                    freeze_channel_masks(model)
                    print(f"\n  [mask_reg] Iteration {current_iter}/{total_iter}: Freezing masks and switching to finetune mode")
                mask_channel_lambda = 0.0
                reg_loss_mask_channel = torch.tensor(0.0, device=device)
        elif prune_method is None:
            # Trainable layer에만 regularization 적용 (freeze된 layer는 제외)
            reg_loss_mask_channel = 0
        # Random, Weight-based, Loss-based, Prob-based, Entropy_stuck 옵션이 켜져 있으면 
        # regularization을 적용하지 않음 (모든 mask가 freeze됨)
        
    # 5. 모든 손실을 가중합
    total_loss = (task_loss + 
                  (lam_reg_wgs * reg_loss_wgs) + 
                  (lam_reg_crossbar * reg_loss_crossbar) +
                  (mask_channel_lambda * reg_loss_mask_channel))

    # --- 역전파 과정 ---
    optimizer.zero_grad()
    total_loss.backward()

    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

    optimizer.step()
    
    return total_loss.item(), task_loss.item(), reg_loss_wgs.item(), reg_loss_crossbar.item()


def eval(model, loader, mode):
    orig_mode = model.training
    model.train(mode=mode)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            correct += (output.argmax(-1) == y).sum().item()
            total += y.size(0)
    model.train(orig_mode)
    return correct / total if total > 0 else 0




####################################################################################
############################### scheduler funcitons #################################


def get_current_tau(
    iteration: int, 
    total_iterations: int, 
    args, 
    model: nn.Module = None,
    initial_weight_sum: float = None, # WGS 동적 스케줄링을 위한 초기 가중치 합
) -> float:
    """
    [수정됨] tau 값을 계산합니다. 'dynamic' 스케줄을 포함하도록 확장되었습니다.
    """
    # --- 새로운 'dynamic' 스케줄 로직 ---
    if args.tau_sched == 'dynamic':
        if model is None:
            raise ValueError("Dynamic tau scheduling requires the 'model' argument.")
        
        wgs_layer = None
        for module in model.modules():
            if isinstance(module, WeightedGroupSum):
                wgs_layer = module
                break
        
        if wgs_layer is None:
            # WGS 레이어가 없는 경우, 기본값 반환
            return args.tau_start

        # 적응형 tau 계산
        with torch.no_grad():
            current_weight_sum = wgs_layer.weight_raw.round().sum().item()

        if initial_weight_sum is not None and initial_weight_sum > 0:
            # 비례 스케일링 공식 적용
            new_tau = args.tau_start * (current_weight_sum / initial_weight_sum)
        else:
            new_tau = args.tau_start # Fallback
        
        # tau가 너무 작아지지 않도록 tau_end를 최소값으로 사용
        return max(new_tau, args.tau_end)

    # --- 기존 시간 기반 스케줄 로직 ---
    if args.tau_start == args.tau_end:
        return args.tau_start
    
    if total_iterations <= 1:
        return args.tau_start

    progress = iteration / (total_iterations - 1)
    
    if args.tau_sched == 'linear':
        return args.tau_start + (args.tau_end - args.tau_start) * progress
    elif args.tau_sched == 'exp':
        return args.tau_start * (args.tau_end / args.tau_start) ** progress
    else: # 'none' 또는 기타
        return args.tau_start


def set_logic_layer_tau(net: nn.Module, tau_val: float):
    """Recursively find all LogicLayer modules and set their tau value."""
    for module in net.modules():
        if isinstance(module, LogicLayer):
            # LogicLayer에 tau 파라미터가 있다고 가정
            module.tau = tau_val

def set_wgs_layer_tau(net: nn.Module, tau_val: float):
    """Recursively find all LogicLayer modules and set their tau value."""
    for module in net.modules():
        if isinstance(module, WeightedGroupSum):
            # LogicLayer에 tau 파라미터가 있다고 가정
            module.tau = tau_val

  
# -----------------------------------------------------------------------------
# 4. MAIN SCRIPT (기능 추가)
# -----------------------------------------------------------------------------

def packbits_eval(model, loader, implementation='cuda', use_packbits_features=False, bit_count=64):
    """
    Evaluate conv model using PackBitsTensor for faster inference.
    Similar to main.py's packbits_eval but adapted for conv models.
    
    Args:
        model: Conv model (features + classifier)
        loader: DataLoader
        implementation: Implementation backend ('cuda' or 'triton')
        use_packbits_features: If True, use PackBitsTensor for TreeConvLayer features
        bit_count: Bit count for PackBitsTensor
    
    Returns:
        accuracy: Model accuracy
    """
    orig_mode = model.training
    device = next(model.parameters()).device
    
    with torch.no_grad():
        model.eval()
        
        if use_packbits_features:
            # Use PackBitsTensor for both features and classifier
            # This requires TreeConvLayer to support PackBitsTensor
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                
                # Forward through features with PackBitsTensor support
                if isinstance(model, nn.Sequential) and len(model) == 2:
                    features = model[0]
                    classifier = model[1]
                else:
                    # Fallback: extract classifier
                    from experiments.eval_conv_with_packbits import extract_classifier_from_conv_model
                    classifier = extract_classifier_from_conv_model(model)
                    features = None
                
                if features is not None:
                    # Forward through features (TreeConvLayer with PackBitsTensor support)
                    features_output = x
                    for module in features:
                        if isinstance(module, TreeConvLayer):
                            features_output = module(features_output, use_packbits=True, bit_count=bit_count)
                        else:
                            features_output = module(features_output)
                else:
                    # Fallback: use regular forward
                    features_output = model[0](x) if isinstance(model, nn.Sequential) else x
                
                # Flatten for classifier
                if len(features_output.shape) == 4:
                    B, C, H, W = features_output.shape
                    features_flat = features_output.view(B, -1)
                else:
                    features_flat = features_output.view(features_output.shape[0], -1)
                
                # Convert to boolean and PackBitsTensor
                features_bool = features_flat.round().bool()
                pb = PackBitsTensor(features_bool, bit_count=bit_count, device=device, implementation=implementation)
                
                # Forward through classifier
                output = classifier(pb)
                
                # Get predictions
                if isinstance(output, PackBitsTensor):
                    # Unpack if needed (shouldn't happen with GroupSum)
                    predictions = output.argmax(dim=1) if hasattr(output, 'argmax') else None
                else:
                    predictions = output.argmax(dim=1)
                
                if predictions is not None:
                    correct += (predictions == y).sum().item()
                    total += y.size(0)
            
            res = correct / total if total > 0 else 0.0
        else:
            # Use PackBitsTensor only for classifier (simpler, default)
            res = np.mean(
                [
                    (
                        model(
                            PackBitsTensor(
                                x.to(device).reshape(x.shape[0], -1).round().bool(),
                                implementation=implementation,
                                bit_count=bit_count
                            )
                        ).argmax(-1) == y.to(device)
                    ).to(torch.float32).mean().item()
                    for x, y in loader
                ]
            )
        
        model.train(mode=orig_mode)
    
    return res if isinstance(res, float) else res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Convolutional Differentiable Logic Gate Networks.')
    
    # 기본 인자
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='cifar-10-3-thresholds', choices=['cifar-10-3-thresholds', 'mnist'])
    parser.add_argument('--batch-size', '-bs', type=int, default=None)
    parser.add_argument('--learning-rate', '-lr', type=float, default=None)
    parser.add_argument('--num-iterations', '-ni', type=int, default=50000)
    parser.add_argument('--eval-freq', '-ef', type=int, default=1000)
    parser.add_argument('--valid-set-size', '-vss', type=float, default=0.1)
    parser.add_argument('--seed', '-s', type=int, default=0)

    # 모델 구조 인자
    parser.add_argument('--model-size', type=str, default='S', choices=['S','M','B','L','G'])
    parser.add_argument('--num-neurons', '-k', type=int, default=None)
    parser.add_argument('--tree-depth', '-td', type=int, default=3)
    parser.add_argument('--init', type=str, default='residual', choices=['random', 'residual'])
    parser.add_argument('--groups', type=int, default=None)
    parser.add_argument('--use-fused-tree-conv', action='store_true', default=False,
                        help='Use FusedTreeConvLayer instead of TreeConvLayer (fused CUDA kernel for logic layers, ORPool2d is separate). Only affects model creation in get_model().')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--tau', '-t', type=float, default=1.0)
    parser.add_argument('--no-ste', dest='ste', action='store_false')
    parser.add_argument('--implementation', type=str, default='triton', choices=['triton', 'im2col'],
                        help='Implementation type: "triton" (FusedLogicTreeBlock) or "im2col" (TreeConvLayer with crossbar support)')

    # --- 새로 추가된 인자 ---
    parser.add_argument('--classifier-arch', type=str, default='groupsum', choices=['groupsum', 'wgs', 'soft_wgs', 'im2col_sparse'],
                        help='Architecture for the final classifier layer.')
    parser.add_argument('--retrain-eid', type=str, default=None,
                        help='Path to a pre-trained model file to load and retrain/prune. Can be: (1) Integer (e.g., "700090") -> loads "results_conv/700090.pth", (2) Full path (e.g., "results_conv/700090_mask_prune_phase3.pt") -> loads directly. .pt files are loaded as model objects, .pth files are loaded as state_dict.')
    parser.add_argument('--pruned-eid', type=int, default=None,
                        help='New experiment ID for the pruned/retrained model.')
    parser.add_argument('--wgs-lam-reg', type=float, default=1e-5,
                        help='L1 regularization lambda for WeightedGroupSum during retraining.')


    parser.add_argument('--scheduler', type=str, default='none', choices=['cosine', 'step', 'none'])
    parser.add_argument('--scheduler-step-size', type=int, default=10000)
    parser.add_argument('--scheduler-gamma', type=float, default=0.1)

    parser.add_argument('--tau-start', type=float, default=40,
                        help='Initial tau for LogicLayers at the beginning of training.')
    parser.add_argument('--tau-end', type=float, default=40,
                        help='Final tau for LogicLayers at the end of training.')
    parser.add_argument('--tau-sched', type=str, default='exp', choices=['linear', 'exp', 'dynamic'],
                        help='Schedule for tau annealing (linear or exponential).')
    
    parser.add_argument('--clip-grad', type=float, default=0.0,
                        help='Max norm for gradient clipping. (e.g., 1.0). Set to 0 to disable.')

    # [추가] 2단계 재학습을 위한 단계별 반복 횟수

    parser.add_argument('--crossbar-retrain', action='store_true', default=False,
                        help='crossbar retrain for features: sequentially convert Crossbar1x1Conv connections from unique to ste.')
    parser.add_argument('--crossbar-retrain-iterative', action='store_true', default=False,
                        help='If set, insert all crossbars at once, then reinitialize and retrain iteratively (iterative mode).')
    parser.add_argument('--block-size', type=int, default=128,
                        help='block size for crossbar. If num_blocks is not specified, will be used to calculate num_blocks.')
    parser.add_argument('--num-blocks', type=int, default=None,
                        help='number of blocks for crossbar. If not specified, will be calculated from block_size and out_dim.')
    parser.add_argument('--crossbar-retrain-start-idx', type=int, default=None,
                        help='Starting Crossbar1x1Conv index for crossbar retraining (0-based, excluding first layer).')
    parser.add_argument('--crossbar-retrain-end-idx', type=int, default=None,
                        help='Ending Crossbar1x1Conv index for crossbar retraining (0-based, excluding first layer, inclusive).')
    parser.add_argument('--crossbar-bottleneck', action='store_true', default=False,
                        help='Use bottleneck structure (two crossbars in Sequential) for crossbar retraining.')
    parser.add_argument('--crossbar-connections', type=str, default='ste', choices=['ste', 'softmax', 'unique'],
                        help='Connection type for crossbar layers (default: ste).')
    parser.add_argument('--bottleneck-factor', type=float, default=0.125,
                        help='Bottleneck factor (ratio) for crossbar retraining. Default: 0.125 (1/8).')

                        
    parser.add_argument('--mask-channel-prune', action='store_true', default=False,
                        help='Use MaskLayer for channel pruning after TreeConvLayer layers.')
    parser.add_argument('--mask-channel-prune-oneshot', action='store_true', default=False,
                        help='One-shot mask channel pruning: collect scores from all layers and select top prune_pct globally across all layers.')
    parser.add_argument('--mask-channel-prune-lambda', type=float, default=1e-2,
                        help='Regularization strength for mask channel pruning.')
    parser.add_argument('--mask-channel-prune-tau', type=float, default=1.0,
                        help='Temperature parameter for mask channel pruning.')
    parser.add_argument('--prune-pct', type=float, default=None,
                        help='Target prune percentage for mask channel pruning. Applied uniformly to all layers.')
    parser.add_argument('--prune-method', type=str, default=None, choices=['random', 'weight', 'loss', 'prob', 'entropy_stuck', 'mask_reg', 'loss_approx'],
                        help='Pruning method for mask channel pruning: "random" (randomly select channels), "weight" (weight-based selection), "loss" (loss-based selection), "prob" (probability-based selection), "entropy_stuck" (entropy-based stuck neuron detection), "mask_reg" (KL divergence regularization to learn mask). If None, mask will be trained. The mask will be frozen after initialization when using these methods (except mask_reg).')
    parser.add_argument('--mask-channel-prune-loss-samples', type=int, default=100,
                        help='Number of samples to use for loss-based pruning evaluation (default: 100).')
    parser.add_argument('--loss-prune-type', type=str, default='mse', choices=['l1', 'mse'],
                        help='Loss type for loss-based pruning: "l1" (L1 loss) or "mse" (MSE loss, default).')
    parser.add_argument('--stuck-max-batches', type=int, default=10,
                        help='Number of batches to analyze for entropy-based stuck neuron detection (default: 10).')
    parser.add_argument('--mask-reg-lambda', type=float, default=1e-2,
                        help='Regularization strength for mask_reg pruning method (default: 1e-2).')
    parser.add_argument('--mask-reg-tau', type=float, default=1.0,
                        help='Temperature parameter for softmax in mask_reg regularization (default: 1.0).')
    parser.add_argument('--mask-reg-type', type=str, default='bypass', choices=['kl', 'bypass'],
                        help='Regularization type for mask_reg: "kl" (KL divergence) or "bypass" (simple bypass probability penalty, default).')
    parser.add_argument('--target-bypass-prob', type=float, default=0.8,
                        help='Target bypass probability for mask_reg bypass penalty method (default: 0.8 = 80%%).')
    parser.add_argument('--target-layer-idx', type=int, default=None,
                        help='Target layer index for mask channel pruning. If None, all layers will be processed.')
    parser.add_argument('--iterative', action='store_true', default=False,
                        help='Iterative all mode for mask channel pruning.')
    parser.add_argument('--reverse-iterative', action='store_true', default=False,
                        help='Process layers in reverse order (from last to first) in iterative mode. Phase numbers still start from 1.')
    parser.add_argument('--resume-from-phase', type=int, default=None,
                        help='Resume mask channel pruning from a specific phase (1-indexed). Loads the model from results_conv/{pruned_eid}_mask_prune_phase{N}.pt and continues from phase N+1.')
    parser.add_argument('--residual-mask', action='store_true', default=False,
                        help='Use ResidualChannelMaskLayer instead of ChannelMaskLayer. Supports bypass-a, bypass-b, neg bypass-a, neg bypass-b, and bypass operations.')
    parser.add_argument('--residual-mask-include-tie', action='store_true', default=False,
                        help='Include 0-tie and 1-tie options in ResidualChannelMaskLayer (7 options total). Default: False (5 options: bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass).')
    
    parser.add_argument('--weight-prune', action='store_true', default=False,
                        help='Weight pruning: directly modify LogicLayer weights to tie operations based on weight softmax probabilities.')
    parser.add_argument('--weight-prune-pct', type=float, default=None,
                        help='Target prune percentage for weight pruning. Applied uniformly to all layers.')
    parser.add_argument('--weight-prune-iterative', action='store_true', default=False,
                        help='Iterative mode for weight pruning: process layers sequentially.')

    parser.add_argument('--improve-acc', action='store_true', default=False,
                        help='improve accuracy for the final model.')
    parser.add_argument('--improve-start-progress', type=float, default=0.8,
                        help='the start progress of the improve phase.')
    parser.add_argument('--improve-learning-rate', type=float, default=0.001,
                        help='the learning rate of the improve phase.')

                        
    # -------------------------

    args = parser.parse_args()
    
    print(vars(args))

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
    # -----------------------------------------------------------------

    assert args.num_iterations % args.eval_freq == 0
    if args.experiment_id is not None:
        results = ResultsJSON(eid=args.experiment_id, path='./results_conv/')
        results.store_args(args)
        
        # wandb 초기화
        if WANDB_AVAILABLE:
            wandb.init(
                project="birel",
                name=f"exp_{args.experiment_id}",
                id=f"exp_{args.experiment_id}",
                config=vars(args),
                reinit=True
            )
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader, in_channels = load_dataset(args)
        
    model, loss_fn, optim = get_model(args, in_channels)
    
    # 기본 optimizer 생성 함수 (재사용)
    def create_default_optimizer(model):
        """기본 optimizer 생성 (lr=args.learning_rate, weight_decay=0.002)"""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=0.002
        )
    
    # Finetune용 optimizer 생성 함수 (lr이 낮음)
    def create_finetune_optimizer(model, lr_multiplier=0.1):
        """Finetune용 optimizer 생성 (lr=args.learning_rate * lr_multiplier, weight_decay=0.002)"""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate * lr_multiplier,
            weight_decay=0.002
        )
    
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_iterations, eta_min=0)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        scheduler = None


    # --- 재학습/가지치기 로직 ---
    if args.retrain_eid is not None:
        if args.retrain_eid.isdigit():
            model_path = f'results_conv/{args.retrain_eid}.pt'
        else:
            model_path = args.retrain_eid
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}"); exit()
        
        print(f"\n[RETRAIN MODE] Loading model from: {model_path}")
        
        
        try:
            loaded_data = torch.load(model_path, map_location=device)
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model = loaded_data['model']
                if 'args' in loaded_data:
                    print(f"  Model metadata: {loaded_data['args']}")
            elif isinstance(loaded_data, torch.nn.Module):
                # 모델 객체 자체가 저장된 경우
                #state_dict = loaded_data.state_dict()
                #model.load_state_dict(state_dict)
                model = loaded_data
            elif isinstance(loaded_data, dict) and not isinstance(loaded_data, torch.nn.Module):
                # state_dict만 저장된 경우 (OrderedDict)
                print(f"  Detected state_dict format. Loading into new model...")
                model, loss_fn, _ = get_model(args, in_channels)
                model.load_state_dict(loaded_data)
            else:
                model = loaded_data
            
            # model이 실제로 nn.Module인지 확인
            if isinstance(model, torch.nn.Module):
                model.to(device)
            else:
                raise ValueError(f"Loaded data is not a model object: {type(model)}")
        except Exception as e:
            print(f"ERROR: Failed to load .pt model: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        

        acc_before = eval(model, test_loader, mode=False)
        print(f"Accuracy of loaded model before pruning: {acc_before:.4f}")

        # retrain 모드에서 wandb 초기화 (pruned_eid 또는 retrain_eid 사용)
        if WANDB_AVAILABLE:
            retrain_eid = args.pruned_eid if args.pruned_eid is not None else args.retrain_eid
            if retrain_eid is not None:
                # wandb id는 파일 경로를 직접 사용할 수 없으므로 안전한 문자열로 변환
                if isinstance(retrain_eid, str) and (retrain_eid.endswith('.pt') or retrain_eid.endswith('.pth') or '/' in retrain_eid):
                    # 파일 경로인 경우: 파일명만 추출하고 확장자 제거
                    import os
                    safe_id = os.path.basename(retrain_eid).replace('.pt', '').replace('.pth', '')
                    # 허용되지 않는 문자 제거
                    safe_id = safe_id.replace(':', '_').replace(';', '_').replace(',', '_').replace('#', '_').replace('?', '_').replace('/', '_').replace("'", '_')
                else:
                    # 숫자나 일반 문자열인 경우 그대로 사용
                    safe_id = str(retrain_eid)
                
                wandb.init(
                    project="birel-retrain",
                    name=f"retrain_{safe_id}",
                    id=f"retrain_{safe_id}",
                    config=vars(args),
                    reinit=True
                )
                wandb.log({'results/acc_initial': acc_before})

        print("\n--- Running Pruning Analysis (Channel-wise) ---")
        analysis_results = finding_live_nodes_by_channel_with_fusion(
            model, in_channels, args, device='cuda', verbose=False
        )
        summarize_and_print_analysis(analysis_results)
        
        # 분석 결과 wandb 로깅
        if WANDB_AVAILABLE and args.pruned_eid is not None:
            stats = analysis_results.get('stats', {})
            total_dead = sum(res.get('dead', 0) for key, res in stats.items() if key != 'classifier_input')
            total_nodes = sum(res.get('total', 0) for key, res in stats.items() if key != 'classifier_input')
            overall_dead_ratio = (100 * total_dead / total_nodes) if total_nodes > 0 else 0.0
            feat_dead_ratio = stats.get('features', {}).get('dead_ratio', 0.0)
            cl_int_dead_ratio = stats.get('classifier_internal', {}).get('dead_ratio', 0.0)
            wandb.log({
                'results/dead_ratio': overall_dead_ratio,
            })
        train_iter = iter(load_n(train_loader, args.num_iterations * 2))  # prune + recovery

        # ================== Weight Pruning ==================
        if args.weight_prune:
            if args.implementation != 'im2col':
                print(f"ERROR: --weight-prune requires --implementation im2col")
                exit()
            
            # pruned_results 초기화
            pruned_results = None
            if args.pruned_eid:
                try:
                    pruned_results = ResultsJSON(eid=args.pruned_eid, path='./results_conv/')
                except:
                    pruned_results = None
            
            print("\n--- Weight Pruning Mode: Directly modify LogicLayer weights to tie operations ---")
            print("\n--- Model structure before weight pruning ---")
            print(model)
            
            # pruning을 진행할 타켓 레이어 설정
            all_layers = []
            
            
            # TreeConvLayer의 cascade 내부 마지막 LogicLayer만 찾기 (출력 레이어)
            if isinstance(model[0], nn.Sequential):
                for idx, module in enumerate(model[0]):
                    if isinstance(module, TreeConvLayer):
                        if hasattr(module, 'cascade'):
                            # cascade의 마지막 LogicLayer 찾기 (출력 레이어)
                            for cascade_module in reversed(module.cascade):
                                if isinstance(cascade_module, LogicLayer):
                                    if cascade_module.out_dim == module.out_dim:
                                        all_layers.append(('treeconv', idx, None, f"features_{idx}", cascade_module, 0))
                                        break
            
            # Classifier의 LogicLayer들 찾기
            if isinstance(model[1], nn.Sequential):
                for idx, module in enumerate(model[1][:-1]):  # 마지막 레이어 제외
                    if isinstance(module, LogicLayer):
                        all_layers.append(('logic', idx, None, f"classifier_{idx}", module, 1))
            
            print(f"Found {len(all_layers)} LogicLayers to process:")
            for idx, (layer_type, pos, cascade_pos, name, module, seq_idx) in enumerate(all_layers):
                print(f"  {idx}. {name} (out_dim={module.out_dim})")
            
            is_iterative = getattr(args, 'iterative', False)
            
            print(f"\n{'='*80}")
            if is_iterative:
                print(f"ITERATIVE MODE: Process layers sequentially")
            else:
                print(f"ONE-SHOT MODE: Process all layers at once")
            print(f"{'='*80}")
            
            num_total_phases = len(all_layers)
            target_pct = getattr(args, 'prune_pct', None)
            
            if target_pct is None:
                print(f"ERROR: --weight-prune-pct must be specified for weight pruning")
                exit()
            
            print(f"Target prune percentage: {target_pct:.2f}% (applied to all layers)")
            
            for phase_idx, (layer_type, pos, cascade_pos, layer_name, logic_layer, seq_idx) in enumerate(all_layers):
                layer_type_str = "TreeConv" if layer_type == 'treeconv' else "Classifier"
                print(f"\n{'='*80}")
                print(f"Phase {phase_idx + 1}/{num_total_phases}: Processing {layer_name} ({layer_type_str})")
                print(f"{'='*80}")
                
                out_dim = logic_layer.out_dim
                target_prune_neurons = int(out_dim * target_pct / 100.0)
                print(f"  Target prune neurons: {target_prune_neurons}/{out_dim}")
                
                # Weight softmax 계산
                logic_weights = logic_layer.weights.data  # [out_dim, 16]
                logic_probs = torch.softmax(logic_weights, dim=-1)  # [out_dim, 16]
                
                prob_0tie = logic_probs[:, 0]  # [out_dim]
                prob_1tie = logic_probs[:, 15]  # [out_dim]
                
                # 높은 tie probability를 가진 neuron 선택
                score = torch.maximum(prob_0tie, prob_1tie)  # [out_dim] - 높을수록 prune할 가치가 높음
                tie_type_mask = (prob_1tie > prob_0tie)  # True면 1-tie, False면 0-tie
                
                # Score 기반으로 정렬해서 top prune pct만큼 선택
                _, sorted_indices = torch.sort(score, descending=True)
                selected_indices = sorted_indices[:target_prune_neurons]
                
                # LogicLayer의 weight를 직접 tie로 교체
                with torch.no_grad():
                    for neuron_idx in selected_indices:
                        neuron_idx = neuron_idx.item()
                        if tie_type_mask[neuron_idx]:
                            # 1-tie로 교체: weight를 [-10, -10, ..., -10, 10] 형태로 (15번 인덱스만 높은 값)
                            logic_layer.weights.data[neuron_idx, :] = -10.0
                            logic_layer.weights.data[neuron_idx, 15] = 10.0  # 높은 값으로 설정
                        else:
                            # 0-tie로 교체: weight를 [10, -10, ..., -10] 형태로 (0번 인덱스만 높은 값)
                            logic_layer.weights.data[neuron_idx, :] = -10.0
                            logic_layer.weights.data[neuron_idx, 0] = 10.0  # 높은 값으로 설정
                        
                        # indices 설정 (tie operation에는 필요 없지만 일관성을 위해)
                        if hasattr(logic_layer, 'indices') and logic_layer.indices is not None:
                            logic_layer.indices[0][neuron_idx] = 0
                            logic_layer.indices[1][neuron_idx] = 0
                
                # Weight 수정 후 requires_grad 유지 (재학습을 위해)
                logic_layer.weights.requires_grad = True
                
                num_0tie = (~tie_type_mask[selected_indices]).sum().item()
                num_1tie = tie_type_mask[selected_indices].sum().item()
                print(f"  Pruned {target_prune_neurons} neurons: {num_0tie} to 0-tie, {num_1tie} to 1-tie")
                
                if is_iterative:
                    # Iterative 모드: 각 레이어 처리 후 재학습
                    print(f"\n  === Phase {phase_idx + 1}/{num_total_phases} Retraining ({layer_type_str}) ===")
                    
                    retrain_loader = train_loader
                    weight_prune_optimizer = create_default_optimizer(model)
                    acc_before_phase = eval(model, test_loader, mode=False)
                    print(f"  Accuracy before phase {phase_idx + 1}: {acc_before_phase:.4f}")
                    
                    best_phase_acc = acc_before_phase
                    best_phase_model_state = copy.deepcopy(model.state_dict())
                    
                    train_iter_phase = iter(load_n(retrain_loader, args.num_iterations))
                    
                    model.train()
                    for i in tqdm(range(args.num_iterations), desc=f'Phase {phase_idx+1}/{num_total_phases} retraining'):
                        x, y = next(train_iter_phase)
                        x, y = x.to(device), y.to(device)
                        
                        train_step(model, x, y, loss_fn, weight_prune_optimizer, 0.0, 0.0, args.clip_grad, args,
                                   current_iter=i, total_iter=args.num_iterations)
                        
                        if (i + 1) % args.eval_freq == 0:
                            model.eval()
                            with torch.no_grad():
                                test_acc = eval(model, test_loader, mode=False)
                            model.train()
                            
                            if test_acc > best_phase_acc:
                                best_phase_acc = test_acc
                                best_phase_model_state = copy.deepcopy(model.state_dict())
                                print(f"\n  Iter {i+1}: New best accuracy in phase {phase_idx+1}: {test_acc:.4f}")
                    
                    model.load_state_dict(best_phase_model_state)
                    acc_after_phase = eval(model, test_loader, mode=False)
                    print(f"\n  Phase {phase_idx + 1} retraining finished. Best accuracy: {best_phase_acc:.4f}")
                    print(f"  Accuracy improvement: {acc_after_phase - acc_before_phase:+.4f}")
                    
                    if args.pruned_eid:
                        save_path = f'results_conv/{args.pruned_eid}_weight_prune_phase{phase_idx+1}.pt'
                        remove_residual_mask_hooks(model)
                        torch.save(model, save_path)
                        print(f"  ✓ Saved best model: {save_path}")
                    
                    analysis_bundle_phase = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
                    summarize_and_print_analysis(analysis_bundle_phase)
                    
                    if WANDB_AVAILABLE and args.pruned_eid is not None:
                        stats_phase = analysis_bundle_phase.get('stats', {})
                        total_dead_phase = sum(res.get('dead', 0) for key, res in stats_phase.items() if key != 'classifier_input')
                        total_nodes_phase = sum(res.get('total', 0) for key, res in stats_phase.items() if key != 'classifier_input')
                        overall_dead_ratio_phase = (100 * total_dead_phase / total_nodes_phase) if total_nodes_phase > 0 else 0.0
                        wandb.log({
                            f"results/phase_{phase_idx+1}_best_acc": best_phase_acc,
                            f"results/phase_{phase_idx+1}_dead_ratio": overall_dead_ratio_phase
                        })
                else:
                    print(f"\n  === Phase {phase_idx + 1}/{num_total_phases} (One-shot mode: skipping training) ===")
            
            # Iterative 모드일 때 최종 결과 출력
            if is_iterative:
                final_acc = eval(model, test_loader, mode=False)
                print(f"\n{'='*80}")
                print(f"Weight Pruning (iterative mode) completed. Final accuracy: {final_acc:.4f}")
                print(f"{'='*80}")
            
            # One-shot 모드일 때 모든 레이어 처리 후 한 번에 재학습
            if not is_iterative:
                print(f"\n{'='*80}")
                print(f"One-shot training: Retraining after all weight pruning")
                print(f"{'='*80}")
                
                # 모든 파라미터 활성화
                for param in model.parameters():
                    param.requires_grad = True
                
                # Optimizer 설정
                weight_prune_optimizer = create_default_optimizer(model)
                
                # 초기 accuracy 측정
                acc_before_retrain = eval(model, test_loader, mode=False)
                print(f"\n  Accuracy before retraining: {acc_before_retrain:.4f}")
                
                best_retrain_acc = acc_before_retrain
                best_retrain_model_state = copy.deepcopy(model.state_dict())
                
                # Retraining loop
                train_iter_retrain = iter(load_n(train_loader, args.num_iterations))
                
                for i in tqdm(range(args.num_iterations), desc='Weight pruning retraining'):
                    x, y = next(train_iter_retrain)
                    x, y = x.to(device), y.to(device)
                    
                    train_step(model, x, y, loss_fn, weight_prune_optimizer, 0.0, 0.0, args.clip_grad, args,
                               current_iter=i, total_iter=args.num_iterations)
                    
                    if (i + 1) % args.eval_freq == 0:
                        test_acc = eval(model, test_loader, mode=False)
                        
                        print(f"\n  Iter {i+1}: Test Acc={test_acc:.4f}")
                        
                        if test_acc > best_retrain_acc:
                            best_retrain_acc = test_acc
                            best_retrain_model_state = copy.deepcopy(model.state_dict())
                            print(f"  *** New best accuracy: {test_acc:.4f} ***")
                
                if best_retrain_model_state is not None:
                    model.load_state_dict(best_retrain_model_state)
                
                acc_after_retrain = eval(model, test_loader, mode=False)
                print(f"\n  Retraining finished. Best accuracy: {best_retrain_acc:.4f}")
                print(f"  Accuracy improvement: {acc_after_retrain - acc_before_retrain:+.4f}")
                
                final_acc = eval(model, test_loader, mode=False)
                print(f"\n{'='*80}")
                print(f"Weight Pruning (one-shot mode) completed. Final accuracy: {final_acc:.4f}")
                print(f"  Initial accuracy: {acc_before_retrain:.4f}")
                print(f"  Final accuracy: {final_acc:.4f}")
                print(f"  Improvement: {final_acc - acc_before_retrain:+.4f}")
                print(f"{'='*80}")
                
                # wandb 로깅
                if WANDB_AVAILABLE and args.pruned_eid is not None:
                    final_analysis = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
                    stats_final = final_analysis.get('stats', {})
                    total_dead_final = sum(res.get('dead', 0) for key, res in stats_final.items() if key != 'classifier_input')
                    total_nodes_final = sum(res.get('total', 0) for key, res in stats_final.items() if key != 'classifier_input')
                    overall_dead_ratio_final = (100 * total_dead_final / total_nodes_final) if total_nodes_final > 0 else 0.0
                    
                    wandb.log({
                        "results/oneshot_best_acc": best_retrain_acc,
                        "results/oneshot_dead_ratio": overall_dead_ratio_final
                    })
            
            # 최종 분석
            print("\n--- Final Analysis after Weight Pruning ---")
            analysis_bundle_final = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=True)
            summarize_and_print_analysis(analysis_bundle_final)
            
            if args.pruned_eid:
                torch.save(model, f'results_conv/{args.pruned_eid}_weight_prune_final.pt')
                if pruned_results:
                    summary_data = {
                        'source_eid': args.retrain_eid,
                        'accuracy_before_prune': acc_before,
                        'final_accuracy': final_acc,
                        'improvement': final_acc - acc_before,
                        'weight_prune_pct': target_pct,
                    }
                    pruned_results.store_final_results(summary_data)
                    pruned_results.save()
            
            exit()

        # ================== Mask Channel Prune One-Shot ==================
        if args.mask_channel_prune_oneshot:
            if args.implementation != 'im2col':
                print(f"ERROR: --mask-channel-prune-oneshot requires --implementation im2col")
                exit()
            
            # pruned_results 초기화
            pruned_results = None
            if args.pruned_eid:
                try:
                    pruned_results = ResultsJSON(eid=args.pruned_eid, path='./results_conv/')
                except:
                    pruned_results = None
            
            print("\n--- Mask Channel Pruning One-Shot Mode: Collect scores from all layers and select top prune_pct globally ---")
            print("\n--- Model structure before mask channel pruning ---")
            print(model)
            
            # 모든 타겟 레이어 수집
            all_layers = []
            
            if isinstance(model[0], nn.Sequential):
                for idx, module in enumerate(model[0]):
                    if isinstance(module, TreeConvLayer):
                        all_layers.append(('treeconv', idx, f"features_{idx}", module, 0))
            
            if isinstance(model[1], nn.Sequential):
                for idx, module in enumerate(model[1][:-1]):  # 마지막 레이어 제외
                    if isinstance(module, LogicLayer):
                        all_layers.append(('logic', idx, f"classifier_{idx}", module, 1))
            
            print(f"Found {len(all_layers)} layers to process:")
            for idx, (layer_type, pos, name, module, seq_idx) in enumerate(all_layers):
                if layer_type == 'treeconv':
                    print(f"  {idx}. {name} (features[{pos}], out_dim={module.out_dim})")
                else:
                    print(f"  {idx}. {name} (classifier[{pos}], out_dim={module.out_dim})")
            
            target_pct = getattr(args, 'prune_pct', None)
            if target_pct is None:
                print(f"ERROR: --prune-pct must be specified for mask_channel_prune_oneshot")
                exit()
            
            prune_method = getattr(args, 'prune_method', None)
            if prune_method is None:
                print(f"ERROR: --prune-method must be specified for mask_channel_prune_oneshot")
                exit()
            
            use_residual_mask = getattr(args, 'residual_mask', False)
            residual_mask_include_tie = getattr(args, 'residual_mask_include_tie', False)
            mask_frozen = (prune_method != 'mask_reg')
            
            print(f"\n{'='*80}")
            print(f"ONE-SHOT MODE: Collecting scores from all {len(all_layers)} layers")
            print(f"Target prune percentage: {target_pct:.2f}% (global across all layers)")
            print(f"Prune method: {prune_method}")
            print(f"{'='*80}")
            
            # Step 1: 모든 layer의 score 수집
            all_scores = []  # (layer_idx, channel_idx, score, tie_type/residual_type, layer_info)
            layer_info_list = []  # 각 layer의 정보 저장
            
            for layer_idx, (layer_type, pos, layer_name, module, seq_idx) in enumerate(all_layers):
                print(f"\n  Processing layer {layer_idx + 1}/{len(all_layers)}: {layer_name}")
                
                out_channels = module.out_dim
                score = None
                tie_type_mask = None
                residual_type_mask = None
                
                # Score 계산 (기존 로직 재사용)
                if prune_method == 'weight':
                    target_logic_layer = None
                    if layer_type == 'treeconv':
                        if hasattr(module, 'cascade'):
                            for logic_module in reversed(module.cascade):
                                if isinstance(logic_module, LogicLayer):
                                    if logic_module.out_dim == module.out_dim:
                                        target_logic_layer = logic_module
                                        break
                    else:  # logic
                        target_logic_layer = module
                    
                    if target_logic_layer is not None:
                        logic_weights = target_logic_layer.weights.data
                        logic_probs = torch.softmax(logic_weights, dim=-1)
                        
                        if use_residual_mask:
                            if residual_mask_include_tie:
                                prob_0tie = logic_probs[:, 0]
                                prob_1tie = logic_probs[:, 15]
                                prob_bypass_a = logic_probs[:, 3]
                                prob_bypass_b = logic_probs[:, 5]
                                prob_neg_bypass_b = logic_probs[:, 10]
                                prob_neg_bypass_a = logic_probs[:, 12]
                                
                                residual_probs = torch.stack([
                                    prob_0tie, prob_1tie, prob_bypass_a, prob_bypass_b,
                                    prob_neg_bypass_a, prob_neg_bypass_b
                                ], dim=-1)
                                max_residual_score, max_residual_op = torch.max(residual_probs, dim=-1)
                                score = max_residual_score
                                residual_type_mask = max_residual_op
                            else:
                                prob_bypass_a = logic_probs[:, 3]
                                prob_bypass_b = logic_probs[:, 5]
                                prob_neg_bypass_b = logic_probs[:, 10]
                                prob_neg_bypass_a = logic_probs[:, 12]
                                
                                residual_probs = torch.stack([
                                    prob_bypass_a, prob_bypass_b, prob_neg_bypass_a, prob_neg_bypass_b
                                ], dim=-1)
                                max_residual_score, max_residual_op = torch.max(residual_probs, dim=-1)
                                score = max_residual_score
                                residual_type_mask = max_residual_op
                        else:
                            prob_0tie = logic_probs[:, 0]
                            prob_1tie = logic_probs[:, 15]
                            score = torch.maximum(prob_0tie, prob_1tie)
                            tie_type_mask = (prob_1tie > prob_0tie)
                
                elif prune_method == 'random':
                    score = torch.rand(out_channels, device='cuda')
                    tie_type_mask = torch.rand(out_channels, device='cuda') > 0.5
                
                elif prune_method == 'entropy_stuck':
                    target_layer_name = None
                    target_module_for_hook = None
                    
                    if layer_type == 'treeconv':
                        target_module_for_hook = module
                        for name, m in model.named_modules():
                            if id(m) == id(module):
                                target_layer_name = name
                                break
                    else:
                        for name, m in model.named_modules():
                            if id(m) == id(module):
                                target_layer_name = name
                                break
                    
                    if target_module_for_hook is not None or target_layer_name is not None:
                        stuck_max_batches = getattr(args, 'stuck_max_batches', 10)
                        stuck_list = detect_stuck_by_entropy_single_layer(
                            model, train_loader,
                            target_layer_name=target_layer_name,
                            target_module=target_module_for_hook,
                            prune_pct=1.0,  # 전체를 분석하기 위해 100%로 설정
                            max_batches=stuck_max_batches,
                            device=device
                        )
                        
                        if stuck_list:
                            score = torch.zeros(out_channels, device='cuda')
                            tie_type_mask = torch.zeros(out_channels, dtype=torch.bool, device='cuda')
                            
                            for layer_name, channel_idx, stuck_type, avg_prob, entropy in stuck_list:
                                if channel_idx < out_channels:
                                    score[channel_idx] = 1.0 / (1.0 + entropy)
                                    tie_type_mask[channel_idx] = (stuck_type == 'sa1')
                        else:
                            print(f"    Warning: No stuck neurons found. Using random scores.")
                            score = torch.rand(out_channels, device='cuda')
                            tie_type_mask = torch.rand(out_channels, device='cuda') > 0.5
                
                else:
                    print(f"    ERROR: Prune method '{prune_method}' not supported in oneshot mode yet.")
                    exit()
                
                if score is None:
                    print(f"    ERROR: Could not compute score for layer {layer_name}")
                    exit()
                
                # 모든 채널의 score를 all_scores에 추가
                for channel_idx in range(out_channels):
                    if use_residual_mask:
                        if residual_type_mask is not None:
                            all_scores.append((
                                layer_idx, channel_idx, score[channel_idx].item(),
                                residual_type_mask[channel_idx].item(), (layer_type, pos, layer_name, module, seq_idx)
                            ))
                        else:
                            all_scores.append((
                                layer_idx, channel_idx, score[channel_idx].item(),
                                0, (layer_type, pos, layer_name, module, seq_idx)
                            ))
                    else:
                        tie_type = 1 if (tie_type_mask is not None and tie_type_mask[channel_idx]) else 0
                        all_scores.append((
                            layer_idx, channel_idx, score[channel_idx].item(),
                            tie_type, (layer_type, pos, layer_name, module, seq_idx)
                        ))
                
                # Layer 정보 저장
                layer_info_list.append({
                    'layer_idx': layer_idx,
                    'layer_type': layer_type,
                    'pos': pos,
                    'layer_name': layer_name,
                    'module': module,
                    'seq_idx': seq_idx,
                    'out_channels': out_channels,
                    'tie_type_mask': tie_type_mask,
                    'residual_type_mask': residual_type_mask
                })
                
                print(f"    Collected {out_channels} channel scores (min={score.min().item():.4f}, max={score.max().item():.4f}, mean={score.mean().item():.4f})")
            
            # Step 2: 전체 score를 기준으로 정렬하여 top prune_pct 선택
            print(f"\n  Total channels across all layers: {len(all_scores)}")
            all_scores.sort(key=lambda x: x[2], reverse=True)  # score 기준 내림차순 정렬
            
            total_channels = len(all_scores)
            target_prune_channels = int(total_channels * target_pct / 100.0)
            selected_channels = all_scores[:target_prune_channels]
            
            print(f"  Selected top {target_prune_channels}/{total_channels} channels ({target_pct:.2f}%) for pruning")
            
            # Step 3: Layer별로 선택된 채널 그룹화
            layer_selected_channels = {}  # layer_idx -> list of (channel_idx, score, type)
            for layer_idx, channel_idx, score, type_val, layer_info in selected_channels:
                if layer_idx not in layer_selected_channels:
                    layer_selected_channels[layer_idx] = []
                layer_selected_channels[layer_idx].append((channel_idx, score, type_val))
            
            # Step 4: 각 layer에 mask 삽입 및 초기화
            print(f"\n  Inserting and initializing masks for {len(layer_selected_channels)} layers...")
            
            for layer_idx, layer_info in enumerate(layer_info_list):
                layer_type = layer_info['layer_type']
                pos = layer_info['pos']
                layer_name = layer_info['layer_name']
                module = layer_info['module']
                seq_idx = layer_info['seq_idx']
                out_channels = layer_info['out_channels']
                tie_type_mask = layer_info['tie_type_mask']
                residual_type_mask = layer_info['residual_type_mask']
                
                current_modules = list(model[seq_idx])
                new_modules = []
                mask_inserted = False
                
                for idx, m in enumerate(current_modules):
                    new_modules.append(m)
                    if id(m) == id(module) and not mask_inserted:
                        # 이미 mask가 있는지 확인
                        has_mask_already = False
                        if idx + 1 < len(current_modules):
                            next_module = current_modules[idx + 1]
                            if isinstance(next_module, nn.Sequential):
                                if len(next_module) > 0 and isinstance(next_module[0], ChannelMaskLayer):
                                    has_mask_already = True
                            elif isinstance(next_module, (ChannelMaskLayer, ResidualChannelMaskLayer)):
                                has_mask_already = True
                        
                        if has_mask_already:
                            continue
                        
                        # Mask layer 생성
                        prev_logic_layer = None
                        if use_residual_mask:
                            if layer_type == 'treeconv':
                                if hasattr(module, 'cascade'):
                                    for logic_module in reversed(module.cascade):
                                        if isinstance(logic_module, LogicLayer):
                                            if logic_module.out_dim == module.out_dim:
                                                prev_logic_layer = logic_module
                                                break
                            elif layer_type == 'logic':
                                prev_logic_layer = module
                            
                            from birel.conv import ResidualChannelMaskLayer
                            if prev_logic_layer is None:
                                print(f"    ERROR: Could not find prev_logic_layer for {layer_name}")
                                exit()
                            mask_layer = ResidualChannelMaskLayer(
                                num_channels=out_channels,
                                prev_logic_layer=prev_logic_layer,
                                device='cuda',
                                frozen=mask_frozen,
                                include_tie=residual_mask_include_tie
                            )
                        else:
                            mask_layer = ChannelMaskLayer(num_channels=out_channels, device='cuda', frozen=mask_frozen)
                        
                        # 선택된 채널에 대해 mask 초기화
                        selected_for_this_layer = layer_selected_channels.get(layer_idx, [])
                        all_indices = torch.arange(out_channels, device='cuda')
                        
                        with torch.no_grad():
                            mask_layer.mask_weights.fill_(0.0)
                            
                            if len(selected_for_this_layer) > 0:
                                prune_indices = torch.tensor([ch_idx for ch_idx, _, _ in selected_for_this_layer], device='cuda')
                                
                                if use_residual_mask:
                                    if residual_mask_include_tie:
                                        for ch_idx, _, type_val in selected_for_this_layer:
                                            mask_layer.mask_weights[ch_idx, int(type_val)] = 5.0
                                    else:
                                        for ch_idx, _, type_val in selected_for_this_layer:
                                            mask_layer.mask_weights[ch_idx, int(type_val)] = 5.0
                                else:
                                    indices_0tie = prune_indices[torch.tensor([t == 0 for _, _, t in selected_for_this_layer], device='cuda')]
                                    indices_1tie = prune_indices[torch.tensor([t == 1 for _, _, t in selected_for_this_layer], device='cuda')]
                                    
                                    if len(indices_0tie) > 0:
                                        mask_layer.mask_weights[indices_0tie, 0] = 5.0
                                    if len(indices_1tie) > 0:
                                        mask_layer.mask_weights[indices_1tie, 1] = 5.0
                            
                            # 선택되지 않은 채널들은 bypass로 설정
                            selected_channel_indices = set([ch_idx for ch_idx, _, _ in selected_for_this_layer])
                            keep_indices = torch.tensor([i for i in range(out_channels) if i not in selected_channel_indices], device='cuda')
                            
                            if len(keep_indices) > 0:
                                if use_residual_mask:
                                    bypass_idx = 6 if residual_mask_include_tie else 4
                                    mask_layer.mask_weights[keep_indices, bypass_idx] = 5.0
                                else:
                                    mask_layer.mask_weights[keep_indices, 2] = 5.0
                        
                        num_pruned = len(selected_for_this_layer)
                        print(f"    {layer_name}: Pruned {num_pruned}/{out_channels} channels")
                        
                        new_modules.append(mask_layer)
                        mask_inserted = True
                
                model[seq_idx] = nn.Sequential(*new_modules)
                model.to(device)
            
            # Step 5: One-shot training
            print(f"\n{'='*80}")
            print(f"One-shot training: Training all mask layers together")
            print(f"{'='*80}")
            
            for param in model.parameters():
                param.requires_grad = True
            
            mask_prune_optimizer = create_default_optimizer(model)
            
            acc_before_retrain = eval(model, test_loader, mode=False)
            print(f"\n  Accuracy before retraining: {acc_before_retrain:.4f}")
            
            best_retrain_acc = acc_before_retrain
            best_retrain_model_state = copy.deepcopy(model.state_dict())
            
            train_iter_retrain = iter(load_n(train_loader, args.num_iterations))
            
            for i in tqdm(range(args.num_iterations), desc='Mask channel pruning oneshot retraining'):
                x, y = next(train_iter_retrain)
                x, y = x.to(device), y.to(device)
                
                train_step(model, x, y, loss_fn, mask_prune_optimizer, 0.0, 0.0, args.clip_grad, args,
                        current_iter=i, total_iter=args.num_iterations)
                
                if (i + 1) % args.eval_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        test_acc = eval(model, test_loader, mode=False)
                    model.train()
                    
                    total_channels = 0
                    pruned_channels = 0
                    for name, m in model.named_modules():
                        if isinstance(m, ChannelMaskLayer):
                            mask_selection = m.mask_weights.argmax(-1)
                            pruned = ((mask_selection == 0) | (mask_selection == 1)).sum().item()
                            total_channels += m.num_channels
                            pruned_channels += pruned
                    
                    prune_ratio = (pruned_channels / total_channels * 100) if total_channels > 0 else 0.0
                    
                    print(f"\n  Iter {i+1}: Test Acc={test_acc:.4f}, Pruned={pruned_channels}/{total_channels} ({prune_ratio:.2f}%)")
                    
                    if test_acc > best_retrain_acc:
                        best_retrain_acc = test_acc
                        best_retrain_model_state = copy.deepcopy(model.state_dict())
                        print(f"  *** New best accuracy: {test_acc:.4f} ***")
            
            if best_retrain_model_state is not None:
                model.load_state_dict(best_retrain_model_state)
            
            acc_after_retrain = eval(model, test_loader, mode=False)
            print(f"\n  Retraining finished. Best accuracy: {best_retrain_acc:.4f}")
            print(f"  Accuracy improvement: {acc_after_retrain - acc_before_retrain:+.4f}")
            
            final_acc = eval(model, test_loader, mode=False)
            print(f"\n{'='*80}")
            print(f"Mask Channel Pruning (oneshot mode) completed. Final accuracy: {final_acc:.4f}")
            print(f"  Initial accuracy: {acc_before_retrain:.4f}")
            print(f"  Final accuracy: {final_acc:.4f}")
            print(f"  Improvement: {final_acc - acc_before_retrain:+.4f}")
            print(f"{'='*80}")
            
            # 최종 분석
            print("\n--- Final Analysis after Mask Channel Pruning One-Shot ---")
            
            final_total_channels = 0
            final_pruned_channels = 0
            for name, m in model.named_modules():
                if isinstance(m, ChannelMaskLayer):
                    mask_selection = m.mask_weights.argmax(-1)
                    pruned = ((mask_selection == 0) | (mask_selection == 1)).sum().item()
                    out_dim = m.num_channels
                    final_total_channels += out_dim
                    final_pruned_channels += pruned
                    print(f"  {name}: Pruned {pruned}/{out_dim} channels")
            
            final_prune_ratio = (final_pruned_channels / final_total_channels * 100) if final_total_channels > 0 else 0.0
            print(f"\nFinal results: accuracy = {final_acc:.4f}, pruned_channels = {final_pruned_channels}/{final_total_channels} ({final_prune_ratio:.2f}%)")
            
            analysis_bundle_final = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=True)
            summarize_and_print_analysis(analysis_bundle_final)
            
            if args.pruned_eid:
                remove_residual_mask_hooks(model)
                save_path = f'results_conv/{args.pruned_eid}_mask_channel_prune_oneshot_final.pt'
                torch.save(model, save_path)
                print(f"  ✓ Saved final model: {save_path}")
                
                if pruned_results:
                    summary_data = {
                        'source_eid': args.retrain_eid,
                        'accuracy_before_prune': acc_before,
                        'final_accuracy': final_acc,
                        'improvement': final_acc - acc_before,
                        'pruned_channels': final_pruned_channels,
                        'total_channels': final_total_channels,
                        'prune_ratio': final_prune_ratio,
                        'mask_channel_prune_lambda': args.mask_channel_prune_lambda,
                        'mask_channel_prune_tau': args.mask_channel_prune_tau,
                    }
                    pruned_results.store_final_results(summary_data)
                    pruned_results.save()
                    print(f"  ✓ Saved results to EID: {args.pruned_eid}")
            
            # wandb 로깅
            if WANDB_AVAILABLE and args.pruned_eid is not None:
                stats_final = analysis_bundle_final.get('stats', {})
                total_dead_final = sum(res.get('dead', 0) for key, res in stats_final.items() if key != 'classifier_input')
                total_nodes_final = sum(res.get('total', 0) for key, res in stats_final.items() if key != 'classifier_input')
                overall_dead_ratio_final = (100 * total_dead_final / total_nodes_final) if total_nodes_final > 0 else 0.0
                
                wandb.log({
                    "results/oneshot_best_acc": best_retrain_acc,
                    "results/oneshot_dead_ratio": overall_dead_ratio_final
                })
            
            exit()

        if args.mask_channel_prune:
            if args.implementation != 'im2col':
                print(f"ERROR: --mask-channel-prune requires --implementation im2col")
                exit()
            
            # pruned_results 초기화
            pruned_results = None
            if args.pruned_eid:
                try:
                    pruned_results = ResultsJSON(eid=args.pruned_eid, path='./results_conv/')
                except:
                    pruned_results = None
            
            print("\n--- Mask Channel Pruning Mode: Sequential insertion of MaskLayer after TreeConvLayer and LogicLayer layers ---")
            print("\n--- Model structure before mask channel pruning ---")
            print(model)
            
            # pruning을 진행할 타켓 레이어 설정
            # classifier의 마지막 LogicLayer 제외하고 모든 레이어 타켓
            all_layers = []
            
            
            
            if isinstance(model[0], nn.Sequential):
                for idx, module in enumerate(model[0]):
                    if isinstance(module, TreeConvLayer):
                        all_layers.append(('treeconv', idx, f"features_{idx}", module, 0))  # (type, pos, name, module, sequential_idx)
            
            
            if isinstance(model[1], nn.Sequential):
                for idx, module in enumerate(model[1][:-1]):  # 마지막 레이어 제외
                    if isinstance(module, LogicLayer):
                        all_layers.append(('logic', idx, f"classifier_{idx}", module, 1))  # (type, pos, name, module, sequential_idx)
            
        
            print(f"Found {len(all_layers)} layers to process:")
            for idx, (layer_type, pos, name, module, seq_idx) in enumerate(all_layers):
                if layer_type == 'treeconv':
                    print(f"  {idx}. {name} (features[{pos}], out_dim={module.out_dim})")
                else:
                    print(f"  {idx}. {name} (classifier[{pos}], out_dim={module.out_dim})")
            
            is_iterative = getattr(args, 'iterative', False)  # iterative 모드 (기본 구조 사용)
            is_reverse = getattr(args, 'reverse_iterative', False)  # 역순 처리 모드
            
            # 역순 모드: 레이어 순서를 역순으로 만들기 (처리 순서만 역순, phase 번호는 1부터 시작)
            if is_reverse and is_iterative:
                all_layers = list(reversed(all_layers))
                print(f"\n{'='*80}")
                print(f"REVERSE ITERATIVE MODE: Processing layers in reverse order (last to first)")
                print(f"Phase numbers will still start from 1")
            else:
                print(f"\n{'='*80}")
            
            if is_iterative:
                if is_reverse:
                    print(f"ITERATIVE MODE (REVERSE): Inserting ChannelMaskLayer sequentially from last to first layer")
                else:
                    print(f"ITERATIVE MODE: Inserting ChannelMaskLayer sequentially for all layers")
            else:
                print(f"ONE-SHOT MODE: Inserting all ChannelMaskLayers, then one-shot training")
            print(f"{'='*80}")
            
            num_total_phases = len(all_layers)
            
            # Resume from specific phase if requested
            start_phase_idx = 0
            if getattr(args, 'resume_from_phase', None) is not None:
                resume_phase = args.resume_from_phase
                if args.pruned_eid is None:
                    print(f"ERROR: --resume-from-phase requires --pruned-eid to be specified")
                    exit()
                
                if resume_phase < 1 or resume_phase > num_total_phases:
                    print(f"ERROR: --resume-from-phase must be between 1 and {num_total_phases}, got {resume_phase}")
                    exit()
                
                resume_model_path = f'results_conv/{args.pruned_eid}_mask_prune_phase{resume_phase}.pt'
                if not os.path.exists(resume_model_path):
                    print(f"ERROR: Resume model file not found: {resume_model_path}")
                    exit()
                
                print(f"\n{'='*80}")
                print(f"RESUME MODE: Loading model from phase {resume_phase}")
                print(f"Model path: {resume_model_path}")
                print(f"{'='*80}")
                
                try:
                    loaded_model = torch.load(resume_model_path, map_location=device)
                    if isinstance(loaded_model, dict) and 'model' in loaded_model:
                        model = loaded_model['model']
                    elif isinstance(loaded_model, torch.nn.Module):
                        model = loaded_model
                    else:
                        raise ValueError(f"Unexpected model format in {resume_model_path}")
                    
                    model.to(device)
                    print(f"✓ Successfully loaded model from phase {resume_phase}")
                    
                    # Resume from the next phase (phase indices are 0-indexed)
                    start_phase_idx = resume_phase  # resume_phase는 1-indexed이므로, 다음 phase부터 시작
                    print(f"  Will continue from phase {start_phase_idx + 1}/{num_total_phases}")
                    
                    # Verify that the model has the expected number of layers
                    current_mask_count = sum(1 for m in model.modules() if isinstance(m, (ChannelMaskLayer, ResidualChannelMaskLayer)))
                    expected_mask_count = resume_phase
                    if current_mask_count != expected_mask_count:
                        print(f"  WARNING: Model has {current_mask_count} mask layers, expected {expected_mask_count}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to load resume model: {e}")
                    import traceback
                    traceback.print_exc()
                    exit()
            
            for phase_idx, (layer_type, pos, layer_name, module, seq_idx) in enumerate(all_layers):
                # Skip phases that were already completed (resume mode)
                if phase_idx < start_phase_idx:
                    print(f"\n{'='*80}")
                    print(f"Phase {phase_idx + 1}/{num_total_phases}: Skipping (already completed in previous run)")
                    print(f"{'='*80}")
                    continue
                layer_type_str = "TreeConv" if layer_type == 'treeconv' else "Classifier"
                print(f"\n{'='*80}")
                print(f"Phase {phase_idx + 1}/{num_total_phases}: Inserting ChannelMaskLayer after {layer_name} ({layer_type_str})")
                print(f"{'='*80}")
                
                print(f"\n  [{phase_idx + 1}/{num_total_phases}] Inserting ChannelMaskLayer after {layer_name}")
                
                # 모든 레이어 타입에서 out_dim 사용
                total_channels = module.out_dim
                target_pct = getattr(args, 'prune_pct', None)
                if target_pct is not None:
                    target_prune_channels = int(total_channels * target_pct / 100.0)
                    print(f"  Target prune percentage: {target_pct:.2f}% (applied to all layers, dead channels not considered)")
                    print(f"  Target prune channels: {target_prune_channels}/{total_channels}")
                else:
                    target_prune_channels = 0
                    target_pct = None
                    print(f"  No target prune percentage specified (using random/weight-based option if enabled)")
                
                current_modules = list(model[seq_idx])
                
                # id(module)을 사용해서 정확한 모듈을 찾고 바로 뒤에 mask 삽입
                new_modules = []
                mask_inserted = False
                
                for idx, m in enumerate(current_modules):
                    new_modules.append(m)
                    # 현재 레이어를 찾으면 바로 뒤에 ChannelMaskLayer 삽입
                    if id(m) == id(module) and not mask_inserted:
                        # 현재 레이어 뒤에 이미 ChannelMaskLayer가 있는지 확인
                        has_mask_already = False
                        if idx + 1 < len(current_modules):
                            next_module = current_modules[idx + 1]
                            if isinstance(next_module, nn.Sequential):
                                if len(next_module) > 0 and isinstance(next_module[0], ChannelMaskLayer):
                                    has_mask_already = True
                                    print(f"  ChannelMaskLayer already exists after {layer_name} (wrapped in Sequential), skipping insertion.")
                            elif isinstance(next_module, ChannelMaskLayer):
                                has_mask_already = True
                                print(f"  ChannelMaskLayer already exists after {layer_name}, skipping insertion.")
                        
                        if has_mask_already:
                            continue
                        # 모든 레이어 타입에서 out_dim 사용
                        out_channels = module.out_dim
                        
                        prune_method = getattr(args, 'prune_method', None)
                        # mask_reg 방식은 학습 가능하도록 frozen=False, 나머지는 frozen=True
                        mask_frozen = (prune_method != 'mask_reg')
                        
                        # Residual mask 옵션 확인
                        use_residual_mask = getattr(args, 'residual_mask', False)
                        residual_mask_include_tie = getattr(args, 'residual_mask_include_tie', False)
                        
                        # 이전 LogicLayer 찾기 (residual mask를 위해)
                        # ResidualChannelMaskLayer는 현재 레이어의 LogicLayer를 참조합니다
                        prev_logic_layer = None
                        if use_residual_mask:
                            if layer_type == 'treeconv':
                                # TreeConvLayer인 경우, cascade 내부 마지막 LogicLayer 찾기
                                if hasattr(module, 'cascade'):
                                    for logic_module in reversed(module.cascade):
                                        if isinstance(logic_module, LogicLayer):
                                            if logic_module.out_dim == module.out_dim:
                                                prev_logic_layer = logic_module
                                                break
                            elif layer_type == 'logic':
                                # LogicLayer인 경우, 그 자체를 사용
                                prev_logic_layer = module
                        
                        if use_residual_mask:
                            from birel.conv import ResidualChannelMaskLayer
                            if prev_logic_layer is None:
                                print(f"  ERROR: Could not find prev_logic_layer for ResidualChannelMaskLayer at {layer_name}. Residual mask requires a previous LogicLayer.")
                                exit()
                            mask_layer = ResidualChannelMaskLayer(
                                num_channels=out_channels, 
                                prev_logic_layer=prev_logic_layer,
                                device='cuda', 
                                frozen=mask_frozen,
                                include_tie=residual_mask_include_tie
                            )
                        else:
                            mask_layer = ChannelMaskLayer(num_channels=out_channels, device='cuda', frozen=mask_frozen)
                        all_indices = torch.arange(out_channels, device='cuda')
                        
                        # score와 tie_type_mask 초기화
                        score = None
                        tie_type_mask = None
                        residual_type_mask = None  # ResidualChannelMaskLayer용
                        
                        if prune_method == 'weight':
                            target_logic_layer = None
                            if layer_type == 'treeconv':
                                if hasattr(module, 'cascade'):
                                    # cascade의 마지막 LogicLayer 찾기 (출력 레이어)
                                    for logic_module in reversed(module.cascade):
                                        if isinstance(logic_module, LogicLayer):
                                            if logic_module.out_dim == module.out_dim:
                                                target_logic_layer = logic_module
                                                break
                            else:  # logic
                                target_logic_layer = module  # module 자체가 LogicLayer
                            
                            if target_logic_layer is not None:
                                logic_weights = target_logic_layer.weights.data  # [out_dim, 16]
                                logic_probs = torch.softmax(logic_weights, dim=-1)  # [out_dim, 16]
                                
                                if use_residual_mask:
                                    if residual_mask_include_tie:
                                        # ResidualChannelMaskLayer with tie: 7가지 옵션 (0-tie, 1-tie, bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass)
                                        # Operation 0: 0-tie
                                        # Operation 15: 1-tie
                                        # Operation 3 (a): bypass-a
                                        # Operation 5 (b): bypass-b
                                        # Operation 10 (NOT b): neg bypass-b
                                        # Operation 12 (NOT a): neg bypass-a
                                        
                                        prob_0tie = logic_probs[:, 0]  # [out_dim] - Operation 0
                                        prob_1tie = logic_probs[:, 15]  # [out_dim] - Operation 15
                                        prob_bypass_a = logic_probs[:, 3]  # [out_dim] - Operation 3 (a)
                                        prob_bypass_b = logic_probs[:, 5]  # [out_dim] - Operation 5 (b)
                                        prob_neg_bypass_b = logic_probs[:, 10]  # [out_dim] - Operation 10 (NOT b)
                                        prob_neg_bypass_a = logic_probs[:, 12]  # [out_dim] - Operation 12 (NOT a)
                                        
                                        # 6개 operation의 probability를 stack (0-tie, 1-tie, bypass-a, bypass-b, neg bypass-a, neg bypass-b)
                                        residual_probs = torch.stack([
                                            prob_0tie,          # 0: 0-tie
                                            prob_1tie,          # 1: 1-tie
                                            prob_bypass_a,      # 2: bypass-a
                                            prob_bypass_b,      # 3: bypass-b
                                            prob_neg_bypass_a,  # 4: neg bypass-a
                                            prob_neg_bypass_b   # 5: neg bypass-b
                                        ], dim=-1)  # [out_dim, 6]
                                        
                                        # 각 channel에 대해 가장 높은 probability를 가진 operation 선택
                                        max_residual_score, max_residual_op = torch.max(residual_probs, dim=-1)  # [out_dim], [out_dim]
                                        
                                        # Score는 max_residual_score를 사용 (높을수록 prune할 가치가 높음)
                                        score = max_residual_score
                                        # Residual connection 타입 (0: 0-tie, 1: 1-tie, 2: bypass-a, 3: bypass-b, 4: neg bypass-a, 5: neg bypass-b)
                                        residual_type_mask = max_residual_op
                                    else:
                                        # ResidualChannelMaskLayer without tie: 5가지 옵션 (bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass)
                                        # Operation 3 (a): bypass-a
                                        # Operation 5 (b): bypass-b
                                        # Operation 10 (NOT b): neg bypass-b
                                        # Operation 12 (NOT a): neg bypass-a
                                        
                                        prob_bypass_a = logic_probs[:, 3]  # [out_dim] - Operation 3 (a)
                                        prob_bypass_b = logic_probs[:, 5]  # [out_dim] - Operation 5 (b)
                                        prob_neg_bypass_b = logic_probs[:, 10]  # [out_dim] - Operation 10 (NOT b)
                                        prob_neg_bypass_a = logic_probs[:, 12]  # [out_dim] - Operation 12 (NOT a)
                                        
                                        # 4개 operation의 probability를 stack (bypass-a, bypass-b, neg bypass-a, neg bypass-b)
                                        residual_probs = torch.stack([
                                            prob_bypass_a,      # 0: bypass-a
                                            prob_bypass_b,      # 1: bypass-b
                                            prob_neg_bypass_a,  # 2: neg bypass-a
                                            prob_neg_bypass_b   # 3: neg bypass-b
                                        ], dim=-1)  # [out_dim, 4]
                                        
                                        # 각 channel에 대해 가장 높은 probability를 가진 residual connection 선택
                                        max_residual_score, max_residual_op = torch.max(residual_probs, dim=-1)  # [out_dim], [out_dim]
                                        
                                        # Score는 max_residual_score를 사용 (높을수록 prune할 가치가 높음)
                                        score = max_residual_score
                                        # Residual connection 타입 (0: bypass-a, 1: bypass-b, 2: neg bypass-a, 3: neg bypass-b)
                                        residual_type_mask = max_residual_op
                                else:
                                    # ChannelMaskLayer: 3가지 옵션 (0-tie, 1-tie, bypass)
                                    prob_0tie = logic_probs[:, 0]  # [out_dim]
                                    prob_1tie = logic_probs[:, 15]  # [out_dim]
                                    
                                    score = torch.maximum(prob_0tie, prob_1tie)  # [out_dim] - 높을수록 prune할 가치가 높음
                                    tie_type_mask = (prob_1tie > prob_0tie)  # True면 1-tie, False면 0-tie
                            else:
                                print(f"  Warning: Could not find LogicLayer for weight-based pruning. Falling back to default initialization.")
                                prune_method = None  # Fallback to default
                        
                        elif prune_method == 'loss':
                            print(f"  Computing loss changes for {out_channels} channels...")
                            
                            # Loss 타입 선택 (L1 또는 MSE)
                            loss_type = getattr(args, 'loss_prune_type', 'mse')  # 'l1' or 'mse'
                            
                            def prune_loss_fn(output, target):
                                """ Logits의 변화(Distortion)를 측정하는 L1 또는 L2 Loss """
                                if loss_type == 'l1':
                                    return torch.nn.functional.l1_loss(output, target, reduction='mean')
                                else:  # 'mse' or default
                                    return torch.nn.functional.mse_loss(output, target, reduction='mean')
                            
                            # 먼저 평가에 사용할 데이터를 미리 수집하여 일관성 보장
                            model.train()
                            num_samples = getattr(args, 'mask_channel_prune_loss_samples', 100)
                            eval_data_list = []
                            collected_samples = 0
                            
                            with torch.no_grad():
                                for batch_idx, (x, y) in enumerate(train_loader):
                                    if collected_samples >= num_samples:
                                        break
                                    x, y = x.to(device), y.to(device)
                                    eval_data_list.append((x.detach().clone(), y.detach().clone()))
                                    collected_samples += x.size(0)
                            
                            if len(eval_data_list) == 0:
                                print(f"    Warning: No evaluation data collected. Skipping loss-based pruning.")
                                continue
                            
                            # Baseline logits 계산 (mask 삽입 전 원본 모델에서 측정)
                            print(f"    Computing baseline logits from original model (before mask insertion)...")
                            with torch.no_grad():
                                baseline_logits_list = []
                                for x, y in eval_data_list:
                                    output = model(x)
                                    baseline_logits_list.append(output.detach().clone())
                                baseline_logits = torch.cat(baseline_logits_list, dim=0)
                            
                            print(f"    Baseline logits collected: {baseline_logits.shape} (from {collected_samples} samples)")
                            
                            # 이제 mask를 삽입
                            with torch.no_grad():
                                mask_layer.mask_weights.fill_(0.0)
                                mask_layer.mask_weights[:, 2] = 5.0  # 모든 채널을 bypass로 설정 (초기값)
                            
                            # id(module)을 사용해서 정확한 모듈의 위치를 찾기
                            temp_target_idx = None
                            for temp_idx, temp_m in enumerate(current_modules):
                                if id(temp_m) == id(module):
                                    temp_target_idx = temp_idx
                                    break
                            
                            if temp_target_idx is None:
                                print(f"  Warning: Could not find module for loss-based pruning. Skipping...")
                                continue
                            
                            temp_modules = list(current_modules)
                            temp_modules.insert(temp_target_idx + 1, mask_layer)
                            model[seq_idx] = nn.Sequential(*temp_modules)
                            model.to(device)
                            
                            # 각 채널에 대해 loss 변화 계산
                            loss_changes_0tie = torch.zeros(out_channels, device='cuda')
                            loss_changes_1tie = torch.zeros(out_channels, device='cuda')
                            
                            print(f"    Evaluating loss changes for each channel...")
                            with torch.no_grad():
                                for channel_idx in tqdm(range(out_channels), desc=f"      Channel", leave=False):
                                    # 0-tie로 교체
                                    mask_layer.mask_weights.fill_(0.0)
                                    mask_layer.mask_weights[:, 2] = 5.0  # 나머지는 bypass
                                    mask_layer.mask_weights[channel_idx, 2] = 0.0  # 해당 채널은 bypass 아님
                                    mask_layer.mask_weights[channel_idx, 0] = 5.0  # 해당 채널만 0-tie
                                    pruned_logits_list = []
                                    for x, y in eval_data_list:
                                        output = model(x)
                                        pruned_logits_list.append(output.detach().clone())
                                    
                                    pruned_logits_0tie = torch.cat(pruned_logits_list, dim=0)
                                    loss_0tie = prune_loss_fn(pruned_logits_0tie, baseline_logits)
                                    loss_changes_0tie[channel_idx] = loss_0tie.item()
                                    
                                    # 1-tie로 교체
                                    mask_layer.mask_weights.fill_(0.0)
                                    mask_layer.mask_weights[:, 2] = 5.0  # 나머지는 bypass
                                    mask_layer.mask_weights[channel_idx, 2] = 0.0  # 해당 채널은 bypass 아님
                                    mask_layer.mask_weights[channel_idx, 1] = 5.0  # 해당 채널만 1-tie
                                    
                                    pruned_logits_list = []
                                    for x, y in eval_data_list:
                                        output = model(x)
                                        pruned_logits_list.append(output.detach().clone())
                                    
                                    pruned_logits_1tie = torch.cat(pruned_logits_list, dim=0)
                                    loss_1tie = prune_loss_fn(pruned_logits_1tie, baseline_logits)
                                    loss_changes_1tie[channel_idx] = loss_1tie.item()
                            
                            min_loss_changes = torch.minimum(loss_changes_0tie, loss_changes_1tie)  # [out_dim]
                            score = -min_loss_changes  # [out_dim] - 낮은 loss 변화일수록 높은 score (음수이므로)
                            tie_type_mask = (loss_changes_1tie < loss_changes_0tie)  # True면 1-tie가 더 낮음, False면 0-tie가 더 낮음
                            
                            temp_modules = list(model[seq_idx])
                            # id(module)을 사용해서 정확한 모듈의 위치를 찾기
                            temp_target_idx = None
                            for temp_idx, temp_m in enumerate(temp_modules):
                                if id(temp_m) == id(module):
                                    temp_target_idx = temp_idx
                                    break
                            
                            if temp_target_idx is not None and temp_target_idx + 1 < len(temp_modules):
                                temp_modules.pop(temp_target_idx + 1)
                            model[seq_idx] = nn.Sequential(*temp_modules)
                            model.to(device)
                        
                        elif prune_method == 'loss_approx':
                            print(f"  Using Hessian-based approx loss pruning for {layer_name}")

                            # 1) eval loader 준비 (이미 위에서 eval_data_list 뽑던 로직 대신)
                            num_samples = getattr(args, 'mask_channel_prune_loss_samples', 100)
                            tmp_list = []
                            collected = 0
                            model.eval()

                            with torch.no_grad():
                                for x, y in train_loader:
                                    x, y = x.to(device), y.to(device)
                                    tmp_list.append((x, y))
                                    collected += x.size(0)
                                    if collected >= num_samples:
                                        break

                            # 작은 DataLoader로 다시 감싸기
                            small_dataset = [(x.cpu(), y.cpu()) for x, y in tmp_list]
                            small_loader = torch.utils.data.DataLoader(
                                small_dataset, batch_size=args.batch_size, shuffle=False
                            )

                            # 2) 근사 Δloss 계산
                            approx_loss0, approx_loss1 = approximate_channel_loss_scores(
                                model,
                                target_module=module,      # 지금 phase에서 다루는 TreeConvLayer or LogicLayer
                                eval_batches=small_dataset,
                                device=device,
                                num_batches=min(5, len(small_loader)),
                                num_probes=1,
                            )

                            # 3) 채널별 score / tie type 결정
                            # ΔL가 작을수록 prune하기 좋으니 score = -ΔL
                            if approx_loss0 is None or approx_loss1 is None:
                                raise RuntimeError("approx_loss0/1 is None, check tie options")

                            # 두 tie 중 더 작은 ΔL 선택
                            all_losses = torch.stack([approx_loss0, approx_loss1], dim=0)  # [2, C]
                            min_loss, best_tie = torch.min(all_losses, dim=0)              # [C]

                            score = -min_loss               # 큰 score일수록 prune할 가치 ↑
                            tie_type_mask = (best_tie == 1) # True면 1-tie, False면 0-tie

                        
                        elif prune_method == 'random':
                            # 랜덤 score 생성
                            score = torch.rand(out_channels, device='cuda')  # [out_dim] - 랜덤하게 선택
                            tie_type_mask = torch.rand(out_channels, device='cuda') > 0.5  # True면 1-tie, False면 0-tie
                        
                        elif prune_method == 'prob':
                            print(f"  Computing probability-based scores for {out_channels} channels...")
                            
                            # 모델을 training mode로 설정
                            model.train()
                            
                            # target_logic_layer 찾기
                            target_logic_layer = None
                            if layer_type == 'treeconv':
                                if hasattr(module, 'cascade'):
                                    # cascade의 마지막 LogicLayer 찾기 (출력 레이어)
                                    for logic_module in reversed(module.cascade):
                                        if isinstance(logic_module, LogicLayer):
                                            if logic_module.out_dim == module.out_dim:
                                                target_logic_layer = logic_module
                                                break
                            else:  # logic
                                target_logic_layer = module  # module 자체가 LogicLayer
                            
                            if 'mnist' in args.dataset:
                                dummy_input_shape = (1, 1, 28, 28)
                            else:  # cifar-10
                                dummy_input_shape = (1, 9, 32, 32)
                            
                            prob_input = torch.full(dummy_input_shape, 0.5, device=device)
                            
                            # Forward pass를 통해 target_logic_layer의 output 계산
                            # target_logic_layer의 출력을 얻기 위해 hook 사용
                            target_output = [None]  # list로 감싸서 nonlocal 문제 해결
                            
                            def get_output_hook(module_hook, input, output):
                                # LogicLayer의 output은 2D tensor (B, C)이므로 채널별로 평균
                                if output.dim() == 2:
                                    target_output[0] = output.mean(dim=0)  # [C] - 채널별 평균
                                elif output.dim() == 4:
                                    target_output[0] = output.mean(dim=(0, 2, 3))  # [C] - 채널별 평균
                                else:
                                    target_output[0] = output.flatten()  # 1D로 변환
                            
                            hook = target_logic_layer.register_forward_hook(get_output_hook)
                            
                            with torch.no_grad():
                                _ = model(prob_input)
                            
                            hook.remove()
                            
                            out = target_output[0]  # list에서 값 추출
                                        
                            
                            if phase_idx != 0:
                                score = torch.maximum(1 - out, out)  # [out_channels] - 높을수록 prune할 가치가 높음                            
                                tie_type_mask = (1 - out) < out  # True면 1-tie, False면 0-tie
                            else:
                                logic_weights = target_logic_layer.weights.data  # [out_dim, 16]
                                logic_probs = torch.softmax(logic_weights, dim=-1)  # [out_dim, 16]
                                
                                prob_0tie = logic_probs[:, 0]  # [out_dim]
                                prob_1tie = logic_probs[:, 15]  # [out_dim]
                                
                                score = torch.maximum(prob_0tie, prob_1tie)  # [out_dim] - 높을수록 prune할 가치가 높음
                                tie_type_mask = (prob_1tie > prob_0tie)  # True면 1-tie, False면 0-tie

                            print(f"    Computed prob-based scores: min={score.min().item():.4f}, max={score.max().item():.4f}, mean={score.mean().item():.4f}")
                        
                        elif prune_method == 'entropy_stuck':
                            print(f"  Computing entropy-based scores for {out_channels} channels...")
                            
                            # TreeConvLayer의 경우 직접 모듈 사용, LogicLayer의 경우 이름 사용
                            target_layer_name = None
                            target_module_for_hook = None
                            
                            if layer_type == 'treeconv':
                                # TreeConvLayer의 경우 직접 모듈에 hook 등록
                                target_module_for_hook = module
                                # 디버깅을 위한 이름 찾기
                                for name, m in model.named_modules():
                                    if id(m) == id(module):
                                        target_layer_name = name
                                        break
                            else:  # logic layer
                                # LogicLayer의 경우 이름으로 찾기
                                for name, m in model.named_modules():
                                    if id(m) == id(module):
                                        target_layer_name = name
                                        break
                            
                            if target_module_for_hook is None and target_layer_name is None:
                                print(f"  Warning: Could not find layer for entropy-based pruning. Falling back to default initialization.")
                                prune_method = None  # Fallback to default
                            else:
                                # entropy stuck 방식으로 채널 선택
                                stuck_max_batches = getattr(args, 'stuck_max_batches', 10)
                                stuck_list = detect_stuck_by_entropy_single_layer(
                                    model,
                                    train_loader,
                                    target_layer_name=target_layer_name,
                                    target_module=target_module_for_hook,
                                    prune_pct=target_pct / 100.0 if target_pct is not None else 0.1,
                                    max_batches=stuck_max_batches,
                                    device=device
                                )
                                
                                if stuck_list:
                                    # stuck_list에서 channel_idx와 stuck_type 추출
                                    score = torch.zeros(out_channels, device='cuda')
                                    tie_type_mask = torch.zeros(out_channels, dtype=torch.bool, device='cuda')
                                    
                                    for layer_name, channel_idx, stuck_type, avg_prob, entropy in stuck_list:
                                        if channel_idx < out_channels:
                                            # entropy가 낮을수록 높은 score (prune할 가치가 높음)
                                            # entropy를 역수로 변환하여 score 계산
                                            score[channel_idx] = 1.0 / (1.0 + entropy)
                                            tie_type_mask[channel_idx] = (stuck_type == 'sa1')  # True면 1-tie, False면 0-tie
                                    
                                    print(f"    Computed entropy-based scores: {len(stuck_list)} channels selected")
                                else:
                                    print(f"  Warning: No stuck neurons found. Falling back to default initialization.")
                                    prune_method = None  # Fallback to default

                        # score 기반으로 정렬해서 top prune pct만큼을 잘라내도록 마스크 초기화                                
                        if prune_method is not None and score is not None:
                            if target_prune_channels > 0 and len(all_indices) > 0:
                                num_to_prune = min(target_prune_channels, len(all_indices))
                                
                                _, sorted_indices = torch.sort(score, descending=True)
                                selected_indices = sorted_indices[:num_to_prune]
                                prune_indices = all_indices[selected_indices]
                            else:
                                prune_indices = torch.tensor([], dtype=torch.long, device='cuda')
                            
                            num_channels_to_prune = len(prune_indices)
                            
                            # residual_type_mask 확인
                            if use_residual_mask:
                                if residual_type_mask is None:
                                    print(f"  ERROR: residual_type_mask is None for ResidualChannelMaskLayer. Cannot initialize mask.")
                                    exit()
                            
                            with torch.no_grad():
                                mask_layer.mask_weights.fill_(0.0)
                                
                                if len(prune_indices) > 0:
                                    if use_residual_mask:
                                        if residual_mask_include_tie:
                                            # ResidualChannelMaskLayer with tie: 7가지 옵션 (0-tie, 1-tie, bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass)
                                            # residual_type_mask: 0=0-tie, 1=1-tie, 2=bypass-a, 3=bypass-b, 4=neg bypass-a, 5=neg bypass-b
                                            # residual_type_mask는 0-5까지만 있음 (bypass는 6번 인덱스로 별도 처리)
                                            for res_type in range(6):
                                                mask_type_indices = prune_indices[residual_type_mask[prune_indices] == res_type]
                                                if len(mask_type_indices) > 0:
                                                    mask_layer.mask_weights[mask_type_indices, res_type] = 5.0
                                        else:
                                            # ResidualChannelMaskLayer without tie: 5가지 옵션 (bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass)
                                            # residual_type_mask: 0=bypass-a, 1=bypass-b, 2=neg bypass-a, 3=neg bypass-b
                                            # residual_type_mask는 0-3까지만 있음 (bypass는 4번 인덱스로 별도 처리)
                                            for res_type in range(4):
                                                mask_type_indices = prune_indices[residual_type_mask[prune_indices] == res_type]
                                                if len(mask_type_indices) > 0:
                                                    mask_layer.mask_weights[mask_type_indices, res_type] = 5.0
                                    else:
                                        # ChannelMaskLayer: 3가지 옵션 (0-tie, 1-tie, bypass)
                                        indices_0tie = prune_indices[~tie_type_mask[prune_indices]]
                                        indices_1tie = prune_indices[tie_type_mask[prune_indices]]
                                        
                                        if len(indices_0tie) > 0:
                                            mask_layer.mask_weights[indices_0tie, 0] = 5.0  # 0-tie
                                        if len(indices_1tie) > 0:
                                            mask_layer.mask_weights[indices_1tie, 1] = 5.0  # 1-tie
                                
                                # 선택되지 않은 channel들은 bypass로 설정
                                keep_indices = torch.ones(out_channels, dtype=torch.bool, device='cuda')
                                keep_indices[prune_indices] = False
                                if keep_indices.any():
                                    if use_residual_mask:
                                        bypass_idx = 6 if residual_mask_include_tie else 4
                                        mask_layer.mask_weights[keep_indices, bypass_idx] = 5.0  # bypass (ResidualChannelMaskLayer의 마지막 옵션)
                                    else:
                                        mask_layer.mask_weights[keep_indices, 2] = 5.0  # bypass (ChannelMaskLayer의 마지막 옵션)
                            
                            if use_residual_mask:
                                if residual_mask_include_tie:
                                    num_0tie = (mask_layer.mask_weights[:, 0] > 0).sum().item()
                                    num_1tie = (mask_layer.mask_weights[:, 1] > 0).sum().item()
                                    num_bypass_a = (mask_layer.mask_weights[:, 2] > 0).sum().item()
                                    num_bypass_b = (mask_layer.mask_weights[:, 3] > 0).sum().item()
                                    num_neg_bypass_a = (mask_layer.mask_weights[:, 4] > 0).sum().item()
                                    num_neg_bypass_b = (mask_layer.mask_weights[:, 5] > 0).sum().item()
                                    num_bypass = (mask_layer.mask_weights[:, 6] > 0).sum().item()
                                    print(f"  Initialized mask: 0-tie={num_0tie}, 1-tie={num_1tie}, bypass-a={num_bypass_a}, bypass-b={num_bypass_b}, neg bypass-a={num_neg_bypass_a}, neg bypass-b={num_neg_bypass_b}, bypass={num_bypass}")
                                else:
                                    num_bypass_a = (mask_layer.mask_weights[:, 0] > 0).sum().item()
                                    num_bypass_b = (mask_layer.mask_weights[:, 1] > 0).sum().item()
                                    num_neg_bypass_a = (mask_layer.mask_weights[:, 2] > 0).sum().item()
                                    num_neg_bypass_b = (mask_layer.mask_weights[:, 3] > 0).sum().item()
                                    num_bypass = (mask_layer.mask_weights[:, 4] > 0).sum().item()
                                    print(f"  Initialized mask: bypass-a={num_bypass_a}, bypass-b={num_bypass_b}, neg bypass-a={num_neg_bypass_a}, neg bypass-b={num_neg_bypass_b}, bypass={num_bypass}")
                            else:
                                num_0tie_total = (mask_layer.mask_weights[:, 0] > 0).sum().item()
                                num_1tie_total = (mask_layer.mask_weights[:, 1] > 0).sum().item()
                                num_bypass = (mask_layer.mask_weights[:, 2] > 0).sum().item()
                            
                        elif prune_method == 'mask_reg':
                            print(f"  Mask initialized for mask_reg (will be trained with KL divergence regularization)")
                        else:
                            print(f"  Mask initialized (will be trained to achieve target prune percentage)")
                        
                        #mask_wrapper = nn.Sequential(mask_layer)
                        new_modules.append(mask_layer)
                        mask_inserted = True
                
                model[seq_idx] = nn.Sequential(*new_modules)
                model.to(device)
                
                for param in model.parameters():
                    param.requires_grad = True
                
                if is_iterative:
                    print(f"\n  === Phase {phase_idx + 1}/{num_total_phases} Retraining ({layer_type_str}) ===")
                    
                    retrain_loader = train_loader                        
                    mask_prune_optimizer = create_default_optimizer(model)
                    acc_before_phase = eval(model, test_loader, mode=False)
                    print(f"  Accuracy before phase {phase_idx + 1}: {acc_before_phase:.4f}")
                    
                    best_phase_acc = acc_before_phase
                    best_phase_model_state = copy.deepcopy(model.state_dict())
                    
                    train_iter_phase = iter(load_n(retrain_loader, args.num_iterations))
                    
                    model.train()
                    for i in tqdm(range(args.num_iterations), desc=f'Phase {phase_idx+1}/{num_total_phases} training mask'):
                        x, y = next(train_iter_phase)
                        x, y = x.to(device), y.to(device)
                        
                        train_step(model, x, y, loss_fn, mask_prune_optimizer, 0.0, 0.0, args.clip_grad, args, 
                                   current_iter=i, total_iter=args.num_iterations)

                        if (i + 1) % args.eval_freq == 0:
                            model.eval()
                            with torch.no_grad():
                                test_acc = eval(model, test_loader, mode=False)
                                '''
                                if WANDB_AVAILABLE and args.pruned_eid is not None:
                                    global_step = phase_idx * args.num_iterations + (i + 1)
                                    wandb.log({
                                        f'training_curve/test_acc_phase{phase_idx+1}': test_acc,
                                    }, step=global_step)
                                '''
                            model.train()
                            
                            if test_acc > best_phase_acc:
                                best_phase_acc = test_acc
                                best_phase_model_state = copy.deepcopy(model.state_dict())
                                print(f"\n  Iter {i+1}: New best accuracy in phase {phase_idx+1}: {test_acc:.4f}")
                    
                    model.load_state_dict(best_phase_model_state)
                    acc_after_phase = eval(model, test_loader, mode=False)
                    print(f"\n  Phase {phase_idx + 1} retraining finished. Best accuracy: {best_phase_acc:.4f}")
                    print(f"  Accuracy improvement: {acc_after_phase - acc_before_phase:+.4f}")
                    
                    if args.pruned_eid:
                        save_path = f'results_conv/{args.pruned_eid}_mask_prune_phase{phase_idx+1}.pt'
                        remove_residual_mask_hooks(model)
                        torch.save(model, save_path)
                        print(f"  ✓ Saved best model: {save_path}")
                    
                    analysis_bundle_phase = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
                    summarize_and_print_analysis(analysis_bundle_phase)
                    
                    if WANDB_AVAILABLE and args.pruned_eid is not None:
                        stats_phase = analysis_bundle_phase.get('stats', {})
                        total_dead_phase = sum(res.get('dead', 0) for key, res in stats_phase.items() if key != 'classifier_input')
                        total_nodes_phase = sum(res.get('total', 0) for key, res in stats_phase.items() if key != 'classifier_input')
                        overall_dead_ratio_phase = (100 * total_dead_phase / total_nodes_phase) if total_nodes_phase > 0 else 0.0
                        wandb.log({
                            f"results/phase_{phase_idx+1}_best_acc": best_phase_acc,
                            f"results/phase_{phase_idx+1}_dead_ratio": overall_dead_ratio_phase
                        })
                else:
                    print(f"\n  === Phase {phase_idx + 1}/{num_total_phases} (One-shot mode: skipping training) ===")
                
            # iterative 모드일 때 최종 결과 출력
            if is_iterative:
                final_acc = eval(model, test_loader, mode=False)
                print(f"\n{'='*80}")
                print(f"Mask Channel Pruning (iterative mode) completed. Final accuracy: {final_acc:.4f}")
                print(f"{'='*80}")
            
            # one-shot 모드일 때 모든 mask 삽입 후 한 번에 학습
            if not is_iterative:
                # ========== ONE-SHOT MODE: 모든 mask layer 삽입 완료 후 한 번에 retrain ==========
                print(f"\n{'='*80}")
                print(f"One-shot training: Training all mask layers together")
                print(f"{'='*80}")
                
                # 모든 파라미터 활성화
                for param in model.parameters(): 
                    param.requires_grad = True
                
                # Optimizer 설정
                mask_prune_optimizer = create_default_optimizer(model)
                  
                # 초기 accuracy 측정
                acc_before_retrain = eval(model, test_loader, mode=False)
                print(f"\n  Accuracy before retraining: {acc_before_retrain:.4f}")
                
                best_retrain_acc = acc_before_retrain
                best_retrain_model_state = copy.deepcopy(model.state_dict())
                
                # Retraining loop
                train_iter_retrain = iter(load_n(train_loader, args.num_iterations))
                
                for i in tqdm(range(args.num_iterations), desc='Mask channel pruning retraining'):
                    x, y = next(train_iter_retrain)
                    x, y = x.to(device), y.to(device)
                    
                    train_step(model, x, y, loss_fn, mask_prune_optimizer, 0.0, 0.0, args.clip_grad, args,
                            current_iter=i, total_iter=args.num_iterations)

                    if (i + 1) % args.eval_freq == 0:
                        model.eval()
                        with torch.no_grad():
                            test_acc = eval(model, test_loader, mode=False)
                        model.train()
                        
                        # Prune ratio 계산 (루프 밖에서 계산)
                        total_channels = 0
                        pruned_channels = 0
                        for name, m in model.named_modules():
                            if isinstance(m, ChannelMaskLayer):
                                mask_selection = m.mask_weights.argmax(-1)
                                pruned = ((mask_selection == 0) | (mask_selection == 1)).sum().item()
                                total_channels += m.num_channels
                                pruned_channels += pruned
                        
                        prune_ratio = (pruned_channels / total_channels * 100) if total_channels > 0 else 0.0
                        
                        print(f"\n  Iter {i+1}: Test Acc={test_acc:.4f}, Pruned={pruned_channels}/{total_channels} ({prune_ratio:.2f}%)")
                        
                        if test_acc > best_retrain_acc:
                            best_retrain_acc = test_acc
                            best_retrain_model_state = copy.deepcopy(model.state_dict())
                            print(f"  *** New best accuracy: {test_acc:.4f} ***")
                
                if best_retrain_model_state is not None:
                    model.load_state_dict(best_retrain_model_state)
                
                acc_after_retrain = eval(model, test_loader, mode=False)
                print(f"\n  Retraining finished. Best accuracy: {best_retrain_acc:.4f}")
                print(f"  Accuracy improvement: {acc_after_retrain - acc_before_retrain:+.4f}")
                
                final_acc = eval(model, test_loader, mode=False)
                print(f"\n{'='*80}")
                print(f"Mask Channel Pruning (one-shot mode) completed. Final accuracy: {final_acc:.4f}")   
                print(f"  Initial accuracy: {acc_before_retrain:.4f}")
                print(f"  Final accuracy: {final_acc:.4f}")
                print(f"  Improvement: {final_acc - acc_before_retrain:+.4f}")
                print(f"{'='*80}")
                
                # wandb 로깅
                if WANDB_AVAILABLE and args.pruned_eid is not None:
                    final_analysis = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
                    stats_final = final_analysis.get('stats', {})
                    total_dead_final = sum(res.get('dead', 0) for key, res in stats_final.items() if key != 'classifier_input')
                    total_nodes_final = sum(res.get('total', 0) for key, res in stats_final.items() if key != 'classifier_input')
                    overall_dead_ratio_final = (100 * total_dead_final / total_nodes_final) if total_nodes_final > 0 else 0.0
                    
                    wandb.log({
                        "results/oneshot_best_acc": best_retrain_acc,
                        "results/oneshot_dead_ratio": overall_dead_ratio_final
                    })
                
            # 최종 분석
            print("\n--- Final Analysis after Mask Channel Pruning ---")
            
            # 최종 accuracy 재측정
            final_acc = eval(model, test_loader, mode=False)
            
            final_total_channels = 0
            final_pruned_channels = 0
            for name, m in model.named_modules():
                if isinstance(m, ChannelMaskLayer):
                    mask_selection = m.mask_weights.argmax(-1)
                    pruned = ((mask_selection == 0) | (mask_selection == 1)).sum().item()
                    out_dim = m.num_channels
                    final_total_channels += out_dim
                    final_pruned_channels += pruned
                    print(f"  {name}: Pruned {pruned}/{out_dim} channels")
            
            final_prune_ratio = (final_pruned_channels / final_total_channels * 100) if final_total_channels > 0 else 0.0
            print(f"\nFinal results: accuracy = {final_acc:.4f}, pruned_channels = {final_pruned_channels}/{final_total_channels} ({final_prune_ratio:.2f}%)")
        
            # 최종 분석 (finding_live_nodes_by_channel이 이미 ChannelMaskLayer를 처리함)
            analysis_bundle_final = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=True)
            summarize_and_print_analysis(analysis_bundle_final)
            

            if args.pruned_eid:
                # Hook 제거 후 저장
                remove_residual_mask_hooks(model)
                save_path = f'results_conv/{args.pruned_eid}_mask_channel_prune_final.pt'
                torch.save(model, save_path)
                print(f"  ✓ Saved final model: {save_path}")
                
                if pruned_results:
                    summary_data = {
                        'source_eid': args.retrain_eid,
                        'accuracy_before_prune': acc_before,
                        'final_accuracy': final_acc,
                        'improvement': final_acc - acc_before,
                        'pruned_channels': final_pruned_channels,
                        'total_channels': final_total_channels,
                        'prune_ratio': final_prune_ratio,
                        'mask_channel_prune_lambda': args.mask_channel_prune_lambda,
                        'mask_channel_prune_tau': args.mask_channel_prune_tau,
                    }
                    pruned_results.store_final_results(summary_data)
                    pruned_results.save()
                    print(f"  ✓ Saved results to EID: {args.pruned_eid}")
                
            exit()

        

        # ================== PHASE 1: WGS Pruning ==================
        # pruned_results 초기화 (WGS pruning의 경우)
        pruned_results = None
        if args.pruned_eid:
            try:
                from results_json import ResultsJSON
                pruned_results = ResultsJSON(eid=args.pruned_eid, path='./results_conv/')
            except Exception as e:
                print(f"Warning: Could not initialize ResultsJSON: {e}")
                pruned_results = None
        
        print(f"\n--- PHASE 1: Pruning Classifier ({args.num_iterations} iterations) ---")
        print("\n--- Model structure before WGS pruning ---")
        print(model)
        
        # 마지막 LogicLayer 찾기 (ChannelMaskLayer를 건너뛰기 위해 역순으로 검색)
        last_logic_layer = None
        for idx in range(len(model[1]) - 2, -1, -1):  # -2부터 역순으로 (마지막은 WeightedGroupSum/GroupSum)
            if isinstance(model[1][idx], LogicLayer):
                last_logic_layer = model[1][idx]
                break
        
        if last_logic_layer is None:
            raise ValueError("Could not find LogicLayer in classifier for WGS pruning")
        
        wgs_in_dim = last_logic_layer.out_dim
        wgs_layer_ref = WeightedGroupSum(k=model[1][-1].k, in_dim=wgs_in_dim, tau=model[1][-1].tau, init="ones").to(device)
        model[1][-1] = wgs_layer_ref
        model.to(device)
        

        for param in model.parameters():
            param.requires_grad = True
            
        for module in model.modules():
            if isinstance(module, (BlockEfficientCrossbarLayer, ChannelMaskLayer)):
                for param in module.parameters():
                    param.requires_grad = False

        prune_optimizer_p1 = create_default_optimizer(model)
        #prune_scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(prune_optimizer_p1, T_max=args.num_iterations)
        
        best_acc_p1 = 0
        best_model_p1_state = copy.deepcopy(model.state_dict())

        for i in tqdm(range(args.num_iterations), desc='Phase 1: Pruning Classifier'):
            x, y = next(train_iter); x, y = x.to(device), y.to(device)
            
            train_step(model, x, y, loss_fn, prune_optimizer_p1, args.wgs_lam_reg, 0.0, args.clip_grad, args)
            
            
            #prune_scheduler_p1.step()

            if (i + 1) % args.eval_freq == 0:
                test_acc = eval(model, test_loader, mode=False)
                '''
                if WANDB_AVAILABLE and args.pruned_eid is not None:
                    wandb.log({
                        'training_curve/test_acc_wgs_phase1': test_acc,
                    }, step=i + 1)
                '''
                print(f"--- Phase 1: Iter {i+1} Test Acc={test_acc:.4f} ---")
                analysis_bundle = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
                summarize_concise_analysis(analysis_bundle)

                if test_acc > best_acc_p1:
                    best_acc_p1 = test_acc
                    best_model_p1_state = copy.deepcopy(model.state_dict())
                    print(f"\nIter {i+1}: New best accuracy in Phase 1: {test_acc:.4f}")
        
        # Phase 1의 best model state를 로드
        # 주의: finding_live_nodes_by_channel_with_fusion은 복사본을 사용하므로
        # phase 1 중간 평가에서 fuse를 해도 원본 모델은 변경되지 않습니다.
        model.load_state_dict(best_model_p1_state)
        if args.pruned_eid:
            torch.save(model, f'results_conv/{args.pruned_eid}_phase1_wgs_pruned.pt')
        print(f"--- Phase 1 Finished. Best accuracy: {best_acc_p1:.4f} ---")
        print(f"--- Loaded Phase 1 best model state for Phase 2 recovery ---")
        # 분석 시 복사본을 사용하므로 원본 모델은 안전합니다.
        analysis_bundle_p1 = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=True)
        summarize_and_print_analysis(analysis_bundle_p1)            

        # Phase 1 결과 wandb 로깅 (즉시 로깅)
        if WANDB_AVAILABLE and args.pruned_eid is not None:
            stats_p1 = analysis_bundle_p1.get('stats', {})
            total_dead_p1 = sum(res.get('dead', 0) for key, res in stats_p1.items() if key != 'classifier_input')
            total_nodes_p1 = sum(res.get('total', 0) for key, res in stats_p1.items() if key != 'classifier_input')
            overall_dead_ratio_p1 = (100 * total_dead_p1 / total_nodes_p1) if total_nodes_p1 > 0 else 0.0
            wandb.log({
                f"results/phase_1_wgs_pruning_best_acc": best_acc_p1,
                f"results/phase_1_wgs_pruning_dead_ratio": overall_dead_ratio_p1
            })
        
        # ================== PHASE 2: Final Recovery ==================

        #finetune step을 원본 lr의 1/10으로 진행행
        print(f"\n--- PHASE 2: Final Recovery ({args.num_iterations} iterations) ---")
        for param in model[1][-1].parameters(): param.requires_grad = False

        
        recovery_optimizer_p2 = create_finetune_optimizer(model, lr_multiplier=0.1)
        # Phase 1의 best accuracy를 기준으로 시작
        best_recovery_acc = best_acc_p1
        best_model_p2_state = copy.deepcopy(model.state_dict())

        for i in tqdm(range(args.num_iterations), desc='Phase 2: Final Recovery'):
            x, y = next(train_iter); x, y = x.to(device), y.to(device)
            
            
            train_step(model, x, y, loss_fn, recovery_optimizer_p2, 0.0, 0.0, args.clip_grad, args)
            

            if (i + 1) % args.eval_freq == 0:
                test_acc = eval(model, test_loader, mode=False)
                '''
                if WANDB_AVAILABLE and args.pruned_eid is not None:
                    wandb.log({
                        'training_curve/test_acc_recovery_phase2': test_acc,
                    }, step=i + 1)
                '''
                print(f"\nIter {i+1}: Test Acc={test_acc:.4f}")
                if test_acc > best_recovery_acc:
                    best_recovery_acc = test_acc
                    best_model_p2_state = copy.deepcopy(model.state_dict())
                    print(f"*** New best final accuracy: {best_recovery_acc:.4f} ***")

        model.load_state_dict(best_model_p2_state)
        if args.pruned_eid:
            # .pt 파일은 무조건 모델 객체로 저장
            torch.save(model, f'results_conv/{args.pruned_eid}_final.pt')

        print(f"--`- Retraining finished. Best final accuracy: {best_recovery_acc:.4f} ---")
        analysis_bundle_p2 = finding_live_nodes_by_channel_with_fusion(model, in_channels, args, device='cuda', verbose=False)
        summarize_and_print_analysis(analysis_bundle_p2)

        # Phase 2 결과 wandb 로깅 (즉시 로깅)
        if WANDB_AVAILABLE and args.pruned_eid is not None:
            stats_p2 = analysis_bundle_p2.get('stats', {})
            total_dead_p2 = sum(res.get('dead', 0) for key, res in stats_p2.items() if key != 'classifier_input')
            total_nodes_p2 = sum(res.get('total', 0) for key, res in stats_p2.items() if key != 'classifier_input')
            overall_dead_ratio_p2 = (100 * total_dead_p2 / total_nodes_p2) if total_nodes_p2 > 0 else 0.0
            wandb.log({
                f"results/phase_2_recovery_best_acc": best_recovery_acc,
                f"results/phase_2_recovery_dead_ratio": overall_dead_ratio_p2
            })
        
        

        

        #acc_after = eval(model, test_loader, mode=False)
        #print(f"\nAccuracy after 2-stage fine-tuning: {acc_after:.4f}")
        
        
        with torch.no_grad():
            final_weights = model[1][-1].weight_raw.data.round()
            pruned_count = (final_weights == 0).sum().item()
            total_count = final_weights.numel()
            prune_ratio = (100 * pruned_count / total_count) if total_count > 0 else 0.0
            print(f"\n--- Final WGS Pruning Stats ---")
            print(f"Final WGS Pruning ratio: {prune_ratio:.2f}% ({pruned_count} / {total_count} neurons pruned)")
            print(f"Final WGS Weight distribution: {final_weights.unique(return_counts=True)}")
        
        analysis_results = finding_live_nodes_by_channel_with_fusion(
            model, in_channels, args, device='cuda', verbose=False
        )

        summarize_and_print_analysis(analysis_results)
        
        # pruned_results 초기화 (WGS pruning의 경우)
        pruned_results = None
        if args.pruned_eid:
            try:
                from results_json import ResultsJSON
                pruned_results = ResultsJSON(args.pruned_eid)
            except Exception as e:
                print(f"Warning: Could not initialize ResultsJSON: {e}")
                pruned_results = None
        
        if pruned_results:
            summary_data = {
                'source_eid': args.retrain_eid,
                'accuracy_before_prune': acc_before,
                'best_accuracy_after_pruning': best_acc_p1,
                'best_accuracy_after_recovery': best_recovery_acc,
                'final_wgs_pruning_ratio': prune_ratio,
                'final_pruning_analysis': analysis_results
            }
            
                
            # Phase 1 (WGS pruning) 결과 추가
            summary_data['best_accuracy_after_wgs_pruning'] = best_acc_p1
            
            pruned_results.store_final_results(summary_data)
            pruned_results.save()
            print(f"Saved retraining curve and final results to EID: {args.pruned_eid}")
        
        exit()
    # --- 재학습 로직 끝 ---

    best_acc = 0
    best_acc_test = 0
    
    # Lambda 값 초기화 (메인 학습에서는 정규화 사용 안 함)
    current_lam_wgs = 0.0
    current_lam_crossbar = 0.0

    for m in model.modules():
        for param in m.parameters():
            param.requires_grad = True

    # --- 메인 학습 루프 ---
    for i, (x, y) in tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc='iteration',
        total=args.num_iterations,
    ):
        x = x.to(device)
        y = y.to(device)



        # -------------------------------------------
        total_l, task_l, wgs_l, crossbar_l = train_step(
            model, x, y, loss_fn, optim,
            current_lam_wgs,
            current_lam_crossbar,
            clip_grad_norm=args.clip_grad,
            args=args,
            current_iter=i,
            total_iter=args.num_iterations
        )
        
        if scheduler is not None:
            scheduler.step()



        if (i+1) % args.eval_freq == 0:
            valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
            valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
            train_accuracy_eval_mode = eval(model, train_loader, mode=False)
            train_accuracy_train_mode = -1 # Full train accuracy is slow
            test_accuracy_eval_mode = eval(model, test_loader, mode=False)
            test_accuracy_train_mode = eval(model, test_loader, mode=True)

            if test_accuracy_eval_mode > best_acc_test:
                best_acc_test = test_accuracy_eval_mode

            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'valid_acc_train_mode': valid_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
                'best_acc_test': best_acc_test,
            }

            if args.experiment_id is not None:
                results.store_results(r)
                # wandb 로깅
                if WANDB_AVAILABLE:
                    wandb.log({
                        'training_curve/test_acc_main': test_accuracy_eval_mode,
                    }, step=i + 1)
            else:
                print(r)

            if test_accuracy_eval_mode > best_acc:
                best_acc = test_accuracy_eval_mode
                
                analysis_bundle = finding_live_nodes_by_channel(
                    model, in_channels, args, device='cuda', verbose=False
                )
                stats = analysis_bundle.get('stats', {})
                total_dead = sum(res.get('dead', 0) for key, res in stats.items() if key != 'classifier_input')
                total_nodes = sum(res.get('total', 0) for key, res in stats.items() if key != 'classifier_input')
                overall_dead_ratio = (100 * total_dead / total_nodes) if total_nodes > 0 else 0.0

                feat_dead_ratio = stats.get('features', {}).get('dead_ratio', 0.0)
                cl_int_dead_ratio = stats.get('classifier_internal', {}).get('dead_ratio', 0.0)

                summary_for_json = {
                    'classifier_dead_ratio': cl_int_dead_ratio,
                    'features_dead_ratio': feat_dead_ratio,
                    'overall_dead_ratio': overall_dead_ratio
                }
                
                # 기본 결과 딕셔너리 'r'에 분석 결과(요약+분포)를 추가
                r.update(summary_for_json)
                r['distributions'] = analysis_bundle.get('distributions', {})
                
                if args.experiment_id is not None:
                    # 프루닝 정보가 포함된 풍부한 결과를 final_results에 저장
                    results.store_final_results(r)
                    # .pt 파일은 무조건 모델 객체로 저장
                    torch.save(model, f'results_conv/{args.experiment_id}.pt')
                    # wandb 로깅
                    if WANDB_AVAILABLE:
                        wandb.log({
                            'results/best_acc': test_accuracy_eval_mode,
                            'results/dead_ratio': overall_dead_ratio,
                        }, step=i + 1)
                else:
                                        # 콘솔에 한 줄 요약 출력
                    print("IS THE BEST UNTIL NOW.")
                    print(
                        f"Analysis -> Overall Dead: {overall_dead_ratio:.2f}% | "
                        f"Features Dead: {feat_dead_ratio:.2f}% | "
                        f"Classifier Dead: {cl_int_dead_ratio:.2f}%"
                    )
                # ---

            if args.experiment_id is not None:
                results.save()

    # 1. 함수를 호출하고 결과를 딕셔너리로 받습니다.
    analysis_results = finding_live_nodes_by_channel(
        model, in_channels, args, device='cuda', verbose=False
    )

    summarize_and_print_analysis(analysis_results)

    timestamp = datetime.datetime.now()
    if args.experiment_id is None:
        filename = f'results_conv/{timestamp.strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")


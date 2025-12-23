"""
mask_channel_prune_simple.py
간단한 Mask Channel Pruning (Iterative 모드) 실험용 스크립트
"""
import argparse
import math
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum
from birel.model import *
from birel.conv import ChannelMaskLayer, ResidualChannelMaskLayer, Crossbar1x1Conv, TreeConvLayer, ORPool2d
from birel.utils import finding_live_nodes_by_channel

torch.set_num_threads(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def remove_residual_mask_hooks(model):
    """Remove all forward hooks from ResidualChannelMaskLayer instances."""
    for module in model.modules():
        if isinstance(module, ResidualChannelMaskLayer):
            module._remove_input_hook()


def load_dataset(args):
    """데이터셋 로딩"""
    if 'cifar-10' in args.dataset:
        if args.model_size in ['S', 'M', 'toy']:
            def custom_transform(x):
                outputs = [(x > (i + 1) / 4.0).float() for i in range(3)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 3
        else:
            def custom_transform(x):
                outputs = [(x > (i + 1) / 32.0).float() for i in range(31)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 31
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        
        train_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=True, download=True, transform=train_transforms)
        test_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=False, transform=test_transforms)
        
        args.valid_set_size = 5000 / 50000.0
    elif 'mnist' in args.dataset:
        def custom_transform(x):
            return (x > 0.5).float()
        final_channels = 1
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform)
        ])
        
        train_set = torchvision.datasets.MNIST(root='./data-mnist', train=True, download=True, transform=train_transforms)
        test_set = torchvision.datasets.MNIST(root='./data-mnist', train=False, transform=test_transforms)
        
        args.valid_set_size = 10000 / 60000.0
    else:
        raise NotImplementedError
    
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


def get_model(args, in_channels):
    """모델 생성 (간단한 버전)"""
    class_count = 10
    k = args.num_neurons
    
    base_logic_layer_kw = dict(
        ste=False,
        implementation='cuda',
        init=args.init,
        tau=1.0
    )
    
    torch.manual_seed(0)
    
    if 'cifar-10' in args.dataset:
        base_features = [
            Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
            TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=3, padding=1, stride=1, k=k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
            Crossbar1x1Conv(in_channels=k, out_channels=4*k*2, num_blocks=k//8, connections='unique'),
            TreeConvLayer(in_channels=4*k*2, out_channels=4*k, kernel_size=3, padding=1, stride=1, k=4*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
            Crossbar1x1Conv(in_channels=4*k, out_channels=16*k*2, num_blocks=k//8, connections='unique'),
            TreeConvLayer(in_channels=16*k*2, out_channels=16*k, kernel_size=3, padding=1, stride=1, k=16*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
            Crossbar1x1Conv(in_channels=16*k, out_channels=32*k*2, num_blocks=k//8, connections='unique'),
            TreeConvLayer(in_channels=32*k*2, out_channels=32*k, kernel_size=3, padding=1, stride=1, k=32*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
        ]
        features = nn.Sequential(*base_features)
        final_feature_dim = 32 * k * 2 * 2
        
        if args.model_size == 'toy':
            layer_dims = [320*k, 160*k, 80*k]
        else:
            layer_dims = [1280*k, 640*k, 320*k]
        if args.model_size in ['B', 'L']:
            layer_dims = [d * 2 for d in layer_dims]
        
        classifier_layers = [
            nn.Flatten(),
            LogicLayer(final_feature_dim, layer_dims[0], **base_logic_layer_kw),
            LogicLayer(layer_dims[0], layer_dims[1], **base_logic_layer_kw),
            LogicLayer(layer_dims[1], layer_dims[2], **base_logic_layer_kw),
            GroupSum(k=class_count, tau=args.tau),
        ]
    elif 'mnist' in args.dataset:
        base_features = [
            Crossbar1x1Conv(in_channels=in_channels, out_channels=k*2, num_blocks=1, connections='unique'),
            TreeConvLayer(in_channels=k*2, out_channels=k, kernel_size=5, padding=0, stride=1, k=k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
            Crossbar1x1Conv(in_channels=k, out_channels=3*k*2, num_blocks=k//8, connections='unique'),
            TreeConvLayer(in_channels=3*k*2, out_channels=3*k, kernel_size=3, padding=1, stride=1, k=3*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
            Crossbar1x1Conv(in_channels=3*k, out_channels=9*k*2, num_blocks=k//8, connections='unique'),
            TreeConvLayer(in_channels=9*k*2, out_channels=9*k, kernel_size=3, padding=1, stride=1, k=9*k, logic_layer_kwargs=base_logic_layer_kw),
            ORPool2d(2, 2),
        ]
        features = nn.Sequential(*base_features)
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
        raise NotImplementedError
    
    classifier = nn.Sequential(*classifier_layers)
    model = nn.Sequential(features, classifier)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.002)
    
    return model, loss_fn, optimizer



def compute_2nd_mse_hutch_scores(
    model,
    target_module,          # LogicLayer 또는 TreeConvLayer (TreeConvLayer 내부 LogicLayer 포함 가능)
    eval_batches,          # list of (x_cpu, y_cpu) 또는 DataLoader iterable
    device='cuda',
    num_batches=40,
    num_probes=1,
    tie_to_zero=True,
    tie_to_one=True,
    eps=1e-8,
):
    """
    2nd order MSE approximation with Hutchinson trick (2nd_MSE_hutch)
    
    p-space Gauss-Newton 기반 채널별 ΔL_dist 근사

    ΔL_dist(c, tie) ≈ 0.5 * E_v [ (δp_c^T g_p,c)^2 ]
    여기서 g_p = ∂(v^T z) / ∂p, δp_c: 해당 채널의 확률 변화 (e0 - p, e15 - p)

    구현은 g_w = ∂(v^T z) / ∂w 와 p 를 이용해
    δp^T g_p = sum_k δp_k * g_w_k / p_k 로 계산.
    """

    model.train()  # approximate_channel_loss_scores와 동일하게 train 모드 사용

    # target_module에서 LogicLayer 찾기
    # TreeConvLayer인 경우 내부의 LogicLayer를 찾고, LogicLayer인 경우 그 자체를 사용
    target_layer = None
    if isinstance(target_module, LogicLayer):
        target_layer = target_module
    elif hasattr(target_module, 'cascade'):
        # TreeConvLayer인 경우, cascade 내부의 마지막 LogicLayer 찾기
        for logic_module in reversed(target_module.cascade):
            if isinstance(logic_module, LogicLayer):
                if logic_module.out_dim == target_module.out_dim:
                    target_layer = logic_module
                    break
    else:
        raise ValueError(f"target_module must be LogicLayer or TreeConvLayer, got {type(target_module)}")
    
    if target_layer is None:
        raise ValueError(f"Could not find LogicLayer in target_module {type(target_module)}")

    # eval_batches가 DataLoader든 list든 상관없이 iterator로 만듦
    if hasattr(eval_batches, '__iter__') and not isinstance(eval_batches, list):
        eval_iter = iter(eval_batches)
    else:
        eval_iter = iter(eval_batches)

    C = target_layer.out_dim
    h0 = torch.zeros(C, device=device) if tie_to_zero else None
    h1 = torch.zeros(C, device=device) if tie_to_one else None

    tau = getattr(target_layer, 'tau', 1.0)
    batches_used = 0

    for b in range(num_batches):
        try:
            x_cpu, y_cpu = next(eval_iter)
        except StopIteration:
            break

        x = x_cpu.to(device)
        y = y_cpu.to(device)

        for _ in range(num_probes):
            model.zero_grad(set_to_none=True)

            logits = model(x)  # [B, num_classes]
            v = torch.randn_like(logits)
            s = (logits * v).sum()
            s.backward()

            W = target_layer.weights         # [C, 16]
            G = target_layer.weights.grad    # [C, 16]
            if G is None:
                raise RuntimeError("target_layer.weights.grad is None")

            # softmax 확률 p, 그리고 gate 선택 인덱스 k_curr
            p = torch.softmax(W / tau, dim=-1)        # [C, 16]
            g_over_p = G / (p + eps)                  # [C, 16]

            curr_idx = W.argmax(dim=-1)               # [C]
            idx = torch.arange(C, device=device)

            # 0-tie: δp = e_0 - e_curr
            if tie_to_zero:
                delta_p0 = torch.zeros_like(p)        # [C, 16]
                # +1 at op 0
                delta_p0[idx, 0] += 1.0
                # -1 at current op
                delta_p0[idx, curr_idx] -= 1.0
                # curr_idx == 0 인 채널은 δp0 == 0 이 됨 (당연히 ΔL = 0 근사)
                inner0 = (delta_p0 * g_over_p).sum(dim=1)  # [C]
                h0 += inner0.pow(2).detach()

            # 1-tie: δp = e_15 - e_curr
            if tie_to_one:
                delta_p1 = torch.zeros_like(p)
                delta_p1[idx, 15] += 1.0
                delta_p1[idx, curr_idx] -= 1.0
                inner1 = (delta_p1 * g_over_p).sum(dim=1)  # [C]
                h1 += inner1.pow(2).detach()

        batches_used += 1

    approx_loss0 = 0.5 * h0 if tie_to_zero and h0 is not None else None
    approx_loss1 = 0.5 * h1 if tie_to_one and h1 is not None else None

    return approx_loss0, approx_loss1



def compute_2nd_ce_scores(model, target_layer, data_loader, device='cuda', loss_type='ce', num_batches=40):
    """
    2nd order CE approximation (Fisher Information / Empirical Fisher)
    Batch Effect를 제거하기 위해 Per-sample gradient squaring을 수행함.
    """
    model_mode = model.training
    model.train() # BN 등을 위해 train 모드 유지 (단, eval 모드로 해야 deterministic 할 수 있음)
    
    used_samples = 0 # 배치 수가 아니라 샘플 수로 카운트
    
    # Baseline p (softmax(weights/tau)) - 고정
    W = target_layer.weights.detach()
    tau = getattr(target_layer, 'tau', 1.0)
    p0 = torch.softmax(W / tau, dim=-1)
    C, K = p0.shape
    
    idx = torch.arange(C, device=device)
    
    # delta_p 미리 계산
    delta_p0 = torch.zeros_like(p0)
    delta_p0[idx, 0] = 1.0
    delta_p0 -= p0
    
    delta_p1 = torch.zeros_like(p0)
    delta_p1[idx, 15] = 1.0
    delta_p1 -= p0
    
    sum_s0_sq = torch.zeros(C, device=device)
    sum_s1_sq = torch.zeros(C, device=device)
    
    data_iter = iter(data_loader)
    
    for _ in range(num_batches):
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            break
        
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        batch_size = x_batch.size(0)

        # [핵심 변경] 배치를 통째로 forward/backward 하지 않고,
        # 샘플 단위(혹은 아주 작은 미니배치)로 쪼개서 gradient 제곱을 누적합니다.
        # 이렇게 하면 배치 사이즈가 변해도 결과(평균)는 수렴합니다.
        
        # 방법: autograd의 효율성을 위해 for문을 돌며 개별 처리
        for i in range(batch_size):
            # 1. 개별 샘플 추출 (차원 유지: [1, C, H, W])
            x_sample = x_batch[i:i+1]
            y_sample = y_batch[i:i+1]

            model.zero_grad(set_to_none=True)
            
            # 2. Forward & Loss
            if loss_type == 'dist':
                # Distillation loss logic (생략 가능하면 생략, 예시용)
                with torch.no_grad():
                    logits_orig = model(x_sample).detach()
                logits = model(x_sample)
                diff = logits - logits_orig
                loss = 0.5 * diff.pow(2).mean() # reduction doesn't matter for size 1
            else:
                logits = model(x_sample)
                loss = F.cross_entropy(logits, y_sample) # default reduction is mean, but size is 1
            
            # 3. Backward (개별 샘플에 대한 gradient 생성)
            loss.backward()
            
            # 4. Gradient 제곱 누적 (Fisher Information 근사)
            # p.grad는 이 샘플 하나에 대한 gradient임
            assert hasattr(target_layer, "_last_p"), "Layer Hook이 동작하지 않음"
            p = target_layer._last_p
            g_p = p.grad # [C, 16] (Batch dim is already averaged inside logic layer or handled by hook)
            
            # LogicLayer 구현 특성상 _last_p가 배치 차원을 가지고 있는지 확인 필요
            # 만약 _last_p가 [B, C, 16]이라면 i번째를 가져와야 하고,
            # LogicLayer가 내부적으로 mean을 취했다면 배치 사이즈 1이므로 그대로 사용.
            # 보통 LogicLayer 구현상 forward에서 mean을 취하지 않고 [B, ...] 라면 아래와 같이 처리:
            if g_p.dim() > 2: # [B, C, 16] 형태라면
                g_p = g_p.mean(dim=0) # 배치 1이라도 차원이 살아있을 수 있음

            # Score 계산: (delta * g)^2
            s0 = (delta_p0 * g_p).sum(dim=1)
            s1 = (delta_p1 * g_p).sum(dim=1)
            
            sum_s0_sq += s0.pow(2)
            sum_s1_sq += s1.pow(2)
            
            used_samples += 1
            
    if used_samples == 0:
        raise RuntimeError("No samples used for grad^2 approx")
    
    # 총 샘플 수로 나누어 평균 (Expectation)
    approx_loss0 = 0.5 * (sum_s0_sq / used_samples)
    approx_loss1 = 0.5 * (sum_s1_sq / used_samples)
    
    all_losses = torch.stack([approx_loss0, approx_loss1], dim=0)
    min_loss, best_tie = torch.min(all_losses, dim=0)
    
    score = -min_loss
    tie_type_mask = (best_tie == 1)
    
    model.train(model_mode)
    return approx_loss0, approx_loss1, score, tie_type_mask



def compute_2nd_mse_scores(
    model,
    target_layer: LogicLayer,
    x_eval: torch.Tensor = None,
    device='cuda',
    tau=None,
    data_loader=None,
    num_batches=40,
):
    """
    2nd order MSE approximation without Hutchinson trick (2nd_MSE)
    
    Hutchinson 없이, p-space Gauss-Newton ΔL_dist(c, tie)를 직접 계산.
    매우 느리기 때문에 conv1 + 작은 batch에서만 검증용으로 사용할 것.
    
    x_eval 또는 data_loader 중 하나를 제공해야 함.
    data_loader가 제공되면 여러 배치에 대해 평균 계산.

    Returns:
        approx_loss0: [C] tensor, 0-tie GN ΔL 근사
        approx_loss1: [C] tensor, 1-tie GN ΔL 근사
        score: [C] tensor
        tie_mask: [C] bool tensor
    """
    model_mode = model.training
    model.train()
    
    # data_loader가 제공되면 여러 배치 수집
    if data_loader is not None:
        eval_data_list = []
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            eval_data_list.append(x.detach().clone())
        
        if len(eval_data_list) == 0:
            raise RuntimeError("No batches collected for compute_2nd_mse_scores")
    else:
        if x_eval is None:
            raise ValueError("Either x_eval or data_loader must be provided")
        eval_data_list = [x_eval.to(device)]
    
    # baseline p (softmax(weights/tau)) - 모든 배치에서 동일
    W = target_layer.weights.detach()
    if tau is None:
        tau = getattr(target_layer, 'tau', 1.0)
    p0 = torch.softmax(W / tau, dim=-1)         # [C,16]
    C, K = p0.shape
    assert K == 16

    # tie 방향: absolute 버전 δp = e_tie - p0
    idx = torch.arange(C, device=device)

    delta_p0 = torch.zeros_like(p0)
    delta_p0[idx, 0] = 1.0
    delta_p0 -= p0

    delta_p1 = torch.zeros_like(p0)
    delta_p1[idx, 15] = 1.0
    delta_p1 -= p0

    # 누적 공간 (모든 배치에 대해 누적)
    acc0 = torch.zeros(C, device=device)
    acc1 = torch.zeros(C, device=device)
    total_logits = 0  # 전체 logit 수 추적

    # 각 배치에 대해 처리
    for x_eval in eval_data_list:
        # autograd용: forward를 다시 해야 하므로 grad 켜고 다시 계산
        logits = model(x_eval)
        z_flat = logits.view(-1)
        num_logits_in_batch = z_flat.numel()
        total_logits += num_logits_in_batch

        # 각 logit z_k에 대해 ∂z_k/∂p를 계산
        for k in range(num_logits_in_batch):
            s = z_flat[k]
            g_p = torch.autograd.grad(
                s, target_layer._last_p,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]  # j_k

            inner0 = (delta_p0 * g_p).sum(dim=1)
            inner1 = (delta_p1 * g_p).sum(dim=1)

            acc0 += inner0.pow(2)
            acc1 += inner1.pow(2)

    # 4) 최종 GN ΔL ≈ 1/2 * sum_k (δp^T j_k)^2
    # 전체 logit 수로 나눠서 배치 크기와 무관하게 정규화
    if total_logits == 0:
        raise RuntimeError("No logits processed in compute_2nd_mse_scores")
    approx_loss0 = 0.5 * acc0 / total_logits
    approx_loss1 = 0.5 * acc1 / total_logits

    # tie 선택 및 score 계산
    all_losses = torch.stack([approx_loss0, approx_loss1], dim=0)  # [2, C]
    min_loss, best_tie = torch.min(all_losses, dim=0)              # [C]
    score = -min_loss
    tie_mask = (best_tie == 1)  # True → 1-tie, False → 0-tie

    model.train(model_mode)
    return approx_loss0, approx_loss1, score, tie_mask



def compute_1st_relative_scores(target_layer: LogicLayer, model=None, data_loader=None, device='cuda', num_batches=40):
    """1st order relative approximation (1st_relative) - 1차 Taylor 근사 (Relative Loss)"""
    C = target_layer.out_dim
    W = target_layer.weights.detach()
    curr_idx = W.argmax(dim=-1)
    
    sum_delta0 = torch.zeros(C, device=device)
    sum_delta15 = torch.zeros(C, device=device)
    used_batches = 0
    
    # data_loader가 제공되면 여러 배치에 대해 gradient 누적
    if data_loader is not None and model is not None:
        model.train()
        train_iter = iter(data_loader)
        
        for _ in range(num_batches):
            try:
                x, y = next(train_iter)
            except StopIteration:
                break
            
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            
            out = model(x)
            loss = F.cross_entropy(out, y, reduction='mean')
            loss.backward()
            
            assert hasattr(target_layer, "_last_p"), "target_layer._last_p가 없습니다."
            p = target_layer._last_p
            g_p = p.grad
            assert g_p is not None, "target_layer._last_p.grad가 없습니다."
            
            g_curr = g_p.gather(1, curr_idx.unsqueeze(1)).squeeze(1)
            
            delta0 = g_p[:, 0] - g_curr
            delta15 = g_p[:, 15] - g_curr
            
            sum_delta0 += delta0.detach()
            sum_delta15 += delta15.detach()
            used_batches += 1
        
        if used_batches == 0:
            raise RuntimeError("No batches used for loss_relative")
        
        delta0_mean = sum_delta0 / used_batches
        delta15_mean = sum_delta15 / used_batches
        
        best_delta = torch.minimum(delta0_mean, delta15_mean)
        score = -best_delta
        tie_type = (delta15_mean < delta0_mean)
        
        return score, tie_type, delta0_mean, delta15_mean
    else:
        # 단일 배치 처리 (기존 로직) - _last_p가 이미 존재해야 함
        assert hasattr(target_layer, "_last_p"), "target_layer._last_p가 없습니다."
        assert target_layer._last_p.grad is not None, "target_layer._last_p.grad가 없습니다."
        
        p = target_layer._last_p.detach()
        g_p = target_layer._last_p.grad.detach()
        
        g_curr = g_p.gather(1, curr_idx.unsqueeze(1)).squeeze(1)
        
        delta0 = g_p[:, 0] - g_curr
        delta15 = g_p[:, 15] - g_curr
        
        best_delta = torch.minimum(delta0, delta15)
        score = -best_delta
        tie_type = (delta15 < delta0)
        
        return score, tie_type, delta0, delta15


def compute_1st_absolute_scores(target_layer: LogicLayer, tie_indices=(0, 15), model=None, data_loader=None, device='cuda', num_batches=40):
    """1st order absolute approximation (1st_absolute) - 1차 Taylor 근사 (Absolute Loss)"""
    C = target_layer.out_dim
    k0, k1 = tie_indices
    
    sum_delta0 = torch.zeros(C, device=device)
    sum_delta1 = torch.zeros(C, device=device)
    used_batches = 0
    
    # data_loader가 제공되면 여러 배치에 대해 gradient 누적
    if data_loader is not None and model is not None:
        model.train()
        train_iter = iter(data_loader)
        
        for _ in range(num_batches):
            try:
                x, y = next(train_iter)
            except StopIteration:
                break
            
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            
            out = model(x)
            loss = F.cross_entropy(out, y, reduction='mean')
            loss.backward()
            
            assert hasattr(target_layer, "_last_p"), "target_layer._last_p가 없습니다."
            p = target_layer._last_p
            g_p = p.grad
            assert g_p is not None, "target_layer._last_p.grad가 없습니다."
            
            S = (p * g_p).sum(dim=1)
            delta0 = g_p[:, k0] - S
            delta1 = g_p[:, k1] - S
            
            sum_delta0 += delta0.detach()
            sum_delta1 += delta1.detach()
            used_batches += 1
        
        if used_batches == 0:
            raise RuntimeError("No batches used for loss_absolute")
        
        delta0_mean = sum_delta0 / used_batches
        delta1_mean = sum_delta1 / used_batches
        
        best_delta = torch.minimum(delta0_mean, delta1_mean)
        score = -best_delta
        tie_type = (delta1_mean < delta0_mean)
        
        return score, tie_type, delta0_mean, delta1_mean
    else:
        # 단일 배치 처리 (기존 로직) - _last_p가 이미 존재해야 함
        assert hasattr(target_layer, "_last_p"), "target_layer._last_p가 없습니다."
        assert target_layer._last_p.grad is not None, "target_layer._last_p.grad가 없습니다."
        
        p = target_layer._last_p.detach()
        g_p = target_layer._last_p.grad.detach()
        
        S = (p * g_p).sum(dim=1)
        delta0 = g_p[:, k0] - S
        delta1 = g_p[:, k1] - S
        
        best_delta = torch.minimum(delta0, delta1)
        score = -best_delta
        tie_type = (delta1 < delta0)
        
        return score, tie_type, delta0, delta1


def compute_loss_scores(model, target_module, mask_layer, data_loader, device='cuda', num_batches=40, loss_type='mse'):
    """
    Actual loss-based pruning (loss method)
    
    각 채널을 0-tie/1-tie로 교체했을 때의 실제 loss 변화를 계산.
    
    Args:
        model: 전체 모델
        target_module: mask가 삽입될 모듈 (TreeConvLayer 또는 LogicLayer)
        mask_layer: ChannelMaskLayer 인스턴스 (이미 모델에 삽입되어 있어야 함)
        data_loader: 평가용 데이터 로더
        device: 디바이스
        num_batches: 사용할 배치 수
        loss_type: 'mse', 'l1', 또는 'ce'
    
    Returns:
        score: [C] tensor, 낮은 loss 변화일수록 높은 score
        tie_type_mask: [C] bool tensor, True면 1-tie, False면 0-tie
    """
    from tqdm import tqdm
    
    # 평가에 사용할 배치 수집
    eval_data_list = []
    collected_batches = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if collected_batches >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            eval_data_list.append((x.detach().clone(), y.detach().clone()))
            collected_batches += 1
    
    if len(eval_data_list) == 0:
        raise RuntimeError("No evaluation data collected for loss-based pruning")
    
    # Baseline logits 및 labels 수집
    print(f"    Computing baseline logits...")
    with torch.no_grad():
        baseline_logits_list = []
        baseline_labels_list = []
        for x, y in eval_data_list:
            output = model(x)
            baseline_logits_list.append(output.detach().clone())
            baseline_labels_list.append(y.detach().clone())
        baseline_logits = torch.cat(baseline_logits_list, dim=0)
        baseline_labels = torch.cat(baseline_labels_list, dim=0)
    
    collected_samples = baseline_logits.shape[0]
    print(f"    Baseline logits collected: {baseline_logits.shape} (from {collected_batches} batches, {collected_samples} samples)")
    
    # Baseline loss 계산 (CE loss의 경우 필요)
    baseline_loss = None
    if loss_type == 'ce':
        with torch.no_grad():
            baseline_loss = F.cross_entropy(baseline_logits, baseline_labels, reduction='mean')
            print(f"    Baseline CE loss: {baseline_loss.item():.4f}")
    
    # Loss 함수 정의
    def prune_loss_fn(output, target, labels=None):
        """Logits의 변화(Distortion)를 측정하는 Loss"""
        if loss_type == 'ce':
            return F.cross_entropy(output, labels, reduction='mean')
        elif loss_type == 'l1':
            return F.l1_loss(output, target, reduction='mean')
        else:  # 'mse' or default
            return F.mse_loss(output, target, reduction='mean')
    
    # 각 채널에 대해 loss 변화 계산
    out_channels = mask_layer.num_channels
    loss_changes_0tie = torch.zeros(out_channels, device=device)
    loss_changes_1tie = torch.zeros(out_channels, device=device)
    
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
            if loss_type == 'ce':
                pruned_loss_0tie = prune_loss_fn(pruned_logits_0tie, None, labels=baseline_labels)
                loss_0tie = pruned_loss_0tie - baseline_loss
            else:
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
            if loss_type == 'ce':
                pruned_loss_1tie = prune_loss_fn(pruned_logits_1tie, None, labels=baseline_labels)
                loss_1tie = pruned_loss_1tie - baseline_loss
            else:
                loss_1tie = prune_loss_fn(pruned_logits_1tie, baseline_logits)
            loss_changes_1tie[channel_idx] = loss_1tie.item()
    
    min_loss_changes = torch.minimum(loss_changes_0tie, loss_changes_1tie)
    score = -min_loss_changes  # 낮은 loss 변화일수록 높은 score
    tie_type_mask = (loss_changes_1tie < loss_changes_0tie)  # True면 1-tie가 더 낮음
    
    return score, tie_type_mask







def train_step(model, x, y, loss_fn, optimizer, clip_grad_norm, args=None, current_iter=0, total_iter=0):
    """Training step"""
    model.train()
    output = model(x)
    task_loss = loss_fn(output, y)
    
    optimizer.zero_grad()
    task_loss.backward()
    
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    
    optimizer.step()
    return task_loss.item()


def eval(model, loader, mode):
    """Evaluation"""
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


def load_n(loader, n):
    """Load n batches from loader"""
    i = 0
    while i < n:
        for d in loader:
            yield d
            i += 1
            if i >= n:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Mask Channel Pruning (Iterative)')
    
    # 기본 인자
    parser.add_argument('--retrain-eid', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument('--pruned-eid', type=str, default=None, help='New experiment ID')
    parser.add_argument('--dataset', type=str, default='cifar-10-3-thresholds', choices=['cifar-10-3-thresholds', 'mnist'])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--num-iterations', type=int, default=50000)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    
    # 모델 구조
    parser.add_argument('--model-size', type=str, default='S', choices=['toy', 'S', 'M', 'B', 'L', 'G'])
    parser.add_argument('--num-neurons', type=int, default=None)
    parser.add_argument('--init', type=str, default='residual', choices=['random', 'residual'])
    parser.add_argument('--tau', type=float, default=1.0)
    
    # Pruning 인자
    parser.add_argument('--prune-method', type=str, required=True, choices=['weight', 'loss', '2nd_CE', '2nd_MSE', '2nd_MSE_hutch', '1st_relative', '1st_absolute', 'random'])
    parser.add_argument('--prune-pct', type=float, required=True, help='Target prune percentage')
    parser.add_argument('--prune-eval-batches', type=int, default=40)
    parser.add_argument('--prune-eval-batch-size', type=int, default=None, help='Mini-batch size for pruning evaluation (default: same as --batch-size)')
    parser.add_argument('--prune-eval-samples', type=int, default=1000)
    parser.add_argument('--prune-eval-probes', type=int, default=10)
    parser.add_argument('--loss-prune-type', type=str, default='mse', choices=['l1', 'mse', 'ce'])
    parser.add_argument('--clip-grad', type=float, default=0.0)
    
    args = parser.parse_args()
    
    # 자동 설정
    if args.num_neurons is None:
        if 'cifar-10' in args.dataset:
            k_map = {'toy': 8, 'S': 32, 'M': 256, 'B': 512, 'L': 1024, 'G': 2048}
        else:
            k_map = {'S': 16, 'M': 64, 'L': 1024}
        args.num_neurons = k_map.get(args.model_size)
    
    if args.learning_rate is None:
        args.learning_rate = 0.02 if 'cifar-10' in args.dataset else 0.01
    
    if args.batch_size is None:
        if 'cifar-10' in args.dataset:
            args.batch_size = 128
        else:
            bs_map = {'S': 512, 'M': 256, 'L': 128}
            args.batch_size = bs_map.get(args.model_size, 128)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    train_loader, validation_loader, test_loader, in_channels = load_dataset(args)
    # Pruning 전용 평가 배치 크기 설정 (학습 배치와 분리 가능)
    prune_eval_bs = args.prune_eval_batch_size or args.batch_size
    prune_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=prune_eval_bs,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )
    print(f"Prune eval batch size: {prune_eval_bs}")
    
    # 모델 로딩
    model_path = args.retrain_eid if not args.retrain_eid.isdigit() else f'results_conv/{args.retrain_eid}.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        exit()
    
    print(f"Loading model from: {model_path}")
    loaded_data = torch.load(model_path, map_location=device)
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        model = loaded_data['model']
    elif isinstance(loaded_data, torch.nn.Module):
        model = loaded_data
    else:
        model, _, _ = get_model(args, in_channels)
        model.load_state_dict(loaded_data)
    
    model.to(device)
    
    acc_before = eval(model, test_loader, mode=False)
    print(f"Accuracy before pruning: {acc_before:.4f}")

    # Baseline live-node/ops 분석 (pruning 전)
    def _compute_totals(stats_dict):
        total = sum(v.get('total', 0) for k, v in stats_dict.items() if k != 'classifier_input')
        alive = sum(v.get('alive', 0) for k, v in stats_dict.items() if k != 'classifier_input')
        return total, alive

    print("Running baseline live-node analysis...")
    baseline_analysis = finding_live_nodes_by_channel(model, in_channels, args, device=device, verbose=False)
    base_ops_total, base_ops_alive = _compute_totals(baseline_analysis.get('stats', {}))
    base_gates_total, base_gates_alive = _compute_totals(baseline_analysis.get('gate_results', {}))
    print(f"Baseline OPs total={base_ops_total}, alive={base_ops_alive}")
    print(f"Baseline Gates total={base_gates_total}, alive={base_gates_alive}")
    
    # Pruning할 레이어 찾기
    all_layers = []
    if isinstance(model[0], nn.Sequential):
        for idx, module in enumerate(model[0]):
            if isinstance(module, TreeConvLayer):
                all_layers.append(('treeconv', idx, f"features_{idx}", module, 0))
    
    if isinstance(model[1], nn.Sequential):
        for idx, module in enumerate(model[1][:-1]):
            if isinstance(module, LogicLayer):
                all_layers.append(('logic', idx, f"classifier_{idx}", module, 1))
    
    print(f"Found {len(all_layers)} layers to process")
    
    num_total_phases = len(all_layers)
    
    # Iterative pruning
    for phase_idx, (layer_type, pos, layer_name, module, seq_idx) in enumerate(all_layers):
        print(f"\n{'='*80}")
        print(f"Phase {phase_idx + 1}/{num_total_phases}: {layer_name}")
        print(f"{'='*80}")
        
        total_channels = module.out_dim
        target_prune_channels = int(total_channels * args.prune_pct / 100.0)
        print(f"Target prune: {target_prune_channels}/{total_channels} channels")
        
        # target_logic_layer 찾기
        target_logic_layer = None
        if layer_type == 'treeconv' and hasattr(module, 'cascade'):
            for logic_module in reversed(module.cascade):
                if isinstance(logic_module, LogicLayer) and logic_module.out_dim == module.out_dim:
                    target_logic_layer = logic_module
                    break
        elif layer_type == 'logic':
            target_logic_layer = module
        
        if target_logic_layer is None:
            print(f"  Warning: Could not find LogicLayer, skipping...")
            continue
        
        # Mask layer 생성
        mask_layer = ChannelMaskLayer(num_channels=total_channels, device='cuda', frozen=True)
        
        # Score 계산
        score = None
        tie_type_mask = None
        
        if args.prune_method == 'loss':
            print(f"  Computing loss scores...")
            # loss 방식은 mask를 먼저 삽입해야 함
            current_modules = list(model[seq_idx])
            new_modules = []
            mask_inserted = False
            
            for idx, m in enumerate(current_modules):
                new_modules.append(m)
                if id(m) == id(module) and not mask_inserted:
                    new_modules.append(mask_layer)
                    mask_inserted = True
            
            model[seq_idx] = nn.Sequential(*new_modules)
            model.to(device)
            
            # Loss 기반 score 계산
            score, tie_type_mask = compute_loss_scores(
                model, module, mask_layer, prune_eval_loader, device=device, 
                num_batches=args.prune_eval_batches, loss_type=args.loss_prune_type
            )
            
            # 임시로 삽입한 mask 제거 (나중에 다시 삽입)
            temp_modules = list(model[seq_idx])
            temp_target_idx = None
            for temp_idx, temp_m in enumerate(temp_modules):
                if id(temp_m) == id(module):
                    temp_target_idx = temp_idx
                    break
            
            if temp_target_idx is not None and temp_target_idx + 1 < len(temp_modules):
                temp_modules.pop(temp_target_idx + 1)
            model[seq_idx] = nn.Sequential(*temp_modules)
            model.to(device)
        
        elif args.prune_method == 'weight':
            logic_weights = target_logic_layer.weights.data
            logic_probs = torch.softmax(logic_weights, dim=-1)
            prob_0tie = logic_probs[:, 0]
            prob_1tie = logic_probs[:, 15]
            score = torch.maximum(prob_0tie, prob_1tie)
            tie_type_mask = (prob_1tie > prob_0tie)
        
        elif args.prune_method == '2nd_CE':
            print(f"  Computing 2nd_CE scores...")
            approx_loss0, approx_loss1, score, tie_type_mask = compute_2nd_ce_scores(
                model, target_logic_layer, prune_eval_loader, device=device, loss_type='ce', num_batches=args.prune_eval_batches
            )
        
        
        elif args.prune_method == '1st_relative':
            print(f"  Computing 1st_relative scores...")
            score, tie_type_mask, _, _ = compute_1st_relative_scores(
                target_logic_layer, model=model, data_loader=prune_eval_loader, device=device, num_batches=args.prune_eval_batches
            )
        
        elif args.prune_method == '1st_absolute':
            print(f"  Computing 1st_absolute scores...")
            score, tie_type_mask, _, _ = compute_1st_absolute_scores(
                target_logic_layer, model=model, data_loader=prune_eval_loader, device=device, num_batches=args.prune_eval_batches
            )
        
        elif args.prune_method == '2nd_MSE':
            print(f"  Computing 2nd_MSE scores (VERY SLOW)...")
            approx_loss0, approx_loss1, score, tie_type_mask = compute_2nd_mse_scores(
                model, target_logic_layer, data_loader=prune_eval_loader, device=device, 
                num_batches=args.prune_eval_batches, tau=getattr(target_logic_layer, 'tau', 1.0)
            )
        
        elif args.prune_method == '2nd_MSE_hutch':
            print(f"  Computing 2nd_MSE_hutch scores (Hutchinson)...")
            approx_loss0, approx_loss1 = compute_2nd_mse_hutch_scores(
                model,
                target_module=module,
                eval_batches=prune_eval_loader,
                device=device,
                num_batches=args.prune_eval_batches,
                num_probes=args.prune_eval_probes,
            )
            
            if approx_loss0 is None or approx_loss1 is None:
                raise RuntimeError("approx_loss0/1 is None, check tie options")
            
            # 두 tie 중 더 작은 ΔL 선택
            all_losses = torch.stack([approx_loss0, approx_loss1], dim=0)  # [2, C]
            min_loss, best_tie = torch.min(all_losses, dim=0)              # [C]
            
            score = -min_loss               # 큰 score일수록 prune할 가치 ↑
            tie_type_mask = (best_tie == 1) # True면 1-tie, False면 0-tie
        
        elif args.prune_method == 'random':
            score = torch.rand(total_channels, device='cuda')
            tie_type_mask = torch.rand(total_channels, device='cuda') > 0.5
        
        if score is None:
            print(f"  ERROR: Could not compute score")
            continue

        # Score 통계 출력
        with torch.no_grad():
            score_mean = score.mean().item()
            score_max = score.max().item()
            score_min = score.min().item()
        print(f"  Score stats — mean: {score_mean:.4f}, max: {score_max:.4f}, min: {score_min:.4f}")
        
        # Mask 초기화
        _, sorted_indices = torch.sort(score, descending=True)
        selected_indices = sorted_indices[:target_prune_channels]
        
        with torch.no_grad():
            mask_layer.mask_weights.fill_(0.0)
            indices_0tie = selected_indices[~tie_type_mask[selected_indices]]
            indices_1tie = selected_indices[tie_type_mask[selected_indices]]
            if len(indices_0tie) > 0:
                mask_layer.mask_weights[indices_0tie, 0] = 5.0
            if len(indices_1tie) > 0:
                mask_layer.mask_weights[indices_1tie, 1] = 5.0
            
            keep_indices = torch.ones(total_channels, dtype=torch.bool, device='cuda')
            keep_indices[selected_indices] = False
            if keep_indices.any():
                mask_layer.mask_weights[keep_indices, 2] = 5.0
        
        # Mask 삽입
        current_modules = list(model[seq_idx])
        new_modules = []
        mask_inserted = False
        
        for idx, m in enumerate(current_modules):
            new_modules.append(m)
            if id(m) == id(module) and not mask_inserted:
                new_modules.append(mask_layer)
                mask_inserted = True
        
        model[seq_idx] = nn.Sequential(*new_modules)
        model.to(device)
        
        # Retraining
        print(f"\n  Retraining phase {phase_idx + 1}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.002)
        acc_before_phase = eval(model, test_loader, mode=False)
        print(f"  Accuracy before phase {phase_idx + 1}: {acc_before_phase:.4f}")
        
        best_phase_acc = acc_before_phase
        best_phase_model_state = copy.deepcopy(model.state_dict())
        
        train_iter_phase = iter(load_n(train_loader, args.num_iterations))
        
        for param in model.parameters():
            param.requires_grad = True
        
        for i in tqdm(range(args.num_iterations), desc=f'Phase {phase_idx+1}/{num_total_phases}'):
            x, y = next(train_iter_phase)
            x, y = x.to(device), y.to(device)
            train_step(model, x, y, nn.CrossEntropyLoss(), optimizer, args.clip_grad, args, i, args.num_iterations)
            
            if (i + 1) % args.eval_freq == 0:
                test_acc = eval(model, test_loader, mode=False)
                if test_acc > best_phase_acc:
                    best_phase_acc = test_acc
                    best_phase_model_state = copy.deepcopy(model.state_dict())
                    print(f"\n  Iter {i+1}: New best accuracy: {test_acc:.4f}")
        
        model.load_state_dict(best_phase_model_state)
        acc_after_phase = eval(model, test_loader, mode=False)
        print(f"\n  Phase {phase_idx + 1} finished. Best accuracy: {best_phase_acc:.4f}")
        print(f"  Improvement: {acc_after_phase - acc_before_phase:+.4f}")

        # Phase별 모델 저장 (옵션)
        if args.pruned_eid:
            save_path = f'pruned_model/{args.pruned_eid}_phase{phase_idx+1}.pt'
            remove_residual_mask_hooks(model)
            torch.save(model, save_path)
            print(f"  Saved phase model: {save_path}")

    # 최종 결과
    final_acc = eval(model, test_loader, mode=False)
    print(f"\n{'='*80}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Improvement: {final_acc - acc_before:+.4f}")
    print(f"{'='*80}")
    
    # Pruned 모델 live-node/ops 분석 및 baseline 대비 감소율 계산
    try:
        final_analysis = finding_live_nodes_by_channel(model, in_channels, args, device=device, verbose=False)
        final_ops_total, final_ops_alive = _compute_totals(final_analysis.get('stats', {}))
        final_gates_total, final_gates_alive = _compute_totals(final_analysis.get('gate_results', {}))

        ops_reduction = 100.0 * (1 - (final_ops_alive / base_ops_alive)) if base_ops_alive > 0 else 0.0
        gates_reduction = 100.0 * (1 - (final_gates_alive / base_gates_alive)) if base_gates_alive > 0 else 0.0

        print(f"OPs alive: baseline {base_ops_alive} -> pruned {final_ops_alive} (reduction {ops_reduction:.2f}%)")
        print(f"Gates alive: baseline {base_gates_alive} -> pruned {final_gates_alive} (reduction {gates_reduction:.2f}%)")
    except Exception as e:
        final_analysis = None
        print(f"Warning: finding_live_nodes_by_channel failed on pruned model: {e}")
    
    if args.pruned_eid:
        remove_residual_mask_hooks(model)
        torch.save(model, f'pruned_model/{args.pruned_eid}.pt')
        print(f"Saved final model: pruned_model/{args.pruned_eid}.pt")


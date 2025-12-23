


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruning_two_stage.py

Two-stage mask channel pruning:
- Objective for refinement (Stage-B): loss_MSE distortion to the original (all-bypass) model.
- Stage-A metric (screening): configurable per layer-type (features / classifier):
    * loss_mse_fd_ps : per-channel finite-diff on distortion, using prefix/suffix caching
    * 2nd_mse        : exact p-space Gauss-Newton 2nd-order MSE approximation

- Stage-A picks (target + overshoot) channels.
- Stage-B removes overshoot channels using binary/n-way split search or tail rescoring
  (each group evaluation uses refine_eval_batches, objective is loss_MSE).

Defaults match your usual command:
  python pruning_two_stage.py
    --retrain-eid results_conv/baseline.pt
    --dataset cifar-10-3-thresholds --model-size M
    --prune-pct 50
    --overshoot-k 5 --score-eval-batches 1 --refine-eval-batches 20
    --batch-size 4 --num-iterations 0
"""

import argparse
import math
import os
import copy
import random
import time
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from difflogic import LogicLayer
from birel.conv import (
    ChannelMaskLayer,
    ResidualChannelMaskLayer,
    TreeConvLayer,
)

torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Basic utils
# -----------------------------------------------------------------------------
def remove_residual_mask_hooks(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, ResidualChannelMaskLayer):
            module._remove_input_hook()


def eval_acc(model: nn.Module, loader, mode: bool = False) -> float:
    orig_mode = model.training
    model.train(mode=mode)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.size(0)
    model.train(orig_mode)
    return correct / total if total > 0 else 0.0


def train_step(model: nn.Module, x, y, optimizer, clip_grad_norm: float = 0.0) -> float:
    model.train()
    out = model(x)
    loss = F.cross_entropy(out, y, reduction="mean")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    return float(loss.item())


def load_n(loader, n: int):
    i = 0
    while i < n:
        for d in loader:
            yield d
            i += 1
            if i >= n:
                return


def _find_logic_layer_for_module(target_module: nn.Module) -> LogicLayer:
    if isinstance(target_module, LogicLayer):
        return target_module
    if hasattr(target_module, "cascade"):
        for lm in reversed(target_module.cascade):
            if isinstance(lm, LogicLayer) and lm.out_dim == target_module.out_dim:
                return lm
    raise ValueError(f"Could not find LogicLayer inside {type(target_module)}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def load_dataset(args):
    if "cifar-10" in args.dataset:
        if args.model_size in ["S", "M", "toy"]:
            def custom_transform(x):
                outputs = [(x > (i + 1) / 4.0).float() for i in range(3)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 3
        else:
            def custom_transform(x):
                outputs = [(x > (i + 1) / 32.0).float() for i in range(31)]
                return torch.cat(outputs, dim=0)
            final_channels = 3 * 31

        train_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform),
        ])
        test_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root="./data-cifar", train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data-cifar", train=False, transform=test_tf
        )
        args.valid_set_size = 5000 / 50000.0

    elif "mnist" in args.dataset:
        def custom_transform(x):
            return (x > 0.5).float()
        final_channels = 1

        train_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform),
        ])
        test_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(custom_transform),
        ])

        train_set = torchvision.datasets.MNIST(
            root="./data-mnist", train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.MNIST(
            root="./data-mnist", train=False, transform=test_tf
        )
        args.valid_set_size = 10000 / 60000.0
    else:
        raise NotImplementedError(args.dataset)

    train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
    valid_set_size = len(train_set) - train_set_size
    if valid_set_size > 0:
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )
    else:
        validation_set = test_set

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader, final_channels


# -----------------------------------------------------------------------------
# Mask helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def mask_all_bypass(mask_layer: ChannelMaskLayer) -> None:
    mask_layer.mask_weights.fill_(0.0)
    mask_layer.mask_weights[:, 2] = 5.0  # bypass


@torch.no_grad()
def mask_set_prune_set(mask_layer: ChannelMaskLayer, prune_indices: Set[int], tie_type_mask: torch.Tensor) -> None:
    """
    tie_type_mask: bool[C]  True => 1-tie else 0-tie
    """
    mask_all_bypass(mask_layer)
    if not prune_indices:
        return
    idx = torch.as_tensor(sorted(list(prune_indices)), device=mask_layer.mask_weights.device, dtype=torch.long)
    if idx.numel() == 0:
        return

    tie1 = tie_type_mask[idx]
    idx1 = idx[tie1]
    idx0 = idx[~tie1]

    if idx0.numel() > 0:
        mask_layer.mask_weights[idx0, 2] = 0.0
        mask_layer.mask_weights[idx0, 0] = 5.0
    if idx1.numel() > 0:
        mask_layer.mask_weights[idx1, 2] = 0.0
        mask_layer.mask_weights[idx1, 1] = 5.0


# -----------------------------------------------------------------------------
# Prefix/Suffix split builder (for loss_MSE scoring)
# -----------------------------------------------------------------------------
def insert_mask_after_module(model: nn.Sequential, seq_idx: int, module: nn.Module, mask_layer: ChannelMaskLayer) -> bool:
    current = list(model[seq_idx])
    new = []
    inserted = False
    for m in current:
        new.append(m)
        if id(m) == id(module) and not inserted:
            new.append(mask_layer)
            inserted = True
    if inserted:
        model[seq_idx] = nn.Sequential(*new).to(device)
    return inserted


def find_mask_index(seq: nn.Sequential, mask_layer: nn.Module) -> int:
    for i, m in enumerate(seq):
        if id(m) == id(mask_layer):
            return i
    raise RuntimeError("mask_layer not found in sequence")


def build_prefix_suffix(model: nn.Sequential, seq_idx: int, mask_layer: ChannelMaskLayer) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (prefix, suffix) such that:
        h = prefix(x)
        logits = suffix(mask_layer(h))
    """
    assert isinstance(model, nn.Sequential) and len(model) == 2, "Expect model = nn.Sequential(features, classifier)"

    features = model[0]
    classifier = model[1]
    assert isinstance(features, nn.Sequential)
    assert isinstance(classifier, nn.Sequential)

    if seq_idx == 0:
        # mask in features
        mi = find_mask_index(features, mask_layer)
        prefix = nn.Sequential(*list(features[:mi]))
        suffix = nn.Sequential(*list(features[mi + 1 :]), *list(classifier))
    elif seq_idx == 1:
        # mask in classifier
        mi = find_mask_index(classifier, mask_layer)
        prefix = nn.Sequential(features, *list(classifier[:mi]))
        suffix = nn.Sequential(*list(classifier[mi + 1 :]))
    else:
        raise ValueError("seq_idx must be 0 or 1")

    return prefix.to(device), suffix.to(device)


@torch.no_grad()
def collect_eval_batches(data_loader, num_batches: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    it = iter(data_loader)
    for _ in range(num_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        batches.append((x.to(device), y.to(device)))
    if len(batches) == 0:
        raise RuntimeError("No eval batches collected")
    return batches



@torch.no_grad()
def distortion_per_sample(
    out: torch.Tensor,
    teacher: torch.Tensor,
    loss_type: str = "mse",
    T: float = 1.0,
) -> torch.Tensor:
    """
    out, teacher: [N, num_classes]
    returns: [N] per-sample distortion
    """
    out = out.float()
    teacher = teacher.float()

    lt = loss_type.lower()
    if lt == "kl":
        log_p_s = F.log_softmax(out / T, dim=1)
        p_t     = F.softmax(teacher / T, dim=1)
        # [N, C] -> [N]
        return F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T ** 2)

    if lt == "mse":
        # MSE on logits (always)
        diff = (out / T) - (teacher / T)
        return diff.pow(2).mean(dim=1) * (T ** 2)

    raise ValueError(f"Unknown loss_type={loss_type} (use 'mse' or 'kl')")



class PSEvalContext:
    """
    loss_MSE distortion evaluation using cached prefix activations.
    """
    def __init__(
        self,
        prefix: nn.Module,
        suffix: nn.Module,
        mask_layer: ChannelMaskLayer,
        eval_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        cache_fp16: bool = False,
    ):
        self.prefix = prefix.eval()
        self.suffix = suffix.eval()
        self.mask = mask_layer
        self.eval_batches = eval_batches
        self.cache_fp16 = cache_fp16

        # cache prefix outputs
        self.h_list: List[torch.Tensor] = []
        for x, _ in self.eval_batches:
            h = self.prefix(x).detach()
            if self.cache_fp16:
                h = h.to(torch.float16)
            self.h_list.append(h)

        # baseline logits (all bypass)
        mask_all_bypass(self.mask)
        self.base_logits: List[torch.Tensor] = []
        for h in self.h_list:
            hh = h
            if self.cache_fp16 and hh.dtype == torch.float16:
                hh = hh.to(torch.float32)  # mask/suffix likely expect fp32
            out = self.suffix(self.mask(hh)).detach()
            self.base_logits.append(out)

    @torch.no_grad()
    def distortion_current_mask(self, loss_type="mse", T: float = 1.0) -> float:
        total = 0.0
        for h, base in zip(self.h_list, self.base_logits):
            hh = h
            if self.cache_fp16 and hh.dtype == torch.float16:
                hh = hh.to(torch.float32)
            out = self.suffix(self.mask(hh))
            loss_vec = distortion_per_sample(out, base, loss_type=loss_type, T=T)
            total += float(loss_vec.mean().item())
        return total / max(len(self.h_list), 1)

    @torch.no_grad()
    def distortion_for_prune_set(self, prune_set: Set[int], tie_type_mask: torch.Tensor,
                                 loss_type="mse", T: float = 1.0) -> float:
        mask_set_prune_set(self.mask, prune_set, tie_type_mask)
        return self.distortion_current_mask(loss_type=loss_type, T=T)

    @torch.no_grad()
    def distortion_for_single_channel(self, c: int, tie_one: bool,
                                      loss_type="mse", T: float = 1.0) -> float:
        mask_all_bypass(self.mask)
        self.mask.mask_weights[c, 2] = 0.0
        self.mask.mask_weights[c, 1 if tie_one else 0] = 5.0
        return self.distortion_current_mask(loss_type=loss_type, T=T)


# -----------------------------------------------------------------------------
# Stage-A metric: loss_mse_fd_ps (prefix/suffix cached FD)
# -----------------------------------------------------------------------------


@torch.no_grad()
def stageA_scores_loss_mse_fd_ps(
    ctx: PSEvalContext,
    num_channels: int,
    chunk_size: int = 32,
    loss_type: str = "kl",          # "kl" or "mse"
    T: float = 1.0,                 # temperature (KD-style)
) -> Tuple[torch.Tensor, torch.Tensor]:

    score_0 = torch.zeros(num_channels, device=device)
    score_1 = torch.zeros(num_channels, device=device)

    def compute_chunk_loss(c_start, c_end, tie_val_float):
        current_chunk_size = c_end - c_start
        chunk_losses = torch.zeros(current_chunk_size, device=device)
        total_counts = 0

        for h, base in zip(ctx.h_list, ctx.base_logits):
            h = h.to(device)
            base = base.to(device)

            B = h.shape[0]
            ndim = h.ndim

            if ndim == 4:
                h_exp = h.unsqueeze(0).expand(current_chunk_size, -1, -1, -1, -1).clone()
            else:
                h_exp = h.unsqueeze(0).expand(current_chunk_size, -1, -1).clone()

            indices = torch.arange(current_chunk_size, device=device)
            target_channels = torch.arange(c_start, c_end, device=device)

            if ndim == 4:
                h_exp[indices, :, target_channels, :, :] = tie_val_float
                C, H, W = h.shape[1], h.shape[2], h.shape[3]
                h_input = h_exp.reshape(-1, C, H, W)
            else:
                h_exp[indices, :, target_channels] = tie_val_float
                C = h.shape[1]
                h_input = h_exp.reshape(-1, C)

            out_flat = ctx.suffix(h_input)              # [current_chunk_size * B, num_classes]
            base_exp = base.repeat(current_chunk_size, 1)  # [current_chunk_size * B, num_classes]

            # ----------------------------
            # Loss switch: KL vs MSE
            # ----------------------------
            if loss_type.lower() == "kl":
                # KD-style KL (teacher prob, student log-prob)
                log_prob_student = F.log_softmax(out_flat / T, dim=1)
                prob_teacher     = F.softmax(base_exp / T, dim=1)
                # per-sample KL: [N]
                loss_per_sample = F.kl_div(
                    log_prob_student, prob_teacher, reduction="none"
                ).sum(dim=1) * (T ** 2)

            elif loss_type.lower() == "mse":
                # MSE on logits (always)
                # KD-style temperature scaling: compare scaled logits, then rescale by T^2 for gradient magnitude match
                diff = (out_flat / T) - (base_exp / T)
                loss_per_sample = diff.pow(2).mean(dim=1) * (T ** 2)

            else:
                raise ValueError(f"Unknown loss_type={loss_type} (use 'kl' or 'mse')")

            # chunk-wise average over B (keep per-channel in chunk)
            loss_per_chunk = loss_per_sample.view(current_chunk_size, B).mean(dim=1)
            chunk_losses += loss_per_chunk
            total_counts += 1

            del h_exp, h_input, out_flat, base_exp, loss_per_sample, loss_per_chunk

        return chunk_losses / max(total_counts, 1)

    for c_start in range(0, num_channels, chunk_size):
        c_end = min(c_start + chunk_size, num_channels)
        score_0[c_start:c_end] = compute_chunk_loss(c_start, c_end, 0.0)
        score_1[c_start:c_end] = compute_chunk_loss(c_start, c_end, 1.0)

    min_loss = torch.minimum(score_0, score_1)
    tie_mask = (score_1 < score_0)

    return -min_loss, tie_mask


# -----------------------------------------------------------------------------
# Stage-A metric: 2nd_MSE (exact Gauss-Newton)
# -----------------------------------------------------------------------------

def _compute_delta_p(target_module: nn.Module, tie_indices=(0, 15)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Compute delta_p0 and delta_p1 for tie(0) and tie(1) perturbations.
    
    Returns:
        delta_p0: [C, K] delta for tie(0)
        delta_p1: [C, K] delta for tie(1)
        p0: [C, K] baseline probability distribution
        C: number of channels
    """
    target_layer = _find_logic_layer_for_module(target_module)
    W = target_layer.weights.detach()
    tau = getattr(target_layer, "tau", 1.0)
    
    # p0 calculation
    p0 = torch.softmax(W / tau, dim=-1)  # [C, K]
    C, K = p0.shape
    
    k0, k1 = tie_indices
    idx = torch.arange(C, device=device)
    
    # Pre-calculate deltas [C, K]
    delta_p0 = torch.zeros_like(p0)
    delta_p0[idx, k0] = 1.0
    delta_p0 = delta_p0 - p0
    
    delta_p1 = torch.zeros_like(p0)
    delta_p1[idx, k1] = 1.0
    delta_p1 = delta_p1 - p0
    
    return delta_p0, delta_p1, p0, C


def stageA_scores_2nd_mse(
    model: nn.Module,
    mask_layer: ChannelMaskLayer,
    target_module: nn.Module,
    data_loader,
    num_batches: int,
    tie_indices=(0, 15),
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Gauss-Newton (p-space) for MSE distortion.
    Uses loop-based Jacobian computation for robustness with custom CUDA kernels.
    """
    model_mode = model.training
    model.train()  # Gradient computation requires train mode

    # Ensure mask does not affect logits during this stage-A metric
    with torch.no_grad():
        mask_all_bypass(mask_layer)

    # Get target_layer for accessing _last_p
    target_layer = _find_logic_layer_for_module(target_module)
    
    # Compute delta_p using helper function
    delta0, delta1, p0, C = _compute_delta_p(target_module, tie_indices)

    acc0 = torch.zeros(C, device=device)
    acc1 = torch.zeros(C, device=device)
    denom = 0

    it = iter(data_loader)
    
    # Enable grad explicitly
    with torch.enable_grad():
        for _ in range(num_batches):
            try:
                x, _y = next(it)
            except StopIteration:
                break
            x = x.to(device)

            # 1. Forward pass
            logits = model(x)       # [B, NumClasses]
            z = logits.reshape(-1)  # Flatten outputs: [N_total]
            denom += z.numel()

            # Retrieve p
            if not hasattr(target_layer, "_last_p") or target_layer._last_p is None:
                 raise RuntimeError("target_layer._last_p not found.")
            p = target_layer._last_p  # [C, 16]

            # 2. Loop-based Jacobian (Robust for custom CUDA kernels)
            for i in range(z.numel()):
                g_p_i = torch.autograd.grad(
                    outputs=z[i],
                    inputs=p,
                    retain_graph=True, 
                    create_graph=False
                )[0]
                
                if g_p_i is None:
                    continue

                inner0 = (g_p_i * delta0).sum(dim=1) 
                inner1 = (g_p_i * delta1).sum(dim=1)

                acc0 += inner0.pow(2)
                acc1 += inner1.pow(2)
                
                del g_p_i

            del logits, z

    denom = max(denom, 1)

    # 4. Final Score
    approx0 = 0.5 * acc0 / float(denom)
    approx1 = 0.5 * acc1 / float(denom)

    # Determine score
    all_losses = torch.stack([approx0, approx1], dim=0)
    min_loss, best_tie = torch.min(all_losses, dim=0)

    score = -min_loss
    tie_mask = (best_tie == 1)

    model.train(model_mode)
    return score, tie_mask



def stageA_scores_2nd_kl(
    model: nn.Module,
    mask_layer: ChannelMaskLayer,
    target_module: nn.Module,
    data_loader,
    num_batches: int,
    tie_indices=(0, 15),
    T: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Gauss-Newton (p-space) for KL distortion:
      L = T^2 * KL( softmax(z0/T) || softmax(z/T) )
    Around z=z0, 2nd-order:
      ΔL ≈ 0.5 * Δz^T F(q) Δz,  q=softmax(z0/T), F=diag(q)-qq^T
    with Δz ≈ J δp.

    Implementation:
      For each sample, compute v_k(c)=<∂z_k/∂p, δp_c> for k in classes,
      then quad(c)=E_q[v^2]-E_q[v]^2 (variance under q).
    """
    model_mode = model.training
    model.train()  # 필요하면 eval로 바꿔서 deterministic하게 써도 됨

    with torch.no_grad():
        mask_all_bypass(mask_layer)

    # Get target_layer for accessing _last_p
    target_layer = _find_logic_layer_for_module(target_module)
    
    # Compute delta_p using helper function
    delta0, delta1, p0, C = _compute_delta_p(target_module, tie_indices)

    acc0 = torch.zeros(C, device=device)
    acc1 = torch.zeros(C, device=device)
    used_samples = 0

    it = iter(data_loader)

    with torch.enable_grad():
        for _ in range(num_batches):
            try:
                x, _y = next(it)
            except StopIteration:
                break
            x = x.to(device)

            logits = model(x)  # [B, num_classes]
            B, Kout = logits.shape

            q_all = torch.softmax((logits.detach() / T), dim=1)  # [B, Kout]

            if not hasattr(target_layer, "_last_p") or target_layer._last_p is None:
                raise RuntimeError("target_layer._last_p not found. Make sure LogicLayer stores p without detach.")
            p = target_layer._last_p  # [C,16] or [B,C,16] depending on impl

            for b in range(B):
                q = q_all[b].to(device)  # [Kout]

                v0 = torch.zeros(Kout, C, device=device)
                v1 = torch.zeros(Kout, C, device=device)

                for k in range(Kout):
                    g_p = torch.autograd.grad(
                        outputs=logits[b, k],
                        inputs=p,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if g_p is None:
                        continue

                    if g_p.dim() == 3:   # [B, C, 16]
                        g = g_p[b]
                    else:                # [C, 16]
                        g = g_p

                    v0[k] = (g * delta0).sum(dim=1)
                    v1[k] = (g * delta1).sum(dim=1)

                qw = q.view(Kout, 1)  # [Kout,1]

                Ev0   = (qw * v0).sum(dim=0)          # [C]
                Ev0_2 = (qw * (v0 * v0)).sum(dim=0)   # [C]
                quad0 = Ev0_2 - Ev0 * Ev0             # [C]

                Ev1   = (qw * v1).sum(dim=0)
                Ev1_2 = (qw * (v1 * v1)).sum(dim=0)
                quad1 = Ev1_2 - Ev1 * Ev1

                acc0 += quad0
                acc1 += quad1
                used_samples += 1

            del logits, q_all

    if used_samples == 0:
        raise RuntimeError("No samples used for 2nd_KL approx")

    approx0 = 0.5 * acc0 / float(used_samples)
    approx1 = 0.5 * acc1 / float(used_samples)

    all_losses = torch.stack([approx0, approx1], dim=0)  # [2, C]
    min_loss, best_tie = torch.min(all_losses, dim=0)

    score = -min_loss
    tie_type_mask = (best_tie == 1)

    model.train(model_mode)
    return score, tie_type_mask



def stageA_scores_2nd_ce_empfisher(
    model: nn.Module,
    mask_layer: ChannelMaskLayer,
    target_module: nn.Module,
    data_loader,
    num_batches: int,
    tie_indices=(0, 15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2nd-order CE approximation in p-space using Empirical Fisher:
      ΔL ≈ 0.5 * E[ ( <∇_p L(x,y), δp> )^2 ]

    - Uses per-sample gradients (batch effect 줄이기).
    - δp는 tie(0/1)로 강제할 때의 p 변화 (delta_p0/delta_p1).
    """
    model_mode = model.training
    model.train()  # gradient 필요

    # mask 영향 제거: all-bypass
    with torch.no_grad():
        mask_all_bypass(mask_layer)

    # Get target_layer for accessing _last_p
    target_layer = _find_logic_layer_for_module(target_module)
    
    # Compute delta_p using helper function
    delta_p0, delta_p1, p0, C = _compute_delta_p(target_module, tie_indices)

    sum_s0_sq = torch.zeros(C, device=device)
    sum_s1_sq = torch.zeros(C, device=device)
    used_samples = 0

    data_iter = iter(data_loader)

    with torch.enable_grad():
        for _ in range(num_batches):
            try:
                x_batch, y_batch = next(data_iter)
            except StopIteration:
                break

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)

            for i in range(B):
                x = x_batch[i:i+1]
                y = y_batch[i:i+1]

                model.zero_grad(set_to_none=True)

                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="mean")

                # p must be connected to graph (NOT detached)
                if not hasattr(target_layer, "_last_p") or target_layer._last_p is None:
                    raise RuntimeError("target_layer._last_p not found. Make sure LogicLayer saves p without detach.")
                p = target_layer._last_p

                # safer than p.grad: works even without retain_grad()
                g_p = torch.autograd.grad(
                    outputs=loss,
                    inputs=p,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if g_p is None:
                    continue

                # handle [1,C,16] or [C,16]
                if g_p.dim() == 3:
                    g = g_p[0]
                else:
                    g = g_p

                s0 = (delta_p0 * g).sum(dim=1)  # [C]
                s1 = (delta_p1 * g).sum(dim=1)  # [C]

                sum_s0_sq += s0.pow(2)
                sum_s1_sq += s1.pow(2)
                used_samples += 1

    if used_samples == 0:
        raise RuntimeError("No samples used for 2nd_CE empirical fisher")

    approx0 = 0.5 * (sum_s0_sq / float(used_samples))
    approx1 = 0.5 * (sum_s1_sq / float(used_samples))

    all_losses = torch.stack([approx0, approx1], dim=0)  # [2,C]
    min_loss, best_tie = torch.min(all_losses, dim=0)

    score = -min_loss
    tie_type_mask = (best_tie == 1)

    model.train(model_mode)
    return score, tie_type_mask


def stageA_scores_1st_ce(
    model: nn.Module,
    mask_layer: ChannelMaskLayer,
    target_module: nn.Module,
    data_loader,
    num_batches: int,
    tie_indices=(0, 15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1st order CE approximation:
    ΔL(c, tie) ≈ E[ <∂L/∂p, δp> ]
      δp0 = e0  - p0
      δp1 = e15 - p0
    """
    model_mode = model.training
    model.train()  # gradient 필요

    # mask 영향 제거: all-bypass
    with torch.no_grad():
        mask_all_bypass(mask_layer)

    # Get target_layer for accessing _last_p
    target_layer = _find_logic_layer_for_module(target_module)
    
    # Compute delta_p using helper function
    delta_p0, delta_p1, p0, C = _compute_delta_p(target_module, tie_indices)

    # accumulate 1st-order projections (signed)
    sum_s0 = torch.zeros(C, device=device)
    sum_s1 = torch.zeros(C, device=device)
    used_samples = 0

    data_iter = iter(data_loader)

    for _ in range(num_batches):
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            break

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        B = x_batch.size(0)

        for i in range(B):
            x = x_batch[i:i+1]
            y = y_batch[i:i+1]

            model.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()

            if not hasattr(target_layer, "_last_p") or target_layer._last_p is None:
                raise RuntimeError("target_layer._last_p not found. Make sure LogicLayer saves p without detach.")
            p = target_layer._last_p
            g_p = p.grad
            if g_p is None:
                continue

            # if [B,C,16] 형태면 평균(배치 1이지만 차원 유지될 수 있음)
            if g_p.dim() > 2:
                g_p = g_p.mean(dim=0)  # [C,16]

            # 1st-order: <g_p, delta_p>
            s0 = (delta_p0 * g_p).sum(dim=1)  # [C]
            s1 = (delta_p1 * g_p).sum(dim=1)  # [C]

            sum_s0 += s0.detach()
            sum_s1 += s1.detach()
            used_samples += 1

    if used_samples == 0:
        raise RuntimeError("No samples used for 1st_CE approx")

    approx_loss0 = sum_s0 / used_samples   # signed ΔL prediction
    approx_loss1 = sum_s1 / used_samples

    # Same decision rule style as others: pick smaller predicted ΔL
    all_losses = torch.stack([approx_loss0, approx_loss1], dim=0)
    min_loss, best_tie = torch.min(all_losses, dim=0)

    score = -min_loss
    tie_type_mask = (best_tie == 1)

    model.train(model_mode)
    return score, tie_type_mask



@torch.no_grad()
def stageA_scores_weight(target_module: nn.Module, tie_indices=(0, 15)) -> Tuple[torch.Tensor, torch.Tensor]:
    target_layer = _find_logic_layer_for_module(target_module)
    W = target_layer.weights.detach()
    tau = getattr(target_layer, "tau", 1.0)
    p0 = torch.softmax(W / tau, dim=-1)  # [C,16]

    k0, k1 = tie_indices
    prob0 = p0[:, k0]
    prob1 = p0[:, k1]

    score = torch.maximum(prob0, prob1)        # 큰 게 prune 후보
    tie_mask = (prob1 > prob0)                 # True -> 1-tie
    return score, tie_mask


@torch.no_grad()
def stageA_scores_random(num_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    score = torch.rand(num_channels, device=device)
    tie_mask = (torch.rand(num_channels, device=device) > 0.5)
    return score, tie_mask



# -----------------------------------------------------------------------------
# Stage-B refinement: n-way split removal (uses prefix/suffix distortion)
# -----------------------------------------------------------------------------

def _split_list_nway(cand: List[int], n: int) -> List[List[int]]:
    n = max(2, int(n))
    n = min(n, len(cand))
    q, r = divmod(len(cand), n)
    out = []
    s = 0
    for i in range(n):
        e = s + q + (1 if i < r else 0)
        if s < e:
            out.append(cand[s:e])
        s = e
    return out


@torch.no_grad()
def find_most_harmful_channel_by_nway_split(
    ctx: PSEvalContext,
    prune_set: Set[int],
    tie_type_mask: torch.Tensor,
    split_way: int = 2,
    early_exit_size: int = 1,
    cand_order: Optional[List[int]] = None,
    group_chunk: int = 2,
    use_amp: bool = False,
    loss_type: str = "mse",
    T: float = 1.0,

) -> List[int]:
    """
    n-way split search with vectorized group loss evaluation.
    Returns: early_exit_size개의 채널 리스트 (이 채널들을 KEEP = prune_set에서 제거하면 됨)
    
    split_way=2면 binary split과 동일.
    """
    if cand_order is None:
        cand = sorted(list(prune_set))
    else:
        cand = [c for c in cand_order if c in prune_set]

    if len(cand) <= early_exit_size:
        return cand

    cur_full = set(prune_set)

    # --- 0) cur_full 전체 prune 적용한 h_full_pruned를 배치별로 1회만 생성 ---
    prune_idx = torch.as_tensor(sorted(list(cur_full)), device=device, dtype=torch.long)
    prune_val = tie_type_mask[prune_idx].to(device=device, dtype=torch.float32)  # 0/1

    h_orig_list = []
    h_pruned_full_list = []
    base_list = []

    for h, base in zip(ctx.h_list, ctx.base_logits):
        ho = h
        if ctx.cache_fp16 and ho.dtype == torch.float16:
            ho = ho.to(torch.float32)
        bt = base
        if bt.dtype in (torch.float16, torch.bfloat16):
            bt = bt.to(torch.float32)

        hp = ho.clone()
        nd = hp.ndim
        view_shape = (1, -1) + (1,) * (nd - 2)  # (1,P,1,1...) or (1,P)
        hp[:, prune_idx, ...] = prune_val.view(*view_shape)

        h_orig_list.append(ho)
        h_pruned_full_list.append(hp)
        base_list.append(bt)

    # --- 1) 그룹들을 vectorize해서 loss 계산하는 내부 함수 ---
    def eval_keep_groups_vec(groups: List[List[int]]) -> List[float]:
        G = len(groups)
        losses_cpu = torch.zeros(G, device="cpu")

        for ho, hp_full, bt in zip(h_orig_list, h_pruned_full_list, base_list):
            B = hp_full.shape[0]
            feat_shape = hp_full.shape[1:]

            # groups를 chunk로 나눠서 메모리 관리
            for s in range(0, G, max(1, group_chunk)):
                e = min(s + max(1, group_chunk), G)
                sub = groups[s:e]
                g = len(sub)

                h_exp = hp_full.unsqueeze(0).expand(g, *hp_full.shape).clone()

                for j, Gi in enumerate(sub):
                    if len(Gi) == 0:
                        continue
                    idx = torch.as_tensor(Gi, device=device, dtype=torch.long)
                    h_exp[j, :, idx, ...] = ho[:, idx, ...]  # restore keep group

                h_in = h_exp.reshape(g * B, *feat_shape)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = ctx.suffix(h_in)
                else:
                    out = ctx.suffix(h_in)

                base_rep = bt.repeat(g, 1)
                loss_vec = distortion_per_sample(out, base_rep, loss_type=loss_type, T=T)  # [g*B]
                loss_g = loss_vec.view(g, B).mean(dim=1)  # [g]

                losses_cpu[s:e] += loss_g.detach().to("cpu")

                del h_exp, h_in, out, base_rep, loss_vec, loss_g

        losses_cpu /= max(len(h_orig_list), 1)
        return [float(x) for x in losses_cpu.tolist()]

    # --- 2) n-way split search ---
    while len(cand) > early_exit_size:
        n = min(max(2, int(split_way)), len(cand))
        groups = _split_list_nway(cand, n)

        losses = eval_keep_groups_vec(groups)

        best = int(np.argmin(losses))
        cand = groups[best]

    return cand



def tail_rescore_refine(
    ctx: PSEvalContext,
    cand_sorted: List[int],
    target_prune: int,
    overshoot: int,
    tie_type_mask: torch.Tensor,
    tail_m: float,
    loss_type: str = "mse",
    T: float = 1.0,
) -> Set[int]:
    """
    Tail rescoring refinement:
      - cand_sorted: Stage-A sorted candidates (length = target_prune + overshoot)
      - overshoot:   number of extra channels beyond target_prune
      - tail_m:      tail size multiplier (tail_size = overshoot * tail_m)

    Algorithm:
      1) tail_indices = last tail_size elements of cand_sorted
      2) For each ch in tail_indices, compute distortion when we KEEP ch
         (i.e., prune_set = all candidates - {ch})
      3) Choose overshoot channels with smallest distortion when kept
      4) Final prune_set = all candidates - keep_channels
    """
    M = len(cand_sorted)
    if M == 0 or overshoot <= 0:
        return set(cand_sorted)

    # 실제 제거할(=prune_set에서 뺄) 개수는 overshoot와 M-target_prune 중 작은 값
    k_remove = min(overshoot, max(0, M - target_prune))
    if k_remove == 0:
        return set(cand_sorted)

    # tail size: 최소 k_remove, 최대 M
    tail_size = int(round(k_remove * tail_m))
    tail_size = max(k_remove, tail_size)
    tail_size = min(M, tail_size)

    tail_indices = cand_sorted[-tail_size:]

    prune_all = set(cand_sorted)
    loss_keep: Dict[int, float] = {}

    for ch in tqdm(tail_indices, desc="    Stage-B tail_rescore", leave=False):
        S_without = prune_all.difference({ch})
        loss_keep[ch] = ctx.distortion_for_prune_set(S_without, tie_type_mask, loss_type=loss_type, T=T)

    # KEEP 했을 때 distortion이 가장 작은 overshoot개 선택
    sorted_keep = sorted(tail_indices, key=lambda c: loss_keep[c])
    keep_channels = set(sorted_keep[:k_remove])

    prune_set = prune_all.difference(keep_channels)
    return prune_set



@torch.no_grad()
def _eval_keep_losses_fdps(
    ctx: PSEvalContext,
    prune_set: Set[int],
    keep_candidates: List[int],
    tie_type_mask: torch.Tensor,
    chunk_size: int = 16,
    use_amp: bool = False,
) -> List[float]:
    """
    현재 prune_set(=pruned 채널들)이 고정일 때,
    각 c in keep_candidates에 대해 'c를 KEEP(즉 prune_set에서 제거)' 했을 때의 distortion(MSE)을 계산.
    prefix는 ctx.h_list로 캐시되어 있고, suffix만 batched로 평가.
    """
    if len(keep_candidates) == 0:
        return []

    # prune_set 전체를 0/1로 덮어쓸 인덱스/값 준비
    prune_idx = torch.as_tensor(sorted(list(prune_set)), device=device, dtype=torch.long)
    prune_val = tie_type_mask[prune_idx].to(device=device, dtype=torch.float32)  # 0/1

    # candidates 텐서
    cand_all = torch.as_tensor(keep_candidates, device=device, dtype=torch.long)

    losses_cpu = torch.zeros(cand_all.numel(), device="cpu")

    for h, base in zip(ctx.h_list, ctx.base_logits):
        ho = h
        if ctx.cache_fp16 and ho.dtype == torch.float16:
            ho = ho.to(torch.float32)
        bt = base
        if bt.dtype in (torch.float16, torch.bfloat16):
            bt = bt.to(torch.float32)

        # 1) prune_set 전체 적용한 h_pruned 생성 (한 배치당 1회)
        hp = ho.clone()
        nd = hp.ndim
        view_shape = (1, -1) + (1,) * (nd - 2)  # (1,P,1,1...) or (1,P)
        hp[:, prune_idx, ...] = prune_val.view(*view_shape)

        B = hp.shape[0]
        feat_shape = hp.shape[1:]

        # 2) keep 후보들을 chunk로 묶어, 각 variant에서 해당 채널만 원복(restore)
        for s in range(0, cand_all.numel(), chunk_size):
            e = min(s + chunk_size, cand_all.numel())
            ch = cand_all[s:e]
            g = ch.numel()

            h_exp = hp.unsqueeze(0).expand(g, *hp.shape).clone()

            # 채널별 원복 (g가 작으니 loop가 안전/간단)
            for j in range(g):
                cj = int(ch[j].item())
                h_exp[j, :, cj, ...] = ho[:, cj, ...]

            h_in = h_exp.reshape(g * B, *feat_shape)

            # suffix forward (mask는 bypass 상태라 그냥 통과)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = ctx.suffix(h_in)
            else:
                out = ctx.suffix(h_in)

            base_rep = bt.repeat(g, 1)
            loss_vec = distortion_per_sample(out, base_rep, loss_type=loss_type, T=T)  # [g*B]
            loss_g = loss_vec.view(g, B).mean(dim=1)  # [g]
            losses_cpu[s:e] += loss_g.detach().to("cpu")

            del h_exp, h_in, out, base_rep, loss_vec, loss_g

        del hp

    losses_cpu /= max(len(ctx.h_list), 1)
    return [float(x) for x in losses_cpu.tolist()]



@torch.no_grad()
def stageB_refine_fdps(
    ctx: PSEvalContext,
    cand_sorted: List[int],
    candidate_set: Set[int],
    target_prune: int,
    overshoot: int,
    tie_type_mask: torch.Tensor,
    chunk_size: int = 16,
    max_cands: int = 5000,
    tail_m: float = 8.0,
    use_amp: bool = False,
    loss_type: str = "mse",
    T: float = 1.0,
) -> Set[int]:
    """
    Stage-B (fd_ps) one-shot refinement:
      - candidate_set 전체를 tie(0/1)로 prune한 상태를 기준으로
      - 각 채널 c에 대해 'c만 keep(bypass)' 했을 때 distortion을 1회 평가
      - distortion이 가장 작은 overshoot개 채널을 keep으로 선택
      - prune_set = candidate_set - keep_set

    NOTE: keep 채널 간 상호작용은 무시(1-shot leave-one-out).
    """

    prune_all = set(candidate_set)
    k_remove = min(overshoot, max(0, len(prune_all) - target_prune))
    if k_remove <= 0:
        return prune_all

    # 후보가 너무 크면 tail subset으로 제한 (classifier에서 필수)
    cand_list = list(prune_all)
    if len(cand_list) > max_cands:
        tail_size = int(round(k_remove * tail_m))
        tail_size = max(k_remove, tail_size)
        tail_size = min(len(cand_sorted), tail_size)
        cand_list = [c for c in cand_sorted[-tail_size:] if c in prune_all]

    if len(cand_list) == 0:
        return prune_all

    # mask는 "identity"로 고정 (h를 직접 수정하므로 mask가 건드리면 안 됨)
    mask_all_bypass(ctx.mask)

    # prune_all을 h에 적용하기 위한 idx/val
    prune_idx = torch.as_tensor(sorted(list(prune_all)), device=device, dtype=torch.long)
    prune_val = tie_type_mask[prune_idx].to(device=device, dtype=torch.float32)  # 0/1

    cand_all = torch.as_tensor(cand_list, device=device, dtype=torch.long)
    losses = torch.zeros(cand_all.numel(), device="cpu")

    for h, base in zip(ctx.h_list, ctx.base_logits):
        ho = h
        if ctx.cache_fp16 and ho.dtype == torch.float16:
            ho = ho.to(torch.float32)
        bt = base
        if bt.dtype in (torch.float16, torch.bfloat16):
            bt = bt.to(torch.float32)

        # 1) prune_all 적용한 hp 생성 (배치당 1회)
        hp = ho.clone()
        nd = hp.ndim
        view_shape = (1, -1) + (1,) * (nd - 2)  # (1,P,1,1...) or (1,P)
        hp[:, prune_idx, ...] = prune_val.view(*view_shape)

        B = hp.shape[0]
        feat_shape = hp.shape[1:]

        for s in range(0, cand_all.numel(), chunk_size):
            e = min(s + chunk_size, cand_all.numel())
            ch = cand_all[s:e]
            g = ch.numel()

            h_exp = hp.unsqueeze(0).expand(g, *hp.shape).clone()

            for j in range(g):
                cj = int(ch[j].item())
                h_exp[j, :, cj, ...] = ho[:, cj, ...]  # keep(restore)

            h_in = h_exp.reshape(g * B, *feat_shape)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = ctx.suffix(ctx.mask(h_in))
            else:
                out = ctx.suffix(ctx.mask(h_in))

            base_rep = bt.repeat(g, 1)
            loss_vec = distortion_per_sample(out, base_rep, loss_type=loss_type, T=T)  # [g*B]
            loss_g = loss_vec.view(g, B).mean(dim=1)  # [g]

            losses[s:e] += loss_g.detach().to("cpu")

            del h_exp, h_in, out, base_rep, loss_vec, loss_g

        del hp

    losses /= max(len(ctx.h_list), 1)

    # 3) distortion이 작은 것부터 k_remove개 keep
    order = torch.argsort(losses)  # ascending
    keep_channels = set([cand_list[int(i)] for i in order[:k_remove].tolist()])

    prune_set = prune_all.difference(keep_channels)
    return prune_set



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Two-stage Mask Channel Pruning (loss_MSE with prefix/suffix + exact 2nd_MSE)")

    # defaults = your usual command
    parser.add_argument("--retrain-eid", type=str, default="results_conv/baseline.pt")
    parser.add_argument("--pruned-eid", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="cifar-10-3-thresholds",
                        choices=["cifar-10-3-thresholds", "mnist"])
    parser.add_argument("--model-size", type=str, default="M",
                        choices=["toy", "S", "M", "B", "L", "G"])

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-iterations", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--clip-grad", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # pruning
    parser.add_argument("--prune-pct", type=float, default=50.0)

    # overshoot: k or fraction of target (frac overrides k)
    parser.add_argument("--overshoot-k", type=int, default=40)
    parser.add_argument("--overshoot-frac", type=float, default=0.0,
                        help="If >0, overshoot = round(target * overshoot_frac) and overrides overshoot-k")

    parser.add_argument("--score-eval-batches", type=int, default=2)
    parser.add_argument("--refine-eval-batches", type=int, default=20)

    # Stage-B refinement method
    parser.add_argument(
        "--stage2-method",
        type=str,
        default="binary_split",
        choices=["binary_split", "tail_rescore", "fd_ps"],
        help="Stage-B refinement method: binary_split or tail_rescore",
    )
    parser.add_argument(
        "--tail-m",
        type=float,
        default=8.0,
        help="tail_rescore only: tail size = overshoot * m",
    )

    # Stage-A metric per layer type
    parser.add_argument("--metric-features", type=str, default="2nd_mse",
                        choices=["loss_mse_fd_ps", "2nd_mse", "2nd_kl", "2nd_ce", "weight", "1st_ce", "random"])
    parser.add_argument("--metric-classifier", type=str, default="2nd_mse",
                        choices=["loss_mse_fd_ps", "2nd_mse", "2nd_kl", "2nd_ce", "weight", "1st_ce", "random"])

    # loss_MSE prefix cache dtype
    parser.add_argument("--cache-fp16", action="store_true",
                        help="Cache prefix activations in fp16 (saves memory; may affect numeric tiny)")


    parser.add_argument("--early-exit-size", type=int, default=1,
                        help="binary_split only: 한 번의 binary-split에서 반환할 후보 개수(=한 번에 keep할 채널 수)")
    parser.add_argument("--split-way", type=int, default=2,
                        help="binary_split only: 2=binary, 4/8=n-way")
    parser.add_argument("--nway-group-chunk", type=int, default=2,
                        help="n-way에서 한 번에 suffix로 평가할 그룹 개수(메모리/속도 트레이드오프)")


    # --- Stage-A distortion objective ---
    parser.add_argument(
        "--stageA-loss-type",
        type=str,
        default="mse",
        choices=["mse", "kl"],
        help="Stage-A distortion objective for loss_mse_fd_ps metric (mse or kl).",
    )
    parser.add_argument(
        "--stageA-T",
        type=float,
        default=3.0,
        help="Temperature for Stage-A distortion (used for KL, and optional for MSE KD-style scaling).",
    )

    # --- Stage-B distortion objective ---
    parser.add_argument(
        "--stageB-loss-type",
        type=str,
        default="mse",
        choices=["mse", "kl"],
        help="Stage-B distortion objective against all-bypass teacher (mse or kl).",
    )
    parser.add_argument(
        "--stageB-T",
        type=float,
        default=3.0,
        help="Temperature for Stage-B distortion (used for KL, and optional for MSE KD-style scaling).",
    )


    args = parser.parse_args()

    if args.learning_rate is None:
        args.learning_rate = 0.02 if "cifar-10" in args.dataset else 0.01

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, _, test_loader, _ = load_dataset(args)

    # load model
    model_path = args.retrain_eid if not args.retrain_eid.isdigit() else f"results_conv/{args.retrain_eid}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    print(f"Loading model from: {model_path}")
    loaded = torch.load(model_path, map_location=device)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    elif isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        raise RuntimeError("Unsupported checkpoint format (expect torch.nn.Module or dict{model})")

    model.to(device)

    acc_before = eval_acc(model, test_loader, mode=False)
    print(f"Accuracy before pruning: {acc_before:.4f}")

    # find layers
    all_layers = []
    if isinstance(model[0], nn.Sequential):
        for idx, m in enumerate(model[0]):
            if isinstance(m, TreeConvLayer):
                all_layers.append(("treeconv", idx, f"features_{idx}", m, 0))
    if isinstance(model[1], nn.Sequential):
        for idx, m in enumerate(model[1][:-1]):
            if isinstance(m, LogicLayer):
                all_layers.append(("logic", idx, f"classifier_{idx}", m, 1))
    all_layers = all_layers[1:-1]

    print(f"Found {len(all_layers)} layers to process")


    
    for phase_idx, (layer_type, _pos, layer_name, module, seq_idx) in enumerate(all_layers):
        print(f"\n{'='*80}")
        print(f"Phase {phase_idx+1}/{len(all_layers)}: {layer_name}  (type={layer_type})")
        print(f"{'='*80}")

        total_channels = module.out_dim
        target_prune = int(total_channels * args.prune_pct / 100.0)
        if target_prune <= 0:
            print("  target_prune <= 0, skipping")
            continue

        if args.overshoot_frac and args.overshoot_frac > 0.0:
            overshoot = int(round(target_prune * float(args.overshoot_frac)))
            overshoot_src = f"frac={args.overshoot_frac}"
        else:
            overshoot = max(0, int(args.overshoot_k))
            overshoot_src = f"k={args.overshoot_k}"

        cand_prune = min(total_channels, target_prune + overshoot)
        print(f"Target prune = {target_prune}/{total_channels}, overshoot({overshoot_src}) -> pick {cand_prune}")

        # insert mask layer after module in the correct sequence
        mask_layer = ChannelMaskLayer(num_channels=total_channels, device=device, frozen=True)
        inserted = insert_mask_after_module(model, seq_idx, module, mask_layer)
        if not inserted:
            print("  Warning: could not insert mask (module not found), skipping")
            continue

        # build prefix/suffix for loss_MSE evaluations
        prefix, suffix = build_prefix_suffix(model, seq_idx, mask_layer)

        # choose stage-A metric by layer type
        metric = args.metric_features if layer_type == "treeconv" else args.metric_classifier
        print(f"  Stage-A metric: {metric}  (score_eval_batches={args.score_eval_batches})")

        # ---- Stage-A ----
        tA0 = time.time()
        if metric == "loss_mse_fd_ps":
            model.eval()
            evalA = collect_eval_batches(train_loader, args.score_eval_batches)
            ctxA = PSEvalContext(prefix, suffix, mask_layer, evalA, cache_fp16=args.cache_fp16)
            score, tie_type_mask = stageA_scores_loss_mse_fd_ps(
                ctxA, total_channels, 
                loss_type=args.stageA_loss_type, 
                T=args.stageA_T
            )

        elif metric == "2nd_mse":
            # exact GN metric uses backward; keep model in train-mode internally
            score, tie_type_mask = stageA_scores_2nd_mse(
                model=model,
                mask_layer=mask_layer,
                target_module=module,
                data_loader=train_loader,
                num_batches=args.score_eval_batches,
            )
        elif metric == "2nd_kl":
            score, tie_type_mask = stageA_scores_2nd_kl(
                model=model,
                mask_layer=mask_layer,
                target_module=module,
                data_loader=train_loader,
                num_batches=args.score_eval_batches,
                T=args.stageA_T,   # 이미 넣은 stageA_T 재사용

            )
        elif metric == "2nd_ce":
            score, tie_type_mask = stageA_scores_2nd_ce_empfisher(
                model=model,
                mask_layer=mask_layer,
                target_module=module,
                data_loader=train_loader,
                num_batches=args.score_eval_batches,
            )

        elif metric == "weight":
            score, tie_type_mask = stageA_scores_weight(module)   # module이 treeconv여도 OK (내부 logic layer 찾아줌)
        elif metric == "random":
            score, tie_type_mask = stageA_scores_random(total_channels)
        elif metric == "1st_ce":
            score, tie_type_mask = stageA_scores_1st_ce(
                model=model,
                mask_layer=mask_layer,
                target_module=module,
                data_loader=train_loader,
                num_batches=args.score_eval_batches,
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
        tA1 = time.time()
        print(f"  Stage-A runtime: {tA1 - tA0:.2f} sec")

        # pick top (target + overshoot)
        _, sorted_idx = torch.sort(score, descending=True)
        cand_sorted = sorted_idx[:cand_prune].tolist()
        candidate_set: Set[int] = set(cand_sorted)

        # ---- Stage-B: refinement using TRUE objective (loss_MSE) with prefix/suffix cache ----
        if overshoot > 0:
            print(
                f"  Stage-B refine (method={args.stage2_method}): overshoot={overshoot} "
                f"(refine_eval_batches={args.refine_eval_batches})"
            )
            model.eval()

            evalB = collect_eval_batches(train_loader, args.refine_eval_batches)
            ctxB = PSEvalContext(prefix, suffix, mask_layer, evalB, cache_fp16=args.cache_fp16)

            t0 = time.time()

            if args.stage2_method == "binary_split":
                prune_set = set(candidate_set)
                cache: Dict[Tuple[int, ...], float] = {}

                keep_total = min(overshoot, max(0, len(prune_set) - target_prune))
                kept = 0

                while kept < keep_total and len(prune_set) > target_prune:
                    step = min(
                        args.early_exit_size,
                        keep_total - kept,
                        len(prune_set) - target_prune,
                    )
                    if step <= 0:
                        break
                    

                    ##### binary split is default ######
                    harmful_list = find_most_harmful_channel_by_nway_split(
                        ctx=ctxB,
                        prune_set=prune_set,
                        tie_type_mask=tie_type_mask,
                        split_way=2,      
                        early_exit_size=1,
                        cand_order=None,
                        group_chunk=2,
                        use_amp=False,
                        loss_type=args.stageB_loss_type,
                        T=args.stageB_T,
                    )

                    for h in harmful_list:
                        if h in prune_set:
                            prune_set.remove(h)  # keep (= prune에서 제외)
                            kept += 1
                            if kept >= keep_total or len(prune_set) <= target_prune:
                                break

                if len(prune_set) > target_prune:
                    remain = sorted(list(prune_set), key=lambda i: float(score[i].item()), reverse=True)
                    prune_set = set(remain[:target_prune])
                    
            elif args.stage2_method == "tail_rescore":
                prune_set = tail_rescore_refine(
                    ctx=ctxB,
                    cand_sorted=cand_sorted,
                    target_prune=target_prune,
                    overshoot=overshoot,
                    tie_type_mask=tie_type_mask,
                    tail_m=args.tail_m,
                    loss_type=args.stageB_loss_type,
                    T=args.stageB_T,
                )
            elif args.stage2_method == "fd_ps":
                prune_set = stageB_refine_fdps(
                    ctx=ctxB,
                    cand_sorted=cand_sorted,
                    candidate_set=candidate_set,
                    target_prune=target_prune,
                    overshoot=overshoot,
                    tie_type_mask=tie_type_mask,
                    chunk_size=32,
                    loss_type=args.stageB_loss_type,
                    T=args.stageB_T,
                )
            else:
                raise ValueError(f"Unknown stage2-method: {args.stage2_method}")

            t1 = time.time()
            print(f"  Stage-B ({args.stage2_method}) total runtime: {t1 - t0:.2f} sec")

        else:
            prune_set = candidate_set

        # finalize exact size
        if len(prune_set) > target_prune:
            remain = sorted(list(prune_set), key=lambda i: float(score[i].item()), reverse=True)
            prune_set = set(remain[:target_prune])

        print(f"  Final prune set size: {len(prune_set)} (target {target_prune})")
        
        with torch.no_grad():
            mask_set_prune_set(mask_layer, prune_set, tie_type_mask)

        if False:
            #---- Retrain (optional) ----
            print(f"\n  Retraining phase {phase_idx+1}... (num_iterations={args.num_iterations})")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.002)

            acc_before_phase = eval_acc(model, test_loader, mode=False)
            print(f"  Accuracy before phase {phase_idx+1}: {acc_before_phase:.4f}")

            best_acc = acc_before_phase
            best_state = copy.deepcopy(model.state_dict())

            train_iter_phase = iter(load_n(train_loader, args.num_iterations))
            for it in tqdm(range(args.num_iterations), desc=f"Phase {phase_idx+1}/{len(all_layers)}"):
                x, y = next(train_iter_phase)
                x, y = x.to(device), y.to(device)
                train_step(model, x, y, optimizer, clip_grad_norm=args.clip_grad)
                if (it + 1) % args.eval_freq == 0:
                    test_acc = eval_acc(model, test_loader, mode=False)
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_state = copy.deepcopy(model.state_dict())
                        print(f"\n  Iter {it+1}: New best accuracy: {test_acc:.4f}")
            model.load_state_dict(best_state)
            acc_after_phase = eval_acc(model, test_loader, mode=False)
            print(f"\n  Phase {phase_idx+1} finished. Best accuracy: {best_acc:.4f}")
            print(f"  Improvement: {acc_after_phase - acc_before_phase:+.4f}")

            if args.pruned_eid is not None:
                save_path = f"results_conv/{args.pruned_eid}_mask_prune_phase{phase_idx+1}.pt"
                remove_residual_mask_hooks(model)
                torch.save(model, save_path)
                print(f"  Saved: {save_path}")

    final_acc = eval_acc(model, test_loader, mode=False)
    print(f"\n{'='*80}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Improvement: {final_acc - acc_before:+.4f}")
    print(f"{'='*80}")

    if args.pruned_eid is not None:
        remove_residual_mask_hooks(model)
        final_path = f"pruned_model/{args.pruned_eid}_mask_prune_final.pt"
        torch.save(model, final_path)
        print(f"Saved final model: {final_path}")

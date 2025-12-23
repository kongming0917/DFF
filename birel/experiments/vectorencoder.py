#!/usr/bin/env python3
# diff_f16_autoenc_compare_updated.py
"""
Auto‑encoder comparison framework (MLP vs. ResidualLogicNet)

Updates (2025‑05‑07)
───────────────────
1.   Train/evaluate **one** model per run – selectable with `--model {mlp,logic}`.
2.   Pluggable LR schedulers via `--scheduler` and related hyper‑params.
3.   Temperature‑annealed `SigmoidHardSwitch` (tau schedule: linear / exp).
4.   Full experiment config & results are logged to JSON.
     • directory = today (YYYYMMDD)
     • filename  = summary_YYYYMMDD_HHMMSS.json
     • training curves: loss, LR, tau per epoch
5.   Repeated trials (`--trials`) with mean/σ of MSE stored in the same JSON.
6.   Every `eval_every` epochs (default 50) evaluate on val‑set and keep the
     best model.  Saved to `<exp_dir>/best_model.pt`, best MSE recorded.
"""
from __future__ import annotations
import argparse, json, os, math, datetime, functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import functools
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
#  ❶  Original helpers & ResidualLogicNet import (unchanged)
#      * float8/float16 conversion utils
#      * ResidualLogicNet / GroupSum etc. assumed available
# ────────────────────────────────────────────────────────────────
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet  # noqa: F401
from birel.model import * # noqa: F403, F401
from birel.utils import * # noqa: F403, F401
from birel.verilog import * # noqa: F403, F401

# (the long helper definitions from the original script follow unchanged →
#  float32_to_float8_e4m3, float8_e4m3_to_float32, f16_to_bits_torch, bits16_to_float32, etc.)

###############################################################
# … [helpers copied without modification — truncated for brevity]
###############################################################
# Paste the **exact** helper implementations from the user‑supplied script
# here.  They were omitted in this snippet to save space but are present in
# the full file stored in the canvas.
###############################################################
import numpy as np
import inspect, pprint



USEED = 4         # 2 ** (2 ** es)  (es = 1)
def _decode_posit8(byte: int) -> float:
    """단일 8-bit Posit 값 → float32 (Python float)."""
    assert 0 <= byte < 256
    if byte == 0x00:          # 0
        return 0.0
    if byte == 0x80:          # NaR
        return math.nan

    sign = -1.0 if (byte & 0x80) else 1.0          # MSB
    # 음수값: two's-complement 로 magnitude 복원
    if sign < 0:
        byte = ((~byte) & 0xFF) + 1

    # ① sign bit 제거 → 7-bit 스트림
    bitstr = f"{byte:08b}"[1:]                     # e.g. '1010101'

    # ② regime: 첫 비트와 동일한 연속 run 길이
    reg_sign = bitstr[0]
    run_len  = 1
    for b in bitstr[1:]:
        if b == reg_sign:
            run_len += 1
        else:
            break

    k = run_len - 1 if reg_sign == '1' else -run_len
    idx_after_regime = run_len + 1                  # run + terminating bit
    rest = bitstr[idx_after_regime:]

    # ③ exponent (es = 1)
    exp_bits = rest[:1] if len(rest) >= 1 else '0'
    exp_val  = int(exp_bits, 2)

    # ④ fraction
    frac_bits = rest[1:]
    frac_val  = 0.0
    for i, bit in enumerate(frac_bits, start=1):
        frac_val += int(bit) / (2 ** i)

    # ⑤ 합산
    return sign * (USEED ** k) * (2.0 ** exp_val) * (1.0 + frac_val)


# ───────── LUT 구축 & 벡터화된 encode/decode ──────────
@functools.lru_cache(maxsize=1)
def _get_posit8_lut() -> np.ndarray:
    """index (0..255) → float32 값 LUT (NaR=nan)."""
    table = np.array([_decode_posit8(i) for i in range(256)], dtype=np.float32)
    return table


def posit8_to_float32(p: np.ndarray) -> np.ndarray:
    """
    p : np.uint8 배열
    ↳  동일 shape 의 float32 배열 반환
    """
    lut = _get_posit8_lut()
    return lut[p.astype(np.uint8)].astype(np.float32)


def float32_to_posit8(x: np.ndarray) -> np.ndarray:
    """
    x : float32 배열
    ↳  최근접 8-bit Posit(정수) 배열  (np.uint8)
    (NaR 은 제외하고 최댓값/최솟값에서 saturate)
    """
    lut = _get_posit8_lut()                      # (256,)
    # NaR(0x80) 은 비교 대상에서 제외
    valid_idx = np.array([i for i in range(256) if i != 0x80], dtype=np.uint8)
    valid_lut = lut[valid_idx]                  # (255,)

    # broadcasting: |x - lut| 최소값 인덱스
    diff = np.abs(x.astype(np.float32)[..., None] - valid_lut)   # (..., 255)
    nearest = diff.argmin(axis=-1)                               # (...)
    return valid_idx[nearest].astype(np.uint8)


# ─────────────── float16 비트 ↔ np.float32  ─────────────── #


def f16_to_bits_torch(x_f16: torch.Tensor) -> torch.Tensor:
    """
    float16 Tensor  ➜  (B,16) bit Tensor  (LSB-first, float32)
    * torch.uint16 은 없으므로 int16 로 view
    * 배치 크기 B 는 어떤 값이든 자동 처리
    """
    # 1) float16 → int16 재해석  (메모리 공유, 연산 비용 0)
    x_i16 = x_f16.view(torch.int16)                  # (B,)

    # 2) 두 바이트 추출
    byte0 = (x_i16 & 0x00FF).to(torch.uint8)           # LSB
    byte1 = ((x_i16 >> 8) & 0x00FF).to(torch.uint8)   # MSB
    bytes2 = torch.stack([byte0, byte1], dim=1)      # (B, 2)

    # 3) 각 바이트를 비트로 전개  (B,2,8)
    bits = torch.stack([(bytes2 >> i) & 1
                         for i in range(8)], dim=-1)   # (B,2,8)

    # 4) 마지막 두 축(2,8) → 16 flatten → float32
    return bits.flatten(start_dim=1).float()            # (B,16)




import torch

def bits16_to_float32(bits: torch.Tensor) -> torch.Tensor:
    """
    bits : (..., 16) — 이미 0/1 로 binarize 된 float16 비트
             [0:10]  : 10-bit mantissa  (LSB → MSB)
             [10:15] : 5-bit exponent   (LSB → MSB)
             [15]    : sign (0=+, 1=−)

    returns : torch.float32  (동일 leading dims)
    """
    assert bits.shape[-1] == 16, "expect last dim = 16"
    dtype, device = bits.dtype, bits.device

    # ────────── (3) 지수·가수 정수화 ──────────
    pow2_10 = 2.0 ** torch.arange(10, dtype=dtype, device=device)   # 2^0 … 2^9
    pow2_5  = 2.0 ** torch.arange(5 , dtype=dtype, device=device)   # 2^0 … 2^4

    mant_bits = bits[..., 0:10]         # (…,10)
    exp_bits  = bits[..., 10:15]        # (…, 5)

    mant_int = (mant_bits * pow2_10).sum(dim=-1)       # Σ b_i·2^i
    exp_int  = (exp_bits  * pow2_5 ).sum(dim=-1)       # Σ b_i·2^i

    # ────────── (4) 부호 처리 ──────────
    sign_bit = bits[..., 15]                            # 0 또는 1
    sign     = 1.0 - 2.0 * sign_bit                    # 0→+1, 1→−1

    # ────────── (5) float16 값 재구성 ──────────
    bias = 15.0
    mantissa = mant_int / 1024.0                        # 2^10 로 나눔
    value = sign * (1.0 + mantissa) * torch.pow(
        torch.full_like(exp_int, 2.0), exp_int - bias
    )

    return value.to(torch.float32)


def weighted_mse(pred, target, alpha=5.0):
    """
    pred, target : (B,) float32  (Decoder가 회귀한 실수)
    alpha        : exponent 비트가 하나라도 틀리면 MSE × alpha
    """
    # 1) float → 16-bit 벡터
    pred_bits = f16_to_bits_torch(pred.to(torch.float16))
    true_bits = f16_to_bits_torch(target.to(torch.float16))

    # 2) exponent (bit10~14) 비교
    exp_equal = (pred_bits[:, 10:15] == true_bits[:, 10:15]).all(dim=1)  # (B,)
    w = torch.where(exp_equal, 1.0, alpha).to(pred.device)               # (B,)

    # 3) 가중 MSE
    return ((pred - target) ** 2 * w).mean()

def ulp_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    float16 Tensor a, b → 각 원소별 ULP 거리(float32)
    (NaN/Inf 는 결과가 정의되지 않음)
    """
    # 1) bit-pattern을 16-bit 부호없는 정수로 재해석
    a_bits = a.to(torch.float16).view(torch.int16).to(torch.int32)
    b_bits = b.to(torch.float16).view(torch.int16).to(torch.int32)

    # 2) 부호 비트(15) 추출
    a_ord = a_bits ^ ((a_bits >> 15) * 0x7FFF)
    b_ord = b_bits ^ ((b_bits >> 15) * 0x7FFF)

    # 3) 절댓값 차이가 곧 ULP 거리
    return (a_ord - b_ord).abs().to(torch.float32)



def custom_loss(output, target, lambda_mse=0.0, lambda_cos=0.0, lambda_range=0.0, lambda_ulp = 1.0):
    """
    output: model output tensor (float32 or float16)
    target: original fp16 activation (float32 or float16)
    """
    # Ensure float32 for stability in loss calculations
    output = output.float()
    target = target.float()

    # 1. Mean Squared Error
    mse_loss = nn.functional.mse_loss(output, target)

    # 2. Cosine Similarity Loss (1 - cosine_similarity)
    cos_sim = nn.functional.cosine_similarity(output, target, dim=-1)
    cosine_loss = 1.0 - cos_sim.mean()

    # 3. Range penalty (values should stay within [-1.0, 1.0] or desired fp16-safe range)
    range_penalty = torch.mean(torch.clamp(torch.abs(output) - 1.0, min=0.0))

    # 4. ULP error
    ulp_penalty = ulp_error(output, target).mean()

    
    #total_loss = nn.functional.l1_loss(output, target)

    
    # Combine
    total_loss = (
        lambda_mse * mse_loss +
        lambda_cos * cosine_loss +
        lambda_range * range_penalty +
        lambda_ulp * ulp_penalty
    )
    
    return total_loss

def kd_loss(student_out, teacher_out):
    # 예) 기본은 MSE, 필요하면 SmoothL1 등으로 교체 가능
    return nn.functional.mse_loss(student_out, teacher_out)



USEED = 4         # 2 ** (2 ** es)  (es = 1)

def _decode_posit8(byte: int) -> float:
    """단일 8-bit Posit 값 → float32 (Python float)."""
    assert 0 <= byte < 256
    if byte == 0x00:          # 0
        return 0.0
    if byte == 0x80:          # NaR
        return math.nan

    sign = -1.0 if (byte & 0x80) else 1.0          # MSB
    # 음수값: two's-complement 로 magnitude 복원
    if sign < 0:
        byte = ((~byte) & 0xFF) + 1

    # ① sign bit 제거 → 7-bit 스트림
    bitstr = f"{byte:08b}"[1:]                     # e.g. '1010101'

    # ② regime: 첫 비트와 동일한 연속 run 길이
    reg_sign = bitstr[0]
    run_len  = 1
    for b in bitstr[1:]:
        if b == reg_sign:
            run_len += 1
        else:
            break

    k = run_len - 1 if reg_sign == '1' else -run_len
    idx_after_regime = run_len + 1                  # run + terminating bit
    rest = bitstr[idx_after_regime:]

    # ③ exponent (es = 1)
    exp_bits = rest[:1] if len(rest) >= 1 else '0'
    exp_val  = int(exp_bits, 2)

    # ④ fraction
    frac_bits = rest[1:]
    frac_val  = 0.0
    for i, bit in enumerate(frac_bits, start=1):
        frac_val += int(bit) / (2 ** i)

    # ⑤ 합산
    return sign * (USEED ** k) * (2.0 ** exp_val) * (1.0 + frac_val)


# ───────── LUT 구축 & 벡터화된 encode/decode ──────────
@functools.lru_cache(maxsize=1)
def _get_posit8_lut() -> np.ndarray:
    """index (0..255) → float32 값 LUT (NaR=nan)."""
    table = np.array([_decode_posit8(i) for i in range(256)], dtype=np.float32)
    return table


def posit8_to_float32(p: np.ndarray) -> np.ndarray:
    """
    p : np.uint8 배열
    ↳  동일 shape 의 float32 배열 반환
    """
    lut = _get_posit8_lut()
    return lut[p.astype(np.uint8)].astype(np.float32)


def float32_to_posit8(x: np.ndarray) -> np.ndarray:
    """
    x : float32 배열
    ↳  최근접 8-bit Posit(정수) 배열  (np.uint8)
    (NaR 은 제외하고 최댓값/최솟값에서 saturate)
    """
    lut = _get_posit8_lut()                      # (256,)
    # NaR(0x80) 은 비교 대상에서 제외
    valid_idx = np.array([i for i in range(256) if i != 0x80], dtype=np.uint8)
    valid_lut = lut[valid_idx]                  # (255,)

    # broadcasting: |x - lut| 최소값 인덱스
    diff = np.abs(x.astype(np.float32)[..., None] - valid_lut)   # (..., 255)
    nearest = diff.argmin(axis=-1)                               # (...)
    return valid_idx[nearest].astype(np.uint8)


# ────────── float8e4m3 helpers ──────────
_FLOAT8_E4M3FN_MAX_NORMAL = 448.0
_FLOAT8_E4M3FN_MIN_NORMAL = 2**-6

def float32_to_float8_e4m3(x: np.ndarray) -> np.ndarray:
    """
    x : float32 배열
    ↳  최근접 float8e4m3 (정수) 배열  (np.uint8)
    (NaN/Inf 는 제외하고 최댓값/최솟값에서 saturate)
    """
    # Create the lookup table for float8_e4m3fn
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if i == 0b10000000: # NaN
            lut[i] = np.nan
        else:
            sign = -1.0 if (i & 0b10000000) else 1.0
            byte_val = i & 0b01111111 # Remove sign bit
            
            exponent = (byte_val >> 3) & 0b00001111 # 4 exponent bits
            mantissa = byte_val & 0b00000111 # 3 mantissa bits

            if exponent == 0: # Subnormal or zero
                if mantissa == 0:
                    val = 0.0
                else:
                    val = (mantissa / 8.0) * 2**-6 # 2**(1-7)
            elif exponent == 0b1111: # Inf (not applicable for e4m3fn) or NaN (handled above)
                val = np.inf # Should not be reached for e4m3fn if 0x80 handled
            else: # Normal
                val = (1.0 + mantissa / 8.0) * (2**(exponent - 7)) # Exponent bias of 7
            
            lut[i] = sign * val
            
    # NaR(0x80) 은 비교 대상에서 제외
    valid_idx = np.array([i for i in range(256) if i != 0x80], dtype=np.uint8)
    valid_lut = lut[valid_idx]

    # broadcasting: |x - lut| 최소값 인덱스
    diff = np.abs(x.astype(np.float32)[..., None] - valid_lut)
    nearest = diff.argmin(axis=-1)
    return valid_idx[nearest].astype(np.uint8)


def float8_e4m3_to_float32(p: np.ndarray) -> np.ndarray:
    """
    p : np.uint8 배열
    ↳  동일 shape 의 float32 배열 반환
    """
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if i == 0b10000000: # NaN
            lut[i] = np.nan
        else:
            sign = -1.0 if (i & 0b10000000) else 1.0
            byte_val = i & 0b01111111 # Remove sign bit
            
            exponent = (byte_val >> 3) & 0b00001111 # 4 exponent bits
            mantissa = byte_val & 0b00000111 # 3 mantissa bits

            if exponent == 0: # Subnormal or zero
                if mantissa == 0:
                    val = 0.0
                else:
                    val = (mantissa / 8.0) * 2**-6 # 2**(1-7)
            elif exponent == 0b1111: # Inf (not applicable for e4m3fn) or NaN (handled above)
                val = np.inf # Should not be reached for e4m3fn if 0x80 handled
            else: # Normal
                val = (1.0 + mantissa / 8.0) * (2**(exponent - 7)) # Exponent bias of 7
            
            lut[i] = sign * val
    return lut[p.astype(np.uint8)].astype(np.float32)


# ────────── float8e5m2 helpers ──────────
def float32_to_float8_e5m2(x: np.ndarray) -> np.ndarray:
    """
    x : float32 배열
    ↳  최근접 float8e5m2 (정수) 배열  (np.uint8)
    """
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if (i & 0x80) else 1.0
        byte_val = i & 0x7F # Remove sign bit

        exponent = (byte_val >> 2) & 0x1F # 5 exponent bits
        mantissa = byte_val & 0x03 # 2 mantissa bits

        if exponent == 0: # Subnormal or zero
            if mantissa == 0:
                val = 0.0
            else:
                val = (mantissa / 4.0) * 2**(-14) # 2**(1-15)
        elif exponent == 0x1F: # Inf or NaN
            if mantissa == 0:
                val = np.inf
            else:
                val = np.nan
        else: # Normal
            val = (1.0 + mantissa / 4.0) * (2**(exponent - 15)) # Exponent bias of 15
        
        lut[i] = sign * val
        
    # Exclude NaN (0x7F for +NaN, 0xFF for -NaN) and Inf values from direct comparison for finding nearest
    # Instead, we handle saturation to the max/min finite values.
    # The default behavior of argmin on abs diff will usually pick the closest finite value
    # or 0 if NaN is present, so explicit handling might be needed if exact NaN/Inf conversion is critical.
    
    # For now, let's just use the LUT directly for lookup for simplicity and assume finite values for training.
    # If the input contains NaNs/Infs, their nearest finite representation will be found.
    valid_idx = np.array([i for i in range(256) if not np.isnan(lut[i]) and not np.isinf(lut[i])], dtype=np.uint8)
    valid_lut = lut[valid_idx]

    # broadcasting: |x - lut| 최소값 인덱스
    diff = np.abs(x.astype(np.float32)[..., None] - valid_lut)
    nearest = diff.argmin(axis=-1)
    return valid_idx[nearest].astype(np.uint8)


def float8_e5m2_to_float32(p: np.ndarray) -> np.ndarray:
    """
    p : np.uint8 배열
    ↳  동일 shape 의 float32 배열 반환
    """
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if (i & 0x80) else 1.0
        byte_val = i & 0x7F # Remove sign bit

        exponent = (byte_val >> 2) & 0x1F # 5 exponent bits
        mantissa = byte_val & 0x03 # 2 mantissa bits

        if exponent == 0: # Subnormal or zero
            if mantissa == 0:
                val = 0.0
            else:
                val = (mantissa / 4.0) * 2**(-14) # 2**(1-15)
        elif exponent == 0x1F: # Inf or NaN
            if mantissa == 0:
                val = np.inf
            else:
                val = np.nan
        else: # Normal
            val = (1.0 + mantissa / 4.0) * (2**(exponent - 15)) # Exponent bias of 15
        
        lut[i] = sign * val
    return lut[p.astype(np.uint8)].astype(np.float32)

def attach_grad2norm_logger(model: torch.nn.Module, freq: int = 100):
    """
    모델의 모든 파라미터의 gradient norm을 로깅하는 클로저를 반환합니다.
    """
    grad_norms = []
    
    def log_grad_norm():
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        #print(f"  Grad Norm: {total_norm:.4f}")
        return total_norm

    # For debugging: print norms periodically
    # def print_grad_norms_hook(module, grad_input, grad_output):
    #     if module.training and len(grad_norms) % freq == 0:
    #         log_grad_norm()

    # for module in model.modules():
    #     module.register_backward_hook(print_grad_norms_hook)
            
    return log_grad_norm


# ────────────────────────────────────────────────────────────────
#  ❷  Modules with tau‑annealed sigmoid
# ────────────────────────────────────────────────────────────────
class SigmoidHardSwitch(nn.Module):
    """Sigmoid with temperature‐annealing, hard switch at eval time."""
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.register_buffer("tau", torch.tensor(float(tau)))

    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.sigmoid(x / self.tau)
        else:
            #return torch.sigmoid(x / self.tau)
            return (x >= 0).to(x.dtype)

# ────────────────────────────────────────────────────────────────
#  ❸  Model builders (identical topology, new Sigmoid class used)
# ────────────────────────────────────────────────────────────────
# MLP helpers ----------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

def build_mlp_autoenc(bitwidth: int, latent: int, enc_w: List[int], dec_w: List[int], device: str, vector_size: int):
    # MLP는 벡터 길이에 따라 입력/출력 차원이 변경됩니다.
    input_dim = bitwidth * vector_size  # ex: 16 * 4 = 64
    output_dim = vector_size            # output is float value, not bit

    enc_layers: List[nn.Module] = []
    in_d = input_dim
    for h in enc_w:
        enc_layers += [MLPBlock(in_d, h)]
        in_d = h
    enc_layers += [nn.Linear(in_d, latent), SigmoidHardSwitch()] # Latent bottleneck has `latent` dimensions

    dec_layers: List[nn.Module] = []
    in_d = latent
    for h in dec_w:
        dec_layers += [MLPBlock(in_d, h)]
        in_d = h
    dec_layers += [nn.Linear(in_d, output_dim)] # Decoder outputs `vector_size` float values

    return nn.Sequential(*enc_layers, *dec_layers).to(device)


# LogicNet builder ----------------------------------------------
def build_logic_autoenc(bitwidth: int, latent: int, enc_w: List[int], dec_w: List[int],
                        n_layers: int, device: str, k_history: int = 8,
                        enc_logic_layer_ste: bool = False, dec_logic_layer_ste: bool = False, k_keep: int = 0,
                        encoder_noise_prob: float = 0.0, decoder_noise_prob: float = 0.0,
                        implementation: str = 'python', connections: str = 'ste',
                        use_ternary: bool = True, k_history_include_input: bool = False,
                        last_connections: str = 'unique',
                        group_sum_tau: float = 100,
                        initialization: str = 'normal', vector_size: int = 1, block_size: int = 16):

    # LogicNet도 벡터 길이에 따라 입력/출력 차원 변경
    input_dim = bitwidth * vector_size
    output_dim = vector_size

    enc = torch.nn.Sequential(
            LogicBlock(n_in=input_dim, n_out=enc_w[0], width=[enc_w[0]], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections='ste', use_crossbar_tree=False, initialization="normal", mean = 0.0, std = 0.2, block_size = 16),
            LogicBlock(n_in=enc_w[0], n_out=enc_w[-2], width=enc_w[1:-1], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections='ste', use_crossbar_tree=False, initialization="normal", mean = 0.0, std = 0.2, block_size = 16),
            
            LogicBlock(n_in=enc_w[-2], n_out=latent, width=enc_w[-1], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections=last_connections, use_crossbar_tree=False, initialization="normal", mean = 0.0, std = 0.2),
            VotingLayer(latent, tau=group_sum_tau, use_ternary=use_ternary)
    )
    dec = torch.nn.Sequential(
            LogicBlock(n_in=latent, n_out=dec_w[0], width=[dec_w[0]], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections='ste', use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            LogicBlock(n_in=dec_w[0], n_out=dec_w[-2], width=dec_w[1:-1], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections='unique', use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
                                
            LogicBlock(n_in=dec_w[-2], n_out=dec_w[-1], width=dec_w[-1], implementation='cuda', k_history=1,
                                logic_layer_ste=True, crossbar_ste=True, connections=last_connections, use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            RegressionLayer(output_dim, use_ternary=use_ternary) # Decoder outputs `output_dim` float values
    )
    
    return nn.Sequential(enc, dec).to(device)

# ────────────────────────────────────────────────────────────────
#  ❹  Argument parser
# ────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model / topology -------------------------------------------------------
    p.add_argument('--model', choices=['mlp', 'logic'], default='logic',
                    help='Which auto‑encoder to run')

    p.add_argument('--dataset', choices=['uniform', 'normal', 'normal1p2', 'normal1p5', 'normal2', 'llama3'], default='uniform',
                    help='Which dataset to use')

    p.add_argument('--enc-width', type=lambda s: eval(s), default='[1024, 1024, 1024]')
    p.add_argument('--dec-width', type=lambda s: eval(s), default='[1024, 1024, 1024]')
    p.add_argument('--latent', type=int, default=8, help='Latent bit‑width')
    p.add_argument('--layers', type=int, default=3, help='Layers (for LogicNet, not used in MLP)')
    p.add_argument('--connections', type=str, default='ste', help='Connections')
    p.add_argument('--vector-size', type=int, default=1,
                    help='Number of scalar values to combine into a vector for compression')
    p.add_argument('--block-size', type=int, default=None,
                    help='Number of scalar values to combine into a vector for compression')


    # ▼ 새로 추가된 activation-loader 옵션
    p.add_argument('--data-type', choices=[
        'query', 'key', 'value', 'dkey', 'dvalue', 'attn', 'dattn'
    ], default='key',
        help='Which tensor to load from activation dump files')
    p.add_argument('--layer', type=int, default=0,
        help='Layer index to load')
    p.add_argument('--head', type=int, default=None,
        help='Head index (None = all heads)')
    p.add_argument('--root-dir', type=str, default='../calibration_sets/opt-125m',
        help='Directory where .pt activation files are stored')






    # optimiser & scheduler --------------------------------------------------
    p.add_argument('--lr', type=float, default=0.1, help='Base learning‑rate')
    p.add_argument('--scheduler', choices=['manual', 'step', 'cosine', 'plateau', 'none', 'cosineanneal'],
                    default='step', help='LR scheduler type')
    p.add_argument('--step-size', type=int, default=150, help='StepLR: step_size')
    p.add_argument('--gamma', type=float, default=0.1, help='LR decay factor')
    p.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    p.add_argument('--patience', type=int, default=10, help='Plateau patience')

    # tau annealing ----------------------------------------------------------
    p.add_argument('--tau-start', type=float, default=1.0, help='Initial tau')
    p.add_argument('--tau-end',   type=float, default=1.0, help='Final tau')
    p.add_argument('--tau-sched', choices=['linear', 'exp'], default='linear')

    p.add_argument('--noise-start', type=float, default=0.1, help='Initial noise')
    p.add_argument('--noise-end',   type=float, default=None, help='Final noise')
    p.add_argument('--noise-sched', choices=['linear', 'exp'], default='linear')

    p.add_argument('--clip-grad', type=float, default=None, help='Clip grad')


    # misc -------------------------------------------------------------------
    p.add_argument('--epochs', type=int, default=400, help='Training epochs')
    p.add_argument('--batch',  type=int, default=512, help='Batch size')
    p.add_argument('--eval-every', type=int, default=10, help='Eval interval (epochs)')
    p.add_argument('--trials', type=int, default=1, help='# repeated experiments')

    # logic‑specific
    p.add_argument('--no-enc-logic-layer-ste',dest='enc_logic_layer_ste', action='store_false', default=True)
    p.add_argument('--no-dec-logic-layer-ste',dest='dec_logic_layer_ste', action='store_false', default=True)
    p.add_argument('--k-history', type=int, default=1)
    p.add_argument('--k-keep', type=int, default=0, help='-3 is identity, -4 is regression')
    p.add_argument('--input-in-history', dest='k_history_include_input', action='store_true',  default=False, help='Include input in history')
    p.add_argument('--train-size', type=int, default=50000, help='Training dataset size')
    p.add_argument('--test-size', type=int, default=5000, help='Test dataset size')

    p.add_argument('--encoder-noise-prob', type=float, default=0.1, help='Encoder noise probability')
    p.add_argument('--decoder-noise-prob', type=float, default=0.1, help='Decoder noise probability')

    p.add_argument('--implementation', type=str, default='cuda', help='Implementation')
    p.add_argument('--no-ternary', dest='use_ternary', action='store_false', default=True, help='Use ternary')
    p.add_argument('--log-gradnorm', dest='log_gradnorm', action='store_true', default=False, help='Log grad norm')


    # initialize option
    p.add_argument('--initialization', choices=['uniform', 'normal', 'residual'], default='normal')
    
    p.add_argument('--last-connections', type=str, default='unique', help='Last connections')
    p.add_argument('--group-sum-tau', type=float, default=1, help='Scaler')


    # get_args()
    p.add_argument('--kd-weight', type=float, default=0.3,
                    help='λ_kd : MSE_teacher ↔ student 가중치')

    p.add_argument('--pruning-epochs', type=int, default=20,
               help='0-mask freeze 기간(epoch)')

    return p.parse_args()

# ────────────────────────────────────────────────────────────────
#  ❺  Dataset utilities (unchanged)
# ────────────────────────────────────────────────────────────────
# make_dataset, f16_to_bits_torch, bits16_to_float32 … already defined above

# ────────────────────────────────────────────────────────────────
#  ❻  Training / evaluation helpers
# ────────────────────────────────────────────────────────────────

def create_scheduler(opt, args, total_epochs):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'manual':
        # handled explicitly inside train loop (same as original script)
        return 'manual'
    if args.scheduler == 'step':
        if args.warmup_epochs > 0:
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers = [
                    # ① 선형 워밍업: 0 → 1·lr 로 증가
                    torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs),
                    # ② 본격 스텝 디케이
                    torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
                    ],
                milestones = [args.warmup_epochs]          # 첫 번째 스케줄이 끝나는 시점
            )
        else:
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
        return sched
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    if args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=args.gamma,
                                                            patience=args.patience, verbose=False)
    if args.scheduler == 'cosineanneal':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-4)
    raise ValueError(f"Unknown scheduler {args.scheduler}")


def current_tau(epoch: int, total_epochs: int, args) -> float:
    total_epochs = total_epochs # *1//3
    epoch = min(epoch, total_epochs)

    if args.tau_sched == 'linear':
        return args.tau_start + (args.tau_end - args.tau_start) * (epoch / (total_epochs - 1))
    # exponential decay
    ratio = (epoch / (total_epochs - 1))
    return args.tau_start * (args.tau_end / args.tau_start) ** ratio


def set_module_tau(net: nn.Module, tau_val: float):
    for m in net.modules():
        if isinstance(m, SigmoidHardSwitch):
            m.set_tau(tau_val)
        elif isinstance(m, VotingLayer):
            m.tau = tau_val
        elif isinstance(m, Binarize):
            m.tau = tau_val

        #elif isinstance(m, LogicLayer):
        #    m.tau = tau_val

def current_noise(epoch: int, total_epochs: int, args) -> float:
 #   total_epochs = total_epochs//3
 #   epoch = min(epoch, total_epochs)


    if args.noise_sched == 'linear':
        return args.noise_start + (args.noise_end - args.noise_start) * (epoch / (total_epochs - 1))
    # exponential decay
    ratio = (epoch / (total_epochs - 1))
    return args.noise_start * (args.noise_end / args.noise_start) ** ratio

def set_module_noise(net: nn.Module, noise_val: float):
    for m in net.modules():
        if isinstance(m, VotingLayer):
            m.noise_prob = noise_val
        elif isinstance(m, RegressionLayer):
            m.noise_prob = noise_val
        elif isinstance(m, BitFlip):
            m.noise_prob = noise_val


# ───────── 설정 ─────────
ENC_PREFIX = "enc."    # Sequential 구조라면 "0."
DEC_PREFIX = "dec."    # Sequential 구조라면 "1."

ENC_LAYERS_TO_MASK = {"0", "1", "2"}   # enc.0,1,2
DEC_LAYERS_TO_MASK = {}          # dec.0


def is_enc_param(name: str) -> bool:
    return name.startswith(ENC_PREFIX) and "weights" in name


def is_target_param(name: str) -> bool:
    """
    True  → 0-mask / 프리즈 대상 파라미터
    False → 그대로 학습
    """
    # ---------- Encoder ----------
    if name.startswith(ENC_PREFIX):
        after = name[len(ENC_PREFIX):]          # "0.weights" ...
        layer_idx = after.split(".")[0]          # "0"
        return layer_idx in ENC_LAYERS_TO_MASK and "weights" in name

    # ---------- Decoder ----------
    if name.startswith(DEC_PREFIX):
        after = name[len(DEC_PREFIX):]
        layer_idx = after.split(".")[0]
        return layer_idx in DEC_LAYERS_TO_MASK and "weights" in name

    return False          # 그 외 파라미터는 마스크 X


# ------------------------------------------------------------------
# 1) 맨티사 추출 함수 – torch 연산만 사용 (GPU 상에서 그대로 동작)
# ------------------------------------------------------------------
_MANT_MASK_F16 = 0x03FF          # float16 10-bit mantissa
_SCALE_F16     = 1024.0          # 2^10  (= 정규화용)

class MantissaSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        # --- bitcast & mask (gradient는 신경 안 써도 됨) ---
        x_f16 = x.to(torch.float16).contiguous()
        bits  = x_f16.view(torch.int16).to(torch.int32)
        mant  = (bits & _MANT_MASK_F16).to(torch.float32)
        if scale:
            mant = mant / _SCALE_F16
        # 역전파 때 원본 dtype이 필요할 수도 있어 저장
        ctx.save_for_backward(x)
        ctx.scale = scale
        return mant

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through: dL/dx ≈ dL/dmant
        (x,) = ctx.saved_tensors
        return grad_output.to(x.dtype), None          # scale에는 grad 없음


def _mantissa_f16(x: torch.Tensor, scale: bool = True) -> torch.Tensor:
    return MantissaSTE.apply(x, scale)

def mantissa_mse(pred: torch.Tensor,
                 tgt:  torch.Tensor,
                 scale: bool = True) -> torch.Tensor:
    p_m = _mantissa_f16(pred, scale)
    t_m = _mantissa_f16(tgt.detach(), scale)   # 대상은 gradient 필요 X
    return torch.mean((p_m - t_m) ** 2)


def evaluate(net: nn.Module, bits: torch.Tensor, ref_flat_scalars: np.ndarray, device: str) -> float:
    """
    vector_size에 맞게 처리된 모델 입력과 평탄화된 실수 타겟을 받아 MSE를 계산합니다.
    net: 모델 (AutoEncoder)
    bits: 모델의 입력 (N_vectors, 16 * vector_size)
    ref_flat_scalars: 원본 실수 값 (N_total_scalars,)
    device: 연산을 수행할 디바이스
    """
    net.eval()
    with torch.no_grad():
        # 모델 출력은 (Batch, vector_size) 형태
        rec_vectors = net(bits.to(device)).cpu().numpy()
        
        # MSE 계산을 위해 (Batch * vector_size,) 형태로 평탄화
        rec_flat_scalars = rec_vectors.reshape(-1) 
        
    return float(np.mean((rec_flat_scalars - ref_flat_scalars) ** 2))

# evaluate_extended 함수는 더 이상 사용하지 않음
# def evaluate_extended(...):
#     ...


def train_single_trial(args, device, x_tr, x_te, bits_tr, bits_te, p8_tr, p8_te,
                         mse_f8: float, trial_id: int,
                         save_dir: Path, encoder_noise_prob: float, decoder_noise_prob: float) -> Dict:
    # build model
    if args.model == 'mlp':
        net = build_mlp_autoenc(16, args.latent, args.enc_width, args.dec_width, device, args.vector_size)
    else:
        net = build_logic_autoenc(16, args.latent, args.enc_width, args.dec_width,
                                  args.layers, device, args.k_history,
                                  args.enc_logic_layer_ste, args.dec_logic_layer_ste, args.k_keep,
                                  encoder_noise_prob, decoder_noise_prob, args.implementation,
                                  args.connections, args.use_ternary, args.k_history_include_input,
                                  args.last_connections, args.group_sum_tau, args.initialization, args.vector_size, args.block_size)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = create_scheduler(opt, args, args.epochs)


    #opt = torch.optim.AdamW(net.parameters(),
    #                         lr=0.1,
    #                         betas=(0.9, 0.999),
    #                         weight_decay=1e-4)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(opt,
    #                                              step_size=100,  # 1/4 of epochs
    #                                              gamma=0.3)


    # DataLoader 생성시 x_tr과 bits_tr의 shape은 이미 (N_vectors, vector_size)와 (N_vectors, 16 * vector_size)
    loader = DataLoader(TensorDataset(bits_tr, torch.from_numpy(x_tr), torch.from_numpy(p8_tr)),
                         batch_size=args.batch, shuffle=True)
    lossf =  nn.MSELoss()

    # ------------------------------------------------------------------
    # A) 준비 단계
    # ------------------------------------------------------------------
    masks = {n: torch.ones_like(p, dtype=torch.bool)     # True = 학습 가능
              for n,p in net.named_parameters() if "weights" in n}
    
    check_freq   = args.pruning_epochs                  # 20 epoch마다 점검
    freeze_after = int(args.epochs*0.3)
    epsilon      = 0.05                 # |w|<ε 면 0 처리
    min_live_frac = 0.2                  # 레이어별 최소 20 % 남김




    # tracking -------------------------------------------------------------
    hist_loss: List[float] = []
    hist_lr:   List[float] = []
    hist_tau:  List[float] = []
    hist_noise:    List[float] = []
    hist_val_mse:  List[float] = []

    best_mse: float = math.inf
    best_state: Dict | None = None

    if args.log_gradnorm:
        get_last_norm = attach_grad2norm_logger(net)
    
    zero_masks = {}
    with torch.no_grad():
        for n, p in net.named_parameters():
            if is_enc_param(n):
                zero_masks[n] = (p.abs() < epsilon)
    
    
    last50_start = args.epochs - 49            # 예: epochs=1200 → 1151

    
    for ep in range(1, args.epochs + 1):
        net.train()
        tau_now = current_tau(ep - 1, args.epochs, args)

        
        set_module_tau(net, tau_now)
        if args.noise_end is not None:
            noise_now = current_noise(ep - 1, args.epochs, args)
            set_module_noise(net, noise_now)
        else:
            noise_now = args.encoder_noise_prob


        running = 0.0
        for xb, target_vectors, teacher_vectors in loader: # target_vectors: (Batch, vector_size)
            xb, target_vectors = xb.to(device), target_vectors.to(device)

            # normalize input
            xb = xb 
            opt.zero_grad()
            
            recon_val_vectors = net(xb).view(-1, args.vector_size) # recon_val_vectors: (Batch, vector_size)
            
            # Loss 계산 시 벡터 단위로 맞추기
            loss = lossf(recon_val_vectors, target_vectors)
            
            loss.backward()
            
            if args.clip_grad is not None:
                utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad) 
            opt.step()
                            
            running += loss.item() * xb.size(0)
            

        epoch_loss = running / len(loader.dataset)
        hist_loss.append(epoch_loss)
        hist_lr.append(opt.param_groups[0]['lr'])
        hist_tau.append(tau_now)
        hist_noise.append(noise_now)
        
        # manual LR schedule (same as original) ---------------------------
        if scheduler == 'manual':
            if ep == 1:
                opt.param_groups[0]['lr'] = args.lr * 0.1
            elif ep == 10:
                opt.param_groups[0]['lr'] = args.lr
            elif ep == 150:
                opt.param_groups[0]['lr'] = args.lr * 0.1
            elif ep == 300:
                opt.param_groups[0]['lr'] = args.lr * 0.01
        elif scheduler is not None and args.scheduler != 'plateau':
            scheduler.step()



        # evaluation -------------------------------------------------------
        if ep % args.eval_every == 0 or ep == args.epochs:
            # evaluate 함수 호출 (bits_te는 (N_vectors, 16 * vector_size), x_te.reshape(-1)은 (N_total_scalars,))
            mse_now = evaluate(net, bits_te, x_te.reshape(-1), device)
            
            hist_val_mse.append(mse_now)

            if mse_now < best_mse:
                best_mse = mse_now
                best_state = {
                    'epoch': ep,
                    'model_state_dict': net.state_dict(),
                    'mse': best_mse
                }
            print(f"[Trial {trial_id}] Epoch {ep:03d}/{args.epochs}  "
                  f"train-MSE={epoch_loss:.3e}  val-MSE={mse_now:.3e}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  tau={tau_now:.3f}  noise={noise_now:.3f}")
            if args.log_gradnorm:
                get_last_norm()
        else:
            hist_val_mse.append(0) # Not evaluated, append 0 or None

    # save best model -------------------------------------------------------
    model_path = save_dir / 'best_model.pt'
    if best_state is not None:
        torch.save(best_state, model_path)

    # final evaluation ------------------------------------------------------
    final_mse = evaluate(net, bits_te, x_te.reshape(-1), device) # Final eval uses the best model, reshape x_te

    return {
        'best_mse': best_mse,
        'final_mse': final_mse,
        'loss_curve': hist_loss,
        'val_mse_curve': hist_val_mse,
        'lr_curve': hist_lr,
        'tau_curve': hist_tau,
        'noise_curve': hist_noise,
        'model_path': str(model_path)
    }


FILE_TEMPLATES = {          # data_type → 파일명 패턴
    "query"   : "{prefix}_query.pt",
    "key"     : "{prefix}_key.pt",
    "value"   : "{prefix}_value.pt",
    "dkey"    : "{prefix}_dkey.pt",
    "dvalue"  : "{prefix}_dvalue.pt",
    "attn"    : "{prefix}_attn.pt",     # (B,H,S,S)
    "dattn"   : "{prefix}_dattn.pt",
}

# ─────────────────────────────────────────────────────────────
# 2) 공통 로더: (B,S,D) → (B·S, D) 로 평탄화
def _flatten_tokens(t: torch.Tensor, pad_to: int | None = None) -> torch.Tensor:
    """
    (B,S,D) → (B·S, D)
    (B,S,S) → (B·S, S)    ← attn
    pad_to  : int  → 두 번째 축을 그 길이까지 zero-pad
    """
    if t.dim() == 3:
        flat = t.reshape(-1, t.size(-1))
        if pad_to is not None and flat.size(-1) < pad_to:
            pad = (0, pad_to - flat.size(-1))
            flat = F.pad(flat, pad)      # 오른쪽 zero-padding
        elif pad_to is not None and flat.size(-1) > pad_to:
            flat = flat[:, :pad_to]      # truncate
        return flat
    return t             # dim≠3 → 그대로

def _load_activation_dataset(prefix: str,    # 모델 이름 or 파일 prefix
                             data_type: str,  # "query" | "key" | "value"
                             layer: int,
                             head : Optional[int],
                             n: int,
                             seed: int,
                             root_dir: str = ".",
                             sample_mode: str = "random") -> np.ndarray:
    path = os.path.join(root_dir,
                        FILE_TEMPLATES[data_type].format(prefix=prefix))
    dct = torch.load(path, map_location="cpu")        # {layer:int → list[head]}

    tensors = []
    heads = [head] if head is not None else range(len(dct[layer]))
    for h in heads:
        for t in dct[layer][h]:                     # list[tensor]
            tensors.append(_flatten_tokens(t))

    acts = torch.cat(tensors, dim=0)

    # ── 샘플링 옵션 ─────────────────────────────────────
    total = len(acts)
    if sample_mode == "all" or n == -1 or n >= total:
        return acts.numpy().astype(np.float32)

    if sample_mode == "sequential":
        acts = acts[:n]
    else:  # "random"
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(total, size=n, replace=False)
        acts = acts[idx]

    return acts.numpy().astype(np.float32).reshape(-1)




# ──────────────────── 데이터 셋 ────────────────────
def make_dataset(dataset: str                 = "uniform",
                 n: int                       = 20_000,
                 seed: int                    = 0,
                 # Model-mode 추가 옵션 ↓ (synthetic 모드에선 무시)
                 data_type: str               = "key",
                 layer: int                   = 0,
                 head: Optional[int]          = None,
                 root_dir: str                = ".",
                 vector_size: int             = 1) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    raw_data: np.ndarray

    if dataset == 'uniform':
        raw_data = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
    elif dataset == 'normal':
        raw_data = rng.normal(0, 1.0, size=(n,)).astype(np.float32)
    elif dataset == 'normal1p2':
        raw_data = rng.normal(0, 1.2, size=(n,)).astype(np.float32)
    elif dataset == 'normal1p5':
        raw_data = rng.normal(0, 1.5, size=(n,)).astype(np.float32)
    elif dataset == 'normal2':
        raw_data = rng.normal(0, 2, size=(n,)).astype(np.float32)
    elif dataset == 'llama3':
        #llama3 데이터셋은 (sequence_length, hidden_dim) 또는 (sequence_length, num_heads, head_dim) 형태
        # 이를 평탄화하여 (N,) 스칼라 배열로 만든 후 vector_size에 맞춰 재구성해야 함
        data_full = torch.load('/data/calibration_sets/llama-3-8B/calibration_dump/wikitext2_1024_llama-3-8b_key.pt')
        tensor_list = []
        # 예시: 특정 레이어와 헤드에서 일부 데이터만 로드 (실제 사용 시 범위 조절)
        # 이 예시에서는 data[10][:3] -> layer 10, first 3 heads
        # 모든 헤드를 사용하거나, 특정 헤드를 지정하는 argparse 인자를 활용해야 함
        
        # NOTE: _load_activation_dataset 함수가 이미 N개의 값을 sampling 하므로,
        # 여기서는 단순히 호출해서 N개의 스칼라 값을 가져오는 것으로 변경합니다.
        raw_data = _load_activation_dataset(prefix="wikitext2_1024_llama-3-8b",
                                            data_type=data_type,
                                            layer=layer,
                                            head=head,
                                            n=n, # make_dataset의 n과 일치시키기 위해
                                            seed=seed,
                                            root_dir=root_dir,
                                            sample_mode="random")
    elif dataset.startswith('wikitext2_2048_opt-'): # OPT models
         raw_data = _load_activation_dataset(prefix=dataset,
                                             data_type=data_type,
                                             layer=layer,
                                             head=head,
                                             n=n,
                                             seed=seed,
                                             root_dir=root_dir,
                                             sample_mode="random")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # vector_size에 맞춰 데이터 재구성
    # 전체 데이터 크기가 vector_size의 배수가 되도록 자르거나 패딩
    num_elements = raw_data.size
    # n이 vector_size의 배수가 되도록 조정
    n = (n // vector_size) * vector_size
    raw_data = raw_data[:n] # 필요한 만큼만 사용
    
    num_vectors = n // vector_size # 보정된 n을 사용
    
    reshaped_data = raw_data.reshape(num_vectors, vector_size)
    print(f"Original data size (total scalars): {num_elements}, Using {n} scalars for {reshaped_data.shape[0]} vectors of size {reshaped_data.shape[1]}")
    return reshaped_data.astype(np.float32)


# ────────────────────────────────────────────────────────────────
#  ❼  Main entry
# ────────────────────────────────────────────────────────────────


def main():

    # print(torch.cuda.memory_summary())    # 전체 요약
    # print("allocated:", torch.cuda.memory_allocated())   # 실제 사용 중
    # print("reserved: ", torch.cuda.memory_reserved())    # 캐시 예약 중
    args = get_args()

    args.enc_width = list(args.enc_width)  # type: ignore
    args.dec_width = list(args.dec_width)  # type: ignore

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # construct output directory ------------------------------------------------
    date_dir = Path(datetime.datetime.now().strftime('%Y%m%d'))
    now = datetime.datetime.now().strftime('%H%M%S')
    date_dir.mkdir(parents=True, exist_ok=True)
    date_dir = Path(f"{date_dir}/{now}")
    print(' '.join(sys.argv) +" # " + str(date_dir))
    date_dir.mkdir(parents=True, exist_ok=True)

    # dataset -------------------------------------------------------------------
    Ntrain, Ntest = args.train_size, args.test_size
    # make_dataset에서 반환되는 x_all은 (N_total_vectors, vector_size) 형태
    x_all_vectors = make_dataset(dataset = args.dataset, n = Ntrain + Ntest, 
                                 data_type=args.data_type, layer = args.layer, 
                                 head = args.head, root_dir= args.root_dir,
                                 vector_size=args.vector_size)
    
    # 훈련/테스트 세트 분리
    # x_tr_vectors: (Ntrain_vectors, vector_size), x_te_vectors: (Ntest_vectors, vector_size)
    num_train_vectors = Ntrain // args.vector_size
    num_test_vectors = Ntest // args.vector_size # 실제 테스트 벡터 수
    x_tr_vectors, x_te_vectors = x_all_vectors[:num_train_vectors], x_all_vectors[num_train_vectors:num_train_vectors + num_test_vectors]

    # MLP의 입력은 (Batch, 16 * vector_size) 형태가 되어야 함
    # LogicNet의 입력도 (Batch, 16 * vector_size) 형태가 되어야 함
    
    # 각 스칼라 값을 float16 비트로 변환한 후 평탄화
    # bits_tr: (Ntrain_vectors, vector_size * 16)
    bits_tr = f16_to_bits_torch(torch.from_numpy(x_tr_vectors).to(torch.float16).reshape(-1)).reshape(x_tr_vectors.shape[0], -1)
    bits_te = f16_to_bits_torch(torch.from_numpy(x_te_vectors).to(torch.float16).reshape(-1)).reshape(x_te_vectors.shape[0], -1)
    
    # evaluate 함수에 전달할 x_te_flat은 (N_total_test_scalars,)
    x_te_flat_for_eval = x_te_vectors.reshape(-1)

    # Posit-교사 출력 (float32) 한 번에 계산
    # p8_labels: (N_total_vectors, vector_size)
    with torch.no_grad():
        p8_labels_flat = posit8_to_float32(float32_to_posit8(x_all_vectors.reshape(-1))) 
        p8_labels = p8_labels_flat.reshape(x_all_vectors.shape[0], args.vector_size)
    p8_tr, p8_te = p8_labels[:num_train_vectors], p8_labels[num_train_vectors:num_train_vectors + num_test_vectors]


    # float8 baseline -----------------------------------------------------------
    # baseline은 스칼라 값으로 계산합니다.
    x_te_flat = x_te_vectors.reshape(-1) # (Ntest * vector_size,)

    f8   = float32_to_float8_e4m3(x_te_flat)
    f8rec= float8_e4m3_to_float32(f8)
    mse_f8 = float(np.mean((f8rec - x_te_flat) ** 2))

    f8_e3m4 = float32_to_float8_e3m4(x_te_flat)
    f8_e3m4rec = float8_e3m4_to_float32(f8_e3m4)
    mse_f8_e3m4 = float(np.mean((f8_e3m4rec - x_te_flat) ** 2))  

    f8_e5m2 = float32_to_float8_e5m2(x_te_flat)
    f8_e5m2rec = float8_e5m2_to_float32(f8_e5m2)
    mse_f8_e5m2 = float(np.mean((f8_e5m2rec - x_te_flat) ** 2))
    
    
    p8          = float32_to_posit8(x_te_flat)          # NEW
    p8_rec      = posit8_to_float32(p8)
    mse_p8      = float(np.mean((p8_rec - x_te_flat) ** 2))

    

    posit = float32_to_posit8(x_te_flat)
    positrec = posit8_to_float32(posit)
    mse_posit = float(np.mean((positrec - x_te_flat) ** 2))  

    # repeat trials -------------------------------------------------------------
    trial_metrics: List[Dict] = []
    for t in range(1, args.trials + 1):
        trial_dir = date_dir / f"trial_{t:02d}"
        trial_dir.mkdir(exist_ok=True)
        # train_single_trial에 x_te_flat_for_eval 전달
        metrics = train_single_trial(args, device, x_tr_vectors, x_te_vectors, bits_tr, bits_te, p8_tr, p8_te,
                                     mse_f8, t, trial_dir, args.encoder_noise_prob, args.decoder_noise_prob)
        trial_metrics.append(metrics)

    # aggregate -----------------------------------------------------------------
    mse_list = [m['best_mse'] for m in trial_metrics]
    mean_mse = float(np.mean(mse_list))
    std_mse  = float(np.std(mse_list))

    # JSON summary --------------------------------------------------------------
    cfg_str = f"{args.model}_enc_width{str(args.enc_width)}_dec_width{str(args.dec_width)}_latent{args.latent}_vecsize{args.vector_size}_lr{args.lr:g}".replace('.', 'p')
    json_path = date_dir / f"{cfg_str}.json"

    # ────────── Experiment Summary 문자열 빌드 ──────────
    summary_text = (
        f"float16→float8_e3m4→float32 MSE : {mse_f8_e3m4:.3e}\n"
        f"float16→float8_e4m3→float32 MSE : {mse_f8:.3e}\n"
        f"float16→float8_e5m2→float32 MSE : {mse_f8_e5m2:.3e}\n"
        f"float16→posit8→float32 MSE : {mse_p8:.3e}\n"
        f"{args.model.upper()} {16*args.vector_size}→{args.latent}→{args.vector_size} final MSEs : "
        f"{', '.join(f'{m:.3e}' for m in mse_list)}  "
        f"(μ={mean_mse:.3e}, σ={std_mse:.3e})"
    )
    
    # ────────── JSON summary 작성 ──────────
    summary = {
        "config":   vars(args),
        "float8_mse": {
            "e3m4": mse_f8_e3m4,
            "e4m3": mse_f8,
            "e5m2": mse_f8_e5m2,
        },
        "posit8_mse": mse_p8, # Posit8 MSE 추가
        "trials":   trial_metrics,
        "mse_mean": mean_mse,
        "mse_std":  std_mse,
        "summary_text": summary_text         # ← 추가
    }
    
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    
    # ────────── 화면에도 그대로 출력 ──────────
    print("\n──────── Experiment Summary ────────")

    print(' '.join(sys.argv) +" # " + str(date_dir))
    print(f"float16→float8_e3m4→float32 MSE : {mse_f8_e3m4:.3e}")
    print(f"float16→float8_e4m3→float32 MSE : {mse_f8:.3e}")
    print(f"float16→float8_e5m2→float32 MSE : {mse_f8_e5m2:.3e}")
    print(f"float16→posit→float32 MSE : {mse_posit:.3e}")
    print(f"{args.model.upper()} {16*args.vector_size}→{args.latent}→{args.vector_size} final MSEs : "
          f"{', '.join(f'{m:.3e}' for m in mse_list)}  (μ={mean_mse:.3e}, σ={std_mse:.3e})")

    print(f"Results saved to: {json_path}")



if __name__ == "__main__":
    main()
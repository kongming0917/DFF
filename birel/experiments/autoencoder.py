

#!/usr/bin/env python3
# diff_f16_autoenc_compare_updated.py
"""
Auto‑encoder comparison framework (MLP vs. ResidualLogicNet)

Updates (2025‑05‑07)
───────────────────
1.   Train/evaluate **one** model per run – selectable with `--model {mlp,logic}`.
2.   Pluggable LR schedulers via `--scheduler` and related hyper‑params.
3.   Temperature‑annealed `SigmoidHardSwitch` (tau schedule: linear / exp).
4.   Full experiment config & results are logged to JSON.
     • directory = today (YYYYMMDD)
     • filename  = summary_YYYYMMDD_HHMMSS.json
     • training curves: loss, LR, tau per epoch
5.   Repeated trials (`--trials`) with mean/σ of MSE stored in the same JSON.
6.   Every `eval_every` epochs (default 50) evaluate on val‑set and keep the
     best model.  Saved to `<exp_dir>/best_model.pt`, best MSE recorded.
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

# ────────────────────────────────────────────────────────────────
#  ❶  Original helpers & ResidualLogicNet import (unchanged)
#      * float8/float16 conversion utils
#      * ResidualLogicNet / GroupSum etc. assumed available
# ────────────────────────────────────────────────────────────────
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet  # noqa: F401
from birel.model import *   # noqa: F403, F401
from birel.utils         import *   # noqa: F403, F401
from birel.verilog       import *   # noqa: F403, F401

# (the long helper definitions from the original script follow unchanged →
#  float32_to_float8_e4m3, float8_e4m3_to_float32, f16_to_bits_torch, bits16_to_float32, etc.)

###############################################################
# … [helpers copied without modification — truncated for brevity]
###############################################################
# Paste the **exact** helper implementations from the user‑supplied script
# here.  They were omitted in this snippet to save space but are present in
# the full file stored in the canvas.
###############################################################
import numpy as np
import inspect, pprint



USEED = 4          # 2 ** (2 ** es)  (es = 1)
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
    idx_after_regime = run_len + 1                 # run + terminating bit
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
    lut = _get_posit8_lut()                     # (256,)
    # NaR(0x80) 은 비교 대상에서 제외
    valid_idx = np.array([i for i in range(256) if i != 0x80], dtype=np.uint8)
    valid_lut = lut[valid_idx]                 # (255,)

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
    x_i16 = x_f16.view(torch.int16)                   # (B,)

    # 2) 두 바이트 추출
    byte0 = (x_i16 & 0x00FF).to(torch.uint8)          # LSB
    byte1 = ((x_i16 >> 8) & 0x00FF).to(torch.uint8)   # MSB
    bytes2 = torch.stack([byte0, byte1], dim=1)       # (B, 2)

    # 3) 각 바이트를 비트로 전개  (B,2,8)
    bits = torch.stack([(bytes2 >> i) & 1
                        for i in range(8)], dim=-1)   # (B,2,8)

    # 4) 마지막 두 축(2,8) → 16 flatten → float32
    return bits.flatten(start_dim=1).float()          # (B,16)




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

    mant_bits = bits[..., 0:10]        # (…,10)
    exp_bits  = bits[..., 10:15]       # (…, 5)

    mant_int = (mant_bits * pow2_10).sum(dim=-1)      # Σ b_i·2^i
    exp_int  = (exp_bits  * pow2_5 ).sum(dim=-1)      # Σ b_i·2^i

    # ────────── (4) 부호 처리 ──────────
    sign_bit = bits[..., 15]                           # 0 또는 1
    sign     = 1.0 - 2.0 * sign_bit                   # 0→+1, 1→−1

    # ────────── (5) float16 값 재구성 ──────────
    bias = 15.0
    mantissa = mant_int / 1024.0                      # 2^10 로 나눔
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

def ulp_error(a, b):
    """
    Computes ULP (Unit in the Last Place) error between two float16 tensors.
    """
    # Make sure both tensors are float16
    a = a.to(torch.float16)
    b = b.to(torch.float16)

    # View as int16 to compare bit patterns (IEEE 754)
    a_bits = a.view(torch.int16)
    b_bits = b.view(torch.int16)

    # Compute absolute ULP difference
    return torch.abs(a_bits - b_bits).float()  # return as float for mean



def custom_loss(output, target, lambda_mse=1.0, lambda_cos=0.0, lambda_range=0.0, lambda_ulp = 0.0):
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



USEED = 4          # 2 ** (2 ** es)  (es = 1)

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
    idx_after_regime = run_len + 1                 # run + terminating bit
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
    lut = _get_posit8_lut()                     # (256,)
    # NaR(0x80) 은 비교 대상에서 제외
    valid_idx = np.array([i for i in range(256) if i != 0x80], dtype=np.uint8)
    valid_lut = lut[valid_idx]                 # (255,)

    # broadcasting: |x - lut| 최소값 인덱스
    diff = np.abs(x.astype(np.float32)[..., None] - valid_lut)   # (..., 255)
    nearest = diff.argmin(axis=-1)                               # (...)
    return valid_idx[nearest].astype(np.uint8)





# ────────────────────────────────────────────────────────────────
#  ❷  Modules with tau‑annealed sigmoid
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
#  ❸  Model builders (identical topology, new Sigmoid class used)
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

def build_mlp_autoenc(bitwidth: int, latent: int, enc_w: List[int], dec_w: List[int], device: str):
    enc_layers: List[nn.Module] = []
    in_d = bitwidth
    for h in enc_w:
        enc_layers += [MLPBlock(in_d, h)]
        in_d = h
    enc_layers += [nn.Linear(in_d, latent), SigmoidHardSwitch()]

    dec_layers: List[nn.Module] = []
    in_d = latent
    for h in dec_w:
        dec_layers += [MLPBlock(in_d, h)]
        in_d = h
    #dec_layers += [nn.Linear(in_d, bitwidth), SigmoidHardSwitch()]
    dec_layers += [nn.Linear(in_d, 1)]
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
                        initialization: str = 'normal'):

    #enc = ResidualLogicNet(
    #    device=device, n_in=bitwidth, n_out=latent,
    #    n_layers=n_layers, width=enc_w, k_history=k_history,
   #     logic_layer_ste=enc_logic_layer_ste, k_keep=0, voter_ste=False, noise_prob=encoder_noise_prob,
   #     implementation=implementation, connections=connections,  k_history_include_input=k_history_include_input)
    #enc = HybridEncoder(
    #    device=device, n_in=bitwidth, n_out=latent,
    #    width=enc_w, k_history=k_history,
    #    use_concat=True,
    #    implementation=implementation, connections=connections,  k_history_include_input=k_history_include_input)
    #dec = HybridDecoder(
    #    device=device, n_in=latent, n_out=1,
    #    width=dec_w, k_history=k_history,
    #    use_concat=True,
    #    implementation=implementation, connections=connections,  k_history_include_input=k_history_include_input)

    #logic_layers = []
    #llkw['connections'] = 'ste'
    #logic_layers.append(torch.nn.Flatten())
    #logic_layers.append(LogicBlock(n_in=bitwidth, n_out=latent, width=enc_w, implementation='cuda', k_history=1,
    #                                   logic_layer_ste=True, crossbar_ste=True, connections='ste'))
        #or _ in range(l - 1):
         #   logic_layers.append(LogicBlock(n_in=k, n_out=k, width=[k]*1, **llkw))
    enc = torch.nn.Sequential(
            LogicBlock(n_in=bitwidth, n_out=enc_w[0], width=[enc_w[0]], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections='ste', use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            LogicBlock(n_in=enc_w[0], n_out=enc_w[-2], width=enc_w[1:-1], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections='unique', use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            LogicBlock(n_in=enc_w[-2], n_out=latent, width=enc_w[-1], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections=last_connections, use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            #BitFlip(),
            #GroupSum(latent, tau=30), # group sum is positive 
            #nn.BatchNorm1d(latent), # we can normalize here but bitflip make it difficult
            #Binarize(latent)
            VotingLayer(latent, tau=group_sum_tau, use_ternary=use_ternary)
    )
    dec = torch.nn.Sequential(
            LogicBlock(n_in=latent, n_out=dec_w[0], width=[dec_w[0]], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections='ste', use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            LogicBlock(n_in=dec_w[0], n_out=dec_w[-2], width=dec_w[1:-1], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections='unique', use_crossbar_tree=True, initialization=initialization, mean = 0.0, std = 0.2),
            LogicBlock(n_in=dec_w[-2], n_out=dec_w[-1], width=dec_w[-1], implementation='cuda', k_history=1,
                            logic_layer_ste=True, crossbar_ste=True, connections=last_connections, use_crossbar_tree=False, initialization=initialization, mean = 0.0, std = 0.2),
            #BitFlip(dec_w[-1]),
            #GroupSum(1, tau=30),
            #nn.BatchNorm1d(1),
            RegressionLayer(1, use_ternary=use_ternary)
    )

    #dec = ResidualLogicNet(
    #    device=device, n_in=latent, n_out=1,
    #    n_layers=n_layers, width=dec_w, k_history=k_history,
    #    logic_layer_ste=dec_logic_layer_ste, k_keep=-4, voter_ste=False, noise_prob=decoder_noise_prob,
    #    implementation=implementation, connections=connections,  k_history_include_input=k_history_include_input)

    #dec = HybridDecoder(
    #    device=device, n_in=latent, n_out=1,
    #    width=dec_w, k_history=k_history,
    #    use_concat=True,
    #    implementation=implementation, connections=connections,  k_history_include_input=k_history_include_input)
    #fb1 = nn.Linear(16, 128)
    #fb3 = nn.ReLU()
    #fb4 = nn.Linear(128, 1)


    return nn.Sequential(enc, dec).to(device)

# ────────────────────────────────────────────────────────────────
#  ❹  Argument parser
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
    p.add_argument('--layers', type=int, default=3, help='# ResidualLogic layers')
    p.add_argument('--connections', type=str, default='ste', help='Connections')


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
#  ❺  Dataset utilities (unchanged)
# ────────────────────────────────────────────────────────────────
# make_dataset, f16_to_bits_torch, bits16_to_float32 … already defined above

# ────────────────────────────────────────────────────────────────
#  ❻  Training / evaluation helpers
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
                milestones = [args.warmup_epochs]         # 첫 번째 스케줄이 끝나는 시점
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
DEC_LAYERS_TO_MASK = {}             # dec.0


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
        layer_idx = after.split(".")[0]         # "0"
        return layer_idx in ENC_LAYERS_TO_MASK and "weights" in name

    # ---------- Decoder ----------
    if name.startswith(DEC_PREFIX):
        after = name[len(DEC_PREFIX):]
        layer_idx = after.split(".")[0]
        return layer_idx in DEC_LAYERS_TO_MASK and "weights" in name

    return False      # 그 외 파라미터는 마스크 X



def evaluate(net: nn.Module, bits: torch.Tensor, ref: np.ndarray, device: str) -> float:
    net.eval()
    with torch.no_grad():
        #rec = bits16_to_float32(net(bits.to(device))).cpu().numpy()
        rec = net(bits.to(device)).cpu().numpy()
        if len(rec.shape) == 2:
            rec = rec.squeeze(1)
    return float(np.mean((rec - ref) ** 2))



def train_single_trial(args, device, x_tr, x_te, bits_tr, bits_te, p8_tr, p8_te,
                       mse_f8: float, trial_id: int,
                       save_dir: Path, encoder_noise_prob: float, decoder_noise_prob: float) -> Dict:
    # build model
    if args.model == 'mlp':
        net = build_mlp_autoenc(16, args.latent, args.enc_width, args.dec_width, device)
    else:
        net = build_logic_autoenc(16, args.latent, args.enc_width, args.dec_width,
                                  args.layers, device, args.k_history,
                                  args.enc_logic_layer_ste, args.dec_logic_layer_ste, args.k_keep,
                                  encoder_noise_prob, decoder_noise_prob, args.implementation,
                                  args.connections, args.use_ternary, args.k_history_include_input,
                                  args.last_connections, args.group_sum_tau, args.initialization)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = create_scheduler(opt, args, args.epochs)

    #opt = torch.optim.AdamW(net.parameters(),
    #                    lr=0.05, betas=(0.9,0.99),
    #                    weight_decay=1e-4)

    loader = DataLoader(TensorDataset(bits_tr, torch.from_numpy(x_tr), torch.from_numpy(p8_tr)),
                        batch_size=args.batch, shuffle=True)
    lossf =  nn.MSELoss()

    # ------------------------------------------------------------------
    # A) 준비 단계
    # ------------------------------------------------------------------
    masks = {n: torch.ones_like(p, dtype=torch.bool)     # True = 학습 가능
             for n,p in net.named_parameters() if "weights" in n}
    
    check_freq   = args.pruning_epochs                # 20 epoch마다 점검
    freeze_after = int(args.epochs*0.3)
    epsilon      = 0.05               # |w|<ε 면 0 처리
    min_live_frac = 0.2               # 레이어별 최소 20 % 남김




    # tracking -------------------------------------------------------------
    hist_loss: List[float] = []
    hist_lr:   List[float] = []
    hist_tau:  List[float] = []
    hist_noise:  List[float] = []
    hist_val_mse:  List[float] = []
    #hist_loss_MSE: List[float] = []

    best_mse: float = math.inf
    best_state: Dict | None = None

    if args.log_gradnorm:
        get_last_norm = attach_grad2norm_logger(net)

    # 1) 초기 마스크
    '''
    zero_masks = {
        n: (p.abs() < epsilon) for n, p in net.named_parameters()
        if is_enc_param(n)
    }
    '''

    zero_masks = {}
    with torch.no_grad():
        for n, p in net.named_parameters():
            if is_enc_param(n):
                zero_masks[n] = (p.abs() < epsilon)
    
    
    last50_start = args.epochs - 49          # 예: epochs=1200 → 1151


    
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
        for xb, target, teacher in loader:
            xb, target, teacher = xb.to(device), target.to(device), teacher.to(device)

            # normalize input
            xb = xb 
            opt.zero_grad()
            #recon_bits = net(xb)
            #recon_val  = bits16_to_float32(recon_bits)
            recon_val = net(xb)
            recon_val = recon_val 
            if len(recon_val.shape) == 2:
                recon_val = recon_val.squeeze(1)
            mse_loss = lossf(recon_val, target)
            #kd_l = kd_loss(recon_val, teacher)
            #loss = (1- args.kd_weight) * mse_loss + args.kd_weight * kd_l
            loss = mse_loss
            #loss_plain = lossf_plain(recon_val, target)
            #print(torch.allclose(loss_plain, loss))  # True가 나와야 정상
            #loss = weighted_mse(recon_val, target, 3.0)
            #mse = nn.MSELoss(recon_val, target)
            loss.backward()
            
            # 2-a) grad mask (encoder만)
            for n, p in net.named_parameters():
                if n in zero_masks and p.grad is not None:
                    p.grad[zero_masks[n]] = 0.0

         
            
            if args.clip_grad is not None:
                utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad) 
            opt.step()
            
            
            
            # 2-b) 값 고정
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in zero_masks:
                        p.data[zero_masks[n]] = 0.0
        
            # 3) 주기적 업데이트 (encoder만)
            if ep >= freeze_after and ep % check_freq == 0:
                with torch.no_grad():
                    for n, p in net.named_parameters():
                        if n in zero_masks:
                            newly_zero = (p.abs() < epsilon) & zero_masks[n]  # 누적
                            zero_masks[n] |= newly_zero


                            
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
        if ep >= last50_start or ep % args.eval_every == 0:
        #if ep % args.eval_every == 0 or ep == args.epochs:
            mse_now = evaluate(net, bits_te, x_te, device)
            hist_val_mse.append(mse_now)
            if args.scheduler == 'plateau' and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mse_now)
            if mse_now < best_mse:
                best_mse = mse_now
                best_state = {
                    'epoch': ep,
                    'model_state_dict': net.state_dict(),
                    'mse': best_mse
                }
            print(f"[Trial {trial_id}] Epoch {ep:03d}/{args.epochs}  "
                  f"train‑MSE={epoch_loss:.3e}  val‑MSE={mse_now:.3e}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  tau={tau_now:.3f}  noise={noise_now:.3f}")
            if args.log_gradnorm:
                get_last_norm()
        else:
            hist_val_mse.append(0)

    # save best model -------------------------------------------------------
    model_path = save_dir / 'best_model.pt'
    if best_state is not None:
        torch.save(best_state, model_path)

    # final evaluation ------------------------------------------------------
    final_mse = evaluate(net, bits_te, x_te, device)

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
    (B,S,S) → (B·S, S)   ← attn
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
    return t           # dim≠3 → 그대로

def _load_activation_dataset(prefix: str,     # 모델 이름 or 파일 prefix
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
        for t in dct[layer][h]:                      # list[tensor]
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
def make_dataset(dataset: str                = "uniform",
                 n: int                      = 20_000,
                 seed: int                   = 0,
                 # Model-mode 추가 옵션 ↓ (synthetic 모드에선 무시)
                 data_type: str              = "key",
                 layer: int                  = 0,
                 head: Optional[int]         = None,
                 root_dir: str               = ".") -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    if dataset == 'uniform':
        return rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
    elif dataset == 'normal':

        return rng.normal(0, 1.0, size=(n,)).astype(np.float32)
    elif dataset == 'wikitext2_2048_opt-125m':
        return _load_activation_dataset(prefix=dataset,
                                    data_type=data_type,
                                    layer=layer,
                                    head=head,
                                    n=n,
                                    seed=seed,
                                    root_dir=root_dir,
                                    sample_mode= "random")
    elif dataset == 'wikitext2_2048_opt-350m':
        return _load_activation_dataset(prefix=dataset,
                                    data_type=data_type,
                                    layer=layer,
                                    head=head,
                                    n=n,
                                    seed=seed,
                                    root_dir=root_dir,
                                    sample_mode= "random")

        data = rng.normal(0, 1, size=(n,)).astype(np.float32)
        return data
    elif dataset == 'normal1p2':
        data = rng.normal(0, 1.2, size=(n,)).astype(np.float32)
        return data
    elif dataset == 'normal1p5':
        data = rng.normal(0, 1.5, size=(n,)).astype(np.float32)
        return data
    elif dataset == 'normal2':
        data = rng.normal(0, 2, size=(n,)).astype(np.float32)
        return data
    elif dataset == 'llama3':
        data = torch.load('/data/calibration_sets/llama-3-8B/calibration_dump/wikitext2_1024_llama-3-8b_key.pt')
        tensor_list = []
        for x in data[10][:3]:
            for j in range(1):
                tensor_list.append(x[j].reshape(-1))
        data = torch.concat(tensor_list).cpu().numpy().astype(np.float32)[:n]
        print(data.shape)
        return data

    else:
        raise ValueError(f"Unknown dataset {dataset}")

# ────────────────────────────────────────────────────────────────
#  ❼  Main entry
# ────────────────────────────────────────────────────────────────



def main():

    print(torch.cuda.memory_summary())    # 전체 요약
    print("allocated:", torch.cuda.memory_allocated())   # 실제 사용 중
    print("reserved: ", torch.cuda.memory_reserved())    # 캐시 예약 중
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
    x_all = make_dataset(dataset = args.dataset, n = Ntrain + Ntest, data_type=args.data_type, layer = args.layer, head = args.head, root_dir= args.root_dir)
    x_tr, x_te = x_all[:Ntrain], x_all[Ntrain:]
    
    bits_tr = f16_to_bits_torch(torch.from_numpy(x_tr).to(torch.float16))
    bits_te = f16_to_bits_torch(torch.from_numpy(x_te).to(torch.float16))

    # Posit-교사 출력 (float32) 한 번에 계산
    with torch.no_grad():
        p8_labels = posit8_to_float32(float32_to_posit8(x_all))  # shape = (N,)
    p8_tr, p8_te = p8_labels[:Ntrain], p8_labels[Ntrain:]


    # float8 baseline -----------------------------------------------------------
    f8   = float32_to_float8_e4m3(x_te)
    f8rec= float8_e4m3_to_float32(f8)
    mse_f8 = float(np.mean((f8rec - x_te) ** 2))

    f8_e3m4 = float32_to_float8_e3m4(x_te)
    f8_e3m4rec = float8_e3m4_to_float32(f8_e3m4)
    mse_f8_e3m4 = float(np.mean((f8_e3m4rec - x_te) ** 2))  

    f8_e5m2 = float32_to_float8_e5m2(x_te)
    f8_e5m2rec = float8_e5m2_to_float32(f8_e5m2)
    mse_f8_e5m2 = float(np.mean((f8_e5m2rec - x_te) ** 2))
    
    
    p8          = float32_to_posit8(x_te)                    # NEW
    p8_rec      = posit8_to_float32(p8)
    mse_p8      = float(np.mean((p8_rec - x_te) ** 2))

    

    posit = float32_to_posit8(x_te)
    positrec = posit8_to_float32(posit)
    mse_posit = float(np.mean((positrec - x_te) ** 2))  

    # repeat trials -------------------------------------------------------------
    trial_metrics: List[Dict] = []
    for t in range(1, args.trials + 1):
        trial_dir = date_dir / f"trial_{t:02d}"
        trial_dir.mkdir(exist_ok=True)
        metrics = train_single_trial(args, device, x_tr, x_te, bits_tr, bits_te, p8_tr, p8_te,
                                     mse_f8, t, trial_dir, args.encoder_noise_prob, args.decoder_noise_prob)
        trial_metrics.append(metrics)

    # aggregate -----------------------------------------------------------------
    mse_list = [m['best_mse'] for m in trial_metrics]
    mean_mse = float(np.mean(mse_list))
    std_mse  = float(np.std(mse_list))

    # JSON summary --------------------------------------------------------------
    cfg_str = f"{args.model}_enc_width{str(args.enc_width)}_dec_width{str(args.dec_width)}_lr{args.lr:g}".replace('.', 'p')
    json_path = date_dir / f"{cfg_str}.json"

    # ────────── Experiment Summary 문자열 빌드 ──────────
    summary_text = (
        f"float16→float8_e3m4→float32 MSE : {mse_f8_e3m4:.3e}\n"
        f"float16→float8_e4m3→float32 MSE : {mse_f8:.3e}\n"
        f"float16→float8_e5m2→float32 MSE : {mse_f8_e5m2:.3e}\n"
        f"float16→posit8→float32 MSE : {mse_p8:.3e}\n"
        f"{args.model.upper()} 16→{args.latent}→16 final MSEs : "
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
    print(f"{args.model.upper()} {16}→{args.latent}→16 final MSEs : "
          f"{', '.join(f'{m:.3e}' for m in mse_list)}  (μ={mean_mse:.3e}, σ={std_mse:.3e})")

    print(f"Results saved to: {json_path}")



if __name__ == "__main__":
    main()

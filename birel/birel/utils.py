
import numpy as np


#!/usr/bin/env python3
# corr_plots.py
import torch
import matplotlib
matplotlib.use('Agg') # pyplot을 임포트하기 전에 백엔드를 설정해야 합니다.
import matplotlib.pyplot as plt
import itertools
import os, torch, matplotlib.pyplot as plt
from pysat.formula import IDPool, CNF
from pysat.card import CardEnc
from pysat.solvers import Glucose4        # 어떤 SAT solver든 OK
import os, torch, matplotlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from ortools.sat.python import cp_model

import os

from difflogic import *
from difflogic.difflogic import *
# birel.pruning과 birel.model은 순환 import를 피하기 위해 함수 내부에서 lazy import
# birel.conv는 순환 import를 피하기 위해 함수 내부에서 lazy import
from collections import Counter
import torch.nn as nn


def save_corr_plot(A, B):
    #A = torch.randint(0, 2, (256, 60, 5), dtype=torch.float32)  # replace with your tensor
    #B = torch.randint(0, 2, (256, 5),  dtype=torch.float32)     # replace with your tensor

    os.makedirs("corr_plots", exist_ok=True)

# ─── 3.  Loop over the 5 channels ───
    print(A.shape)
    print(B.shape)
    for ch in range(A.shape[1]):
        b_vec   = B[:, ch]                # (256,)
        corr    = torch.empty(A.shape[2])         # store correlation values for this channel

        for feat in range(A.shape[2]):
            a_vec = A[:, ch, feat]        # (256,)
            # torch.corrcoef returns a 2×2 matrix; [0,1] (or [1,0]) is the coefficient
            corr[feat] = torch.corrcoef(torch.stack((a_vec, b_vec)))[0, 1]

        # ─── 4.  Plot ───
        plt.figure(figsize=(10, 4))
        plt.bar(range(A.shape[1]), corr.numpy())
        plt.title(f"Channel {ch}: correlation of A[:, i, {ch}] with B[:, {ch}]")
        plt.xlabel(f"Feature index i (0–{A.shape[2]-1})")
        plt.ylabel("Pearson r")
        plt.tight_layout()
        plt.savefig(f"corr_plots/corr_channel_{ch}.png", dpi=150)

def save_corr_plot(A: torch.Tensor,
                   B: torch.Tensor,
                   top_n: int = 5,
                   out_dir: str = "corr_plots") -> None:
    """
    A : (N, 5, 60)  — binary tensor
    B : (N, 5)      — binary tensor

    • 각 채널(ch = 0‥4)에 대해 Pearson r(A[:, ch, feat], B[:, ch]) 계산
    • 바-그래프 PNG 저장
    • 완전 상관( |r| == 1 ) feature 탐색
      · 없으면 |r|이 가장 큰 top-n feature를 다수결(median vote)로 결합해
        새 상관계수를 구하고 결과 출력
    """
    # ─── sanity check ───────────────────────────────────────────────
    assert A.ndim == 3 and A.shape[1] == 5 and A.shape[2] == 60, \
        f"expect A.shape == (N, 5, 60), got {A.shape}"
    assert B.shape == (A.shape[0], 5), \
        f"expect B.shape == (N, 5), got {B.shape}"
    assert 2 <= top_n <= 60

    N = A.shape[0]
    os.makedirs(out_dir, exist_ok=True)

    # ─── per-channel 작업 ───────────────────────────────────────────
    for ch in range(5):
        b_vec = B[:, ch]               # (N,)
        corr  = torch.empty(60)        # 60 features

        # 1) feature별 상관 계산
        for feat in range(60):
            a_vec = A[:, ch, feat]     # (N,)
            corr[feat] = torch.corrcoef(
                torch.stack((a_vec, b_vec))
            )[0, 1]

        # 2) 그래프 저장
        plt.figure(figsize=(10, 4))
        plt.bar(range(60), corr.numpy())
        plt.title(f"Channel {ch}: corr(A[:, {ch}, i], B[:, {ch}])")
        plt.xlabel("Feature index i (0–59)")
        plt.ylabel("Pearson r")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/corr_channel_{ch}.png", dpi=150)
        plt.close()

        # 3) 완전 상관 여부 확인
        perfect_idx = (corr.abs() == 1).nonzero(as_tuple=True)[0]

        if perfect_idx.numel():
            signs = ["+" if corr[i] > 0 else "−" for i in perfect_idx]
            print(f"✓ 채널 {ch}: feature {perfect_idx.tolist()} 가 "
                  f"완전 상관 (sign {signs})")
            continue

        # 4) top-n 다수결
        top_idx  = torch.topk(corr.abs(), top_n).indices
        votes    = A[:, ch, top_idx]                       # (N, top_n)
        majority = (votes.sum(dim=1) >= (top_n // 2 + 1)).float()

        r_vote = torch.corrcoef(torch.stack((majority, b_vec)))[0, 1]

        msg = (f"채널 {ch}: 단일 feature 완전 상관 없음 → "
               f"top-{top_n} {top_idx.tolist()} 다수결 corr = {r_vote:.3f}")
        if abs(r_vote) == 1.0:
            msg += "  (🎯 voting으로 완전 상관 달성!)"
        print(msg)

    print(f"모든 채널 처리 완료 — PNG는 “{out_dir}/” 폴더에 저장되었습니다.")

    plt.close()

def save_corr_plot_and_min_subset(A: torch.Tensor,
                                  B: torch.Tensor,
                                  max_k: int = 4,
                                  out_dir: str = "corr_plots"):
    """
    A : (N, 5, 60)  binary (float 0/1)
    B : (N, 5)      binary (float 0/1)

    1) 각 채널마다 feature-별 Pearson r 막대그래프 저장
    2) 최소 서브셋(majority-vote)으로 완전 상관이 되는지 탐색
       - 찾으면 어떤 feature들이며 r=±1 인지 콘솔에 출력
       - max_k 넘어서도 없으면 '미발견' 메시지
    """
    assert A.ndim == 3 and A.shape[1] == 5 and A.shape[2] == 60
    assert B.shape == (A.shape[0], 5)
    os.makedirs(out_dir, exist_ok=True)
    N = A.shape[0]

    for ch in range(5):
        b = B[:, ch]                       # (N,)
        corr = torch.empty(60)

        # ── 1. 개별 feature 상관계수 계산 & 그래프 ────────────────
        for i in range(60):
            corr[i] = torch.corrcoef(torch.stack((A[:, ch, i], b)))[0, 1]

        plt.figure(figsize=(10, 4))
        plt.bar(range(60), corr.numpy())
        plt.title(f"Channel {ch}: corr(A[:, {ch}, i], B[:, {ch}])")
        plt.xlabel("Feature index (0–59)");  plt.ylabel("Pearson r")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/corr_channel_{ch}.png", dpi=150);  plt.close()

        # ── 2. 최소 서브셋 탐색 ───────────────────────────────────
        # (1) 단일 feature?
        perfect_1 = (corr.abs() == 1).nonzero(as_tuple=True)[0]
        if perfect_1.numel():
            idx = perfect_1[0].item()
            sign = '+' if corr[idx] > 0 else '−'
            print(f"✓ 채널 {ch}: 1-feature 완전 상관  ⇒  [{idx}]  (sign {sign})")
            continue

        # (2) 다수결 서브셋 k = 2 … max_k
        found = False
        for k in range(2, max_k + 1):
            majority_th = k // 2 + 1        # ⌈k/2⌉
            # 빠른 계산을 위해 feature 벡터를 미리 스택
            feats = [A[:, ch, i] for i in range(60)]  # 리스트로 추출 → indexing 빠름
            # itertools.combinations 범위 경고: k 큰 경우 시간 ↑
            for combo in itertools.combinations(range(60), k):
                votes = torch.stack([feats[i] for i in combo], dim=1)  # (N,k)
                pred  = (votes.sum(dim=1) >= majority_th).float()      # majority

                if torch.equal(pred, b) or torch.equal(1 - pred, b):
                    sign = '+' if torch.equal(pred, b) else '−'
                    print(f"✓ 채널 {ch}: "
                          f"최소 {k}-feature 서브셋 {list(combo)}  (sign {sign})")
                    found = True
                    break   # break inner for-combo
            if found:
                break       # break for-k

        if not found:
            print(f"✗ 채널 {ch}: k ≤ {max_k} 까지 완전 상관 서브셋 없음")

    print(f"모든 채널 완료 — 그래프는 “{out_dir}/” 폴더.")


    print("✓ Finished — graphs saved in the “corr_plots” folder.")

#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────
def save_corr_plot_and_min_subset(A: torch.Tensor,
                                      B: torch.Tensor,
                                      out_dir: str = "corr_plots"):
    """
    A : (N, C, F)  binary float 0/1
    B : (N, C)     binary float 0/1
    """
    assert A.ndim == 3 and B.shape == (A.shape[0], A.shape[1])
    max_k = A.shape[2]

    N, C, F = A.shape
    os.makedirs(out_dir, exist_ok=True)

    info = []
    for ch in range(C):
        b_vec = B[:, ch]                       # (N,)
        corr  = torch.empty(F)

        # ── Pearson r 막대그래프 ───────────────────────────────
        for feat in range(F):
            corr[feat] = torch.corrcoef(torch.stack((A[:, ch, feat], b_vec)))[0, 1]

        plt.figure(figsize=(10, 4))
        plt.bar(range(F), corr.numpy())
        plt.title(f"Channel {ch}: corr(A[:, {ch}, i], B[:, {ch}])")
        plt.xlabel(f"Feature index (0–{F-1})")
        plt.ylabel("Pearson r")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/corr_channel_{ch}.png", dpi=150)
        plt.close()

        # ── SAT 기반 최소 서브셋 탐색 ───────────────────────────
        found, subset, sign = _sat_min_subset(ch, A[:, ch, :], b_vec, max_k=max_k)
        if found:
            sign_str = '+' if sign == 1 else '−'
            print(f"✓ 채널 {ch}: 최소 k={len(subset)} subset {subset}  (sign {sign_str})")
            info.append((sign, subset))
        else:
            print(f"✗ 채널 {ch}: k ≤ {max_k} 로 완전 상관 서브셋 없음")
            info.append(None)


    print(f"완료 — 그래프는 “{out_dir}/” 폴더에 저장되었습니다.")
    return info
# ──────────────────────────────────────────────────────────────────

#!/usr/bin/env python3
import os, torch, matplotlib
matplotlib.use("Agg")                         # 다중 프로세스에서 안전
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from ortools.sat.python import cp_model

# ──────────────────── 채널 하나 처리 ────────────────────
def _solve_one_channel(ch: int, A_ch: torch.Tensor, B_ch: torch.Tensor, out_dir: str):
    N, F = A_ch.shape

    # ── 1) Pearson r 그래프 ─────────────────────────────
    corr = torch.stack([
        torch.corrcoef(torch.stack((A_ch[:, i], B_ch)))[0, 1]
        for i in range(F)
    ])
    plt.figure(figsize=(10, 4))
    plt.bar(range(F), corr.numpy())
    plt.title(f"Channel {ch} – Pearson r")
    plt.xlabel("feature"); plt.ylabel("r")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"corr_channel_{ch}.png"), dpi=150)
    plt.close()

    # ── 2) ILP(=CP-SAT) 최소 subset ─────────────────────
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(F)]
    k = model.NewIntVar(0, F, "k");  model.Add(k == sum(x))
    t = model.NewIntVar(0, F, "t")
    model.Add(2 * t >= k );  model.Add(2 * t <= k+1)
    s = model.NewBoolVar("sign")        # 1=+, 0=−

    for n in range(N):
        pos_idx = [i for i in range(F) if A_ch[n, i] == 1]
        pos_sum = model.NewIntVar(0, F, f"sum{n}")
        model.Add(pos_sum == sum(x[i] for i in pos_idx))

        if B_ch[n] == 1:
            model.Add(pos_sum >= t).OnlyEnforceIf(s)
            model.Add(pos_sum <= t - 1).OnlyEnforceIf(s.Not())
        else:
            model.Add(pos_sum <= t - 1).OnlyEnforceIf(s)
            model.Add(pos_sum >= t).OnlyEnforceIf(s.Not())

    model.Minimize(k)
    solver = cp_model.CpSolver()
    #solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        subset = [i for i in range(F) if solver.Value(x[i])]
        sign   = +1 if solver.Value(s) else -1
        return ch, True, subset, sign
    else:
        return ch, False, [], 0

def _sat_min_subset(ch: int, A_ch: torch.Tensor, B_ch: torch.Tensor, max_k: int):
    """
    A_ch : (N, F)  binary float 0/1
    B_ch : (N,)    binary float 0/1
    """
    N, F = A_ch.shape
    idp   = IDPool()
    xvars = [idp.id(f"x{i}") for i in range(F)]      # F 개 변수

    for k in range(1, max_k + 1):
        t = k // 2 + 1                               # majority threshold
        base_cnf = CardEnc.equals(lits=xvars, bound=k, vpool=idp, encoding=1)

        for sign in (+1, -1):                        # +1: 직접, −1: 역상
            cnf = CNF(from_clauses=base_cnf.clauses)
            feasible = True

            for n in range(N):
                pos = [xvars[i] for i in range(F) if A_ch[n, i] == 1]

                if (B_ch[n] == 1 and sign == +1) or (B_ch[n] == 0 and sign == -1):
                    if len(pos) < t:          # 만족 불가
                        feasible = False
                        break
                    cnf.extend(CardEnc.atleast(pos, t, vpool=idp, encoding=1).clauses)
                else:
                    ub = t - 1
                    if ub < 0:
                        feasible = False
                        break
                    cnf.extend(CardEnc.atmost(pos, ub, vpool=idp, encoding=1).clauses)

            if not feasible:
                continue

            with Glucose4(bootstrap_with=cnf.clauses) as solver:
                if solver.solve():
                    model = solver.get_model()
                    subset = [i for i, var in enumerate(xvars) if model[var - 1] > 0]
                    return ch, True, subset, sign

    return ch, False, [], 0

def save_corr_plot_and_min_subset_parallel(A: torch.Tensor,
                                           B: torch.Tensor,
                                           out_dir: str = "corr_plots_ilp",
                                           max_workers: int | None = None):
    """
    A : (N, C, F) 0/1 float
    B : (N, C)    0/1 float
    """
    assert A.ndim == 3 and B.shape == (A.shape[0], A.shape[1])
    N, C, F = A.shape
    os.makedirs(out_dir, exist_ok=True)

    # 필요하다면 BLAS 스레드 수를 1로 줄여 과잉 오버섭스크립션 방지
    torch.set_num_threads(1)

    info = [None]*C
    with ThreadPoolExecutor(max_workers=C) as pool:
        futures = {
            pool.submit(_solve_one_channel, ch, A[:, ch, :], B[:, ch], out_dir): ch
            #pool.submit(_sat_min_subset, ch, A[:, ch, :], B[:, ch], A.shape[2]): ch
            for ch in range(C)
        }

        for fut in as_completed(futures):
            ch, ok, subset, sign = fut.result()
            info[ch] = (sign, subset)
            if ok:
                print(f"✓ 채널 {ch}: k={len(subset)}, subset={subset}, sign={'+' if sign==1 else '−'}")
            else:
                print(f"✗ 채널 {ch}: feasible subset 없음")

    print(f"완료 – 그래프는 “{out_dir}/” 폴더에 저장되었습니다.")
    return info

def attach_grad2norm_logger(model, verbose=True, eps=1e-12):
    """
    각 leaf-module의 output-gradient에 대해
    ‖∂L/∂activation‖₂ 를 계산해서 step별 dict에 저장한다.

    반환값
    -------
    get_last_norms() : callable → {layer_name: scalar-norm}
    """
    last_norms = {}

    def make_hook(name):
        def _hook(_, grad_in, grad_out):
            g = grad_out[0]                        # tuple → tensor
            # 안전하게 2-norm 계산
            norm = g.float().pow(2).sum().add(eps).sqrt().item()
            last_norms[name] = norm
        return _hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:           # leaf layer만
            m.register_full_backward_hook(make_hook(name))

    def get_last_norms():
        if verbose:
            print("── activation gradient 2-norms ──")
            for k, v in last_norms.items():
                print(f"{k:30s}: {v:.4e}")
            print("────────────────────────────────")
        return dict(last_norms)

    return get_last_norms
# ─────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  e4m3  ( 1 sign | 4 exp | 3 mant )
#  bias = 7,  max exp  =  7  (exp field 0xF → e=8, value ≈ 480)
#            min norm = -6
# ──────────────────────────────────────────────────────────────

def float32_to_float8_e4m3(x32: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 → uint8(e4m3)  • sub-normals=0, overflow=clamp max"""
    ui32  = x32.view(np.uint32)

    sign  = (ui32 >> 31) & 0x1
    exp32 = (ui32 >> 23) & 0xFF           # 0‥255
    man32 =  ui32        & 0x7FFFFF       # 23-bit mantissa

    # true exponent value (unbiased)
    e_val = exp32.astype(np.int32) - 127
    e8    = e_val + 7                     # re-bias for 4-bit field

    # ──────────── 범위 판정 ────────────
    sub   = e8 <= 0                       # underflow  → subnormal/zero
    over  = e8 >  15                      # overflow   → clamp (max norm)

    # mantissa 상위 3bit (round-to-nearest-even 대신 simple trunc.)
    man8  = (man32 >> 20) & 0x7

    # pack
    f8 = (sign << 7) | ((np.clip(e8, 0, 15) & 0xF) << 3) | man8

    # special cases
    f8[sub & (man32 == 0) & (exp32 == 0)] = sign[sub & (man32 == 0) &
                                                 (exp32 == 0)] << 7   # ±0
    f8[sub & ~((man32 == 0) & (exp32 == 0))] = (sign[sub] << 7) | man8[sub]  # sub-norm
    f8[over] = (sign[over] << 7) | (0xF << 3) | 0x7                         # clamp max

    return f8.astype(np.uint8)


def float8_e4m3_to_float32(f8: np.ndarray) -> np.ndarray:
    """uint8(e4m3) → IEEE-754 float32  (정규수·sub-normal·0 모두 지원)"""
    sign = (f8 >> 7) & 0x1
    exp  = (f8 >> 3) & 0xF
    man  =  f8       & 0x7

    sign_f = (-1.0) ** sign
    man_f  = man.astype(np.float32) / 8.0   # 3-bit → [0, 0.875]

    # 정규수 vs sub-normal
    norm   = exp != 0
    sub    = exp == 0

    # (norm)  value = ± (1 + m) * 2^(e-bias)
    val_norm = sign_f[norm] * (1.0 + man_f[norm]) * np.ldexp(
        1.0, exp[norm].astype(np.int32) - 7
    )

    # (sub)   value = ± (m) * 2^(1-bias)
    val_sub  = sign_f[sub]  * (man_f[sub])  * np.ldexp(1.0, -6)

    out = np.empty_like(f8, dtype=np.float32)
    out[norm] = val_norm
    out[sub ] = val_sub
    return out

def float32_to_float8_e5m2(x32: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 → uint8(e5m2)  • sub-normals=0, overflow=clamp max"""
    ui32  = x32.view(np.uint32)

    sign  = (ui32 >> 31) & 0x1
    exp32 = (ui32 >> 23) & 0xFF
    man32 =  ui32        & 0x7FFFFF

    e_val = exp32.astype(np.int32) - 127
    e8    = e_val + 15  # re-bias for 5-bit exp (bias=15)

    sub   = e8 <= 0
    over  = e8 >  31

    man8  = (man32 >> 21) & 0x3  # 2-bit mantissa

    f8 = (sign << 7) | ((np.clip(e8, 0, 31) & 0x1F) << 2) | man8

    f8[sub & (man32 == 0) & (exp32 == 0)] = sign[sub & (man32 == 0) & (exp32 == 0)] << 7
    f8[sub & ~((man32 == 0) & (exp32 == 0))] = (sign[sub] << 7) | man8[sub]
    f8[over] = (sign[over] << 7) | (0x1F << 2) | 0x3

    return f8.astype(np.uint8)


def float8_e5m2_to_float32(f8: np.ndarray) -> np.ndarray:
    """uint8(e5m2) → IEEE-754 float32"""
    sign = (f8 >> 7) & 0x1
    exp  = (f8 >> 2) & 0x1F
    man  =  f8       & 0x3

    sign_f = (-1.0) ** sign
    man_f  = man.astype(np.float32) / 4.0  # 2-bit mantissa

    norm = exp != 0
    sub  = exp == 0

    val_norm = sign_f[norm] * (1.0 + man_f[norm]) * np.ldexp(1.0, exp[norm].astype(np.int32) - 15)
    val_sub  = sign_f[sub]  * (man_f[sub])        * np.ldexp(1.0, -14)

    out = np.empty_like(f8, dtype=np.float32)
    out[norm] = val_norm
    out[sub ] = val_sub
    return out


def float32_to_float8_e3m4(x32: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 → uint8(e3m4)  • sub-normals=0, overflow=clamp max"""
    ui32  = x32.view(np.uint32)

    sign  = (ui32 >> 31) & 0x1
    exp32 = (ui32 >> 23) & 0xFF
    man32 =  ui32        & 0x7FFFFF

    e_val = exp32.astype(np.int32) - 127
    e8    = e_val + 3  # re-bias for 3-bit exp (bias=3)

    sub   = e8 <= 0
    over  = e8 >  7

    man8  = (man32 >> 19) & 0xF  # 4-bit mantissa

    f8 = (sign << 7) | ((np.clip(e8, 0, 7) & 0x7) << 4) | man8

    f8[sub & (man32 == 0) & (exp32 == 0)] = sign[sub & (man32 == 0) & (exp32 == 0)] << 7
    f8[sub & ~((man32 == 0) & (exp32 == 0))] = (sign[sub] << 7) | man8[sub]
    f8[over] = (sign[over] << 7) | (0x7 << 4) | 0xF

    return f8.astype(np.uint8)


def float8_e3m4_to_float32(f8: np.ndarray) -> np.ndarray:
    """uint8(e3m4) → IEEE-754 float32"""
    sign = (f8 >> 7) & 0x1
    exp  = (f8 >> 4) & 0x7
    man  =  f8       & 0xF

    sign_f = (-1.0) ** sign
    man_f  = man.astype(np.float32) / 16.0  # 4-bit mantissa

    norm = exp != 0
    sub  = exp == 0

    val_norm = sign_f[norm] * (1.0 + man_f[norm]) * np.ldexp(1.0, exp[norm].astype(np.int32) - 3)
    val_sub  = sign_f[sub]  * (man_f[sub])        * np.ldexp(1.0, -2)

    out = np.empty_like(f8, dtype=np.float32)
    out[norm] = val_norm
    out[sub ] = val_sub
    return out







######################################################################################################
#########################  모델 압축 관련 함수 #########################################################################################













######################################################################################################


### conv difflogic utils
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
    # Lazy import to avoid circular dependency
    from birel.conv import TreeConvLayer
    
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



####################################################################################
############################### anlayze funcitons #################################

def analyze_crossbar_connections(model: nn.Module):
    """
    [UPDATED]
    모델 내의 모든 Crossbar 계열 레이어(BlockEfficientCrossbarLayer 포함)를 찾아,
    각 입력 채널의 팬아웃(fan-out) 분포를 분석하고 출력합니다.
    """
    # Lazy import to avoid circular dependency
    from birel.model import CrossbarLayer, BlockEfficientCrossbarLayer
    
    print("\n--- Crossbar Connection Fan-out Analysis ---")
    
    found_crossbar = False
    for name, module in model.named_modules():
        # [수정] 모든 Crossbar 유형을 분석 대상으로 포함
        if isinstance(module, (CrossbarLayer, BlockEfficientCrossbarLayer)):
            found_crossbar = True
            print(f"\nAnalyzing [{name}] ({type(module).__name__}):")
            
            with torch.no_grad():
                connection_indices = None # 최종 절대 연결 인덱스를 저장할 변수

                # --- 각 Crossbar 유형에 맞게 '절대 입력 인덱스'를 찾는 로직 ---
                if isinstance(module, BlockEfficientCrossbarLayer):
                    # 1. 각 출력이 선택한 '상대' 인덱스를 찾음
                    relative_indices = module.weights.argmax(dim=-1) # (out_dim,)
                    
                    # 2. 각 출력이 속한 블록 그룹의 인덱스를 계산
                    output_indices = torch.arange(module.out_dim, device=module.weights.device)
                    block_indices = output_indices // module.out_per_block
                    
                    # 3. 절대 입력 인덱스 = (블록 시작점) + (블록 내 상대 인덱스)
                    connection_indices = block_indices * module.block_size + relative_indices

                else: # 기본 CrossbarLayer
                    connection_indices = module.weights.argmax(dim=-1)

                # 4. 이후 분석 로직은 모든 Crossbar 유형에 대해 동일
                fan_out_counts = torch.bincount(connection_indices, minlength=module.in_dim)
                usage_distribution = Counter(fan_out_counts.cpu().numpy())
                
                print("  - Input Channel Usage Distribution (Fan-out):")
                for num_connections, num_channels in sorted(usage_distribution.items()):
                    print(f"    - Connected to {num_connections} outputs: {num_channels} input channels")
                
                unused_channels = usage_distribution.get(0, 0)
                if module.in_dim > 0:
                    pruning_ratio = (unused_channels / module.in_dim) * 100
                    print(f"  - Summary: {unused_channels}/{module.in_dim} ({pruning_ratio:.2f}%) of input channels are completely unused.")

    if not found_crossbar:
        print("  - No Crossbar-like layers found in the model.")
        
    print("------------------------------------------")


def _analyze_logic_cascade(modules_in_block, start_mask, part, verbose):
    """
    [UPDATED VERSION]
    블록 내부를 분석하고, (최종 입력 마스크, 총 게이트 수, 살아있는 게이트 수, operation 통계)를 반환합니다.
    """
    # Lazy import to avoid circular dependency
    from birel.model import CrossbarLayer, BlockEfficientCrossbarLayer
    from birel.conv import ChannelMaskLayer
    
    alive_mask = start_mask
    total_gates_in_cascade = 0
    alive_gates_in_cascade = 0
    
    # Operation type 통계를 위한 카운터
    op_stats = {"tie": 0, "bypass": 0, "logic": 0}

    for name, m in reversed(list(modules_in_block)):
        if isinstance(m, LogicLayer):
            if m.out_dim != alive_mask.numel():
                if verbose:
                    print(f"  [!] Skipping logic layer {name} due to mask size mismatch ({m.out_dim} vs {alive_mask.numel()})")
                continue
            new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=alive_mask.device)
            alive_output_indices = torch.where(alive_mask)[0]
            op_indices = m.weights.argmax(-1)
            
            # 이 레이어의 operation 통계
            layer_op_stats = {"tie": 0, "bypass": 0, "logic": 0}
            
            for j in alive_output_indices:
                op_id = op_indices[j].item()
                
                # Operation type 분류
                if op_id in {0, 15}:  # tie operations
                    layer_op_stats["tie"] += 1
                elif op_id in {3, 5}:  # bypass operations  
                    layer_op_stats["bypass"] += 1
                else:  # logic operations
                    layer_op_stats["logic"] += 1
                
                # 입력 연결 설정 (기존 로직)
                if op_id not in {0, 5, 10, 15}: new_alive_mask[m.indices[0][j].item()] = True
                if op_id not in {0, 3, 12, 15}: new_alive_mask[m.indices[1][j].item()] = True
            
            # 전체 통계에 누적
            for key in op_stats:
                op_stats[key] += layer_op_stats[key]
            
            total_gates_in_cascade += m.out_dim
            alive_gates_in_cascade += alive_mask.sum().item()
            if verbose:
                print(f"  [{part.upper()}] {name:<45s}: Gates: {m.out_dim}, Alive: {alive_mask.sum().item()}, "
                      f"Tie: {layer_op_stats['tie']}, Bypass: {layer_op_stats['bypass']}, Logic: {layer_op_stats['logic']}")
            alive_mask = new_alive_mask.bool()

        elif isinstance(m, (CrossbarLayer, BlockEfficientCrossbarLayer)):
            if m.out_dim != alive_mask.numel():
                if verbose:
                    print(f"  [!] Skipping crossbar layer {name} due to mask size mismatch ({m.out_dim} vs {alive_mask.numel()})")
                continue
            new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=alive_mask.device)
            alive_output_indices = torch.where(alive_mask)[0]
            if alive_output_indices.numel() > 0:
                required_input_indices = None
                if isinstance(m, BlockEfficientCrossbarLayer):
                    connection_indices_relative = None
                    if m.connections == 'unique':
                        connection_indices_relative = m.connection_indices
                    else:
                        connection_indices_relative = m.weights.argmax(dim=-1)
                    chosen_relative_indices = connection_indices_relative[alive_output_indices]
                    block_indices = alive_output_indices // m.out_per_block
                    required_input_indices = block_indices * m.block_size + chosen_relative_indices
                else: # 기본 CrossbarLayer
                    connection_indices = m.weights.argmax(dim=-1)
                    required_input_indices = connection_indices[alive_output_indices]
                new_alive_mask[required_input_indices.unique()] = True

            alive_mask = new_alive_mask.bool()
            if verbose:
                print(f"  [{part.upper()}] {name:<45s}: Tracing liveness through {m.out_dim} selections...")
        
        elif isinstance(m, nn.Sequential):
            # Sequential로 감싸진 모듈 처리
            # 먼저 ChannelMaskLayer인지 확인
            if len(m) > 0 and isinstance(m[0], ChannelMaskLayer):
                # ChannelMaskLayer 처리
                channel_mask = m[0]
                if channel_mask.num_channels != alive_mask.numel():
                    if verbose:
                        print(f"  [!] Skipping ChannelMaskLayer {name} (wrapped in Sequential) due to mask size mismatch ({channel_mask.num_channels} vs {alive_mask.numel()})")
                    continue
                
                # mask_weights 확인
                mask_weights = channel_mask.mask_weights  # [out_dim, 3]
                mask_selection = mask_weights.argmax(-1)  # [out_dim], values: 0 (0-tie/pruned), 1 (1-tie/pruned), 2 (bypass/alive)
                
                # mask_selection에 따라 alive/dead 결정
                new_alive_mask = (mask_selection == 2)  # bypass인 채널만 alive
                alive_mask = alive_mask & new_alive_mask.to(alive_mask.device)
                
                # Pruned channel 통계 계산
                pruned_channels = ((mask_selection == 0) | (mask_selection == 1)).sum().item()
                kept_channels = (mask_selection == 2).sum().item()
                
                if verbose:
                    num_0tie = (mask_selection == 0).sum().item()
                    num_1tie = (mask_selection == 1).sum().item()
                    print(f"  [{part.upper()}] {name:<45s}: ChannelMaskLayer (wrapped in Sequential) - Pruned {pruned_channels}/{channel_mask.num_channels} channels (0-tie: {num_0tie}, 1-tie: {num_1tie}), Kept {kept_channels}/{channel_mask.num_channels} channels (bypass)")
            else:
                # Sequential로 감싸진 crossbar들을 순차적으로 처리
                # 역순으로 처리하여 출력부터 입력까지 역전파
                sequential_modules = list(m.named_children())
                for seq_name, seq_module in reversed(sequential_modules):
                    if isinstance(seq_module, (CrossbarLayer, BlockEfficientCrossbarLayer)):
                        if seq_module.out_dim != alive_mask.numel():
                            if verbose:
                                print(f"  [!] Skipping sequential crossbar {seq_name} due to mask size mismatch ({seq_module.out_dim} vs {alive_mask.numel()})")
                            continue
                        new_alive_mask = torch.zeros(seq_module.in_dim, dtype=torch.bool, device=alive_mask.device)
                        alive_output_indices = torch.where(alive_mask)[0]
                        if alive_output_indices.numel() > 0:
                            required_input_indices = None
                            if isinstance(seq_module, BlockEfficientCrossbarLayer):
                                connection_indices_relative = None
                                if seq_module.connections == 'unique':
                                    connection_indices_relative = seq_module.connection_indices
                                else:
                                    connection_indices_relative = seq_module.weights.argmax(dim=-1)
                                chosen_relative_indices = connection_indices_relative[alive_output_indices]
                                block_indices = alive_output_indices // seq_module.out_per_block
                                required_input_indices = block_indices * seq_module.block_size + chosen_relative_indices
                            else: # 기본 CrossbarLayer
                                connection_indices = seq_module.weights.argmax(dim=-1)
                                required_input_indices = connection_indices[alive_output_indices]
                            new_alive_mask[required_input_indices.unique()] = True
                        alive_mask = new_alive_mask.bool()
                        if verbose:
                            print(f"  [{part.upper()}] {name}.{seq_name:<45s}: Tracing liveness through {seq_module.out_dim} selections...")
        
        elif isinstance(m, ChannelMaskLayer):
            # ChannelMaskLayer 처리: mask_weights를 확인하여 bypass인 채널만 alive로 처리
            if m.num_channels != alive_mask.numel():
                if verbose:
                    print(f"  [!] Skipping ChannelMaskLayer {name} due to mask size mismatch ({m.num_channels} vs {alive_mask.numel()})")
                continue
            
            # mask_weights 확인 (옛날 버전 호환)
            if hasattr(m, 'mask_weights'):
                mask_weights = m.mask_weights  # [out_dim, 3]
            elif hasattr(m, 'mask_layer') and hasattr(m.mask_layer, 'mask_weights'):
                mask_weights = m.mask_layer.mask_weights  # 옛날 버전
            else:
                if verbose:
                    print(f"  [!] Skipping ChannelMaskLayer {name} - no mask_weights found")
                continue
            mask_selection = mask_weights.argmax(-1)  # [out_dim], values: 0 (0-tie/pruned), 1 (1-tie/pruned), 2 (bypass/alive)
            
            # mask_selection에 따라 alive/dead 결정
            # mask_selection == 0 (0-tie) 또는 mask_selection == 1 (1-tie): pruned (dead)
            # mask_selection == 2 (bypass): alive (kept)
            new_alive_mask = (mask_selection == 2)  # bypass인 채널만 alive
            
            # 기존 alive_channels_mask와 새로운 mask를 결합 (AND 연산)
            # MaskLayer는 출력 채널을 mask하므로, 출력 채널의 alive 상태만 업데이트
            alive_mask = alive_mask & new_alive_mask.to(alive_mask.device)
            
            # Pruned channel 통계 계산
            pruned_channels = ((mask_selection == 0) | (mask_selection == 1)).sum().item()  # 0-tie와 1-tie 모두 pruned
            kept_channels = (mask_selection == 2).sum().item()  # bypass만 kept
            
            if verbose:
                num_0tie = (mask_selection == 0).sum().item()
                num_1tie = (mask_selection == 1).sum().item()
                print(f"  [{part.upper()}] {name:<45s}: ChannelMaskLayer - Pruned {pruned_channels}/{m.num_channels} channels (0-tie: {num_0tie}, 1-tie: {num_1tie}), Kept {kept_channels}/{m.num_channels} channels (bypass)")
            
            # MaskLayer 자체는 연산량을 추가하지 않으므로 counters는 업데이트하지 않음

    return alive_mask, total_gates_in_cascade, alive_gates_in_cascade, op_stats

def finding_live_nodes_by_channel(model, in_channels, args, device="cuda", verbose: bool = False,
                                  compare_with_random: bool = False, random_live_nodes: int = None,
                                  fixed_classifier_outputs: int = None):
    """
    [MODIFIED FOR BINARY OPS COUNT]
    counters 딕셔너리('total', 'alive')를 그대로 사용하여,
    단순 노드 수가 아닌 총 바이너리 연산(Ops)을 누적하여 계산합니다.
    """
    # Lazy import to avoid circular dependency
    from birel.conv import ORPool2d, TreeConvLayer, Crossbar1x1Conv, ChannelShuffle, MaskLayer, ChannelMaskLayer
    from birel.model import CrossbarLayer, BlockEfficientCrossbarLayer
    
    # --- 1. Get intermediate shapes (기존과 동일) ---
    model.eval()
    module_shapes = {}
    hooks = []
    def get_shape_hook(name):
        def hook(module, input, output):
            inp = input[0] if isinstance(input, (tuple, list)) else input
            module_shapes[name] = {'input_shape': inp.shape, 'output_shape': output.shape}
        return hook
    
    for name, module in model.named_modules():
        # 존재하지 않는 클래스들 제거: PatchLogicBlock, EfficientTreeConv, CrossbarPatchLogicBlock, 
        # PruningBlock, RandomChannelPruning, RandomNodePruning, CrossbarPoolingConv
        if isinstance(module, (ORPool2d, nn.Flatten, TreeConvLayer,
                               CrossbarLayer, BlockEfficientCrossbarLayer, 
                               Crossbar1x1Conv, ChannelShuffle, MaskLayer, ChannelMaskLayer)):
            hooks.append(module.register_forward_hook(get_shape_hook(name)))
            
    dummy_input_shape = (1, in_channels, 28, 28) if 'mnist' in args.dataset else (1, in_channels, 32, 32)
    with torch.no_grad(): model(torch.randn(dummy_input_shape).to(device))
    for h in hooks: h.remove()

    # --- 2. Initialize dictionaries for results ---
    # 변수명은 유지하되, 이제부터 Ops를 저장합니다.
    counters = {
        'classifier_internal': {'total': 0, 'alive': 0},
        'classifier_input':    {'total': 0, 'alive': 0}, # 이 부분은 Ops 계산에 사용되지 않음
        'features':            {'total': 0, 'alive': 0}
    }
    weight_distributions = {} # 변경 없음
    
    # Operation type 통계를 위한 딕셔너리 추가
    node_counters = {
        'classifier_internal': {'total': 0, 'alive': 0},
        'classifier_input':    {'total': 0, 'alive': 0}, # 이 부분은 Ops 계산에 사용되지 않음
        'features':            {'total': 0, 'alive': 0}
    }


    # --- 3. Stage 1: Analyze Classifier ---
    if verbose: print("\n--- Stage 1: Analyzing Classifier Ops ---")
    # Classifier에서 LogicLayer와 crossbar 모두 포함
    classifier_logic_modules = list(model[1][:-1].named_children())
    
    final_layer = model[1][-1]
    
    # 마지막 LogicLayer 찾기 (crossbar 제외)
    last_logic_layer = None
    for module in reversed(model[1][:-1]):
        if isinstance(module, LogicLayer):
            last_logic_layer = module
            break
    
    if last_logic_layer is None:
        raise ValueError("No LogicLayer found in classifier")
    
    classifier_logic_output_dim = last_logic_layer.out_dim

    # Lazy import to avoid circular dependency
    from difflogic.difflogic import WeightedGroupSum, PrunedGroupSum
    
    if isinstance(final_layer, (WeightedGroupSum, PrunedGroupSum)):
        weights = final_layer.weight_raw.data.round()
        alive_mask = (weights.flatten() != 0)
    else:
        alive_mask = torch.ones(classifier_logic_output_dim, dtype=torch.bool, device=device)
        
    # Classifier의 Ops는 게이트 수와 동일합니다 (슬라이딩 윈도우 없음).
    classifier_input_alive_mask, total_gates, alive_gates, classifier_op_stats = _analyze_logic_cascade(
        classifier_logic_modules, alive_mask, 'classifier_internal', verbose
    )
    counters['classifier_internal']['total'] += total_gates
    counters['classifier_internal']['alive'] += alive_gates
    node_counters['classifier_internal']['total'] += total_gates
    node_counters['classifier_internal']['alive'] += alive_gates
    

    # --- 4. Stage 2: Analyze Features ---
    if verbose: print("\n--- Stage 2: Analyzing Feature Ops ---")
    flatten_layer_name = '1.0'
    feature_output_shape = module_shapes[flatten_layer_name]['input_shape']
    C, H, W = feature_output_shape[1], feature_output_shape[2], feature_output_shape[3]
    
    alive_channels_mask = classifier_input_alive_mask.view(C, H * W).any(dim=1)
    
    # 각 TreeConvLayer의 출력에서의 alive_channels_mask를 저장하기 위한 딕셔너리
    tree_conv_output_masks = {}
    
    feature_top_level_modules = list(model[0].named_children())
    for name, m in reversed(feature_top_level_modules):
        full_name = f"0.{name}"
        if not full_name in module_shapes: continue

        output_shape = module_shapes[full_name]['output_shape']
        input_shape = module_shapes[full_name]['input_shape']
        
        # 슬라이딩 윈도우 개수 (출력 피처맵 크기)
        num_windows = output_shape[2] * output_shape[3]

        # Crossbar1x1Conv의 crossbar가 Sequential인 경우 실제 출력 차원 확인
        actual_output_channels = output_shape[1]
        if isinstance(m, Crossbar1x1Conv) and isinstance(m.crossbar, nn.Sequential):
            # Sequential의 마지막 모듈의 출력 차원을 실제 출력으로 사용
            last_module = list(m.crossbar.children())[-1]
            if isinstance(last_module, (CrossbarLayer, BlockEfficientCrossbarLayer)):
                    actual_output_channels = last_module.out_dim
            elif hasattr(last_module, 'out_dim'):
                actual_output_channels = last_module.out_dim
        
        if alive_channels_mask.numel() != actual_output_channels:
            print(f"[ERROR] Mismatch at {full_name}: Mask size {alive_channels_mask.numel()} != Module output channels {actual_output_channels} (expected {output_shape[1]}). Skipping...")
            # 크기가 맞지 않으면 입력 채널 전체를 활성화 (안전한 fallback)
            alive_channels_mask = torch.ones(input_shape[1], dtype=torch.bool, device=device)
            continue
            
        # ❗ [핵심] 레이어 타입별 Ops 계산 및 누적 ❗
        if isinstance(m, ORPool2d):
            # [수정] m.out_channels 대신, 미리 구해놓은 output_shape에서 채널 수를 가져옵니다.
            num_output_channels = output_shape[1]
            
            # Total Ops = (출력 채널 수) * (윈도우 개수) * 3
            total_ops = num_output_channels * num_windows * 3
            # Alive Ops = (살아있는 출력 채널 수) * (윈도우 개수) * 3
            alive_ops = alive_channels_mask.sum().item() * num_windows * 3
            counters['features']['total'] += total_ops
            counters['features']['alive'] += alive_ops

            node_counters['features']['total'] += num_output_channels
            node_counters['features']['alive'] += alive_channels_mask.sum().item()
            pass

        elif isinstance(m, TreeConvLayer):
            # TreeConvLayer 처리 (PatchLogicBlock, CrossbarPatchLogicBlock, EfficientTreeConv는 더 이상 존재하지 않음)
            # TreeConvLayer의 출력에서의 alive_channels_mask를 저장 (지나기 전 상태)
            tree_conv_output_masks[full_name] = alive_channels_mask.clone()
            
            modules_to_analyze = m.cascade.named_children()
            in_ch, k_size = m.in_dim, m.kernel_size

            logic_input_mask, total_gates, alive_gates, feature_op_stats = _analyze_logic_cascade(
                modules_to_analyze, alive_channels_mask, 'features', verbose
            )
            
            counters['features']['total'] += total_gates * num_windows
            counters['features']['alive'] += alive_gates * num_windows
            node_counters['features']['total'] += total_gates
            node_counters['features']['alive'] += alive_gates

            # 기존의 liveness 마스크 업데이트 로직
            kernel_flat_dim = k_size ** 2
            if logic_input_mask.numel() != in_ch * kernel_flat_dim:
                raise ValueError(f"Shape mismatch in {full_name} logic... Expected {in_ch * kernel_flat_dim}, got {logic_input_mask.numel()}")
            alive_channels_mask = logic_input_mask.view(in_ch, kernel_flat_dim).any(dim=1)

        elif isinstance(m, Crossbar1x1Conv):
            
            # Liveness 마스크 추적
            # crossbar가 Sequential인 경우 (bottleneck 구조) 처리
            if isinstance(m.crossbar, nn.Sequential):
                # Sequential로 감싸진 경우: 역순으로 처리하여 최종 출력부터 입력까지 역전파
                sequential_modules = list(m.crossbar.named_children())
                current_mask = alive_channels_mask
                
                for seq_name, seq_module in reversed(sequential_modules):
                    if isinstance(seq_module, (CrossbarLayer, BlockEfficientCrossbarLayer)):
                        if seq_module.out_dim != current_mask.numel():
                            if verbose:
                                print(f"  [!] Skipping sequential crossbar {seq_name} in {full_name} due to mask size mismatch ({seq_module.out_dim} vs {current_mask.numel()})")
                            # 크기가 맞지 않으면 전체 입력을 활성화
                            current_mask = torch.ones(seq_module.in_dim, dtype=torch.bool, device=current_mask.device)
                        else:
                            new_alive_mask = torch.zeros(seq_module.in_dim, dtype=torch.bool, device=current_mask.device)
                            alive_output_indices = torch.where(current_mask)[0]
                            if alive_output_indices.numel() > 0:
                                required_input_indices = None
                                if isinstance(seq_module, BlockEfficientCrossbarLayer):
                                    connection_indices_relative = None
                                    if seq_module.connections == 'unique':
                                        connection_indices_relative = seq_module.connection_indices
                                    else:
                                        connection_indices_relative = seq_module.weights.argmax(dim=-1)
                                    chosen_relative_indices = connection_indices_relative[alive_output_indices]
                                    block_indices = alive_output_indices // seq_module.out_per_block
                                    required_input_indices = block_indices * seq_module.block_size + chosen_relative_indices
                                else: # 기본 CrossbarLayer
                                    connection_indices = seq_module.weights.argmax(dim=-1)
                                    required_input_indices = connection_indices[alive_output_indices]
                                new_alive_mask[required_input_indices.unique()] = True
                            current_mask = new_alive_mask.bool()
                            if verbose:
                                print(f"  [FEATURES] {full_name}.crossbar.{seq_name:<45s}: Tracing liveness through {seq_module.out_dim} selections...")
                alive_channels_mask = current_mask
            else:
                # 단일 crossbar인 경우 (기존 로직)
                internal_crossbar_modules = [('crossbar', m.crossbar)]
                new_alive_mask, _, _, _ = _analyze_logic_cascade(
                    internal_crossbar_modules, alive_channels_mask, 'features', verbose
                )
                alive_channels_mask = new_alive_mask
        elif isinstance(m, ChannelShuffle):
            if verbose: 
                print(f"[{'FEATURES'}] {full_name:<45s}: Tracing liveness through channel shuffle...")
            
            # 정방향 연산: view(B, G, C/G, H, W) -> transpose(1, 2) -> view(B, C, H, W)
            # 역방향 추적: view(G, C/G) -> transpose(0, 1) -> view(C)
            
            num_channels = alive_channels_mask.numel()
            groups = m.groups
            channels_per_group = num_channels // groups
            
            # 1. 마스크를 transpose된 후의 모양으로 재구성
            mask_transposed_shape = alive_channels_mask.view(channels_per_group, groups)
            
            # 2. transpose의 역연산 수행 (차원 0과 1을 다시 뒤집음)
            mask_original_shape = torch.transpose(mask_transposed_shape, 0, 1).contiguous()
            
            # 3. 원래의 1D 마스크 형태로 복원
            alive_channels_mask = mask_original_shape.view(-1)
            
            # ChannelShuffle 자체는 연산량을 추가하지 않으므로 counters는 업데이트하지 않음
            pass
        
        elif isinstance(m, ChannelMaskLayer):
            if verbose:
                print(f"[{'FEATURES'}] {full_name:<45s}: Processing ChannelMaskLayer...")
            
            # mask_weights 확인 (옛날 버전 호환)
            if hasattr(m, 'mask_weights'):
                mask_weights = m.mask_weights
            elif hasattr(m, 'mask_layer') and hasattr(m.mask_layer, 'mask_weights'):
                mask_weights = m.mask_layer.mask_weights  # 옛날 버전
            else:
                if verbose:
                    print(f"  [!] Skipping ChannelMaskLayer {full_name} - no mask_weights found")
                continue
            
            mask_selection = mask_weights.argmax(-1)  # [out_dim]
            
            # ChannelMaskLayer의 경우 num_channels 확인
            out_dim = m.num_channels
            
            if alive_channels_mask.numel() != out_dim:
                if verbose:
                    print(f"  [!] Skipping {layer_type} {full_name} due to mask size mismatch ({out_dim} vs {alive_channels_mask.numel()})")
                # 크기가 맞지 않으면 입력 채널 전체를 활성화 (안전한 fallback)
                alive_channels_mask = torch.ones(input_shape[1], dtype=torch.bool, device=device)
                continue
            
            # ChannelMaskLayer: 0=0-tie, 1=1-tie, 2=bypass
            bypass_idx = 2
            new_alive_mask = (mask_selection == bypass_idx)  # bypass인 채널만 alive
            pruned_channels = ((mask_selection == 0) | (mask_selection == 1)).sum().item()  # 0-tie와 1-tie 모두 pruned
            kept_channels = (mask_selection == bypass_idx).sum().item()  # bypass만 kept
            
            if verbose:
                num_0tie = (mask_selection == 0).sum().item()
                num_1tie = (mask_selection == 1).sum().item()
                print(f"  MaskLayer {full_name}: Pruned {pruned_channels}/{out_dim} channels (0-tie: {num_0tie}, 1-tie: {num_1tie}), Kept {kept_channels}/{out_dim} channels (bypass)")
            
            alive_channels_mask = alive_channels_mask & new_alive_mask.to(alive_channels_mask.device)
            
            # MaskLayer 자체는 연산량을 추가하지 않으므로 counters는 업데이트하지 않음
            pass

        # --- Ops가 없는 레이어들은 기존처럼 liveness 마스크만 추적 ---
        # PruningBlock, RandomChannelPruning은 더 이상 존재하지 않음

    # --- 5. Final results ---
    ops_results = {}
    gate_results = {}
    for part in counters:
        # 'classifier_input'은 Ops 계산이 아니므로 최종 요약에서 제외
        if part == 'classifier_input': continue

        if counters[part]['total'] > 0:
            total, alive = counters[part]['total'], counters[part]['alive']
            dead = total - alive
            ratio = (100 * dead / total) if total > 0 else 0.0
            ops_results[part] = {'total': total, 'alive': alive, 'dead': dead, 'dead_ratio': ratio}
        if node_counters[part]['total'] > 0:
            total, alive = node_counters[part]['total'], node_counters[part]['alive']
            dead = total - alive
            ratio = (100 * dead / total) if total > 0 else 0.0
            gate_results[part] = {'total': total, 'alive': alive, 'dead': dead, 'dead_ratio': ratio}

    # tree_conv_output_masks를 반환값에 추가
    return {'stats': ops_results, 'gate_results': gate_results, 'distributions': weight_distributions, 'tree_conv_output_masks': tree_conv_output_masks}


def summarize_and_print_analysis(analysis_bundle: dict):
    """
    [FINAL CORRECTED VERSION]
    - Fixes AttributeError by using the correct parameter name ('weight_raw').
    - Fixes the bug that caused the Overall Summary to be all zeros.
    - No longer requires the 'model' object as an argument.
    """
    # 1. liveness 통계 출력
    stats_results = analysis_bundle.get('stats', {})
    gate_results = analysis_bundle.get('gate_results', {})
    print("\n--- Pruning Analysis Results ---")
    
    parts_map = {
        'classifier_internal': 'Classifier - Internal Gates',
        'classifier_input':    'Classifier - Input Nodes (from Features)',
        'features':            'Features - Internal Gates'
    }


    # [버그 수정] 'stats' 딕셔너리 내부에서 값을 합산하도록 수정
    total_dead = sum(res.get('dead', 0) for key, res in stats_results.items() if key != 'classifier_input')
    total_nodes = sum(res.get('total', 0) for key, res in stats_results.items() if key != 'classifier_input')
    total_alive = sum(res.get('alive', 0) for key, res in stats_results.items() if key != 'classifier_input')
    
    overall_dead_ratio = (100 * total_dead / total_nodes) if total_nodes > 0 else 0.0

    print("\n------------------------------------")
    print("OPs Summary (OPs):")
    print(f"  - Total OPs: {total_nodes:,}")
    print(f"  - Alive OPs: {total_alive:,}")
    print(f"  - OPs reduction: {overall_dead_ratio:.2f}%")
    print("------------------------------------")
    
    parts_map = {
        'classifier_internal': 'Classifier - Internal Gates',
        'classifier_input':    'Classifier - Input Nodes (from Features)',
        'features':            'Features - Internal Gates'
    }


    total_dead = sum(res.get('dead', 0) for key, res in gate_results.items() if key != 'classifier_input')
    total_nodes = sum(res.get('total', 0) for key, res in gate_results.items() if key != 'classifier_input')
    total_alive = sum(res.get('alive', 0) for key, res in gate_results.items() if key != 'classifier_input')
    
    overall_dead_ratio = (100 * total_dead / total_nodes) if total_nodes > 0 else 0.0

    print("Gates Summary (Gates):")
    print(f"  - Total Gates: {total_nodes:,}")
    print(f"  - Alive Gates: {total_alive:,}")
    print(f"  - Gates reduction: {overall_dead_ratio:.2f}%")
    print("------------------------------------")
    


def summarize_concise_analysis(analysis_bundle: dict):
    stats = analysis_bundle.get('stats', {})
    # 전체 비율 계산
    total_dead = sum(res.get('dead', 0) for key, res in stats.items() if key != 'classifier_input')
    total_nodes = sum(res.get('total', 0) for key, res in stats.items() if key != 'classifier_input')
    overall_dead_ratio = (100 * total_dead / total_nodes) if total_nodes > 0 else 0.0
    # 파트별 비율 추출
    feat_dead_ratio = stats.get('features', {}).get('dead_ratio', 0.0)
    cl_int_dead_ratio = stats.get('classifier_internal', {}).get('dead_ratio', 0.0)
    
    print(f"\nFeatures Dead Ratio={feat_dead_ratio:.2f}%, Classifier Dead Ratio={cl_int_dead_ratio:.2f}%, Overall Dead Ratio={overall_dead_ratio:.2f}%")
    return overall_dead_ratio


def generate_feature_map_saliency(model, loader, loss_fn, target_layer_name, args, device="cuda", image_idx=0):
    """
    [FINAL CORRECTED VERSION]
    - Loss-based backpropagation: 단일 점수 대신 손실 함수에 대해 역전파를 수행하여
      Custom Kernel과의 호환성 및 그래디언트 안정성을 보장합니다.
    """
    print(f"\n--- Generating Saliency via Loss Backpropagation for layer '{target_layer_name}' ---")
    
    try:
        target_layer = dict(model.named_modules())[target_layer_name]
    except KeyError:
        print(f"[Error] Layer '{target_layer_name}' not found.")
        return

    saliency_from_hook = None
    def backward_hook(module, grad_input, grad_output):
        nonlocal saliency_from_hook
        saliency_from_hook = grad_output[0]

    handle = target_layer.register_full_backward_hook(backward_hook)

    # train() 모드로 설정하여 STE 경로를 타도록 보장
    original_mode = model.training
    model.train()
    
    # 분석할 단일 데이터 샘플 준비
    transformed_img, label = loader.dataset[image_idx]
    input_tensor = transformed_img.to(device).unsqueeze(0)
    label_tensor = torch.tensor([label], device=device) # label도 텐서로 변환

    with torch.set_grad_enabled(True):
        output = model(input_tensor)
        
        # ❗❗❗ [핵심 수정] ❗❗❗
        # 단일 점수(score) 대신 손실(loss)을 계산합니다.
        loss = loss_fn(output, label_tensor)
        
        model.zero_grad()
        
        # 손실에 대해 역전파를 수행합니다.
        loss.backward()
    
    # 분석 후 원래 모드로 복원
    model.train(original_mode)
    handle.remove()

    if saliency_from_hook is None:
        print(f"[Error] Failed to capture gradient via backward hook.")
        return

    saliency = saliency_from_hook.data.abs().squeeze(0).cpu()
    
    # (이후 시각화 코드는 동일합니다)
    # ... (이전과 동일한 시각화 코드) ...
    channel_saliency = torch.mean(saliency, dim=[1, 2])
    spatial_saliency, _ = torch.max(saliency, dim=0)
    
    fig = plt.figure(figsize=(18, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(range(len(channel_saliency)), channel_saliency)
    ax1.set_title(f"Channel-wise Saliency (Loss-based)\nLayer: {target_layer_name}")
    ax1.set_xlabel("Channel Index")
    ax1.set_ylabel("Average Gradient Magnitude")
    
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(spatial_saliency.numpy(), cmap='hot')
    ax2.set_title(f"Spatial Saliency (Max across channels)\nLayer: {target_layer_name}")
    ax2.axis('off')
    fig.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f"feature_saliency_loss_based_{target_layer_name.replace('.', '_')}.png")
    print(f"Saved feature_saliency_loss_based_{target_layer_name.replace('.', '_')}.png")
    plt.close()

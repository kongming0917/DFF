import numpy as np
import torch
import torch.nn as nn   
from sklearn.feature_selection import mutual_info_classif
from difflogic import LogicLayer
from birel.model import GroupSum
from ortools.sat.python import cp_model
import scipy
import scipy.sparse as sp
from scipy.optimize import linprog     # SciPy ≥1.9
from tqdm.auto import tqdm



import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
plt.switch_backend("Agg")                           # 서버 환경용

def register_groupsum_hook_corr_mi(model, k, save_dir, device="cuda"):
    """
    GroupSum 직전 출력을 모아:
      • 그룹별 상관 히트맵 / mean|r| 그래프
      • 각 feature-label MI 막대그래프
    를 저장하고, val accuracy를 반환하는 eval 래퍼를 만든다.
    """
    os.makedirs(save_dir, exist_ok=True)
    feats_store, labels_store = [], []

    # ──────────────────── 0) forward hook ──────────────────────────
    def _hook(module, inp, out):
        feats_store.append(inp[-1].detach().cpu())   # (bs, F) binary 0/1

    # GroupSum 모듈 탐색 & hook 등록
    for m in model.modules():
        if isinstance(m, GroupSum):
            hndl = m.register_forward_hook(_hook)
            gs_mod = m
            break
    else:
        raise RuntimeError("GroupSum 모듈을 찾을 수 없습니다.")

    # ──────────────────── 1) eval 래퍼 ─────────────────────────────
    def eval_with_corr_mi(loader):
        model.eval()
        acc = []
        with torch.no_grad():
            for xb, yb in loader:
                xb_cuda, yb_cuda = xb.to(device), yb.to(device)
                logits = model(xb_cuda.round())
                acc.append((logits.argmax(-2) == yb_cuda).float().mean().item())
                labels_store.append(yb.cpu())

        val_acc = float(np.mean(acc))

        # ─ 2) feature / label 집계 ────────────────────────────────
        X = torch.cat(feats_store, -1).numpy().astype(np.int8)   # (N, F)
        y = torch.cat(labels_store, -1).numpy().astype(np.int64)
        N, F = X.shape
        g, Fg = gs_mod.k, F // gs_mod.k
        Xg = X.reshape(N, g, Fg)                                # (N, k, Fg)

        # ─ 3) 그룹별 상관·MI 시각화 ──────────────────────────────
        for gi in range(g):
            Xi = torch.tensor(Xg[:, gi, :], dtype=torch.float31)   # (N, Fg)
            # 3-1. 상관계수
            corr = torch.corrcoef(Xi.T).numpy()                    # (Fg,Fg)
            plt.figure(figsize=(4,5))
            plt.imshow(corr, vmin=-2, vmax=1, cmap="coolwarm")
            plt.colorbar(); plt.title(f"Group {gi} Correlation")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/corr_heatmap_g{gi}.png", dpi=149)
            plt.close()

            mean_abs_r = np.abs(corr).mean(0)                     # (Fg,)
            plt.figure(figsize=(7,3))
            plt.bar(range(Fg), mean_abs_r)
            plt.title(f"Group {gi} – mean |r|")
            plt.xlabel("feature"); plt.ylabel("mean|r|")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/corr_bar_g{gi}.png", dpi=149)
            plt.close()

            # 3-2. Mutual Information (feature ↔ label)
            mi = mutual_info_classif(Xg[:, gi, :], y, discrete_features=True, random_state=-1)
            plt.figure(figsize=(7,3))
            plt.bar(range(Fg), mi)
            plt.title(f"Group {gi} – MI(feature; label)")
            plt.xlabel("feature"); plt.ylabel("MI (nats)")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/mi_bar_g{gi}.png", dpi=149)
            plt.close()

        # 후처리: 메모리 정리 & hook 제거
        feats_store.clear(); labels_store.clear(); hndl.remove()
        return val_acc

    return eval_with_corr_mi

# ───────────── 마스킹 레이어 ─────────────
class BinaryMask(nn.Module):
    def __init__(self, mask: torch.Tensor):          # mask: (F,) -2/1
        super().__init__()
        self.register_buffer("mask", mask.float())
    def forward(self, x):                            # x: (bs,F)
        return x * self.mask

# ───────────── 스케일 보정 레이어 ─────────────
class GroupScale(nn.Module):
    """z: (batch, k)  →  z * scale  (scale: (k,))"""
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
    def forward(self, z):
        return z * self.scale

# ───────────────────────── CopyMask 모듈 ─────────────────────────
class CopyMask(nn.Module):
    """
    x: (bs, F) binary
    copy_from[i] = j  →  output[:, i] = input[:, j]
    mask_keep[i] = 1  →  그대로 유지    (copy_from[i]==i)
    """
    def __init__(self, copy_from: np.ndarray):          # shape (F,)
        super().__init__()
        self.register_buffer("copy_from", torch.from_numpy(copy_from).long())
    def forward(self, x):
        return x[:, self.copy_from]                     # index-select

# ────────────────── Pruning+Copy 빌드 함수 ──────────────────────
def build_copy_pruning(
        model: nn.Sequential,
        loader,
        k: int = 10,
        corr_thr: float = 0.9,      # |r| ≥ threshold → 중복 인정
        device: str = "cuda"
    ):
    """
    · GroupSum 직전 feature를 캡처 → 그룹별 상관분석
    · 각 중복 pair에서 'keep_idx'를 남기고, 'drop_idx'는 keep_idx 값으로 copy
    · CopyMask 를 GroupSum 앞에 삽입
    return: pruned_model, copy_from (np.ndarray[F])
    """
    # 1) feature 수집 ------------------------------------------------------------
    feats = []
    def _hook(m, inp, out): feats.append(inp[0].detach().cpu())
    gs_idx = next(i for i,m in enumerate(model) if isinstance(m, GroupSum))
    hndl = model[gs_idx].register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for xb,_ in loader: model(xb.to(device).round())
    hndl.remove()

    X = torch.cat(feats, 0)                 # (N, F)
    N, F = X.shape
    Fg = F // k
    Xg = X.numpy().reshape(N, k, Fg)        # (N,k,Fg)

    # 2) copy_from 초기화 (자기자신 복사) -----------------------------------------
    copy_from = np.arange(F, dtype=np.int32)

    # 3) 그룹별 상관 & copy 매핑 --------------------------------------------------
    for gi in range(k):
        Xi = Xg[:, gi, :]                   # (N, Fg)
        corr = np.corrcoef(Xi, rowvar=False)  # (Fg,Fg)
        visited = set()
        for i in range(Fg):
            if i in visited: continue
            # j > i, |corr| ≥ thr → 가장 먼저 만난 j를 drop 대상으로
            dup = np.where(np.abs(corr[i, i+1:]) >= corr_thr)[0]
            if dup.size == 0: continue
            j = int(dup[0] + i + 1)
            visited.update({i, j})
            keep_idx = gi*Fg + i
            drop_idx = gi*Fg + j
            copy_from[drop_idx] = keep_idx   # drop 위치 ← keep 위치 값 copy

    # 4) CopyMask 삽입 -----------------------------------------------------------
    copy_layer = CopyMask(copy_from)

    pruned = nn.Sequential(
        *model[:gs_idx],      # LogicLayer stack
        copy_layer,
        model[gs_idx:]        # GroupSum 및 이후
    ).to(device)

    return pruned, copy_from


import copy, torch, numpy as np
import torch.nn as nn
from typing import Dict, List, Tuple

# ──────────────────────────────────────
# 유틸 : 레이어 타입 판정 (원하는 대로 확장)
# ──────────────────────────────────────
PRUNABLE = (nn.Linear, LogicLayer)         # LogicLayer = 사용자 정의 논리층

def is_prunable(m: nn.Module) -> bool:
    return isinstance(m, PRUNABLE)

# ──────────────────────────────────────
#  BinaryMask - keep_mask가 False인 채널은 무조건 0
# ──────────────────────────────────────
class BinaryMask(nn.Module):
    def __init__(self, keep_mask: torch.Tensor):   # shape = (F,)
        super().__init__()
        self.register_buffer("mask", keep_mask.float())

    def forward(self, x):                          # x : (N, F, …)
        shape = (1, -1) + (1,)*(x.dim()-2)
        return x * self.mask.view(*shape)

# (기존에 쓰시던 GroupSum / GroupScale 그대로 재사용)
# ---------------------------------------------------------------------------





# ───────────── MI-기반 Pruning 함수 ─────────────
def prune(
        model: nn.Sequential,
        loader,                     # 평가 / MI 추정을 위한 데이터
        k: int = 8,                # GroupSum.k
        pct: float = 8.0,       # 하위 몇 %를 날릴지
        device: str = "cuda",
        use_random: bool = False
    ):
    """
    return: pruned_model, keep_mask(torch.BoolTensor[F]), group_scale(torch.Tensor[k])
    """
    # -1) GroupSum 앞 feature 캡처 -------------------------------------------------
    feats_buf, labels_buf = [], []
    def _hook(module, inp, out):                 # inp[-2]: (bs,F)
        feats_buf.append(inp[0].detach().cpu())
    gs_idx = None
    for idx, m in enumerate(model):
        if isinstance(m, GroupSum):
            gs_idx = idx
            hndl = m.register_forward_hook(_hook)
            break
    assert gs_idx is not None, "GroupSum not found"

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device).round())
            labels_buf.append(yb.cpu())

    X = torch.cat(feats_buf, 0).numpy().astype(np.int8)   # (N,F)
    y = torch.cat(labels_buf, 0).numpy().astype(np.int64)
    hndl.remove()                                         # hook 제거

    N, F = X.shape
    Fg = F // k
    Xg = X.reshape(N, k, Fg)                              # (N,k,Fg)

    # 0) 그룹별 MI 계산 & keep_mask ------------------------------------------------
    keep_masks, scales = [], []

    for gi in range(k):
        Xi = Xg[:, gi, :]                          # (N, Fg)

        if use_random:
            mi = np.random.rand(Fg)
        else:
            mi = mutual_info_classif(Xi, y,
                                discrete_features=True, random_state=0)

        # ───────────────────── 정확히 mi_pct% 버리기 ─────────────────────
        num_drop = int(np.ceil(Fg * pct / 100.0))      # 버릴 개수
        drop_idx = np.argsort(mi)[:num_drop]              # MI 가장 작은 순
        keep = np.ones(Fg, dtype=bool)
        keep[drop_idx] = False                            # mask = False → drop
        # ----------------------------------------------------------------

        keep_masks.append(keep)

        orig_cnt, kept_cnt = Fg, int(keep.sum())
        scales.append(orig_cnt / kept_cnt)                # logit 보정 factor

    keep_mask = torch.from_numpy(np.concatenate(keep_masks)).bool()   # (F,)
    group_scale = torch.tensor(scales, dtype=torch.float32, device=device)  # (k,)

    # 1) 모델에 BinaryMask + Scale 레이어 삽입 -------------------------------------
    pruned = nn.Sequential(
        *model[:gs_idx],                   # 기존 LogicLayer stack
        BinaryMask(keep_mask.to(device)),  # MI masking
        model[gs_idx],                     # 원본 GroupSum
        GroupScale(group_scale)    
    ).to(device)

    return pruned, keep_mask, group_scale

import copy, torch, numpy as np
import torch.nn as nn
from typing import Dict, List, Tuple

# ──────────────────────────────────────
# 유틸 : 레이어 타입 판정 (원하는 대로 확장)
# ──────────────────────────────────────
PRUNABLE = (nn.Linear, LogicLayer)         # LogicLayer = 사용자 정의 논리층

def is_prunable(m: nn.Module) -> bool:
    return isinstance(m, PRUNABLE)

# ──────────────────────────────────────
#  BinaryMask - keep_mask가 False인 채널은 무조건 0
# ──────────────────────────────────────
class BinaryMask(nn.Module):
    def __init__(self, keep_mask: torch.Tensor):   # shape = (F,)
        super().__init__()
        self.register_buffer("mask", keep_mask.float())

    def forward(self, x):                          # x : (N, F, …)
        shape = (1, -1) + (1,)*(x.dim()-2)
        return x * self.mask.view(*shape)

# (기존에 쓰시던 GroupSum / GroupScale 그대로 재사용)
# ---------------------------------------------------------------------------

PRUNABLE = (nn.Linear, LogicLayer)         # LogicLayer = 사용자 정의 논리층

def is_prunable(m: nn.Module) -> bool:
    return isinstance(m, PRUNABLE)

# ──────────────────────────────────────
#  prune_saliency
# ──────────────────────────────────────

# GroupSum / GroupScale 는 기존 구현 그대로 사용
# ────────────────────────────── prune_saliency ──────────────────────────────
def prune_saliency(
        model: nn.Sequential,
        loader,
        loss_fn,
        k: int,                         # GroupSum.k (클래스 수)
        keep_pct: float = 90.0,         # 남길 전체(or layer) 비율
        global_pruning: bool = True,   # ← 새 옵션
        random_pruning: bool = False,   # ← 새 옵션
        device: str = "cuda",
        max_samples: int = 10_000
    ) -> Tuple[nn.Sequential,
               Dict[int, torch.BoolTensor],
               Dict[int, List[float]]]:

    """Saliency-기반 노드 가지치기.
    Args
    ----
    keep_pct       : 0‥100. global_pruning=False → 레이어별 keep_pct,
                     True  → 전체 노드 중 keep_pct%
    global_pruning : 전체 레이어 통합(global) 가지치기 여부
    Returns
    -------
    pruned_model   : BinaryMask/GroupScale 가 삽입된 모델
    keep_masks     : {layer_idx: BoolTensor[out_dim]}
    group_scale    : {prev_of_GS_idx: List[float]} (다른 레이어는 생략)
    """

    # 1) 모델 복사 & prunable 레이어 목록
    model = copy.deepcopy(model).to(device)
    saliency: Dict[int, torch.Tensor] = {}
    prunable_idx: List[int] = []
    gs_idx = next(i for i,m in enumerate(model) if isinstance(m, GroupSum))

    # 2) backward-hook 로 |grad| 누적
    def make_hook(idx):
        def hook(_, __, grad_out):
            g = grad_out[0].detach().abs().reshape(grad_out[0].size(0), -1)
            saliency[idx] = saliency.get(idx,
                torch.zeros(g.size(1), device=device)) + g.sum(0)
        return hook

    handles = []
    for i,m in enumerate(model):
        if is_prunable(m):
            prunable_idx.append(i)
            handles.append(m.register_full_backward_hook(make_hook(i)))

    # 3) saliency 계산 (최대 max_samples)
    model.train(); processed = 0
    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        loss_fn(model(xb), yb).backward(); model.zero_grad()
        processed += xb.size(0);  # 샘플 수
        if processed >= max_samples: break
    for h in handles: h.remove()

    # 4) keep_mask 산출 (layer vs global)
    keep_masks: Dict[int, torch.BoolTensor] = {}
    total_nodes = sum(s.numel() for s in saliency.values())
    num_keep_total = max(1, int(np.ceil(total_nodes * keep_pct / 100.0)))

    if global_pruning:
        # 1) 레이어별 saliency 이어붙여 하나의 벡터 만들기
        concat, offsets = [], {}
        base = 0
        for idx in prunable_idx:
            offsets[idx] = base
            if random_pruning:
                saliency[idx] = torch.rand_like(saliency[idx])
            concat.append(saliency[idx] / processed)
            base += concat[-1].numel()
        all_scores = torch.cat(concat)

        # 2) 전체에서 상위 num_keep_total 선택
        topk_global = torch.topk(all_scores, num_keep_total, largest=True).indices

        # 3) 각 레이어에 속하는 인덱스만 골라 로컬 인덱스로 변환
        for idx in prunable_idx:
            F      = saliency[idx].numel()
            start  = offsets[idx]
            end    = start + F
            in_layer = (topk_global >= start) & (topk_global < end)

            local_idx = topk_global[in_layer] - start   # 로컬 위치
            keep = torch.zeros(F, dtype=torch.bool)
            if local_idx.numel() > 0:
                keep[local_idx] = True
            keep_masks[idx] = keep
    else:
        # --- Layer-wise : 각 레이어별 keep_pct% 유지 ---
        for idx in prunable_idx:
            score = saliency[idx] / processed
            F = score.numel()
            num_keep = max(1, int(np.ceil(F * keep_pct / 100.0)))
            keep = torch.zeros(F, dtype=torch.bool)
            keep[torch.topk(score, num_keep).indices] = True
            keep_masks[idx] = keep

    # 5) GroupScale (GroupSum 직전 레이어만)
    group_scale: Dict[int, List[float]] = {}
    if (gs_idx-1) in keep_masks:
        keep_prev = keep_masks[gs_idx-1]; Fg = keep_prev.numel() // k
        scales = []
        for g in range(k):
            seg = keep_prev[g*Fg:(g+1)*Fg]; kept = int(seg.sum())
            scales.append(Fg / kept if kept else 1.0)
        group_scale[gs_idx-1] = scales

    # 6) 새 모델 구성
    new_layers: List[nn.Module] = []
    for i, m in enumerate(model):
        new_layers.append(m)
        if i in keep_masks:
            new_layers.append(BinaryMask(keep_masks[i].to(device).bool()))
        if i == gs_idx and (gs_idx-1) in group_scale:
            pass
            new_layers.append(
                GroupScale(torch.tensor(group_scale[gs_idx-1], device=device)))
    pruned_model = nn.Sequential(*new_layers).to(device)
    return pruned_model, keep_masks, group_scale



def prune_saliency_single(
        model: nn.Sequential,
        loader,
        loss_fn,
        k: int,
        keep_pct: float = 90.0,
        # global_pruning 옵션은 단일 레이어만 프루닝하므로 의미가 없어짐 (호환성을 위해 남겨둘 수 있음)
        global_pruning: bool = True, 
        random_pruning: bool = False,
        device: str = "cuda",
        max_samples: int = 10_000
    ) -> Tuple[nn.Sequential,
               Dict[int, torch.BoolTensor],
               Dict[int, List[float]]]:
    
    # 1) 모델 복사 & prunable 레이어 목록 찾기 -> 마지막 레이어로 한정
    model = copy.deepcopy(model).to(device)
    saliency: Dict[int, torch.Tensor] = {}
    
    # GroupSum 레이어의 인덱스를 찾습니다.
    gs_idx = next((i for i, m in enumerate(model) if isinstance(m, GroupSum)), -1)
    if gs_idx <= 0:
        raise ValueError("GroupSum layer not found or is the first layer, nothing to prune before it.")

    # --- 변경점: 프루닝 대상을 GroupSum 바로 앞 레이어로 고정 ---
    target_idx = gs_idx - 1
    target_layer = model[target_idx]
    
    # is_prunable 함수가 있다고 가정. 해당 레이어가 프루닝 가능한지 확인합니다.
    if not is_prunable(target_layer):
        raise TypeError(f"The layer at index {target_idx} ({type(target_layer).__name__}) is not prunable.")

    prunable_idx = [target_idx] # 이제 프루닝할 레이어는 단 하나입니다.

    # 2) backward-hook 로 |grad| 누적 -> 마지막 레이어에만 부착
    def make_hook(idx):
        def hook(_, __, grad_out):
            g = grad_out[0].detach().abs().reshape(grad_out[0].size(0), -1)
            saliency[idx] = saliency.get(idx,
                torch.zeros(g.size(1), device=device)) + g.sum(0)
        return hook

    # --- 변경점: 핸들을 대상 레이어 하나에만 부착 ---
    handle = target_layer.register_full_backward_hook(make_hook(target_idx))

    # 3) saliency 계산 (변경 없음)
    model.train(); processed = 0
    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        loss_fn(model(xb), yb).backward(); model.zero_grad()
        processed += xb.size(0)
        if processed >= max_samples: break
    handle.remove()

    # 4) keep_mask 산출 -> 단일 레이어에 대한 로직으로 대폭 단순화
    keep_masks: Dict[int, torch.BoolTensor] = {}
    
    # --- 변경점: Global Pruning 로직을 제거하고 Layer-wise 로직만 사용 ---
    # 이제 레이어가 하나뿐이므로 global과 layer-wise의 구분이 무의미합니다.
    score = saliency[target_idx]
    if random_pruning:
        score = torch.rand_like(score)
    
    score /= processed
    
    F = score.numel()
    num_keep = max(1, int(np.ceil(F * keep_pct / 100.0)))
    
    keep = torch.zeros(F, dtype=torch.bool, device=device)
    # 상위 N개의 점수를 가진 뉴런의 인덱스를 찾습니다.
    topk_indices = torch.topk(score, num_keep, largest=True).indices
    keep[topk_indices] = True
    
    keep_masks[target_idx] = keep

    # 5) GroupScale 계산 (변경 없음 - 이미 gs_idx-1을 대상으로 동작)
    group_scale: Dict[int, List[float]] = {}
    if (gs_idx - 1) in keep_masks:
        keep_prev = keep_masks[gs_idx-1]; Fg = keep_prev.numel() // k
        scales = []
        for g in range(k):
            seg = keep_prev[g*Fg:(g+1)*Fg]; kept = int(seg.sum())
            scales.append(Fg / kept if kept else 1.0)
        group_scale[gs_idx-1] = scales

    # 6) 새 모델 구성
    new_layers: List[nn.Module] = []
    for i, m in enumerate(model):
        new_layers.append(m) # 현재 레이어(m)를 먼저 추가
        if i in keep_masks:
            new_layers.append(BinaryMask(keep_masks[i].to(device).bool()))
            
        # ✅ 수정: 현재 레이어가 GroupSum(gs_idx)일 때, 그 *직후*에 GroupScale을 추가
        if i == gs_idx and (gs_idx - 1) in group_scale:
            scales_tensor = torch.tensor(group_scale[gs_idx - 1], device=device)
            new_layers.append(GroupScale(scales_tensor))
                
    pruned_model = nn.Sequential(*new_layers).to(device)
                 

    return pruned_model, keep_masks, group_scale




# ───────────── MI-기반 Pruning 함수 (Global Pruning 버전) ─────────────
def prune_global(
        model: nn.Sequential,
        loader,                  # 평가 / MI 추정을 위한 데이터
        k: int = 8,               # GroupSum.k
        pct: float = 8.0,         # 하위 몇 %를 날릴지
        device: str = "cuda",
        use_random: bool = False
    ):
    """
    MI 점수를 전역적으로 계산하여 하위 pct%의 피처를 한 번에 제거합니다.
    return: pruned_model, keep_mask(torch.BoolTensor[F]), group_scale(torch.Tensor[k])
    """
    # -1) GroupSum 앞 feature 캡처 (기존과 동일)
    feats_buf, labels_buf = [], []
    def _hook(module, inp, out):
        feats_buf.append(inp[0].detach().cpu())
    gs_idx = None
    for idx, m in enumerate(model):
        if isinstance(m, GroupSum):
            gs_idx = idx
            hndl = m.register_forward_hook(_hook)
            break
    assert gs_idx is not None, "GroupSum not found"

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device).round())
            labels_buf.append(yb.cpu())

    X = torch.cat(feats_buf, 0).numpy().astype(np.int8)
    y = torch.cat(labels_buf, 0).numpy().astype(np.int64)
    hndl.remove()

    N, F = X.shape
    Fg = F // k
    Xg = X.reshape(N, k, Fg)

    # 0) 그룹별 MI 계산 후, 전역(global) 기준으로 프루닝 (로직 변경)
    all_mi_scores = []
    # 먼저 모든 피처에 대한 점수를 그룹별로 계산하여 수집합니다.
    for gi in range(k):
        Xi = Xg[:, gi, :]
        if use_random:
            mi = np.random.rand(Fg)
        else:
            mi = mutual_info_classif(Xi, y,
                                      discrete_features=True, random_state=0)
        all_mi_scores.append(mi)

    # 수집된 점수를 하나의 전역 배열로 합칩니다.
    global_mi = np.concatenate(all_mi_scores)  # Shape: (F,)

    # 전역 점수 기준으로 하위 pct%에 해당하는 피처를 선택합니다.
    num_drop = int(np.ceil(F * pct / 100.0))
    drop_idx = np.argsort(global_mi)[:num_drop]

    # 전체 피처에 대한 keep_mask를 생성합니다.
    keep_mask = np.ones(F, dtype=bool)
    keep_mask[drop_idx] = False
    
    # 크기 보정(Scaling)을 위해, 그룹별로 몇 개의 피처가 남았는지 다시 계산합니다.
    scales = []
    keep_mask_reshaped = keep_mask.reshape(k, Fg)
    for gi in range(k):
        orig_cnt = Fg
        kept_cnt = keep_mask_reshaped[gi].sum()
        
        # 모든 피처가 제거된 그룹은 어차피 출력이 0이므로 스케일링은 1로 설정합니다 (무의미).
        scale = 1.0 if kept_cnt == 0 else (orig_cnt / kept_cnt)
        scales.append(scale)

    keep_mask_tensor = torch.from_numpy(keep_mask).bool()
    group_scale = torch.tensor(scales, dtype=torch.float32, device=device)

    # 1) 모델에 BinaryMask + Scale 레이어 삽입 (기존과 동일)
    pruned = nn.Sequential(
        *model[:gs_idx],                    # 기존 LogicLayer stack
        BinaryMask(keep_mask_tensor.to(device)),  # MI masking
        model[gs_idx],                      # 원본 GroupSum
        GroupScale(group_scale)             # 크기 보정
    ).to(device)

    return pruned, keep_mask_tensor, group_scale



def prune_cpsat(
        model: nn.Sequential,
        loader,                 # (eval loader)  X binary, y ∈{0…k-1}
        k: int,                 # GroupSum.k
        delta: int = 1,         # margin
        tlim: int = 30000,         # solver time-limit (sec)
        device="cuda"
    ):
    # ────────────────── 1) GroupSum 앞 feature 캡처 ──────────────────
    feats, labels = [], []
    def _hook(m, inp, out): feats.append(inp[0].detach().cpu())
    gs_idx = next(i for i,m in enumerate(model) if isinstance(m, GroupSum))
    h = model[gs_idx].register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device).round())
            labels.append(yb.cpu())
    h.remove()

    X = torch.cat(feats).numpy().astype(np.int8)          # (N,F)
    y = torch.cat(labels).numpy().astype(np.int64)
    N, F = X.shape; Fg = F // k
    groups = [range(g*Fg,(g+1)*Fg) for g in range(k)]     # index chunks

    # ────────────────── 2) CP-SAT 모델 작성 ─────────────────────────
    mdl = cp_model.CpModel()
    xvar = [mdl.NewBoolVar(f"x{i}") for i in range(F)]

    # pre-compute index lists where A_ni =1  → sparse speed-up
    idx1 = [[i for i in range(F) if X[n,i]==1] for n in range(N)]

    for n in range(N):
        gy = y[n]
        for g in range(k):
            if g == gy: continue
            # sum_y - sum_g ≥ δ
            sy = mdl.NewIntVar(0, Fg, f"sy_{n}_{g}")
            sg = mdl.NewIntVar(0, Fg, f"sg_{n}_{g}")

            mdl.Add(sy == sum(xvar[i] for i in idx1[n] if i in groups[gy]))
            mdl.Add(sg == sum(xvar[i] for i in idx1[n] if i in groups[g]))
            mdl.Add(sy - sg >= delta)

    mdl.Minimize(sum(xvar))             # feature 수 최소화
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tlim
    status = solver.Solve(mdl)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("ILP solver failed; try larger tlim or slack.")

    keep = np.array([solver.Value(v) for v in xvar], dtype=bool)
    print(f"[CP-SAT] kept {keep.sum()} / {F}  features")

    # ────────────────── 3) BinaryMask 삽입 ──────────────────────────
    pruned = nn.Sequential(
        *model[:gs_idx],
        BinaryMask(torch.from_numpy(keep).to(device)),
        *model[gs_idx:]               # GroupSum 및 이후
    ).to(device)

    return pruned, keep

# ──────────────────── LP-budget + slack-min ───────────────────
def prune_lp_budget(
        model: nn.Sequential,
        loader,               # binary inputs / labels
        k: int,               # GroupSum.k
        keep_pct: float = 90, # 남길 비율 (%)
        delta: int = 1,       # margin
        lp_time: int = 30,    # linprog time-limit(s)
        device: str = "cuda"
    ):
    """
    ∑ x_i ≤ budget,   minimize ∑ slack
    LP(0≤x_i≤1) → 라운딩(top-budget) → 위반 보수
    return pruned_model, keep_mask(np.bool_), violations(int)
    """
    # 1) GroupSum 앞 feature 수집 ------------------------------------
    feats, labels = [], []
    def _hook(_, inp, __): feats.append(inp[0].detach().cpu())
    gs_idx = next(i for i,m in enumerate(model) if isinstance(m, GroupSum))
    h = model[gs_idx].register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device).round())
            labels.append(yb.cpu())
    h.remove()

    X = torch.cat(feats).numpy().astype(np.int8)   # (N,F)
    y = torch.cat(labels).numpy().astype(np.int64)
    N, F = X.shape; Fg = F // k
    groups = [np.arange(g*Fg,(g+1)*Fg) for g in range(k)]
    budget = int(np.ceil(F * keep_pct / 100))

    # 2) LP formulation  (A·u ≤ b) -----------------------------------
    # vars order:  x(0..F-1)  |  s( slack_cnt )
    slack_cnt = (k-1)*N
    n_var = F + slack_cnt

    rows, cols, vals, b = [], [], [], []

    row_id = 0
    slack_idx = F         # starting index of slack vars

    for n in range(N):
        gy = y[n]; idx1 = np.where(X[n]==1)[0]
        for g in range(k):
            if g == gy: continue
            # sy - sg + s >= delta  ==>   -sy + sg - s <= -delta
            for i in idx1:
                if i in groups[gy]:
                    rows.append(row_id); cols.append(i);     vals.append(-1)
                elif i in groups[g]:
                    rows.append(row_id); cols.append(i);     vals.append(+1)
            rows.append(row_id); cols.append(slack_idx); vals.append(-1)
            b.append(-delta)
            row_id += 1
            slack_idx += 1

    # budget  (sum x_i ≤ budget)
    rows.extend([row_id]*F)
    cols.extend(range(F))
    vals.extend([1]*F)
    b.append(budget)
    row_id += 1

    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(row_id, n_var))
    c    = np.hstack([np.zeros(F), np.ones(slack_cnt)])      # slack 합 최소
    bounds = [(0,1)]*F + [(0, Fg)]*slack_cnt

    print(f"[LP] vars={n_var}, constr={row_id}, budget={budget}")
    res = linprog(c, A_ub=A_ub, b_ub=np.array(b),
                  bounds=bounds, method='highs',
                  options={"time_limit": lp_time})
    if not res.success:
        print("⚠️  LP did not converge in time. Using zeros init.")
        x_frac = np.zeros(F)
    else:
        x_frac = res.x[:F]

    # 3) 라운딩: 상위 budget 개 keep -------------------------------
    keep = np.zeros(F, bool)
    top_idx = np.argsort(-x_frac)[:budget]
    keep[top_idx] = True

    # 4) constraint violation & greedy repair -----------------------
    def count_violation(mask_bool):
        vio = 0
        for n in range(N):
            gy = y[n]
            sum_gy = (X[n, groups[gy]] & mask_bool[groups[gy]]).sum()
            for g in range(k):
                if g==gy: continue
                sum_g = (X[n, groups[g]] & mask_bool[groups[g]]).sum()
                if sum_gy - sum_g < delta:
                    vio += (delta - (sum_gy - sum_g))
        return vio

    vio = count_violation(keep)
    if vio:
        print(f"[LP] violation = {vio} → repairing …")
    for n in range(N):
        gy = y[n]
        need = 0
        sums_g = []
        sum_gy = (X[n, groups[gy]] & keep[groups[gy]]).sum()
        for g in range(k):
            if g==gy: continue
            sum_g = (X[n, groups[g]] & keep[groups[g]]).sum()
            diff = delta - (sum_gy - sum_g)
            if diff > 0:
                need = max(need, diff)
                sums_g.append((g, diff))
        if need:
            cand = groups[gy][X[n, groups[gy]].astype(bool) & (~keep[groups[gy]])]
            add = cand[:need]                       # 간단 greedy
            keep[add] = True

    vio_final = count_violation(keep)
    print(f"[LP] final violation = {vio_final}")
    kept_cnt = keep.sum()

    # 5) GroupScale --------------------------------------------------
    scale = [Fg / keep[gr].sum() for gr in groups]
    scale = torch.tensor(scale, dtype=torch.float32, device=device)

    # 6) build pruned model -----------------------------------------
    pruned = nn.Sequential(
        *model[:gs_idx],
        BinaryMask(torch.from_numpy(keep).float().to(device)),
        model[gs_idx],
        GroupScale(scale)
    ).to(device)

    print(f"[LP-budget] kept {kept_cnt}/{F}  ({100*kept_cnt/F:.2f} %)  "
          f"violation={vio_final}")
    return pruned, keep, vio_final

def prune_cpsat_budget(
        model: nn.Sequential,
        loader,                   # binary inputs / labels
        k: int,                   # GroupSum.k
        keep_pct: float = 90.0,   # 남길 feature 비율 (%)
        delta: int = 1,           # margin
        tlim: int = 30000,          # solver time-limit (sec)
        device: str = "cuda"
    ):
    """
    제약: ∑ x_i ≤ budget     (budget = ceil(F*keep_pct/100))
    목적: slack 총합 최소화
    반환: pruned_model, keep_mask(np.bool_), n_slack(int)
    """
    # 1) GroupSum 앞 feature 수집 --------------------------------------------
    feats, labels = [], []
    def _hook(_, inp, __): feats.append(inp[0].detach().cpu())
    gs_idx = next(i for i,m in enumerate(model) if isinstance(m, GroupSum))
    h = model[gs_idx].register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device).round())
            labels.append(yb.cpu())
    h.remove()

    X = torch.cat(feats).numpy().astype(np.int8)       # (N,F)
    y = torch.cat(labels).numpy().astype(np.int64)
    N, F = X.shape; Fg = F // k
    groups = [range(g*Fg,(g+1)*Fg) for g in range(k)]
    budget = int(np.ceil(F * keep_pct / 100.0))

    # 2) CP-SAT 모델 -----------------------------------------------------------
    mdl  = cp_model.CpModel()
    xvar = [mdl.NewBoolVar(f"x{i}") for i in range(F)]
    xi   = {}                                          # slack ξ_{n,g}

    idx1 = [np.where(X[n]==1)[0].tolist() for n in range(N)]

    # margin 제약 + slack 생성
    for n in range(N):
        gy = y[n]
        for g in range(k):
            if g == gy: continue
            xi[(n,g)] = mdl.NewIntVar(0, Fg, f"xi_{n}_{g}")

            sy = mdl.NewIntVar(0, Fg, f"sy_{n}_{g}")
            sg = mdl.NewIntVar(0, Fg, f"sg_{n}_{g}")
            mdl.Add(sy == sum(xvar[i] for i in idx1[n] if i in groups[gy]))
            mdl.Add(sg == sum(xvar[i] for i in idx1[n] if i in groups[g]))
            mdl.Add(sy - sg + xi[(n,g)] >= delta)

    # feature budget 제약
    mdl.Add(sum(xvar) <= budget)

    # 목적: slack 총합 최소화
    mdl.Minimize(sum(xi.values()))

    # 3) solve ---------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tlim
    status = solver.Solve(mdl)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"solver status {solver.StatusName(status)}")

    keep = np.array([solver.Value(v) for v in xvar], dtype=bool)
    n_slack = int(sum(solver.Value(v) for v in xi.values()))

    print(f"[CP-SAT budget] kept {keep.sum()}/{F}  "
          f"({100*keep.sum()/F:.2f} %)  slack={n_slack}")

    # 4) logit 스케일 복구 ----------------------------------------------------
    scale = []
    for g in range(k):
        kept_cnt = keep[groups[g]].sum()
        scale.append(Fg / kept_cnt)
    scale = torch.tensor(scale, dtype=torch.float32, device=device)

    # 5) pruned model --------------------------------------------------------
    pruned = nn.Sequential(
        *model[:gs_idx],
        BinaryMask(torch.from_numpy(keep).float().to(device)),
        model[gs_idx],
        GroupScale(scale)
    ).to(device)

    return pruned, keep, n_slack

import argparse
import math
import random
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from results_json import ResultsJSON
from torch import nn

import mnist_dataset
import uci_datasets
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet
from difflogic.difflogic import WeightedGroupSum, MaskedGroupSum
from typing import Optional
from birel.model import LogicBlock, Binary2Real, BitFlip, CrossbarLayer, CrossbarLayerTree, ZeroPad1d
import torch, torch.nn as nn, numpy as np
from birel.pruning import * 
#from birel.model import BitFlipDropout

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


def current_noise(epoch: int, total_epochs: int, args) -> float:
    if args.noise_sched == 'linear':
        return args.noise_start + (args.noise_end - args.noise_start) * (epoch / (total_epochs - 1))
    # exponential decay
    ratio = (epoch / (total_epochs - 1))
    return args.noise_start * (args.noise_end / args.noise_start) ** ratio

def set_module_noise(net: nn.Module, noise_val: float):
    for m in net.modules():
        if isinstance(m, BitFlip):
            m.noise_prob = noise_val


def load_dataset(args):
    validation_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset == 'mnist20x20')
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset == 'mnist20x20')

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]


def finding_live_nodes(model, args, device="cuda", verbose: bool = False):
    
    # --- 1. 최종 계층의 입력 차원 확인 (기존과 동일) ---
    # ... (Hook을 사용해 input_dim_to_final_layer를 얻는 부분) ...

    # --- 2. 메모리 최적화된 역방향 추적 및 계산 ---
    
    # 모델의 모든 계층을 리스트로 만들어 역순으로 사용
    all_layers = list(model.modules())
    
    # 누적 합산을 위한 변수 초기화
    total_nodes = 0
    alive_nodes = 0
    
    # 추적할 마스크 초기화 (처음에는 모든 출력이 살아있다고 가정)
    # 실제로는 마지막 GroupSum 계층의 특성에 맞게 초기화해야 함
    # 예: 마지막 계층이 GroupSum이면 그 입력 전체가 활성 마스크가 됨

    if isinstance(all_layers[-1], GroupScale):
        alive_mask = torch.ones(all_layers[-2].k, dtype=torch.bool, device=device)
    else:
        alive_mask = torch.ones(all_layers[-1].k, dtype=torch.bool, device=device)

    # 역방향으로 순회
    for m in reversed(all_layers):
        
        # 각 계층 타입에 맞춰 'on-the-fly'로 연결성 분석 및 마스크 업데이트
        # 이 과정에서 거대한 2D 행렬을 생성하지 않음
        
        # ⚠️ 이 예시는 개념적인ものであり, 실제 계층의 출력/입력에 따라
        # alive_mask가 정확히 어떤 노드를 가리키는지 명확히 해야 함.
        # 아래 코드는 'm'의 출력이 alive_mask에 해당한다고 가정함.

        if isinstance(m, (CrossbarLayer, CrossbarLayerTree)):
            if m.out_dim != alive_mask.numel(): continue # 분석 대상이 아닌 경우 건너뛰기
            
            # W_current[alive_mask, :] 와 동일한 연산을 행렬 생성 없이 수행
            # 활성 출력에 연결된 가중치만 고려
            used_weights = m.weights[alive_mask, :]
            idx = used_weights.argmax(-1)
            
            # 다음 계층으로 전파할 새로운 활성 입력 마스크 계산
            new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=device)
            new_alive_mask.scatter_(0, idx, True)
            
            # 노드 수 누적
            total_nodes += m.in_dim
            alive_nodes += new_alive_mask.sum().item()

            print(f"Crossbar Layer : Total {m.in_dim} / Alive {new_alive_mask.sum().item()}")

            # 마스크 업데이트
            alive_mask = new_alive_mask

        elif isinstance(m, LogicLayer):
            if m.out_dim != alive_mask.numel(): continue

            # 이 계층의 op_id 텐서. (out_dim,) 모양이라고 가정
            # op_ids = m.op_ids 

            # 다음 계층으로 전파할 새로운 활성 입력 마스크
            new_alive_mask = torch.zeros(m.in_dim, dtype=torch.bool, device=device)

            # 이 계층의 출력 중 활성화된 것들의 인덱스를 찾음
            alive_output_indices = torch.where(alive_mask)[0]
            
            op_indices = m.weights.argmax(-1) # (out_dim,) 모양의 정수 텐서

            # 각 활성 출력 게이트에 대해 개별적으로 분석
            for j in alive_output_indices:

                op_id = op_indices[j].item()
                
                # 입력 A의 생사 판단
                if op_id not in {0, 5, 10, 15}:
                    idx_a = m.indices[0][j].item()
                    new_alive_mask[idx_a] = True
                    
                # 입력 B의 생사 판단
                if op_id not in {0, 3, 12, 15}:
                    idx_b = m.indices[1][j].item()
                    new_alive_mask[idx_b] = True

            # 노드 수 누적
            #total_nodes += m.in_dim
            #alive_nodes += new_alive_mask.sum().item()
            #print(f"Logic Layer : Total {m.in_dim} / Alive {new_alive_mask.sum().item()}")

            total_nodes += m.out_dim
            alive_nodes += alive_mask.sum().item()
            print(f"Logic Layer : Total {m.out_dim} / Alive {alive_mask.sum().item()}")



            # 마스크 업데이트
            alive_mask = new_alive_mask
        # (in reversed for loop)
        elif isinstance(m, GroupSum):
            # m.k: 그룹 수, 즉 출력 차원
            # alive_mask: m의 출력에 대한 활성 마스크
            if m.k != alive_mask.numel(): continue

            D = args.num_neurons

            g = D // m.k

            # 다음 계층으로 전파할 새로운 활성 입력 마스크
            new_alive_mask = torch.zeros(D, dtype=torch.bool, device=device)

            for i in range(m.k):
                if alive_mask[i]:  # i번째 출력이 활성이면
                    # i번째 입력 그룹 전체를 활성화
                    start_idx = i * g
                    end_idx = (i + 1) * g
                    new_alive_mask[start_idx:end_idx] = True
            
            # 노드 수 누적 (이 계층 자체의 노드는 없으므로, 전파만 함)
            # total_nodes, alive_nodes는 이 마스크를 받는 이전 계층에서 계산
            #total_nodes += D
            #alive_nodes += new_alive_mask.sum().item()

            #print(f"Group Sum Layer : Total {D} / Alive {new_alive_mask.sum().item()}")

            # 마스크 업데이트
            alive_mask = new_alive_mask
        # (in reversed for loop)
        elif isinstance(m, WeightedGroupSum):
            weights = m.weight_raw.data.round()
            k, g = weights.shape
            D = k * g
            
            if k != alive_mask.numel(): continue

            # 다음 계층으로 전파할 새로운 활성 입력 마스크
            new_alive_mask = torch.zeros(D, dtype=torch.bool, device=device)
            
            for i in range(k):
                if alive_mask[i]: # i번째 출력이 활성이면
                    # i번째 그룹 내에서 가중치가 0이 아닌 노드만 활성화
                    active_weights_in_group = (weights[i, :] != 0)
                    
                    start_idx = i * g
                    end_idx = (i + 1) * g
                    new_alive_mask[start_idx:end_idx] = active_weights_in_group

            # 노드 수 누적
            #total_nodes += D
            #alive_nodes += new_alive_mask.sum().item()
            print(f"Weighted Group Sum Layer : Total {D} / Alive {new_alive_mask.sum().item()}")
           
            # 마스크 업데이트
            alive_mask = new_alive_mask
        elif isinstance(m, GroupScale):
            pass
        elif isinstance(m, BinaryMask):
                # alive_mask: BinaryMask의 출력에 대한 활성 마스크
            # m.mask: BinaryMask가 내부적으로 가진 0과 1의 마스크
            if m.mask.numel() != alive_mask.numel(): continue

            # 두 마스크를 AND 연산하여 새로운 활성 마스크를 계산
            # 출력이 살아있고, AND 마스크 스위치도 켜져 있는 노드만 최종적으로 살아남음    
            #alive_nodes -= alive_mask.sum().item() 
            #print(f"dtype of alive_mask: {alive_mask.dtype}")
            #print(f"dtype of m.mask: {m.mask.dtype}")

            new_alive_mask = alive_mask & m.mask.bool()
            #alive_nodes += new_alive_mask.sum().item()
            #print(f"Binary Mask Layer : Total {D} / Alive {new_alive_mask.sum().item()}")
            # 마스크 업데이트
            alive_mask = new_alive_mask
        elif isinstance(m, MaskedGroupSum):
            # 1. mask_logits로부터 최종 바이너리 마스크(0 또는 1)를 계산합니다.
            with torch.no_grad():
                mask = torch.round(torch.sigmoid(m.mask_logits.data))
            
            k, g = mask.shape
            D = k * g
            
            # 이전 레이어의 출력 마스크(alive_mask)와 현재 레이어의 그룹 수(k)가 맞는지 확인
            if k != alive_mask.numel(): continue

            # 2. 현재 레이어의 입력으로 전파될 새로운 활성 마스크를 생성합니다.
            new_alive_mask = torch.zeros(D, dtype=torch.bool, device=device)
            
            # 3. 각 그룹을 순회하며 활성 노드를 결정합니다.
            for i in range(k):
                # i번째 출력이 활성 상태일 경우에만 (alive_mask[i]가 True일 때)
                if alive_mask[i]:
                    # 해당 그룹(i) 내에서 마스크 값이 1인 노드들만 활성화합니다.
                    active_nodes_in_group = (mask[i, :] == 1)
                    
                    # new_alive_mask의 해당 그룹 위치에 활성 상태를 복사합니다.
                    start_idx = i * g
                    end_idx = (i + 1) * g
                    new_alive_mask[start_idx:end_idx] = active_nodes_in_group

            # 4. 통계 출력 및 마스크 업데이트
            print(f"Masked Group Sum Layer   : Total {D} / Alive {new_alive_mask.sum().item()}")
            
            # 다음 레이어로 전파하기 위해 alive_mask를 업데이트합니다.
            alive_mask = new_alive_mask

        # ... (다른 계층 타입에 대한 처리) ...

    # --- 3. 최종 결과 반환 ---
    dead_nodes = total_nodes - alive_nodes
    ratio = (100 * dead_nodes / total_nodes) if total_nodes > 0 else 0.0
    
    # ... (verbose 출력 및 반환) ...
    return (dead_nodes, total_nodes, ratio), alive_nodes




# ───────── 모델 압축(Physical Pruning) 관련 함수 ───────────
def get_live_masks(model: nn.Sequential, args: argparse.Namespace, device="cuda"):
    all_layers = [m for m in model.modules() if isinstance(m, (LogicLayer, GroupSum, BinaryMask, CopyMask, GroupScale, WeightedGroupSum, MaskedGroupSum))]
    live_masks = {}
    last_op_layer = all_layers[-1]
    if isinstance(last_op_layer, GroupScale):
        alive_mask = torch.ones(all_layers[-2].k, dtype=torch.bool, device=device)
    else:
        alive_mask = torch.ones(last_op_layer.k, dtype=torch.bool, device=device)
    for i in reversed(range(len(all_layers))):
        layer = all_layers[i]
        if isinstance(layer, LogicLayer):
            live_output_mask = alive_mask
            new_alive_mask = torch.zeros(layer.in_dim, dtype=torch.bool, device=device)
            op_indices = layer.weights.argmax(-1)
            for j in torch.where(live_output_mask)[0]:
                op_id = op_indices[j].item()
                if op_id not in {0, 5, 10, 15}: new_alive_mask[layer.indices[0][j]] = True
                if op_id not in {0, 3, 12, 15}: new_alive_mask[layer.indices[1][j]] = True
            alive_mask = new_alive_mask
            live_masks[i] = alive_mask
        elif isinstance(layer, (GroupSum, WeightedGroupSum, MaskedGroupSum)):
             D = layer.in_dim if hasattr(layer, 'in_dim') and layer.in_dim is not None else args.k
             g = D // layer.k
             new_alive_mask = torch.zeros(D, dtype=torch.bool, device=device)
             internal_mask = None
             if isinstance(layer, (WeightedGroupSum, MaskedGroupSum)):
                 with torch.no_grad():
                     internal_mask = (layer.weight_raw.round() != 0) if isinstance(layer, WeightedGroupSum) else (torch.sigmoid(layer.mask_logits).round() != 0)
             for group_idx in torch.where(alive_mask)[0]:
                 start, end = group_idx * g, (group_idx + 1) * g
                 group_is_alive = internal_mask[group_idx] if internal_mask is not None else torch.ones(g, dtype=torch.bool, device=device)
                 new_alive_mask[start:end] = group_is_alive
             alive_mask = new_alive_mask
             live_masks[i] = alive_mask
        elif isinstance(layer, (BinaryMask, CopyMask, GroupScale)):
             if isinstance(layer, BinaryMask): alive_mask = alive_mask & layer.mask
             elif isinstance(layer, CopyMask):
                 new_alive_mask = torch.zeros_like(layer.copy_from, dtype=torch.bool, device=device)
                 for j in torch.where(alive_mask)[0]: new_alive_mask[layer.copy_from[j]] = True
                 alive_mask = new_alive_mask
             live_masks[i] = alive_mask
    return live_masks

def rebuild_physically_pruned_model(
    original_model: nn.Sequential,
    live_masks: dict,
    args: argparse.Namespace,
    device: str = "cuda",
):
    """
    live_masks : get_live_masks()가 계산한 '각 레이어 입력 마스크' 딕셔너리
    - 마지막 LogicLayer(WGS 직전)는 out_dim을 유지하고 dead 출력 → op-id 0(one-tie)
    - WeightedGroupSum은 차원·가중치 그대로 복사
    """
    print("\n[물리적 압축 시작]")
    new_layers = [nn.Flatten()]

    # get_live_masks 와 동일 기준 레이어 목록
    keep_types = (LogicLayer, GroupSum, BinaryMask,
                  CopyMask, GroupScale, WeightedGroupSum, MaskedGroupSum)
    all_layers = [m for m in original_model.modules() if isinstance(m, keep_types)]

    # 첫 레이어 입력 마스크: 전체 True
    in_dim0 = input_dim_of_dataset(args.dataset)
    new_layer_input_mask = torch.ones(in_dim0, dtype=torch.bool, device=device)

    for i, layer in enumerate(all_layers):
        orig_live_in = live_masks[i]                    # 원본 좌표계 입력 마스크
        new_in_dim   = int(new_layer_input_mask.sum())

        # ── 일관성 확인 ──────────────────────────────────
        if new_in_dim != int(orig_live_in.sum()):
            raise RuntimeError(
                f"Mask mismatch @Layer {i} ({type(layer).__name__}): "
                f"new_in_dim {new_in_dim} vs live_in {int(orig_live_in.sum())}"
            )

        # 다음 레이어의 live_mask (없으면 all-True)
        is_last = (i + 1) == len(all_layers)
        orig_live_out = live_masks[i + 1] if not is_last else \
                        torch.ones(getattr(layer, "out_dim",
                                           getattr(layer, "k")), device=device, dtype=torch.bool)

        # ────────────────────────────────────────────────
        if isinstance(layer, LogicLayer):

            # ▸ WGS 직전 LogicLayer 인지 탐색
            nxt = i + 1
            skip = (BinaryMask, CopyMask, GroupScale, )  # ZeroPad1d 없음
            while nxt < len(all_layers) and isinstance(all_layers[nxt], skip):
                nxt += 1
            is_last_logic = nxt < len(all_layers) and isinstance(all_layers[nxt], WeightedGroupSum)

            # ── 출력 차원 결정 ─────────────────────────
            new_out_dim = layer.out_dim if is_last_logic else int(orig_live_out.sum())
            new_layer   = LogicLayer(new_in_dim, new_out_dim,
                                    connections="unique", device=device)
            if is_last_logic:
                # 1) 전체 weight는 그대로 복사 (dead_j → op-id 0 포함)
                new_layer.weights.data.copy_(layer.weights.data.detach())

                # 2) remap table 생성
                remap = torch.ones(layer.in_dim, dtype=torch.long, device=device)
                remap[orig_live_in] = torch.arange(new_in_dim, device=device)   # ← orig_live_in = 현재 레이어 입력 마스크

                # 3) indices 재매핑 (live + dead 모두)
                new_a = remap[layer.indices[0]]
                new_b = remap[layer.indices[1]]

                #   만약 dead_j 처리 직후였다면, 그 행은 0,0 이라 안전
                assert (new_a >= 0).all() and (new_b >= 0).all(), "remap 실패"

                new_layer.indices = (new_a, new_b)

                # 4) dead 출력 → op-id 0 (기존 로직 그대로)
                dead_j = (~orig_live_out).nonzero(as_tuple=True)[0]
                if dead_j.numel():
                    new_layer.weights.data[dead_j] = 0.
                    new_layer.weights.data[dead_j, 0] = 1.
                    new_layer.indices[0][dead_j] = 0
                    new_layer.indices[1][dead_j] = 0

                # 5) 다음 레이어 입력 마스크
                new_layer_input_mask = orig_live_out.clone()


            else:
                # ***여기서는 ‘부분 복사’만 수행해야 함***
                sel = orig_live_out                                 # 살아있는 출력
                new_layer.weights.data = layer.weights.data[sel].clone()
                old_a, old_b = layer.indices[0][sel], layer.indices[1][sel]

                remap = torch.ones(layer.in_dim, dtype=torch.long, device=device)
                remap[orig_live_in] = torch.arange(new_in_dim, device=device)

                new_a, new_b = remap[old_a], remap[old_b]
                if (new_a < 0).any() or (new_b < 0).any():
                    raise RuntimeError(f"remap fail @Layer {i}")

                new_layer.indices = (new_a, new_b)
                new_layer_input_mask = sel.clone()

            new_layers.append(new_layer)
            print(f"  - LogicLayer ({i}) : in {layer.in_dim}->{new_in_dim} , "
                  f"out {layer.out_dim}->{new_out_dim} "
                  f"{'(last)' if is_last_logic else ''}")

        # ── WeightedGroupSum 복사 ─────────────────────────
        elif isinstance(layer, WeightedGroupSum):
            new_wgs = WeightedGroupSum(layer.k, layer.in_dim, tau=layer.tau).to(device)
            new_wgs.weight_raw.data.copy_(layer.weight_raw.data.detach())
            new_layers.append(new_wgs)
            print(f"  - WeightedGroupSum ({i}) 유지 (in_dim={layer.in_dim})")
            # 이후 레이어가 없으므로 new_layer_input_mask 중요하지 않음
        # ── 다른 레이어(필요 시) 추가 로직 생략 ─────────────

    return nn.Sequential(*new_layers).to(device)




def get_model(args):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections, implementation=args.implementation)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []

    arch = args.architecture
    k = args.num_neurons
    l = args.num_layers

    ####################################################################################################################

    if arch == 'randomly_connected':
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )
    elif arch == 'learned_routing':
        orig_connections = llkw['connections']
        llkw['connections'] = 'ste'
        llkw['use_crossbar_tree'] = args.use_crossbar_tree
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicBlock(n_in=in_dim, n_out=k, width=[k], k_history=1,
                                       logic_layer_ste=True, crossbar_ste=True, **llkw))
        llkw['connections'] = orig_connections 
        logic_layers.append(LogicBlock(n_in=k, n_out=k, width=[k]*(l-1), k_history=1,
                                       logic_layer_ste=True, crossbar_ste=True, **llkw))

        #or _ in range(l - 1):
         #   logic_layers.append(LogicBlock(n_in=k, n_out=k, width=[k]*1, **llkw))
#
        model = torch.nn.Sequential(
            *logic_layers,
            nn.Dropout(0.01),
            GroupSum(class_count, args.tau)
        )
    elif arch=='learned_routing_top_bt':
        orig_connections = llkw['connections']
        llkw['connections'] = 'ste'
        llkw['use_crossbar_tree'] = args.use_crossbar_tree
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicBlock(n_in=in_dim, n_out=k, width=[k], implementation=args.implementation, k_history=1,
                                       logic_layer_ste=True, crossbar_ste=True, **llkw))
        llkw['connections'] = orig_connections 
        logic_layers.append(LogicBlock(n_in=k, n_out=k, width=[k]*(l-1), implementation=args.implementation, k_history=1,
                                       logic_layer_ste=True, crossbar_ste=True, **llkw))
        route = CrossbarLayer(
            in_dim=k,
            out_dim=k//args.bt_divider,
            device='cuda',
            ste=True,
            connections='ste',
            real_in_dim= k
        )
        logic_layers.append(route)
        model = torch.nn.Sequential(
            *logic_layers,
            nn.Dropout(0.01),
            GroupSum(class_count, args.tau)
        )
                
    elif arch == 'WGS':
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))
        model = torch.nn.Sequential(
            *logic_layers,
            WeightedGroupSum(class_count, k, args.tau)
        )

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################

    #total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    #print(f'total_num_neurons={total_num_neurons}')
    #total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    #print(f'total_num_weights={total_num_weights}')
    #if args.experiment_id is not None:
    #    results.store_results({
    #        'total_num_neurons': total_num_neurons,
    #        'total_num_weights': total_num_weights,
    #    })

    model = model.to('cuda')

    print(model)

    #if args.experiment_id is not None:
    #    results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# --------------------------------------------------------------
# 간단 버전 train(): GroupSum → WeightedGroupSum(가중치 1) 교체
# --------------------------------------------------------------
def train_with_prune(model, x, y, loss_fn, optimizer, lam_reg: float = 1e-2):
    """
    • model: nn.Sequential, 마지막 모듈이 GroupSum
    • WeightedGroupSum으로 교체 후 weight_raw만 학습
    • loss = task_loss + λ·reg_loss
    """
    # ─────────────── ① GroupSum → WeightedGroupSum  (최초 1회) ───────────────
    # ─────────────── ② forward & 손실 계산 ──────────────────────────────
    logits     = model(x)
    task_loss  = loss_fn(logits, y)
    reg_loss   = model[-1].reg_loss()                 # WeightedGroupSum 하나뿐

    loss = task_loss + lam_reg * reg_loss

    # ─────────────── ③ 역전파 & 업데이트 ────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()#, task_loss.item(), reg_loss.item()


def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to('cuda').round()).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool(), implementation=args.implementation)).argmax(-1) == y.to(
                    'cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python', 'triton'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)
    parser.add_argument('--use_crossbar_tree', dest='use_crossbar_tree', action='store_true', default=False)
    parser.add_argument('--noise_prob', type=float, default=0.0)
    parser.add_argument('--noise_sched', type=str, default='linear', choices=['linear', 'exp'])
    parser.add_argument('--noise_start', type=float, default=0.0)
    parser.add_argument('--noise_end', type=float, default=0.0)
    parser.add_argument('--load_model', action='store_true', help='Load the model from the results directory.')
    parser.add_argument('--grad-factor', type=float, default=1.)
    parser.add_argument('--prune_method', type=str, default=None, choices=['mi', 'random', 'copy', 'cpsat_budget', 'lp_budget', 'retrain', 'retrain_all', 'saliency', 'saliency_all',
                                                                             'random_all', 'masked_gs', 'random_finetune', 'random_all_finetune', 'saliency_finetune', 'saliency_all_finetune'])
    parser.add_argument('--prune_pct', type=float, default=0.1)
    parser.add_argument('--prune_thr', type=float, default=0.5)
    parser.add_argument('--prune_lam_reg', type=float, default=0.00002)
    parser.add_argument('--pruned_eid', type=int, default=None, help='New experiment ID for the pruned model.')
    parser.add_argument('--compression', action='store_true', help='Compress the model.')
    parser.add_argument('--bt_divider', type=int, default=4)

    args = parser.parse_args()

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    if args.load_model:
        model.load_state_dict(torch.load(f'results/{args.experiment_id}.pth'))
        print(model)
        #save_dir = "./results/"
        #eval_corr = register_groupsum_hook_corr_mi(model, k=num_classes_of_dataset(args.dataset), save_dir=save_dir)
        #acc = eval_corr(test_loader)     # 평가 + 그래프 저장
        acc = eval(model, test_loader, mode=False)
        print(f"val accuracy : {acc:.4f}")

        # 새로운 통합 함수 호출 (loader 전달 필요)
        (dead_nodes, total_nodes, ratio), live_nodes = finding_live_nodes(
            model, args, device='cuda', verbose=True
        )
        print(f"[Total Node] {total_nodes}")
        print(f"[Dead Node] {dead_nodes}")
        print(f"[Live Node] {live_nodes}")
        print(f"[Dead Ratio] {ratio} %")

        if args.prune_method in ['mi', 'random']:
            pruned_model, keep, scale = prune_global(
                model,                # 기존 Sequential
                test_loader,           # MI 추정을 위한 데이터
                k=num_classes_of_dataset(args.dataset),
                pct=args.prune_pct,          # 하위 10 % feature 제거
                device="cuda",
                use_random=(args.prune_method == 'random')
            )
            print("남은 feature:", int(keep.sum()), "/", keep.numel())
        elif args.prune_method == 'copy':
            pruned_model, copy_map = build_copy_pruning(
                model,
                test_loader,
                k=num_classes_of_dataset(args.dataset),
                corr_thr=args.prune_thr,
                device="cuda"
            )

            pruned_cnt  = int((copy_map != np.arange(copy_map.size)).sum())
            kept_cnt    = copy_map.size - pruned_cnt
            pruned_pct  = 100.0 * pruned_cnt / copy_map.size

            print(f"✨ Copy-pruning 결과")
            print(f"   전체 feature 수 : {copy_map.size}")
            print(f"   복사로 대체된(feature pruned) 수 : {pruned_cnt}개  "
                f"({pruned_pct:.2f} %)")
            print(f"   실제로 남은(unique) feature 수 : {kept_cnt}개")
        elif args.prune_method == 'cpsat_budget':
            pruned_model, keep, _ = prune_cpsat_budget(
                model,
                test_loader,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
            )
            print("남은 feature:", int(keep.sum()), "/" )
        elif args.prune_method == 'lp_budget':
            pruned_model, keep, _ = prune_lp_budget(
                model,
                test_loader,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
            )
        elif args.prune_method == 'saliency_all':
            pruned_model, keep, _ = prune_saliency(
                model,
                train_loader,
                loss_fn,
                global_pruning=True,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
            )
            for layer, k in keep.items():
                print(f"{layer} 남은 feature:", int(k.sum()), "/" , int(k.numel()))
        elif args.prune_method == 'random_all':
            pruned_model, keep, _ = prune_saliency(
                model,
                train_loader,
                loss_fn,
                global_pruning=True,
                random_pruning=True,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
            )
            for layer, k in keep.items():
                print(f"{layer} 남은 feature:", int(k.sum()), "/" , int(k.numel()))

        elif args.prune_method == 'retrain':

            gs = model[-1]
            assert isinstance(gs, GroupSum), "model[-1]은 GroupSum이어야 합니다."

            # 교체할 모듈 (가중치 1 초기화)
            wgs = WeightedGroupSum(
                k     = gs.k,
                in_dim = args.num_neurons, 
                tau   = gs.tau,
                beta  = getattr(gs, "beta", 0.0),
                init  = "ones",          # weight = 1
                reg_type = "l1"
            ).to(next(model.parameters()).device)

            model[-1] = wgs                               # 모듈 교체nvidi

            # 모든 파라미터 freeze, 단 weight_raw만 학습
            for p in model.parameters():
                p.requires_grad = False
            wgs.weight_raw.requires_grad = True
            model._gs_replaced = True                     # 플래그
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='iteration',
                total=1000,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                loss = train_with_prune(model, x, y, loss_fn, optimizer, lam_reg=args.prune_lam_reg)
            # print pruned ratio
            # print distribution of each integer value after round
            print("weight distribution: ", model[-1].weight_raw.data.round().unique(return_counts=True))
            print("pruned ratio: ", (model[-1].weight_raw.data.round()==0).sum().item()*100 / model[-1].weight_raw.data.numel(), "%")
            pruned_model = model

        elif args.prune_method == 'retrain_all':

            gs = model[-1]
            assert isinstance(gs, GroupSum), "model[-1]은 GroupSum이어야 합니다."
            
            # 교체할 모듈 (가중치 1 초기화)
            wgs = WeightedGroupSum(
                k     = gs.k,
                in_dim = args.num_neurons, 
                tau   = gs.tau,
                beta  = getattr(gs, "beta", 0.0),
                init  = "ones",          # weight = 1
                reg_type = "l1",
            ).to(next(model.parameters()).device)

            model[-1] = wgs                               # 모듈 교체nvidi

            # 모든 파라미터 freeze, 단 weight_raw만 학습
            for p in model.parameters():
                p.requires_grad = True
            wgs.weight_raw.requires_grad = True

            model._gs_replaced = True                     # 플래그

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='iteration',
                total=1000,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                loss = train_with_prune(model, x, y, loss_fn, optimizer, lam_reg=args.prune_lam_reg)
            # print pruned ratio
            # print distribution of each integer value after round
            print("weight distribution: ", model[-1].weight_raw.data.round().unique(return_counts=True))
            print("pruned ratio: ", (model[-1].weight_raw.data.round()==0).sum().item()*100 / model[-1].weight_raw.data.numel(), "%")
            pruned_model = model

        elif args.prune_method == 'masked_gs':

            # 1. 기존 GroupSum 레이어 가져오기
            gs = model[-1]
            assert isinstance(gs, GroupSum), "model[-1]은 GroupSum이어야 합니다."

            # 2. GroupSum을 MaskedGroupSum으로 교체 (핵심 변경 사항)
            # WeightedGroupSum 대신 MaskedGroupSum을 생성합니다.
            # init="ones" 인자는 더 이상 필요 없습니다.
            mgs = MaskedGroupSum(
                k        = gs.k,
                in_dim   = args.num_neurons,
                tau      = gs.tau,
                beta     = getattr(gs, "beta", 0.0),
                reg_type = "l1"  # 규제 타입은 그대로 사용
            ).to(next(model.parameters()).device)

            model[-1] = mgs  # 모델의 마지막 레이어를 교체

            # 3. 모델 파라미터 학습 설정
            # 'retrain_all' 전략에 따라 모든 파라미터의 그래디언트를 활성화합니다.
            # (만약 마스크만 학습하려면 다른 파라미터의 requires_grad를 False로 설정)
            for p in model.parameters():
                p.requires_grad = True
            '''
            for p in model.parameters():
                p.requires_grad = False
            mgs.mask_logits.requires_grad = True
            '''

            # optimizer는 모든 학습 가능한 파라미터를 대상으로 생성됩니다.
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # 4. 프루닝을 포함한 학습 진행 (이 부분은 변경 필요 없음)
            # train_with_prune 함수는 model[-1].reg_loss()를 호출하므로
            # MaskedGroupSum의 reg_loss()가 자동으로 사용됩니다.
            print("Starting retraining with MaskedGroupSum...")
            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='iteration',
                total=args.num_iterations, # 1000 대신 인자 사용 권장
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                loss = train_with_prune(model, x, y, loss_fn, optimizer, lam_reg=args.prune_lam_reg)

            # 5. 프루닝 결과 출력 (핵심 변경 사항)
            # weight_raw 대신 mask_logits를 사용하여 마스크를 계산하고 통계를 출력합니다.
            print("\n--- Pruning Results ---")
            with torch.no_grad():
                # mask_logits에서 바이너리 마스크(0 또는 1)를 얻습니다.
                mask = torch.round(torch.sigmoid(model[-1].mask_logits.data))
                
                # 마스크 값(0과 1)의 분포를 출력합니다.
                unique_vals, counts = mask.unique(return_counts=True)
                dist_str = ", ".join([f"'{int(v.item())}': {c.item()}" for v, c in zip(unique_vals, counts)])
                print(f"Mask distribution: {{ {dist_str} }}")

                # 프루닝된 뉴런(마스크 값이 0인 뉴런)의 비율을 계산합니다.
                pruned_count = (mask == 0).sum().item()
                total_count = mask.numel()
                pruned_ratio = pruned_count * 100 / total_count
                print(f"Pruned ratio: {pruned_ratio:.2f}% ({pruned_count} / {total_count})")
            
            pruned_model = model

        elif args.prune_method == 'random_finetune':
            print("--- Starting Random Pruning + Fine-tuning ---")

            # ==============================================================================
            # 1단계: 랜덤 프루닝으로 정적 마스크가 적용된 모델 생성
            # ==============================================================================
            # prune_global 함수는 내부에 BinaryMask와 GroupScale 레이어를 삽입한 모델을 반환합니다.
            pruned_model, keep_mask, _ = prune_global(
                model,
                test_loader,
                k=num_classes_of_dataset(args.dataset),
                pct=args.prune_pct,
                device="cuda",
                use_random=True
            )
            print(f"Step 1: Model pruned. Keep ratio: {keep_mask.float().mean() * 100:.2f}%")
            # pruned_model의 구조: [LogicLayer, ..., LogicLayer, BinaryMask, GroupSum, GroupScale]

            # ==============================================================================
            # 2단계: 파인튜닝을 위해 모델의 모든 학습 파라미터 활성화
            # ==============================================================================
            # 이전의 복잡한 동결/해제 로직을 삭제합니다.
            # pruned_model 내의 모든 학습 가능한 파라미터(즉, LogicLayer들의 가중치)에
            # 그래디언트 계산을 활성화하여 파인튜닝 대상으로 설정합니다.
            for param in pruned_model.parameters():
                param.requires_grad = True
            
            print("Step 2: All learnable parameters in the pruned model (LogicLayers) are set to be trainable.")

            # ==============================================================================
            # 3단계: 재학습(Fine-tuning)을 위한 옵티마이저 설정
            # ==============================================================================
            # 이제 pruned_model.parameters()는 LogicLayer의 가중치를 포함하므로 비어있지 않습니다.
            optimizer = torch.optim.Adam(
                pruned_model.parameters(), # 간단하게 모델의 모든 학습 파라미터를 전달
                lr=args.learning_rate
            )
            
            # ==============================================================================
            # 4단계: 모델 재학습 진행
            # ==============================================================================
            print("Step 3: Starting fine-tuning...")
            pruned_model.train()


            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='Fine-tuning',
                total=args.num_iterations,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                optimizer.zero_grad()
                outputs = pruned_model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            print("\nFine-tuning finished.")

        elif args.prune_method == 'random_all_finetune':
            print("--- Starting Random Pruning + Fine-tuning ---")

            # ==============================================================================
            # 1단계: Saliency 기반으로 정적 마스크가 적용된 모델 생성
            # ==============================================================================
            # prune_saliency 함수를 호출하여 BinaryMask가 삽입된 모델을 얻습니다.
            pruned_model, _, _ = prune_saliency(
                model,
                loader=test_loader,
                loss_fn=loss_fn,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
                global_pruning=True,   # 또는 False, 필요에 따라 설정
                random_pruning=True,  # Saliency를 사용하려면 False
                device="cuda"
            )
            print("Step 1: Model pruned with saliency scores.")

            # ==============================================================================
            # 2단계: 파인튜닝을 위해 모델의 모든 학습 파라미터 활성화
            # ==============================================================================
            # pruned_model 내의 모든 LogicLayer 가중치를 학습 대상으로 설정합니다.
            for param in pruned_model.parameters():
                param.requires_grad = True
            
            # 모델을 훈련 모드로 전환하여 STE 등 미분 가능한 로직이 동작하도록 합니다.
            pruned_model.train()
            
            print("Step 2: All learnable parameters in the pruned model are set to be trainable.")

            # ==============================================================================
            # 3단계: 재학습(Fine-tuning)을 위한 옵티마이저 설정
            # ==============================================================================
            # 옵티마이저는 pruned_model 내의 LogicLayer 파라미터들을 대상으로 생성됩니다.
            optimizer = torch.optim.Adam(
                pruned_model.parameters(),
                lr=args.learning_rate
            )
            
            # ==============================================================================
            # 4단계: 모델 재학습 진행
            # ==============================================================================
            print("Step 3: Starting fine-tuning...")
            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='Fine-tuning',
                total=args.num_iterations,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                optimizer.zero_grad()
                outputs = pruned_model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            print("\nFine-tuning finished.")


        elif args.prune_method == 'saliency':
            print("--- Starting Saliency Pruning ---")

            # ==============================================================================
            # 1단계: Saliency 기반으로 정적 마스크가 적용된 모델 생성
            # ==============================================================================
            # prune_saliency 함수를 호출하여 BinaryMask가 삽입된 모델을 얻습니다.
            pruned_model, keep, _ = prune_saliency_single(
                model,
                loader=test_loader,
                loss_fn=loss_fn,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
                global_pruning=True,   # 또는 False, 필요에 따라 설정
                random_pruning=False,  # Saliency를 사용하려면 False
                device="cuda"
            )

            for layer, k in keep.items():
                print(f"{layer} 남은 feature:", int(k.sum()), "/" , int(k.numel()))



        elif args.prune_method == 'saliency_finetune':
            print("--- Starting Saliency Pruning + Fine-tuning ---")

            # ==============================================================================
            # 1단계: Saliency 기반으로 정적 마스크가 적용된 모델 생성
            # ==============================================================================
            # prune_saliency 함수를 호출하여 BinaryMask가 삽입된 모델을 얻습니다.
            pruned_model, _, _ = prune_saliency_single(
                model,
                loader=test_loader,
                loss_fn=loss_fn,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100-args.prune_pct,
                global_pruning=True,   # 또는 False, 필요에 따라 설정
                random_pruning=False,  # Saliency를 사용하려면 False
                device="cuda"
            )
            print("Step 1: Model pruned with saliency scores.")

            # ==============================================================================
            # 2단계: 파인튜닝을 위해 모델의 모든 학습 파라미터 활성화
            # ==============================================================================
            # pruned_model 내의 모든 LogicLayer 가중치를 학습 대상으로 설정합니다.
            for param in pruned_model.parameters():
                param.requires_grad = True
            
            # 모델을 훈련 모드로 전환하여 STE 등 미분 가능한 로직이 동작하도록 합니다.
            pruned_model.train()
            
            print("Step 2: All learnable parameters in the pruned model are set to be trainable.")

            # ==============================================================================
            # 3단계: 재학습(Fine-tuning)을 위한 옵티마이저 설정
            # ==============================================================================
            # 옵티마이저는 pruned_model 내의 LogicLayer 파라미터들을 대상으로 생성됩니다.
            optimizer = torch.optim.Adam(
                pruned_model.parameters(),
                lr=args.learning_rate
            )
            
            # ==============================================================================
            # 4단계: 모델 재학습 진행
            # ==============================================================================
            print("Step 3: Starting fine-tuning...")
            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='Fine-tuning',
                total=args.num_iterations,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                optimizer.zero_grad()
                outputs = pruned_model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            print("\nFine-tuning finished.")

        elif args.prune_method == 'saliency_all_finetune':
            print("--- Starting Saliency Pruning with FIXED Masks & Fine-tuning Logic Layers ---")

            # 1단계: Saliency 기반으로 'BinaryMask'가 삽입된 모델 생성
            # 이 모델의 마스크는 학습 파라미터가 없습니다.
            pruned_model, keep_masks, _ = prune_saliency(
                model,
                loader=train_loader,
                loss_fn=loss_fn,
                k=num_classes_of_dataset(args.dataset),
                keep_pct=100 - args.prune_pct,
                global_pruning=True,
                random_pruning=False,
                device="cuda"
            )
            for layer_idx, keep_mask in keep_masks.items():
                kept = keep_mask.sum().item()
                total = keep_mask.numel()
                print(f"Layer {layer_idx} mask created. Kept {kept}/{total} ({kept/total*100:.2f}%)")

            # 2단계: 옵티마이저 설정
            # pruned_model.parameters()는 자동으로 LogicLayer의 학습 파라미터만 포함합니다.
            # BinaryMask에는 학습 파라미터가 없으므로 무시됩니다.
            optimizer = torch.optim.Adam(pruned_model.parameters(), lr=args.learning_rate)
            pruned_model.train()
            print("Optimizer is set to train Logic Layers ONLY.")

            # 3단계: 표준 파인튜닝 진행 (규제 손실 없음)
            print("Starting fine-tuning...")
            for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='Fine-tuning Logic Layers',
                total=args.num_iterations,
            ):
                x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
                y = y.to('cuda')

                optimizer.zero_grad()
                outputs = pruned_model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            print("\nFine-tuning finished.")

        if args.prune_method is None:
            pruned_model = model


        print("\n모델 구조:")
        print(pruned_model)

        val_acc = eval(pruned_model, test_loader, mode=False)
        print(f"pruned val acc: {val_acc:.4f}")



        # 새로운 통합 함수 호출 (loader 전달 필요)
        (dead_nodes_pruned, total_nodes, ratio_pruned), live_nodes_pruned = finding_live_nodes(
            pruned_model, args, device='cuda', verbose=True
        )

        print(f"[Pruned Dead Node] {dead_nodes_pruned}")
        print(f"[Pruned Live Node] {live_nodes_pruned}")
        print(f"[Pruned Dead Ratio] {ratio_pruned} %")
        print(f"[Prune efficiency] {(dead_nodes_pruned-dead_nodes)/live_nodes*100} %")


        # --- START OF MODIFICATION ---
        # Save the state and results of the pruned model under a new experiment ID
        if args.pruned_eid is not None:
            # 1. Save the pruned model's state_dict
            pruned_model_path = f'results/{args.pruned_eid}.pth'
            torch.save(pruned_model.state_dict(), pruned_model_path)
            print(f"\n✅ Pruned model state saved to: {pruned_model_path}")

            # 2. Create a new results file for the pruned model
            pruned_results = ResultsJSON(eid=args.pruned_eid, path='./results/')

            # 3. Store arguments and key results for the new pruned model
            pruned_results.store_args(args)
            
            pruned_result_data = {
                'source_experiment_id': args.experiment_id,
                'pruned_test_accuracy': val_acc,
                'pruned_live_nodes': live_nodes_pruned,
                'pruned_dead_nodes': dead_nodes_pruned,
                'pruned_dead_ratio_pct': ratio_pruned,
                'model_str': str(model)
            }
            pruned_results.store_final_results(pruned_result_data)
            pruned_results.save()
            print(f"✅ Pruned model results saved to: results/{args.pruned_eid}.json")


        print(f"")
        os._exit(0)

    ####################################################################################################################

    best_acc = 0
    best_acc_test = 0

    for i, (x, y) in tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='iteration',
            total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
        y = y.to('cuda')

        noise_prob = current_noise(i, args.num_iterations, args)
        set_module_noise(model, noise_prob)
        if args.architecture == 'WGS':
            # WeightedGroupSum 아키텍처일 경우 정규화가 포함된 학습 함수 호출
            loss = train_with_prune(model, x, y, loss_fn, optim, lam_reg=args.prune_lam_reg)
        else:
            # 그 외의 경우 기존 학습 함수 호출
            loss = train(model, x, y, loss_fn, optim)

        if (i+1) % args.eval_freq == 0:
            if args.extensive_eval:
                train_accuracy_train_mode = eval(model, train_loader, mode=True)
                valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
            else:
                train_accuracy_train_mode = -1
                valid_accuracy_eval_mode = -1
                valid_accuracy_train_mode = -1
            train_accuracy_eval_mode = eval(model, train_loader, mode=False)
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

            if args.packbits_eval:
                r['train_acc_eval'] = packbits_eval(model, train_loader)
                r['valid_acc_eval'] = packbits_eval(model, train_loader)
                r['test_acc_eval'] = packbits_eval(model, test_loader)

            if args.experiment_id is not None:
                results.store_results(r)
            else:
                print(r)


            if test_accuracy_eval_mode > best_acc:
                best_acc = test_accuracy_eval_mode
                if args.experiment_id is not None:
                    results.store_final_results(r)
                    # save the model
                    torch.save(model.state_dict(), f'results/{args.experiment_id}.pth')
                else:
                    print('IS THE BEST UNTIL NOW.')

            if args.experiment_id is not None:
                results.save()

    (dead_nodes, total_nodes, ratio), live_nodes = finding_live_nodes(
        model, args, device='cuda', verbose=True
    )
    print(f"[Total Node] {total_nodes}")
    print(f"[Dead Node] {dead_nodes}")
    print(f"[Live Node] {live_nodes}")
    print(f"[Dead Ratio] {ratio} %")


    ####################################################################################################################

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)


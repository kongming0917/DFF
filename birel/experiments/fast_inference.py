#!/usr/bin/env python3
import argparse
import torch
import sys
import json
from main import load_dataset, input_dim_of_dataset, num_classes_of_dataset
from main import eval as eval_fn
# WeightedGroupSum, MaskedGroupSumを difflogic에서 가져옵니다.
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet # <<< [수정] CompiledLogicNet 추가
from difflogic.difflogic import WeightedGroupSum, MaskedGroupSum, PrunedGroupSum, PrunedWeightedGroupSum
from birel.pruning import BinaryMask, CopyMask, GroupScale
import random
import numpy as np
import torch.nn as nn
import copy
import time
import math
import uci_datasets
import mnist_dataset
import torchvision
import os # <<< [수정] os 모듈 추가

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
    validation_loader = None
    test_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        # 변경: 전체 데이터셋을 배치 크기로 설정
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        # 변경: 전체 데이터셋을 배치 크기로 설정
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        # 변경: 전체 데이터셋을 배치 크기로 설정
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set_full = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset == 'mnist20x20')
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset == 'mnist20x20')

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set_full))
        valid_set_size = len(train_set_full) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set_full, [train_set_size, valid_set_size])

        # 변경: 각 분할된 데이터셋의 전체 크기를 배치 크기로 설정
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_set), shuffle=False, pin_memory=True)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    elif 'cifar-10' in args.dataset:
        transform_key = 'cifar-10-3-thresholds' if '3-thresholds' in args.dataset else 'cifar-10-31-thresholds'
        transform_func = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[transform_key]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform_func),
        ])
        train_set_full = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set_full))
        valid_set_size = len(train_set_full) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set_full, [train_set_size, valid_set_size])
        
        # 변경: 각 분할된 데이터셋의 전체 크기를 배치 크기로 설정
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_set), shuffle=False, pin_memory=True)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader










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
        layers.append(LogicLayer(in_dim, out_dim, connections='unique', implementation='cuda'))
        
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
    if prune_method in ['mi', 'random', 'saliency', 'saliency_finetune',
                        'lp_budget', 'cpsat_budget', 'random_finetune', 'saliency_single']:
        keep_mask = torch.ones(ks[-1], dtype=torch.bool)
        group_scale = torch.ones(n_classes, dtype=torch.float32)

        final_layers = [
            *base_layers,
            BinaryMask(keep_mask),
            GroupSum(n_classes, tau=10),
            GroupScale(group_scale)
        ]

    elif prune_method == 'copy':
        final_layers = [
            *base_layers,
            CopyMask(num_features=ks[-1]),
            GroupSum(n_classes, tau=10)
        ]

    elif prune_method in ['saliency_all', 'random_all', 'saliency_all_finetune']:
        final_layers = [base_layers[0]]          # Flatten
        for logic_layer in base_layers[1:]:
            final_layers.append(logic_layer)        # LogicLayer
            final_layers.append(BinaryMask(num_features=logic_layer.out_dim))
        final_layers.append(GroupSum(n_classes, tau=10))
        final_layers.append(GroupScale(num_groups=n_classes))

    elif prune_method in ['retrain', 'retrain_all']:
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
            D = layer.in_dim if hasattr(layer, 'in_dim') and layer.in_dim is not None else args.num_neurons
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
            if isinstance(layer, BinaryMask): alive_mask = alive_mask & layer.mask.bool()
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
) -> nn.Sequential:
    """
    주어진 live_masks를 기반으로 모델을 물리적으로 재구성(압축)하는 최종 통합 버전.
    상태 추적 변수를 사용하여 BinaryMask와 같은 중간 레이어의
    프루닝 효과를 올바르게 전파합니다.
    """
    print("\n[모델 물리적 압축 시작]")
    pruned_layers = [nn.Flatten()]

    keep_types = (LogicLayer, GroupSum, BinaryMask, CopyMask, 
                  GroupScale, WeightedGroupSum, MaskedGroupSum)
    all_layers = [m for m in original_model.modules() if isinstance(m, keep_types)]
    
    # (1) 압축 후 텐서의 '활성 마스크'를 추적할 상태 변수를 초기화합니다.
    # 이 변수가 레이어를 거치며 프루닝 효과를 누적하여 전달합니다.
    in_dim0 = input_dim_of_dataset(args.dataset)
    current_live_mask = torch.ones(in_dim0, dtype=torch.bool, device=device)

    for i, layer in enumerate(all_layers):
        # 새로운 입/출력 차원을 '상태 변수'와 'live_masks'를 기반으로 계산합니다.
        new_in_dim = int(current_live_mask.sum())
        
        is_last_layer = (i + 1) == len(all_layers)
        if not is_last_layer:
            live_out_mask = live_masks[i + 1]
        else:
            out_dim_val = getattr(layer, "out_dim", getattr(layer, "k", args.num_neurons))
            live_out_mask = torch.ones(out_dim_val, device=device, dtype=torch.bool)
        
        new_out_dim = int(live_out_mask.sum())
            
        # ───────────────── 레이어 타입별 재구성 ─────────────────
        
        if isinstance(layer, LogicLayer):
            new_layer = LogicLayer(new_in_dim, new_out_dim, connections="unique", device=device)
            
            remap_table = torch.full((layer.in_dim,), -1, dtype=torch.long, device=device)
            remap_table[current_live_mask] = torch.arange(new_in_dim, device=device) # ✅ 수정된 부분

            remapped_a_all = remap_table[layer.indices[0]]
            remapped_b_all = remap_table[layer.indices[1]]
            
            live_out_indices = live_out_mask.nonzero(as_tuple=True)[0]
            
            new_weights = layer.weights.data[live_out_indices].clone()
            new_a = remapped_a_all[live_out_indices]
            new_b = remapped_b_all[live_out_indices]
            
            new_a[new_a == -1] = 0
            new_b[new_b == -1] = 0

            new_layer.weights.data = new_weights
            new_layer.indices = (new_a, new_b)
            
            pruned_layers.append(new_layer)
            print(f"  - LogicLayer ({i}): in {layer.in_dim} -> {new_in_dim}, out {layer.out_dim} -> {new_out_dim}")

        elif isinstance(layer, GroupSum):
            # (2) GroupSum의 입력 차원을 결정하기 위해 'out_dim'을 가진 이전 레이어를 탐색합니다.
            prev_layer_with_dim = None
            for j in range(i - 1, -1, -1):
                if hasattr(all_layers[j], 'out_dim'):
                    prev_layer_with_dim = all_layers[j]
                    break
            
            if prev_layer_with_dim is None:
                raise RuntimeError(f"Could not find a preceding layer with 'out_dim' for {type(layer).__name__}")

            original_in_dim = prev_layer_with_dim.out_dim
            original_class_size = original_in_dim // layer.k
            
            new_group_sizes = []
            for group_idx in range(layer.k):
                start = group_idx * original_class_size
                end = start + original_class_size
                
                # (3) [핵심] live_masks[i]가 아닌, 올바르게 전파된 current_live_mask를 사용합니다.
                live_count = current_live_mask[start:end].sum().item()
                new_group_sizes.append(live_count)
            
            new_group_sizes_tensor = torch.tensor(new_group_sizes, dtype=torch.int, device=device)
            print(new_group_sizes_tensor)
            # `PrunedGroupSum`이 올바른 group_sizes로 생성됩니다.
            new_layer = PrunedGroupSum(
                k=layer.k,
                group_sizes=new_group_sizes_tensor,
                tau=layer.tau,
                beta=layer.beta,
                noise_prob=layer.noise_prob
            ).to(device)
            pruned_layers.append(new_layer)
            print(f"  - GroupSum ({i}) -> PrunedGroupSum (in: {original_in_dim} -> {new_layer.in_dim}, k: {layer.k})")

        elif isinstance(layer, WeightedGroupSum):
            # GroupSum과 동일한 로직으로 이전 레이어와 원래 입력 차원을 찾습니다.
            prev_layer_with_dim = next((all_layers[j] for j in range(i - 1, -1, -1) if hasattr(all_layers[j], 'out_dim')), None)
            if prev_layer_with_dim is None:
                raise RuntimeError(f"Could not find a preceding layer with 'out_dim' for {type(layer).__name__}")
            
            original_in_dim = prev_layer_with_dim.out_dim
            original_class_size = original_in_dim // layer.k

            # GroupSum과 동일한 로직으로 새로운 그룹 크기를 계산합니다.
            new_group_sizes = []
            for group_idx in range(layer.k):
                start = group_idx * original_class_size
                end = start + original_class_size
                live_count = current_live_mask[start:end].sum().item()
                new_group_sizes.append(live_count)
            
            new_group_sizes_tensor = torch.tensor(new_group_sizes, dtype=torch.int, device=device)

            # 활성 뉴런에 해당하는 가중치만 추출합니다.
            # 원래 가중치는 (k, class_size) 형태이므로 1D로 펼쳐서 마스크를 적용합니다.
            original_weights_flat = layer.weight_raw.view(-1)
            pruned_weights_flat = original_weights_flat[current_live_mask].clone()
            print(new_group_sizes_tensor)
            # `PrunedWeightedGroupSum` 레이어를 생성합니다.
            new_layer = PrunedWeightedGroupSum(
                k=layer.k,
                group_sizes=new_group_sizes_tensor,
                weights=pruned_weights_flat, # 프루닝된 1D 가중치 전달
                tau=layer.tau
            ).to(device)
            pruned_layers.append(new_layer)
            print(f"  - WeightedGroupSum ({i}) -> PrunedWeightedGroupSum (in: {original_in_dim} -> {new_layer.in_dim}, k: {layer.k})")

        elif isinstance(layer, (MaskedGroupSum, GroupScale)):
            new_layer = copy.deepcopy(layer).to(device)
            pruned_layers.append(new_layer)
            print(f"  - {type(layer).__name__} ({i}): Copied without structural changes.")

        # ───────── BinaryMask ───────────────────────────────────────────
        elif isinstance(layer, BinaryMask):
            # BinaryMask 자체는 계산이 없지만, feature-level 활성 마스크를
            current_live_mask &= layer.mask.bool().to(device)
            print(f"  - BinaryMask ({i}): applied ({int(current_live_mask.sum())}/{len(current_live_mask)})")
            continue          # live_out_mask로 덮어쓰지 말고 바로 다음 레이어로

        # ───────── CopyMask ─────────────────────────────────────────────
        elif isinstance(layer, CopyMask):
            # copy_from 테이블에 따라 인덱스를 재매핑
            new_mask = torch.zeros_like(layer.copy_from, dtype=torch.bool, device=device)
            new_mask[layer.copy_from] = current_live_mask
            current_live_mask = new_mask
            print(f"  - CopyMask ({i}): remapped ({int(current_live_mask.sum())}/{len(current_live_mask)})")
            continue          # 동일하게 덮어쓰기 방지
        
        # (4) 다음 레이어를 위해, '상태 변수'를 현재 레이어의 활성 출력 마스크로 업데이트합니다.
        current_live_mask = live_out_mask.clone()
        
    return nn.Sequential(*pruned_layers).to(device)


# ───────── main ─────────────────
pa = argparse.ArgumentParser()
pa.add_argument('-eid', '--experiment_id', type=int, default=None)

pa.add_argument('--dataset', type=str, choices=[
    'adult', 'breast_cancer',
    'monk1', 'monk2', 'monk3',
    'mnist', 'mnist20x20',
    'cifar-10-3-thresholds',
    'cifar-10-31-thresholds',
], required=True, help='the dataset to use')
pa.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
pa.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
pa.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
pa.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
pa.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

pa.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                     'implementation intended for helping with the understanding.')

pa.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                             'additional eval step.')
pa.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

pa.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
pa.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

pa.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
pa.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

pa.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
pa.add_argument('--architecture', '-a', type=str, default='randomly_connected')
pa.add_argument('--num_neurons', '-k', type=int, default=512)
pa.add_argument('--num_layers', '-l', type=int, default=2)
pa.add_argument('--use_crossbar_tree', dest='use_crossbar_tree', action='store_true', default=False)
pa.add_argument('--noise_prob', type=float, default=0.0)
pa.add_argument('--noise_sched', type=str, default='linear', choices=['linear', 'exp'])
pa.add_argument('--noise_start', type=float, default=0.0)
pa.add_argument('--noise_end', type=float, default=0.0)
pa.add_argument('--load_model', action='store_true', help='Load the model from the results directory.')
pa.add_argument('--grad-factor', type=float, default=1.)
pa.add_argument('--prune_method', type=str, default=None, choices=['mi', 'random', 'copy', 'cpsat_budget', 'lp_budget', 'retrain', 'retrain_all', 'saliency', 'saliency_all',
                                                                    'random_all', 'masked_gs', 'random_finetune', 'random_all_finetune', 'saliency_finetune', 'saliency_all_finetune'])
pa.add_argument('--prune_pct', type=float, default=0.1)
pa.add_argument('--prune_thr', type=float, default=0.5)
pa.add_argument('--prune_lam_reg', type=float, default=0.00002)
pa.add_argument('--pruned_eid', type=int, default=None, help='New experiment ID for the pruned model.')
pa.add_argument('--compression', action='store_true', help='Compress the model.')
pa.add_argument('--bt_divider', type=int, default=4)

pa.add_argument("--bit", type=int, default=64, choices=[8, 16, 32, 64])
pa.add_argument("--iters", type=int, default=5000)


pa.add_argument('--load-so-path', type=str, default=None, help='Path to a pre-compiled .so file to load for speed testing.')


args = pa.parse_args()



dev = "cuda" if torch.cuda.is_available() else sys.exit("CUDA가 필요합니다.")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

train_loader, _, _ = load_dataset(args)

# Prune Method 인자에 따라 모델 빌더를 선택
if args.prune_method:
    print(f"'{args.prune_method}' Pruning이 적용된 모델을 로드합니다.")
    model = build_pruned_model(args.dataset, [args.num_neurons] * args.num_layers, args.prune_method)
    model.load_state_dict(torch.load(f"results/{args.pruned_eid}.pth"))
else:
    print("Original 모델을 로드합니다.")
    model = build_model(args.dataset, [args.num_neurons] * args.num_layers)
    model.load_state_dict(torch.load(f"results/{args.experiment_id}.pth"))


print("\n모델 구조:")
print(model)



print(model)
model.to(dev).eval()
acc = eval_fn(model, train_loader, mode=False)

print(f"Accuracy: {acc:.4f}")


if args.compression:
    with torch.no_grad():
        live_masks = get_live_masks(model, args, device="cuda")
        model = rebuild_physically_pruned_model(model, live_masks, args, device="cuda")
    
    
    print("\n[압축 후 모델 구조]")
    print(model)
    model.to(dev).eval()

    print("\n[압축 후 모델 정확도 측정]")
    compressed_acc = eval_fn(model, train_loader, mode=False)
    print(f"  - 정확도: {compressed_acc:.4f}")

    print("\n[정확도 비교]")
    if abs(acc - compressed_acc) < 1e-5:
        print("  ✅ 정확도가 일치합니다. 압축이 성공적으로 완료되었습니다.")
    else:
        print(f"  ❌ 경고: 정확도가 일치하지 않습니다! 초기: {acc:.4f}, 압축 후: {compressed_acc:.4f}")


# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# [사용자 요청] 최종 모델의 마지막 레이어를 가중치 없는 버전으로 교체
# 이 로직은 압축 여부와 상관없이, 속도 측정 직전에 최종 모델에 대해 실행됩니다.
# ==============================================================================
print("\n" + "="*70)
print("INFO: 속도 측정을 위해 마지막 레이어를 가중치 없는 버전으로 교체 시도...")

# 현재 모델의 마지막 레이어를 가져옵니다.
last_layer = model[-1]
layer_swapped = False

# 1. 마지막 레이어가 PrunedWeightedGroupSum인 경우
if isinstance(last_layer, PrunedWeightedGroupSum):
    print("  - 감지된 레이어: PrunedWeightedGroupSum")
    print("  - 교체할 레이어: PrunedGroupSum (가중치 없는 버전)")
    
    # PWGS의 속성(k, group_sizes, tau)을 사용하여 PGS를 생성
    new_layer = PrunedGroupSum(
        k=last_layer.k,
        group_sizes=last_layer.group_sizes, # PWGS는 이 속성을 가지고 있음
        tau=last_layer.tau
    ).to(dev)
    
    model[-1] = new_layer # 모델의 마지막 레이어를 교체
    layer_swapped = True

# 2. 마지막 레이어가 (압축되지 않은) WeightedGroupSum인 경우
elif isinstance(last_layer, WeightedGroupSum):
    print("  - 감지된 레이어: WeightedGroupSum")
    print("  - 교체할 레이어: GroupSum (가중치 없는 버전)")

    # WGS의 속성(k, tau)을 사용하여 GS를 생성
    new_layer = GroupSum(
        k=last_layer.k,
        tau=last_layer.tau
    ).to(dev)
    
    model[-1] = new_layer # 모델의 마지막 레이어를 교체
    layer_swapped = True

if layer_swapped:
    print("  ✅ 레이어 교체가 완료되었습니다.")
    print("="*70)
    print("\n[교체 후 최종 모델 구조]")
    print(model)
else:
    print("  - 교체 대상 레이어(WGS, PWGS)가 없어 원본 모델을 그대로 사용합니다.")
    print("="*70)

# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================


model.to(dev).eval()            # ← self.in_dim 확정
X, _ = next(iter(train_loader))
B = X.size(0)
pb = PackBitsTensor(X.to('cuda').reshape(B, -1).round().bool())
pb_orig = pb.t.clone()                          # 원본 비트 백업

# bit-packed 입력용 Bool 행렬 (고정)

#bool_mat = X.view(B, -1).gt(0).bool().to(dev)  # (batch, in_dim)
#pb = PackBitsTensor(bool_mat, bit_count=args.bit)  # ★ 매번 새로 생성

warmup = 10



# 모드 1: 이미 컴파일된 .so 파일 로드 및 측정
if args.load_so_path:
    print("\n" + "="*80)
    print(f"사전 컴파일된 모델 로드 및 측정 모드")
    print(f"대상 파일: {args.load_so_path}")
    print("="*80)

    if not os.path.exists(args.load_so_path):
        sys.exit(f"오류: 지정된 .so 파일을 찾을 수 없습니다: {args.load_so_path}")

    # 1. 컴파일된 모델과 동일한 구조의 PyTorch 모델로 CNet 객체 생성
    #    .compile() 호출 시, 파일이 이미 존재하면 컴파일을 건너뛰고 로드만 수행
    print("컴파일된 라이브러리를 로드하기 위해 CNet 객체를 초기화합니다...")
    compiled_model = CompiledLogicNet(
        model=model.to('cpu'), # 구조 동기화를 위해 CPU 버전 모델 전달
        num_bits=args.bit,
        verbose=True
    )
    compiled_model.compile(save_lib_path=args.load_so_path, verbose=False)
    print("✅ 컴파일된 모델 로드 완료.")

    # 2. CPU 추론용 데이터 준비 및 속도 측정
    numpy_input = X.reshape(B, -1).round().bool().numpy()
    
    def cpu_ns(compiled_net, data, it=1000):
        for _ in range(warmup): _ = compiled_net(data)
        start_time = time.perf_counter_ns()
        for _ in range(it): _ = compiled_net(data)
        end_time = time.perf_counter_ns()
        return (end_time - start_time) / it

    num_runs = 10
    timings_ns_cpu = []
    print(f"\n[Loaded Compiled CPU | bit-{args.bit}] 추론 속도 측정 시작...")
    for i in range(num_runs):
        ns_per_batch_cpu = cpu_ns(compiled_model, numpy_input, it=args.iters)
        timings_ns_cpu.append(ns_per_batch_cpu)
        print(f"  - 실행 {i + 1}/{num_runs}: {ns_per_batch_cpu / B:.2f} ns / sample")
        time.sleep(0.1)

    if timings_ns_cpu:
        average_ns_per_batch_cpu = sum(timings_ns_cpu) / len(timings_ns_cpu)
        print(f"\n[Loaded Compiled CPU | bit-{args.bit}] {average_ns_per_batch_cpu / B:.2f} ns / sample (batch={B})")
    
    sys.exit()
    




for _ in range(warmup):
    _ = model(pb)
    pb.t = pb_orig      # 복구
torch.cuda.synchronize()


# 3) 측정 (이벤트 + 동기화)                     # ★
def cuda_ns(it=3000, bool_mat=None):
    with torch.no_grad():
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()

        for i in range(it):
            _ = model(pb)
            pb.t = pb_orig    # 원본 비트로 되돌림


        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e6 / it  # ns

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 코드 수정: 10회 실행하여 평균 시간을 계산
# ==============================================================================
num_runs = 10
timings_ns = []
print(f"\n[GPU PackBitsTensor | bit-{args.bit}] 추론 속도 측정 시작 (총 {num_runs}회 실행 후 평균)")

for i in range(num_runs):
    ns_per_batch = cuda_ns(args.iters)
    timings_ns.append(ns_per_batch)
    print(f"  - 실행 {i + 1}/{num_runs}: {ns_per_batch / B:.2f} ns / sample")
    time.sleep(0.1) # 다음 측정에 대한 안정성 확보

# 최종 평균 계산 및 출력
if timings_ns:
    average_ns_per_batch = sum(timings_ns) / len(timings_ns)
    print(f"\n[GPU PackBitsTensor | bit-{args.bit}] {average_ns_per_batch / B:.2f} ns / sample (batch={B})")
else:
    print("측정에 실패했습니다.")
# ==============================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# [추가된 코드] --compile_model이 True일 경우, 컴파일된 모델의 CPU 추론 속도를 측정합니다.
# ==============================================================================
if args.compile_model:
    # 1. 모델 컴파일
    print("\n" + "="*80)
    print(f"모델을 C 코드로 컴파일합니다... (bit={args.bit})")
    print("="*80)
    
    # 컴파일된 라이브러리를 저장할 경로 설정
    lib_dir = 'lib'
    os.makedirs(lib_dir, exist_ok=True)
    save_lib_path = os.path.join(lib_dir, f'model_{args.experiment_id}_bs{args.batch_size}_{args.bit}bit.so')

    # ▼▼▼▼▼▼▼▼▼▼ [수정된 부분 시작] ▼▼▼▼▼▼▼▼▼▼
    
    # 컴파일러 객체를 생성합니다. 이 객체가 컴파일 후 추론에 바로 사용됩니다.
    compiled_model = CompiledLogicNet(
        model=model.to('cpu'), # 컴파일은 CPU 모델 기반으로 수행
        num_bits=args.bit,
        cpu_compiler='gcc',
        verbose=True
    )
    # .compile()을 호출하면 compiled_model 객체가 추론 가능한 상태가 됩니다.
    compiled_model.compile(
        opt_level=3,
        save_lib_path=save_lib_path,
        verbose=True
    )
    print(f"✅ 모델이 컴파일되어 '{save_lib_path}'에 저장되었습니다.")

    # 2. 컴파일된 모델 로드 (이 단계가 필요 없어졌습니다)
    print("\n컴파일된 모델의 속도 측정을 준비합니다...")
    # ❌ 삭제: compiled_model = CompiledLogicNet(load_lib_path=save_lib_path)

    # ▲▲▲▲▲▲▲▲▲▲ [수정된 부분 끝] ▲▲▲▲▲▲▲▲▲▲

    # 3. CPU 추론을 위한 입력 데이터 준비 (NumPy boolean array)
    numpy_input = X.reshape(B, -1).round().bool().numpy()
    print(f"NumPy 입력 데이터 준비 완료. Shape: {numpy_input.shape}, Dtype: {numpy_input.dtype}")

    # 4. CPU 속도 측정 함수 정의
    def cpu_ns(compiled_net, data, it=1000):
        # 워밍업
        for _ in range(warmup):
            _ = compiled_net(data)
        
        # 측정
        start_time = time.perf_counter_ns()
        for _ in range(it):
            _ = compiled_net(data)
        end_time = time.perf_counter_ns()
        
        return (end_time - start_time) / it

    # 5. CPU 추론 속도 측정 실행
    timings_ns_cpu = []
    print(f"\n[Compiled CPU | bit-{args.bit}] 추론 속도 측정 시작 (총 {num_runs}회 실행 후 평균)")
    for i in range(num_runs):
        # 컴파일에 사용된 'compiled_model' 객체를 그대로 전달합니다.
        ns_per_batch_cpu = cpu_ns(compiled_model, numpy_input, it=args.iters)
        timings_ns_cpu.append(ns_per_batch_cpu)
        print(f"  - 실행 {i + 1}/{num_runs}: {ns_per_batch_cpu / B:.2f} ns / sample")
        time.sleep(0.1)

    # 최종 평균 계산 및 출력
    if timings_ns_cpu:
        average_ns_per_batch_cpu = sum(timings_ns_cpu) / len(timings_ns_cpu)
        print(f"\n[Compiled CPU | bit-{args.bit}] {average_ns_per_batch_cpu / B:.2f} ns / sample (batch={B})")
    else:
        print("CPU 속도 측정에 실패했습니다.")







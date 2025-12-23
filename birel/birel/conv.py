import argparse
import math
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum      # LogicLayer 코드 그대로 사용
from difflogic.functional import (
    bin_op_s,
    get_pairwise_connections,
    get_random_connections_in_groups
)
from birel.model import CrossbarLayer, VotingCrossbarLayer, BlockEfficientCrossbarLayer

# CUDA extension import
try:
    import difflogic_cuda
except ImportError:
    difflogic_cuda = None
    print("Warning: difflogic_cuda not available. FusedTreeConvLayer will use Python fallback.")

class ORPool2d(nn.Module):
    """Logical OR Pooling implemented via Max Pooling."""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # [수정됨] kernel_size 와 stride를 인스턴스 속성으로 저장합니다.
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.max_pool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding)
    def forward(self, x):
        return self.max_pool(x)

class MaskLayer(torch.nn.Module):
    """
    Mask layer that applies mask operations based on mask_weights.
    mask_weights: [out_dim, 3] where:
    - [:, 0]: 0-tie weight (output = 0)
    - [:, 1]: 1-tie weight (output = 1)
    - [:, 2]: bypass weight (output = original)
    
    Uses softmax + argmax with STE for backward pass when trainable.
    When frozen, uses register_buffer (non-trainable).
    """
    def __init__(self, out_dim: int, device: str = 'cuda', frozen: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.frozen = frozen
        
        if frozen:
            # Frozen mode: use register_buffer (non-trainable)
            self.register_buffer('mask_weights', torch.ones(out_dim, 3, device=device))
        else:
            # Trainable mode: use Parameter
            self.mask_weights = torch.nn.parameter.Parameter(torch.ones(out_dim, 3, device=device))
        
        with torch.no_grad():
            self.mask_weights.fill_(0.0)
            self.mask_weights[:, 2] = 5.0  # bypass weight (default)
    
    def forward(self, x):
        """
        Apply mask based on mask_weights.
        
        Args:
            x: input tensor [batch_size, out_dim]
        
        Returns:
            x_masked: masked output [batch_size, out_dim]
        """
        if self.frozen:
            # Frozen mode: use simple argmax (no softmax needed)
            mask_selection = torch.argmax(self.mask_weights, dim=-1)  # [out_dim]
            
            # Expand mask_selection to match x shape [batch_size, out_dim]
            batch_size = x.shape[0]
            mask_selection_expanded = mask_selection.unsqueeze(0).expand(batch_size, -1)  # [batch_size, out_dim]
            
            # Apply mask operations directly using torch.where
            # mask_selection: 0 (0-tie), 1 (1-tie), 2 (bypass)
            x_masked = torch.where(mask_selection_expanded == 0, torch.zeros_like(x),
                          torch.where(mask_selection_expanded == 1, torch.ones_like(x), x))
        else:
            # Training mode: use softmax + STE for gradient flow
            # Compute softmax probabilities
            mask_probs = torch.nn.functional.softmax(self.mask_weights, dim=-1)  # [out_dim, 3]
            
            # Get hard selection (argmax)
            mask_selection = torch.argmax(self.mask_weights, dim=-1)  # [out_dim]
            mask_hard = torch.nn.functional.one_hot(mask_selection, 3).to(x.dtype)  # [out_dim, 3]
            
            # Apply STE: use hard selection for forward, soft probabilities for backward
            mask_ste = mask_hard.detach() + (mask_probs - mask_probs.detach())  # [out_dim, 3]
            
            # Apply mask operations
            # mask_ste[:, 0]: 0-tie (output = 0)
            # mask_ste[:, 1]: 1-tie (output = 1)
            # mask_ste[:, 2]: bypass (output = original)
            
            # x shape: [batch_size, out_dim]
            # Expand mask_ste to [batch_size, out_dim, 3]
            batch_size = x.shape[0]
            mask_expanded = mask_ste.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, out_dim, 3]
            
            # Create operations: [0-tie, 1-tie, bypass]
            ops = torch.stack([
                torch.zeros_like(x),  # 0-tie
                torch.ones_like(x),   # 1-tie
                x                     # bypass
            ], dim=-1)  # [batch_size, out_dim, 3]
            
            # Apply mask: sum over operations weighted by mask_ste
            x_masked = torch.sum(ops * mask_expanded, dim=-1)  # [batch_size, out_dim]
        
        return x_masked


class Crossbar1x1Conv(nn.Module):
    """
    [통합 버전]
    - shuffle=False (기본): 1x1 컨볼루션처럼 동작
    - shuffle=True: (B, H*W*C)로 flatten하고 채널을 섞어 FC 레이어처럼 동작
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, 
                 connections: str = 'ste', shuffle: bool = False,
                 input_height: int = None, input_width: int = None, init : str = "normal"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle = shuffle

        if self.shuffle:
            # --- FC 모드 초기화 ---
            assert input_height is not None and input_width is not None, \
                "'shuffle=True' 모드에서는 'input_height'와 'input_width'가 반드시 필요합니다."
            self.input_height = input_height
            self.input_width = input_width
            
            flat_in_dim = in_channels * input_height * input_width
            flat_out_dim = out_channels * input_height * input_width
            self.crossbar = BlockEfficientCrossbarLayer(
                in_dim=flat_in_dim,
                out_dim=flat_out_dim, # 이 경우 out_channels는 out_features를 의미
                num_blocks=num_blocks,
                connections=connections,
                device="cuda"
            )
        else:
            # --- 1x1 Conv 모드 초기화 ---
            self.crossbar = BlockEfficientCrossbarLayer(
                in_dim=in_channels,
                out_dim=out_channels,
                num_blocks=num_blocks,
                connections=connections,
                device="cuda",
                init = init
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        if self.shuffle:
            # --- FC 모드 Forward ---
            # 1. (B, C*H*W) 형태로 Flatten
            x_flat = x.view(B, -1)

            # 2. 학습 중에만 flatten된 피처 순서를 섞음
            if self.training:
                permutation = torch.randperm(x_flat.shape[1], device=x.device)
                x_flat = x_flat[:, permutation]
            
            # 3. Crossbar 적용: (B, C*H*W) -> (B, out_features)
            output = self.crossbar(x_flat)
            output = output.view(B, self.out_channels, H, W)

            return output
        
        else:
            # --- 1x1 Conv 모드 Forward ---
            # 1. (B*H*W, C_in) 형태로 Reshape
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C_in)
            
            # 2. Crossbar 적용
            out_reshaped = self.crossbar(x_reshaped)
            
            # 3. 원래 4D 형태로 복원
            output = out_reshaped.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
            return output

    def extra_repr(self) -> str:
        if self.shuffle:
            flat_dim = self.in_channels * self.input_height * self.input_width
            return (f"mode=FC+Shuffle, in_shape=({self.in_channels}, {self.input_height}, {self.input_width}), "
                    f"out_features={self.out_channels}\n"
                    f"  (Flattened in_dim={flat_dim})")
        else:
            return f"mode=1x1-Conv, in_channels={self.in_channels}, out_channels={self.out_channels}"



    

    



class TreeConvLayer(nn.Module):
    """
    [MANUAL VERSION]
    CascadingLogicLayer 대신 수동으로 정의된 LogicLayer들을 사용하는 버전.
    내부 레이어들을 nn.Sequential(self.cascade)으로 묶어 분석을 용이하게 합니다.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, k: int = None, 
                 k_in: int = None,
                 logic_layer_kwargs: dict = None, compress: bool = False):
        super().__init__()
        self.in_channels = None
        self.out_channels = None
        self.in_dim = in_channels  # in_channels를 in_dim으로 통일
        self.out_dim = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 입력 특징 벡터의 전체 차원을 올바르게 계산합니다.
        self.logic_in_features = self.in_dim * (self.kernel_size ** 2)

        self.k_in = k_in
        self.k = k
        # 중간 레이어 차원 (예시, 필요에 따라 조절 가능)
        intermediate_dim1 = self.out_dim * 4
        intermediate_dim2 = self.out_dim * 2

        print(f"Creating Manual TreeConvLayer (in={in_channels}, out={out_channels}, K={kernel_size})")
        print(f"  - Manual cascade path: {self.logic_in_features} -> {intermediate_dim1} -> {intermediate_dim2} -> {self.out_dim}")

        pairwise_logic_layer_kwargs = logic_layer_kwargs.copy()
        pairwise_logic_layer_kwargs['connections'] = 'pairwise'

        grouped_logic_layer_kwargs = logic_layer_kwargs.copy()
        grouped_logic_layer_kwargs['connections'] = 'unique_in_groups'
        grouped_logic_layer_kwargs['k'] = self.out_dim

        grouped_input_logic_layer_kwargs = logic_layer_kwargs.copy()
        grouped_input_logic_layer_kwargs['connections'] = 'unique_in_groups'
        grouped_input_logic_layer_kwargs['k'] = self.k

        random_logic_layer_kwargs = logic_layer_kwargs.copy()
        random_logic_layer_kwargs['connections'] = 'random_in_groups'
        random_logic_layer_kwargs['k'] = self.k
        random_logic_layer_kwargs['k_in'] = self.k_in

        #-----------------------------------------------
        # conv_compression.py에서 rebuild를 위한 추가 코드
        # ❗ [핵심] 수동으로 정의한 레이어들을 nn.Sequential로 묶습니다. 
        # 첫 번째 레이어: k_in이 None이거나 logic_in_features가 k_in으로 나누어 떨어지지 않으면 'unique' 사용
        # (이렇게 하면 k_in 제약 없이 레이어를 생성할 수 있고, 나중에 cascade를 교체할 때 원본 연결 정보를 유지할 수 있음)
        first_layer_kwargs = random_logic_layer_kwargs.copy()
        if compress:
            first_layer_kwargs['connections'] = 'unique'
            first_layer_kwargs.pop('k_in', None)  # k_in 제거 (unique 연결에서는 필요 없음)
            first_layer_kwargs.pop('k', None)  # k도 제거 (unique 연결에서는 필요 없음)
        #-----------------------------------------------
        self.cascade = nn.Sequential(
            LogicLayer(self.logic_in_features, intermediate_dim1, **first_layer_kwargs),
            LogicLayer(intermediate_dim1, intermediate_dim2, **pairwise_logic_layer_kwargs),
            LogicLayer(intermediate_dim2, self.out_dim, **pairwise_logic_layer_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        patches_flat = patches.permute(0, 2, 1).contiguous().view(-1, self.logic_in_features)

        # self.cascade를 통해 한 번에 forward pass를 수행합니다.
        logic_output = self.cascade(patches_flat)

        # out_dim이 없으면 cascade의 마지막 레이어 출력 차원 사용 (이전 버전 호환성)
        if not hasattr(self, 'out_dim'):
            self.out_dim = self.out_channels
        # out_dim이 없으면 cascade의 마지막 레이어 출력 차원 사용 (이전 버전 호환성)
        if not hasattr(self, 'in_dim'):
            self.in_dim = self.in_channels

        output_reshaped = logic_output.view(batch_size, patches.shape[2], self.out_dim).permute(0, 2, 1)

        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return F.fold(output_reshaped, output_size=(out_h, out_w), kernel_size=1)
    def extra_repr(self) -> str:
        return f'k_in={self.k_in}, k={self.k}'



class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # Reshape -> Transpose -> Flatten
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class FusedTreeConvORPool2dAB(nn.Module):
    def __init__(self,
                 weights_L, a_idx_L, b_idx_L,
                 leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
                 kernel_size, stride, padding,
                 groups, in_channels_per_group):
        super().__init__()
        self.k, self.s, self.p = kernel_size, stride, padding
        self.groups = groups
        self.icpg = in_channels_per_group

        # weights_L는 학습 파라미터: [L, Co, max_nodes, 16]
        self.weights_L = nn.Parameter(weights_L)  # ← 원본 cascade에서 추출 후 초기화
        # 인덱스들은 고정 버퍼
        self.register_buffer("a_idx_L", a_idx_L)
        self.register_buffer("b_idx_L", b_idx_L)
        self.register_buffer("leaf_ici", leaf_ici)
        self.register_buffer("leaf_ipx", leaf_ipx)
        self.register_buffer("leaf_ipy", leaf_ipy)
        self.register_buffer("nodes_per_level", nodes_per_level)

    def forward(self, x):
        if self.p > 0: x_pad = F.pad(x, (self.p, self.p, self.p, self.p))
        else:          x_pad = x

        B, C, Hp, Wp = x_pad.shape
        H, W = x.shape[-2:]
        Ho = (H + 2*self.p - self.k) // self.s + 1
        Wo = (W + 2*self.p - self.k) // self.s + 1
        out_h, out_w = Ho // 2, Wo // 2  # ORPool(2,2)

        return FusedLogicTreeOrPoolFunction.apply(
            x_pad, self.weights_L, self.a_idx_L, self.b_idx_L,
            self.leaf_ici, self.leaf_ipx, self.leaf_ipy, self.nodes_per_level,
            out_h, out_w, self.k, self.s, self.groups, self.icpg
        )


class FusedLogicTreeOrPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_padded, weights_L, a_idx_L, b_idx_L,
                leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
                out_h, out_w, kernel_size, stride, groups, in_channels_per_group):
        # y: [B, Co, out_h, out_w], winners: uint8 [B, Co, out_h, out_w]
        y, winners = difflogic_cuda.logictree_forward_ablevels(
            x_padded, weights_L, a_idx_L, b_idx_L,
            leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
            int(out_h), int(out_w), int(kernel_size), int(stride),
            int(groups), int(in_channels_per_group)
        )
        # 재계산 전략: x와 indices, weights 등만 저장 (중간 활성은 저장 안함)
        ctx.save_for_backward(x_padded, winners, weights_L, a_idx_L, b_idx_L,
                              leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level)
        ctx.meta = (out_h, out_w, kernel_size, stride, groups, in_channels_per_group)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_padded, winners, weights_L, a_idx_L, b_idx_L, \
        leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level = ctx.saved_tensors
        out_h, out_w, k, s, groups, icpg = ctx.meta

        grad_y = grad_y.contiguous()

        grad_x_padded, grad_w_L = difflogic_cuda.logictree_backward_ablevels(
            x_padded, grad_y, winners,
            weights_L, a_idx_L, b_idx_L,
            leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
            int(out_h), int(out_w),
            int(k), int(s), int(groups), int(icpg)
        )
        # inputs: (x_padded, weights_L, a_idx_L, b_idx_L, leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level, out_h, out_w, k, s, g, icpg)
        # grads for non-tensor ints are None:
        return (grad_x_padded, grad_w_L, None, None, None, None, None, None,
                None, None, None, None, None, None)





class FusedTreeConvLayer(nn.Module):
    """
    TreeConvLayer와 동일한 구조를 가지되,
    내부 3개의 LogicLayer를 TripleLogicFunction + fused CUDA kernel로
    한 번에 처리하는 버전.

    - 파라미터/연결 구조는 TreeConvLayer와 100% 동일
      (LogicLayer 3개: D0->4C->2C->C)
    - unfold/fold는 그대로 사용.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0,
                 k: int = None,
                 k_in: int = None,
                 logic_layer_kwargs: dict | None = None):
        super().__init__()
        self.in_dim = in_channels  # in_channels를 in_dim으로 통일
        self.out_dim = out_channels  # out_channels를 out_dim으로 통일
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.k            = k
        self.k_in         = k_in

        self.logic_in_features = self.in_dim * (self.kernel_size ** 2)
        intermediate_dim1 = self.out_dim * 4   # D1
        intermediate_dim2 = self.out_dim * 2   # D2

        print(f"[FusedTreeConvLayer] in={in_channels}, out={out_channels}, K={kernel_size}")
        print(f"  - cascade path: {self.logic_in_features} -> "
              f"{intermediate_dim1} -> {intermediate_dim2} -> {self.out_dim}")

        assert logic_layer_kwargs is not None, "logic_layer_kwargs must be provided"

        # --- kwargs 설정 (TreeConvLayer와 동일) ---
        pairwise_logic_layer_kwargs = logic_layer_kwargs.copy()
        pairwise_logic_layer_kwargs['connections'] = 'pairwise'

        random_logic_layer_kwargs = logic_layer_kwargs.copy()
        random_logic_layer_kwargs['connections'] = 'random_in_groups'
        random_logic_layer_kwargs['k'] = self.k
        random_logic_layer_kwargs['k_in'] = self.k_in

        # --- LogicLayer 3개를 직접 들고 있게 만들기 ---
        # 원래 TreeConvLayer: random_in_groups -> pairwise -> pairwise
        self.l1 = LogicLayer(self.logic_in_features,
                             intermediate_dim1,
                             **random_logic_layer_kwargs)
        self.l2 = LogicLayer(intermediate_dim1,
                             intermediate_dim2,
                             **pairwise_logic_layer_kwargs)
        self.l3 = LogicLayer(intermediate_dim2,
                             self.out_dim,
                             **pairwise_logic_layer_kwargs)

        # 디버깅/프린트용으로 기존과 동일한 cascade도 만들어 둠
        self.cascade = nn.Sequential(self.l1, self.l2, self.l3)

    def _get_logic_params(self, layer: LogicLayer):
        """
        LogicLayer로부터 (a_idx, b_idx, weight) 텐서를 추출.
        LogicLayer는 self.indices = (a, b) 튜플과 self.weights를 가짐.
        """
        a = layer.indices[0]  # shape [out_dim]
        b = layer.indices[1]  # shape [out_dim]
        w = layer.weights     # shape [out_dim, 16]
        return a, b, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Cin, H, W]
        반환: [B, Cout, H_out, W_out]
        """
        B, _, H, W = x.shape

        # 1) Unfold: [B, Cin*K*K, H_out*W_out]
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        # 2) [B, num_patches, D0]
        patches_flat = patches.permute(0, 2, 1).contiguous()
        B, num_patches, D0 = patches_flat.shape
        assert D0 == self.logic_in_features, \
            f"logic_in_features mismatch: {D0} vs {self.logic_in_features}"

        # 3) [N, D0] (N = B * num_patches)
        x_flat = patches_flat.view(-1, self.logic_in_features)
        if not x_flat.is_cuda:
            raise RuntimeError("FusedTreeConvLayer: input must be CUDA tensor.")
        x_flat = x_flat.contiguous()

        # 4) LogicLayer 3개 파라미터/인덱스 추출
        a1, b1, w1 = self._get_logic_params(self.l1)
        a2, b2, w2 = self._get_logic_params(self.l2)
        a3, b3, w3 = self._get_logic_params(self.l3)

        # 5) Triple fused CUDA 경로
        y_flat = TripleLogicFunction.apply(
            x_flat,
            a1, b1, w1,
            a2, b2, w2,
            a3, b3, w3,
            1,
        )  # [N, Cout]

        # 6) 다시 [B, num_patches, Cout] -> [B, Cout, num_patches]
        y = y_flat.view(B, num_patches, self.out_dim).permute(0, 2, 1)

        # 7) Fold back
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        return F.fold(y, output_size=(out_h, out_w), kernel_size=1)

    def extra_repr(self) -> str:
        return f'k_in={self.k_in}, k={self.k}'





class TripleLogicFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,
                a1, b1, w1,
                a2, b2, w2,
                a3, b3, w3,
                groups: int):
        y = difflogic_cuda.logic_triple_forward(
            x, a1, b1, w1,
               a2, b2, w2,
               a3, b3, w3,
               groups,
        )
        ctx.groups = groups
        ctx.save_for_backward(x, a1, b1, w1, a2, b2, w2, a3, b3, w3)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, a1, b1, w1,
            a2, b2, w2,
            a3, b3, w3) = ctx.saved_tensors
        groups = ctx.groups
        grad_y = grad_y.contiguous()
        gx, gw1, gw2, gw3 = difflogic_cuda.logic_triple_backward(
            x, grad_y,
            a1, b1, w1,
            a2, b2, w2,
            a3, b3, w3,
            groups,
        )
        return gx, None, None, gw1, None, None, gw2, None, None, gw3, None


class TripleFusedTreeConvLayer(nn.Module):
    """
    TripleLogicFunction을 사용하는 FusedTreeConvLayer의 대안 구현.
    
    TreeConvLayer와 동일하게 LogicLayer 객체를 사용하지만,
    forward에서 TripleLogicFunction을 호출하여 3개 레벨을 하나의 CUDA 커널로 처리합니다.
    
    Args:
        in_channels, out_channels, kernel_size, stride, padding:
            TreeConvLayer와 동일한 의미.
        k, k_in:
            첫 레벨 random_in_groups 연결용 (group 수/크기).
        device:
            파라미터/버퍼 device.
        logic_layer_kwargs:
            LogicLayer 생성에 사용.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 k: int = None,
                 k_in: int = None,
                 device: str = "cuda",
                 logic_layer_kwargs: dict | None = None):
        super().__init__()
        self.in_dim = in_channels  # in_channels를 in_dim으로 통일
        self.out_dim = out_channels  # out_channels를 out_dim으로 통일
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.k_in = k_in

        # unfold 한 patch의 feature dimension
        self.logic_in_features = self.in_dim * (self.kernel_size ** 2)

        # TreeConvLayer와 동일한 hidden dim 설정
        intermediate_dim1 = self.out_dim * 4
        intermediate_dim2 = self.out_dim * 2

        print(f"[TripleFusedTreeConvLayer] in={in_channels}, out={out_channels}, K={kernel_size}")
        print(f"  - Triple cascade path: {self.logic_in_features} -> "
              f"{intermediate_dim1} -> {intermediate_dim2} -> {self.out_dim}")

        assert logic_layer_kwargs is not None, "logic_layer_kwargs must be provided"

        # --- kwargs 설정 (TreeConvLayer와 동일) ---
        pairwise_logic_layer_kwargs = logic_layer_kwargs.copy()
        pairwise_logic_layer_kwargs['connections'] = 'pairwise'

        random_logic_layer_kwargs = logic_layer_kwargs.copy()
        random_logic_layer_kwargs['connections'] = 'random_in_groups'
        random_logic_layer_kwargs['k'] = self.k
        random_logic_layer_kwargs['k_in'] = self.k_in

        # --- LogicLayer 3개를 직접 들고 있게 만들기 ---
        # 원래 TreeConvLayer: random_in_groups -> pairwise -> pairwise
        self.l1 = LogicLayer(self.logic_in_features,
                             intermediate_dim1,
                             device=device,
                             **random_logic_layer_kwargs)
        self.l2 = LogicLayer(intermediate_dim1,
                             intermediate_dim2,
                             device=device,
                             **pairwise_logic_layer_kwargs)
        self.l3 = LogicLayer(intermediate_dim2,
                             self.out_dim,
                             device=device,
                             **pairwise_logic_layer_kwargs)

    def _get_logic_params(self, layer: LogicLayer):
        """
        LogicLayer로부터 (a_idx, b_idx, weight) 텐서를 추출.
        LogicLayer는 self.indices = (a, b) 튜플과 self.weights를 가짐.
        """
        a = layer.indices[0]  # shape [out_dim]
        b = layer.indices[1]  # shape [out_dim]
        w = layer.weights     # shape [out_dim, 16]
        return a, b, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Cin, H, W]
        반환: [B, Cout, H_out, W_out]
        """
        B, _, H, W = x.shape

        # 1) Unfold: [B, Cin*K*K, H_out*W_out]
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        # 2) [B, num_patches, D0]
        patches_flat = patches.permute(0, 2, 1).contiguous()
        B, num_patches, D0 = patches_flat.shape
        assert D0 == self.logic_in_features, \
            f"logic_in_features mismatch: {D0} vs {self.logic_in_features}"

        # 3) [N, D0] (N = B * num_patches)
        x_flat = patches_flat.view(-1, self.logic_in_features)
        if not x_flat.is_cuda:
            raise RuntimeError("TripleFusedTreeConvLayer: input must be CUDA tensor.")
        x_flat = x_flat.contiguous()

        # 4) LogicLayer 3개 파라미터/인덱스 추출
        a1, b1, w1 = self._get_logic_params(self.l1)
        a2, b2, w2 = self._get_logic_params(self.l2)
        a3, b3, w3 = self._get_logic_params(self.l3)

        # 5) Triple fused CUDA 경로 (groups=1: per-patch 처리)
        y_flat = TripleLogicFunction.apply(
            x_flat,
            a1, b1, w1,
            a2, b2, w2,
            a3, b3, w3,
            16,  # groups=1 for per-patch processing
        )  # [N, Cout]

        # 6) [B, num_patches, Cout] -> [B, Cout, num_patches]
        y_flat = y_flat.view(B, num_patches, self.out_dim)
        output_reshaped = y_flat.permute(0, 2, 1)

        # 7) Fold back to feature map
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        return F.fold(output_reshaped, output_size=(out_h, out_w), kernel_size=1)

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"k={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"logic_in_features={self.logic_in_features}")


class TripleFusedClassifierLayer(nn.Module):
    """
    Classifier의 LogicLayer 3개를 TripleLogicFunction으로 fuse한 버전.
    
    입력: [B, in_dim] (2D tensor)
    출력: [B, out_dim] (2D tensor)
    
    Args:
        in_dim: 첫 번째 LogicLayer의 입력 차원
        out_dim: 마지막 LogicLayer의 출력 차원
        intermediate_dims: 중간 차원들 (예: [dim1, dim2])
        device: 파라미터/버퍼 device
        logic_layer_kwargs: LogicLayer 생성에 사용
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 intermediate_dims: list[int],
                 device: str = "cuda",
                 logic_layer_kwargs: dict | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.intermediate_dims = intermediate_dims
        
        assert len(intermediate_dims) == 2, "intermediate_dims must have exactly 2 dimensions"
        intermediate_dim1 = intermediate_dims[0]
        intermediate_dim2 = intermediate_dims[1]
        
        print(f"[TripleFusedClassifierLayer] in={in_dim}, intermediate={intermediate_dims}, out={out_dim}")
        print(f"  - Triple cascade path: {in_dim} -> {intermediate_dim1} -> {intermediate_dim2} -> {out_dim}")
        
        assert logic_layer_kwargs is not None, "logic_layer_kwargs must be provided"
        
        # --- kwargs 설정 (pairwise 연결 사용) ---
        pairwise_logic_layer_kwargs = logic_layer_kwargs.copy()
        pairwise_logic_layer_kwargs['connections'] = 'pairwise'
        
        # LogicLayer 3개 생성
        self.l1 = LogicLayer(in_dim,
                             intermediate_dim1,
                             device=device,
                             **pairwise_logic_layer_kwargs)
        self.l2 = LogicLayer(intermediate_dim1,
                             intermediate_dim2,
                             device=device,
                             **pairwise_logic_layer_kwargs)
        self.l3 = LogicLayer(intermediate_dim2,
                             out_dim,
                             device=device,
                             **pairwise_logic_layer_kwargs)
    
    def _get_logic_params(self, layer: LogicLayer):
        """
        LogicLayer로부터 (a_idx, b_idx, weight) 텐서를 추출.
        LogicLayer는 self.indices = (a, b) 튜플과 self.weights를 가짐.
        """
        a = layer.indices[0]  # shape [out_dim]
        b = layer.indices[1]  # shape [out_dim]
        w = layer.weights     # shape [out_dim, 16]
        return a, b, w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim] (2D tensor)
        반환: [B, out_dim] (2D tensor)
        """
        if not x.is_cuda:
            raise RuntimeError("TripleFusedClassifierLayer: input must be CUDA tensor.")
        x = x.contiguous()
        
        # LogicLayer 3개 파라미터/인덱스 추출
        a1, b1, w1 = self._get_logic_params(self.l1)
        a2, b2, w2 = self._get_logic_params(self.l2)
        a3, b3, w3 = self._get_logic_params(self.l3)
        
        # Triple fused CUDA 경로 (groups=1: 2D 입력이므로 per-sample 처리)
        y = TripleLogicFunction.apply(
            x,
            a1, b1, w1,
            a2, b2, w2,
            a3, b3, w3,
            64,  # groups=1 for 2D input
        )  # [B, out_dim]
        
        return y
    
    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, intermediate_dims={self.intermediate_dims}, "
                f"out_dim={self.out_dim}")



class ChannelMaskLayer(nn.Module):
    """
    Channel mask layer that applies mask operations to channel dimension.
    Supports [B, C, H, W] (4D) and [B, C] (2D) tensors.
    mask_weights: [num_channels, 3] where:
    - [:, 0]: 0-tie weight (output = 0)
    - [:, 1]: 1-tie weight (output = 1)
    - [:, 2]: bypass weight (output = original)
    """
    def __init__(self, num_channels: int, device: str = 'cuda', frozen: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.frozen = frozen
        
        if frozen:
            # Frozen mode: use register_buffer (non-trainable)
            self.register_buffer('mask_weights', torch.ones(num_channels, 3, device=device))
        else:
            # Trainable mode: use Parameter
            self.mask_weights = torch.nn.parameter.Parameter(torch.ones(num_channels, 3, device=device))
        
        with torch.no_grad():
            self.mask_weights.fill_(0.0)
            self.mask_weights[:, 2] = 5.0  # bypass weight (default)
    
    def _apply_mask_1d(self, x):
        """
        Apply mask to 1D tensor [batch_size, num_channels].
        """
        # 기존 모델 호환성: mask_weights가 없으면 처리
        if not hasattr(self, 'mask_weights'):
            # 옛날 버전: mask_layer.mask_weights를 사용
            if hasattr(self, 'mask_layer') and hasattr(self.mask_layer, 'mask_weights'):
                # 옛날 버전 구조: mask_layer를 통해 접근
                mask_weights = self.mask_layer.mask_weights
            else:
                # 완전히 없으면 초기화 (옛날 버전에서는 항상 Parameter로 있었음)
                device = getattr(self, 'device', 'cuda')
                self.mask_weights = torch.nn.parameter.Parameter(torch.ones(self.num_channels, 3, device=device))
                with torch.no_grad():
                    self.mask_weights.fill_(0.0)
                    self.mask_weights[:, 2] = 5.0  # bypass weight (default)
                mask_weights = self.mask_weights
        else:
            mask_weights = self.mask_weights
        
        # 기존 모델 호환성을 위해 getattr 사용 (frozen 속성이 없으면 False로 간주)
        frozen = getattr(self, 'frozen', False)
        if frozen:
            # Frozen mode: use simple argmax (no softmax needed)
            mask_selection = torch.argmax(mask_weights, dim=-1)  # [num_channels]
            
            # Expand mask_selection to match x shape [batch_size, num_channels]
            batch_size = x.shape[0]
            mask_selection_expanded = mask_selection.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_channels]
            
            # Apply mask operations directly using torch.where
            # mask_selection: 0 (0-tie), 1 (1-tie), 2 (bypass)
            x_masked = torch.where(mask_selection_expanded == 0, torch.zeros_like(x),
                          torch.where(mask_selection_expanded == 1, torch.ones_like(x), x))
        else:
            # Training mode: use softmax + STE for gradient flow
            # Compute softmax probabilities
            mask_probs = torch.nn.functional.softmax(mask_weights, dim=-1)  # [num_channels, 3]
            
            # Get hard selection (argmax)
            mask_selection = torch.argmax(mask_weights, dim=-1)  # [num_channels]
            mask_hard = torch.nn.functional.one_hot(mask_selection, 3).to(x.dtype)  # [num_channels, 3]
            
            # Apply STE: use hard selection for forward, soft probabilities for backward
            mask_ste = mask_hard.detach() + (mask_probs - mask_probs.detach())  # [num_channels, 3]
            
            # Apply mask operations
            # mask_ste[:, 0]: 0-tie (output = 0)
            # mask_ste[:, 1]: 1-tie (output = 1)
            # mask_ste[:, 2]: bypass (output = original)
            
            # x shape: [batch_size, num_channels]
            # Expand mask_ste to [batch_size, num_channels, 3]
            batch_size = x.shape[0]
            mask_expanded = mask_ste.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_channels, 3]
            
            # Create operations: [0-tie, 1-tie, bypass]
            ops = torch.stack([
                torch.zeros_like(x),  # 0-tie
                torch.ones_like(x),   # 1-tie
                x                     # bypass
            ], dim=-1)  # [batch_size, num_channels, 3]
            
            # Apply mask: sum over operations weighted by mask_ste
            x_masked = torch.sum(ops * mask_expanded, dim=-1)  # [batch_size, num_channels]
        
        return x_masked
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] tensor (4D) or [B, C] tensor (2D)
        Returns:
            x_masked: same shape as input with channels masked
        """
        if x.ndim == 4:
            # 4D tensor: [B, C, H, W] (TreeConvLayer output)
            B, C, H, W = x.shape
            assert C == self.num_channels, f"Channel mismatch: {C} != {self.num_channels}"
            
            # Reshape to [B*H*W, C] for mask application
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
            
            # Apply mask
            x_masked = self._apply_mask_1d(x_reshaped)  # [B*H*W, C]
            
            # Reshape back to [B, C, H, W]
            x_masked = x_masked.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            
            return x_masked
        elif x.ndim == 2:
            # 2D tensor: [B, C] (LogicLayer output)
            B, C = x.shape
            assert C == self.num_channels, f"Channel mismatch: {C} != {self.num_channels}"
            
            # Apply mask directly
            x_masked = self._apply_mask_1d(x)  # [B, C]
            
            return x_masked
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}. Expected 2 or 4.")


class ResidualChannelMaskLayer(nn.Module):
    """
    Residual-style channel mask layer that applies mask operations with residual connections.
    Supports [B, C, H, W] (4D) and [B, C] (2D) tensors.
    mask_weights: [num_channels, 7] where:
    - [:, 0]: 0-tie weight (output = 0)
    - [:, 1]: 1-tie weight (output = 1)
    - [:, 2]: bypass-a weight (output = previous logic layer's input a)
    - [:, 3]: bypass-b weight (output = previous logic layer's input b)
    - [:, 4]: neg bypass-a weight (output = negation of previous logic layer's input a)
    - [:, 5]: neg bypass-b weight (output = negation of previous logic layer's input b)
    - [:, 6]: bypass weight (output = original)
    
    Requires reference to previous LogicLayer to access its input a and b.
    """
    def __init__(self, num_channels: int, prev_logic_layer: nn.Module = None, device: str = 'cuda', frozen: bool = False, include_tie: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.frozen = frozen
        self.include_tie = include_tie  # Whether to include 0-tie and 1-tie options
        self.prev_logic_layer = prev_logic_layer  # Reference to previous LogicLayer
        self.prev_input = None  # Store previous LogicLayer's input
        self._hook_handle = None
        
        # Number of mask options: 7 if include_tie, 5 otherwise
        self.num_mask_options = 7 if include_tie else 5
        # Bypass index: 6 if include_tie, 4 otherwise
        self.bypass_idx = 6 if include_tie else 4
        
        # Register forward hook on previous LogicLayer to capture its input
        if self.prev_logic_layer is not None:
            self._register_input_hook()
        
        if frozen:
            # Frozen mode: use register_buffer (non-trainable)
            self.register_buffer('mask_weights', torch.ones(num_channels, self.num_mask_options, device=device))
        else:
            # Trainable mode: use Parameter
            self.mask_weights = torch.nn.parameter.Parameter(torch.ones(num_channels, self.num_mask_options, device=device))
        
        with torch.no_grad():
            self.mask_weights.fill_(0.0)
            self.mask_weights[:, self.bypass_idx] = 5.0  # bypass weight (default)
    
    def _input_hook(self, module, input, output):
        """Hook function to capture previous LogicLayer's input."""
        # Store the input tensor (first element of input tuple)
        if len(input) > 0:
            self.prev_input = input[0].detach() if isinstance(input[0], torch.Tensor) else input[0]
    
    def _register_input_hook(self):
        """Register forward hook on previous LogicLayer to capture its input."""
        if self.prev_logic_layer is None:
            return
        
        self._hook_handle = self.prev_logic_layer.register_forward_hook(self._input_hook)
    
    def _remove_input_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def _get_prev_logic_inputs(self, prev_input):
        """
        Extract input a and b from previous LogicLayer's input using its indices.
        
        Args:
            prev_input: Previous LogicLayer's input tensor [B, in_dim] or [B*H*W, in_dim]
        
        Returns:
            prev_input_a: Input a tensor [B, num_channels] or [B*H*W, num_channels]
            prev_input_b: Input b tensor [B, num_channels] or [B*H*W, num_channels]
        """
        if self.prev_logic_layer is None or not hasattr(self.prev_logic_layer, 'indices'):
            # Fallback: return zeros if no previous layer
            return torch.zeros_like(prev_input[:, :self.num_channels]), torch.zeros_like(prev_input[:, :self.num_channels])
        
        a_indices = self.prev_logic_layer.indices[0]  # [out_dim] or [num_channels]
        b_indices = self.prev_logic_layer.indices[1]  # [out_dim] or [num_channels]
        
        # Ensure indices are within bounds
        in_dim = prev_input.shape[-1]
        a_indices = torch.clamp(a_indices, 0, in_dim - 1)
        b_indices = torch.clamp(b_indices, 0, in_dim - 1)
        
        # Extract input a and b
        # prev_input: [B, in_dim] or [B*H*W, in_dim]
        # We need to select channels based on indices, but only up to num_channels
        num_channels_to_use = min(self.num_channels, len(a_indices))
        prev_input_a = prev_input[:, a_indices[:num_channels_to_use]]  # [B, num_channels_to_use] or [B*H*W, num_channels_to_use]
        prev_input_b = prev_input[:, b_indices[:num_channels_to_use]]  # [B, num_channels_to_use] or [B*H*W, num_channels_to_use]
        
        # Pad if necessary
        if num_channels_to_use < self.num_channels:
            padding_a = torch.zeros(prev_input.shape[0], self.num_channels - num_channels_to_use, 
                                   device=prev_input.device, dtype=prev_input.dtype)
            padding_b = torch.zeros(prev_input.shape[0], self.num_channels - num_channels_to_use,
                                   device=prev_input.device, dtype=prev_input.dtype)
            prev_input_a = torch.cat([prev_input_a, padding_a], dim=-1)
            prev_input_b = torch.cat([prev_input_b, padding_b], dim=-1)
        
        return prev_input_a, prev_input_b
    
    def _apply_mask_1d(self, x, prev_input=None):
        """
        Apply mask to 1D tensor [batch_size, num_channels].
        
        Args:
            x: Current input tensor [batch_size, num_channels]
            prev_input: Previous LogicLayer's input tensor [batch_size, in_dim] (optional)
        """
        if not hasattr(self, 'mask_weights'):
            device = getattr(self, 'device', 'cuda')
            include_tie = getattr(self, 'include_tie', False)
            num_options = 7 if include_tie else 5
            bypass_idx = 6 if include_tie else 4
            self.mask_weights = torch.nn.parameter.Parameter(torch.ones(self.num_channels, num_options, device=device))
            with torch.no_grad():
                self.mask_weights.fill_(0.0)
                self.mask_weights[:, bypass_idx] = 5.0  # bypass weight (default)
        
        mask_weights = self.mask_weights
        
        # Get previous layer's input a and b if available
        if prev_input is not None:
            prev_input_a, prev_input_b = self._get_prev_logic_inputs(prev_input)
        else:
            # Fallback: use zeros if no previous input
            prev_input_a = torch.zeros_like(x)
            prev_input_b = torch.zeros_like(x)
        
        frozen = getattr(self, 'frozen', False)
        include_tie = getattr(self, 'include_tie', False)
        if frozen:
            # Frozen mode: use simple argmax (no softmax needed)
            mask_selection = torch.argmax(mask_weights, dim=-1)  # [num_channels]
            
            # Expand mask_selection to match x shape [batch_size, num_channels]
            batch_size = x.shape[0]
            mask_selection_expanded = mask_selection.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_channels]
            
            # Apply mask operations directly using torch.where
            if include_tie:
                # mask_selection: 0 (0-tie), 1 (1-tie), 2 (bypass-a), 3 (bypass-b), 4 (neg bypass-a), 5 (neg bypass-b), 6 (bypass)
                x_masked = torch.where(mask_selection_expanded == 0, torch.zeros_like(x),  # 0-tie
                              torch.where(mask_selection_expanded == 1, torch.ones_like(x),  # 1-tie
                              torch.where(mask_selection_expanded == 2, prev_input_a,  # bypass-a
                              torch.where(mask_selection_expanded == 3, prev_input_b,  # bypass-b
                              torch.where(mask_selection_expanded == 4, 1.0 - prev_input_a,  # neg bypass-a
                              torch.where(mask_selection_expanded == 5, 1.0 - prev_input_b, x))))))  # neg bypass-b, bypass
            else:
                # mask_selection: 0 (bypass-a), 1 (bypass-b), 2 (neg bypass-a), 3 (neg bypass-b), 4 (bypass)
                x_masked = torch.where(mask_selection_expanded == 0, prev_input_a,  # bypass-a
                              torch.where(mask_selection_expanded == 1, prev_input_b,  # bypass-b
                              torch.where(mask_selection_expanded == 2, 1.0 - prev_input_a,  # neg bypass-a
                              torch.where(mask_selection_expanded == 3, 1.0 - prev_input_b, x))))  # neg bypass-b, bypass
        else:
            # Training mode: use softmax + STE for gradient flow
            # Compute softmax probabilities
            num_options = getattr(self, 'num_mask_options', 7 if include_tie else 5)
            mask_probs = torch.nn.functional.softmax(mask_weights, dim=-1)  # [num_channels, num_options]
            
            # Get hard selection (argmax)
            mask_selection = torch.argmax(mask_weights, dim=-1)  # [num_channels]
            mask_hard = torch.nn.functional.one_hot(mask_selection, num_options).to(x.dtype)  # [num_channels, num_options]
            
            # Apply STE: use hard selection for forward, soft probabilities for backward
            mask_ste = mask_hard.detach() + (mask_probs - mask_probs.detach())  # [num_channels, num_options]
            
            # x shape: [batch_size, num_channels]
            # Expand mask_ste to [batch_size, num_channels, num_options]
            batch_size = x.shape[0]
            mask_expanded = mask_ste.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_channels, num_options]
            
            # Create operations based on include_tie
            if include_tie:
                # [0-tie, 1-tie, bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass]
                ops = torch.stack([
                    torch.zeros_like(x),      # 0: 0-tie
                    torch.ones_like(x),       # 1: 1-tie
                    prev_input_a,             # 2: bypass-a
                    prev_input_b,             # 3: bypass-b
                    1.0 - prev_input_a,       # 4: neg bypass-a
                    1.0 - prev_input_b,       # 5: neg bypass-b
                    x                         # 6: bypass
                ], dim=-1)  # [batch_size, num_channels, 7]
            else:
                # [bypass-a, bypass-b, neg bypass-a, neg bypass-b, bypass]
                ops = torch.stack([
                    prev_input_a,             # 0: bypass-a
                    prev_input_b,             # 1: bypass-b
                    1.0 - prev_input_a,       # 2: neg bypass-a
                    1.0 - prev_input_b,       # 3: neg bypass-b
                    x                         # 4: bypass
                ], dim=-1)  # [batch_size, num_channels, 5]
            
            # Apply mask: sum over operations weighted by mask_ste
            x_masked = torch.sum(ops * mask_expanded, dim=-1)  # [batch_size, num_channels]
        
        return x_masked
    
    def forward(self, x, prev_input=None):
        """
        Args:
            x: [B, C, H, W] tensor (4D) or [B, C] tensor (2D)
            prev_input: Previous LogicLayer's input tensor (optional, for residual connections)
                       If None, uses self.prev_input captured by forward hook
        Returns:
            x_masked: same shape as input with channels masked
        """
        # Use stored prev_input if not provided
        if prev_input is None:
            prev_input = self.prev_input
        
        if x.ndim == 4:
            # 4D tensor: [B, C, H, W] (TreeConvLayer output)
            B, C, H, W = x.shape
            assert C == self.num_channels, f"Channel mismatch: {C} != {self.num_channels}"
            
            # Reshape to [B*H*W, C] for mask application
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
            
            # Reshape prev_input if provided
            if prev_input is not None and prev_input.ndim == 4:
                prev_input_reshaped = prev_input.permute(0, 2, 3, 1).contiguous().view(-1, prev_input.shape[1])  # [B*H*W, in_dim]
            else:
                prev_input_reshaped = prev_input
            
            # Apply mask
            x_masked = self._apply_mask_1d(x_reshaped, prev_input_reshaped)  # [B*H*W, C]
            
            # Reshape back to [B, C, H, W]
            x_masked = x_masked.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            
            return x_masked
        elif x.ndim == 2:
            # 2D tensor: [B, C] (LogicLayer output)
            B, C = x.shape
            assert C == self.num_channels, f"Channel mismatch: {C} != {self.num_channels}"
            
            # Apply mask directly
            x_masked = self._apply_mask_1d(x, prev_input)  # [B, C]
            
            return x_masked
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}. Expected 2 or 4.")


import math
import torch, torch.nn as nn
from typing import Callable, Union, List
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Sequence
# birel.utils는 순환 import를 피하기 위해 함수 내부에서 lazy import
from difflogic.functional import bin_op_s, get_unique_connections         # 이미 주신 모듈 그대로 import
from difflogic.difflogic import LogicLayer, GroupSum      # LogicLayer 코드 그대로 사용
try:
    import difflogic_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class ZeroPad1d(nn.Module):
    """
    입력 텐서 (B, N)에 대해 뒤쪽에 pad_right 개의 0을 붙여 (B, N+pad) 로 반환.
    """
    def __init__(self, pad_right: int):
        super().__init__()
        self.pad_right = pad_right

    def forward(self, x: torch.Tensor):
        if self.pad_right == 0:
            return x
        zeros = x.new_zeros(x.size(0), self.pad_right)
        return torch.cat([x, zeros], dim=1)



class BinaryNumberLayer(nn.Module):
    """
    입력:   (..., n_bits)  – 이미 STE 처리된 0/1(혹은 soft 0-1) 비트
    출력:   (...,)         – 가중 합산된 정수값 (float tensor)
    """
    def __init__(self, n_bits: int):
        super().__init__()
        self.register_buffer(
            "weights", 2 ** torch.arange(n_bits, dtype=torch.float32)
        )
        self.n_bits = n_bits

    def forward(self, bits: torch.Tensor):
        assert bits.shape[-1] == self.weights.numel(), \
            "마지막 차원이 n_bits와 같아야 합니다."
        code = (bits * self.weights).sum(dim=-1)
        return code


class BinaryNumberLayerSTE(nn.Module):
    def __init__(self, n_bits, tau=1.0):
        super().__init__()
        self.register_buffer("weights", 2 ** torch.arange(n_bits).float())
        self.tau = tau         # 필요하면 temperature 조절

    def forward(self, x):      # x: (..., n_bits)  – 실수 logits
        probs = torch.sigmoid(x / self.tau)      # 여전히 sigmoid는 gradient용!
        bits_hard = (probs > 0.5).float()
        bits = bits_hard.detach() - probs.detach() + probs  # STE
        code = (bits * self.weights).sum(dim=-1)
        return code

class RegressionLayer(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = True
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.device = device
        self.tau = 0.0
        self.linear = nn.Linear(1, 1)
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary

    # ------------------------- forward -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2

        *batch_dims, feats = x.shape
        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)
        #xg = x.reshape(*batch_dims, self.k, g)

        if self.use_ternary:
            xg = x.reshape(*batch_dims, self.k, g//2, 2)
            xg =(xg[..., 0] - xg[..., 1]).view(*batch_dims, self.k, g//2)
            g = g/2
        else:
            xg = x.reshape(*batch_dims, self.k, g)

        s = xg.sum(dim=-1)
        #print(x.shape)
        #logits = s/(g)
        #logits = (s - g / 2) / (g/2) # -1 ~ 1
        logits = s/g #(s - g / 2) / (g/2) # -1 ~ 1
        #print(logits)
        #logits = (s - g / 2) / (g/2)  # -1 ~ 1
        return self.linear(logits.view(-1, 1)).view(-1)

    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'


class MultiOutputRegressionLayer(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        ste: bool = False,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = True
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        ste   : bool  – Straight-Through Estimator 사용 여부
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.ste    = ste
        self.device = device
        self.tau = tau
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary
        self.voter_in = None
    # ------------------------- forward -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m

        *batch_dims, feats = x.shape

        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)

        xg = x.reshape(*batch_dims, self.k, g)

        # 다수결 거리: s - g/2  (양수 → 과반)
        #print(xg.sum(dim=-1))
        s = xg.sum(dim=-1)

        #print(x.shape)
        logits = (s - g/2) / math.sqrt((g/4))  # -1 ~ 1
        # Gradient는 p를 따르고, forward는 y_hard를 사용
        return logits
    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'



import torch
import torch.nn as nn

class Float16RegressionLayer(nn.Module):
    """
    입력 x : (..., 16·k) ― 마지막 차원을 16씩 끊어
               [ sign(1) | exponent(5) | mantissa(10) ] … k개
               값 범위는 임의의 실수(0‥1 확률·출력 등)라고 가정
    출력    : (..., k)   ― IEEE-754 half-precision 해석 결과 (float32 텐서)
    """
    def __init__(self, *, device="cuda"):
        super().__init__()
        self.device = device

        # 5-bit, 10-bit 가중치 벡터(2^i)
        self.alpha = nn.Parameter(torch.tensor(1.0, device=device))
        self.beta = nn.Parameter(torch.tensor(1.0, device=device))
        self.register_buffer("exp_w", 2 ** torch.arange(5,  dtype=torch.float32, device=device))
        self.register_buffer("man_w", 2 ** torch.arange(10, dtype=torch.float32, device=device))

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, feat = x.shape
        assert feat % 16 == 0, f"feature dim {feat} not multiple of 16"
        k = feat // 16

        # (..., k, 16) 로 재배열
        bits = x.view(*batch, 16, k)

        sign_grp = bits[..., 0, :]              # (..., k)
        exp_grp  = bits[..., 1:  6, :]          # (..., k, 5)
        man_grp  = bits[..., 6: 16, :]          # (..., k,10)

        # ─── 1) sign : [0,1] → (-1, +1) 연속 값
        #     평균 0 → +1, 1 → -1   (확률이면 ±중간값)
        sign_prob   = sign_grp.mean(-1)                  # (..., k)
        sign_factor = 1.0 - 2.0 * sign_prob              # +1 … -1

        # ─── 2) exponent : 합계 → [0‥31] 로 스케일 (5-bit 최대)
        exp_sum = exp_grp.sum(-1)                        # 최대값 = 5
        #exp_val = torch.clamp(exp_sum * (31.0 / k), 0.0, 31.0)
        exp_val = torch.clamp(exp_sum * self.alpha, 0.0, 31.0)

        # ─── 3) mantissa : 합계 → [0‥1023] 로 스케일 (10-bit 최대)
        man_sum = man_grp.sum(-1)                        # 최대값 = 10
        man_val = torch.clamp(man_sum * self.beta, 0.0, 1023.0)

        # ─── 4) float16 해석 (연속 버전) ───────────────
        bias  = 15.0
        two   = torch.tensor(2.0, device=x.device)

        normal     = (1.0 + man_val / 1024.0) * two.pow(exp_val - bias)
        subnormal  = (man_val / 1024.0)       * two.pow(1.0 - bias)
        value_mag  = torch.where(exp_val < 1e-6, subnormal, normal)   # exp==0 → subnormal
        value_mag  = torch.where(exp_val >= 31.0 - 1e-6, torch.full_like(value_mag, float('inf')), value_mag)

        return sign_factor * value_mag         # shape (..., k)

    # ----------------------------------------------------------
    def extra_repr(self):
        return "Float16-Regression (1:5:10)"

class DiscreteRegressionLayer(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = True
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.device = device
        self.tau = 0.0
        self.linear = nn.Linear(1, 1)
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary

    # ------------------------- forward -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2

        *batch_dims, feats = x.shape
        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)
        #xg = x.reshape(*batch_dims, self.k, g)

        xg = x.reshape(*batch_dims, self.k, g)

        s = xg.sum(dim=-1)
        logits = (s/g)*30  #0~1


        hard_logit = torch.round(logits)
        #print(logits)
        #logits = (s - g / 2) / (g/2)  # -1 ~ 1
        return (hard_logit.detach() + logits - logits.detach()).view(-1)
        #return self.linear(a.view(-1, 1)).view(-1)

    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'


class GroupBinaryLayer(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        ste: bool = False,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = True
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        ste   : bool  – Straight-Through Estimator 사용 여부
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.ste    = ste
        self.device = device
        self.tau = 0.0
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary
    # ------------------------- forward -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m

        *batch_dims, feats = x.shape

        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)

        w = (2**(-torch.arange(g//2, dtype=torch.float32, device='cuda')-1)).view(1, 1, -1)
        xg = x.reshape(*batch_dims, self.k, g)*torch.cat([w, w], dim=-1)

        # 다수결 거리: s - g/2  (양수 → 과반)
        #print(xg.sum(dim=-1))
        s = xg.sum(dim=-1)
        #print(x.shape)
        #logits = (s - g/2) / math.sqrt((g/4))  # -1 ~ 1
        logits = (s - 0.5)*2
        if self.training:
            if self.tau == 0.0:
                p = torch.sigmoid(logits)                               # (..., k)
            else:
                p = torch.sigmoid(logits/self.tau)                               # (..., k)
            #if self.ste:
        else:
            p = (logits>=0).float()
            return p
            #else:
            #    p = torch.sigmoid(logits*math.sqrt(g))
            #return p
        #p = logits
        #self.tau = 2.0/g 

        if not self.ste:
            return p                                            # soft 출력

        # ---------- STE (hard 0/1 + straight-through gradient) ----------
        y_hard = (p > 0.5).float()
        # Gradient는 p를 따르고, forward는 y_hard를 사용
        return y_hard.detach() - p.detach() + p

    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'

class StepSTEClip(torch.autograd.Function):
    """
    forward : hard step 0/1
    backward: grad = upstream_grad  if |x| ≤ 3
              grad = 0             if |x| >  3
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= 3).float()       # 1 inside [-3,3], else 0
        return grad_output * mask           # element-wise
        



class VotingLayer(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        ste: bool = False,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = True,
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        ste   : bool  – Straight-Through Estimator 사용 여부
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.ste    = ste
        self.device = device
        self.tau = tau
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary
        self.voter_in = None
    # ------------------------- forward -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m

        *batch_dims, feats = x.shape

        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)

        if self.use_ternary:
            xg = x.reshape(*batch_dims, self.k, g//2, 2)
            xg =(xg[..., 0] - xg[..., 1]).view(*batch_dims, self.k, g//2)
            g = g/2
        else:
            xg = x.reshape(*batch_dims, self.k, g)

        # 다수결 거리: s - g/2  (양수 → 과반)
        #print(xg.sum(dim=-1))
        if not self.training:
            self.voter_in = xg
        s = xg.sum(dim=-1)

        #print(x.shape)
        if self.use_ternary:
            logits = (s) / math.sqrt((g/2))  # -1 ~ 1
        else:
            logits = (s - g/2) / math.sqrt((g/4))  # -1 ~ 1
        if self.training:
            p = torch.sigmoid(logits/self.tau)                               # (..., k)
        else:
            p = (logits>=0).float()
            #save_corr_plot_and_min_subset(A, p)
            return p
            #else:
            #    p = torch.sigmoid(logits*math.sqrt(g))
            #return p
        #p = logits
        #self.tau = 2.0/g 

        if not self.ste:
            return p                                            # soft 출력

        # ---------- STE (hard 0/1 + straight-through gradient) ----------
        y_hard = (p > 0.5).float()
        # Gradient는 p를 따르고, forward는 y_hard를 사용
        return y_hard.detach() - p.detach() + p

    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'


# birel/model.py  ─ 기존 정의를 전부 대체하거나 patch
class WeightedVotingLayer(nn.Module):
    def __init__(self,
                 k: int,
                 in_dim: int | None = None,      # ← None 허용
                 tau: float = 1.0,
                 ste: bool = True,
                 *,
                 reg_type: str = "l1",
                 init: str = "ones",
                 w_max: float | None = None,
                 device: str | torch.device = "cuda"):
        super().__init__()
        self.k        = k
        self.tau      = tau
        self.ste      = ste
        self.reg_type = reg_type
        self.init     = init
        self.w_max    = w_max
        self.device   = device

        # 아직 in_dim 을 모를 수도 있으므로…
        self.in_dim: int | None = None
        self.g      : int | None = None
        self.register_parameter("weight_raw", None)
        self.voter_in = None      # ← 추가


        # in_dim 이 주어졌다면 즉시 초기화
        if in_dim is not None:
            self._init_weights(in_dim)

    # ────────────────────────────────────────────────────────
    def _init_weights(self, in_dim: int):
        assert in_dim % self.k == 0, \
            f"in_dim({in_dim}) must be multiple of k({self.k})"
        self.in_dim = in_dim
        self.g      = in_dim // self.k

        if self.init == "ones":
            w0 = torch.ones(self.k, self.g, device=self.device)
        else:                     # "randn"
            w0 = torch.randn(self.k, self.g, device=self.device)

        # 파라미터 등록(처음 한 번만)
        self.weight_raw = nn.Parameter(w0, requires_grad=True)

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _ste_round(w: torch.Tensor) -> torch.Tensor:
        return (w.round() - w).detach() + w

    # ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lazy init ─ 입력 차원으로 가중치 초기화
        if self.weight_raw is None:
            self._init_weights(x.size(-1))

        # ① non-negative, optional max-clamp
        with torch.no_grad():
            self.weight_raw.clamp_(min=0)
            if self.w_max is not None:
                self.weight_raw.clamp_(max=self.w_max)

        w_q = self._ste_round(self.weight_raw)          # (k,g)
        xg  = x.view(*x.shape[:-1], self.k, self.g)      # (...,k,g)
        if not self.training:
            self.voter_in = xg

        s   = (xg * w_q).sum(-1)                        # (...,k)

        threshold = w_q.sum(-1) / 2                     # (k,)
        logits    = s - threshold

        p = torch.sigmoid(logits / self.tau)

        if not self.ste:
            return p if self.training else (p > 0.5).float()

        y_hard = (p > 0.5).float()
        return y_hard.detach() - p.detach() + p

    # ────────────────────────────────────────────────────────
    def reg_loss(self) -> torch.Tensor:
        if self.weight_raw is None:
            return torch.tensor(0., device=self.device)
        if   self.reg_type == "l1": return self.weight_raw.abs().sum()
        elif self.reg_type == "l2": return (self.weight_raw**2).sum()
        else:                       return torch.tensor(0., device=self.device)

    def extra_repr(self):
        g = "?" if self.g is None else self.g
        return (f'k={self.k}, g={g}, tau={self.tau}, ste={self.ste}, '
                f'reg_type={self.reg_type}')



class PrunedVotingLayer(nn.Module):
    """
    corr_plot 분석으로 얻은 최소 입력 subset 정보만을 사용하여 투표를 수행하는 레이어.
    info 객체를 받아, 각 출력 채널에 필요한 입력만 'gather'하여 계산한다.
    """
    def __init__(self, k: int, info: dict, device="cuda"):
        super().__init__()
        self.k = k
        self.info = info # 채널별 subset 정보가 담긴 딕셔너리
        self.device = device
        # 분석 함수와의 호환성을 위해 out_dim 속성 추가
        self.out_dim = k
        # in_dim은 이전 레이어의 out_dim과 같아야 하므로, 모델 생성 시 동적으로 설정
        self.in_dim = -1 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x의 shape: (batch_size, in_dim)
        batch_size = x.shape[0]
        outputs = []

        for i in range(self.k):
            if i in self.info:
                # info 객체에서 해당 채널의 subset 정보를 가져옴
                _k, subset, sign = self.info[i]
                
                subset_indices = torch.tensor(subset, dtype=torch.long, device=self.device)
                
                # torch.gather를 사용해 필요한 입력(subset)만 선택
                selected_inputs = torch.gather(x, 1, subset_indices.expand(batch_size, -1))
                
                # 선택된 입력들로만 투표 수행
                s = selected_inputs.sum(dim=-1)
                threshold = len(subset) / 2.0
                
                # sign에 따라 로짓 계산
                logits = (s - threshold) if sign == '+' else (threshold - s)
                
                # 간단한 시그모이드 처리
                p = torch.sigmoid(logits)
                outputs.append(p)
            else:
                # info에 없는 채널은 0으로 처리
                outputs.append(torch.zeros(batch_size, device=self.device))
        
        return torch.stack(outputs, dim=1)





class Binary2Real(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        ste: bool = False,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0,
        use_ternary: bool = False 
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        ste   : bool  – Straight-Through Estimator 사용 여부
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.ste    = ste
        self.device = device
        self.tau = tau
        self.noise_prob = noise_prob
        self.use_ternary = use_ternary
        self.voter_in = None
        #self.linear = nn.Linear(k, k)
    # ------------------------- forward -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m

        *batch_dims, feats = x.shape

        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)

        if self.use_ternary:
            xg = x.reshape(*batch_dims, self.k, g//2, 2)
            xg =(xg[..., 0] - xg[..., 1]).view(*batch_dims, self.k, g//2)
            g = g/2
        else:
            xg = x.reshape(*batch_dims, self.k, g)

        # 다수결 거리: s - g/2  (양수 → 과반)
        #print(xg.sum(dim=-1))
        if not self.training:
            self.voter_in = xg
        s = xg.sum(dim=-1)

        #print(x.shape)
        if self.use_ternary:
            logits = (s) / math.sqrt((g/2))  # -1 ~ 1
        else:
            logits = (s - g/2) / math.sqrt((g/4))  # -1 ~ 1

        return logits
        #return self.linear(logits) #.view(-1)
        #if self.training:
        #    p = torch.sigmoid(logits/self.tau)                               # (..., k)
        #else:
        #    p = (logits>=0).float()
        #    #save_corr_plot_and_min_subset(A, p)
        #    return p
            #else:
            #    p = torch.sigmoid(logits*math.sqrt(g))
            #return p
        #p = logits
        #self.tau = 2.0/g 

        if not self.ste:
            return p                                            # soft 출력

        # ---------- STE (hard 0/1 + straight-through gradient) ----------
        y_hard = (p > 0.5).float()
        # Gradient는 p를 따르고, forward는 y_hard를 사용
        return y_hard.detach() - p.detach() + p

    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'


class Binarize(nn.Module):
    def __init__(self, tau=1.0):
        super(Binarize, self).__init__()
        self.tau = tau

    def forward(self, x):
        if self.training:
            p = torch.sigmoid(x/self.tau)                               # (..., k)
            return p
        else:
            p = (x>=0).float()
            return p


 
class OutputSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        smoothing: 정답 레이블을 얼마나 부드럽게 만들지 (예: 0.1이면 1 → 0.9, 0 → 0.1)
        """
        super(OutputSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs):
        """
        inputs: 예측값 (logits 또는 probabilities)
        targets: ground truth (0 또는 1)

        inputs: shape (batch_size,) or (batch_size, 1)
        targets: shape (batch_size,) or (batch_size, 1)
        """
        if self.training:
            # label smoothing 적용
            smooth_output = (0.6)*inputs+(0.4)*(1-inputs)
        else:
            # 평가 모드에서는 smoothing 없이 원래 target 사용
            smooth_output = inputs 

        return smooth_output

class VotingLayerForNonSTEInput(nn.Module):
    """
    Soft (or STE-hard) majority-voting layer.

    마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹 내부 합이 과반수(> g/2)인지 판정한다.

    * `ste=False`  : 부드러운 확률값   (0‥1, gradient = logistic)
    * `ste=True`   : 0/1 하드 스위치   (forward 0/1,   gradient = straight-through)

    출력 차원은 (..., k), 값 범위는 [0, 1] 또는 {0, 1}.
    """
    def __init__(
        self,
        k: int,
        tau: float = 1.0,
        *,
        ste: bool = False,
        device: str | torch.device = "cuda",
        noise_prob: float = 0.0
    ):
        """
        Args
        ----
        k     : int   – 의도한 출력 수(예: 클래스 수)
        tau   : float – softmax/로지스틱 온도. 작을수록 0/1에 수렴
        ste   : bool  – Straight-Through Estimator 사용 여부
        device: 연산 디바이스
        """
        super().__init__()
        self.k      = k
        self.tau    = tau
        self.ste    = ste
        self.device = device
        self.tau = 0.0
        self.noise_prob = noise_prob
    # ------------------------- forward -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m

        *batch_dims, feats = x.shape

        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                                     # 그룹 크기

        # (..., k, g)
        xg = x.reshape(*batch_dims, self.k, g)
        #xg = x.reshape(*batch_dims, self.k, g//2, 2)
        #xg =(xg[..., 0] - xg[..., 1]).view(*batch_dims, self.k, g//2)
        #g = g/2

        # 다수결 거리: s - g/2  (양수 → 과반)
        #print(xg.sum(dim=-1))
        s = xg.sum(dim=-1)

        s = s/g #+ 0.5 # 0 ~ 1. 확률로 해석 가능j
        #print(x.shape)

        #p = torch.sigmoid(logits*(g/2))                               # (..., k)
        if self.training:
            p = s
        else:
            p = (s>=0.5).float()
        return p
  
    # ----------------------- extra_repr -----------------------
    def extra_repr(self) -> str:
        return f'k={self.k}, tau={self.tau}, ste={self.ste}'



import math, torch
import torch.nn as nn


import math, torch
import torch.nn as nn



class ConditionalBernoulliNoise(nn.Module):
    """
    입력: 0‥1 실수 텐서
    출력: {0,1} 또는 원본 값 (조건부로 변형)
    
    y = g * ber(x) + (1-g) * x
        g ~ Bernoulli(noise_prob)
        ber(x) ~ Bernoulli(x)
    
    Args
    ----
    noise_prob   : ρ   – 노이즈를 넣을 확률 (0 ⇒ 패스스루, 1 ⇒ 항상 샘플링)
    ste          : bool – True 면 straight-through gradient
    training_only: bool – True 면 eval() 모드에서 노이즈 없이 통과
    """
    def __init__(self,
                 noise_prob: float = 0.1):
        super().__init__()
        self.register_buffer('noise_prob', torch.tensor(noise_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # g: 노이즈 여부 게이트
            g = torch.bernoulli(torch.full_like(x, self.noise_prob))
            # 노이즈가 켜진 위치에서만 새 샘플링
            noisy = torch.bernoulli(x)
            y = g * noisy +  x * (1.0-g)
            y = x  # 평가 단계: 원본 값 유지
        return y


class BitFlip(nn.Module):
    """
    입력: 0‥1 실수 텐서
    출력: {0,1} 또는 원본 값 (조건부로 변형)
    
    y = g * ber(x) + (1-g) * x
        g ~ Bernoulli(noise_prob)
        ber(x) ~ Bernoulli(x)
    
    Args
    ----
    noise_prob   : ρ   – 노이즈를 넣을 확률 (0 ⇒ 패스스루, 1 ⇒ 항상 샘플링)
    ste          : bool – True 면 straight-through gradient
    training_only: bool – True 면 eval() 모드에서 노이즈 없이 통과
    """
    def __init__(self,
                 noise_prob: float = 0.1):
        super().__init__()
        self.noise_prob = noise_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            y = (x + m) % 2
        else:
            y = x

        return y




class WeightedTopKVote(nn.Module):
    """
    Weighted + Top-k voting layer.
    입력 마지막 차원을 k개의 그룹으로 나눈 뒤,
    그룹마다 softmax 가중치로 합산하되 상위 k_keep개만 살린다.

    Parameters
    ----------
    k        : 출력 클래스 수 (= 그룹 수)
    tau      : softmax temperature
    k_keep   : 테스트 단계에서 남길 게이트 수 (1‥g)
    ste      : 학습 중 straight-through hard top-k 사용 여부
    """
    def __init__(self,
                 k: int,
                 tau: float = 1.0,
                 *,
                 k_keep: int = 2,
                 ste: bool = True,
                 device: str | torch.device = "cuda"):
        super().__init__()
        self.k       = k
        self.tau     = tau
        self.k_keep  = k_keep
        self.ste     = ste
        self.device  = device
        # α 는 forward 시 in_dim 에 따라 초기화되므로 일단 None
        #self.register_parameter("alpha", None)
        #self.alpha = nn.Parameter(torch.zeros(self.k, 100, device=self.device))

    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, feats = x.shape
        assert feats % self.k == 0, (x.shape, self.k)
        g = feats // self.k                              # 그룹 크기

        # (B…, k, g)
        xg = x.view(*batch_dims, self.k, g)

        # α 파라미터를 첫 forward 에서만 생성

        self.tau = 0.05

        # -------- 1) softmax 가중치 ----------
        w_soft = torch.softmax(self.alpha / self.tau, dim=-1)  # (k,g)

        if self.training:                                  # ─ Train mode ─
            if self.ste:
                # hard top-k (1/k_keep 로 정규화) + straight-through
                topk_idx = w_soft.topk(self.k_keep, dim=-1).indices
                w_hard   = torch.zeros_like(w_soft)
                #w_hard.scatter_(-1, topk_idx, 1.0 / self.k_keep)
                w_hard.scatter_(-1, topk_idx, 1.0)
                w = (w_hard - w_soft).detach() + w_soft     # STE
            else:
                w = w_soft                                  # 부드러운 가중
        else:                                               # ─ Eval mode ─
            w = torch.zeros_like(w_soft)
            topk_idx = self.alpha.topk(self.k_keep, dim=-1).indices
            #w.scatter_(-1, topk_idx, 1.0 / self.k_keep)     # 진짜 hard top-k
            w.scatter_(-1, topk_idx, 1.0 )     # 진짜 hard top-k

        # -------- 2) 가중 합산 ----------
        y = (xg * w).sum(-1) #                                # (..., k)
        if self.ste: # 0 ~ k_keep
            p = torch.sigmoid((y - self.k_keep/2))
        else: # 0~1
            assert False
            p = torch.sigmoid((y - 0.5)*(self.k_keep/2))
        return p

    def entropy_penalty(self) -> torch.Tensor:
        """
        평균 엔트로피  H_c  =  -Σ_j w_cj log w_cj
        (softmax 구간에서만 계산; hard-top-k일 때는 0)
        축소하려면   loss += β * entropy_penalty
        """
        # 학습 중이 아닐 때 하드게이트면 패널티 0
        if not self.training:
            return torch.zeros(1, device=self.alpha.device)

        w_soft = torch.softmax(self.alpha / self.tau, dim=-1)   # (k, g)
        ent    = -(w_soft * w_soft.clamp_min(1e-9).log()).sum(-1)  # (k,)
        return ent.mean()           

    # ----------------------- extra_repr -------------------------
    def extra_repr(self):
        return f'k={self.k}, tau={self.tau}, k_keep={self.k_keep}, ste={self.ste}'



def quota_by_prob_torch(prob: torch.Tensor,
                        K: int,
                        g: int,
                        q_min: int = 5) -> torch.Tensor:
    """
    prob  : (k,)  softmax 확률 (Σ=1, dtype=float, device任意)
    K     : 총 quota
    g     : 그룹별 quota 상한
    q_min : 그룹별 quota 하한 (default 5)

    반환  : (k,) int64  ─  조건
            • q_min ≤ k_c[i] ≤ g
            • Σ k_c = K
            • k_c 가 확률에 비례하도록 L¹ 오차 최소
    """
    k = prob.numel()
    if K < k * q_min or K > k * g:
        raise ValueError(f"K={K}는 할당 불가 (k·q_min={k*q_min} ≤ K ≤ k·g={k*g})")

    # ─ 1. 하한 선분배 ─────────────────────────────────────────────
    kc = torch.full((k,), q_min, dtype=torch.int64, device=prob.device)
    remK = K - k * q_min
    if remK == 0:
        return kc

    # ─ 2. water-filling with tensor ops ─────────────────────────
    cap = g - q_min                              # (상한까지 여유)
    active = kc < g                              # (k,) bool

    while remK > 0:
        # 2-a. 활성 클래스 확률 재정규화
        p_act = prob * active                    # (k,)
        p_sum = p_act.sum()
        # p_sum 이 0이면 active 가 모두 False → 이론적 불가(K 범위 검사 통과했음)
        p_act = p_act / p_sum

        # 2-b. 연속 비례 할당 → floor → 정수
        alloc_float = p_act * remK               # (k,)
        alloc_int   = torch.floor(alloc_float).to(torch.int64)

        # 2-c. 남은 잔여(residue) → fractional top-r
        residue = remK - alloc_int.sum().item()
        if residue > 0:
            frac = alloc_float - alloc_int.float()      # (k,)
            # active 중에서 frac 큰 순서 top-residue
            top_val, top_idx = torch.topk(frac, residue)
            alloc_int[top_idx] += 1

        # 2-d. kc 갱신 & 상한 초과 처리
        kc += alloc_int
        over = kc - g
        over_mask = over > 0
        if over_mask.any():
            excess = torch.where(over_mask, over, torch.zeros_like(over)).sum().item()
            kc = torch.where(over_mask, torch.full_like(kc, g), kc)
            remK = excess
            active = kc < g
        else:
            remK = 0

    return kc

class WeightedGlobalTopKVote(nn.Module):
    def __init__(self, k:int, in_dim:int, tau:float=0.05, *, k_keep:int=4,
                 tau_out:float=1.0,      # ← sigmoid 전용 τ
                 ste:bool=True, device="cuda"):
        super().__init__()
        self.k, self.tau, self.K = k, tau, k_keep
        self.tau_out = tau_out
        self.g = in_dim // self.k
        self.ste = True
        self.device = device
        #self.register_parameter("alpha", None)
        self.alpha = nn.Parameter(torch.zeros(self.k, self.g, device=self.device))
        self.k_c = None



    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, feats = x.shape
        xg = x.view(*batch_dims, self.k, self.g)

        # ---- α 초기화 -------------------------------------------------
        #if self.alpha is None or self.alpha.shape != (self.k, self.g):
        #    self.alpha.requires_grad = True
        #    self.alpha.data.uniform_(-0.01, 0.01)

        # ---- 1) global softmax ---------------------------------------
        w_soft = torch.softmax(self.alpha.flatten() / self.tau, dim=0)\
                     .view(self.k, self.g)

        # ---- 2) quota k_c --------------------------------------------
        k_c = quota_by_prob_torch(w_soft.sum(-1), self.K, self.g)

        #real_quota = w_soft.sum(-1) * self.K           # (k,)
        #k_c = torch.floor(real_quota)
        #remain = int(self.K - k_c.sum().item())
        #if remain > 0:
        #    frac = (real_quota - k_c).tolist()
        #    top_frac = sorted(range(self.k),
        #                      key=lambda i: frac[i],
        #                      reverse=True)[:remain]
        #    k_c[top_frac] += 1

        #k_c = k_c.to(torch.int64).clamp_max(self.g)         # (k,)
        #k_c = k_c.to(torch.int64).clamp_min(5)         # (k,)
        k_c = k_c.to(torch.int64)
        self.k_c = k_c

        # ---- 3) 그룹별 top-k (STE) -----------------------------------
        w_hard = torch.zeros_like(w_soft)
        for c in range(self.k):
            kc = k_c[c].item()
            if kc > 0:
                top_idx = w_soft[c].topk(kc).indices
                w_hard[c].scatter_(0, top_idx, 1.0)

        w_soft = w_soft / (w_soft.sum(dim=-1, keepdim=True) + 1e-12)


        if self.training and self.ste:
            w = w_hard + (w_soft - w_soft.detach())     # STE
        else:
            w = w_hard 

        # ---- 4) 가중 합산 & **신규 로짓 스케일** -----------------------
        y   = (xg * w).sum(-1)                         # (..., k)
        k_c_float = k_c.to(dtype=y.dtype)              # broadcast 준비
        logits = (y - k_c_float/ 2)/ (k_c_float/2)     # ← 변경 핵심
        p = torch.sigmoid(logits*torch.sqrt(k_c_float))
        p_hard = (p>0.5).float()
        if self.training and self.ste:
            p = p_hard.detach() + (p - p.detach())
        else:
            p = p_hard

        return p

import torch, torch.nn as nn
from typing import Optional

# ────────────────────────────────────────────────────────────
#  O(log n) 토너먼트-argmax  (vectorised, GPU friendly)
# ────────────────────────────────────────────────────────────
def _tree_argmax(scores: torch.Tensor) -> torch.Tensor:
    """
    scores : (R, N)  – R rows, N candidates
    returns: (R,)    – arg-max index per row
    """
    idx = torch.arange(scores.shape[-1], device=scores.device).expand_as(scores)
    val = scores
    while val.shape[-1] > 1:                      # ↓ halve each step
        # 홀수일 때 -∞ padding
        if val.shape[-1] & 1:
            pad_val = val.new_full((*val.shape[:-1], 1), -1e30)
            pad_idx = idx[..., :1]
            val = torch.cat([val, pad_val], -1)
            idx = torch.cat([idx, pad_idx], -1)

        even, odd   = val[..., ::2],  val[..., 1::2]
        even_i, odd_i = idx[..., ::2], idx[..., 1::2]
        choose_odd  = odd > even                    # (R, N/2)

        val = torch.where(choose_odd, odd, even)    # winners’ value
        idx = torch.where(choose_odd, odd_i, even_i)# winners’ index
    return idx.squeeze(-1)                          # (R,)

# ────────────────────────────────────────────────────────────
#  Cross-bar with tree-search
# ────────────────────────────────────────────────────────────
class CrossbarLayerTree(nn.Module):
    """
    Drop-in replacement for CrossbarLayer.
    Forward-pass complexity:  O(M log₂ N)
    """
    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        *,
        device: str = "cuda",
        grad_factor: float = 1.0,
        hard_weights: bool = False,
        implementation: Optional[str] = None,
        ste: bool = False,
        connections: str = "random",
        real_in_dim: int = None,
        alpha: float = 0.9,
    ):
        super().__init__()
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.device   = device
        self.grad_factor = grad_factor
        self.hard_weights = hard_weights
        self.ste      = ste
        self.connections = connections
        self.real_in_dim = real_in_dim
        self.alpha    = alpha

        self.weights = nn.Parameter(
            torch.randn(out_dim, in_dim, device=device).float()
        )
        if implementation is None:
            implementation = "cuda" if device == "cuda" else "python"
        self.implementation = implementation
        self.reset_parameters()

    # 기존 초기화 그대로 복사 (중략) ─────────────────────────
    def reset_parameters(self):
        nn.init.normal_(self.weights, mean=0.0, std=0.2)

    # ───────────────────────── forward ───────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, feat = x.shape
        assert feat == self.in_dim, f"{feat=}, {self.in_dim=}"

        # ① 필요하면 Grad-Factor
        if self.grad_factor != 1.0:
            x = x * self.grad_factor + (x.detach() * (1 - self.grad_factor))

        # ② STE / random / soft 선택  
        soft_w = torch.softmax(self.weights, -1)      # (M, N)

        if self.training:
            if self.connections == "ste":
                # STE: forward=hard, backward=soft
                idx       = _tree_argmax(self.weights)            # (M,)
                y_hard    = torch.take_along_dim(
                    x.unsqueeze(-2).expand(*batch_dims, self.out_dim, self.in_dim),
                    idx.expand(*batch_dims, self.out_dim).unsqueeze(-1),
                    dim=-1,
                ).squeeze(-1)                                     # (..., M)

                y_soft    = x @ soft_w.t()                        # (..., M)
                y = y_hard + (y_soft - y_soft.detach())           # straight-through
            else:  # soft routing
                y = x @ soft_w.t()

        else:  # ─────────────── inference ────────────────
            idx = _tree_argmax(self.weights)                      # (M,)
            y   = torch.take_along_dim(
                x.unsqueeze(-2).expand(*batch_dims, self.out_dim, self.in_dim),
                idx.expand(*batch_dims, self.out_dim).unsqueeze(-1),
                dim=-1,
            ).squeeze(-1)                                         # (..., M)

        return y

    def extra_repr(self) -> str:
        mode = "train" if self.training else "eval"
        return f"{self.in_dim}, {self.out_dim}, {mode} (tree-search)"

def _make_block_mask(out_dim, in_dim, block, mapping="round_robin"):
    """
    returns (out_dim, in_dim) 0/1 mask.
    mapping = "round_robin" | "random" | "tail_head"
    """
    mask = torch.zeros(out_dim, in_dim)
    n_blocks       = in_dim // block
    out_per_block  = math.ceil(out_dim / n_blocks)       # 균등 분배

    rng = torch.Generator().manual_seed(0)
    for o in range(out_dim):
        if mapping == "random":
            blk = torch.randint(n_blocks, (1,), generator=rng).item()
        elif mapping == "tail_head":        # 앞쪽 half, 뒤쪽 half 번갈아
            blk = (o % 2) * (n_blocks//2) + (o // 2) % (n_blocks//2)
        else:                               # round_robin
            blk = o // out_per_block
        c0, c1 = blk*block, (blk+1)*block
        mask[o, c0:c1] = 1.0
    return mask



class CrossbarLayer(torch.nn.Module):
    """
    Differentiable N×M cross-bar.  
    Each of the `out_dim` outputs forwards *one* of the `in_dim` inputs.
    During training the choice is a softmax distribution; at eval time it
    collapses to a hard arg-max (one-hot) so every output forwards a
    discrete input line.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        device: str = "cuda",
        grad_factor: float = 1.0,
        hard_weights: bool = False,
        implementation: Optional[str] = None,   # 'cuda' or 'python'
        ste: bool = False,
        connections: str = 'ste',
        real_in_dim: int = None,
        alpha: float = 0.9,
        block_size: int | None = None,   # ← 추가 (16, 32 …)
        block_mapping: str = "round_robin",  # ← 추가
        tau: float = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.hard_weights = hard_weights
        self.ste = ste
        # one trainable score per (output, input) pair
        self.weights = torch.nn.Parameter(torch.randn(out_dim, in_dim, device=device).float())
        self.tau = tau
        # choose backend automatically if not given
        if implementation is None:
            implementation = "cuda" if device == "cuda" else "python"
        assert implementation in ("cuda", "python")
        self.implementation = implementation
        self.connections = connections
        self.real_in_dim = real_in_dim
        self.alpha = alpha
        self.in_dim = in_dim

        self.block_size   = block_size
        self.block_mapping= block_mapping


        self.reset_parameters()


    #def reset_parameters(self):
    #    with torch.no_grad():
    #        torch.nn.init.normal_(self.weights, mean=0.0, std=0.2)
    #        self.weights.requires_grad_(True)

    def reset_parameters(self):
        """
        Initialise weights with N(0,0.2) *inside the allowed columns*.
        Disallowed columns are filled with -1e4 so softmax ≈ 0.
        """
        with torch.no_grad():
            # ❶ 전체 가중치 기본값 N(0,0.2)
            nn.init.normal_(self.weights, mean=0.0, std=0.2)

            '''
            # ───── block mask 적용 ─────
            if self.block_size is not None:
                mask = _make_block_mask(
                        self.out_dim, self.in_dim,
                        self.block_size, self.block_mapping).to(self.weights.device)

                neg_inf = -1e4
                self.weights.masked_fill_(mask == 0, neg_inf)

            # ❷ head / tail column 범위
            head_cols = slice(0, self.in_dim - self.real_in_dim)     # 0 .. -real_in_dim-1
            tail_cols = slice(self.in_dim - self.real_in_dim,        # -real_in_dim ..
                               self.in_dim)                           #        .. end

            # ❸ 행마다 Bernoulli(α) 샘플
            tail_mask = torch.bernoulli(
                torch.full((self.out_dim,), self.alpha, device=self.weights.device)
            ).bool()                       # True  → tail_cols만 허용

            # ❹ 금지 영역 = -1e4
            neg_inf = -1e4
            # rows → tail_cols만 허용 ⇒ head_cols 억제
            if tail_mask.any():
                self.weights[tail_mask, head_cols] = neg_inf
            # rows → head_cols만 허용 ⇒ tail_cols 억제
            if (~tail_mask).any():
                self.weights[~tail_mask, tail_cols] = neg_inf
            '''

            self.weights.fill_(-1e4)
            
            # 2. 대각선 요소에만 큰 양수 값을 설정합니다.
            #    이는 argmax가 항상 대각선(i==j)을 선택하도록 강제합니다.
            diag_len = min(self.out_dim, self.in_dim)
            for i in range(diag_len):
                self.weights[i, i] = 5.0
            #print(self.weights)
            #print(self.weights.shape)
                    # ❹ gradient 활성화
        self.weights.requires_grad_(True)

    # ────────────────────────────── forward ──────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim)
        returns: (..., out_dim)    with the same leading batch dims
        """
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)  # same helper you already use

        # shape sanity
        *batch_dims, feat = x.shape
        assert feat == self.in_dim, (feat, self.in_dim)


        if self.training:
            # soft selection – each row of soft_w is a probability over inputs
            if self.connections == 'random': # fixed weights
                idx = self.weights.argmax(dim=-1)                           # (out_dim,)
                hard_mask = torch.nn.functional.one_hot(idx, self.in_dim).float()   # (out_dim, in_dim)
                hard_mask = hard_mask.detach()
                w_ste = hard_mask
            else:

                soft_w = torch.softmax(self.weights/self.tau, dim=-1)  # (out_dim, in_dim)

                if self.connections == 'ste':
                    idx = self.weights.argmax(dim=-1)                           # (out_dim,)
                    hard_mask = torch.nn.functional.one_hot(idx, self.in_dim).float()   # (out_dim, in_dim)
                    w_ste = hard_mask.detach() + (soft_w - soft_w.detach())
                elif self.connections == 'soft':
                    w_ste = soft_w
                else:
                    raise ValueError(f"unknown connections: {self.connections}")

            y = x @ w_ste.T  
            #y = (w_ste @ x.T).T
            return y

        else:  # ───────────── inference / hard routing
            # one-hot mask of winning inputs per output
            idx = self.weights.argmax(dim=-1)                           # (out_dim,)
            hard_mask = torch.nn.functional.one_hot(idx, self.in_dim).float()   # (out_dim, in_dim)
            #y = torch.einsum("...n, mn -> ...m", x, hard_mask.float())
            
            y= x @ hard_mask.T  
            
            #y = (hard_mask @ x.T).T
            return y

    # ───────────────────────────── extras ────────────────────────────────
    def extra_repr(self) -> str:
        mode = "train" if self.training else "eval"
        return f"{self.in_dim}, {self.out_dim}, {mode}"







class BlockEfficientCrossbarLayer(nn.Module):
    """
    [FINAL VERSION]
    'unique' 모드가 블록 내에서 무작위 연결을 생성하도록 수정된 버전.
    (기존의 순차적 unique 모드는 삭제됨)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        *,
        connections: str = 'ste', # 'ste', 'softmax', 'unique' (이제 random)
        tau: float = 1.0,
        device: str = "cuda",
        init: str = "normal",
        implementation: Optional[str] = "python",  # 'cuda' or 'python'
    ):
        super().__init__()
        assert in_dim % num_blocks == 0, "in_dim must be divisible by num_blocks"
        assert out_dim % num_blocks == 0, "out_dim must be divisible by num_blocks"
        if init == 'residual':
            # residual 초기화는 블록 내에서 입/출력 개수가 같거나 입력이 더 많을 때 가장 이상적입니다.
            assert in_dim // num_blocks >= out_dim // num_blocks, \
                "For 'residual' init, block_size must be >= out_per_block."

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.block_size = in_dim // num_blocks
        self.connections = connections
        self.tau = tau
        self.device = device
        self.init = init
        self.out_per_block = out_dim // self.num_blocks
        
        # implementation 설정
        if implementation is None:
            implementation = "cuda" if device == "cuda" else "python"
        assert implementation in ("cuda", "python")
        self.implementation = implementation

        # ❗ [핵심 수정] 'unique'가 랜덤 연결을 생성하도록 로직 변경 ❗
        if self.connections == 'unique':    
            all_indices = []
            for _ in range(self.num_blocks):
                # 각 블록마다 독립적인 랜덤 인덱스를 생성
                if self.out_per_block <= self.block_size:
                    # 출력이 입력보다 적거나 같으면, 중복 없이 랜덤 샘플링
                    perm = torch.randperm(self.block_size, device=device)
                    indices_per_block = perm[:self.out_per_block]
                else:
                    # 출력이 입력보다 많으면, 중복을 허용하여 랜덤 샘플링
                    indices_per_block = torch.randint(
                        0, self.block_size, (self.out_per_block,), device=device
                    )
                all_indices.append(indices_per_block)
            
            indices = torch.cat(all_indices)
            self.register_buffer('connection_indices', indices.long())
            self.weights = None
        elif self.connections == 'unique_contiguous':
            # ❗ [핵심 수정] get_unique_connections 기반의 새로운 'unique' 로직 ❗
            assert self.block_size >= 2, "For 'unique' connection type, block_size must be at least 2."
            all_indices = []
            for _ in range(self.num_blocks):
                # 1. 블록 내에서 get_unique_connections를 사용해 '규칙 기반'으로 후보 쌍을 생성
                #    인덱스는 블록 내 상대 좌표 (0 ~ block_size-1)
                group_a, group_b = get_unique_connections(self.block_size, self.out_per_block // 2, device=device)
                
                # 2. 두 후보를 쌓아서 선택을 준비 (Shape: 2, out_per_block)
                indices_per_block = torch.stack([group_a, group_b], dim=0)
                
                all_indices.append(indices_per_block)
            
            indices = torch.cat(all_indices)
            self.register_buffer('connection_indices', indices.long())
            self.weights = None

        else: # ste, softmax
            self.weights = nn.Parameter(
                torch.empty(out_dim, self.block_size, device=device).float()
            )
            self.reset_parameters()

    def reset_parameters(self):
        if self.weights is None:
            return

        # ❗ [핵심 수정] self.init 값에 따라 다른 초기화 수행 ❗
        if self.init == 'residual':
            # --- Residual 초기화 로직 ---
            with torch.no_grad():
                # 1. 먼저 모든 가중치를 작은 랜덤 노이즈로 채웁니다. (대칭성 파괴)
                self.weights.normal_(mean=0.0, std=0.01)

                # 2. "대각선"에 해당하는 가중치에 큰 값을 더해줍니다.
                #    (블록 단위로 동작함을 고려해야 함)
                for j in range(self.out_dim):
                    # j번째 출력이 속한 블록 내에서, j와 동일한 상대적 위치의 입력 인덱스
                    # 예: out_per_block=8이면 j=0~7은 각각 입력 0~7에, j=8은 다시 입력 0에...
                    identity_input_idx = j % self.out_per_block
                    
                    # 해당 위치의 가중치를 크게 만듭니다.
                    self.weights[j, identity_input_idx] += 5.0
        else:
            # --- 기본 Normal 초기화 로직 ---
            nn.init.normal_(self.weights, mean=0.0, std=0.2)
        
        self.weights.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        w_sparse = None

        # ─────────────────────────────
        # 1) connections == 'unique'
        #    -> 각 블록에서 하나의 입력만 선택하는 고정 크로스바
        #    -> matmul 대신 gather로 구현 (훨씬 가벼움)
        # ─────────────────────────────
        if self.connections == 'unique':
            # connection_indices: [out_dim] (블록마다 out_per_block개씩 이어붙임)
            # reshape해서 [num_blocks, out_per_block] 로 변환
            idx_per_block = self.connection_indices.view(self.num_blocks,
                                                         self.out_per_block)  # [N, Ob]

            # x: [B, in_dim] -> [B, num_blocks, block_size]
            x_blocked = x.view(B, self.num_blocks, self.block_size)  # [B, N, Cb]

            # idx_expanded: [B, num_blocks, out_per_block]
            idx_expanded = idx_per_block.unsqueeze(0).expand(B, -1, -1)

            # gather: 마지막 dim(=block_size)에서 인덱스 선택
            # 결과 y_blocked: [B, num_blocks, out_per_block]
            y_blocked = torch.gather(x_blocked, dim=2, index=idx_expanded)

            # [B, num_blocks, out_per_block] -> [B, out_dim]
            output = y_blocked.contiguous().view(B, -1)
            return output

        # ─────────────────────────────
        # 2) 그 외 모드 (ste, softmax 등)
        #    -> 기존 로직 그대로 유지
        # ─────────────────────────────
        if self.connections == 'unique_contiguous':
            # unique_contiguous는 여전히 one_hot + matmul 경로 사용
            w_sparse = F.one_hot(self.connection_indices, self.block_size).float()

        elif self.training:  # ste, softmax
            soft_w = F.softmax(self.weights / self.tau, dim=-1)
            if self.connections == 'ste':
                idx = soft_w.argmax(dim=-1)
                hard_mask = F.one_hot(idx, self.block_size).float()
                w_ste = hard_mask.detach() + (soft_w - soft_w.detach())
            else:  # 'softmax'
                w_ste = soft_w
            w_sparse = w_ste

        else:  # eval for ste, softmax
            with torch.no_grad():
                idx = self.weights.argmax(dim=-1)
                hard_mask = F.one_hot(idx, self.block_size).float()
                w_sparse = hard_mask

        # CUDA kernel 사용 여부 결정 (unique 이외의 경우만)
        use_cuda = (
            self.implementation == "cuda"
            and CUDA_AVAILABLE
            and x.is_cuda
            and x.is_contiguous()
            and w_sparse.is_contiguous()
        )

        if use_cuda:
            return BlockEfficientCrossbarCudaFunction.apply(
                x, w_sparse, self.num_blocks, self.block_size, self.out_per_block
            )
        else:
            # Fallback to PyTorch operations (einsum)
            x_blocked = x.view(B, self.num_blocks, self.block_size)
            w_blocked = w_sparse.view(self.num_blocks, self.out_per_block, self.block_size)
            output_blocked = torch.einsum('bnc,ndc->bnd', x_blocked, w_blocked)
            output = output_blocked.contiguous().view(B, -1)
            return output

    def reg_loss(self) -> torch.Tensor:
            """
            Group Sparsity (L1-on-L2) 정규화 손실을 계산합니다.
            각 입력 채널(column)을 그룹으로 간주합니다.
            """
            if self.weights is None:
                return torch.tensor(0., device=self.device)
                
            # 1. 각 세로 열(dim=0)에 대해 L2-norm을 계산합니다.
            #    결과 텐서의 크기는 (in_dim,)이 됩니다.
            column_norms = torch.norm(self.weights, p=2, dim=0)
            
            # 2. 계산된 모든 norm 값들을 더합니다 (L1-sum).
            return column_norms.sum()

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"num_blocks={self.num_blocks}, block_size={self.block_size} (derived), "
                f"connections='{self.connections}', implementation='{self.implementation}'")


class BlockEfficientCrossbarCudaFunction(torch.autograd.Function):
    """
    CUDA kernel을 사용하는 BlockEfficientCrossbarLayer의 autograd Function
    """
    @staticmethod
    def forward(ctx, x, w_sparse, num_blocks, block_size, out_per_block):
        ctx.save_for_backward(x, w_sparse)
        ctx.num_blocks = num_blocks
        ctx.block_size = block_size
        ctx.out_per_block = out_per_block
        
        # CUDA kernel 호출
        return difflogic_cuda.block_efficient_crossbar_forward(
            x, w_sparse, num_blocks, block_size, out_per_block
        )
    
    @staticmethod
    def backward(ctx, grad_y):
        x, w_sparse = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        
        grad_x = grad_w_sparse = None
        
        if ctx.needs_input_grad[0]:
            # Input gradient
            grad_x = difflogic_cuda.block_efficient_crossbar_backward_x(
                w_sparse, grad_y, ctx.num_blocks, ctx.block_size, ctx.out_per_block
            )
        
        if ctx.needs_input_grad[1]:
            # Weight gradient (w_sparse에 대한 gradient)
            grad_w_sparse = difflogic_cuda.block_efficient_crossbar_backward_w(
                x, grad_y, ctx.num_blocks, ctx.block_size, ctx.out_per_block
            )
        
        return grad_x, grad_w_sparse, None, None, None









class VotingCrossbarLayer(nn.Module):
    """
    [NEW] 'VotingLayer'의 설계를 따르는 메모리 효율적인 Crossbar.
    - 각 출력은 `num_active_inputs`개의 무작위 입력 후보의 '가중치 없는 합'으로 계산됩니다.
    - 그 합이 과반수(num_active_inputs / 2)를 넘는지 판정하여 0 또는 1을 출력합니다.
    - STE/Soft 학습 방식과 Training/Eval 모드를 모두 지원합니다.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_active_inputs: int,
        *,
        ste: bool = True,
        tau: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        assert num_active_inputs <= in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_active_inputs = num_active_inputs
        self.ste = ste
        self.tau = tau
        self.device = device

        # 이 레이어는 학습 가능한 가중치(weights)를 사용하지 않습니다.
        
        # 연결 후보의 인덱스를 저장할 버퍼
        self.register_buffer(
            'active_input_indices',
            torch.empty(out_dim, num_active_inputs, dtype=torch.long, device=device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """연결 후보 인덱스를 무작위로 초기화합니다."""
        with torch.no_grad():
            # 메모리 효율적인 방식으로 연결 후보 인덱스 샘플링
            indices = torch.randint(
                0, self.in_dim,
                (self.out_dim, self.num_active_inputs),
                device=self.device
            )
            self.active_input_indices.copy_(indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim), 여기서 B는 배치 크기. 입력값은 0 또는 1로 가정.
        returns: (B, out_dim)
        """
        
        # 1. Gather: 입력 x에서 연결 후보에 해당하는 값들을 가져옴
        # shape: (B, out_dim, num_active_inputs)
        gathered_inputs = x[:, self.active_input_indices]
        
        # 2. [핵심] VotingLayer와 동일한 로직: 가중치 없이 합산
        # shape: (B, out_dim)
        s = gathered_inputs.sum(dim=-1)
        
        # 3. 다수결 판정을 위한 logits 계산
        # 그룹 크기(g)는 self.num_active_inputs가 됨
        g = self.num_active_inputs
        logits = (s - g / 2) / math.sqrt(g / 4) # VotingLayer의 정규화 방식 적용

        # 4. 학습/평가/STE 모드에 따라 최종 0/1 출력 결정
        if self.training:
            p = torch.sigmoid(logits / self.tau)
            
            if not self.ste:
                return p  # Soft-forward
            
            # STE-forward
            y_hard = (p > 0.5).float()
            return y_hard.detach() - p.detach() + p
        else: # eval mode
            # 평가 시에는 logits이 0보다 큰지(s > g/2)로 hard 판정
            return (logits >= 0).float()

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"num_active_inputs={self.num_active_inputs}, method='unweighted_voting', ste={self.ste}")



def init_logic_weights(t: torch.Tensor, mode: str = "normal", mean: float = 0.0, std: float = 0.2,  boost_ids: Union[int, Sequence[int]] = (3),   # ex) 3  or  (3,6,9)
    boost_scale: float = 6.0, noise_std: float = -1.0

):

    if mode == "normal":
        nn.init.normal_(t, mean=mean, std=std)
        #t = abs(t)


    elif mode == "residual":

        # ① 전체를 N(mean, std²) 로 초기화
        nn.init.normal_(t, mean=mean, std=std)

        # ② boost 할 열만 스케일 업
        if isinstance(boost_ids, int):
            boost_ids = (boost_ids,)           # 단일 int → 튜플로

        for idx in boost_ids:
            t[:, idx] *= boost_scale

        # ③ 소량의 노이즈로 symmetry break
        if noise_std > 0:
            t.add_(torch.randn_like(t) * noise_std)
        return



    else:
        raise ValueError(f"unknown init mode: {mode}")



class LogicBlock(nn.Module):
    """
    Crossbar-Logic ladder with skip-windowed routing.

    At every CrossbarLayer i, the *input vector* is:
        concat( outputs from the last k LogicLayers )
    (If fewer than k LogicLayers have been seen so far, concat all of them;
     for the very first CrossbarLayer that set is empty, so we use the raw
     external input x₀.)

    Args
    ----
    n_in        : dimension of external input  x₀
    n_out       : number of vote groups / final classes
    n_layers    : number of LogicLayers               (default 5)
    width       : hidden width per LogicLayer (except last) (default 5)
    k_history   : how many past LogicLayer outputs to expose to each crossbar
    crossbar_ste, voter_ste : STE options (as before)
    """
    def __init__(
        self,
        n_in: int,
        n_out: int,
        width: int = 5,
        k_history: int = 2,
        device: str = "cuda",
        crossbar_ste: bool = True,
        connections: str = 'ste',
        k_history_include_input: bool = True,
        logic_layer_ste: bool = True,
        noise_prob: float = 0.0,
        implementation : str = 'python',
        grad_factor: float = 1.0,
        use_crossbar_tree: bool = False,
        initialization: str = 'normal',
        mean: float = 0.0,
        std: float = 0.2,
        block_size: int = None,
        block_mapping: str = "round_robin"
    ):
        super().__init__()
        self.k_history = max(1, k_history)      # safety
        self.k_history_include_input = k_history_include_input
        self.layers    = nn.ModuleList()
        self.in_dim = n_in
        self.out_dim = n_out
        assert grad_factor == 1.0, "grad_factor is not supported in LogicBlock"

        if isinstance(width, int):
            widths = [width]                      # broadcast → old behaviour
        else:
            widths = width

        self.n_layers = len(widths)    

        #prev_logic_dims = []                    # rolling list of LogicLayer out-dims
        if k_history_include_input:
            prev_logic_dims = [n_in]                    # rolling list of LogicLayer out-dims
        else:
            prev_logic_dims = []
        cur_in_dim      = n_in
        #self.min_out_dim = width[0]   
        #self.embedding_layer_idx = 0

        #if self.embedding_layer_idx!=-1:
        #    for i, out_dim in enumerate(widths):
        #        if self.embedding_layer_idx!=-1 and out_dim < self.min_out_dim:
        #            self.embedding_layer_idx = i
        #            self.min_out_dim = out_dim

        ## NOTE: INPUT을 포함하게 바꿈 /2025.05.01
        for i, out_dim in enumerate(widths):
        
            # ---------- decide LogicLayer out-dim ----------
            #if i == n_layers - 1:               # last hidden → shrink for voter
            #    out_dim = width                 # can customise
            #else:
            #    out_dim = width

            # ---------- crossbar input dim = concat(k last logic outs) ----------
            if prev_logic_dims:
                cur_in_dim = sum(prev_logic_dims[-self.k_history:])  # window
                real_in_dim = prev_logic_dims[-1]
            else:
                cur_in_dim = n_in                                   # first pass
                real_in_dim = n_in


            if use_crossbar_tree:
                crossbar_class = CrossbarLayerTree
            else:
                crossbar_class = CrossbarLayer

            # ---------- modules ----------
            if connections == 'ste':
                route = crossbar_class(
                    in_dim=cur_in_dim,
                    out_dim=out_dim * 2,
                    device=device,
                    ste=crossbar_ste,
                    connections=connections,
                    real_in_dim= real_in_dim
                    #block_size = block_size,
                    #block_mapping = block_mapping
                )
                logic_in = out_dim * 2
            else:
                logic_in = cur_in_dim 

            print(f"Stage {i}: {cur_in_dim} (concat) -> {logic_in} -> {out_dim}")

            logic = LogicLayer(
                in_dim=logic_in,
                out_dim=out_dim,
                device=device,
                hard_weights=False,
                connections="unique",
                grad_factor= 1.0, #2 ** ((n_layers - i) / 2),
                ste=logic_layer_ste,
                implementation=implementation
            )
            # ---------- weight init ----------
            with torch.no_grad():
                init_logic_weights(logic.weights, initialization, std=std, mean=mean)
                logic.weights.requires_grad_(True)
                if connections == 'ste':
                    if type(route) == CrossbarLayer:
                        nn.init.normal_(route.weights, mean=mean, std=std)
                        route.weights.requires_grad_(True)

            # ---------- register ----------
            if connections == 'ste':
                self.layers.append(route)
            self.layers.append(logic)

            #ivote = VotingLayer(out_dim//5, 0.1, ste=True)
            #self.layers.append(ivote)
            #curr_in_dim = out_dim//5

            # pair-wise mask indices for logic layer
            #a_idx = torch.arange(0, out_dim, device=device)
            #logic.indices = (a_idx * 2, a_idx * 2 + 1)
            #TODO: Investigate above

            #if i==self.embedding_layer_idx: # actually stage index
            #    prev_logic_dims = [out_dim]
            #else:
            prev_logic_dims.append(out_dim)     # track for next stage
        # ---------------- voting layer ----------------

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Keep a buffer of recent LogicLayer outputs so each CrossbarLayer
        gets the concatenated window it expects.
        """
        if self.k_history_include_input:
            recent_logic = [x]
        else:
            recent_logic = []             # list of tensors

        real_in_size = {}
        logic_layer_idx = 0
        for layer in self.layers:
            # ---- Crossbar: feed concat of last k logic outputs ----
            if isinstance(layer, CrossbarLayer):
                if recent_logic:
                    concat_in = torch.cat(
                        recent_logic[-self.k_history :], dim=-1
                    )
                    real_in_size[layer] = recent_logic[-1].shape[-1]
                else:
                    concat_in = x                      # first crossbar
                    real_in_size[layer] = x.shape[-1]
                x = layer(concat_in)

            # ---- LogicLayer: normal forward, store output ----
            elif isinstance(layer, LogicLayer):
                x = layer(x)
                recent_logic.append(x)
                logic_layer_idx += 1
            # ---- anything else (e.g., VotingLayer) ----
            else:
                x = layer(x)

        return x 

import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            dim (int): 입력 및 출력 차원 (skip connection을 위해 동일해야 함)
            hidden_dim (int): 내부 확장 차원
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_out)
        self.bn2 = nn.BatchNorm1d(dim_out)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, D)
        """
        residual = x  # skip connection
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        #out = self.dropout(out)
        out += residual  # skip connection
        return out


class BinarizeSTE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            p = torch.sigmoid(x)                               # (..., k)
            p_hard = (p > 0.5).float()
            p = p_hard.detach() - p.detach() + p
            return p
        else:
            p = (x>=0).float()
            return p


        x_hard = (x > 0).float()
        x = x_hard.detach() - x.detach() + x
        return x
    
    
  
class HybridEncoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 5, k_history: int = 2, device: str = "cuda", crossbar_ste: bool = True, connections: str = 'ste', k_history_include_input: bool = True, logic_layer_ste: bool = True, noise_prob: float = 0.0, implementation : str = 'python', use_concat=False):
        super().__init__()
        n_hidden1 = 16
        self.mlp = MLPBlock(8, 8, 8)
        #self.logic_block1 = LogicBlock(n_in, 8, width, k_history, device, crossbar_ste, connections, k_history_include_input, logic_layer_ste, noise_prob, implementation)
        self.b2r1 = Binary2Real(8)
        self.logic_block2 = LogicBlock(n_in, n_out, width, k_history, device, crossbar_ste, connections, k_history_include_input, logic_layer_ste, noise_prob, implementation)
        self.b2r2 = Binary2Real(n_out)
        self.b1 = BinarizeSTE()
        self.binarize = Binarize() #(1.0)
        self.use_concat = use_concat
        #if self.use_concat:
        #    self.relu3 = nn.ReLU()
        #    self.linear4 = nn.Linear(n_out + n_out, n_out)
    
    def forward(self, x):
        #x = self.mlp(x)
        #x = self.b1(x)

        #x_hard = (x > 0.5).float()
        #x = x_hard.detach() - x.detach() + x


       
        #x = self.logic_block1(x)
        #x = self.b2r1(x)
        #x = self.b1(x)

        y = self.logic_block2(x)
        y = self.b2r2(y)
        #x = self.linear1(x)
        #x = self.relu1(x)
        #x = self.linear2(x)
        #x = self.relu2(x)
        #z = self.linear3(x)
        #if self.use_concat:
        #    z = self.relu3(z)
        #    x = torch.cat([y, z], dim=-1)
        #    x = self.linear4(x)
        #else:
        #    x = y + z
        x = self.binarize(y)
        return x


class HybridDecoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 5, k_history: int = 2, device: str = "cuda", crossbar_ste: bool = True, connections: str = 'ste', k_history_include_input: bool = True, logic_layer_ste: bool = True, noise_prob: float = 0.0, implementation : str = 'python', use_concat=False):
        super().__init__()
        self.mlp = MLPBlock(n_in, 8, 64)
        self.logic_block = LogicBlock(8, n_out, width, k_history, device, crossbar_ste, connections, k_history_include_input, logic_layer_ste, noise_prob, implementation)
        #self.b2r = #Binary2Real(n_out)
        self.b2r = RegressionLayer(n_out, 0.2, noise_prob=noise_prob, use_ternary=False)
        self.b1 = BinarizeSTE()
        self.binarize = Binarize() #(1.0)

        self.use_concat = use_concat
        #if self.use_concat:
        #    self.relu3 = nn.ReLU()
        #    self.linear4 = nn.Linear(n_out + n_out, n_out)
    
    def forward(self, x):
        #x = self.mlp(x)
        #x = self.b1(x)
        #x_hard = (x > 0.5).float()
        #x = x_hard.detach() - x.detach() + x
        y = self.logic_block(x)
        y = self.b2r(y)
        #x = self.linear1(x)
        #x = self.relu1(x)
        #x = self.linear2(x)
        #x = self.relu2(x)
        #z = self.linear3(x)
        #if self.use_concat:
        #    z = self.relu3(z)
        #    x = torch.cat([y, z], dim=-1)
        #    x = self.linear4(x)
        #else:
        #    x = y + z
        return y
 


class ResidualLogicNet(nn.Module):
    """
    Crossbar-Logic ladder with skip-windowed routing.

    At every CrossbarLayer i, the *input vector* is:
        concat( outputs from the last k LogicLayers )
    (If fewer than k LogicLayers have been seen so far, concat all of them;
     for the very first CrossbarLayer that set is empty, so we use the raw
     external input x₀.)

    Args
    ----
    n_in        : dimension of external input  x₀
    n_out       : number of vote groups / final classes
    n_layers    : number of LogicLayers               (default 5)
    width       : hidden width per LogicLayer (except last) (default 5)
    k_history   : how many past LogicLayer outputs to expose to each crossbar
    crossbar_ste, voter_ste : STE options (as before)
    """
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 5,
        width: int = 5,
        k_history: int = 2,
        device: str = "cuda",
        crossbar_ste: bool = True,
        voter_ste: bool = False,
        connections: str = 'ste',
        k_keep: int = 0,
        k_history_include_input: bool = True,
        logic_layer_ste: bool = True,
        noise_prob: float = 0.0,
        implementation : str = 'python',
        use_ternary: bool = True,
        bitflip_prob: float = 0.00,
        initialize: str = 'normal',
        last : bool = False
    ):
        super().__init__()
        self.k_history = max(1, k_history)
        self.k_history_include_input = k_history_include_input
        self.layers = nn.ModuleList()
        self.in_dim = n_in
        self.out_dim = n_out
        self.last = last # 'last' 옵션을 멤버 변수로 저장

        if isinstance(width, int):
            # n_layers가 명시되지 않은 경우, width 리스트의 길이가 n_layers가 됨
            if n_layers == 5 and isinstance(width, list):
                 widths = width
            else:
                 widths = [width] * n_layers
        else:
            widths = width

        self.n_layers = len(widths)

        if k_history_include_input:
            prev_logic_dims = [n_in]
        else:
            prev_logic_dims = []
        
        # --- 메인 루프: (Crossbar -> Logic) 쌍을 순서대로 생성 ---
        for i, out_dim in enumerate(widths):
            if prev_logic_dims:
                cur_in_dim = sum(prev_logic_dims[-self.k_history:])
                real_in_dim = prev_logic_dims[-1]
            else:
                cur_in_dim = n_in
                real_in_dim = n_in

            print(f"Stage {i}: {cur_in_dim} (concat) -> {out_dim*2} -> {out_dim}")

            route = CrossbarLayer(
                in_dim=cur_in_dim, out_dim=out_dim * 2, device=device,
                ste=crossbar_ste, connections=connections, real_in_dim=real_in_dim
            )
            logic = LogicLayer(
                in_dim=out_dim * 2, out_dim=out_dim, device=device,
                hard_weights=False, connections="unique", grad_factor=1.0,
                implementation=implementation, ste=logic_layer_ste
            )
            
            # 가중치 초기화
            if initialize == 'normal':
                with torch.no_grad():
                    nn.init.normal_(logic.weights, mean=0.0, std=0.2)
            elif initialize == 'uniform':
                with torch.no_grad():
                    logic.weights = nn.Parameter(torch.randn(out_dim, 16, device=device))
            elif initialize == 'residual':
                with torch.no_grad():
                    probs = torch.zeros(16, device=device)
                    probs[0] = 0.90
                    probs[1:] = (1.0 - probs[0]) / (len(probs) - 1)
                    distribution = torch.distributions.Categorical(probs)
                    sampled_indices = distribution.sample((out_dim,))
                    one_hot_weights = torch.nn.functional.one_hot(sampled_indices, num_classes=16).float()
                    logic.weights = nn.Parameter(one_hot_weights)

            logic.weights.requires_grad_(True)
            nn.init.normal_(route.weights, mean=0.0, std=0.2)
            route.weights.requires_grad_(True)

            self.layers.append(route)
            self.layers.append(logic)

            if bitflip_prob > 0.0:
                self.layers.append(BitFlip(bitflip_prob))
                assert noise_prob == 0.0, "noise_prob and bitflip_prob cannot be set at the same time"
            
            prev_logic_dims.append(out_dim)

        # --- 루프 종료 후 마지막 레이어 처리 ---
        final_logic_out_dim = prev_logic_dims[-1]

        if self.last:
            # 마지막 Crossbar에 들어올 입력의 차원을 forward와 동일한 로직으로 계산
            final_crossbar_in_dim = sum(prev_logic_dims[-self.k_history:])
            final_crossbar_real_in = prev_logic_dims[-1]

            # 계산된 차원으로 마지막 CrossbarLayer 생성
            last_route = CrossbarLayer(
                in_dim=final_crossbar_in_dim, 
                out_dim=final_logic_out_dim, # 출력 차원은 이전 로직과 동일하게 유지
                device=device, ste=crossbar_ste, 
                connections=connections, real_in_dim=final_crossbar_real_in
            )
            nn.init.normal_(last_route.weights, mean=0.0, std=0.2)
            print(f"Last Crossbar: {final_crossbar_in_dim} -> {final_logic_out_dim}")
            
            self.layers.append(last_route)

        # --- 최종 Voting Layer ---
        # Voter의 입력 차원은 마지막 레이어(추가된 Crossbar 또는 원래의 Logic)의 출력 차원과 같습니다.
        voter_in_dim = final_logic_out_dim

        
        # ---------------- voting layer ----------------
        #self.vote = VotingLayer(n_out, 0.2, ste=voter_ste)
        if k_keep > 0:
            self.vote = WeightedGlobalTopKVote(n_out, out_dim, 0.2, k_keep=k_keep, ste=voter_ste)
        elif k_keep==0:
            #if logic_layer_ste:
            self.vote = VotingLayer(n_out, 1.0, ste=voter_ste, noise_prob=noise_prob, use_ternary=use_ternary)
            #else:
            #    self.vote = VotingLayerForNonSTEInput(n_out, 0.2)
        elif k_keep==-1:
            self.vote = torch.nn.Sigmoid()
        elif k_keep==-2:
            self.vote = BinaryNumberLayer(n_out)
        elif k_keep==-3:
            #self.vote = nn.Identity()
            self.vote = OutputSmoothing()
        elif k_keep==-4:
            self.vote = RegressionLayer(n_out, 0.2, noise_prob=noise_prob, use_ternary=use_ternary)
        elif k_keep==-5:
            self.vote = GroupBinaryLayer(n_out, 0.2)
        elif k_keep==-6:
            self.vote = DiscreteRegressionLayer(n_out, 0.2)
        elif k_keep==-7:
            self.vote = Float16RegressionLayer()
        elif k_keep==-8:
            self.vote = MultiOutputRegressionLayer(n_out, 0.2)
        elif k_keep==-9:
            self.vote = WeightedVotingLayer(n_out, in_dim=last_logic_dim, ste=voter_ste)

        #self.vote = MaxVotingLayer(n_out, 0.2, ste=voter_ste)
        self.layers.append(self.vote)

    # ------------------------------------------------------------------
    #def last_embedding(self):
    #    return self.last_embedding_output

    def set_tau(self, tau):
        for l in self.layers:
            if isinstance(l, CrossbarLayerGS):
                l.set_tau(tau)

    # ------------------------------------------------------------------
    def make_pairwise_mask(self, M: int) -> torch.Tensor:
        mask = torch.zeros((M, 2 * M), dtype=torch.float32, device="cuda")
        for i in range(M):
            mask[i, 2 * i] = 1.0
            mask[i, 2 * i + 1] = 1.0
        return mask

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Keep a buffer of recent LogicLayer outputs so each CrossbarLayer
        gets the concatenated window it expects.
        """
        if self.k_history_include_input:
            recent_logic = [x]
        else:
            recent_logic = []             # list of tensors

        real_in_size = {}
        logic_layer_idx = 0
        for layer in self.layers:
            # ---- Crossbar: feed concat of last k logic outputs ----
            if isinstance(layer, CrossbarLayer):
                if recent_logic:
                    concat_in = torch.cat(
                        recent_logic[-self.k_history :], dim=-1
                    )
                    real_in_size[layer] = recent_logic[-1].shape[-1]
                else:
                    concat_in = x                      # first crossbar
                    real_in_size[layer] = x.shape[-1]
                x = layer(concat_in)

            # ---- LogicLayer: normal forward, store output ----
            elif isinstance(layer, LogicLayer):
                x = layer(x)
                recent_logic.append(x)
                logic_layer_idx += 1
            # ---- anything else (e.g., VotingLayer) ----
            else:
                x = layer(x)

        # --- unused-input accounting (unchanged) ---
        unused_sum = 0

        """
        acc_mask = None
        saved_mask = []
        for layer in reversed(self.layers):
            if isinstance(layer, LogicLayer):
                weight_mask = self.make_pairwise_mask(layer.out_dim)
            elif isinstance(layer, (CrossbarLayer, CrossbarLayerGS)):
                soft_w = torch.softmax(layer.weights, dim=-1)
                weight_mask = soft_w
                #weight_mask = weight_mask[:, :layer.out_dim]
                # break down weight_mask to parts from concat of previous layers
                weight_mask = weight_mask[:, -real_in_size[layer]:]
            # ------------------------------------------------ VotingLayer 등

        # ─── 누적 곱 ( ← 순서 중요 ) ───────────────────────────────────

            else:
                continue

            
            acc_mask = weight_mask if acc_mask is None else acc_mask @ weight_mask
            if isinstance(layer, (CrossbarLayer, CrossbarLayerGS)):
                input_used = torch.minimum(
                    acc_mask.sum(dim=0), torch.tensor(1.0, device=x.device)
                )
                unused = (acc_mask.shape[1] - input_used.sum())
                unused_sum += unused
        """

        return x 

    def state_dict(self, *args, **kwargs):
        """
        기존 state_dict에 각 LogicLayer의 'indices'를 추가하여 반환합니다.
        """
        # 1. PyTorch의 기본 state_dict를 먼저 가져옵니다 (가중치 등이 포함됨).
        sd = super().state_dict(*args, **kwargs)

        # 2. 모델의 모든 LogicLayer를 순회하며 'indices'를 찾습니다.
        for name, module in self.named_modules():
            if isinstance(module, LogicLayer):
                # 3. '모듈이름.indices' 라는 새로운 키로 state_dict에 추가합니다.
                # self.indices는 튜플이므로 torch.stack으로 묶어 저장합니다.
                if isinstance(module.indices, tuple):
                    indices_tensor = torch.stack(module.indices, dim=0)
                    sd[f'{name}.indices'] = indices_tensor
                else: # 이미 텐서인 경우 (수정된 클래스)
                    sd[f'{name}.indices'] = module.indices
        
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """
        'indices' 키를 state_dict에서 분리하여 수동으로 복원하고,
        나머지는 PyTorch의 기본 load_state_dict에 맡깁니다.
        """
        # 1. 불러온 state_dict에서 'indices' 키만 따로 분리합니다.
        indices_state = {}
        model_state = {}
        for k, v in state_dict.items():
            if k.endswith('.indices'):
                indices_state[k] = v
            else:
                model_state[k] = v
        
        # 2. 'indices'를 제외한 나머지(가중치 등)를 기본 로직으로 불러옵니다.
        # strict=False를 사용하여 'indices' 키가 없어도 오류가 나지 않도록 합니다.
        super().load_state_dict(model_state, strict=False)

        # 3. 분리해둔 'indices'를 해당하는 LogicLayer에 직접 할당합니다.
        for name, module in self.named_modules():
            if isinstance(module, LogicLayer):
                indices_key = f'{name}.indices'
                if indices_key in indices_state:
                    # 저장 시 stack으로 묶었으므로, 다시 튜플로 풀어주거나 그대로 사용합니다.
                    # 기존 코드는 튜플을 기대하므로 튜플로 변환해줍니다.
                    indices_tensor = indices_state[indices_key]
                    module.indices = (indices_tensor[0], indices_tensor[1])

# miter_net.py

# ──────────────────────────────────────────────────────────────
# 0. 5-게이트 × 5-레이어 '러프' 네트워크 생성 함수
# ──────────────────────────────────────────────────────────────


# PackBitsTensor가 이미 정의돼 있다면 import / 주석 해제
# from your_project.bits import PackBitsTensor



class LogicNet(nn.Module):
    """
    Stacks `l` LogicLayer blocks, then a GroupSum, Sigmoid, and IdentityWithNone.

    Args
    ----
    n_in  : int   – flattened input dimension
    n_out : int   – number of classes (passed to GroupSum)
    k     : int   – hidden width for each LogicLayer
    l     : int   – number of LogicLayer blocks
    llkw  : dict  – optional kwargs forwarded to each LogicLayer
    """

    def __init__(self, n_in: int, n_out: int, k: int, l: int, llkw: dict | None = None, voter_ste: bool = False):
        super().__init__()

        llkw = llkw or {}             # use empty dict if None
        layers = []

        # 1) flatten
        layers.append(nn.Flatten())

        # 2) first LogicLayer expands from n_in → k
        layers.append(LogicLayer(in_dim=n_in, out_dim=k, **llkw))

        # 3) (l-1) additional LogicLayers of size k → k
        for i in range(l - 1):
            layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw, grad_factor=2**((l-i)/2)))

        # 4) head: GroupSum → Sigmoid → IdentityWithNone
        self.vote = VotingLayer(n_out, 0.2, ste=voter_ste)
        layers.extend([
            self.vote, ]) 
        self.bias = torch.nn.Parameter(torch.randn(1,n_out, device='cuda') )

        # register the whole stack
        self.net = nn.Sequential(*layers)

    def set_tau(self, tau):
        pass

    # -------------------------- forward --------------------------
    def forward(self, x):
        """
        Returns (logits_after_sigmoid, None) because the final
        IdentityWithNone layer emits that tuple.
        """
        x = self.net(x)
        #print(x)
        #x = torch.sigmoid(x + self.bias)


        return x, 0
   



import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet

from birel.model import *
from birel.verilog import *
from birel.utils import *
import itertools
import math


def print_layer_grad_norms(model):
    """
    Prints the L2 norm of the gradients for each layer (module) in the model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
    """
    for name, module in model.named_modules():
        #i Skip the top-level module if desired, or include all
        param_grads = []
        for param in module.parameters(recurse=True):

            if param.grad is not None:
                param_grads.append(param.grad.detach().norm(2))
        if param_grads:
            layer_norm = torch.stack(param_grads).norm(2)
            print(f"Layer '{name}': grad_norm = {layer_norm.item():.4f}")


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(device, mode, dataset, n_layers=5, width=10, verbose=False, adder_size=4, k_history=7, connections='random', k_keep=10, voter_ste=True, logic_layer_ste=True, use_ternary=True):
    if mode == 'learn_gates':
        if dataset=='adder':
            n_out = adder_size+1
        elif dataset=='multiplier':
            n_out = adder_size*2
    print(f"n_layers: {n_layers}, width: {width}, k_history: {k_history}, connections: {connections}, k_keep: {k_keep}, voter_ste: {voter_ste}, logic_layer_ste: {logic_layer_ste}, use_ternary: {use_ternary}")
    if False:
        # [4000, 4000, 1600]
        n_hidden = 80
        model = torch.nn.Sequential(
         ResidualLogicNet(device=device, n_in=adder_size*2, n_out=n_hidden, n_layers=n_layers, width=[2000, 2000, 1600], k_history=k_history, connections=connections, k_keep=k_keep, voter_ste=voter_ste, logic_layer_ste=logic_layer_ste, use_ternary=use_ternary,
                                noise_prob=0.00, implementation='python'),
         ResidualLogicNet(device=device, n_in=n_hidden, n_out=n_out, n_layers=n_layers, width=[2000, 2000, 800], k_history=k_history, connections=connections, k_keep=k_keep, voter_ste=voter_ste, logic_layer_ste=logic_layer_ste, use_ternary=use_ternary,
                                noise_prob=0.00, implementation='python'),
        )
    else:
        model = ResidualLogicNet(device=device, n_in=adder_size*2, n_out=n_out, n_layers=n_layers, width=width, k_history=k_history, connections=connections, k_keep=k_keep, voter_ste=voter_ste, logic_layer_ste=logic_layer_ste, use_ternary=use_ternary,
                                noise_prob=0.00, implementation='python')
    return model.to(device)


def init_variables(mode, net, k, device):
    """
    Returns:
      sel_var: Parameter for selector or input variable
      x_var: Tensor of all input combinations (only for 'learn_select_signal' mode)
    """
    if mode == 'fa':
        # Full-adder-like: directly learn the input vector
        x_var = nn.Parameter(torch.rand(1, len(net.pi_names), device=device))
        sel_var = None
    elif mode in ['learn_select_signals','learn_gates'] :
        # Miter-based selection: learn mix over all combinations
        comb_list = list(itertools.product([0, 1], repeat=INPUT_SIZE))
        x_var = torch.tensor(comb_list, device=device).float()
        if mode == 'learn_select_signals':
            sel_var = nn.Parameter(torch.rand(k, net.new_impl.in_dim-INPUT_SIZE, device=device))
        else:
            sel_var = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return x_var, sel_var


def build_optimizer(var, lr, betas=(0.99, 0.999)):
    return optim.Adam(var, lr=lr, betas=betas)


def build_scheduler(optimizer, scheduler_type, step_size, warmup, t_max, gamma):
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, warmup=warmup)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)


# ─────────────────────────────────────────────────────────────
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



# ──────────────────────────────────────────────────────────────────────────
# temperature_scheduler.py
# ──────────────────────────────────────────────────────────────────────────
import math

class TemperatureScheduler:
    """
    Gumbel-Softmax 온도 τ 를 지수적으로 감소시키는 헬퍼.

        τ_t = max(tau_min, tau0 * decay ** t)

    Args:
        module      : CrossbarLayerGS (또는 set_tau(float) 메서드를 가진 모듈)
        tau0        : 시작 온도
        tau_min     : 최소 온도 (더 낮아지지 않음)
        decay       : 지수 감소 계수 (0<decay<1). 예: 0.95 = 5% 감소/스텝
        update_each : 몇 step 마다 한 번 갱신할지 (기본=1 → 매 스텝)
    """
    def __init__(self, module, tau0=5.0, tau_min=0.1, decay=0.97, update_each=1):
        self.module      = module
        self.tau0        = tau0
        self.tau_min     = tau_min
        self.decay       = decay
        self.update_each = update_each
        self.step_idx    = 0
        self.module.set_tau(tau0)

    def step(self):
        self.step_idx += 1
        if self.step_idx % self.update_each == 0:
            new_tau = max(self.tau_min,
                          self.tau0 * (self.decay ** self.step_idx))
            self.module.set_tau(new_tau)

def create_dataset(input_size: int,
                   name: str = 'adder',     # 'adder' or 'multiplier'
                   mode: str = 'class'):
    """
    input_size : 한 피연산자의 비트수 (예: 3 → 0‥7)
    task       : 'adder'     → a + b  (레이블 비트수 = input_size + 1)
                 'multiplier'→ a * b  (레이블 비트수 = 2*input_size)
    mode       : 'class'     → 각 비트를 one-hot 형태로 반환
    """
    assert name in ('adder', 'multiplier'), "task must be 'adder' or 'multiplier'"

    # 모든 (a,b) 조합 생성 : 0‥2^(2*input_size)-1
    inputs = torch.arange(2 ** (2 * input_size), dtype=torch.int64)

    # a = 하위 input_size비트, b = 상위 input_size비트
    a = inputs & (2 ** input_size - 1)
    b = inputs >> input_size

    # 결과 계산
    if name == 'adder':
        res = a + b
        n_bits = input_size + 1           # carry 때문에 +1
    else:  # multiplier
        res = a * b
        n_bits = 2 * input_size           # 최대 비트폭

    # -------- 레이블 --------
    labels = torch.stack([(res >> i) & 1 for i in range(n_bits)], dim=1) \
                .to(dtype=torch.float32).cuda()

    # -------- 입력 비트 --------
    inp_bits = torch.stack([(inputs >> i) & 1 for i in range(2 * input_size)], dim=1) \
                   .to(torch.float32).cuda()

    return inp_bits, labels

def set_module_logic_layer_ste(net: nn.Module, val: bool):
    for m in net.modules():
        if isinstance(m, LogicLayer):
            m.ste = val



def set_module_tau(net: nn.Module, tau_val: float):
    for m in net.modules():
        if isinstance(m, VotingLayer):
            m.tau = tau_val
        #elif isinstance(m, MaskedVotingLayer):
        #    m.tau = tau_val
        #elif isinstance(m, LogicLayer):
        #    m.tau = tau_val


def current_tau(epoch: int, total_epochs: int, tau_sched: str, tau_start: float, tau_end: float) -> float:
    if tau_sched == 'linear':
        return tau_start + (tau_end - tau_start) * (epoch / (total_epochs - 1))
    # exponential decay
    ratio = (epoch / (total_epochs - 1))
    return tau_start * (tau_end / tau_start) ** ratio



import torch
from torch import Tensor

def pca_sort_indices(vecs: Tensor) -> Tensor:
    """
    vecs : (B, D) – GPU/CPU 아무 장치
    returns : (B,) long – PCA 1-D 값 오름차순 인덱스
    """
    # (1) mean-center
    centered = vecs - vecs.mean(dim=0, keepdim=True)

    # (2) 첫 주성분 방향 (torch.pca_lowrank: GPU 가속 가능)
    _, _, V = torch.pca_lowrank(centered, q=1)      # V: (D,1)
    dir1 = V[:, 0]                                  # (D,)

    # (3) 1-D 투사 후 정렬
    proj = centered @ dir1                          # (B,)
    sort_idx = proj.argsort()                       # 오름차순

    return sort_idx

def greedy_sort_indices(vecs: Tensor) -> Tensor:
    """
    vecs : (B, D) – GPU/CPU
    returns : (B,) long – '가장 가까운 이웃' 경로 순서
    """
    B = vecs.size(0)
    dist = torch.cdist(vecs, vecs)          # (B, B)
    visited = torch.zeros(B, dtype=torch.bool, device=vecs.device)
    order   = torch.empty(B, dtype=torch.long, device=vecs.device)

    cur = 0
    order[0] = cur
    visited[cur] = True
    for t in range(1, B):
        d = dist[cur]
        d[visited] = float('inf')
        cur = torch.argmin(d).item()
        order[t] = cur
        visited[cur] = True

    return order


def random_sort_indices(vecs: Tensor, *, seed: Optional[int] = None) -> Tensor:
    """
    vecs : (B, D) – GPU 또는 CPU tensor
    seed : int | None – 방향 벡터 재현성을 위한 선택적 시드

    returns
    -------
    sort_idx : (B,) long –  랜덤 프로젝션 값 오름차순 인덱스
    """
    if seed is not None:
        # 장치 일관성을 위해 device-local 시드를 설정
        g = torch.Generator(device=vecs.device).manual_seed(seed)
        rand_dir = torch.randn(vecs.size(1), generator=g, device=vecs.device)
    else:
        rand_dir = torch.randn(vecs.size(1), device=vecs.device)

    rand_dir = rand_dir / rand_dir.norm()        # 단위 벡터
    proj     = vecs @ rand_dir                   # (B,)
    sort_idx = proj.argsort()                    # 오름차순 인덱스

    return sort_idx

def train(net, mode, dataset, lr=0.1, steps=1000, k=128, device='cuda',
           tol=1e-3, noise_std0=1e-3, decay=0.5,
          log_interval=20, noise_enable=False, scheduler_type='cosine', step_size=30, warmup=10, t_max=100, gamma=0.1, adder_size=4,
            verbose=False, precondition_steps=0, batch_size=1024, tau_sched='linear', tau_start=1.0, tau_end=0.01):

    inputs, labels = create_dataset(adder_size, dataset, mode=mode)
    dataset_size = inputs.size(0)
    #x_var, sel_var = init_variables(mode, net, k, device)
    optvar = net.parameters()
    optimizer = build_optimizer(optvar, lr)
    scheduler = build_scheduler(optimizer, scheduler_type, step_size, warmup, t_max, gamma)
    net.train()

    get_grad_norms = attach_grad2norm_logger(net, verbose=True)


    if mode=="regression":
        loss_fn = nn.MSELoss()

        #print(##labels)
        #print(labels.shape)
    else:
        loss_fn = nn.BCELoss()

    #sched = TemperatureScheduler(net,
    #                         tau0=5.0,      # 시작 온도
    #                         tau_min=0.05,  # 하한
    #                         decay=0.95,    # 1-에포크당 5% 감소
    #                         update_each=steps/20.0)
    best_loss = 100.0
    batch_size = batch_size 

    # 1.  PRE-CONDITIONING (optional) ­––––––––––––––––––––––––
    if precondition_steps > 0:
        # p(x_i=1)  and  q(y_j=1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr*0.1

        set_module_logic_layer_ste(net, False)
        net.vote.sigmoid = False
        for pc_step in range(1, precondition_steps + 1):

            perm = torch.randperm(dataset_size)

            mean_size = 1
            pre_batch_size = min(batch_size, perm.shape[0])//8 
            for i in range(0, dataset_size, mean_size*pre_batch_size):

                batch_idx = perm[i:i+mean_size*pre_batch_size]
                batch_inputs = inputs[batch_idx].to(device)
                batch_labels = labels[batch_idx]

                sort_idx      = random_sort_indices(batch_labels)  # 1-D 인덱스
                batch_labels  = batch_labels[sort_idx]
                batch_inputs  = batch_inputs[sort_idx, :]


                batch_inputs = batch_inputs.view(pre_batch_size, mean_size, -1)
                batch_labels = batch_labels.view(pre_batch_size, mean_size, -1)


                batch_input = batch_inputs.mean(dim=1)
                batch_label = batch_labels.mean(dim=1)

                optimizer.zero_grad()
                out = net(batch_input)
                loss_pc = loss_fn(out, batch_label) 
                loss_pc.backward()
                optimizer.step()
            if pc_step % 100 ==0 or verbose:
                print(f"[PC] step {pc_step:3d}/{precondition_steps}"
                    f" | loss={loss_pc.item():.3e}")

        # reset the main optimizer’s state so “true” training
        # starts fresh but from a better weight position
        optimizer.state.clear()
        set_module_logic_layer_ste(net, True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        net.vote.sigmoid = True
        print("Warning: STE is enabled after preconditioning")
        print(f"[PC] finished in {precondition_steps} steps.\n")

    for step in range(1, steps + 1):
        #if step<300:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = 0.001  # 원하는 lr로 설정
        #elif step==300:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = lr # scheduler.get_last_lr()[0]

        perm = torch.randperm(dataset_size)

        tau_now = current_tau(step - 1, steps, tau_sched, tau_start, tau_end)
        set_module_tau(net, tau_now)

    # ---------------------------------------------------------

        kd_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_inputs = inputs[batch_idx].to(device)
            batch_labels = labels[batch_idx]


            optimizer.zero_grad()

            out = net(batch_inputs)
            if mode=='learn_gates':
                loss = loss_fn(out, batch_labels) #+ 1e-3*net.vote.l1_mask_loss

            #embedding = net.last_embedding()
            #print(out.shape)
            #print(embedding.shape)
            #print(torch.sigmoid(embedding/net.vote.tau).shape)
            #kd_loss = 0.1*loss_fn2(out, torch.sigmoid(embedding-0.5/net.vote.tau))
            #loss += kd_loss
            #if type(net.vote) != VotingLayer:
            #    loss += 1e-5*net.vote.entropy_penalty()

            min_mismatch = torch.tensor(0)
                        
            loss.backward()
            # gradient noise injection if enabled
            if noise_enable:
                sigma = noise_std0 / ((1 + step) ** decay)
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * sigma)

        optimizer.step()
        scheduler.step()
            #sched.step()


        best_loss_updated = False
        with torch.no_grad():
            if loss.item()<best_loss and loss.item() < 6e-4:
                best_loss_updated = True
                #if best_loss_updated and (step % log_interval == 0):
                if evaluate(net, mode=mode, dataset=dataset, adder_size=adder_size, tol=1e-3, verbose=True):
                    break
                #if step % log_interval == 0:
                #    evaluate(net, sol_idx, x_var, sel_var, mode, tol=1e-3, verbose=False)
                net.train()
                best_loss = min(loss.item(), best_loss)
 

        if step % log_interval == 0 or step == steps-1:
            currnet_lr = scheduler.get_last_lr()[0]
            if hasattr(net, 'vote') and hasattr(net.vote, 'k_c'):
                print(net.vote.k_c)
            if verbose:
                get_grad_norms()
            if verbose or step <= 500:
                print(f"Step {step:4d} | lr={currnet_lr:.3e} | loss={loss.item():.3e} | min_mismatch={min_mismatch.item():.3e} | tau={tau_now:.3e}")
                if step == 500 and loss.item() >= 1.0:
                    break

    return 


def evaluate(net,  mode, dataset, adder_size,  tol=1e-3, verbose=False, subset_info=[], mismatch_info=[]):
    net.eval()

    inputs, labels = create_dataset(adder_size, mode=mode, name=dataset)
    dataset_size = inputs.size(0)
    batch_size = 2**10


    # ⬇️  embed/label 전부 저장할 리스트
    all_embeds, all_labels = [], []
    voter_in = []
    voter_out = []

    with torch.no_grad():
        sum_mismatch = 0
        for i in range(0, dataset_size, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]


            out = net(batch_inputs)
            # [4000, 4000, 1600]
            voter_in.append(net.vote.voter_in) # SHOULD BE FIXED
            voter_out.append(out)

            #last = net.last_embedding()          # (B, emb_dim) or None
            #if last is not None:
            #    all_embeds.append(last.detach())
            #    all_labels.append(batch_labels)  # ← label과 매핑

            if mode=='learn_gates':
                pred = (out > 0.5).int()          # 0/1 예측
                target = (batch_labels == 1).int()         # 0/1 정답

            mismatch = (pred != target).sum().item()
            sum_mismatch += mismatch
        

        # ───────────────── Verilog case 생성 ─────────────────
        if False and all_embeds:                               # 임베딩이 있다면
            emb_cat   = torch.cat(all_embeds,  dim=0)      # (N, E)
            label_cat = torch.cat(all_labels, dim=0)       # (N, L)

            L, E = label_cat.size(1), emb_cat.size(1)

            lines = [f"case (label)"]
            for lab, emb in zip(label_cat, emb_cat):
                lab_bits = "".join(str(int(b.item())) for b in lab)       # '0101…'
                emb_bits = "".join("1" if v > 0.5 else "0"               # '00110…'
                                    for v in emb)
                lines.append(
                    f"    {L}'b{emb_bits}: x = {E}'b{lab_bits};"
                )
            lines.append("endcase")

            verilog_case = "\n".join(lines)
            print("\n───── Generated Verilog case table ─────\n")
            print(verilog_case)
            print("────────────────────────────────────────\n")



        if sum_mismatch == 0:
            print("🎉 All combinations passed.")
            subset_info.append(torch.cat(voter_in, dim=0).cpu())
            subset_info.append(torch.cat(voter_out, dim=0).cpu())
            mismatch_info.append(0)
            return True
        else:
            print(f"⚠️ {sum_mismatch} combinations failed.")
            mismatch_info.append(sum_mismatch)
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='learn_gates',
                        choices=['autoencoder', 'class', 'learn_select_signals','learn_gates', 'regression'],
                        help="Operation mode")
    parser.add_argument('--model', type=str, default=None, help="Model file name")
    parser.add_argument('--dataset', type=str, default='adder',
                        choices=['adder', 'multiplier'],
                        help="Dataset")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument('--noise', action='store_true',
                        help="Enable gradient noise injection")
    parser.add_argument('--scheck', action='store_true',
                        help="Run success check over 50 runs")
    parser.add_argument('--steps', type=int, default=500,
                        help="Number of steps")
    parser.add_argument('--scheduler', type=str, default='step', choices=['cosine', 'step', 'none'],
                        help="Learning rate scheduler type (default: cosine)")
    parser.add_argument('--warmup', type=int, default=10,
                        help="Number of warmup steps for scheduler (default: 10)")
    parser.add_argument('--step_size', type=int, default=400,
                        help="Step size for StepLR (default: 30)")  
    parser.add_argument('--t_max', type=int, default=100,
                        help="T_max for CosineAnnealingLR (default: 100, usually steps)")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Gamma for StepLR (default: 0.1)")
    parser.add_argument('--k', type=int, default=128,
                        help="Number of parallel searches (default: 128)")
    parser.add_argument('--adder_size', type=int, default=4,
                        help="Adder size (default: 4)")
    parser.add_argument('--input_size', type=int, default=-1,
                        help="Input size (default: -1)")
    parser.add_argument('--k_history', type=int, default=7,
                        help="k_history (default: 0)")
    parser.add_argument('--connections', type=str, default='ste',
                        help="Connections (default: random)")
    parser.add_argument('--width', type=lambda s: eval(s) if isinstance(s, str) else s, default=[3000]*7,
                        help="Width can be an integer or a list (default: 3000)")
    parser.add_argument('--n_layers', type=int, default=7,
                        help="Number of layers (default: 7)")
    parser.add_argument('--k_keep', type=int, default=0,
                        help="Number of keep (default: 0)")
    parser.add_argument('--voter_ste', type=bool, default=False,
                        help="Voter straight-through (default: True)")
    parser.add_argument('--no_logic_layer_ste', dest='logic_layer_ste', action='store_false',
                        default=True,
                        help="Logic layer straight-through (default: True)")
    parser.add_argument('--use_ternary', dest="use_ternary", action="store_true", default=False,
                        help="ternary (default: False)")
    parser.add_argument('--precondition_steps', type=int, default=0,
                        help="preconditioning steps (default: 0)")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Batch size (default: 1024)")

    parser.add_argument('--tau_sched', type=str, default='linear', choices=['linear', 'exp'],
                        help="Tau schedule (default: linear)")
    parser.add_argument('--tau_start', type=float, default=1.0,
                        help="Tau start (default: 1.0)")
    parser.add_argument('--tau_end', type=float, default=1.0,
                        help="Tau end (default: 0.01)")
    args = parser.parse_args()
    if args.input_size != -1:
        args.adder_size = args.input_size
    #if type(args.width)==str:
    #    args.width = eval(args.width)
    #print(args.voter_ste)

    device = get_device()

    if args.scheck:
        success = 0
        n_trial = 10
        sum_mismatch = 0
        for i in range(n_trial):
           # if args.mode == 'learn_gates':
           #     net.new_impl.layers[0].weights.data.normal_(0, 0.2)
            #net = build_model(device, args.mode, args.adder_size, args.k_history, args.connections, args.width, args.n_layers)
            #sol_idx, x_var, sel_var = train(net, args.mode, args.lr, args.steps, device,
            #        noise_enable=args.noise)

            net = build_model(device, args.mode, args.dataset, adder_size=args.adder_size, k_history=args.k_history, connections=args.connections, width=args.width, n_layers=args.n_layers, k_keep=args.k_keep, voter_ste=args.voter_ste, logic_layer_ste=args.logic_layer_ste, use_ternary=args.use_ternary)   
            train(net, args.mode, args.dataset, args.lr, args.steps, args.k, device,
                    noise_enable=args.noise, scheduler_type=args.scheduler, step_size=args.step_size, warmup=args.warmup, t_max=args.t_max, gamma=args.gamma, adder_size=args.adder_size,
                    precondition_steps=args.precondition_steps, batch_size=args.batch_size, tau_sched=args.tau_sched, tau_start=args.tau_start, tau_end=args.tau_end)
            mismatch_info = []
            if evaluate(net, args.mode, args.dataset, adder_size=args.adder_size, mismatch_info=mismatch_info):
                success += 1
                # Create directory name from args
                dir_name = f"model_{args.dataset}_s{args.input_size}_w{str(args.width)}"
                os.makedirs(dir_name, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(dir_name, "model.pt"))
                print(f"Model saved in {dir_name}")
            sum_mismatch += mismatch_info[0]
        print(f"## Success: {success}/{n_trial} runs found solution")
        print(f"## Average mismatch: {sum_mismatch/n_trial}")
        return
    net = build_model(device, args.mode, args.dataset, adder_size=args.adder_size, k_history=args.k_history, connections=args.connections, width=args.width, n_layers=args.n_layers, k_keep=args.k_keep, voter_ste=args.voter_ste, logic_layer_ste=args.logic_layer_ste, use_ternary=args.use_ternary)   
    if args.model is not None:
        net.load_state_dict(torch.load(args.model))
    else:
        train(net, args.mode, args.dataset, args.lr, args.steps, args.k, device,
                    noise_enable=args.noise, scheduler_type=args.scheduler, step_size=args.step_size, warmup=args.warmup, t_max=args.t_max, gamma=args.gamma, adder_size=args.adder_size,
                    precondition_steps=args.precondition_steps, batch_size=args.batch_size, tau_sched=args.tau_sched, tau_start=args.tau_start, tau_end=args.tau_end)
    subset_info = []
    if evaluate(net, args.mode, args.dataset, adder_size=args.adder_size, verbose=True, subset_info=subset_info):
        if args.model is None:
            dir_name = f"model_{args.dataset}_s{args.input_size}_w{str(args.width)}"
            os.makedirs(dir_name, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(dir_name, "model.pt"))
            print(f"Model saved in {dir_name}")
        info = save_corr_plot_and_min_subset_parallel(subset_info[0], subset_info[1])
        if args.mode != 'autoencoder':
            residual_net_to_verilog(net, info=info, module_name=f"{args.dataset}_{args.adder_size}")


if __name__ == '__main__':
    main()

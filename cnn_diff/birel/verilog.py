
import sys
import copy
import torch, torch.nn as nn
from collections import defaultdict, deque
from typing import List, Dict, Tuple
from pyverilog.vparser.parser import parse       # pip install pyverilog
from pyverilog.vparser.ast import Ioport, Decl, Input, Output, InstanceList
from pyverilog.vparser.ast import InstanceList
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

#from .functional import bin_op_s        # 이미 주신 모듈 그대로 import
from difflogic import LogicLayer, GroupSum      # LogicLayer 코드 그대로 사용
from birel.model import CrossbarLayer, VotingLayer

# --------------------------- Gate → Opcode table ----------------------
GATE_OPCODE = {   
    # Verilog primitive name (lower) : 4-bit truth-table id
    'and' : 1,
    'nand': 14,
    'or'  : 7,
    'nor' : 8,
    'xor' : 6,
    'xnor': 9,
    'buf' : 3,   # assign y = a;
    'not' : 12,
    'tie0' : 0,
    'tie1' : 15,
    'pi': 3,            # PI 도 identity
    'po': 3             # PO 도 identity(A)
} # 단항 게이트(buf/not)는 두 번째 입력을 상수 0으로 고정(A op 0).

# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |
INDEX_TO_MODULE = {
    0: "tie0",
    1: "and",
    2: "nimp",
    3: "bypass0",
    4: "nrimp",
    5: "bypass1",
    6: "xnor",
    7: "or",
    8: "nor",
    9: "xnor",
    10: "nbypass1",
    11: "rimp",
    12: "nbypass0",
    13: "imp",
    14: "nand",
    15: "tie1"
}

BAISIC_MODULES = """
// ──────────────────────────
//  Constant drivers
// ──────────────────────────
module tie0 (output wire Y, input wire A, input wire B);
    assign Y = 1'b0;      // 0
endmodule

module tie1 (output wire Y, input wire A, input wire B);
    assign Y = 1'b1;      // 1
endmodule

// ──────────────────────────
//  Single-input bypass / inverter
// ──────────────────────────
module bypass0  (output wire Y, input wire A, input wire B);
    assign Y =  A;        // Y =  A
endmodule

module nbypass0 (output wire Y, input wire A, input wire B);
    assign Y = ~A;        // Y = ¬A
endmodule

module bypass1  (output wire Y, input wire A, input wire B);
    assign Y =  B;        // Y =  B
endmodule

module nbypass1 (output wire Y, input wire A, input wire B);
    assign Y = ~B;        // Y = ¬B
endmodule

// ──────────────────────────
//  Implication family
// ──────────────────────────
// A → B  ≡  ¬A ∨ B
module imp   (output wire Y, input wire A, input wire B);
    assign Y = (~A) |  B;        // A implies B
endmodule

// B → A  ≡  ¬B ∨ A
module rimp  (output wire Y, input wire A, input wire B);
    assign Y = (~B) |  A;        // B implies A
endmodule

// ¬(A → B) ≡  A ∧ ¬B
module nimp  (output wire Y, input wire A, input wire B);
    assign Y =  A  & ~B;         // not(A implies B)
endmodule

// ¬(B → A) ≡  B ∧ ¬A
module nrimp (output wire Y, input wire A, input wire B);
    assign Y =  B  & ~A;         // not(B implies A)
endmodule

// ──────────────────────────
//  NAND / NOR / XOR / XNOR 등은
//  Verilog 원시 게이트(primitives)를 그대로 사용하므로 별도 정의 불필요
// ──────────────────────────


//─────────────────────────────────────────────────────────────
// Design-Compiler friendly Majority / k-of-N Voter
//─────────────────────────────────────────────────────────────
module voter #(
    parameter integer N       = 5,                // #inputs (≥1)
    parameter integer THRESH  = (N+1)/2           // 과반수: 기본값
)(
    output wire               Y,                  // vote result
    input  wire [N-1:0]       I                   // input bits
);
    //---------- local clog2() (inside module → OK for DC) ----
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    localparam integer W = clog2(N+1);            // popcount width

    //---------- popcount (adder tree inferred by DC) ----------
    reg [W-1:0] popcnt;
    integer k;                                    // loop index
    always @* begin
        popcnt = {W{1'b0}};
        for (k = 0; k < N; k = k + 1)
            popcnt = popcnt + I[k];
    end

    //---------- threshold comparison --------------------------
    assign Y = (popcnt >= THRESH);
endmodule





"""


def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
        assert False
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)


class Gate:
    def __init__(self, name: str, op: str, ins: Tuple[Optional[str], Optional[str]], out: Optional[str]):
        self.name, self.op, self.ins, self.out = name, op, ins, out
    def __repr__(self):
        return f"{self.name}({self.op})"

def parse_verilog(filename: str, debug: bool = False
                  ) -> Tuple[List[Gate], List[Gate], List[Gate]]:
    """
    Returns (primary_inputs, primary_outputs, gates)
    """
    try:
        ast, _ = parse([filename])
    except Exception as e:
        print(f"[ERROR] Verilog parsing failed: {e}")
        if debug:
            traceback.print_exc()
        sys.exit(1)

    mod = ast.description.definitions[0]
    pis, pos, gates = [], [], []

    # 1) ANSI 헤더 포트
    if hasattr(mod, 'portlist') and mod.portlist is not None:
        for p in mod.portlist.ports:      # p : Ioport
            if isinstance(p, Ioport):
                decl = p.first            # Input / Output / Inout
                if isinstance(decl, Input):
                    g = Gate(decl.name, "pi", () , decl.name)
                    gates.append(g)
                    pis.append(g)
                elif isinstance(decl, Output):
                    g = Gate(decl.name, "po", (decl.name,) , decl.name+"#")
                    gates.append(g)
                    pos.append(g)


    # 2) Non-ANSI Decl
    for item in mod.items:
        if item.__class__.__name__ == 'Decl':
            for decl in item.list:
                if decl.__class__.__name__ == 'Input':
                    g = Gate(decl.name, 'pi', (None, None), decl.name)
                    pis.append(g); gates.append(g)
                elif decl.__class__.__name__ == 'Output':
                    g = Gate(decl.name, 'po', (decl.name, None), decl.name+"#")
                    pos.append(g); gates.append(g)

    # 3) Instances
    for item in mod.items:
        if item.__class__.__name__ == 'InstanceList':
            prim = item.module.lower()
            for inst in item.instances:
                conns = {p.portname: p.argname.name for p in inst.portlist}
                if prim in ('buf','not'):
                    a = conns.get('A'); b = None; y = conns.get('Y', a)
                else:
                    ports = list(conns.values())
                    a, b, y = ports[0], ports[1], ports[-1]
                    y = conns.get('Y', y)
                gates.append(Gate(inst.name, prim, (a,b), y))

    if debug:
        print("┌─[ Parsed Verilog Summary ]─────────────────────────")
        print(f"│  Inputs : {len(pis)}  -> {[g.name for g in pis]}")
        print(f"│ Outputs: {len(pos)}  -> {[g.name for g in pos]}")
        print(f"│ Gates  : {len(gates)} -> {[g.name for g in gates if g.op not in ('pi','po')]}")
        print("└" + "─"*50)

    return pis, pos, gates


def topo_sort_levels(gates: List[Gate], pos: List[Gate], debug: bool = False
                     ) -> Tuple[List[List[Gate]], Dict[str, int]]:
    """
    Returns (levels, net_level_map).
    levels[i] = list of gates at level i.
    """
    # 드라이버 맵
    net_driver = {g.out: g.name for g in gates if g.out}

    # fanout, in-degree, net_lv
    fanout, in_deg, net_lv = defaultdict(list), {}, {}
    for g in gates:
        in_deg[g.name] = 0
        if g.op=='pi' and g.out:
            net_lv[g.out] = 0

    for g in gates:
        for w in g.ins:
            if not w: continue
            if w in net_driver:
                in_deg[g.name] += 1
            fanout[w].append(g.name)

    # 초기 큐
    queue = deque([g.name for g in gates if in_deg[g.name]==0])
    gate_map = {g.name: g for g in gates}
    gate_lv = {}

    while queue:
        name = queue.popleft()
        g = gate_map[name]
        if g.op=='pi':
            lvl = 0
        else:
            parent_lv = [net_lv.get(w,0) for w in g.ins if w]
            lvl = max(parent_lv)+1 if parent_lv else 0
        gate_lv[name] = lvl
        if g.out:
            net_lv[g.out] = lvl
            for succ in fanout[g.out]:
                in_deg[succ] -= 1
                if in_deg[succ]==0:
                    queue.append(succ)

    if len(gate_lv)!=len(gates):
        missing = set(g.name for g in gates) - set(gate_lv.keys())
        raise RuntimeError(f"[ERROR] Combinational loop or undriven nets: {missing}")

    max_lvl = max(gate_lv.values(), default=0)
    for po in pos:
        gate_lv[po.name] = max_lvl

    levels = [[] for _ in range(max_lvl+1)]
    for g in gates:
        levels[gate_lv[g.name]].append(g)

    if debug:
        print("┌─[ Topo Sort Levels ]──────────────────────────────")
        for lvl, grp in enumerate(levels):
            names = [g.name+"("+g.op+")" for g in grp]
            print(f"│ Level {lvl:2d} : {len(grp):3d} gates → {names}")
        print("└" + "─"*50)

    return levels, net_lv



def compute_future_uses(levels):
    """
    future[l] : level l 이 끝난 뒤에도 유지해야 하는 net 집합
                (다음 레벨 이상에서 입력으로 재사용될 net)
    """
    n_lvl  = len(levels)
    future = [set() for _ in range(n_lvl)]

    future_after = set()               # level (l+1) 이후에 필요한 net
    for l in reversed(range(n_lvl)):
        # ─ 현재 레벨에서 생성·소비되는 net들 ─
        outs_l = {g.out for g in levels[l] if g.out}
        ins_l  = {w for g in levels[l] for w in g.ins if w}

        # ① 이 레벨 직후에 남아있어야 하는 net = future_after
        future[l] = future_after.copy()

        # ② 이전 레벨이 유지해야 할 net 집합 계산
        #    - 이미 future_after 였는데, 지금 새로 생성(out)됐다면 더 위로 전달 불필요 → 제거
        #    - 지금 레벨이 참조한(ins) net 은 더 위에서부터 살아 있어야 하므로 추가
        future_after = (future_after - outs_l) | (ins_l - outs_l)

    return future     # len == len(levels)

# --------------------------- DL network builder -----------------------
class DiffLogicNet(nn.Module):
    """
    Verilog netlist → stack of LogicLayer.
    단순 sequential feed-forward 구조를 갖추고, gate truth-table을
    weight로 고정(one-hot)하여 미분도 가능(softmax → argmax).
    """
    def __init__(self, verilog_file:str,
                 device:str='cuda',
                 connection_mode:str='unique', requires_grad=False, hard_init=True, verbose=False):
        super().__init__()
        #pis, pos, gates = parse_verilog(verilog_file)
        #_, _, vls, net_level = toposort_levels(gates)

        # ─ 파싱 → 게이트 리스트 취득
        pis , pos, gates = parse_verilog(verilog_file, debug=verbose)
        #nodes, pis, pos = build_nodes(gates)[:3]   # nodes, pis, pos
        levels, net_level = topo_sort_levels(gates, pos, debug=verbose)
        future_nets     = compute_future_uses(levels)
        self.requires_grad_ = requires_grad
        #print(levels, net_level)

        # level-0 입력 순서
        live_nets   = [ pi.name for pi in pis]
        self.layers            = nn.ModuleList()
        self.keep_idx_per_lvl  = []       # forward 때 사용 (A)
        self.new_dim_per_lvl   = []       # forward 때 사용 (B)
        self.in_dim = len(pis)
        self.out_dim = len(pos)
        for l, lvl_gates in enumerate(levels[1:-1], start=1):
            # ① 레벨 입력 매핑 새로 작성
            net2idx = {n:i for i,n in enumerate(live_nets)}
    
            ## some nets are just bypassing and not used at this level.
            ## so we can reduce in_dim further, but I think it is not a big overhead.
            in_dim  = len(live_nets)
            out_dim = len(lvl_gates)

            # ② 게이트 wiring
            a_idx, b_idx, opcodes = [], [], []
            for g in lvl_gates:
                ia = net2idx[g.ins[0]]
                ib = ia if (len(g.ins) < 2 or g.ins[1] is None) else net2idx[g.ins[1]]
                a_idx.append(ia); b_idx.append(ib)
                opcodes.append(GATE_OPCODE[g.op])

            layer = LogicLayer(in_dim, out_dim, device=device, connections='random', hard_weights=hard_init)
            layer.indices = (torch.tensor(a_idx, dtype=torch.int64, device=device),
                             torch.tensor(b_idx, dtype=torch.int64, device=device))
            layer.requires_grad_(requires_grad)
            if hard_init:
                with torch.no_grad():
                    w = torch.nn.functional.one_hot(torch.tensor(opcodes, device=device), 16).float()
                    layer.weights.copy_(w)
            else:
                # Initialize weights with gradients enabled when hard_init=False
                #w = torch.nn.functional.one_hot(torch.tensor(opcodes, device=device), 16).float()
                #layer.weights.data.copy_(w)
                # Ensure weights require gradients
                layer.weights.data.normal_(0, 0.2)
                layer.weights.requires_grad_(True)
            self.layers.append(layer)

            # ③ 다음 레벨을 위한 live_nets 계산
            keep_nets = [n for n in live_nets if n in future_nets[l]]
            new_nets  = [g.out for g in lvl_gates]
            self.keep_idx_per_lvl.append([net2idx[n] for n in keep_nets])  # (A)
            self.new_dim_per_lvl.append(out_dim)                           # (B)
            live_nets = keep_nets + new_nets                               # concat
            #print("###live", live_nets)


        # 기록용
        self.primary_inputs  = pis
        self.primary_outputs = pos
        self.net2idx = {n:i for i,n in enumerate(live_nets)}


    #def train(self, mode : bool = True):
    #    if self.requires_grad_:
    #        print("## TRAIN MODE!!")
    #        super().train(mode)# ─────────────────────────────────────────────────────────────
#  ResidualLogicNet  →  Verilog  (with VotingLayer support)
# ─────────────────────────────────────────────────────────────
def residual_net_to_verilog(residual_net, module_name="residual_logic"):
    """
    Convert a trained ResidualLogicNet network into synthesizable Verilog.
    - CrossbarLayer : assign 라우팅
    - LogicLayer    : primitive / custom gates (INDEX_TO_MODULE)
    - VotingLayer   : k-way majority using parameterised voter module
    """
    import math, textwrap
    verilog = []

    # ---------- module header ----------
    n_in   = residual_net.in_dim
    n_out  = residual_net.out_dim                      # == 클래스 수

    verilog.append(BAISIC_MODULES)
    header = "module {}(\n    ".format(module_name) + \
             ", ".join([f"input  wire in{i}" for i in range(n_in)] +
                       [f"output wire out{i}" for i in range(n_out)]) + ");"
    verilog.append(header)

    # ---------- internal nets ----------
    wire_cnt = 0
    def new_wire():
        nonlocal wire_cnt
        w = f"w{wire_cnt}"
        wire_cnt += 1
        return w

    # 첫 입력 벡터
    live_vec = [f"in{i}" for i in range(n_in)]

    # ---------- gate opcode → primitive/custom ----------
    INDEX_TO_MODULE = {
        0: "tie0",   1: "and",   2: "nimp",   3: "bypass0",
        4: "nrimp",  5: "bypass1", 6: "xor",  7: "or",
        8: "nor",    9: "xnor", 10: "nbypass1", 11: "rimp",
        12: "nbypass0", 13: "imp", 14: "nand", 15: "tie1"
    }

    # ---------- layer-by-layer ----------
    if residual_net.k_history_include_input:
        recent_logic = [[f"in{i}" for i in range(n_in)]]                          # for concat window
    else:
        recent_logic = []
    k_hist       = residual_net.k_history

    for lid, layer in enumerate(residual_net.layers):
        # ─── CrossbarLayer ────────────────────────────────────
        if isinstance(layer, CrossbarLayer):

            verilog.append(f" // --------------------------------")
            verilog.append(f" // Layer {lid} : {str(layer)} ")
            verilog.append(f" // --------------------------------")
            # concat last k logic outputs (or external inputs for first Xbar)
            if recent_logic:
                concat = [w for group in recent_logic[-k_hist:] for w in group]
            else:
                concat = live_vec
            live_vec = concat                                       # rename

            routes = layer.weights.detach().argmax(dim=1).tolist()
            next_vec = []
            for oid, src_idx in enumerate(routes):
                src_wire = live_vec[src_idx]
                dst_wire = new_wire()
                verilog.append(f"  assign {dst_wire} = {src_wire};")
                next_vec.append(dst_wire)
            live_vec = next_vec

        # ─── LogicLayer (binary gate array) ───────────────────
        elif isinstance(layer, LogicLayer):
            verilog.append(f" // --------------------------------")
            verilog.append(f" // Layer {lid} : {str(layer)} ")
            verilog.append(f" // --------------------------------")
            a_idx, b_idx = layer.indices
            next_vec = []
            for gid, (ia, ib) in enumerate(zip(a_idx.tolist(), b_idx.tolist())):
                A  = live_vec[ia]
                B  = live_vec[ib]
                Y  = new_wire()
                opcode = layer.weights[gid].argmax().item()
                prim   = INDEX_TO_MODULE.get(opcode, "and")
                if prim in ("and","or","xor","xnor","nand","nor"):
                    verilog.append(f"  {prim : <10} i_{lid}_{gid}({Y}, {A}, {B});")
                else:                            # custom truth-table gate
                    verilog.append(f"  {prim : <10} i_{lid}_{gid}({Y}, {A}, {B});")
                next_vec.append(Y)
            recent_logic.append(next_vec)
            live_vec = next_vec

        # ─── VotingLayer ──────────────────────────────────────
        elif isinstance(layer, VotingLayer):
            verilog.append(f" // --------------------------------")
            verilog.append(f" // Layer {lid} : {str(layer)} ")
            verilog.append(f" // --------------------------------")   
            k       = layer.k
            feats   = len(live_vec)
            g       = feats // k
            assert feats % k == 0, "VotingLayer: feats not divisible by k"

            for grp in range(k):
                seg = live_vec[grp*g : (grp+1)*g]      # g wires

                if g == 1:                             # 단일 입력 → 버퍼
                    verilog.append(f"  assign out{grp} = {seg[0]};")
                else:                                  # k-of-N voter
                    bus = "{" + ",".join(reversed(seg)) + "}"  # MSB:Lsb
                    verilog.append(f"  voter #(.N({g})) vote_{grp} ( .Y(out{grp}), .I({bus}) );")

        # ─── 기타(추가 모듈) 생략 ─────────────────────────────

    verilog.append("endmodule")
    code = "\n".join(verilog)
    with open(f"{module_name}.v", "w") as f:
        f.write(code)


# ─────────────────────────────────────────────────────────────
#  ResidualLogicNet  →  Verilog  (VotingLayer 간소화 지원)
# ─────────────────────────────────────────────────────────────
def residual_net_to_verilog(residual_net,
                            info: list | None = None,
                            module_name: str = "residual_logic"):
    """
    Convert a trained ResidualLogicNet into synthesizable Verilog.

    Parameters
    ----------
    residual_net : trained network object
    info         : list returned by `save_corr_plot_and_min_subset`
                   - info[ch] = (sign, subset) or None
                   - sign   ∈ {+1, -1}
                   - subset : list[int]  (feature indices within Voting group)
    module_name  : top-level Verilog module name
    """
    import math, textwrap, os

    verilog = []

    # ---------- module header ----------
    n_in  = residual_net.in_dim
    n_out = residual_net.out_dim

    verilog.append(BAISIC_MODULES)                     # tie0, voter … 정의
    header = "module {}(\n    ".format(module_name) + \
             ", ".join([f"input  wire in{i}"  for i in range(n_in)]  +
                       [f"output wire out{i}" for i in range(n_out)]) + ");"
    verilog.append(header)

    # ---------- helpers ----------
    wire_cnt = 0
    def new_wire():
        nonlocal wire_cnt
        w = f"w{wire_cnt}"
        wire_cnt += 1
        return w

    live_vec = [f"in{i}" for i in range(n_in)]         # 네트 리스트 진행용
    if residual_net.k_history_include_input:
        recent_logic = [live_vec.copy()]               # k-history 윈도용
    else:
        recent_logic = []

    # ---------- gate opcode → primitive/custom ----------
    INDEX_TO_MODULE = {
        0:"tie0",1:"and",2:"nimp",3:"bypass0",4:"nrimp",5:"bypass1",
        6:"xor",7:"or",8:"nor",9:"xnor",10:"nbypass1",11:"rimp",
        12:"nbypass0",13:"imp",14:"nand",15:"tie1"
    }

    # ---------- layer traversal ----------
    info_ptr = 0                                       # VotingLayer channel 인덱스
    k_hist   = residual_net.k_history

    for lid, layer in enumerate(residual_net.layers):

        # ── CrossbarLayer ───────────────────────────────
        if isinstance(layer, CrossbarLayer):
            verilog += [f" // --------------------------------",
                         f" // Layer {lid} : {str(layer)} ",
                         f" // --------------------------------"]
            concat = ([w for g in recent_logic[-k_hist:] for w in g]
                      if recent_logic else live_vec)
            live_vec = concat

            routes   = layer.weights.detach().argmax(dim=1).tolist()
            next_vec = []
            for oid, src_idx in enumerate(routes):
                dst = new_wire()
                verilog.append(f"  assign {dst} = {live_vec[src_idx]};")
                next_vec.append(dst)
            live_vec = next_vec

        # ── LogicLayer ─────────────────────────────────
        elif isinstance(layer, LogicLayer):
            verilog += [f" // --------------------------------",
                         f" // Layer {lid} : {str(layer)} ",
                         f" // --------------------------------"]
            a_idx, b_idx = layer.indices
            next_vec = []
            for gid, (ia, ib) in enumerate(zip(a_idx.tolist(), b_idx.tolist())):
                A, B, Y = live_vec[ia], live_vec[ib], new_wire()
                opcode = layer.weights[gid].argmax().item()
                prim   = INDEX_TO_MODULE.get(opcode, "and")
                verilog.append(f"  {prim:<10} i_{lid}_{gid}({Y}, {A}, {B});")
                next_vec.append(Y)
            recent_logic.append(next_vec)
            live_vec = next_vec

        # ── VotingLayer ────────────────────────────────
        elif isinstance(layer, VotingLayer):
            verilog += [f" // --------------------------------",
                         f" // Layer {lid} : {str(layer)} ",
                         f" // --------------------------------"]
            k      = layer.k
            feats  = len(live_vec)
            g      = feats // k
            assert feats % k == 0, "VotingLayer: feats not divisible by k"

            for grp in range(k):
                seg_all = live_vec[grp*g:(grp+1)*g]    # 해당 그룹 전체 입력
                # corr-SAT 분석 결과 (info) 매핑
                subset_info = None
                if info and info_ptr < len(info):
                    subset_info = info[info_ptr]
                info_ptr += 1

                # ---------- (1) 분석 정보가 없으면 원본 구현 ----------
                if subset_info is None or len(subset_info) == 0:
                    if g == 1:
                        verilog.append(f"  assign out{grp} = {seg_all[0]};")
                    else:
                        bus = "{" + ",".join(reversed(seg_all)) + "}"
                        verilog.append(
                            f"  voter #(.N({g})) vote_{grp} ( .Y(out{grp}), .I({bus}) );")
                    continue

                # ---------- (2) sign / subset 정보가 있을 때 ----------
                sign, subset = subset_info
                subset_wires = [seg_all[idx] for idx in subset]

                # -- subset 길이 1 : direct / NOT 연결 ---------------
                if len(subset_wires) == 1:
                    src = subset_wires[0]
                    if sign == 1:
                        verilog.append(f"  assign out{grp} = {src};")
                    else:
                        verilog.append(
                            f"  not i_inv_{lid}_{grp}(out{grp}, {src});")
                # -- subset 길이 >1 : voter + (선택적) NOT -------------
                else:
                    bus = "{" + ",".join(reversed(subset_wires)) + "}"
                    if sign == 1:
                        verilog.append(
                            f"  voter #(.N({len(subset_wires)})) vote_{grp} "
                            f"( .Y(out{grp}), .I({bus}) );")
                    else:
                        tmp = new_wire()
                        verilog.append(
                            f"  voter #(.N({len(subset_wires)})) vote_{grp} "
                            f"( .Y({tmp}), .I({bus}) );")
                        verilog.append(
                            f"  not i_inv_{lid}_{grp}(out{grp}, {tmp});")

        # ── (추가 레이어 타입은 필요 시 여기에) ─────────────────

    verilog.append("endmodule")

    code = "\n".join(verilog)
    with open(f"{module_name}.v", "w") as f:
        f.write(code)
    print(f"✔ Verilog saved to {module_name}.v")



"""
def residual_net_to_verilog(residual_net, module_name="residual_logic"):
    verilog_lines = []

    # Verilog module header
    n_in = residual_net.in_dim
    n_out = residual_net.out_dim

    verilog_lines.append(BAISIC_MODULES)

    header = f"module {module_name}(" + \
             ", ".join([f"input wire in{i}" for i in range(n_in)] + 
                       [f"output wire out{i}" for i in range(n_out)]) + ");"

    verilog_lines.append(header)

    wire_counter = 0
    wire_map = {}

    recent_logic = []
    current_inputs = [f"in{i}" for i in range(n_in)]

    for idx, layer in enumerate(residual_net.layers):
        if isinstance(layer, CrossbarLayer):
            verilog_lines.append(f" // --------------------------------")
            verilog_lines.append(f" // Layer {idx} : {str(layer)} ")
            verilog_lines.append(f" // --------------------------------")
            if recent_logic:
                current_inputs = [wire for logic_out in recent_logic[-residual_net.k_history:] for wire in logic_out]

            weight = layer.weights.detach().cpu()
            connections = weight.argmax(dim=1)

            crossbar_outputs = []
            for out_idx, in_idx in enumerate(connections):
                src_wire = current_inputs[in_idx]
                wire_name = f"w{wire_counter}"
                wire_counter += 1

                verilog_lines.append(f"  assign {wire_name} = {src_wire};")

                crossbar_outputs.append(wire_name)

            current_inputs = crossbar_outputs

        elif isinstance(layer, LogicLayer):
            next_outputs = []
            ops = layer.weights.argmax(dim=1)
            verilog_lines.append(f" // --------------------------------")
            verilog_lines.append(f" // Layer {idx} : {str(layer)} ")
            verilog_lines.append(f" // --------------------------------")
            for out_idx, (ia, ib) in enumerate(zip(layer.indices[0], layer.indices[1])):
                a_wire = current_inputs[ia]
                b_wire = current_inputs[ib]
                out_wire = f"w{wire_counter}"
                wire_counter += 1

                # For simplicity, assume LogicLayer gates are AND gates

                op = ops[out_idx].item()

                verilog_lines.append(f"  {INDEX_TO_MODULE[op] : <10} i_{idx}_{out_idx}({out_wire}, {a_wire}, {b_wire});")

                next_outputs.append(out_wire)

            recent_logic.append(next_outputs)
            current_inputs = next_outputs

        elif isinstance(layer, VotingLayer):
            for out_idx in range(n_out):
                in_wire = current_inputs[out_idx % len(current_inputs)]
                verilog_lines.append(f"  assign out{out_idx} = {in_wire};")

    verilog_lines.append("endmodule")

    code = "\n".join(verilog_lines)
    with open(f"{module_name}.v", "w") as f:
        f.write(code)
"""
    




# compiled_stages.py
import math
import torch
import numpy as np
from difflogic import LogicLayer, GroupSum
from difflogic.compiled_model import CompiledLogicNet, ALL_OPERATIONS, BITS_TO_DTYPE, BITS_TO_ZERO_LITERAL   # 기존 코드 재사용

# ============================================================
# Base Class
# ============================================================

class MiniLogicCompiler:
    """
    Minimal logic compiler for TreeConvStage.
    Produces a patch-loop function:

        void logic_tree_stage_X(
            const u8* inp,    // [num_patches][in_dim]
            u8* out,          // [num_patches][out_dim]
            size_t num_patches
        )

    Each patch is processed independently.
    """

    def __init__(self, logic_layers, num_bits=8):
        self.layers = []
        self.in_dim = None

        for ll in logic_layers:
            if self.in_dim is None:
                self.in_dim = ll.in_dim
            a = ll.indices[0]
            b = ll.indices[1]
            ops = ll.weights.argmax(1)
            self.layers.append((a, b, ops))

        # prefix sum for intermediate nodes
        self.prefix = []
        cur = 0
        for a, b, ops in self.layers:
            self.prefix.append(cur)
            cur += len(a)

        self.out_dim = len(self.layers[-1][0])
        self.num_bits = num_bits

    def get_op(self, a, b, op):
        name = ALL_OPERATIONS[op]
        if name == "zero": return f"{BITS_TO_ZERO_LITERAL[self.num_bits]}"
        if name == "and": return f"{a} & {b}"
        if name == "not_implies": return f"{a} & ~{b}"
        if name == "a": return f"{a}"
        if name == "not_implied_by": return f"{b} & ~{a}"
        if name == "b": return f"{b}"
        if name == "xor": return f"{a} ^ {b}"
        if name == "or": return f"{a} | {b}"
        if name == "not_or": return f"~({a} | {b})"
        if name == "not_xor": return f"~({a} ^ {b})"
        if name == "not_b": return f"~{b}"
        if name == "implied_by": return f"~{b} | {a}"
        if name == "not_a": return f"~{a}"
        if name == "implies": return f"~{a} | {b}"
        if name == "not_and": return f"~({a} & {b})"
        if name == "one": return f"~{BITS_TO_ZERO_LITERAL[self.num_bits]}"
        raise RuntimeError(name)

    def generate_c(self, func_name):
        in_dim = self.in_dim
        out_dim = self.out_dim

        code = []
        code.append(
f"""
void {func_name}(const u8* inp, u8* out, size_t num_patches)
{{
    for (size_t p = 0; p < num_patches; ++p) {{
        const u8* pin = inp + p * {in_dim};
        u8* pout = out + p * {out_dim};
"""
        )

        # Each layer
        for layer_id, (a_idx, b_idx, ops) in enumerate(self.layers):
            for j in range(len(a_idx)):
                if layer_id == 0:
                    A = f"pin[{a_idx[j]}]"
                    B = f"pin[{b_idx[j]}]"
                else:
                    base = self.prefix[layer_id - 1]
                    A = f"v{base + a_idx[j]}"
                    B = f"v{base + b_idx[j]}"

                op_expr = self.get_op(A, B, ops[j])
                out_idx = self.prefix[layer_id] + j

                if layer_id == len(self.layers) - 1:
                    # last layer -> write to pout
                    code.append(f"        pout[{j}] = {op_expr};")
                else:
                    code.append(
                        f"        u8 v{out_idx} = {op_expr};"
                    )

        code.append(
"""
    }
}
"""
        )
        return "\n".join(code)






class CompiledStage:
    """
    Base class for stage-aware compiler units.
    """
    def __init__(self, module, stage_id):
        self.module = module
        self.stage_id = stage_id

    def trace_shapes(self, Cin, H, W):
        raise NotImplementedError

    def generate_buffers(self, Cin, H, W):
        """
        Returns:
            buffer_plan = {...}
            Cout, Hout, Wout = updated shapes
        """
        raise NotImplementedError

    def generate_c_code(self, buffer_names):
        """
        Generate C code for this stage.
        buffer_names = {
            "inp": "...",
            "out": "...",
            "workspace": [...]
        }
        """
        raise NotImplementedError


# ============================================================
# Crossbar Stage
# ============================================================

class CompiledCrossbarStage(CompiledStage):
    def trace_shapes(self, Cin, H, W):
        # Crossbar1x1Conv는 crossbar 속성을 통해 접근
        crossbar_layer = self.module.crossbar
        Cout = crossbar_layer.out_dim
        return Cout, H, W

    def generate_buffers(self, Cin, H, W):
        Cout, Hout, Wout = self.trace_shapes(Cin, H, W)
        buf_size = Cout * Hout * Wout
        return {
            "type": "crossbar",
            "buf_size": buf_size,
            "Cin": Cin,
            "Cout": Cout,
            "H": H,
            "W": W
        }, Cout, Hout, Wout

    def generate_c_code(self, buffer_names):
        module = self.module
        # Crossbar1x1Conv는 crossbar 속성을 통해 접근
        crossbar_layer = module.crossbar
        idx = crossbar_layer.connection_indices.cpu().numpy().tolist()
        Cout = crossbar_layer.out_dim
        Cin = crossbar_layer.in_dim

        arr = ", ".join(str(i) for i in idx)

        return f"""
// === Crossbar Stage {self.stage_id} ===
static const int cross_idx_{self.stage_id}[{Cout}] = {{ {arr} }};

void run_crossbar_{self.stage_id}(const u8* inp, u8* out, size_t B)
{{
    size_t in_stride = {Cin};
    size_t out_stride = {Cout};

    for (size_t b = 0; b < B; b++) {{
        const u8* ib = inp + b * in_stride;
        u8* ob = out + b * out_stride;
        for (int i = 0; i < {Cout}; i++) {{
            ob[i] = ib[cross_idx_{self.stage_id}[i]];
        }}
    }}
}}
"""


# ============================================================
# TreeConv Stage
# ============================================================

class CompiledTreeStage(CompiledStage):
    """
    TreeConv = unfold → logic cascade → fold
    """

    def trace_shapes(self, Cin, H, W):
        K = self.module.kernel_size
        pad = self.module.padding
        stride = self.module.stride
        Cout = getattr(self.module, "out_dim", None) or self.module.out_channels
        Hout = (H + 2*pad - K) // stride + 1
        Wout = (W + 2*pad - K) // stride + 1
        return Cout, Hout, Wout

    def generate_buffers(self, Cin, H, W):
        Cout, Hout, Wout = self.trace_shapes(Cin, H, W)
        K = self.module.kernel_size

        unfold_dim = Cin * K * K
        num_patches = Hout * Wout

        return {
            "type": "tree",
            "Cin": Cin,
            "Cout": Cout,
            "H": H,
            "W": W,
            "Hout": Hout,
            "Wout": Wout,
            "patch_dim": unfold_dim,
            "num_patches": num_patches,
        }, Cout, Hout, Wout

    def generate_c_code(self, buffer_names):
        Cin = getattr(self.module, "in_dim", None) or self.module.in_channels
        Cout = getattr(self.module, "out_dim", None) or self.module.out_channels
        K = self.module.kernel_size
        pad = self.module.padding
        stride = self.module.stride

        mini = MiniLogicCompiler(self.module.cascade, num_bits=8)
        func_name = f"logic_tree_stage_{self.stage_id}"
        logic_code = mini.generate_c(func_name)

        return f"""
// ===== TreeConv Stage {self.stage_id} =====

void unfold_{self.stage_id}(const u8* inp, u8* out,
                            size_t B, int C, int H, int W,
                            int OH, int OW)
{{
    int P = C * {K} * {K};
    size_t pid = 0;

    for (size_t b=0;b<B;b++) {{
        const u8* ib = inp + b*C*H*W;
        for (int oh=0; oh<OH; oh++) {{
            for (int ow=0; ow<OW; ow++) {{
                for (int ic=0; ic<C; ic++) {{
                    for (int kh=0; kh<{K}; kh++) {{
                        for (int kw=0; kw<{K}; kw++) {{
                            int ih = oh*{stride} - {pad} + kh;
                            int iw = ow*{stride} - {pad} + kw;

                            u8 v = 0;
                            if (ih>=0 && ih<H && iw>=0 && iw<W)
                                v = ib[ic*H*W + ih*W + iw];

                            out[pid*P + ic*{K}*{K} + kh*{K} + kw] = v;
                        }}
                    }}
                }}
                pid++;
            }}
        }}
    }}
}}

{logic_code}

void fold_{self.stage_id}(const u8* patches, u8* out,
                          size_t B, int C, int OH, int OW)
{{
    size_t pid = 0;
    for (size_t b=0;b<B;b++) {{
        u8* ob = out + b*C*OH*OW;
        for (int oh=0; oh<OH; oh++) {{
            for (int ow=0; ow<OW; ow++) {{
                const u8* src = patches + pid*C;
                memcpy(ob + (oh*OW+ow)*C, src, C);
                pid++;
            }}
        }}
    }}
}}
"""


# ============================================================
# ORPool Stage
# ============================================================

class CompiledORPoolStage(CompiledStage):
    def trace_shapes(self, Cin, H, W):
        return Cin, H//2, W//2

    def generate_buffers(self, Cin, H, W):
        Cout, Hout, Wout = self.trace_shapes(Cin, H, W)
        buf_size = Cout * Hout * Wout
        return {
            "type": "orpool",
            "Cin": Cin,
            "Cout": Cout,
            "H": H,
            "W": W,
            "Hout": Hout,
            "Wout": Wout,
            "buf_size": buf_size
        }, Cout, Hout, Wout

    def generate_c_code(self, buffer_names):
        return f"""
// === ORPool Stage {self.stage_id} ===
void orpool_{self.stage_id}(const u8* inp, u8* out,
                            size_t B, int C, int H, int W)
{{
    int OH = H/2, OW = W/2;
    for (size_t b=0; b<B; b++) {{
        const u8* ib = inp + b*C*H*W;
        u8* ob = out + b*C*OH*OW;
        for (int c=0; c<C; c++) {{
            for (int oh=0; oh<OH; oh++) {{
                for (int ow=0; ow<OW; ow++) {{
                    int ih = oh*2;
                    int iw = ow*2;
                    u8 v =
                        ib[c*H*W + ih*W + iw] |
                        ib[c*H*W + ih*W + iw+1] |
                        ib[c*H*W + (ih+1)*W + iw] |
                        ib[c*H*W + (ih+1)*W + iw+1];
                    ob[c*OH*OW + oh*OW + ow] = v;
                }}
            }}
        }}
    }}
}}
"""


# ============================================================
# Classifier Stage (LogicLayers + GroupSum)
# ============================================================

class CompiledClassifierStage(CompiledStage):
    def __init__(self, logic_layers, group_layer, stage_id):
        super().__init__(group_layer, stage_id)
        self.logic_layers = logic_layers
        self.group_layer = group_layer

    def trace_shapes(self, Cin, H, W):
        """
        Classifier input shape is flattened: B × (Cin * H * W)
        Output is GroupSum(k)
        """
        flat_dim = Cin * H * W
        logic_out = self.logic_layers[-1].out_dim
        return flat_dim, logic_out, self.group_layer.k

    def generate_buffers(self, Cin, H, W):
        """
        Must match generate_buffers(self, Cin, H, W)
        """
        flat_dim = Cin * H * W
        logic_out_dim = self.logic_layers[-1].out_dim

        buffer_info = {
            "type": "classifier",
            "flat_dim": flat_dim,
            "logic_out_dim": logic_out_dim,
            "Cin": Cin,
            "H": H,
            "W": W,
            "Cout": self.group_layer.k,
        }

        # Return: buffer_info, Cout, Hout, Wout
        # Classifier outputs shape (B, k, ???)
        return buffer_info, logic_out_dim, 1, 1

    def generate_c_code(self, buffer_names):
        # Local logic compiler
        logic_model = torch.nn.Sequential(*self.logic_layers, self.group_layer)
        comp = CompiledLogicNet(logic_model, num_bits=8)
        return comp.get_c_code()


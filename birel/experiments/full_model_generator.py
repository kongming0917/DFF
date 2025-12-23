# ===============================================================
# full_model_generator.py — Stage-aware Full-Model C Code Builder
# ===============================================================

import torch
from compiled_stages import (
    CompiledTreeStage,
    CompiledCrossbarStage,
    CompiledORPoolStage,
    CompiledClassifierStage,
)


# ---------------------------------------------------------------
# Stage extraction
# ---------------------------------------------------------------

def extract_stages(model):
    stages = []
    stage_id = 0

    # ===== Features =====
    for m in model[0]:
        if hasattr(m, "cascade"):
            stages.append(CompiledTreeStage(m, stage_id))
        elif hasattr(m, "crossbar"):
            stages.append(CompiledCrossbarStage(m, stage_id))
        elif hasattr(m, "max_pool"):
            stages.append(CompiledORPoolStage(m, stage_id))
        else:
            # PASS-LAYER (Identity, etc.) → skip
            stage_id -= 1  # avoid gap
        stage_id += 1

    # ===== Classifier =====
    logic_layers = []
    group_layer = None
    for m in model[1]:
        if isinstance(m, torch.nn.Flatten):
            continue
        elif hasattr(m, "weights"):  # LogicLayer
            logic_layers.append(m)
        elif hasattr(m, "k"):  # GroupSum
            group_layer = m

    if group_layer is None:
        raise RuntimeError("Classifier must end with GroupSum")

    stages.append(CompiledClassifierStage(logic_layers, group_layer, stage_id))

    return stages


# ---------------------------------------------------------------
# Buffer Allocation
# ---------------------------------------------------------------

def allocate_buffers(stages, B, Cin, H, W):
    """
    Walk each stage and compute required memory buffers for C code.
    """
    buffers = []
    prev_C, prev_H, prev_W = Cin, H, W

    for stg in stages:
        buf, outC, outH, outW = stg.generate_buffers(prev_C, prev_H, prev_W)
        buf["buffer_name"] = f"buf_{stg.stage_id}"
        buffers.append(buf)
        prev_C, prev_H, prev_W = outC, outH, outW

    return buffers, (prev_C, prev_H, prev_W)


# ---------------------------------------------------------------
# Stage C Code Emission
# ---------------------------------------------------------------

def emit_stage_code(stages):
    """
    Concatenate all stage C functions:
        unfold_ID, logic_tree_stage_ID, fold_ID, crossbar_ID, orpool_ID, classifier logic
    """
    out = ""
    for stg in stages:
        out += stg.generate_c_code({})
        out += "\n\n"
    return out


# ---------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------


def emit_full_model_wiring(stages, buffers, B):
    """
    Emit C code for the full end-to-end pipeline:
        prev (input pointer)
        → Crossbar
        → TreeConv (unfold → logic → fold)
        → ORPool
        …
        → Classifier (flatten → CompiledLogicNet.apply)

    All intermediate memory allocations happen here.
    """
    code = []
    code.append("void full_model(const u8* inp, int* out, size_t B)\n{")
    code.append("    const u8* prev = inp;")

    for stg, buf in zip(stages, buffers):
        sid   = stg.stage_id
        stype = buf["type"]

        # ----------------------------------------------------
        # 1) CROSSBAR
        # ----------------------------------------------------
        if stype == "crossbar":
            Cin  = buf["Cin"]
            Cout = buf["Cout"]
            H    = buf["H"]
            W    = buf["W"]
            size = f"(B * {Cout} * {H} * {W})"

            code.append("")
            code.append(f"    // --- Crossbar Stage {sid} ---")
            code.append(f"    u8* buf_{sid} = (u8*)malloc({size});")
            code.append(f"    run_crossbar_{sid}(prev, buf_{sid}, B);")
            code.append(f"    prev = buf_{sid};")


        # ----------------------------------------------------
        # 2) TREE CONV
        # ----------------------------------------------------
        elif stype == "tree":
            Cin        = buf["Cin"]
            Cout       = buf["Cout"]
            H          = buf["H"]
            W          = buf["W"]
            Hout       = buf["Hout"]
            Wout       = buf["Wout"]
            patch_dim  = buf["patch_dim"]        # Cin * K*K
            num_patches = buf["num_patches"]     # Hout * Wout

            # logic output dimension
            logic_out_dim = stg.module.cascade[-1].out_dim

            size_unf  = f"(B * {num_patches} * {patch_dim})"
            size_log  = f"(B * {num_patches} * {logic_out_dim})"
            size_fold = f"(B * {Cout} * {Hout} * {Wout})"

            code.append("")
            code.append(f"    // --- TreeConv Stage {sid} ---")
            code.append(f"    u8* buf_unf_{sid} = (u8*)malloc({size_unf});")
            code.append(f"    u8* buf_log_{sid} = (u8*)malloc({size_log});")
            code.append(f"    u8* buf_{sid}     = (u8*)malloc({size_fold});")

            # Call unfold
            code.append(
                f"    unfold_{sid}(prev, buf_unf_{sid}, B, {Cin}, {H}, {W}, {Hout}, {Wout});"
            )

            # Logic: process patches
            code.append(
                f"    logic_tree_stage_{sid}(buf_unf_{sid}, buf_log_{sid}, B * {num_patches});"
            )

            # Fold
            code.append(
                f"    fold_{sid}(buf_log_{sid}, buf_{sid}, B, {Cout}, {Hout}, {Wout});"
            )

            # free temporary
            code.append(f"    free(buf_unf_{sid});")
            code.append(f"    free(buf_log_{sid});")

            code.append(f"    prev = buf_{sid};")

        # ----------------------------------------------------
        # 3) ORPOOL
        # ----------------------------------------------------
        elif stype == "orpool":
            Cin  = buf["Cin"]
            H    = buf["H"]
            W    = buf["W"]
            Hout = buf["Hout"]
            Wout = buf["Wout"]

            size = f"(B * {Cin} * {Hout} * {Wout})"

            code.append("")
            code.append(f"    // --- ORPool Stage {sid} ---")
            code.append(f"    u8* buf_{sid} = (u8*)malloc({size});")
            code.append(
                f"    orpool_{sid}(prev, buf_{sid}, B, {Cin}, {H}, {W});"
            )
            code.append(f"    prev = buf_{sid};")

        # ----------------------------------------------------
        # 4) CLASSIFIER (flatten → logicnet apply)
        # ----------------------------------------------------
        elif stype == "classifier":
            flat_dim      = buf["flat_dim"]        # Cin*H*W
            logic_out_dim = buf["logic_out_dim"]
            Cout          = buf["Cout"]            # k classes

            code.append("")
            code.append(f"    // --- Classifier Stage (GroupSum) ---")
            code.append(f"    size_t flat_dim = {flat_dim};")
            code.append(f"    size_t flat_bytes = B * flat_dim;")

            code.append(f"    u8* flat_u8 = (u8*)malloc(flat_bytes);")

            # Flatten copy
            code.append("    for (size_t b=0; b<B; b++) {")
            code.append("        memcpy(flat_u8 + b*flat_dim, prev + b*flat_dim, flat_dim);")
            code.append("    }")

            # u8 → bool
            code.append("    bool* flat_bool = (bool*)malloc(flat_bytes * sizeof(bool));")
            code.append("    for (size_t i=0; i<B*flat_dim; i++) flat_bool[i] = flat_u8[i] ? true : false;")

            # Call compiled classifier
            code.append("    apply_logic_gate_net(flat_bool, out, B);")

            code.append("    free(flat_u8);")
            code.append("    free(flat_bool);")

    code.append("}")
    return "\n".join(code)




# ---------------------------------------------------------------
# Main Codegen Entry
# ---------------------------------------------------------------

def generate_full_model_c(model, B, Cin, H, W, output_path):
    stages = extract_stages(model)
    buffers, final_shape = allocate_buffers(stages, B, Cin, H, W)

    c = ""
    c += "#include <stdlib.h>\n"
    c += "#include <string.h>\n"
    c += "#include <stdint.h>\n"
    c += "typedef uint8_t u8;\n"
    c += "typedef int32_t i32;\n\n"

    c += emit_stage_code(stages)
    c += emit_full_model_wiring(stages, buffers, B)

    with open(output_path, "w") as f:
        f.write(c)

    print(f"[DONE] Generated {output_path}")
    return output_path

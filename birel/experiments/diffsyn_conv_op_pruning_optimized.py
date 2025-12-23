import argparse
import torch
import torch.nn as nn
import os
import math

# --- User Environment Imports ---
from difflogic import LogicLayer, GroupSum, WeightedGroupSum, PrunedGroupSum, PrunedWeightedGroupSum
from birel.model import *
from birel.pruning import *
from birel.conv import *
from birel.utils import *
from conv_difflogic import ORPool2d

# ==============================================================================
# 0. Verilog Primitives & Helpers
# ==============================================================================
def get_gate_modules():
    return """
`timescale 1ns / 1ps

// Popcount Adder Module
module popcount_adder #(
    parameter WIDTH = 16,
    parameter COUNT_WIDTH = 5
)(
    input wire [WIDTH-1:0] in,
    output wire [COUNT_WIDTH-1:0] sum
);
    integer i;
    reg [COUNT_WIDTH-1:0] count;
    always @(*) begin
        count = 0;
        for (i = 0; i < WIDTH; i = i + 1) begin
            count = count + in[i];
        end
    end
    assign sum = count;
endmodule

// Basic Gates (User Naming Convention)
module tie0 (output wire y, input wire a, input wire b); assign y = 1'b0; endmodule
module tie1 (output wire y, input wire a, input wire b); assign y = 1'b1; endmodule
module and_g (output wire y, input wire a, input wire b); assign y = a & b; endmodule // Avoid Verilog keyword conflict
module or_g (output wire y, input wire a, input wire b); assign y = a | b; endmodule
module xor_g (output wire y, input wire a, input wire b); assign y = a ^ b; endmodule
module nand_g (output wire y, input wire a, input wire b); assign y = ~(a & b); endmodule
module nor_g (output wire y, input wire a, input wire b); assign y = ~(a | b); endmodule
module xnor_g (output wire y, input wire a, input wire b); assign y = ~(a ^ b); endmodule
module imp (output wire y, input wire a, input wire b); assign y = (~a) | b; endmodule
module nimp (output wire y, input wire a, input wire b); assign y = a & (~b); endmodule
module rimp (output wire y, input wire a, input wire b); assign y = a | (~b); endmodule
module nrimp (output wire y, input wire a, input wire b); assign y = (~a) & b; endmodule
module bypass0 (output wire y, input wire a, input wire b); assign y = a; endmodule
module bypass1 (output wire y, input wire a, input wire b); assign y = b; endmodule
module nbypass0 (output wire y, input wire a, input wire b); assign y = ~a; endmodule
module nbypass1 (output wire y, input wire a, input wire b); assign y = ~b; endmodule
"""

# Map op_code to module name
INDEX_TO_MODULE = {
    0:"tie0", 1:"and_g", 2:"nimp", 3:"bypass0", 4:"nrimp", 5:"bypass1", 
    6:"xor_g", 7:"or_g", 8:"nor_g", 9:"xnor_g", 10:"nbypass1", 
    11:"rimp", 12:"nbypass0", 13:"imp", 14:"nand_g", 15:"tie1"
}

# ==============================================================================
# 1. Conv Logic Layer Generator (Modular)
# ==============================================================================
def generate_conv_logic_module(logic_layer, block_idx, layer_idx, prev_width, curr_width):
    module_name = f"Conv{block_idx}_logic{layer_idx}"
    verilog = []
    
    # Define Parameter Names based on layer index
    # Layer 0: IN=CROSSBAR_OUTPUT_WIDTH, OUT=L1_OUTPUT_WIDTH
    # Layer 1: IN=L1_OUT_WIDTH, OUT=L2_OUT_WIDTH
    if layer_idx == 0:
        p_in_name = "CROSSBAR_OUTPUT_WIDTH"
        p_out_name = "L1_OUTPUT_WIDTH"
        in_port = "in"
    else:
        p_in_name = f"L{layer_idx}_OUT_WIDTH"
        p_out_name = f"L{layer_idx+1}_OUT_WIDTH"
        in_port = f"w_{block_idx}_L{layer_idx-1}_n"

    out_port = f"w_{block_idx}_L{layer_idx}_n"

    verilog.append(f"module {module_name} #(")
    verilog.append(f"  parameter {p_in_name} = {prev_width},")
    verilog.append(f"  parameter {p_out_name} = {curr_width}")
    verilog.append(f")(")
    verilog.append(f"    input [{p_in_name}-1:0] {in_port},")
    verilog.append(f"    output [{p_out_name}-1:0] {out_port}")
    verilog.append(f");")
    
    verilog.append(f"\n// Logic Layer {layer_idx}: {prev_width} -> {curr_width}")
    
    indices = logic_layer.indices
    weights = logic_layer.weights.argmax(dim=-1).cpu().numpy()
    
    for out_i in range(curr_width):
        idx_a = indices[0][out_i].item()
        idx_b = indices[1][out_i].item()
        op_code = weights[out_i]
        gate_type = INDEX_TO_MODULE.get(op_code, "and_g")
        
        inst_name = f"inst_{out_port}_{out_i}"
        
        # Access bits
        conn_a = f"{in_port}[{idx_a}]"
        conn_b = f"{in_port}[{idx_b}]"
        res_wire = f"{out_port}[{out_i}]"
        
        verilog.append(f"    {gate_type} {inst_name} ({res_wire}, {conn_a}, {conn_b});")

    verilog.append("endmodule\n")
    return "\n".join(verilog), module_name, in_port, out_port

# ==============================================================================
# 2. Crossbar Generator (Modular)
# ==============================================================================
def generate_crossbar_module(crossbar_layer, block_idx, kernel_size=3):
    module_name = f"crossbar1x1conv_{block_idx}"
    in_channels = crossbar_layer.in_channels
    out_channels = crossbar_layer.out_channels
    indices = crossbar_layer.crossbar.connection_indices.cpu().numpy()
    
    filter_area = kernel_size * kernel_size
    in_total_width = in_channels * filter_area
    out_total_width = out_channels * filter_area

    verilog = []
    verilog.append(f"module {module_name} #(")
    verilog.append(f"  parameter SIZE_PER_FIL = {in_total_width},")
    verilog.append(f"  parameter CROSSBAR_OUTPUT_WIDTH = {out_total_width}")
    verilog.append(f")(")
    verilog.append(f"  input [SIZE_PER_FIL-1:0] data_out,")
    verilog.append(f"  output [CROSSBAR_OUTPUT_WIDTH-1:0] data_arr")
    verilog.append(f");")
    
    # Logic: data_arr[out_ch] <== data_out[in_ch] (block copy)
    for out_c in range(out_channels):
        in_c = indices[out_c]
        start_out = out_c * filter_area
        end_out = start_out + filter_area - 1
        start_in = in_c * filter_area
        end_in = start_in + filter_area - 1
        verilog.append(f"  assign data_arr[{end_out}:{start_out}] = data_out[{end_in}:{start_in}];")
        
    verilog.append("endmodule\n")
    return "\n".join(verilog), module_name

# ==============================================================================
# 3. Conv Wrapper Generator
# ==============================================================================
def generate_conv_wrapper(block_idx, crossbar_mod_name, logic_modules_info, in_ch, out_ch, kernel_size=3):
    module_name = f"conv_block{block_idx}"
    verilog = []
    
    # [수정] 블록 번호에 따른 파라미터 이름 생성 (IN_CH1, IN_CH2, ...)
    p_in_ch = f"IN_CH{block_idx}"
    p_out_ch = f"OUT_CH{block_idx}"

    verilog.append(f"module {module_name} #(")
    verilog.append(f"    parameter FILTER_SIZE = {kernel_size},")
    verilog.append(f"    parameter {p_in_ch} = {in_ch},")
    verilog.append(f"    parameter {p_out_ch} = {out_ch}")
    verilog.append(f")(")
    verilog.append(f"    input [FILTER_SIZE*FILTER_SIZE*{p_in_ch} - 1:0] data_out,")
    verilog.append(f"    output [{p_out_ch} - 1:0] conv{block_idx}_out")
    verilog.append(f");")

    verilog.append(f"\n    localparam FILTER_AREA = FILTER_SIZE * FILTER_SIZE;")
    verilog.append(f"    localparam SIZE_PER_FIL = FILTER_AREA * {p_in_ch};")
    
    # Calculate widths for parameters
    cb_out_width = logic_modules_info[0][3]
    verilog.append(f"    localparam CROSSBAR_OUTPUT_WIDTH = {cb_out_width};")
    for i, (_, _, _, _, out_w) in enumerate(logic_modules_info):
        verilog.append(f"    localparam L{i+1}_OUTPUT_WIDTH = {out_w};") # L1_OUTPUT_WIDTH, etc.

    # 1. Crossbar
    verilog.append(f"\n    /******************************************")
    verilog.append(f"    ********** data unrolling *****************")
    verilog.append(f"    ******************************************/")
    verilog.append(f"    wire [CROSSBAR_OUTPUT_WIDTH-1:0] data_arr;")
    verilog.append(f"\n    {crossbar_mod_name} #(")
    verilog.append(f"        .SIZE_PER_FIL(SIZE_PER_FIL), .CROSSBAR_OUTPUT_WIDTH(CROSSBAR_OUTPUT_WIDTH)")
    verilog.append(f"        ) crossbar_{block_idx} (.data_out(data_out), .data_arr(data_arr));")

    # 2. Logic Layers
    prev_wire = "data_arr"
    
    for i, (l_mod_name, l_in_port, l_out_port, l_in_w, l_out_w) in enumerate(logic_modules_info):
        verilog.append(f"\n    /******************************************")
        verilog.append(f"    ********** Logic Layer {i} ******************")
        verilog.append(f"    ******************************************/")
        
        curr_wire = f"L{i+1}_out"
        verilog.append(f"    wire [L{i+1}_OUTPUT_WIDTH-1:0] {curr_wire};")
        
        # Instantiate
        # Layer 0 params: .CROSSBAR_OUTPUT_WIDTH(...), .L1_OUTPUT_WIDTH(...)
        # Layer >0 params: .Li_OUT_WIDTH(...), .Li+1_OUT_WIDTH(...)
        
        if i == 0:
            p_in = "CROSSBAR_OUTPUT_WIDTH"
            p_out = "L1_OUTPUT_WIDTH"
        else:
            p_in = f"L{i}_OUT_WIDTH"
            p_out = f"L{i+1}_OUT_WIDTH"
            
        # Parameter value mapping (Localparams)
        val_in = "CROSSBAR_OUTPUT_WIDTH" if i == 0 else f"L{i}_OUTPUT_WIDTH"
        val_out = f"L{i+1}_OUTPUT_WIDTH"
        
        verilog.append(f"\n    {l_mod_name} #(")
        verilog.append(f"        .{p_in}({val_in}), .{p_out}({val_out})")
        verilog.append(f"        ) layer{i} (")
        verilog.append(f"        .{l_in_port}({prev_wire}), .{l_out_port}({curr_wire})")
        verilog.append(f"    );")
        
        prev_wire = curr_wire

    # Final Assign
    verilog.append(f"\n    assign conv{block_idx}_out = {prev_wire};")
    verilog.append(f"\nendmodule")
    return "\n".join(verilog)

# ==============================================================================
# 2. Classifier Part Generators (NEW)
# ==============================================================================

def get_classifier_param_names(layer_idx):
    """Returns (param_in_name, param_out_name) based on layer index"""
    if layer_idx == 0:
        return "CLASSIFIER_IN_DIM", "FC1_OUT_DIM"
    elif layer_idx == 1:
        return "FC2_IN_DIM", "FC2_OUT_DIM"
    elif layer_idx == 2:
        return "FC3_IN_DIM", "FC3_OUT_DIM"
    else:
        # Fallback for deeper networks
        return f"FC{layer_idx+1}_IN_DIM", f"FC{layer_idx+1}_OUT_DIM"

def get_classifier_wire_names(layer_idx):
    """Returns (in_wire_name, out_wire_name) for inside the module definition"""
    if layer_idx == 0:
        return "classifier_input", "w_1_n"
    elif layer_idx == 1:
        return "w_1_n", "w_3_n"
    elif layer_idx == 2:
        return "w_3_n", "w_5_n"
    else:
        return f"w_{2*layer_idx-1}_n", f"w_{2*layer_idx+1}_n"

def generate_classifier_logic_module(logic_layer, layer_idx, prev_width, curr_width):
    module_name = f"Classifier_layer{layer_idx}"
    p_in, p_out = get_classifier_param_names(layer_idx)
    in_port, out_port = get_classifier_wire_names(layer_idx)
    
    verilog = []
    verilog.append(f"module {module_name}#(")
    verilog.append(f"    parameter {p_in} = {prev_width},")
    verilog.append(f"    parameter {p_out} = {curr_width}")
    verilog.append(f")(")
    verilog.append(f"    input [{p_in} - 1:0] {in_port},")
    verilog.append(f"    output [{p_out} - 1:0] {out_port}")
    verilog.append(f"    );")
    
    verilog.append(f"\n// --- {2*layer_idx + 1}: LogicLayer ---")
    
    indices = logic_layer.indices
    weights = logic_layer.weights.argmax(dim=-1).cpu().numpy()
    
    for out_i in range(curr_width):
        idx_a = indices[0][out_i].item()
        idx_b = indices[1][out_i].item()
        op_code = weights[out_i]
        gate_type = INDEX_TO_MODULE.get(op_code, "and_g")
        inst_name = f"inst_{out_port}_{out_i}"
        
        verilog.append(f"    {gate_type} {inst_name} ({out_port}[{out_i}], {in_port}[{idx_a}], {in_port}[{idx_b}]);")

    verilog.append("endmodule\n")
    return "\n".join(verilog), module_name, p_in, p_out, in_port, out_port

def generate_groupsum_module(sum_layer, prev_width, n_classes=10):
    module_name = "GroupSum"
    # Parameter matches the output of the last logic layer
    p_in_dim = "FC3_OUT_DIM" # Assuming 3 layer structure usually
    
    verilog = []
    verilog.append(f"module {module_name}#(")
    verilog.append(f"  parameter {p_in_dim} = {prev_width}")
    verilog.append(f")(")
    verilog.append(f"  input wire [{p_in_dim} - 1 : 0] w_5_n,")
    
    # Generate score outputs
    scores = [f"score_{i}" for i in range(n_classes)]
    verilog.append(f"  output wire [12:0] {', '.join(scores)}")
    verilog.append(f");")
    
    verilog.append(f"// --- Final GroupSum & Argmax ---")

    # Processing GroupSum logic
    # Need to reconstruct inputs from w_5_n
    # GroupSum Logic is basically slicing w_5_n
    
    if isinstance(sum_layer, PrunedGroupSum):
        group_sizes_list = sum_layer.group_sizes.tolist()
    else:
        group_size = prev_width // sum_layer.k
        group_sizes_list = [group_size] * sum_layer.k

    current_pos = 0
    # WeightedGroupSum support
    integer_weights = None
    if isinstance(sum_layer, WeightedGroupSum):
        with torch.no_grad():
            integer_weights = sum_layer.weight_raw.round().int().tolist()

    for grp_idx in range(n_classes):
        curr_group_size = int(group_sizes_list[grp_idx])
        
        # Access inputs from w_5_n
        # w_5_n is a vector. We need to grab the bits corresponding to this group.
        # Note: PyTorch logic and Verilog vector indexing need to match.
        # In difflogic, we assume sequential: [Group0][Group1]...
        
        # Create a list of wire bits for this group
        group_bits = []
        for i in range(curr_group_size):
            group_bits.append(f"w_5_n[{current_pos + i}]")
        
        sum_terms = []
        if isinstance(sum_layer, WeightedGroupSum):
            group_weights = integer_weights[grp_idx]
            # Warning: Weights logic might be complex if sizes differ, assuming simple mapping
            # This part is simplified; usually PrunedGroupSum is used for these optimized models
            pass 
        else:
            sum_terms = group_bits

        if sum_terms:
            max_val = len(sum_terms)
            bit_width = 13 # Fixed to 13 as per user request (output wire [12:0])
            
            concat_name = f"pc_in_{grp_idx}"
            verilog.append(f"    wire [{max_val}:0] {concat_name};")
            verilog.append(f"    assign {concat_name} = {{{', '.join(reversed(sum_terms))}}};") # Reversed for MSB..LSB concatenation syntax
            
            # Popcount
            verilog.append(f"    popcount_adder #(.WIDTH({len(sum_terms)}), .COUNT_WIDTH({bit_width})) pc_{grp_idx} (.in({concat_name}[{max_val}-1:0]), .sum(score_{grp_idx}));")

        current_pos += curr_group_size

    verilog.append("endmodule\n")
    return "\n".join(verilog)

def generate_classifier_wrappers(logic_layers_info, groupsum_info, n_classes=10):
    # logic_layers_info: list of (module_name, p_in, p_out, in_port, out_port, width_in, width_out)
    
    wrappers = []
    
    # --- 1. Classifier_block_layer01 (Layer 0 + Layer 1) ---
    # Only if we have at least 2 layers
    if len(logic_layers_info) >= 2:
        l0 = logic_layers_info[0]
        l1 = logic_layers_info[1]
        
        mod_name = "Classifier_block_layer01"
        verilog = []
        verilog.append(f"module {mod_name} #(")
        verilog.append(f"    parameter {l0[1]} = {l0[5]},") # CLASSIFIER_IN_DIM
        verilog.append(f"    parameter {l0[2]} = {l0[6]},") # FC1_OUT_DIM
        verilog.append(f"    parameter {l1[1]} = {l1[5]},") # FC2_IN_DIM
        verilog.append(f"    parameter {l1[2]} = {l1[6]}")  # FC2_OUT_DIM
        verilog.append(f")(")
        verilog.append(f"    input [{l0[1]} - 1:0] classifier_input,")
        verilog.append(f"    output [{l1[2]} - 1:0] L2_out")
        verilog.append(f"    );")
        
        verilog.append(f"\n    /*******************************************************")
        verilog.append(f"    ******************** Logic layer 0 *********************")
        verilog.append(f"    *******************************************************/")
        verilog.append(f"    wire [{l0[2]} - 1:0] L1_out;")
        verilog.append(f"    {l0[0]} #(.{l0[1]}({l0[1]}), .{l0[2]}({l0[2]}))layer0(.{l0[3]}(classifier_input), .{l0[4]}(L1_out));")
        
        verilog.append(f"\n    /*******************************************************")
        verilog.append(f"    ******************** Logic layer 1 *********************")
        verilog.append(f"    *******************************************************/")
        verilog.append(f"    {l1[0]} #(.{l1[1]}({l1[1]}), .{l1[2]}({l1[2]})) layer1(.{l1[3]}(L1_out), .{l1[4]}(L2_out));")
        
        verilog.append("endmodule\n")
        wrappers.append("\n".join(verilog))

    # --- 2. Classifier_block (Full) ---
    mod_name = "Classifier_block"
    verilog = []
    verilog.append(f"module {mod_name} #(")
    
    # 1. 파라미터들을 먼저 리스트에 수집 (중복 제거 포함)
    param_list = []
    seen_params = set()
    
    for info in logic_layers_info:
        p_in = f"    parameter {info[1]} = {info[5]}"
        if p_in not in seen_params:
            param_list.append(p_in)
            seen_params.add(p_in)
        
        p_out = f"    parameter {info[2]} = {info[6]}"
        if p_out not in seen_params:
            param_list.append(p_out)
            seen_params.add(p_out)
            
    # 2. 수집된 파라미터들을 콤마로 연결 (마지막 요소 뒤에는 콤마가 안 붙음!)
    verilog.append(",\n".join(param_list))
    
    verilog.append(f")(")
    verilog.append(f"    input [{logic_layers_info[0][1]} - 1:0] classifier_input,")
    
    class_outs = [f"class_out{i}" for i in range(n_classes)]
    verilog.append(f"    output {', '.join(class_outs)}")
    verilog.append(f");")
    
    # Instantiate Layers
    curr_wire = "classifier_input"
    for i, info in enumerate(logic_layers_info):
        mod, p_in, p_out, port_in, port_out, w_in, w_out = info
        verilog.append(f"\n    /*******************************************************")
        verilog.append(f"    ******************** Logic layer {i} *********************")
        verilog.append(f"    *******************************************************/")
        
        if i == len(logic_layers_info) - 1:
            # Last layer output name fixed to w_5_n for GroupSum connection
            next_wire = "w_5_n"
            verilog.append(f"    wire [{p_out} - 1:0] {next_wire};")
        else:
            next_wire = f"L{i+1}_out" # L1_out, L2_out...
            verilog.append(f"    wire [{p_out} - 1:0] {next_wire};")
            
        verilog.append(f"    {mod} #(.{p_in}({p_in}), .{p_out}({p_out}))layer{i}(.{port_in}({curr_wire}), .{port_out}({next_wire}));")
        curr_wire = next_wire

    # Instantiate GroupSum
    verilog.append(f"\n    /*******************************************************")
    verilog.append(f"    ****************** Score Calculation *******************")
    verilog.append(f"    *******************************************************/")
    
    scores = [f"score_{i}" for i in range(n_classes)]
    verilog.append(f"    wire [12:0] {', '.join(scores)};")
    
    # Map Score Ports
    score_ports = [f".score_{i}(score_{i})" for i in range(n_classes)]
    last_p_out = logic_layers_info[-1][2]
    verilog.append(f"    GroupSum #(.{last_p_out}({last_p_out})) groupsum(.w_5_n(w_5_n), {', '.join(score_ports)});")
    
    # Argmax Logic
    verilog.append(f"\n    /*******************************************************")
    verilog.append(f"    **** Argmax Logic (Winner-Take-All for Multi-class) ****")
    verilog.append(f"    *******************************************************/")
    
    for i in range(n_classes):
        comps = []
        for j in range(n_classes):
            if i == j: continue
            comps.append(f"(score_{i} >= score_{j})")
        verilog.append(f"    assign class_out{i} = &{{{', '.join(comps)}}};")
        
    verilog.append("endmodule\n")
    wrappers.append("\n".join(verilog))
    
    return "\n".join(wrappers)


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def convert_model(model, output_filename, dataset="cifar10"):
    all_verilog = []
    all_verilog.append(get_gate_modules())
    
    # 1. Process Feature Extractor (Conv Blocks)
    print(f"\n[Verilog Gen] Processing Features (Conv Blocks)...")
    features = model[0]
    block_idx = 1
    
    # Logic to iterate standard sequential model
    children = list(features.children())
    i = 0
    final_conv_out_channels = 0
    
    while i < len(children):
        module = children[i]
        
        # Identify ConvBlock start
        if isinstance(module, Crossbar1x1Conv):
            print(f"  - Generating ConvBlock {block_idx}...")
            crossbar_layer = module
            
            # Find next TreeConvLayer
            tree_layer = None
            if i + 1 < len(children) and isinstance(children[i+1], (TreeConvLayer, FusedTreeConvLayer)):
                tree_layer = children[i+1]
                i += 1 
            else:
                print("    Warning: Standalone Crossbar found, skipping.")
                i += 1
                continue
            
            # --- Generate Modular Parts ---
            k_size = tree_layer.kernel_size if hasattr(tree_layer, 'kernel_size') else 3
            
            # A. Crossbar Module
            cb_code, cb_name = generate_crossbar_module(crossbar_layer, block_idx, k_size)
            all_verilog.append(cb_code)
            
            # B. Logic Modules
            logic_modules_info = [] # (name, in_port, out_port, in_w, out_w)
            prev_width = tree_layer.cascade[0].in_dim
            
            for l_idx, logic_layer in enumerate(tree_layer.cascade):
                curr_width = logic_layer.out_dim
                ll_code, ll_name, ll_in, ll_out = generate_conv_logic_module(
                    logic_layer, block_idx, l_idx, prev_width, curr_width
                )
                all_verilog.append(ll_code)
                logic_modules_info.append((ll_name, ll_in, ll_out, prev_width, curr_width))
                prev_width = curr_width
            
            # C. Wrapper Module
            in_ch = crossbar_layer.in_channels
            out_ch = tree_layer.cascade[-1].out_dim
            wrapper_code = generate_conv_wrapper(
                block_idx, cb_name, logic_modules_info, in_ch, out_ch, k_size
            )
            all_verilog.append(wrapper_code)
            
            final_conv_out_channels = out_ch
            block_idx += 1
            
        i += 1
        
    # 2. Classifier
    print(f"[Verilog Gen] Processing Classifier (Modular Mode)...")
    classifier = model[1]
    
    logic_layers = []
    sum_layer = None
    
    for m in classifier.modules():
        if isinstance(m, LogicLayer):
            logic_layers.append(m)
        elif isinstance(m, (GroupSum, PrunedGroupSum, WeightedGroupSum)):
            sum_layer = m

    # Determine input dim for first layer (based on previous logic or model)
    if final_conv_out_channels > 0:
        if dataset == "cifar10": sp_dim = 2*2
        elif dataset == "mnist": sp_dim = 4*4
        else: sp_dim = 1
        clf_input_dim = final_conv_out_channels * sp_dim
    else:
        clf_input_dim = logic_layers[0].in_dim

    # Generate Logic Modules
    clf_logic_info = [] # (name, p_in, p_out, in_port, out_port, w_in, w_out)
    prev_w = clf_input_dim
    
    for idx, layer in enumerate(logic_layers):
        curr_w = layer.out_dim
        code, name, p_in, p_out, port_in, port_out = generate_classifier_logic_module(layer, idx, prev_w, curr_w)
        all_verilog.append(code)
        clf_logic_info.append((name, p_in, p_out, port_in, port_out, prev_w, curr_w))
        prev_w = curr_w

    # Generate GroupSum Module
    n_classes = sum_layer.k
    all_verilog.append(generate_groupsum_module(sum_layer, prev_w, n_classes))
    
    # Generate Wrappers
    all_verilog.append(generate_classifier_wrappers(clf_logic_info, sum_layer, n_classes))

    with open(output_filename, "w") as f:
        f.write("\n".join(all_verilog))
    print(f"✔ Verilog saved to {output_filename}")

# --- Loaders ---
def load_conv_model(model_path, device="cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--module_name', type=str, default="difflogic_net")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--output-dir', type=str, default="../conv_syn/generated/", help="Directory to save the generated Verilog file")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_conv_model(args.model_path, device=device)
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Construct full output path
    output_file = os.path.join(args.output_dir, f"{args.module_name}.v")
    
    convert_model(model, output_file, args.dataset)

if __name__ == '__main__':
    main()
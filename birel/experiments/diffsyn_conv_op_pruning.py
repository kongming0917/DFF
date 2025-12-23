import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

from difflogic import LogicLayer, GroupSum, WeightedGroupSum, PrunedGroupSum, PrunedWeightedGroupSum

from birel.model import *
from birel.pruning import *
from birel.conv import *
from birel.utils import *
from birel.verilog import *
import itertools
import math

from conv_difflogic import ORPool2d


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

# ==============================================================================
# ▼ 3. 모델 로딩 및 평탄화 함수 ▼
# ==============================================================================
def load_conv_model(model_path: str, device="cpu"):
    """
    torch.save()로 저장된 전체 Conv-DiffLogic 모델 객체를 로드합니다.
    """
    print(f"\n[Loading Full Convolutional Model]")
    print(f"  - Model file: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.load(model_path, map_location=device)
    model.to(device).eval()
    print("✅ Model loaded successfully.")
    return model

def get_logic_layers_from_module(module, layer_list):
    """주어진 모듈(Block) 내의 LogicLayer만 순서대로 추출합니다."""
    if isinstance(module, LogicLayer):
        layer_list.append(module)
    elif len(list(module.children())) > 0:
        for child in module.children():
            get_logic_layers_from_module(child, layer_list)

# ================================================
# 압축된 difflogic 모델을 verilog 코드로 변환
# ================================================

def convert_model_to_verilog(model: nn.Sequential, module_name="flattened_conv_net"):
    verilog = []
    INDEX_TO_MODULE = {0:"tie0", 1:"and", 2:"nimp", 3:"bypass0", 4:"nrimp", 5:"bypass1", 6:"xor", 7:"or", 8:"nor", 9:"xnor", 10:"nbypass1", 11:"rimp", 12:"nbypass0", 13:"imp", 14:"nand", 15:"tie1"}


# header
    print(f"\n[Verilog Gen] Generating {module_name}...")

    # --- Verilog 헬퍼 모듈 정의 (사용자 코드와 동일) ---
    verilog.append("`timescale 1ns / 1ps\n")

    popcount_adder_module = """
// Popcount Adder Module: N-bit 입력에서 1의 개수를 셉니다.
module popcount_adder #(
    parameter WIDTH = 16,
    parameter COUNT_WIDTH = 5 // $clog2(WIDTH + 1)
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
"""
    verilog.append(popcount_adder_module)

    full_adder_module = """
// Full Adder
module Full_Adder (
    output wire sum,
    output wire carry_out,
    input  wire a,
    input  wire b,
    input  wire c_in
);
    assign {carry_out, sum} = a + b + c_in;
endmodule
"""
    verilog.append(full_adder_module)

    # ==================================
    # Wire Generation
    # ==================================
    wire_cnt = 0

    # [수정됨] Wire Naming Helper
    # 이름 뒤에 [cnt]를 붙이지 않고, 입력받은 unique name을 그대로 선언합니다.
    def new_wire(name, width=1):
        if width == 1: verilog.append(f"    wire {name};")
        else: verilog.append(f"    wire [{width-1}:0] {name};")
        return name

    # Vector 와이어 선언
    def declare_bus(name, width):
        verilog.append(f"  //  wire [{width-1}:0] {name};")
        return name

    # def dummy_wire(name_prefix='w', width=1):
    #     nonlocal wire_cnt
    #     w_name = f"{name_prefix}[{wire_cnt}]"
    #     width_str = f"[{width-1}:0] " if width > 1 else ""
    #     verilog.append(f"  wire {width_str}{w_name};")
    #     wire_cnt += 1
    #     return w_name


    # 입출력 크기 설정
    dataset = "cifar10"

    if(dataset == "cifar10"):
        input_shape = (9, 32, 32)
    elif(dataset == "mnist"):
        input_shape = (1, 28, 28)

    n_in_channels, in_h, in_w = input_shape
    total_inputs = n_in_channels * in_h * in_w
    
    # 출력 클래스 수 계산
    last_sum_layer = [m for m in model[1].modules() if isinstance(m, (GroupSum,PrunedGroupSum, WeightedGroupSum))][0]
    n_classes = last_sum_layer.k

    verilog.append(f"module {module_name} (")
    verilog.append(f"    input wire [{total_inputs-1}:0] in,")
    verilog.append(f"    output wire [{n_classes-1}:0] class_out")
    verilog.append(");\n")

    # 초기 입력 와이어 리스트 생성
    # 1D 입력을 3D [C][H][W]로 매핑
    current_wires = [[[None for _ in range(in_w)] for _ in range(in_h)] for _ in range(n_in_channels)]
    for c in range(n_in_channels):
        for h in range(in_h):
            for w in range(in_w):
                flat_idx = c * (in_h * in_w) + h * in_w + w
                current_wires[c][h][w] = f"in[{flat_idx}]"
    
    # 현재 공간 차원 추적 (Conv 연산 인덱싱을 위해 필요)
    curr_H, curr_W = in_h, in_w
    tree_conv_count = 0
    
    # ==========================================================================
    # Phase 1: Feature Extractor (model[0])
    # ==========================================================================
    print("[Verilog Gen] Processing Features...")
    
    for name, module in model[0].named_children():
        
        # [1] Crossbar1x1Conv: 채널 간 순서 변경
        if isinstance(module, Crossbar1x1Conv):
            print(f"  - {name}: Crossbar ({module.in_channels} -> {module.out_channels})")
            verilog.append(f"\n    // --- {name}: Crossbar ---")
            
            indices = module.crossbar.connection_indices.cpu().numpy()
            
            # [요청사항] Mapping 정보 출력
            print(f"    [Crossbar Mapping Info]")
            for out_c in range(module.out_channels):
                in_c = indices[out_c]
                print(f"      - Output Ch {out_c:<4} <== Input Ch {in_c:<4}")

            # Wire 연결 (3D 구조 유지)
            next_wires = [[[None for _ in range(curr_W)] for _ in range(curr_H)] for _ in range(module.out_channels)]
            for out_c in range(module.out_channels):
                in_c = indices[out_c]
                for h in range(curr_H):
                    for w in range(curr_W):
                        next_wires[out_c][h][w] = current_wires[in_c][h][w]
            current_wires = next_wires

        # [B] TreeConvLayer (Logic Connections Only, No Unfolding)
        elif isinstance(module, (FusedTreeConvLayer, TreeConvLayer)):
            print(f"  - {name}: TreeConvLayer (K={module.kernel_size}) - Generating Logic Structure Only")
            verilog.append(f"\n    // --- {name}: TreeConvLayer Logic Structure (Abstract Patch Input) ---")
            
            K = module.kernel_size
            final_out_dim = module.cascade[-1].out_dim
            
            # [요청사항] Sliding Window (Unfolding) 제거
            # 대신, 입력 패치를 추상화된 와이어 'patch_in'으로 정의합니다.
            # TreeConv의 입력 차원: C_in * K * K
            logic_in_dim = len(current_wires) * K * K
            
            # [수정됨] 입력 와이어 이름 결정
            patch_wires = []
            if tree_conv_count == 0:
                # 첫 번째 TreeConvLayer: 모듈 입력 'in' 사용
                for i in range(logic_in_dim):
                    patch_wires.append(f"in[{i}]")
            else:
                # 이후 TreeConvLayer: 고유한 입력 이름 생성
                tree_name = f"tree{tree_conv_count}_in"
                declare_bus(tree_name, logic_in_dim) # 선언
                for i in range(logic_in_dim):
                    # treeN_in_M 형식
                    w_name = f"tree{tree_conv_count}_in[{i}]"
                    patch_wires.append(w_name)
            
            tree_conv_count += 1

            # Logic Cascade 생성 (단 한 번만 수행)
            layer_in = patch_wires
            
            # [추가] ConvBlock 번호 주석 추가 (tree_conv_count 사용)
            verilog.append(f"\n// --- ConvBlock_{tree_conv_count} ---")

            for l_idx, logic_layer in enumerate(module.cascade):
                layer_out = []
                indices = logic_layer.indices
                weights = logic_layer.weights.argmax(dim=-1).cpu().numpy()
                
                verilog.append(f"\n// Logic Layer {l_idx}: {logic_layer.in_dim} -> {logic_layer.out_dim}")
                
                # [수정됨] 버스 선언: w_{name}_L{idx}_n
                bus_name = f"w_{name}_L{l_idx}_n"
                declare_bus(bus_name, logic_layer.out_dim)

                for out_idx in range(logic_layer.out_dim):
                    idx_a = indices[0][out_idx].item()
                    idx_b = indices[1][out_idx].item()
                    op_code = weights[out_idx]
                    
                    # 인덱스 기반 mapping
                    wire_a = layer_in[idx_a]
                    wire_b = layer_in[idx_b]
                    
                    # Wire 이름 생성 (좌표 p{h}_{w} 제거됨)
                    res_wire_access = f"{bus_name}[{out_idx}]"
                    inst_name = f"inst_{bus_name}_{out_idx}"
                    
                    gate_type = INDEX_TO_MODULE.get(op_code, "and_g")
                    
                    verilog.append(f"    {gate_type} {inst_name} ({res_wire_access}, {wire_a}, {wire_b});")
                    layer_out.append(res_wire_access)
                
                layer_in = layer_out # 다음 로직 레이어의 입력
            
            # [주의] Unfolding을 하지 않았으므로, 3D Spatial Wire(current_wires)는 여기서 끊깁니다.
            # Classifier 연결을 위해 current_wires를 더미 값이나, 현재 Logic Tree의 출력(1x1 Spatial)으로 대체해야 함.
            # 여기서는 편의상 "Spatial 차원이 1x1로 축소되었다"고 가정하고 current_wires를 업데이트합니다.
            # (실제 전체 이미지 처리는 불가능하지만, 구조 확인용으로는 적합)
            
            
            next_wires = [[[None] for _ in range(1)] for _ in range(final_out_dim)]
            for c_out in range(final_out_dim):
                next_wires[c_out][0][0] = layer_in[c_out]
            curr_H, curr_W = 1, 1 # Crossbar1x1Conv에서 Channel만 처리하기 위해 1로 설정

            current_wires = next_wires

        # [C] ORPool2d
        elif isinstance(module, ORPool2d):
            pass

    # ==========================================================================
    # Phase 2: Classifier (model[1])
    # ==========================================================================
    print("[Verilog Gen] Processing Classifier...")
    verilog.append(f"\n// =============== Block: Classifier_Block ================")
    # --- 3. Classifier Processing (1D Wire Management) ---
    # 1. Classifier Input Dimension Calculation
    # Features의 출력 와이어를 Flatten하여 가져오는 대신, '개수'만 계산합니다.
    if(dataset == "cifar10"):
        curr_H, curr_W = 2, 2
    elif(dataset == "mnist"):
        curr_H, curr_W = 4, 4
    clf_in_dim = len(current_wires) * curr_H * curr_W 
    print(curr_H, curr_W)
    print(f"  - Classifier Input Size: {clf_in_dim}")

    # 2. [수정됨] Features 연결을 끊고, 독립적인 버스 선언
    bus_name = "classifier_input"
    declare_bus(bus_name, width=clf_in_dim)
    
    # assign 문 없음! (Features 출력과 연결되지 않음)
    # live_vec은 이 새로운 버스를 가리킴
    live_vec = [f"{bus_name}[{i}]" for i in range(clf_in_dim)]


    for name, module in model[1].named_children():
        if isinstance(module, nn.Flatten):
            continue

        # [A] LogicLayer Only
        elif isinstance(module, LogicLayer):
            print(f"  - {name}: LogicLayer ({module.in_dim} -> {module.out_dim})")
            verilog.append(f"\n// --- {name}: LogicLayer ---")
            
            indices = module.indices
            weights = module.weights.argmax(dim=-1).cpu().numpy()
            next_vec = []
            
            # [수정] LogicLayer 출력은 버스(Vector)로 선언
            # 예: wire [255:0] w_1_n;
            bus_name = new_wire(f"w_{name}_n", width=module.out_dim)

            for out_idx in range(module.out_dim):
                idx_a = indices[0][out_idx].item()
                idx_b = indices[1][out_idx].item()
                op_code = weights[out_idx]
                
                # LogicLayer의 in_dim은 live_vec의 길이와 같음 (Rebuild로 보장됨)
                wire_a = live_vec[idx_a]
                wire_b = live_vec[idx_b]
                
                # [수정됨] 이름 포맷 변경
                # [수정] 인덱싱을 사용하여 접근 (w_1_n[0])
                res_wire_access = f"{bus_name}[{out_idx}]"
                inst_name = f"inst_{bus_name}_{out_idx}"

                gate_type = INDEX_TO_MODULE[op_code]
                
                verilog.append(f"    {gate_type} {inst_name} ({res_wire_access}, {wire_a}, {wire_b});")
                next_vec.append(res_wire_access)
            
            live_vec = next_vec
    

       # [B] GroupSum & Argmax (기존 로직 유지)
        elif isinstance(module, (GroupSum, PrunedGroupSum, WeightedGroupSum)):
            print(f"- Final Sum & Argmax ({n_classes} classes)")
            verilog.append(f"\n// --- Final GroupSum & Argmax ---")
            
            score_wires = []
            score_widths = []

            if isinstance(module, PrunedGroupSum):
                # PrunedGroupSum: 각 그룹별 크기가 다름 (buffer에서 로드)
                group_sizes_list = module.group_sizes.tolist()
            else:
                if isinstance(module, WeightedGroupSum):
                    in_dim = module.in_dim
                else:
                    in_dim = len(live_vec)
        
                group_size = in_dim // module.k
        
            integer_weights = None
            if isinstance(module, WeightedGroupSum):
                with torch.no_grad():
                    integer_weights = module.weight_raw.round().int().tolist()

            current_pos = 0
            for grp_idx in range(module.k):
                if grp_idx >= n_classes: break

                if isinstance(module, PrunedGroupSum):
                    curr_group_size = int(group_sizes_list[grp_idx])
                    print(f"  - PrunedGroupSum: Group {grp_idx} size: {curr_group_size}")
                else:
                    curr_group_size = group_size

                group_inputs = live_vec[current_pos : current_pos + curr_group_size]

                sum_terms = []
                if isinstance(module, WeightedGroupSum):
                    group_weights = integer_weights[grp_idx]
                    for inp, w in zip(group_inputs, group_weights):
                        if w > 0: sum_terms.extend([inp] * w)
                else: # GroupSum, PrunedGroupSum
                    sum_terms.extend(group_inputs)
            
                if sum_terms:
                    max_val = len(sum_terms)
                    bit_width = max_val.bit_length()
                    
                    concat_wire = new_wire(f"pc_in_{grp_idx}", width=max_val)
                    verilog.append(f"    assign {concat_wire} = {{{', '.join(sum_terms)}}};")
                    
                    s_wire = new_wire(f"score_{grp_idx}", width=bit_width)
                    verilog.append(f"    popcount_adder #(.WIDTH({len(sum_terms)}), .COUNT_WIDTH({bit_width})) pc_{grp_idx} (.in({concat_wire}), .sum({s_wire}));")
                    
                    score_wires.append(s_wire)
                    score_widths.append(bit_width)
                
                current_pos += curr_group_size

            # Argmax Logic (사용자 코드와 동일)
            verilog.append("\n// --- Argmax Logic (Winner-Take-All for Multi-class) ---")
            max_width = max(score_widths) if score_widths else 1
            padded_scores = []
            for w, s in zip(score_widths, score_wires):
                if w < max_width:
                    padded_scores.append(f"{{{{{max_width - w}'b0}}, {s}}}")
                else:
                    padded_scores.append(s)

            for i in range(n_classes):
                is_max_conditions = [f"({padded_scores[i]} >= {padded_scores[j]})" for j in range(n_classes) if i != j]
                if not is_max_conditions:
                    verilog.append(f"  assign class_out{i} = 1'b1;")
                else:
                    verilog.append(f"  assign class_out{i} = &{{{', '.join(is_max_conditions)}}};")
    
    verilog.append("\nendmodule")
    return "\n".join(verilog)
    



def main():
    parser = argparse.ArgumentParser(description="Extract logical blocks (PatchLogicBlock, Classifier) from a .pt model and convert each to a separate Verilog file.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained .pt model file.")
    parser.add_argument('--module_name', type=str, default=None,
                    help="Verilog 모듈의 이름을 직접 지정합니다. 지정하지 않으면 파일 이름에서 자동으로 생성됩니다.")
    args = parser.parse_args()

    # --- 1. 모델 로드 ---
    device = get_device()
    model = load_conv_model(args.model_path, device=device)
    

    # 수정된 Verilog 변환 함수 호출
    verilog_code = convert_model_to_verilog(model, module_name=args.module_name)
    with open(f"{args.module_name}.v", "w") as f:
        f.write(verilog_code)
    print(f"✔ Verilog saved to {args.module_name}.v")

    print("\nAll logical blocks have been converted successfully.")

if __name__ == '__main__':
    main()

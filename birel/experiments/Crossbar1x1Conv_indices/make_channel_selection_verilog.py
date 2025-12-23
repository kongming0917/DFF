import re
import os
import argparse
# 실행: python make_channel_selection_verilog.py --indices-file indices.log
def generate_verilog_from_text_log(log_path, output_file="crossbar_logic.v"):
    BIT_WIDTH = 9

    if not os.path.exists(log_path):
        print(f"❌ Error: Input file '{log_path}' not found.")
        return

    print(f"📂 Reading log file: {log_path}...")

    layers_data = {}
    layer_meta = {} # [추가] 레이어별 (입력 채널 수, 출력 채널 수) 정보를 저장
    current_layer_id = None

    # [수정] 공백(\s*) 처리를 더 유연하게 변경
    # 예: "- 0: Crossbar" 또는 "0: Crossbar" 모두 잡음
    layer_regex = re.compile(r"[:\-\s]+(\d+):\s*Crossbar\s*\(\s*(\d+)\s*->\s*(\d+)\s*\)")
    
    # 예: "- Output Ch 0 <== Input Ch 4" (공백, 탭, 특수문자 무시)
    # "Output Ch"와 숫자, "<=="와 "Input Ch", 숫자 사이의 관계만 봅니다.
    map_regex = re.compile(r"Output\s*Ch\s*(\d+).*Input\s*Ch\s*(\d+)")

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # 앞뒤 공백 제거
            if not line: continue

            # 1. 레이어 감지
            layer_match = layer_regex.search(line)
            if layer_match:
                current_layer_id = int(layer_match.group(1))
                in_ch_count = layer_match.group(2)
                out_ch_count = layer_match.group(3)

                layers_data[current_layer_id] = []
                layer_meta[current_layer_id] = (in_ch_count, out_ch_count) # [추가] 정보 저장
                continue
            
            # 2. 매핑 정보 감지
            if current_layer_id is not None:
                map_match = map_regex.search(line)
                if map_match:
                    out_ch = int(map_match.group(1))
                    in_ch = int(map_match.group(2))
                    layers_data[current_layer_id].append((out_ch, in_ch))

    # --- Verilog 생성 ---
    verilog_lines = []
    verilog_lines.append(f"// Generated from: {log_path}")
    
    sorted_ids = sorted(layers_data.keys())
    
    for layer_id in sorted_ids:
        connections = layers_data[layer_id]
        
        # 저장해둔 채널 정보 가져오기 (없을 경우 '?' 처리)
        in_cnt, out_cnt = layer_meta.get(layer_id, ("?", "?"))

        # [중요] 파싱된 개수 확인
        print(f"  - Layer {layer_id}: Parsed {len(connections)} connections.")
        
        if not connections:
            print(f"    ⚠️ Warning: Layer {layer_id} has 0 connections. Check the log file!")
            continue

        connections.sort(key=lambda x: x[0])
        
        # 입력 변수명 설정 (필요시 수정)
        input_wire = "data_out" 
        output_wire = f"data_arr"

        slices = []
        for out_ch, in_ch in connections:
            base_idx = in_ch * BIT_WIDTH
            slice_str = f"{input_wire}[({BIT_WIDTH}*{in_ch}) +: {BIT_WIDTH}]"
            slices.append(slice_str)

        # LSB가 맨 뒤로 가도록 역순 정렬
        slices.reverse()

        code = f"// Layer {layer_id} Crossbar Mapping ({in_cnt} -> {out_cnt} blocks)\n"
        code += f"     assign {output_wire} = {{ {', '.join(slices)} }};\n"
        verilog_lines.append(code)

    with open(output_file, 'w') as f:
        f.write("\n".join(verilog_lines))

    print(f"\n✅ Verilog code generated: {output_file}")

# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicates the process of fast_inference.py for Conv-DiffLogic models (without compression).")
    parser.add_argument('--indices-file', type=str, required=True, help='Name of the indices file')
    args = parser.parse_args()

    LOG_FILE = args.indices_file
    generate_verilog_from_text_log(LOG_FILE)
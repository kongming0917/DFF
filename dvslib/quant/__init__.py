"""quant layer — one PT2E quantization path shared by every method (FPGA INT8).

prepare_qat / convert / set_qat_mode / get_fpga_quantizer / save_int8 / load_int8 는 `dvslib.quant.pt2e`.
torch.ao 의존이 무거우므로 여기서 re-export하지 않고 submodule에서 직접 import한다.
"""

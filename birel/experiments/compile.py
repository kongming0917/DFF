# compile_model.py
import argparse
import torch
import subprocess
import time
from full_model_generator import generate_full_model_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfile", default="full_model.c")
    parser.add_argument("--sofile", default="full_model.so")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--C", type=int, default=9)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--compiler", default="gcc")
    args = parser.parse_args()

    total_start = time.time()

    print("[1] Loading model…")
    load_start = time.time()
    m = torch.load(args.model, map_location="cpu")
    if isinstance(m, dict) and "model" in m:
        model = m["model"]
    else:
        model = m
    load_time = time.time() - load_start
    print(f"    Load time: {load_time:.3f}s")

    print("[2] Generating C code…")
    codegen_start = time.time()
    generate_full_model_c(
        model,
        args.B, args.C, args.H, args.W,
        args.cfile
    )
    codegen_time = time.time() - codegen_start
    print(f"    Code generation time: {codegen_time:.3f}s")

    print("[3] Compiling shared library…")
    compile_start = time.time()
    result = subprocess.run([
        args.compiler, "-shared", "-O3", "-fPIC",
        args.cfile, "-o", args.sofile
    ], capture_output=True, text=True)
    compile_time = time.time() - compile_start
    
    if result.returncode != 0:
        print(f"    Compilation failed!")
        print(f"    Error: {result.stderr}")
        return
    
    print(f"    Compile time: {compile_time:.3f}s")

    total_time = time.time() - total_start
    print(f"[DONE] Built {args.sofile}")
    print(f"\n=== Compilation Summary ===")
    print(f"  Model loading:     {load_time:.3f}s")
    print(f"  Code generation:   {codegen_time:.3f}s")
    print(f"  Compilation:       {compile_time:.3f}s")
    print(f"  Total time:        {total_time:.3f}s")


if __name__ == "__main__":
    main()

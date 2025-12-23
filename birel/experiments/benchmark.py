#!/usr/bin/env python3
"""
Benchmark script for measuring inference time of .pt model files.
Supports both GPU (regular tensor) and CPU (compiled model) inference.
"""
import argparse
import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np
from collections import defaultdict
from difflogic import CompiledLogicNet
from conv_difflogic import load_dataset, get_model
import ctypes
import numpy.ctypeslib as npct


def load_model_from_pt(model_path, device='cuda'):
    """Load model from .pt file (model object)."""
    loaded_data = torch.load(model_path, map_location=device)
    
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        model = loaded_data['model']
    elif isinstance(loaded_data, torch.nn.Module):
        model = loaded_data
    else:
        raise ValueError(f"Unsupported model format in {model_path}. Expected model object or dict with 'model' key.")
    
    model.to(device).eval()
    return model


def measure_gpu_time(model, data_loader, device='cuda', num_iterations=1000, warmup=10, profile_layers=False):
    """
    Measure GPU inference time using regular tensor (not PackBitsTensor).
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
        device: Device to use
        num_iterations: Number of iterations for measurement
        warmup: Number of warmup iterations
        profile_layers: If True, measure time per layer
    
    Returns:
        Average time per batch in nanoseconds (and layer timings if profile_layers=True)
    """
    model.eval()
    
    # Get a batch of data
    x, _ = next(iter(data_loader))
    x = x.to(device)
    B = x.size(0)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    
    if profile_layers:
        # Use forward hooks with CUDA Events for layer-wise timing
        print(f"\n[GPU Layer-wise Profiling] 각 레이어별 시간 측정 (iterations={num_iterations})")
        print("="*80)
        
        layer_timings = defaultdict(list)
        hooks = []
        
        def make_timing_hooks(layer_name):
            start_events = []
            
            def forward_pre_hook(module, input):
                # Don't synchronize here - just record the start event
                # Synchronization will happen in forward_hook
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                start_events.append(start_event)
            
            def forward_hook(module, input, output):
                if start_events:
                    start_event = start_events.pop(0)
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    # Synchronize only once at the end to get accurate timing
                    # This allows GPU to execute asynchronously between layers
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                    layer_timings[layer_name].append(elapsed_time * 1000)  # Convert to microseconds
            
            return forward_pre_hook, forward_hook
        
        # Register hooks for all leaf modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                pre_hook, hook = make_timing_hooks(name)
                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(hook))
        
        # Measure with both layer-wise and end-to-end timing
        with torch.no_grad():
            # End-to-end timing for comparison
            torch.cuda.synchronize()
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_total.record()
            
            for _ in range(num_iterations):
                _ = model(x)
            
            end_total.record()
            torch.cuda.synchronize()
            total_time_ms = start_total.elapsed_time(end_total)  # milliseconds
            total_time_us = total_time_ms * 1000  # microseconds
            avg_total_time_us = total_time_us / num_iterations
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process results
        layer_avg_times = {}
        sum_layer_times_us = 0
        
        for layer_name, times in layer_timings.items():
            avg_time = sum(times) / len(times)  # Average per iteration in microseconds
            layer_avg_times[layer_name] = avg_time
            sum_layer_times_us += avg_time
        
        # Sort by time (descending)
        sorted_layers = sorted(layer_avg_times.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Layer Name':<60} {'Time (μs)':<15} {'Time (ms)':<15} {'% of Total':<12} {'Time/iter (μs)':<15}")
        print("-" * 120)
        
        for layer_name, time_us in sorted_layers[:30]:  # Top 30 layers
            time_ms = time_us / 1000.0
            percentage = (time_us / sum_layer_times_us * 100) if sum_layer_times_us > 0 else 0
            time_per_iter = time_us
            # Truncate long layer names
            display_name = layer_name[:58] if len(layer_name) > 58 else layer_name
            print(f"{display_name:<60} {time_us:<15.2f} {time_ms:<15.3f} {percentage:<12.2f} {time_per_iter:<15.3f}")
        
        print("-" * 120)
        sum_ms = sum_layer_times_us / 1000.0
        print(f"{'SUM (Layer times)':<60} {sum_layer_times_us:<15.2f} {sum_ms:<15.3f} {'N/A':<12} {sum_layer_times_us:<15.3f}")
        
        # Show actual end-to-end time
        actual_ms = avg_total_time_us / 1000.0
        print(f"{'ACTUAL (End-to-end)':<60} {avg_total_time_us:<15.2f} {actual_ms:<15.3f} {'100.00':<12} {avg_total_time_us:<15.3f}")
        
        # Calculate difference
        diff_us = abs(sum_layer_times_us - avg_total_time_us)
        diff_pct = (diff_us / avg_total_time_us * 100) if avg_total_time_us > 0 else 0
        print(f"\n차이 (Sum vs Actual): {diff_us:.2f} μs ({diff_pct:.2f}%)")
        
        if diff_pct > 10:
            print("⚠️  경고: Layer 시간 합계와 실제 end-to-end 시간이 크게 다릅니다.")
            print("   원인:")
            print("   1. 각 layer마다 torch.cuda.synchronize() 호출로 인한 동기화 오버헤드")
            print("   2. GPU의 비동기 실행이 방해받아 실제 시간이 증가")
            print("   3. Layer 간 overlap이 없어져 순차 실행과 유사해짐")
            print("   → 실제 end-to-end 시간이 더 정확한 측정값입니다.")
        
        avg_time_per_sample = avg_total_time_us / B
        print(f"\n평균 시간 per iteration (실제): {actual_ms:.3f} ms")
        print(f"평균 시간 per sample (batch={B}): {avg_time_per_sample / 1000.0:.3f} ms = {avg_time_per_sample:.2f} μs")
        
        return avg_total_time_us * 1000  # Convert to nanoseconds (use actual end-to-end time)
    
    # Regular measurement (no profiling)
    def gpu_ns(it=num_iterations):
        with torch.no_grad():
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            
            for _ in range(it):
                _ = model(x)
            
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e6 / it  # Convert ms to ns
    
    num_runs = 10
    timings_ns = []
    print(f"\n[GPU Regular Tensor] 추론 속도 측정 시작 (총 {num_runs}회 실행 후 평균)")
    
    for i in range(num_runs):
        ns_per_batch = gpu_ns(num_iterations)
        timings_ns.append(ns_per_batch)
        print(f"  - 실행 {i + 1}/{num_runs}: {ns_per_batch / B:.2f} ns / sample")
        time.sleep(0.1)
    
    if timings_ns:
        average_ns_per_batch = sum(timings_ns) / len(timings_ns)
        print(f"\n[GPU Regular Tensor] {average_ns_per_batch / B:.2f} ns / sample (batch={B})")
        return average_ns_per_batch
    else:
        print("GPU 속도 측정에 실패했습니다.")
        return None


###############################################
# Full-Model Shared Library Wrapper (C So)
###############################################
class FullModelSO:
    """
    Wrapper for full_model.so (stage-aware compiled model).
    full_model(
        const u8* inp,
        int* out,
        size_t B
    )
    """

    def __init__(self, so_path, input_size, num_classes):
        # Convert to absolute path for ctypes.CDLL
        so_path = os.path.abspath(so_path)
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"full_model.so not found: {so_path}")
        
        # Load library with RTLD_NOW on Linux to avoid lazy loading issues
        # This helps prevent munmap_chunk errors during cleanup
        try:
            if hasattr(ctypes, 'RTLD_NOW'):
                # Linux: use RTLD_NOW to load all symbols immediately
                # RTLD_NOW prevents lazy binding which can cause issues
                self.lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_NOW)
            elif sys.platform.startswith('win'):
                # Windows: use LoadLibrary directly
                self.lib = ctypes.windll.LoadLibrary(so_path)
            else:
                # Other platforms: default loading
                self.lib = ctypes.CDLL(so_path)
        except (OSError, AttributeError) as e:
            # Fallback to default loading
            print(f"Warning: Failed to load with RTLD_NOW, trying default: {e}")
            self.lib = ctypes.CDLL(so_path)
        
        # Keep reference to prevent premature garbage collection/unloading
        # This is important to prevent munmap_chunk errors at program exit
        self._so_path = so_path
        self._lib_ref = self.lib  # Explicit reference to prevent GC

        # Set signature
        try:
            self.lib.full_model.argtypes = [
                npct.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),  # inp
                npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"), # out
                ctypes.c_size_t                                                 # B
            ]
            self.lib.full_model.restype = None
        except AttributeError as e:
            raise RuntimeError(f"Failed to find 'full_model' function in {so_path}: {e}")

        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x_bool_np):
        """
        x_bool_np: numpy bool array with shape (B, input_size)
        """
        assert x_bool_np.dtype == np.bool_, "Input must be boolean"
        B = x_bool_np.shape[0]

        # Convert to uint8 because C uses u8
        x_u8 = x_bool_np.astype(np.uint8).reshape(-1)

        out = np.zeros((B * self.num_classes,), dtype=np.int32)

        self.lib.full_model(x_u8, out, B)
        out = out.reshape(B, self.num_classes)

        return out


def measure_full_model_cpu_time(so_model, numpy_input, batch_size, num_iterations=1000, warmup=10):

    # Warmup
    for _ in range(warmup):
        _ = so_model.forward(numpy_input)

    # Timing
    start = time.perf_counter_ns()
    for _ in range(num_iterations):
        _ = so_model.forward(numpy_input)
    end = time.perf_counter_ns()

    return (end - start) / num_iterations

def eval_accuracy_full_model(so_model, torch_model, data_loader, device='cpu'):
    correct = 0
    total = 0

    torch_model = torch_model.to(device)
    torch_model.eval()

    for x, y in data_loader:
        B = x.size(0)

        # PyTorch prediction
        with torch.no_grad():
            torch_out = torch_model(x.to(device)).argmax(-1).cpu()

        # SO model prediction
        x_bool = x.reshape(B, -1).round().bool().numpy()
        out = so_model.forward(x_bool)
        so_pred = out.argmax(-1)

        correct += (so_pred == y.numpy()).sum()
        total += B

    return correct / total


def measure_cpu_time(compiled_model, numpy_input, batch_size, num_iterations=1000, warmup=10):
    """
    Measure CPU inference time using compiled model.
    
    Args:
        compiled_model: CompiledLogicNet instance
        numpy_input: NumPy boolean array input
        batch_size: Batch size
        num_iterations: Number of iterations for measurement
        warmup: Number of warmup iterations
    
    Returns:
        Average time per batch in nanoseconds
    """
    def cpu_ns(compiled_net, data, it=num_iterations):
        # Warmup
        for _ in range(warmup):
            _ = compiled_net(data)
        
        # Measurement
        start_time = time.perf_counter_ns()
        for _ in range(it):
            _ = compiled_net(data)
        end_time = time.perf_counter_ns()
        
        return (end_time - start_time) / it
    
    num_runs = 10
    timings_ns_cpu = []
    print(f"\n[Compiled CPU] 추론 속도 측정 시작 (총 {num_runs}회 실행 후 평균)")
    
    for i in range(num_runs):
        ns_per_batch_cpu = cpu_ns(compiled_model, numpy_input, it=num_iterations)
        timings_ns_cpu.append(ns_per_batch_cpu)
        print(f"  - 실행 {i + 1}/{num_runs}: {ns_per_batch_cpu / batch_size:.2f} ns / sample")
        time.sleep(0.1)
    
    if timings_ns_cpu:
        average_ns_per_batch_cpu = sum(timings_ns_cpu) / len(timings_ns_cpu)
        print(f"\n[Compiled CPU] {average_ns_per_batch_cpu / batch_size:.2f} ns / sample (batch={batch_size})")
        return average_ns_per_batch_cpu
    else:
        print("CPU 속도 측정에 실패했습니다.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference time of .pt model files')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pt model file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (for data loading)')
    parser.add_argument('--model-size', type=str, default='B', choices=['S', 'M', 'B', 'L', 'G'], help='Model size (default: B)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--gpu', action='store_true', help='Measure GPU inference time')
    parser.add_argument('--profile-layers', action='store_true', help='Profile individual layer timings (requires --gpu)')
    parser.add_argument('--cpu', action='store_true', help='Measure CPU inference time (requires compiled model)')
    parser.add_argument('--so-path', type=str, default=None, help='Path to compiled .so file (for CPU measurement)')
    parser.add_argument('--bit', type=int, default=64, choices=[8, 16, 32, 64], help='Bit count for compiled model (default: 64)')
    parser.add_argument('--iters', type=int, default=1000, help='Number of iterations for measurement (default: 1000)')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations (default: 10)')
    parser.add_argument('--full-model-so', type=str, default=None,
                    help='Path to full_model.so (stage-aware compiled FULL model)')

    args = parser.parse_args()
    
    if args.profile_layers and not args.gpu:
        print("Warning: --profile-layers requires --gpu. Enabling --gpu automatically.")
        args.gpu = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("Warning: CUDA is not available. GPU measurement will be skipped.")
        args.gpu = False
        args.profile_layers = False
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        sys.exit(f"Error: Model file not found: {args.model_path}")
    
    model = load_model_from_pt(args.model_path, device=device)
    print("✅ Model loaded successfully")
    
    # Load dataset for getting input shape
    print(f"\nLoading dataset: {args.dataset} (model_size={args.model_size})")
    # Create a minimal args object for load_dataset
    class Args:
        def __init__(self):
            self.dataset = args.dataset
            self.batch_size = args.batch_size
            self.valid_set_size = 0.0
            self.model_size = args.model_size
    
    dataset_args = Args()
    train_loader, validation_loader, test_loader, final_channels = load_dataset(dataset_args)
    
    # Get a batch for shape information
    x, _ = next(iter(test_loader if test_loader is not None else train_loader))
    B = x.size(0)
    print(f"Input shape: {x.shape}, Batch size: {B}")
    
    
    # GPU measurement
    if args.gpu:
        print("\n" + "="*80)
        print("GPU Inference Time Measurement")
        print("="*80)
        model.eval()
        gpu_time = measure_gpu_time(
            model, 
            test_loader if test_loader is not None else train_loader,
            device=device,
            num_iterations=args.iters,
            warmup=args.warmup,
            profile_layers=args.profile_layers
        )
    
    # CPU measurement (compiled model)
    if args.cpu:
        print("\n" + "="*80)
        print("CPU Inference Time Measurement (Compiled Model)")
        print("="*80)
        
        if args.so_path is None:
            print("Error: --so-path is required for CPU measurement")
            print("Please provide the path to a compiled .so file, or compile the model first using compile.py")
            sys.exit(1)
        
        if not os.path.exists(args.so_path):
            print(f"Error: Compiled .so file not found: {args.so_path}")
            sys.exit(1)
        
        # Load compiled model
        print(f"Loading compiled model from: {args.so_path}")
        compiled_model = CompiledLogicNet(
            model=model.to('cpu'),  # Structure sync
            num_bits=args.bit,
            verbose=True
        )
        compiled_model.compile(save_lib_path=args.so_path, verbose=False)
        print("✅ Compiled model loaded successfully")
        
        # Prepare NumPy input
        numpy_input = x.reshape(B, -1).round().bool().numpy()
        print(f"NumPy input shape: {numpy_input.shape}, dtype: {numpy_input.dtype}")
        
        cpu_time = measure_cpu_time(
            compiled_model,
            numpy_input,
            batch_size=B,
            num_iterations=args.iters,
            warmup=args.warmup
        )
    
    # If full_model.so provided → use stage-aware compiled model
    if args.full_model_so is not None:
        print("\n" + "="*80)
        print("CPU Inference (Stage-Aware FULL MODEL .so)")
        print("="*80)

        if not os.path.exists(args.full_model_so):
            sys.exit(f"Error: full_model.so not found: {args.full_model_so}")

        print(f"Loading full_model.so: {args.full_model_so}")

        # Determine input size and num classes
        B, C, H, W = x.shape
        input_size = C * H * W
        num_classes = model[1][-1].k  # GroupSum.k

        so_full = FullModelSO(args.full_model_so, input_size, num_classes)

        # Prepare numpy input
        numpy_input = x.reshape(B, -1).round().bool().numpy()

        # Measure speed
        full_model_time = measure_full_model_cpu_time(
            so_full, numpy_input, B,
            num_iterations=args.iters,
            warmup=args.warmup
        )
        print(f"[Stage-Aware FULL MODEL] {full_model_time/B:.2f} ns/sample")

        # Measure accuracy
        acc_full = eval_accuracy_full_model(
            so_full,
            model.to('cpu'),
            test_loader
        )
        print(f"[Stage-Aware FULL MODEL] Accuracy = {acc_full:.4f}")


    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if args.gpu and gpu_time:
        print(f"GPU (Regular Tensor): {gpu_time / B:.2f} ns / sample")
    if args.cpu and cpu_time:
        print(f"CPU (Compiled): {cpu_time / B:.2f} ns / sample")


if __name__ == '__main__':
    main()

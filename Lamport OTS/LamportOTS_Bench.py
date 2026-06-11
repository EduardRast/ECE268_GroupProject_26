import time
import sys
import os
import cupy as cp

import LOTS_CPU as cpu_ots
import LOTS_GPU as gpu_ots

class SuppressPrint:
    """Context manager to suppress stdout to prevent print I/O from skewing benchmarks."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def format_speedup(cpu_t, gpu_t):
    """Calculates the speedup and formats the winner as a string."""
    speedup = cpu_t / gpu_t
    if speedup > 1:
        return f"GPU is {speedup:.2f}x faster"
    elif speedup < 1:
        return f"CPU is {(1 / speedup):.2f}x faster"
    else:
        return "Equal speed"

def main():
    message = b"hello world"
    print("Initializing Benchmark...\n")

    # --- PHASE 1: WARM-UP (GPU ONLY) ---
    print("Warming up GPU (compiling kernels & allocating memory)...")
    with SuppressPrint():
        pk_warm, sk_warm = gpu_ots.keygen()
        sig_warm = gpu_ots.sign(message, sk_warm)
        gpu_ots.verify(message, sig_warm, pk_warm)
        cp.cuda.Stream.null.synchronize()

    print("Running Benchmarks...\n")

    # --- PHASE 2: KEY GENERATION ---
    # CPU
    cpu_start = time.perf_counter()
    with SuppressPrint():
        pk_cpu, sk_cpu = cpu_ots.keygen()
    cpu_keygen_time = time.perf_counter() - cpu_start

    # GPU
    gpu_start = time.perf_counter()
    with SuppressPrint():
        pk_gpu, sk_gpu = gpu_ots.keygen()
        cp.cuda.Stream.null.synchronize()
    gpu_keygen_time = time.perf_counter() - gpu_start


    # --- PHASE 3: SIGNING ---
    # CPU
    cpu_start = time.perf_counter()
    with SuppressPrint():
        sig_cpu = cpu_ots.sign(message, sk_cpu)
    cpu_sign_time = time.perf_counter() - cpu_start

    # GPU
    gpu_start = time.perf_counter()
    with SuppressPrint():
        sig_gpu = gpu_ots.sign(message, sk_gpu)
        cp.cuda.Stream.null.synchronize()
    gpu_sign_time = time.perf_counter() - gpu_start


    # --- PHASE 4: VERIFICATION ---
    # CPU
    cpu_start = time.perf_counter()
    with SuppressPrint():
        cpu_ots.verify(message, sig_cpu, pk_cpu)
    cpu_verify_time = time.perf_counter() - cpu_start

    # GPU
    gpu_start = time.perf_counter()
    with SuppressPrint():
        gpu_ots.verify(message, sig_gpu, pk_gpu)
        cp.cuda.Stream.null.synchronize()
    gpu_verify_time = time.perf_counter() - gpu_start


    # --- PHASE 5: DISPLAY RESULTS ---
    print("========================================")
    print("        BENCHMARK RESULTS SUMMARY       ")
    print("========================================\n")

    print("1. Key Generation")
    print(f"   CPU Time : {cpu_keygen_time:.6f} s")
    print(f"   GPU Time : {gpu_keygen_time:.6f} s")
    print(f"   Speedup  : {format_speedup(cpu_keygen_time, gpu_keygen_time)}\n")

    print("2. Signing")
    print(f"   CPU Time : {cpu_sign_time:.6f} s")
    print(f"   GPU Time : {gpu_sign_time:.6f} s")
    print(f"   Speedup  : {format_speedup(cpu_sign_time, gpu_sign_time)}\n")

    print("3. Verification")
    print(f"   CPU Time : {cpu_verify_time:.6f} s")
    print(f"   GPU Time : {gpu_verify_time:.6f} s")
    print(f"   Speedup  : {format_speedup(cpu_verify_time, gpu_verify_time)}\n")

    # Overall Summary
    cpu_total = cpu_keygen_time + cpu_sign_time + cpu_verify_time
    gpu_total = gpu_keygen_time + gpu_sign_time + gpu_verify_time
    
    print("----------------------------------------")
    print("Total End-to-End Latency")
    print(f"   CPU Total: {cpu_total:.6f} s")
    print(f"   GPU Total: {gpu_total:.6f} s")
    print(f"   Overall  : {format_speedup(cpu_total, gpu_total)}")
    print("========================================")

if __name__ == "__main__":
    main()
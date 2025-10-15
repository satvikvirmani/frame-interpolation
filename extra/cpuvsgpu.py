import torch
import time

# Check if MPS (Apple GPU) is available
device_cpu = torch.device("cpu")
device_mps = torch.device("mps") if torch.backends.mps.is_available() else None

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Matrix size for benchmarking
N = 4096
A = torch.randn((N, N))
B = torch.randn((N, N))

def benchmark(device, num_iters=10):
    """Benchmark matrix multiplication on a given device."""
    A_dev = A.to(device)
    B_dev = B.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Warmup
    for _ in range(3):
        _ = torch.mm(A_dev, B_dev)

    torch.cuda.synchronize() if device.type == "cuda" else None
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        C = torch.mm(A_dev, B_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iters
    return avg_time

# Run benchmark
cpu_time = benchmark(device_cpu)
print(f"CPU ({device_cpu}) avg time per matmul: {cpu_time:.4f} s")

if device_mps:
    gpu_time = benchmark(device_mps)
    print(f"GPU ({device_mps}) avg time per matmul: {gpu_time:.4f} s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("MPS GPU not available on this system.")
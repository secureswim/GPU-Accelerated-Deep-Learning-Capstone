# GPU-Accelerated Deep Learning Capstone
### CUDA at Scale for the Enterprise — Final Project

A complete GPU-optimized image classification pipeline featuring a **hand-written CUDA kernel**, Automatic Mixed Precision (AMP), GPU profiling, and rigorous CPU/GPU benchmarking.

---

## 🎯 Project Overview

This project trains a CNN on CIFAR-10 while demonstrating advanced GPU programming techniques:

| Technique | Implementation |
|---|---|
| **Custom CUDA Kernel** | Fused GELU + Dropout in a single GPU kernel (`custom_kernel.cu`) |
| **PyTorch Extension** | JIT-compiled via `torch.utils.cpp_extension` — no manual build step |
| **Mixed Precision (AMP)** | `torch.cuda.amp` with `GradScaler` for FP16/FP32 hybrid training |
| **GPU Profiling** | `torch.profiler` with Chrome trace export |
| **Benchmarking** | CUDA event timing across 5 tensor sizes; CPU vs GPU vs custom kernel |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA Toolkit 11.8+ ([download](https://developer.nvidia.com/cuda-downloads))
- A CUDA-capable NVIDIA GPU
- g++ / MSVC (C++17)

### 2. Install PyTorch with CUDA

```bash
# Replace cu121 with your CUDA version (cu118, cu121, cu124, etc.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify your installation:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX XXXX (or your GPU name)
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Step 1 — Train (downloads CIFAR-10 automatically)
python src/train.py --epochs 15 --batch-size 256

# Step 2 — Benchmark: custom kernel vs PyTorch GPU vs CPU
python src/benchmark.py

# Step 3 — Profile with torch.profiler (generates Chrome trace)
python src/profile_run.py --steps 20

# Step 4 — Generate all charts from logs
python src/plot_results.py
```

---

## 📁 Repository Structure

```
capstone-gpu-dl/
├── README.md
├── requirements.txt
│
├── src/
│   ├── custom_kernel.cu      # Hand-written CUDA kernel (fused GELU+Dropout)
│   ├── custom_kernel.py      # PyTorch autograd Function wrapping the kernel
│   ├── train.py              # Main training loop (AMP, CSV logging)
│   ├── benchmark.py          # CPU vs GPU vs custom kernel timing
│   ├── profile_run.py        # torch.profiler with Chrome trace export
│   └── plot_results.py       # Chart generation from CSV logs
│
└── outputs/
    ├── training_log.csv       # Epoch-by-epoch metrics
    ├── benchmark_results.csv  # Latency and speedup table
    ├── best_model.pt          # Saved model weights
    ├── profile_summary.txt    # Top operators by CUDA time
    ├── trace/                 # Chrome trace JSON files
    │   └── trace_step*.json   # Open in chrome://tracing
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── gpu_memory.png
    ├── epoch_time.png
    └── benchmark_plot.png
```

---

## 🔧 How GPU Hardware Is Used

### 1. Custom CUDA Kernel (`src/custom_kernel.cu`)

The kernel **fuses GELU activation and Dropout** into a single GPU operation.  
Normally these are two separate ops, each requiring a global memory round-trip:

```
Standard path:  input → [GELU kernel] → tmp buffer → [Dropout kernel] → output
                         ↑ 2 global mem reads/writes ↑

Fused path:     input → [GELU+Dropout kernel] → output
                         ↑ 1 global mem read/write ↑
```

Key CUDA features used:
- `curandStatePhilox4_32_10_t` — per-thread RNG for dropout sampling
- `__device__ __forceinline__` — inlined device math functions
- `blockIdx.x * blockDim.x + threadIdx.x` — standard 1D thread indexing
- `--use_fast_math` flag — uses hardware fast-math intrinsics

The kernel registers a proper **forward + backward** pass so it participates fully in PyTorch's autograd graph.

### 2. JIT Compilation via `torch.utils.cpp_extension`

```python
_ext = load(
    name="fused_gelu_dropout",
    sources=["custom_kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

No manual `nvcc` invocation required. PyTorch compiles and caches the `.so` automatically on first run.

### 3. Automatic Mixed Precision (AMP)

```python
scaler = GradScaler()
with autocast():
    logits = model(imgs)   # runs in FP16 where safe
    loss   = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Benefits: ~2× memory reduction, ~1.5–2× throughput increase on Tensor Core GPUs.

### 4. GPU Profiling (`torch.profiler`)

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=export_chrome_trace, ...) as prof:
    ...
```

View results: open `chrome://tracing` in Chrome and load any `.json` from `outputs/trace/`.

---

## 🎓 How This Demonstrates GPU Programming

This project goes beyond calling GPU-enabled libraries by:

1. **Writing raw CUDA C++** — the `.cu` file contains hand-authored device functions, thread indexing math, and cuRAND state management
2. **Integrating with PyTorch's autograd** — the kernel isn't just a standalone CUDA program; it's wired into the full training graph including backward pass gradients
3. **Measuring and proving GPU benefit** — the benchmark script uses CUDA Events (not wall-clock time) for accurate GPU-side timing
4. **Profiling at the kernel level** — the Chrome trace reveals exactly which CUDA kernels run, for how long, and how much memory they use

---

## 🛠 Troubleshooting

**`CUDA not available`**  
→ Check `nvidia-smi` and ensure the PyTorch CUDA version matches your driver.

**`nvcc not found` during kernel compilation**  
→ Install CUDA Toolkit and ensure `nvcc` is on your `PATH`.

**Kernel compiles but produces wrong values**  
→ Run `python src/custom_kernel.py` — the smoke test compares output against `torch.nn.functional.gelu`.

**Out of memory**  
→ Reduce `--batch-size` (try 128 or 64).

---

## 📚 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
- [torch.profiler docs](https://pytorch.org/docs/stable/profiler.html)
- [cuRAND Library](https://docs.nvidia.com/cuda/curand/index.html)

"""
benchmark.py

Benchmarks the fused GELU+Dropout CUDA kernel against:
  1. PyTorch built-in nn.GELU + nn.Dropout (GPU)
  2. The same op on CPU

Measures:
  - Mean throughput (elements/sec)
  - Latency (ms) across multiple tensor sizes
  - Speedup ratios

Outputs:
  ../outputs/benchmark_results.csv  — raw numbers
  ../outputs/benchmark_plot.png     — bar chart comparison
"""

import os
import csv
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WARMUP_REPS = 20
BENCH_REPS  = 100

# ─── Timing helper ───────────────────────────────────────────────────────────

def benchmark_fn(fn, x: torch.Tensor, reps: int = BENCH_REPS) -> float:
    """Returns mean wall-clock time in milliseconds."""
    if x.is_cuda:
        # GPU: use CUDA events for accurate timing
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        # Warmup
        for _ in range(WARMUP_REPS):
            _ = fn(x)
        torch.cuda.synchronize()
        start.record()
        for _ in range(reps):
            _ = fn(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps   # ms
    else:
        # CPU: use time.perf_counter
        for _ in range(max(1, WARMUP_REPS // 5)):
            _ = fn(x)
        t0 = time.perf_counter()
        for _ in range(max(1, reps // 5)):
            _ = fn(x)
        return (time.perf_counter() - t0) / max(1, reps // 5) * 1000  # ms


# ─── Baseline (PyTorch built-in, no kernel fusion) ───────────────────────────

class BaselineGeluDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.gelu    = nn.GELU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(self.gelu(x))


# ─── Sizes to benchmark ──────────────────────────────────────────────────────

SIZES = [
    (   "64K",   64 * 1024),
    (  "256K",  256 * 1024),
    (    "1M", 1024 * 1024),
    (    "4M", 4 * 1024 * 1024),
    (   "16M", 16 * 1024 * 1024),
]

# ─── Run benchmarks ───────────────────────────────────────────────────────────

print("=" * 62)
print(f"{'Size':>6}  {'Custom(ms)':>10}  {'Baseline(ms)':>12}  "
      f"{'CPU(ms)':>8}  {'GPU Speedup':>11}")
print("=" * 62)

rows = []
has_cuda   = torch.cuda.is_available()
has_custom = False

if has_cuda:
    try:
        from custom_kernel import FusedGeluDropout
        has_custom = True
        print("✓  Custom CUDA kernel loaded")
    except Exception as e:
        print(f"⚠  Custom kernel unavailable: {e}")
        print("    Comparing PyTorch GPU vs CPU only.")
else:
    print("⚠  No CUDA device — showing CPU-only timings.")

baseline_gpu = BaselineGeluDropout(p=0.1)
baseline_cpu = BaselineGeluDropout(p=0.1)
if has_cuda:
    baseline_gpu = baseline_gpu.cuda().eval()

for label, N in SIZES:
    x_cpu = torch.randn(N)
    x_gpu = x_cpu.cuda() if has_cuda else None

    # CPU baseline
    with torch.no_grad():
        t_cpu = benchmark_fn(baseline_cpu, x_cpu)

    if not has_cuda:
        print(f"{label:>6}  {'N/A':>10}  {'N/A':>12}  {t_cpu:>8.3f}  {'N/A':>11}")
        rows.append([label, N, "N/A", "N/A", f"{t_cpu:.4f}", "N/A", "N/A"])
        continue

    # GPU baseline
    baseline_gpu.eval()
    with torch.no_grad():
        t_base = benchmark_fn(baseline_gpu, x_gpu)

    # GPU custom kernel
    if has_custom:
        custom_layer = FusedGeluDropout(p=0.1).cuda().eval()
        with torch.no_grad():
            t_custom = benchmark_fn(custom_layer, x_gpu)
        speedup_custom = t_base / t_custom if t_custom > 0 else float("nan")
        speedup_cpu    = t_cpu  / t_custom if t_custom > 0 else float("nan")
    else:
        t_custom       = float("nan")
        speedup_custom = float("nan")
        speedup_cpu    = t_cpu / t_base

    print(f"{label:>6}  {t_custom:>10.3f}  {t_base:>12.3f}  "
          f"{t_cpu:>8.3f}  {speedup_custom:>10.2f}x")

    rows.append([
        label, N,
        f"{t_custom:.4f}" if has_custom else "N/A",
        f"{t_base:.4f}",
        f"{t_cpu:.4f}",
        f"{speedup_custom:.4f}" if has_custom else "N/A",
        f"{speedup_cpu:.4f}",
    ])

print("=" * 62)

# ─── Write CSV ────────────────────────────────────────────────────────────────

csv_path = os.path.join(OUT_DIR, "benchmark_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["size_label", "n_elements",
                "custom_kernel_ms", "baseline_gpu_ms", "cpu_ms",
                "speedup_vs_baseline", "speedup_vs_cpu"])
    w.writerows(rows)
print(f"\n✓  Results saved to {csv_path}")

# ─── Plot ─────────────────────────────────────────────────────────────────────

if not has_cuda:
    print("Skipping plot (no CUDA).")
else:
    labels  = [r[0] for r in rows]
    t_cust  = [float(r[2]) if r[2] != "N/A" else None for r in rows]
    t_base  = [float(r[3]) for r in rows]
    t_cpus  = [float(r[4]) for r in rows]

    x_pos = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

    # ── Left: latency bar chart ──
    ax = axes[0]
    if has_custom and None not in t_cust:
        ax.bar(x_pos - width, t_cust,  width, label="Custom Kernel",   color="#7c3aed")
    ax.bar(x_pos,         t_base,  width, label="PyTorch GPU",     color="#2563eb")
    ax.bar(x_pos + width, t_cpus,  width, label="CPU",             color="#64748b")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("Latency (ms)", color="white")
    ax.set_title("Forward Pass Latency by Tensor Size", color="white", fontsize=13)
    ax.legend(facecolor="#1a1d27", labelcolor="white", framealpha=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    # ── Right: speedup over CPU ──
    ax2 = axes[1]
    sp_cust = [t_cpus[i] / t_cust[i] if (t_cust[i] and t_cust[i] > 0) else None
               for i in range(len(rows))]
    sp_base = [t_cpus[i] / t_base[i] if t_base[i] > 0 else 0
               for i in range(len(rows))]

    if has_custom and None not in sp_cust:
        ax2.plot(labels, sp_cust, "o-", color="#7c3aed", linewidth=2,
                 markersize=8, label="Custom Kernel")
    ax2.plot(labels, sp_base, "s-", color="#2563eb", linewidth=2,
             markersize=8, label="PyTorch GPU")
    ax2.axhline(1, color="#64748b", linestyle="--", linewidth=1, label="CPU baseline")
    ax2.set_ylabel("Speedup over CPU", color="white")
    ax2.set_title("GPU Speedup vs CPU", color="white", fontsize=13)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1a1d27", labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "benchmark_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓  Plot saved to {plot_path}")

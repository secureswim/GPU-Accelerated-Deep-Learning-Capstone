"""
plot_results.py

Reads training_log.csv and benchmark_results.csv from ../outputs/
and produces publication-quality figures.

Outputs:
  ../outputs/loss_curve.png        — train/val loss over epochs
  ../outputs/accuracy_curve.png    — val accuracy over epochs
  ../outputs/gpu_memory.png        — GPU memory usage over epochs
  ../outputs/epoch_time.png        — time per epoch
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ─── Style ───────────────────────────────────────────────────────────────────

BG       = "#0f1117"
AX_BG    = "#1a1d27"
GRID_CLR = "#2a2d3a"
C1       = "#7c3aed"   # purple  — train
C2       = "#2563eb"   # blue    — val / GPU
C3       = "#10b981"   # green   — accuracy
C4       = "#f59e0b"   # amber   — time

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    AX_BG,
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "white",
    "xtick.color":       "white",
    "ytick.color":       "white",
    "text.color":        "white",
    "grid.color":        GRID_CLR,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.facecolor":  AX_BG,
    "legend.labelcolor": "white",
    "font.size":         11,
})


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


# ─── Load training log ────────────────────────────────────────────────────────

log_path = os.path.join(OUT_DIR, "training_log.csv")
if not os.path.exists(log_path):
    print(f"training_log.csv not found at {log_path}")
    print("Run train.py first to generate the log.")
    exit(1)

epochs, train_loss, val_loss, val_acc, ep_time, gpu_mb = [], [], [], [], [], []
with open(log_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]) * 100)
        ep_time.append(float(row["epoch_time_s"]))
        gpu_mb.append(float(row["gpu_mem_mb"]))

# ─── 1. Loss curve ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, "o-", color=C1, linewidth=2, markersize=5,
        label="Train Loss")
ax.plot(epochs, val_loss,   "s-", color=C2, linewidth=2, markersize=5,
        label="Val Loss")
ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=C1)
_ax_style(ax, "Training & Validation Loss", "Epoch", "Cross-Entropy Loss")
plt.tight_layout()
p = os.path.join(OUT_DIR, "loss_curve.png")
plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓  {p}")

# ─── 2. Accuracy curve ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, val_acc, "o-", color=C3, linewidth=2.5, markersize=6,
        label="Val Accuracy")
ax.fill_between(epochs, val_acc, alpha=0.12, color=C3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
_ax_style(ax, "Validation Accuracy over Epochs", "Epoch", "Accuracy (%)")
plt.tight_layout()
p = os.path.join(OUT_DIR, "accuracy_curve.png")
plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓  {p}")

# ─── 3. GPU memory ───────────────────────────────────────────────────────────

if any(m > 0 for m in gpu_mb):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(epochs, gpu_mb, color=C2, alpha=0.85, label="Peak GPU Memory (MB)")
    ax.plot(epochs, gpu_mb, "o-", color=C1, linewidth=1.5, markersize=4)
    _ax_style(ax, "Peak GPU Memory per Epoch", "Epoch", "Memory (MB)")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "gpu_memory.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✓  {p}")
else:
    print("  (GPU memory column is zero — skipping gpu_memory.png)")

# ─── 4. Time per epoch ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(epochs, ep_time, color=C4, alpha=0.85, label="Epoch Wall-Clock Time (s)")
ax.axhline(np.mean(ep_time), color="white", linestyle="--", linewidth=1,
           label=f"Mean: {np.mean(ep_time):.1f}s")
_ax_style(ax, "Training Time per Epoch", "Epoch", "Time (s)")
plt.tight_layout()
p = os.path.join(OUT_DIR, "epoch_time.png")
plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓  {p}")

print("\nAll plots saved to", OUT_DIR)

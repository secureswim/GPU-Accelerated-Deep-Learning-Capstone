"""
profile_run.py

Runs a short training loop under torch.profiler to capture:
  - CPU vs CUDA kernel activity
  - Memory allocation timeline
  - Chrome-compatible trace JSON (open at chrome://tracing)

Usage:
    python profile_run.py [--steps N] [--batch-size N]

Outputs:
    ../outputs/trace/         — Chrome trace files (one per step)
    ../outputs/profile_summary.txt — Top operators by CUDA time
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--steps",      type=int, default=20,
                    help="Number of training steps to profile")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--data-dir",   type=str, default="./data")
parser.add_argument("--out-dir",    type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
parser.add_argument("--no-custom-kernel", action="store_true")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
trace_dir = os.path.join(args.out_dir, "trace")
os.makedirs(trace_dir, exist_ok=True)

# ─── Device check ────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠  No CUDA device — profiling CPU only (CUDA activities disabled).")
    activities = [ProfilerActivity.CPU]
else:
    print(f"✓  Profiling on: {torch.cuda.get_device_name(0)}")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

# ─── Model (same as train.py) ─────────────────────────────────────────────────

USE_CUSTOM = (not args.no_custom_kernel) and (device.type == "cuda")
if USE_CUSTOM:
    try:
        from custom_kernel import FusedGeluDropout
        def make_act(p): return FusedGeluDropout(p=p)
        print("✓  Using custom CUDA fused GELU+Dropout kernel")
    except Exception as e:
        print(f"⚠  Custom kernel failed: {e} — falling back to standard ops")
        USE_CUSTOM = False

if not USE_CUSTOM:
    def make_act(p): return nn.Sequential(nn.GELU(), nn.Dropout(p=p))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x))


class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
            ConvBlock(128, 256), ConvBlock(256, 256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), make_act(0.1),
            nn.Linear(512, 256),         make_act(0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model     = CIFAR10Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scaler    = GradScaler(enabled=(device.type == "cuda"))

# ─── Data ────────────────────────────────────────────────────────────────────

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
ds = datasets.CIFAR10(args.data_dir, train=True, transform=tf, download=True)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
data_iter = iter(loader)

def get_batch():
    global data_iter
    try:
        return next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        return next(data_iter)

# ─── Profiler schedule ────────────────────────────────────────────────────────
# wait=2  : skip first 2 steps (JIT warm-up)
# warmup=3: record but discard next 3 (CPU cache warm-up)
# active=args.steps : capture these steps
# repeat=1: one cycle

prof_schedule = schedule(wait=2, warmup=3, active=args.steps, repeat=1)

print(f"\nRunning profiler for {2 + 3 + args.steps} steps …")
model.train()

with profile(
    activities=activities,
    schedule=prof_schedule,
    on_trace_ready=lambda p: p.export_chrome_trace(
        os.path.join(trace_dir, f"trace_step{p.step_num}.json")
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,      # keep file size manageable
) as prof:
    for step in range(2 + 3 + args.steps):
        imgs, labels = get_batch()
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with record_function("forward"):
            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)

        with record_function("backward"):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        prof.step()

        if step % 5 == 0:
            print(f"  step {step:>3}  loss={loss.item():.4f}")

# ─── Print summary ────────────────────────────────────────────────────────────

summary_path = os.path.join(args.out_dir, "profile_summary.txt")
with open(summary_path, "w") as f:
    header = (
        f"Profiler Summary\n"
        f"Device       : {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})\n"
        f"Custom kernel: {USE_CUSTOM}\n"
        f"Batch size   : {args.batch_size}\n"
        f"Steps        : {args.steps}\n"
        f"{'='*60}\n\n"
    )
    f.write(header)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(
        sort_by=sort_key, row_limit=25
    )
    f.write(table)

    # Also write memory stats if available
    if device.type == "cuda":
        mem_table = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=15
        )
        f.write("\n\nTop operators by CUDA memory usage:\n")
        f.write(mem_table)

print(f"\n✓  Chrome traces : {trace_dir}/")
print(f"✓  Text summary  : {summary_path}")
print(f"\nTop 10 CUDA ops:")
print(prof.key_averages().table(
    sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
    row_limit=10
))
print("\nOpen chrome://tracing in Chrome and load any .json file from the trace/ folder.")

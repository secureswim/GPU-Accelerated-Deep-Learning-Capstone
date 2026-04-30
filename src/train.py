"""
train.py

GPU-accelerated CIFAR-10 classifier with:
  - Custom CUDA fused GELU+Dropout kernel (via custom_kernel.py)
  - Automatic Mixed Precision (AMP) via torch.cuda.amp
  - Per-epoch timing, GPU memory tracking, and loss/accuracy CSV logging

Usage:
    python train.py [--epochs N] [--batch-size N] [--no-amp] [--no-custom-kernel]

Outputs (written to ../outputs/):
    training_log.csv   — epoch, loss, accuracy, time, gpu_mem_mb
"""

import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─── CLI args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="GPU CIFAR-10 Trainer")
parser.add_argument("--epochs",           type=int,  default=15)
parser.add_argument("--batch-size",       type=int,  default=256)
parser.add_argument("--lr",               type=float,default=1e-3)
parser.add_argument("--dropout",          type=float,default=0.1)
parser.add_argument("--no-amp",           action="store_true",
                    help="Disable Automatic Mixed Precision")
parser.add_argument("--no-custom-kernel", action="store_true",
                    help="Use standard nn.GELU+nn.Dropout instead of custom kernel")
parser.add_argument("--data-dir",         type=str,  default="./data")
parser.add_argument("--out-dir",          type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
USE_AMP    = not args.no_amp
USE_CUSTOM = not args.no_custom_kernel

# ─── Device ──────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠  No CUDA device found — running on CPU (will be slow).")
    USE_AMP = False
else:
    print(f"✓  Using GPU: {torch.cuda.get_device_name(0)}")

# ─── Activation layer selection ──────────────────────────────────────────────

if USE_CUSTOM and device.type == "cuda":
    from custom_kernel import FusedGeluDropout
    def make_act(p):
        return FusedGeluDropout(p=p)
    print("✓  Using custom CUDA fused GELU+Dropout kernel")
else:
    def make_act(p):
        return nn.Sequential(nn.GELU(), nn.Dropout(p=p))
    tag = "no-custom-kernel flag set" if args.no_custom_kernel else "CPU mode"
    print(f"✓  Using standard nn.GELU + nn.Dropout  ({tag})")

# ─── Model ───────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Activation applied separately so we can swap it out easily
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x))


class CIFAR10Net(nn.Module):
    """
    A compact CNN for CIFAR-10.

    The fully-connected layers use the (optionally custom) fused GELU+Dropout
    kernel so that the GPU optimization is exercised on every forward pass.
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # ── Convolutional backbone ──────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(3,  64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),                    # 16×16

            ConvBlock(64,  128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),                    # 8×8

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),                    # 4×4
        )

        # ── Classifier head with custom fused activation ────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            make_act(dropout),                  # ← custom CUDA kernel here
            nn.Linear(512, 256),
            make_act(dropout),                  # ← and here
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── Data ────────────────────────────────────────────────────────────────────

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print(f"\nDownloading / loading CIFAR-10 …")
train_ds = datasets.CIFAR10(args.data_dir, train=True,  transform=train_tf, download=True)
val_ds   = datasets.CIFAR10(args.data_dir, train=False, transform=val_tf,   download=True)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

# ─── Training setup ──────────────────────────────────────────────────────────

model     = CIFAR10Net(dropout=args.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler    = GradScaler(enabled=USE_AMP)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters : {total_params:,}")
print(f"AMP enabled      : {USE_AMP}")
print(f"Custom kernel    : {USE_CUSTOM and device.type == 'cuda'}")
print(f"Epochs           : {args.epochs}")
print(f"Batch size       : {args.batch_size}\n")

# ─── CSV logger ──────────────────────────────────────────────────────────────

csv_path = os.path.join(args.out_dir, "training_log.csv")
csv_file = open(csv_path, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow(["epoch", "train_loss", "val_loss", "val_acc",
                 "epoch_time_s", "gpu_mem_mb"])

# ─── Train / Eval loops ───────────────────────────────────────────────────────

def train_one_epoch(epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate() -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(val_loader), correct / total


# ─── Main loop ───────────────────────────────────────────────────────────────

best_acc = 0.0
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time(s)':>8}  {'GPU MB':>7}")
print("─" * 60)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    train_loss            = train_one_epoch(epoch)
    val_loss, val_acc     = evaluate()
    scheduler.step()

    elapsed = time.time() - t0

    if device.type == "cuda":
        gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    else:
        gpu_mb = 0.0

    print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
          f"{val_acc*100:>6.2f}%  {elapsed:>8.2f}  {gpu_mb:>7.1f}")

    writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                     f"{val_acc:.6f}", f"{elapsed:.2f}", f"{gpu_mb:.1f}"])
    csv_file.flush()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, "best_model.pt"))

csv_file.close()
print(f"\n✓  Training complete. Best val accuracy: {best_acc*100:.2f}%")
print(f"✓  Log saved to: {csv_path}")

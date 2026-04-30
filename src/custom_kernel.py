"""
custom_kernel.py

Loads and wraps the fused GELU+Dropout CUDA kernel as a PyTorch autograd Function.
Uses torch.utils.cpp_extension.load() for JIT compilation — no separate build step needed.

Usage:
    from custom_kernel import FusedGeluDropout
    layer = FusedGeluDropout(p=0.1)
    out = layer(x)          # x: CUDA float32 tensor
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# ─── JIT compile the CUDA extension ─────────────────────────────────────────

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_FILE = os.path.join(_SRC_DIR, "custom_kernel.cu")

print("[custom_kernel] JIT-compiling CUDA extension … ", end="", flush=True)
_t0 = time.time()

_ext = load(
    name="fused_gelu_dropout",
    sources=[_CU_FILE],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

print(f"done ({time.time()-_t0:.1f}s)")


# ─── Autograd Function ───────────────────────────────────────────────────────

class _FusedGeluDropoutFn(torch.autograd.Function):
    """
    Custom autograd Function that delegates forward/backward
    to the hand-written CUDA kernels.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, p_drop: float, training: bool):
        if not training or p_drop == 0.0:
            # Inference path: GELU only, no dropout
            import math
            k0, k1 = 0.7978845608, 0.044715
            out = 0.5 * x * (1 + torch.tanh(k0 * (x + k1 * x.pow(3))))
            # Save a dummy mask of all-True so backward still works
            mask = torch.ones(x.numel(), dtype=torch.bool, device=x.device)
            ctx.save_for_backward(x, mask)
            ctx.p_drop = 0.0
            return out

        seed = torch.randint(0, 2**62, (1,)).item()
        out, mask = _ext.forward(x.contiguous(), p_drop, seed)
        ctx.save_for_backward(x, mask)
        ctx.p_drop = p_drop
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        grad_x = _ext.backward(
            grad_output.contiguous(), x, mask, ctx.p_drop
        )
        return grad_x, None, None   # no grad for p_drop or training flag


# ─── nn.Module wrapper ───────────────────────────────────────────────────────

class FusedGeluDropout(nn.Module):
    """
    Drop-in replacement for nn.GELU() + nn.Dropout(p) as a single fused op.

    Args:
        p (float): dropout probability (default 0.1)

    Example:
        >>> act = FusedGeluDropout(p=0.1).cuda()
        >>> y = act(torch.randn(1024, device='cuda'))
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0,1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("FusedGeluDropout requires a CUDA tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("FusedGeluDropout requires float32 input")
        return _FusedGeluDropoutFn.apply(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


# ─── Quick sanity check ──────────────────────────────────────────────────────

def _smoke_test():
    if not torch.cuda.is_available():
        print("[smoke test] No CUDA device found — skipping.")
        return

    device = "cuda"
    x = torch.randn(4096, device=device, requires_grad=True)

    layer = FusedGeluDropout(p=0.1).to(device)

    # Forward
    layer.train()
    y = layer(x)
    print(f"[smoke test] forward  OK — output shape {y.shape}, mean {y.mean():.4f}")

    # Backward
    loss = y.sum()
    loss.backward()
    print(f"[smoke test] backward OK — grad shape {x.grad.shape}, norm {x.grad.norm():.4f}")

    # Compare against PyTorch built-in (rough check)
    layer.eval()
    with torch.no_grad():
        x2 = x.detach()
        y_custom = layer(x2)
        y_ref    = torch.nn.functional.gelu(x2)
        max_diff = (y_custom - y_ref).abs().max().item()
    print(f"[smoke test] vs torch.gelu max|diff| = {max_diff:.2e}  (should be < 1e-5)")


if __name__ == "__main__":
    _smoke_test()

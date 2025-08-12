# EvoAttn package

See the root `README.md` and `README_RU.md` for full documentation.

Minimal usage:

```python
from evo_attn import EvoAttn, evo_attn
```

## EvoAttention (Triton + Torch fallback)

[Русская версия README](README_RU.md)

High-performance EvoAttention with causal and non-causal modes implemented in Triton, with a clean Torch fallback for full portability. Matches a reference PyTorch implementation numerically and provides 3–5x forward speedups on large sequences.

### Installation

```bash
pip install -U pip
pip install .  # from repo root
# Optional Triton on supported platforms (Linux, CUDA):
pip install 'evo-attn[triton]'
```

Windows users: Triton is primarily supported on Linux. Use WSL2 for best results; otherwise, the package falls back to the PyTorch reference kernels automatically.

### Quick start

```python
import torch
from evo_attn import EvoAttention, evo_attention

# Pre-projected tensors (B, H, L, D)
B, H, L, D = 2, 8, 4096, 64
q = torch.randn(B, H, L, D, device='cuda')
k = torch.randn(B, H, L, D, device='cuda')
v = torch.randn(B, H, L, D, device='cuda')
y = evo_attention(q, k, v, causal=True)

# Module with Q/K/V projections (drop-in block)
E = H * D
x = torch.randn(B, L, E, device='cuda')
layer = EvoAttention(embed_dim=E, num_heads=H).cuda()
y2 = layer(x, causal=True)
```

### Features

- Causal and non-causal modes
- Triton kernels with numerically stable reductions and compensated prefix-sums
- Automatic Torch fallback when Triton/CUDA is unavailable
- Optional `attention_mask` for padded tokens
- Supports fp16/bf16/fp32 inputs; controlled accumulation precision

### API

- `evo_attention(q, k, v, causal=True, attention_mask=None, accum_dtype=None, block_m=None, num_warps=None, num_stages=None)` → `(B,H,L,D)`
- `EvoAttention(embed_dim, num_heads, bias=False, out_proj=True)` → module for `(B,L,E)` → `(B,L,E)`

### Benchmarks

On a modern NVIDIA GPU, forward speedups of 3–5x vs. PyTorch reference and 1.1–1.6x in backward have been observed for long sequences. Your mileage may vary by GPU/driver/PyTorch.

### Compatibility

- Python ≥ 3.9, PyTorch ≥ 2.1
- Triton ≥ 2.1 (Linux recommended). On Windows, prefer WSL2 for Triton; otherwise the fallback is used.

### License

Apache-2.0


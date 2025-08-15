import torch
import torch.nn as nn

from .evo_causal import EvoAttentionCausalTriton, EvoAttentionCausalTorch
import os
import warnings


def _apply_attention_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        return q, k, v
    # Normalize mask to shape (B, 1, L, 1) with 1.0 for valid tokens
    if attention_mask.dtype != torch.float32 and attention_mask.dtype != torch.float16 and attention_mask.dtype != torch.bfloat16:
        attention_mask = attention_mask.to(dtype=q.dtype)
    if attention_mask.dim() == 2:  # (B, L)
        mask = attention_mask[:, None, :, None]
    elif attention_mask.dim() == 3:  # (B, 1, L)
        mask = attention_mask[:, :, :, None]
    elif attention_mask.dim() == 4:  # (B, 1, L, 1) already
        mask = attention_mask
    else:
        raise ValueError("attention_mask must have shape (B,L) or (B,1,L) or (B,1,L,1)")
    mask = mask.to(device=q.device, dtype=q.dtype)
    return q * mask, k * mask, v * mask


def evo_attn(
    v: torch.Tensor,
    *,
    causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    accum_dtype: torch.dtype | None = None,
    block_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """
    Functional EvoAttention core operating only on Value tensor (B, H, L, D).
    All projections and gating live under the hood.
    """
    if v.dim() != 4:
        raise ValueError("v must have shape (B, H, L, D)")

    # For compatibility, masking still accepted
    if attention_mask is not None:
        v = _apply_attention_mask(v, v, v, attention_mask)[2]

    # Internally, Q=K=V as placeholder; projections done inside the core
    backend = os.getenv("EVO_BACKEND", "auto").lower()
    triton_ok = hasattr(EvoAttentionCausalTriton, "forward")
    if backend == "torch" or (backend == "auto" and not triton_ok):
        core = EvoAttentionCausalTorch()
    else:
        try:
            core = EvoAttentionCausalTriton()
        except Exception as e:
            warnings.warn(f"Falling back to Torch backend due to Triton init error: {e}")
            core = EvoAttentionCausalTorch()
    # Core supports V-only call; single-tensor interface
    return core(v, causal=causal, accum_dtype=accum_dtype, block_m=block_m, num_warps=num_warps, num_stages=num_stages)


class EvoAttn(nn.Module):
    """
    High-level module: user passes only Value embedding (B, L, E),
    all linear projections and gating live inside the core.
    """

    def __init__(self, embed_dim: int, num_heads: int, *, head_dim: int | None = None, bias: bool = False, out_proj: bool = True):
        super().__init__()
        if head_dim is None and embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads when head_dim is None")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads) if head_dim is None else head_dim
        if head_dim is not None and self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError("num_heads * head_dim must equal embed_dim when head_dim is provided")

        # Projections are handled inside the core; keep an Identity for output for API symmetry
        self.out_proj = nn.Identity()

        backend = os.getenv("EVO_BACKEND", "auto").lower()
        try:
            if backend == "torch":
                self.core = EvoAttentionCausalTorch()
            else:
                self.core = EvoAttentionCausalTriton()
        except Exception as e:
            warnings.warn(f"Falling back to Torch backend due to Triton init error: {e}")
            self.core = EvoAttentionCausalTorch()
        self.ref = EvoAttentionCausalTorch()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
        accum_dtype: torch.dtype | None = None,
        block_m: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape (B, L, E)")
        batch, seq_len, _ = x.shape

        v = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if attention_mask is not None:
            v = _apply_attention_mask(v, v, v, attention_mask)[2]

        # Core handles Q/K/V internally; single-tensor call
        y = self.core(v, causal=causal, accum_dtype=accum_dtype, block_m=block_m, num_warps=num_warps, num_stages=num_stages)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(y)

# Backward-compatible aliases
evo_attention = evo_attn

class EvoAttention(EvoAttn):
    pass


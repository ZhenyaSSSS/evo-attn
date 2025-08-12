import torch
import torch.nn as nn

from .evo_causal import EvoAttentionCausalTriton, EvoAttentionCausalTorch


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
    q: torch.Tensor,
    k: torch.Tensor,
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
    Functional EvoAttention core operating on pre-projected tensors shaped (B, H, L, D).

    - q, k, v: tensors of shape (batch, heads, seq_len, head_dim)
    - causal: whether to apply causal prefix accumulation
    - attention_mask: optional mask with 1.0 for valid tokens; shapes supported: (B,L), (B,1,L), (B,1,L,1)
    - accum_dtype: optional accumulation dtype control (e.g., torch.float64)
    - block_m/num_warps/num_stages: low-level Triton tuning hints
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape (B, H, L, D)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")

    if attention_mask is not None:
        q, k, v = _apply_attention_mask(q, k, v, attention_mask)

    core = EvoAttentionCausalTriton()
    return core(
        q,
        k,
        v,
        causal=causal,
        accum_dtype=accum_dtype,
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


class EvoAttn(nn.Module):
    """
    Drop-in attention-like block with Q/K/V linear projections using EvoAttention core.

    Input shape: (B, L, E)
    Output shape: (B, L, E)
    """

    def __init__(self, embed_dim: int, num_heads: int, *, bias: bool = False, out_proj: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim) if out_proj else nn.Identity()

        self.core = EvoAttentionCausalTriton()
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

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _apply_attention_mask(q, k, v, attention_mask)

        y = self.core(
            q,
            k,
            v,
            causal=causal,
            accum_dtype=accum_dtype,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(y)

# Backward-compatible aliases
evo_attention = evo_attn

class EvoAttention(EvoAttn):
    pass


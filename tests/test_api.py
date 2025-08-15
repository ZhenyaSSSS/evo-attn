import os
import torch

from evo_attn import evo_attention, EvoAttention


def test_functional_shapes_cpu():
    B, H, L, D = 1, 2, 16, 8
    v = torch.randn(B, H, L, D)
    y = evo_attention(v)
    assert y.shape == (B, H, L, D)


def test_module_shapes_cpu():
    B, L, E, H = 2, 32, 64, 4
    x = torch.randn(B, L, E)
    mod = EvoAttention(embed_dim=E, num_heads=H)
    y = mod(x)
    assert y.shape == (B, L, E)


@torch.no_grad()
def test_masking_cpu():
    B, H, L, D = 1, 2, 8, 8
    v = torch.randn(B, H, L, D)
    mask = torch.zeros(B, L)
    mask[:, : L // 2] = 1
    y = evo_attention(v, attention_mask=mask)
    assert torch.allclose(y[:, :, L // 2 :, :], torch.zeros_like(y[:, :, L // 2 :, :]), atol=1e-5)


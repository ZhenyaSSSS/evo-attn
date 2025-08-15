import torch

from evo_attn.evo_causal import EvoAttentionCausalTorch


@torch.no_grad()
def test_forward_shapes_v6_torch():
    B, H, L, D = 4, 3, 20, 32
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)

    mod = EvoAttentionCausalTorch()
    out_c = mod(Q, K, V, causal=True)
    out_nc = mod(Q, K, V, causal=False)

    assert out_c.shape == (B, H, L, D)
    assert out_nc.shape == (B, H, L, D)


@torch.no_grad()
def test_causality_strict_v6_torch():
    B, H, L, D = 2, 2, 24, 16
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)

    mod = EvoAttentionCausalTorch()

    token_idx = L // 2
    full_out = mod(Q, K, V, causal=True)
    target = full_out[:, :, token_idx, :]

    Q_tr = Q[:, :, : token_idx + 1, :]
    K_tr = K[:, :, : token_idx + 1, :]
    V_tr = V[:, :, : token_idx + 1, :]
    trunc_out = mod(Q_tr, K_tr, V_tr, causal=True)
    last = trunc_out[:, :, -1, :]

    assert torch.allclose(target, last, atol=1e-6)


@torch.no_grad()
def test_non_causal_diff_v6_torch():
    B, H, L, D = 3, 2, 18, 32
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)

    mod = EvoAttentionCausalTorch()
    out_c = mod(Q, K, V, causal=True)
    out_nc = mod(Q, K, V, causal=False)

    assert not torch.allclose(out_c, out_nc)
    assert not torch.allclose(out_c[:, :, 0, :], out_nc[:, :, 0, :])


@torch.no_grad()
def test_batching_independence_v6_torch():
    B, H, L, D = 4, 2, 12, 16
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)

    mod = EvoAttentionCausalTorch()

    full = mod(Q, K, V, causal=True)
    part1 = mod(Q[: B // 2], K[: B // 2], V[: B // 2], causal=True)
    part2 = mod(Q[B // 2 :], K[B // 2 :], V[B // 2 :], causal=True)

    assert torch.allclose(full[: B // 2], part1, atol=1e-6)
    assert torch.allclose(full[B // 2 :], part2, atol=1e-6)



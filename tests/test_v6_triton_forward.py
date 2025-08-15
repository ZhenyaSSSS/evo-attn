import pytest
import torch

from evo_attn.evo_causal import EvoAttentionCausalTorch, EvoAttentionCausalTriton


def _gpu_and_triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu_and_triton_available(), reason="CUDA or Triton not available")
@torch.no_grad()
def test_v6_triton_forward_causal_matches_torch():
    device = "cuda"
    B, H, L, D = 2, 4, 1024, 64
    Q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)

    out_ref = ref(Q, K, V, causal=True)
    out_trt = trt(Q, K, V, causal=True)
    diff = (out_ref - out_trt).abs().max().item()
    assert diff < 5e-4


@pytest.mark.skipif(not _gpu_and_triton_available(), reason="CUDA or Triton not available")
@torch.no_grad()
def test_v6_triton_forward_noncausal_matches_torch():
    device = "cuda"
    B, H, L, D = 2, 4, 1024, 64
    Q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    ref = EvoAttentionCausalTorch().to(device)
    trt = EvoAttentionCausalTriton().to(device)

    out_ref = ref(Q, K, V, causal=False)
    out_trt = trt(Q, K, V, causal=False)
    diff = (out_ref - out_trt).abs().max().item()
    assert diff < 5e-4



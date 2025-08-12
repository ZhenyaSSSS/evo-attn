from .evo_causal import (
    EvoAttentionCausalTriton,
    EvoAttentionCausalTorch,
)
from .ops import EvoAttn, evo_attn, EvoAttention, evo_attention

__all__ = [
    "EvoAttn",
    "evo_attn",
    # Backward-compatible names
    "EvoAttention",
    "evo_attention",
    "EvoAttentionCausalTriton",
    "EvoAttentionCausalTorch",
]

__author__ = "Eugeny Sautkin"
__email__ = "evgenijsautkin29@gmail.com"
__version__ = "0.1.0"
__license__ = "Apache-2.0"


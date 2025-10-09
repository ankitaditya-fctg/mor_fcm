"""Transformer block with pre-norm residual structure."""

from __future__ import annotations

from typing import Optional, Tuple

from torch import nn, Tensor

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block reused inside the recursion module."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, rotary: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, rotary=rotary)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_override: Optional[Tuple[Tensor, Tensor]] = None,
        rotary_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, kv = self.attn(hidden_states, attn_mask=attn_mask, kv_override=kv_override, rotary_cache=rotary_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.ff(hidden_states)
        return hidden_states, kv

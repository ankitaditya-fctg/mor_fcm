"""Shared recursion block that reuses Transformer weights across steps."""

from __future__ import annotations

from typing import Optional, Tuple

from torch import nn, Tensor

from ..modules.transformer_block import TransformerBlock


class SharedRecursionBlock(nn.Module):
    """A Transformer block that is applied repeatedly across recursion steps."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers_shared: int,
        dropout: float = 0.0,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers_shared = n_layers_shared
        self.block = TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, rotary=rotary)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_override: Optional[Tuple[Tensor, Tensor]] = None,
        rotary_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        kv = kv_override
        output = hidden_states
        for _ in range(self.n_layers_shared):
            output, kv = self.block(output, attn_mask=attn_mask, kv_override=kv, rotary_cache=rotary_cache)
        return output, kv

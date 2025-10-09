"""Multi-head attention with KV cache hooks for Mixture-of-Recursions."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from .embeddings import apply_rotary_pos_emb, rotary_embedding


class MultiHeadAttention(nn.Module):
    """Lightweight multi-head attention supporting rotary embeddings and KV reuse."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rotary: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary = rotary

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: Tensor, seq_len: int, bsz: int) -> Tensor:
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_override: Optional[Tuple[Tensor, Tensor]] = None,
        rotary_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()

        q = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        if kv_override is None:
            k = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
            v = self._shape(self.v_proj(hidden_states), tgt_len, bsz)
        else:
            k, v = kv_override

        if self.rotary and rotary_cache is not None:
            cos, sin = rotary_cache
            cos = cos[:, :tgt_len, :].to(q.device)
            sin = sin[:, :tgt_len, :].to(q.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, (k, v)


def build_rotary_cache(max_len: int, head_dim: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create rotary cache for a given sequence length."""

    return rotary_embedding(max_len, head_dim, device=device)

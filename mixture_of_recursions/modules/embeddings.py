"""Embedding layers and rotary helpers for Mixture-of-Recursions."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class TokenPositionalEmbedding(nn.Module):
    """Token embedding with optional sinusoidal positional encodings."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional", pe, persistent=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        pos = self.positional[: input_ids.size(1)]
        return self.token_embed(input_ids) + pos


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to queries and keys."""

    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def rotary_embedding(sequence_length: int, dim: int, base: float = 10000.0, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """Create rotary embedding tensors."""

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(sequence_length, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb)[None, :, :], torch.sin(emb)[None, :, :]

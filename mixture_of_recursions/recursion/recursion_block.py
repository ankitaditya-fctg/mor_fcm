"""Shared recursion block that reuses Transformer weights across steps."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..modules.transformer_block import TransformerBlock


class _RecursionShard(nn.Module):
    """Sequential stack of Transformer blocks reused for a depth shard."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
        rotary: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, rotary=rotary)
            for _ in range(n_layers)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor],
        kv_override: Optional[Tuple[Tensor, Tensor]],
        rotary_cache: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        output = hidden_states
        kv = kv_override
        for idx, layer in enumerate(self.layers):
            layer_kv = kv if idx == 0 else None
            output, kv = layer(output, attn_mask=attn_mask, kv_override=layer_kv, rotary_cache=rotary_cache)
        return output, kv


class SharedRecursionBlock(nn.Module):
    """Transformer blocks shared across recursion depths with flexible parameter tying."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers_shared: int,
        max_recursions: int,
        sharing: str = "none",
        dropout: float = 0.0,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if n_layers_shared <= 0:
            raise ValueError("n_layers_shared must be positive")
        self.max_recursions = max_recursions
        self.sharing = sharing
        self.n_layers_shared = n_layers_shared
        mapping, shard_count = self._build_mapping(max_recursions, n_layers_shared, sharing)
        self.register_buffer("_depth_to_shard", torch.tensor(mapping, dtype=torch.long), persistent=False)
        self.shards = nn.ModuleList(
            _RecursionShard(d_model, n_heads, d_ff, n_layers_shared, dropout, rotary)
            for _ in range(shard_count)
        )

    @staticmethod
    def _build_mapping(max_recursions: int, n_layers_shared: int, sharing: str) -> Tuple[List[int], int]:
        if sharing not in {"none", "cycle", "middle_cycle"}:
            raise ValueError(f"Unknown sharing strategy: {sharing}")
        if max_recursions <= 0:
            raise ValueError("max_recursions must be positive")
        if sharing == "none":
            mapping = list(range(max_recursions))
            return mapping, max_recursions
        if sharing == "cycle":
            cycle = max(1, min(n_layers_shared, max_recursions))
            mapping = [depth % cycle for depth in range(max_recursions)]
            return mapping, cycle
        # middle_cycle
        cycle = max(1, min(n_layers_shared, max_recursions))
        window = min(max_recursions, cycle * 2)
        start = max(0, (max_recursions - window) // 2)
        end = min(max_recursions, start + window)
        mapping: List[int] = []
        shard_idx = 0
        for depth in range(start):
            mapping.append(shard_idx)
            shard_idx += 1
        window_base = shard_idx
        shard_idx += cycle
        for depth in range(start, end):
            mapping.append(window_base + (depth - start) % cycle)
        for depth in range(end, max_recursions):
            mapping.append(shard_idx)
            shard_idx += 1
        return mapping, shard_idx

    def shard_index(self, depth: int) -> int:
        if depth < 0 or depth >= self._depth_to_shard.numel():
            raise IndexError(f"depth {depth} out of range for max_recursions={self.max_recursions}")
        return int(self._depth_to_shard[depth].item())

    @property
    def head_dim(self) -> int:
        """Return the attention head dimension for rotary cache construction."""

        shard = self.shards[0]
        return shard.layers[0].attn.head_dim

    def forward(
        self,
        hidden_states: Tensor,
        depth: int,
        attn_mask: Optional[Tensor] = None,
        kv_override: Optional[Tuple[Tensor, Tensor]] = None,
        rotary_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        shard = self.shards[self.shard_index(depth)]
        return shard(hidden_states, attn_mask, kv_override, rotary_cache)

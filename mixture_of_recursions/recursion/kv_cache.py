"""Key/value cache management for recursive execution."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class KVCacheManager:
    """Manage per-recursion KV tensors."""

    def __init__(self, mode: str, max_recursions: int) -> None:
        if mode not in {"recursion", "share_first"}:
            raise ValueError(f"Unknown KV cache mode: {mode}")
        self.mode = mode
        self.max_recursions = max_recursions
        self.cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    def reset(self) -> None:
        self.cache.clear()

    def update(self, step: int, kv: Tuple[Tensor, Tensor]) -> None:
        """Store KV for a recursion step."""

        if self.mode == "share_first":
            if step == 0:
                self.cache[0] = tuple(t.detach() for t in kv)
        else:
            self.cache[step] = tuple(t.detach() for t in kv)

    def get(self, step: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Return cached KV tensors if available."""

        if self.mode == "share_first":
            return self.cache.get(0)
        return self.cache.get(step)

    def has(self, step: int) -> bool:
        if self.mode == "share_first":
            return 0 in self.cache
        return step in self.cache

    @staticmethod
    def combine_mask(attn_mask: Tensor, active_mask: Tensor) -> Tensor:
        """Utility to combine causal masks with token activity."""

        extended = attn_mask.clone()
        inactive = (~active_mask).float() * -1e4
        extended = extended + inactive[:, None, None, :]
        return extended

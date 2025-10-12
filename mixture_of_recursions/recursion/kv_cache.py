"""Key/value cache management for recursive execution."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from torch import Tensor


class KVCacheManager:
    """Manage per-recursion KV tensors."""

    def __init__(self, mode: str, max_recursions: int) -> None:
        if mode not in {"recursion", "share_first"}:
            raise ValueError(f"Unknown KV cache mode: {mode}")
        self.mode = mode
        self.max_recursions = max_recursions
        self.cache: Dict[int, Tuple[Tensor, Tensor]] = {}
        self.active_masks: Dict[int, Tensor] = {}

    def reset(self) -> None:
        self.cache.clear()
        self.active_masks.clear()

    def update(self, step: int, kv: Tuple[Tensor, Tensor], token_mask: Optional[Tensor] = None) -> None:
        """Store KV for a recursion step."""

        if self.mode == "share_first":
            if step == 0:
                self.cache[0] = tuple(t.detach() for t in kv)
                if token_mask is not None:
                    self.active_masks[0] = token_mask.detach().clone()
        else:
            self.cache[step] = tuple(t.detach() for t in kv)
            if token_mask is not None:
                self.active_masks[step] = token_mask.detach().clone()

    def get(self, step: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Return cached KV tensors if available."""

        if self.mode == "share_first":
            return self.cache.get(0)
        return self.cache.get(step)

    def get_mask(self, step: int) -> Optional[Tensor]:
        if self.mode == "share_first":
            return self.active_masks.get(0)
        return self.active_masks.get(step)

    def has(self, step: int) -> bool:
        if self.mode == "share_first":
            return 0 in self.cache
        return step in self.cache

    @staticmethod
    def combine_mask(attn_mask: Tensor, active_mask: Tensor) -> Tensor:
        """Utility to combine causal masks with token activity."""

        if attn_mask.dim() != 4:
            raise ValueError("attention mask must be broadcastable to [batch, heads, seq, seq]")
        base = attn_mask
        if attn_mask.size(0) == 1 and active_mask.size(0) > 1:
            base = attn_mask.expand(active_mask.size(0), -1, -1, -1)
        inactive = (~active_mask).unsqueeze(1).unsqueeze(2).float() * -1e4
        return base + inactive

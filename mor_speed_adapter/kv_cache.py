"""Helpers for managing per-token key/value caches."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


class LayerCache:
    """Maintain key/value tensors per token for a single transformer layer."""

    def __init__(self, key: Optional[torch.Tensor], value: Optional[torch.Tensor]):
        if key is None or value is None:
            self.keys: List[Optional[torch.Tensor]] = []
            self.values: List[Optional[torch.Tensor]] = []
        else:
            self.keys = [key[i : i + 1].detach().clone() for i in range(key.shape[0])]
            self.values = [value[i : i + 1].detach().clone() for i in range(value.shape[0])]

    def ensure_batch(self, batch_size: int) -> None:
        if len(self.keys) < batch_size:
            pad = batch_size - len(self.keys)
            self.keys.extend([None] * pad)
            self.values.extend([None] * pad)

    def select(self, indices: torch.Tensor) -> Optional[tuple]:
        if len(indices) == 0:
            return None
        keys = []
        values = []
        for idx in indices.tolist():
            tensor_k = self.keys[idx]
            tensor_v = self.values[idx]
            if tensor_k is None or tensor_v is None:
                return None
            keys.append(tensor_k)
            values.append(tensor_v)
        key = torch.cat(keys, dim=0)
        value = torch.cat(values, dim=0)
        return key, value

    def update(self, indices: torch.Tensor, present: tuple) -> None:
        if present is None:
            return
        key, value = present
        for offset, idx in enumerate(indices.tolist()):
            self.keys[idx] = key[offset : offset + 1].detach().clone()
            self.values[idx] = value[offset : offset + 1].detach().clone()

    def lengths(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        lens = []
        for idx in indices.tolist():
            tensor_k = self.keys[idx]
            if tensor_k is None:
                lens.append(0)
            else:
                lens.append(tensor_k.shape[-2])
        return torch.tensor(lens, device=device, dtype=torch.long).unsqueeze(-1)

    def empty_like(self, batch_size: int, template: torch.Tensor) -> None:
        self.keys = [template.new_empty((1,) + template.shape[1:]) for _ in range(batch_size)]
        self.values = [template.new_empty((1,) + template.shape[1:]) for _ in range(batch_size)]


def convert_past_to_layer_caches(past_key_values: Optional[Sequence[tuple]]) -> List[LayerCache]:
    if past_key_values is None:
        return []
    caches = []
    for key, value in past_key_values:
        caches.append(LayerCache(key, value))
    return caches


def ensure_cache_batch(caches: List[LayerCache], batch_size: int) -> None:
    for cache in caches:
        cache.ensure_batch(batch_size)


def to_standard_cache(caches: List[LayerCache]) -> List[tuple]:
    """Convert back to the standard HuggingFace cache format."""

    if not caches:
        return []
    batch_size = len(caches[0].keys)
    result: List[tuple] = []
    for cache in caches:
        keys = [k if k is not None else caches[0].keys[0].new_zeros((1,) + caches[0].keys[0].shape[1:]) for k in cache.keys]
        values = [v if v is not None else caches[0].values[0].new_zeros((1,) + caches[0].values[0].shape[1:]) for v in cache.values]
        result.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return result

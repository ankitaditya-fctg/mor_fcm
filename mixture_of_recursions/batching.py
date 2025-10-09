"""Utilities for depth-wise batching of active tokens."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def gather_active(hidden_states: Tensor, active_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Gather active tokens into a compact batch."""

    idx = torch.nonzero(active_mask, as_tuple=False)
    if idx.numel() == 0:
        return hidden_states.new_zeros(0, hidden_states.size(-1)), idx
    gathered = hidden_states[idx[:, 0], idx[:, 1]]
    return gathered, idx


def scatter_to_sequence(compact: Tensor, indices: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    """Scatter compact representations back to [batch, seq, dim]."""

    output = torch.zeros(shape, device=compact.device, dtype=compact.dtype)
    if indices.numel() == 0:
        return output
    output[indices[:, 0], indices[:, 1]] = compact
    return output

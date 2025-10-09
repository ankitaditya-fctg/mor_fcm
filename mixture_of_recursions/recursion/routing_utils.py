"""Utility helpers for routing decisions in Mixture-of-Recursions."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def topk_mask(scores: Tensor, keep_ratio: float) -> Tensor:
    """Return boolean mask keeping the top-k fraction of items along the last dim."""

    if keep_ratio >= 1.0:
        return torch.ones_like(scores, dtype=torch.bool)
    if keep_ratio <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    k = max(1, int(torch.ceil(torch.tensor(scores.size(-1) * keep_ratio)).item()))
    topk = torch.topk(scores, k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, topk, True)
    return mask


def entropy_loss(logits: Tensor) -> Tensor:
    """Shannon entropy regulariser for routing logits."""

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def straight_through_sample(logits: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Sample categorical distribution with straight-through gradients."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
    scores = (logits + gumbel) / temperature
    hard = torch.argmax(scores, dim=-1)
    one_hot = torch.nn.functional.one_hot(hard, num_classes=logits.size(-1)).float()
    probs = torch.softmax(logits / temperature, dim=-1)
    st = one_hot + probs - probs.detach()
    return hard, st

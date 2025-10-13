"""Routing modules for depth-adaptive decoding."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterHead(nn.Module):
    """Two-layer MLP producing routing logits over depths ``[0, R]``.

    The implementation intentionally keeps the module tiny – a single hidden
    layer with GELU non-linearity.  The head can optionally be driven with
    precomputed logits during tests via :meth:`set_manual_logits`.
    """

    def __init__(self, hidden_size: int, R: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.R = R
        self.temperature = temperature
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, R + 1)
        self._manual_logits: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._manual_logits is not None:
            return self._manual_logits
        out = self.fc1(hidden_states)
        out = self.act(out)
        logits = self.fc2(out)
        return logits

    def set_manual_logits(self, logits: Optional[torch.Tensor]) -> None:
        """Inject pre-computed logits (used in tests)."""

        self._manual_logits = logits

    def token_choice(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-token depths and logits for ``token_choice`` mode."""

        logits = self.forward(hidden_states)
        scaled = logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        depths = torch.argmax(probs, dim=-1)
        return depths, logits

    def hardness_scores(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hardness scores derived from routing logits."""

        logits = self.forward(hidden_states)
        scaled = logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)
        hardness = -entropy
        return hardness, logits

    def expert_choice(self, hidden_states: torch.Tensor, keep_ratio: float) -> Dict[str, torch.Tensor]:
        """Return indices sorted by hardness for ``expert_choice`` mode."""

        hardness, logits = self.hardness_scores(hidden_states)
        order = torch.argsort(hardness, descending=True)
        return {"order": order, "logits": logits, "hardness": hardness}


def depth_histogram(depths: Iterable[int]) -> Dict[int, int]:
    """Return a ``depth -> count`` histogram."""

    hist: Dict[int, int] = {}
    for depth in depths:
        hist[int(depth)] = hist.get(int(depth), 0) + 1
    return hist

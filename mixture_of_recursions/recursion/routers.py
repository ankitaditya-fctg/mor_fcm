"""Routers for Mixture-of-Recursions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .routing_utils import entropy_loss, straight_through_sample


@dataclass
class RoutingResult:
    """Holds routing metadata used by the main model."""

    depth_map: Tensor
    probs: Optional[Tensor]
    logits: Optional[Tensor]


class BaseRouter(nn.Module):
    """Base router class."""

    def __init__(self, d_model: int, temperature: float = 1.0, entropy_reg: float = 0.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.entropy_reg = entropy_reg
        self.last_loss: Tensor | None = None

    def loss(self) -> Tensor:
        return self.last_loss if self.last_loss is not None else torch.tensor(0.0)


class TokenChoiceRouter(BaseRouter):
    """Assign each token a recursion depth in one shot."""

    def __init__(
        self,
        d_model: int,
        min_depth: int,
        max_depth: int,
        straight_through: bool = True,
        temperature: float = 1.0,
        entropy_reg: float = 0.0,
    ) -> None:
        super().__init__(d_model, temperature=temperature, entropy_reg=entropy_reg)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_choices = max_depth - min_depth + 1
        self.proj = nn.Linear(d_model, self.depth_choices)
        self.straight_through = straight_through

    def assign_depths(self, hidden_states: Tensor) -> RoutingResult:
        logits = self.proj(hidden_states)
        if self.straight_through:
            depths_idx, probs = straight_through_sample(logits, temperature=self.temperature)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            depths_idx = torch.argmax(probs, dim=-1)
        depth_values = depths_idx + self.min_depth
        if self.entropy_reg > 0:
            self.last_loss = self.entropy_reg * entropy_loss(logits)
        else:
            self.last_loss = logits.new_tensor(0.0)
        return RoutingResult(depth_map=depth_values, probs=probs, logits=logits)


class ExpertChoiceRouter(BaseRouter):
    """Filter tokens at every recursion step keeping top fraction of scores."""

    def __init__(
        self,
        d_model: int,
        keep_ratio: float,
        temperature: float = 1.0,
        entropy_reg: float = 0.0,
    ) -> None:
        super().__init__(d_model, temperature=temperature, entropy_reg=entropy_reg)
        self.keep_ratio = keep_ratio
        self.scorer = nn.Linear(d_model, 1)

    def forward_step(self, hidden_states: Tensor, active_mask: Tensor) -> Tensor:
        """Return mask of tokens that continue to the next recursion step."""

        scores = self.scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~active_mask, float("-inf"))
        keep_mask = torch.zeros_like(active_mask)
        batch, seq = scores.shape
        for b in range(batch):
            active_idx = torch.nonzero(active_mask[b], as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                continue
            k = max(1, int(torch.ceil(active_idx.numel() * torch.tensor(self.keep_ratio)).item()))
            chosen = active_idx[torch.topk(scores[b, active_idx], k).indices]
            keep_mask[b, chosen] = True
        if self.entropy_reg > 0:
            logits = self.scorer.weight.new_zeros(batch, seq, 2)
            logits[..., 1] = scores
            logits[..., 0] = 0
            self.last_loss = self.entropy_reg * entropy_loss(logits)
        else:
            self.last_loss = scores.new_tensor(0.0)
        return keep_mask


def build_router(
    router_config,
    d_model: int,
    max_recursions: int,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Factory returning router module and metadata."""

    router_type = router_config.type
    if router_type == "token_choice":
        max_depth = router_config.max_depth or max_recursions
        router = TokenChoiceRouter(
            d_model,
            min_depth=router_config.min_depth,
            max_depth=max_depth,
            straight_through=router_config.straight_through,
            temperature=router_config.temperature,
            entropy_reg=router_config.entropy_reg,
        )
        meta = {"max_depth": max_depth, "min_depth": router_config.min_depth}
        return router, meta
    if router_type == "expert_choice":
        router = ExpertChoiceRouter(
            d_model,
            keep_ratio=router_config.keep_ratio,
            temperature=router_config.temperature,
            entropy_reg=router_config.entropy_reg,
        )
        meta = {"max_depth": max_recursions, "min_depth": 1}
        return router, meta
    raise ValueError(f"Unknown router type: {router_type}")

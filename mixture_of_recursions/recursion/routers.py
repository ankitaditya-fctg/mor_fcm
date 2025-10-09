"""Routers for Mixture-of-Recursions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .routing_utils import entropy_loss, straight_through_sample


@dataclass
class RoutingResult:
    """Holds routing metadata used by the main model."""

    depth_map: Tensor
    probs: Optional[Tensor]
    logits: Optional[Tensor]


@dataclass
class RoutingStatistics:
    """Summary statistics for router behaviour during a forward pass."""

    active_tokens: List[int]
    exits: List[int]
    avg_depth: float


class BaseRouter(nn.Module):
    """Base router class."""

    def __init__(self, d_model: int, temperature: float = 1.0, entropy_reg: float = 0.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.entropy_reg = entropy_reg
        self.last_loss: Tensor | None = None
        self.active_tokens: List[int] = []
        self.exited_tokens: List[int] = []

    def reset_stats(self) -> None:
        self.active_tokens.clear()
        self.exited_tokens.clear()

    def log_step(self, active: int, exited: int) -> None:
        self.active_tokens.append(active)
        self.exited_tokens.append(exited)

    def loss(self) -> Tensor:
        return self.last_loss if self.last_loss is not None else torch.tensor(0.0)

    def statistics(self, depth_map: Tensor, min_depth: int) -> RoutingStatistics:
        if depth_map.numel() == 0:
            avg_depth = float(min_depth)
        else:
            avg_depth = float(depth_map.float().mean().item())
        return RoutingStatistics(
            active_tokens=list(self.active_tokens),
            exits=list(self.exited_tokens),
            avg_depth=avg_depth,
        )


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
        self.reset_stats()
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
        depth_hist = torch.bincount(depths_idx.flatten(), minlength=self.depth_choices)
        cumulative = torch.cumsum(depth_hist.flip(0), dim=0).flip(0)
        for step in range(self.depth_choices):
            active = int(cumulative[step].item())
            exited = int(depth_hist[step].item())
            self.log_step(active, exited)
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
        batch, seq = scores.shape
        scores = scores.masked_fill(~active_mask, float("-inf"))
        keep_mask = torch.zeros_like(active_mask)
        if active_mask.any():
            active_counts = active_mask.sum(dim=-1)
            keep_counts = torch.ceil(active_counts.float() * self.keep_ratio).clamp_min(1).long()
            keep_counts = torch.where(active_counts == 0, active_counts, keep_counts)
            max_keep = int(keep_counts.max().item())
            if max_keep > 0:
                topk_indices = torch.topk(scores, k=max_keep, dim=-1).indices
                selector = (
                    torch.arange(max_keep, device=scores.device)[None, :] < keep_counts[:, None]
                )
                keep_mask.scatter_(1, topk_indices, selector)
            exited = (active_counts - keep_counts).clamp_min(0)
            self.log_step(int(active_counts.sum().item()), int(exited.sum().item()))
        else:
            self.log_step(0, 0)
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

"""Lightweight runtime metrics for the speed adapter."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class DecodeMetrics:
    total_tokens: int = 0
    elapsed: float = 0.0
    depth_hist: Dict[int, int] = field(default_factory=dict)

    def update(self, step_tokens: int, step_elapsed: float, depths: Iterable[int]) -> None:
        self.total_tokens += step_tokens
        self.elapsed += step_elapsed
        for depth in depths:
            depth = int(depth)
            self.depth_hist[depth] = self.depth_hist.get(depth, 0) + 1

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed == 0:
            return 0.0
        return self.total_tokens / self.elapsed

    @property
    def ms_per_token(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return (self.elapsed / self.total_tokens) * 1000.0

    def summary(self) -> Dict[str, float]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "ms_per_token": self.ms_per_token,
        }


class LatencyController:
    """Binary search controller adjusting ``keep_ratio`` to meet a budget."""

    def __init__(self, target_ms: float, initial_keep: float) -> None:
        self.target_ms = target_ms
        self.low = 0.1
        self.high = 1.0
        self.current = initial_keep

    def update(self, observed_ms: float) -> float:
        if observed_ms > self.target_ms * 1.1:
            self.high = self.current
        elif observed_ms < self.target_ms * 0.9:
            self.low = self.current
        self.current = (self.low + self.high) / 2
        return self.current

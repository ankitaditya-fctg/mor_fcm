"""Runtime configuration objects for MOR speed adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeedConfig:
    """Configuration controlling depth-adaptive decoding.

    Parameters mirror the requirements from the project specification.  The
    object intentionally remains lightweight so that callers can instantiate it
    directly or populate it from CLI arguments.
    """

    router: str = "expert_choice"
    R: int = 4
    base_layers: int = 0
    keep_ratio: float = 0.5
    temperature: float = 1.0
    entropy_weight: float = 0.0
    latency_budget_ms: Optional[float] = None
    return_depth_trace: bool = False
    device: Optional[str] = None
    dtype: Optional[str] = None
    seed: int = 0

    def validate(self) -> None:
        """Validate configuration values and raise ``ValueError`` on issues."""

        if self.router not in {"token_choice", "expert_choice"}:
            raise ValueError(f"Unsupported router type: {self.router}")
        if self.R < 0:
            raise ValueError("R must be non-negative")
        if self.base_layers < 0:
            raise ValueError("base_layers must be non-negative")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must lie in (0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive when provided")

    @property
    def router_mode(self) -> str:
        """Return the router mode after validation."""

        self.validate()
        return self.router

    def copy(self) -> "SpeedConfig":
        """Return a shallow copy of the configuration."""

        return SpeedConfig(**self.__dict__)

"""Configuration dataclasses for Mixture-of-Recursions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the shared MoR model backbone."""

    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers_shared: int = 2
    max_recursions: int = 3
    dropout: float = 0.1
    rotary: bool = False
    sharing: str = "none"


@dataclass
class RouterConfig:
    """Configuration for token routing inside Mixture-of-Recursions."""

    type: str = "token_choice"
    keep_ratio: float = 0.5
    min_depth: int = 1
    max_depth: Optional[int] = None
    temperature: float = 1.0
    entropy_weight: float = 0.0
    straight_through: bool = True
    target_depth: Optional[float] = None
    depth_penalty: float = 0.0


@dataclass
class KVConfig:
    """Configuration for the key/value cache strategy."""

    mode: str = "recursion"  # or "share_first"


@dataclass
class TrainConfig:
    """Configuration controlling the demo training loop."""

    seq_len: int = 64
    batch_size: int = 4
    lr: float = 3e-4
    steps: int = 100
    warmup: int = 10
    device: str = "cpu"
    fp16: bool = False
    log_interval: int = 10
    log_depth_hist_every: int = 0
    depth_hist_path: Optional[str] = None
    deterministic: bool = False
    seed: int = 0

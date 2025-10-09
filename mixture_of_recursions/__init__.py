"""Mixture-of-Recursions (MoR) package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import ModelConfig, RouterConfig, TrainConfig, KVConfig

__all__ = [
    "ModelConfig",
    "RouterConfig",
    "TrainConfig",
    "KVConfig",
    "MoRModel",
    "train",
    "generate",
]


def __getattr__(name: str) -> Any:
    if name == "MoRModel":
        module = import_module(".model", __name__)
        return getattr(module, name)
    if name == "train":
        module = import_module(".train", __name__)
        return getattr(module, name)
    if name == "generate":
        module = import_module(".inference", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

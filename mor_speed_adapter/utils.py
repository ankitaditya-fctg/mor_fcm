"""Utility helpers shared across the MOR speed adapter implementation."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the compute device with a small amount of logic.

    Parameters
    ----------
    preferred:
        Optional explicit device string supplied by the user.  When omitted we
        prefer CUDA when available and otherwise fall back to CPU.
    """

    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype: Optional[str], model_dtype: torch.dtype) -> torch.dtype:
    """Resolve the dtype to use for embeddings and router weights."""

    if dtype is None:
        return model_dtype
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unknown dtype string: {dtype}") from exc


def set_seed(seed: int) -> None:
    """Seed PyTorch's RNG for deterministic sampling in unit tests."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_p_sampling(logits: torch.Tensor, top_p: float, temperature: float = 1.0) -> torch.Tensor:
    """Perform nucleus (top-p) sampling over the final token distribution."""

    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must lie in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative <= top_p
    # Always keep at least the first token.
    mask[..., 0] = True
    filtered_probs = sorted_probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, sampled)
    return next_token.squeeze(-1)


@contextmanager
def timed_section() -> Iterable[None]:
    """Context manager returning elapsed wall clock time in seconds."""

    start = time.perf_counter()
    container: Tuple[float] = (0.0,)
    try:
        yield container
    finally:
        container = (time.perf_counter() - start,)


class SimpleTokenizer:
    """A whitespace tokenizer with a reversible vocabulary for testing.

    The tokenizer keeps a growing mapping between tokens and ids.  It is only
    intended for unit tests and CLI smoke runs when a HuggingFace tokenizer is
    not available on disk.
    """

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self.inverse = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

    def encode(self, text: str) -> Sequence[int]:
        tokens = text.strip().split()
        if not tokens:
            return [1]  # BOS
        ids = [1]
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse[idx] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.inverse.get(i, "<unk>") for i in ids if i not in (0, 1)]
        return " ".join(tokens)

    @property
    def eos_token_id(self) -> int:  # pragma: no cover - trivial property
        return 2

    @property
    def bos_token_id(self) -> int:  # pragma: no cover - trivial property
        return 1

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        attention_mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attention_mask}

    def to(self, device: torch.device) -> "SimpleTokenizer":  # pragma: no cover - trivial
        return self

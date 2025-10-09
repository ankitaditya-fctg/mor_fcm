"""Dataset utilities for the Mixture-of-Recursions demo."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset[Tensor]):
    """Simple dataset returning token sequences for language modelling."""

    def __init__(self, sequences: Sequence[Tensor]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tensor:  # type: ignore[override]
        return self.sequences[idx]


def _build_from_wikitext(seq_len: int, vocab_size: int, split: str) -> SequenceDataset:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError("datasets not available") from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(dataset["text"])
    tokens = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)
    sequences: List[Tensor] = []
    for i in range(0, tokens.numel() - seq_len - 1, seq_len):
        sequences.append(tokens[i : i + seq_len + 1])
    if not sequences:
        raise RuntimeError("failed to build sequences")
    return SequenceDataset(sequences)


def _build_synthetic(seq_len: int, vocab_size: int, num_samples: int = 512, seed: int = 0) -> SequenceDataset:
    rng = torch.Generator().manual_seed(seed)
    sequences = torch.randint(0, vocab_size, (num_samples, seq_len + 1), generator=rng)
    return SequenceDataset([seq for seq in sequences])


def get_dataloaders(seq_len: int, vocab_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders."""

    try:
        train_ds = _build_from_wikitext(seq_len, vocab_size, split="train")
        val_ds = _build_from_wikitext(seq_len, vocab_size, split="validation")
    except Exception:
        train_ds = _build_synthetic(seq_len, vocab_size)
        val_ds = _build_synthetic(seq_len, vocab_size, num_samples=64, seed=1)

    def _collate(batch: Sequence[Tensor]) -> Tensor:
        return torch.stack(batch, dim=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, val_loader

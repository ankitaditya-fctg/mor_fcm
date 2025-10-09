"""Inference helpers for Mixture-of-Recursions."""

from __future__ import annotations

from typing import List, Tuple

import torch

from .model import MoRModel


def generate(model: MoRModel, prompt: str, tokenizer=None, max_new_tokens: int = 32) -> Tuple[str, List[int]]:
    """Greedy generation helper with optional tokenizer."""

    model.eval()
    device = next(model.parameters()).device
    if tokenizer is None:
        input_ids = torch.tensor([[ord(c) % model.config.vocab_size for c in prompt]], device=device)
    else:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated, depths = model.generate(input_ids, max_new_tokens=max_new_tokens)
    if tokenizer is None:
        text = "".join(chr(int(t.item())) for t in generated[0])
    else:
        text = tokenizer.decode(generated[0].tolist())
    return text, depths

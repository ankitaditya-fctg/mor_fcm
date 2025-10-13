"""Minimal adapters for HuggingFace causal decoder architectures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class DecoderAdapter:
    model: torch.nn.Module
    layers: Sequence[torch.nn.Module]
    norm: torch.nn.Module
    lm_head: torch.nn.Module
    device: torch.device
    hidden_size: int

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[tuple],
    ) -> Tuple[torch.Tensor, tuple]:
        layer = self.layers[layer_idx]
        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=True,
            output_attentions=False,
        )
        hidden_states = outputs[0]
        present = outputs[1]
        return hidden_states, present

    def apply_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def _find_decoder_module(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the decoder stack (list of transformer blocks) for a model."""

    for attr in ("model", "transformer", "decoder"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if hasattr(candidate, "layers"):
                return candidate
    raise ValueError(
        "Unsupported architecture: expected model.model.layers/transformer.layers to exist."
    )


def _get_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        first_param = next(model.parameters())
    except StopIteration:  # pragma: no cover - models without parameters
        return torch.device("cpu")
    return first_param.device


def _get_hidden_size(model: torch.nn.Module, decoder: torch.nn.Module) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None:
        return hidden_size
    first_layer = getattr(decoder, "layers", None)
    if first_layer and len(first_layer) > 0:
        layer0 = first_layer[0]
        if hasattr(layer0, "hidden_size"):
            return layer0.hidden_size
    raise ValueError("Unable to infer hidden size from model config or layers")


def get_adapter(model: torch.nn.Module) -> DecoderAdapter:
    """Create a :class:`DecoderAdapter` for supported decoder-only models.

    The adapter relies on structural conventions shared by Llama/Mistral/Qwen-like
    architectures used in Hugging Face Transformers. We only require the model to
    expose a decoder module with a ``layers`` attribute and a ``norm`` final layer.
    """

    decoder = _find_decoder_module(model)
    if not hasattr(decoder, "norm"):
        raise ValueError("Unsupported architecture: decoder.norm is required")
    if not hasattr(decoder, "embed_tokens"):
        raise ValueError("Unsupported architecture: decoder.embed_tokens is required")

    return DecoderAdapter(
        model=decoder,
        layers=list(decoder.layers),
        norm=decoder.norm,
        lm_head=getattr(model, "lm_head"),
        device=_get_device(model),
        hidden_size=_get_hidden_size(model, decoder),
    )

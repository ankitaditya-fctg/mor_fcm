"""Minimal adapters for HuggingFace causal decoder architectures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

try:  # Transformers is an optional dependency in CI.
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
except Exception:  # pragma: no cover - optional dependency guard
    LlamaForCausalLM = None
    MistralForCausalLM = None


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


def get_adapter(model: torch.nn.Module) -> DecoderAdapter:
    if LlamaForCausalLM is not None and isinstance(model, LlamaForCausalLM):
        decoder = model.model
    elif MistralForCausalLM is not None and isinstance(model, MistralForCausalLM):
        decoder = model.model
    else:
        raise ValueError("Only LlamaForCausalLM and MistralForCausalLM are supported")

    return DecoderAdapter(
        model=decoder,
        layers=list(decoder.layers),
        norm=decoder.norm,
        lm_head=model.lm_head,
        device=model.device,
        hidden_size=decoder.layers[0].hidden_size,
    )

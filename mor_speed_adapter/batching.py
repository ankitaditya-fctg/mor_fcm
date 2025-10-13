"""Depth-wise batching executor for layer skipping."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .config import SpeedConfig
from .hf_patch import DecoderAdapter
from .kv_cache import LayerCache, ensure_cache_batch
from .router import RouterHead


class DepthwiseExecutor:
    """Run a single decode step with adaptive depth skipping."""

    def __init__(
        self,
        adapter: DecoderAdapter,
        router_head: RouterHead,
        config: SpeedConfig,
    ) -> None:
        self.adapter = adapter
        self.router_head = router_head
        self.config = config

    def _actual_R(self, num_layers: int, force_full: bool) -> int:
        extra_available = max(0, num_layers - self.config.base_layers)
        if force_full:
            return extra_available
        return min(self.config.R, extra_available)

    def __call__(
        self,
        input_ids: torch.Tensor,
        caches: List[LayerCache],
        force_full: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        ensure_cache_batch(caches, batch_size)
        hidden_states = self.adapter.embed(input_ids)

        num_layers = len(self.adapter.layers)
        extra_R = self._actual_R(num_layers, force_full)

        if self.config.base_layers > num_layers:
            raise ValueError("base_layers exceeds model depth")

        # Run the shared base layers for all tokens.
        base_indices = torch.arange(batch_size, device=device)
        for layer_idx in range(self.config.base_layers):
            layer_cache = caches[layer_idx]
            past = layer_cache.select(base_indices)
            position_ids = layer_cache.lengths(base_indices, device)
            outputs, present = self.adapter.forward_layer(
                layer_idx,
                hidden_states,
                position_ids=position_ids,
                past_key_value=past,
            )
            hidden_states = outputs
            layer_cache.update(base_indices, present)

        token_states = hidden_states[:, -1, :]
        depth_trace = torch.zeros(batch_size, dtype=torch.long, device=device)

        if force_full:
            depths = torch.full((batch_size,), extra_R, device=device, dtype=torch.long)
            router_logits = self.router_head.forward(token_states)
        elif self.config.router == "token_choice":
            depths, router_logits = self.router_head.token_choice(token_states)
            depths = depths.clamp(max=extra_R)
        else:
            expert_info = self.router_head.expert_choice(token_states, self.config.keep_ratio)
            router_logits = expert_info["logits"]
            order = expert_info["order"]
            active = order
        remaining = depths.clone() if self.config.router == "token_choice" or force_full else None

        # Depth-wise execution beyond the base layers.
        for offset in range(extra_R):
            layer_idx = self.config.base_layers + offset
            layer_cache = caches[layer_idx]
            if force_full or self.config.router == "token_choice":
                current = torch.nonzero(remaining > 0, as_tuple=False).view(-1)
                if len(current) == 0:
                    break
            else:  # expert choice
                if len(active) == 0:
                    break
                keep = max(1, math.ceil(self.config.keep_ratio * len(active)))
                current = active[:keep]
                active = current
            past = layer_cache.select(current)
            if past is None:
                past = layer_cache.select(current)
            if len(current) == 0:
                continue
            subset_hidden = hidden_states.index_select(0, current)
            position_ids = layer_cache.lengths(current, device)
            outputs, present = self.adapter.forward_layer(
                layer_idx,
                subset_hidden,
                position_ids=position_ids,
                past_key_value=past,
            )
            hidden_states[current] = outputs
            layer_cache.update(current, present)
            depth_trace[current] += 1
            if remaining is not None:
                remaining[current] -= 1

        hidden_states = self.adapter.apply_norm(hidden_states)
        logits = self.adapter.lm_head(hidden_states[:, -1, :])
        return {"logits": logits, "depth_trace": depth_trace}

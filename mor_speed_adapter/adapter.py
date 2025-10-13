"""User facing entry-points for depth-adaptive decoding."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .batching import DepthwiseExecutor
from .config import SpeedConfig
from .hf_patch import DecoderAdapter, get_adapter
from .kv_cache import LayerCache, convert_past_to_layer_caches
from .metrics import DecodeMetrics, LatencyController
from .router import RouterHead
from .utils import resolve_device, resolve_dtype, set_seed, top_p_sampling


class SpeedModel:
    """Wrapper that exposes a ``generate_speed`` method."""

    def __init__(self, model, tokenizer, config: SpeedConfig) -> None:
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = resolve_device(config.device)
        self.model = self.model.to(self.device)
        self.adapter = get_adapter(self.model)
        dtype = resolve_dtype(config.dtype, self.model.dtype)
        self.router_head = RouterHead(self.adapter.hidden_size, config.R, config.temperature).to(self.device, dtype=dtype)
        self.executor = DepthwiseExecutor(self.adapter, self.router_head, self.config)
        self.keep_ratio = config.keep_ratio
        self.latency_controller: Optional[LatencyController] = None
        if config.latency_budget_ms is not None:
            self.latency_controller = LatencyController(config.latency_budget_ms, self.keep_ratio)
        set_seed(config.seed)

    def _decode_step(
        self,
        input_ids: torch.Tensor,
        caches: List[LayerCache],
        force_full: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.config.keep_ratio = self.keep_ratio
        return self.executor(input_ids, caches, force_full=force_full)

    def set_keep_ratio(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError("keep_ratio must lie in (0, 1]")
        self.keep_ratio = value
        self.config.keep_ratio = value

    def set_latency_budget_ms(self, value: Optional[float]) -> None:
        self.config.latency_budget_ms = value
        if value is None:
            self.latency_controller = None
        else:
            self.latency_controller = LatencyController(value, self.keep_ratio)

    def generate_speed(
        self,
        prompt_text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        top_p: float = 0.9,
        temperature: float = 1.0,
        no_skip: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if prompt_text is None and input_ids is None:
            raise ValueError("Either prompt_text or input_ids must be provided")
        if prompt_text is not None:
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"]
        assert input_ids is not None
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        caches = convert_past_to_layer_caches(outputs.past_key_values)
        generated = input_ids.clone()
        metrics = DecodeMetrics()
        depth_traces: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            if do_sample:
                next_token = top_p_sampling(logits, top_p, temperature)
            else:
                next_token = torch.argmax(logits, dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            step_input = next_token.unsqueeze(-1)
            force_full = no_skip or math.isclose(self.keep_ratio, 1.0, rel_tol=1e-6)
            start = time.perf_counter()
            result = self._decode_step(step_input, caches, force_full=force_full)
            elapsed = time.perf_counter() - start
            logits = result["logits"]
            depth_traces.append(result["depth_trace"])
            metrics.update(1, elapsed, result["depth_trace"].tolist())
            if self.latency_controller is not None:
                self.keep_ratio = self.latency_controller.update(metrics.ms_per_token)

        return {
            "sequences": generated,
            "logits": logits,
            "depth_trace": depth_traces,
            "metrics": metrics.summary(),
        }


def wrap_model(model, tokenizer, config: SpeedConfig) -> SpeedModel:
    return SpeedModel(model, tokenizer, config)


@contextmanager
def speed_mode(model, tokenizer, config: SpeedConfig):
    wrapper = wrap_model(model, tokenizer, config)
    try:
        yield wrapper
    finally:
        pass

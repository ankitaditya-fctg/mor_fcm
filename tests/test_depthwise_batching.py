import torch
import pytest

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from mor_speed_adapter import SpeedConfig, wrap_model


def test_depthwise_batching_reduces_tokens():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
    )
    model = LlamaForCausalLM(config)
    tokenizer = lambda text, return_tensors="pt": {"input_ids": torch.tensor([[1, 2]])}
    tokenizer.decode = lambda ids: ""

    speed_model = wrap_model(
        model,
        tokenizer,
        SpeedConfig(router="token_choice", R=3, base_layers=1, keep_ratio=0.5),
    )

    counts = [0 for _ in range(len(model.model.layers))]

    for idx, layer in enumerate(model.model.layers):
        original_forward = layer.forward

        def counted_forward(hidden_states, *args, _idx=idx, **kwargs):
            counts[_idx] += hidden_states.shape[0]
            return original_forward(hidden_states, *args, **kwargs)

        layer.forward = counted_forward

    logits_override = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], device=speed_model.device
    )
    speed_model.router_head.set_manual_logits(logits_override)

    input_ids = torch.tensor([[1, 4], [1, 5]])
    speed_model.generate_speed(input_ids=input_ids, max_new_tokens=1)

    assert counts[2] < counts[0]
    assert counts[3] < counts[0]

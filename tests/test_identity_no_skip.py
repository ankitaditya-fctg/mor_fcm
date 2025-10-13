import torch
import pytest

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from mor_speed_adapter import SpeedConfig, wrap_model


def test_identity_no_skip():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
    )
    model = LlamaForCausalLM(config)
    tokenizer = lambda text, return_tensors="pt": {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode = lambda ids: ""

    speed_model = wrap_model(model, tokenizer, SpeedConfig(router="expert_choice", R=2, keep_ratio=1.0))

    input_ids = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        baseline = model(input_ids=input_ids, use_cache=True)
        baseline_logits = baseline.logits[:, -1, :]
        next_token = torch.argmax(baseline_logits, dim=-1)
        second = model(
            input_ids=next_token.unsqueeze(-1),
            past_key_values=baseline.past_key_values,
            use_cache=True,
        )
    baseline_sequence = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    baseline_logits_next = second.logits[:, -1, :]

    result = speed_model.generate_speed(input_ids=input_ids, max_new_tokens=1, do_sample=False)
    assert torch.equal(result["sequences"], baseline_sequence)
    assert torch.allclose(result["logits"], baseline_logits_next, atol=1e-4, rtol=1e-4)

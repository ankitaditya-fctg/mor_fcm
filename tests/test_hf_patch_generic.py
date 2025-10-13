"""Tests for the generic Hugging Face adapter helpers."""
from types import SimpleNamespace

import torch

from mor_speed_adapter.hf_patch import get_adapter


class _DummyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        new_hidden = hidden_states + 1.0
        present = (
            hidden_states.new_zeros(1, 1, 1, hidden_states.size(-1)),
            hidden_states.new_zeros(1, 1, 1, hidden_states.size(-1)),
        )
        return new_hidden, present


class _DummyDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DummyLayer(hidden_size)])
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)


class _DummyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.model = _DummyDecoder(hidden_size, vocab_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=hidden_size)


def test_get_adapter_supports_qwen_like_model():
    model = _DummyModel()
    adapter = get_adapter(model)

    input_ids = torch.zeros(1, 1, dtype=torch.long)
    hidden = adapter.embed(input_ids)
    assert hidden.shape[-1] == model.config.hidden_size

    position_ids = torch.zeros(1, 1, dtype=torch.long)
    updated, present = adapter.forward_layer(0, hidden, position_ids, past_key_value=None)

    assert isinstance(updated, torch.Tensor)
    assert isinstance(present, tuple)
    assert adapter.apply_norm(updated).shape == updated.shape

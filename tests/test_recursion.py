import pytest

torch = pytest.importorskip("torch")

from mixture_of_recursions.config import ModelConfig, RouterConfig
from mixture_of_recursions.model import MoRModel
from mixture_of_recursions.recursion.recursion_block import SharedRecursionBlock


def test_shared_block_weight_sharing():
    block = SharedRecursionBlock(d_model=16, n_heads=4, d_ff=32, n_layers_shared=2, dropout=0.0)
    x = torch.randn(2, 5, 16)
    out1, _ = block(x)
    out2, _ = block(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_model_forward_deterministic():
    torch.manual_seed(0)
    model = MoRModel(ModelConfig(vocab_size=64, d_model=32, n_heads=4, d_ff=64, n_layers_shared=1, max_recursions=2), RouterConfig(type="token_choice", straight_through=False))
    input_ids = torch.randint(0, 64, (2, 10))
    out1 = model(input_ids)
    torch.manual_seed(0)
    out2 = model(input_ids)
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)

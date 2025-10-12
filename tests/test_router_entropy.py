import pytest

from mixture_of_recursions.config import ModelConfig, RouterConfig
from mixture_of_recursions.model import MoRModel

torch = pytest.importorskip("torch")


def _make_batch(seq_len: int = 8, batch_size: int = 2, vocab: int = 32) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)


def test_token_choice_entropy_gradients():
    batch = _make_batch()
    labels = batch.clone()

    model_cfg = ModelConfig(max_recursions=3, d_model=32, d_ff=64, n_heads=2, vocab_size=32)

    router_cfg_zero = RouterConfig(
        type="token_choice",
        min_depth=1,
        max_depth=3,
        straight_through=False,
        entropy_weight=0.0,
    )
    model_zero = MoRModel(model_cfg, router_cfg_zero)
    outputs_zero = model_zero(batch[:, :-1], labels=labels)
    outputs_zero["loss"].backward()
    grad_zero = model_zero.router.proj.weight.grad
    if grad_zero is not None:
        assert torch.allclose(grad_zero, torch.zeros_like(grad_zero))

    router_cfg = RouterConfig(
        type="token_choice",
        min_depth=1,
        max_depth=3,
        straight_through=False,
        entropy_weight=0.1,
    )
    model = MoRModel(model_cfg, router_cfg)
    outputs = model(batch[:, :-1], labels=labels)
    outputs["loss"].backward()
    grad = model.router.proj.weight.grad
    assert grad is not None and torch.any(grad.abs() > 0)


def test_expert_choice_entropy_gradients():
    torch.manual_seed(0)
    batch = _make_batch()
    labels = batch.clone()
    model_cfg = ModelConfig(max_recursions=3, d_model=32, d_ff=64, n_heads=2, vocab_size=32)

    router_cfg_zero = RouterConfig(type="expert_choice", keep_ratio=0.5, entropy_weight=0.0)
    model_zero = MoRModel(model_cfg, router_cfg_zero)
    outputs_zero = model_zero(batch[:, :-1], labels=labels)
    outputs_zero["loss"].backward()
    grad_zero = model_zero.router.scorer.weight.grad
    if grad_zero is not None:
        assert torch.allclose(grad_zero, torch.zeros_like(grad_zero))

    router_cfg = RouterConfig(type="expert_choice", keep_ratio=0.5, entropy_weight=0.2)
    model = MoRModel(model_cfg, router_cfg)
    outputs = model(batch[:, :-1], labels=labels)
    outputs["loss"].backward()
    grad = model.router.scorer.weight.grad
    assert grad is not None and torch.any(grad.abs() > 0)

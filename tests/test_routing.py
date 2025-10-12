import pytest

from mixture_of_recursions.config import RouterConfig
from mixture_of_recursions.recursion.routers import build_router

torch = pytest.importorskip("torch")


def test_token_choice_depth_assignment():
    config = RouterConfig(type="token_choice", min_depth=1, max_depth=4, straight_through=False)
    router, _ = build_router(config, d_model=8, max_recursions=4)
    hidden = torch.randn(2, 5, 8)
    result = router.assign_depths(hidden)
    assert result.depth_map.min() >= config.min_depth
    assert result.depth_map.max() <= config.max_depth
    assert result.probs.shape == (2, 5, config.max_depth - config.min_depth + 1)


def test_expert_choice_monotonic_activity():
    config = RouterConfig(type="expert_choice", keep_ratio=0.5)
    router, _ = build_router(config, d_model=8, max_recursions=4)
    torch.manual_seed(0)
    hidden = torch.randn(1, 6, 8)
    active = torch.ones(1, 6, dtype=torch.bool)
    counts = []
    for _ in range(3):
        counts.append(int(active.sum()))
        next_mask = router.forward_step(hidden, active)
        active = next_mask
    assert counts[0] >= counts[1] >= counts[2]

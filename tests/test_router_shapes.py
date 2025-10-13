import torch

from mor_speed_adapter.router import RouterHead


def test_router_shapes():
    hidden_size = 16
    batch_size = 5
    R = 3
    router = RouterHead(hidden_size, R)
    hidden = torch.randn(batch_size, hidden_size)

    logits = router(hidden)
    assert logits.shape == (batch_size, R + 1)

    depths, logits_token = router.token_choice(hidden)
    assert depths.shape == (batch_size,)
    assert torch.all((depths >= 0) & (depths <= R))

    expert = router.expert_choice(hidden, keep_ratio=0.5)
    assert expert["logits"].shape == (batch_size, R + 1)
    assert expert["order"].shape == (batch_size,)

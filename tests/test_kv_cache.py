import pytest

from mixture_of_recursions.recursion.kv_cache import KVCacheManager

torch = pytest.importorskip("torch")


def test_recursion_mode_stores_per_step():
    manager = KVCacheManager("recursion", max_recursions=3)
    kv1 = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    kv2 = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    manager.update(0, kv1)
    manager.update(1, kv2)
    got1 = manager.get(0)
    got2 = manager.get(1)
    assert got1[0].shape == kv1[0].shape
    assert not torch.allclose(got1[0], got2[0])


def test_share_first_reuses_initial_step():
    manager = KVCacheManager("share_first", max_recursions=3)
    kv = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    manager.update(0, kv)
    manager.update(1, (torch.randn_like(kv[0]), torch.randn_like(kv[1])))
    got0 = manager.get(0)
    got1 = manager.get(1)
    assert torch.allclose(got0[0], got1[0])
    assert torch.allclose(got0[1], got1[1])

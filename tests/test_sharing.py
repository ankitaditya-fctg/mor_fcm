import pytest

from mixture_of_recursions.recursion.recursion_block import SharedRecursionBlock

torch = pytest.importorskip("torch")


def _make_block(sharing: str, max_recursions: int = 4) -> SharedRecursionBlock:
    return SharedRecursionBlock(
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers_shared=1,
        max_recursions=max_recursions,
        sharing=sharing,
        dropout=0.0,
        rotary=False,
    )


def test_cycle_sharing_ties_parameters():
    block = _make_block("cycle", max_recursions=4)
    idx0 = block.shard_index(0)
    idx2 = block.shard_index(2)
    assert idx0 == idx2
    shard0 = block.shards[idx0]
    shard2 = block.shards[idx2]
    assert shard0 is shard2
    param0 = next(shard0.parameters())
    param2 = next(shard2.parameters())
    assert param0 is param2


def test_cycle_forward_parity():
    torch.manual_seed(0)
    block = _make_block("cycle", max_recursions=4)
    hidden = torch.randn(1, 3, 16)
    out0, _ = block(hidden, depth=0)
    out2, _ = block(hidden, depth=2)
    assert torch.allclose(out0, out2)


def test_middle_cycle_mapping():
    block = SharedRecursionBlock(
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers_shared=1,
        max_recursions=6,
        sharing="middle_cycle",
        dropout=0.0,
        rotary=False,
    )
    mapping = [block.shard_index(i) for i in range(6)]
    assert mapping[2] == mapping[3]
    assert mapping[2] != mapping[0]
    assert mapping[2] != mapping[-1]

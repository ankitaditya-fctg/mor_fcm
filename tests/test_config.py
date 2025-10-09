from mixture_of_recursions.config import ModelConfig, RouterConfig, TrainConfig, KVConfig


def test_config_defaults():
    model = ModelConfig()
    router = RouterConfig()
    train = TrainConfig()
    kv = KVConfig()
    assert model.max_recursions >= 1
    assert router.min_depth == 1
    assert train.seq_len > 0
    assert kv.mode in {"recursion", "share_first"}

import pytest

from mixture_of_recursions.config import KVConfig, ModelConfig, RouterConfig
from mixture_of_recursions.model import MoRModel

torch = pytest.importorskip("torch")


def test_forward_backward_and_generate():
    torch.manual_seed(0)
    model = MoRModel(ModelConfig(vocab_size=32, d_model=16, n_heads=4, d_ff=32, n_layers_shared=1, max_recursions=2), RouterConfig(type="token_choice", straight_through=False), KVConfig(mode="share_first"))
    input_ids = torch.randint(0, 32, (2, 12))
    outputs = model(input_ids[:, :-1], labels=input_ids)
    loss = outputs["loss"]
    loss.backward()
    assert loss.item() > 0
    assert outputs["logits"].shape[:2] == (2, 11)
    generated, depths = model.generate(input_ids[:1, :-1], max_new_tokens=5)
    assert generated.shape[1] == input_ids.shape[1] + 4
    assert len(depths) == 5

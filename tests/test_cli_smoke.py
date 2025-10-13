import subprocess
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM


def test_cli_smoke(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=4,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(tmp_path)

    cmd = [
        sys.executable,
        "speed_decode.py",
        "--model",
        str(tmp_path),
        "--router",
        "token_choice",
        "--R",
        "2",
        "--max_new_tokens",
        "2",
        "--prompt",
        "hello world",
        "--keep_ratio",
        "0.5",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Generated Text" in completed.stdout

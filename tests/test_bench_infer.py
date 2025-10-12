import re
import subprocess
import sys
from pathlib import Path


def test_bench_infer_smoke(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "bench_infer.py"
    cmd = [
        sys.executable,
        str(script),
        "--num_iters=3",
        "--num_warmup=1",
        "--batch_sizes=1",
        "--seq_len=64",
        "--tokens_to_generate=8",
        "--device=cpu",
        "--deterministic",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    numeric_line = next(
        line for line in lines if re.search(r"\d", line) and "batch" not in line and "=" not in line
    )
    parts = numeric_line.split()
    assert len(parts) >= 3
    tokens_per_sec = float(parts[1])
    ms_per_token = float(parts[2])
    assert tokens_per_sec > 0
    assert ms_per_token > 0

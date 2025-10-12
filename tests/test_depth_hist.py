import csv
from pathlib import Path

import pytest

from mixture_of_recursions.train import DepthHistogramLogger

torch = pytest.importorskip("torch")


def test_depth_histogram_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "hist.csv"
    logger = DepthHistogramLogger(max_depth=4, every=1, csv_path=str(csv_path), label="train")
    depth_map = torch.tensor([[0, 1, 2, 2]])
    logger.maybe_log(1, depth_map)
    logger.close()

    with csv_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["global_step", "depth", "count"]
    counts = {int(row[1]): int(row[2]) for row in rows[1:]}
    total = sum(counts.values())
    assert total == depth_map.numel()
    assert counts[0] == 1 and counts[2] == 2

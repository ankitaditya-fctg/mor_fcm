"""Plot depth histogram CSVs produced by Mixture-of-Recursions training."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting helper."""

    parser = argparse.ArgumentParser(description="Plot recursion depth histograms from CSV logs")
    parser.add_argument("csv", type=str, help="CSV file containing depth histogram rows.")
    parser.add_argument("output", type=str, help="Path to the PNG file to write.")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional step to filter on; defaults to aggregating all steps.",
    )
    return parser.parse_args()


def load_counts(path: Path, step_filter: int | None) -> Dict[int, int]:
    """Aggregate depth counts from the histogram CSV, optionally filtering by step."""

    counts: Dict[int, int] = defaultdict(int)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["global_step"])
            if step_filter is not None and step != step_filter:
                continue
            depth = int(row["depth"])
            count = int(row["count"])
            counts[depth] += count
    if not counts:
        raise ValueError("No histogram rows matched the provided filter")
    return counts


def plot_counts(counts: Dict[int, int], output: Path) -> None:
    """Render a bar chart visualising the depth histogram."""

    depths = sorted(counts.keys())
    values = [counts[d] for d in depths]
    plt.figure(figsize=(4, 3))
    plt.bar(depths, values, color="#3b82f6")
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.title("Recursion depth histogram")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    """Entry point for the histogram plotting script."""

    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    counts = load_counts(csv_path, args.step)
    plot_counts(counts, output_path)


if __name__ == "__main__":
    main()

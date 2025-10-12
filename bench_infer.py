"""Throughput microbenchmark for Mixture-of-Recursions inference."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from mixture_of_recursions.config import KVConfig, ModelConfig, RouterConfig
from mixture_of_recursions.model import MoRModel


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


@dataclass
class BenchmarkResult:
    """Aggregated statistics for a single batch-size benchmark run."""

    batch_size: int
    tokens_generated: int
    median_tps: float
    median_ms_per_token: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark script."""

    parser = argparse.ArgumentParser(description="Benchmark Mixture-of-Recursions decoding throughput")
    parser.add_argument("--model", type=str, default=None, help="Optional path to a saved state_dict for the model.")
    parser.add_argument("--seq_len", type=int, default=1024, help="Prompt sequence length for the prefill phase.")
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated list of batch sizes to benchmark.",
    )
    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup iterations excluded from timing.")
    parser.add_argument("--num_iters", type=int, default=20, help="Number of timed iterations per batch size.")
    parser.add_argument(
        "--router",
        choices=["none_fixed", "expert_choice", "token_choice"],
        default="token_choice",
        help="Router mode used during benchmarking.",
    )
    parser.add_argument(
        "--kv_mode",
        choices=["recursion", "share_first"],
        default="recursion",
        help="Key/value cache strategy.",
    )
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Keep ratio for expert-choice routing.")
    parser.add_argument("--R", type=int, default=4, help="Maximum number of recursion steps.")
    parser.add_argument(
        "--depth_fixed",
        type=int,
        default=1,
        help="Fixed recursion depth when router=none_fixed.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Seed all RNGs for reproducibility.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Inference device to run the benchmark on.",
    )
    parser.add_argument("--router_temperature", type=float, default=1.0, help="Temperature for router logits.")
    parser.add_argument("--router_entropy_weight", type=float, default=0.0, help="Entropy penalty weight.")
    parser.add_argument(
        "--tokens_to_generate",
        type=int,
        default=64,
        help="Number of decode steps executed per iteration.",
    )
    parser.add_argument("--report_csv", type=str, default=None, help="Optional CSV path to append benchmark results.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --deterministic is set.")
    return parser.parse_args()


def _determine_device(option: str) -> torch.device:
    """Resolve the torch device to use based on the CLI option."""

    if option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(option)


def _build_model(args: argparse.Namespace, device: torch.device) -> MoRModel:
    """Instantiate and optionally load a Mixture-of-Recursions model."""

    router_type = "token_choice" if args.router == "none_fixed" else args.router
    router_config = RouterConfig(
        type=router_type,
        keep_ratio=args.keep_ratio,
        min_depth=1,
        max_depth=args.R,
        temperature=args.router_temperature,
        entropy_weight=args.router_entropy_weight,
    )
    if args.router == "none_fixed":
        if args.depth_fixed > args.R:
            raise ValueError("depth_fixed cannot exceed R")
        router_config.min_depth = args.depth_fixed
        router_config.max_depth = args.depth_fixed
    model_config = ModelConfig(max_recursions=args.R, vocab_size=128, d_model=128, d_ff=256, n_heads=4)
    kv_config = KVConfig(mode=args.kv_mode)
    model = MoRModel(model_config, router_config, kv_config)
    if args.model is not None:
        path = Path(args.model)
        if path.exists():
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _time_decode(
    model: MoRModel,
    base_tokens: torch.Tensor,
    decode_steps: int,
    device: torch.device,
) -> float:
    """Measure elapsed time for greedy decoding of ``decode_steps`` tokens."""

    with torch.no_grad():
        generated = base_tokens.clone()
        outputs = model(generated)
        if device.type == "cuda":
            torch.cuda.synchronize()  # type: ignore[attr-defined]
        start = time.perf_counter()
        for _ in range(decode_steps):
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            outputs = model(generated)
        if device.type == "cuda":
            torch.cuda.synchronize()  # type: ignore[attr-defined]
        end = time.perf_counter()
    return end - start


def _run_benchmark(args: argparse.Namespace, model: MoRModel, device: torch.device) -> List[BenchmarkResult]:
    """Execute the benchmark for all batch sizes and return aggregated results."""

    vocab = model.config.vocab_size
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    results: List[BenchmarkResult] = []
    for batch_size in batch_sizes:
        base_tokens = torch.randint(0, vocab, (batch_size, args.seq_len), device=device)
        durations: List[float] = []
        iterations = args.num_warmup + args.num_iters
        for it in range(iterations):
            elapsed = _time_decode(model, base_tokens, args.tokens_to_generate, device)
            if it >= args.num_warmup:
                durations.append(elapsed)
        if not durations:
            raise RuntimeError("No timed iterations recorded")
        tokens_generated = batch_size * args.tokens_to_generate
        tokens_per_sec = [tokens_generated / d for d in durations]
        median_tps = float(np.median(tokens_per_sec))
        per_token_ms = [(d / tokens_generated) * 1000.0 for d in durations]
        median_ms = float(np.median(per_token_ms))
        results.append(
            BenchmarkResult(
                batch_size=batch_size,
                tokens_generated=tokens_generated,
                median_tps=median_tps,
                median_ms_per_token=median_ms,
            )
        )
    return results


def _print_table(args: argparse.Namespace, device: torch.device, results: Iterable[BenchmarkResult]) -> None:
    """Pretty-print benchmark results to stdout."""

    print(
        f"router={args.router} kv_mode={args.kv_mode} R={args.R} device={device.type} tokens={args.tokens_to_generate}"
    )
    header = f"{'batch':>7} {'median tok/s':>15} {'p50 ms/token':>15}"
    print(header)
    for result in results:
        print(
            f"{result.batch_size:>7} {result.median_tps:>15.2f} {result.median_ms_per_token:>15.3f}"
        )


def _write_csv(args: argparse.Namespace, results: Iterable[BenchmarkResult]) -> None:
    """Append benchmark results to ``args.report_csv`` when provided."""

    if args.report_csv is None:
        return
    path = Path(args.report_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "router",
                    "kv_mode",
                    "batch_size",
                    "seq_len",
                    "R",
                    "keep_ratio",
                    "tokens_generated",
                    "median_tokens_per_sec",
                    "p50_ms_per_token",
                ]
            )
        for result in results:
            writer.writerow(
                [
                    args.router,
                    args.kv_mode,
                    result.batch_size,
                    args.seq_len,
                    args.R,
                    args.keep_ratio,
                    result.tokens_generated,
                    result.median_tps,
                    result.median_ms_per_token,
                ]
            )


def main() -> None:
    """Entry point for the throughput microbenchmark."""

    args = parse_args()
    device = _determine_device(args.device)
    if args.deterministic:
        _seed_everything(args.seed)
    model = _build_model(args, device)
    results = _run_benchmark(args, model, device)
    _print_table(args, device, results)
    _write_csv(args, results)


if __name__ == "__main__":
    main()

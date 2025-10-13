"""CLI demo for MOR depth-adaptive decoding."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import torch

try:  # pragma: no cover - optional dependency guard
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from mor_speed_adapter import SpeedConfig, wrap_model
from mor_speed_adapter.utils import SimpleTokenizer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF repo id or local path")
    parser.add_argument("--router", choices=["token_choice", "expert_choice"], default="expert_choice")
    parser.add_argument("--R", type=int, default=4, help="Max extra layers after base pass")
    parser.add_argument("--base_layers", type=int, default=0)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0, help="Router temperature")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--latency_budget_ms", type=float, default=None)
    parser.add_argument("--return_depth_trace", action="store_true")
    parser.add_argument("--no_skip", action="store_true", help="Disable skipping for parity checks")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling when >0")
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path for depth histogram")
    return parser


def load_tokenizer(model_path: str):
    if AutoTokenizer is None:
        return SimpleTokenizer()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    except Exception:
        return SimpleTokenizer()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = build_argparser()
    if args is None:
        args = parser.parse_args()

    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers must be installed to use the CLI")

    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        prompt_text = args.prompt or ""

    config = SpeedConfig(
        router=args.router,
        R=args.R,
        base_layers=args.base_layers,
        keep_ratio=args.keep_ratio,
        temperature=args.temperature,
        latency_budget_ms=args.latency_budget_ms,
        return_depth_trace=args.return_depth_trace,
        device=args.device,
    )

    wrapper = wrap_model(model, tokenizer, config)
    result = wrapper.generate_speed(
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.top_p > 0,
        top_p=args.top_p if args.top_p > 0 else 0.9,
        temperature=args.sample_temperature,
        no_skip=args.no_skip,
    )

    text = tokenizer.decode(result["sequences"][0].tolist()) if hasattr(tokenizer, "decode") else str(result["sequences"])
    print("=== Generated Text ===")
    print(text)
    print("=== Metrics ===")
    metrics = result.get("metrics", {})
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

    if args.return_depth_trace:
        traces = torch.stack(result["depth_trace"]).sum(dim=0)
        print("=== Depth Trace ===")
        print(traces.tolist())

    if args.csv:
        histogram = {}
        for trace in result["depth_trace"]:
            for value in trace.tolist():
                histogram[value] = histogram.get(value, 0) + 1
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["depth", "count"])
            for depth, count in sorted(histogram.items()):
                writer.writerow([depth, count])
        print(f"Saved histogram to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()

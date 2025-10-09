"""Command line demo for Mixture-of-Recursions."""

from __future__ import annotations

import argparse

import torch

from mixture_of_recursions.config import ModelConfig, RouterConfig, TrainConfig, KVConfig
from mixture_of_recursions.train import train
from mixture_of_recursions.inference import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixture-of-Recursions demo")
    parser.add_argument("--router", choices=["token_choice", "expert_choice"], default="token_choice")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--kv_mode", choices=["recursion", "share_first"], default="recursion")
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--target_depth", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(max_recursions=args.R, d_model=128, d_ff=256, n_heads=4, vocab_size=128)
    router_config = RouterConfig(type=args.router, keep_ratio=args.keep_ratio)
    if args.target_depth is not None:
        router_config.target_depth = args.target_depth
        router_config.depth_penalty = 0.1
    train_config = TrainConfig(steps=args.steps, device=args.device, seq_len=32, batch_size=4, lr=3e-4)
    kv_config = KVConfig(mode=args.kv_mode)

    model, val_loss = train(model_config, router_config, train_config, kv_config)
    print(f"Validation loss: {val_loss:.4f}")

    with torch.no_grad():
        sample = torch.randint(0, model_config.vocab_size, (1, train_config.seq_len + 1))
        stats = model(sample[:, :-1], labels=sample)
        avg_depth = float(stats.get("router_avg_depth", torch.tensor(0.0)).item())
        active = stats.get("router_active")
        exits = stats.get("router_exits")
        print("Router avg depth:", f"{avg_depth:.2f}")
        if active is not None:
            print("Active tokens per step:", active.tolist())
        if exits is not None:
            print("Exited tokens per step:", exits.tolist())

    prompt = "MiR:"
    text, depths = generate(model, prompt, max_new_tokens=16)
    print("Generated text:", text)
    print("Token depths:", depths)


if __name__ == "__main__":
    main()

"""Training utilities for Mixture-of-Recursions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import ModelConfig, RouterConfig, TrainConfig, KVConfig
from .data import get_dataloaders
from .model import MoRModel


def _lr_schedule(step: int, config: TrainConfig) -> float:
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    return config.lr


def evaluate(model: MoRModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch[:, :-1], labels=batch)
            total_loss += outputs["loss"].item() * batch.size(0)
            total_tokens += batch.size(0)
    model.train()
    return float(total_loss / max(total_tokens, 1))


def train(
    model_config: ModelConfig,
    router_config: RouterConfig,
    train_config: TrainConfig,
    kv_config: KVConfig | None = None,
) -> Tuple[MoRModel, float]:
    device = torch.device(train_config.device)
    model = MoRModel(model_config, router_config, kv_config)
    model.to(device)

    train_loader, val_loader = get_dataloaders(train_config.seq_len, model_config.vocab_size, train_config.batch_size)
    opt = optim.AdamW(model.parameters(), lr=train_config.lr)

    step_iter = iter(train_loader)
    for step in range(train_config.steps):
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(train_loader)
            batch = next(step_iter)
        batch = batch.to(device)
        inputs = batch[:, :-1]
        outputs = model(inputs, labels=batch)
        loss = outputs["loss"]
        loss.backward()
        for param_group in opt.param_groups:
            param_group["lr"] = _lr_schedule(step, train_config)
        opt.step()
        opt.zero_grad()
        if (step + 1) % train_config.log_interval == 0:
            ppl = torch.exp(outputs["loss"]).item()
            print(f"step {step+1}: loss={outputs['loss'].item():.4f} ppl={ppl:.2f}")

    val_loss = evaluate(model, val_loader, device)
    return model, val_loss

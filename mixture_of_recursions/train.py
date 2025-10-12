"""Training utilities for Mixture-of-Recursions."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .config import KVConfig, ModelConfig, RouterConfig, TrainConfig
from .data import get_dataloaders
from .model import MoRModel


class DepthHistogramLogger:
    """Utility to log and optionally persist recursion depth histograms."""

    def __init__(
        self,
        max_depth: int,
        every: int,
        csv_path: Optional[str] = None,
        label: str = "train",
    ) -> None:
        self.max_depth = max_depth
        self.every = every
        self.label = label
        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_writer: Optional[csv.writer] = None
        self._csv_file: Optional[object] = None

    def _ensure_csv(self) -> None:
        if self._csv_path is None or self._csv_writer is not None:
            return
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._csv_path.exists()
        self._csv_file = self._csv_path.open("a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(["global_step", "depth", "count"])

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _log(self, step: int, depth_map: torch.Tensor) -> None:
        counts = torch.bincount(depth_map.to(torch.long).flatten(), minlength=self.max_depth + 1)
        total = int(counts.sum().item())
        histogram = " ".join(f"{depth}:{int(count)}" for depth, count in enumerate(counts.tolist()))
        print(f"[{self.label}] depth-hist step={step} tokens={total} {histogram}")
        if self._csv_path is not None:
            self._ensure_csv()
            assert self._csv_writer is not None
            for depth, count in enumerate(counts.tolist()):
                self._csv_writer.writerow([step, depth, int(count)])
            self._csv_file.flush()  # type: ignore[union-attr]

    def maybe_log(self, step: int, depth_map: torch.Tensor) -> None:
        """Log the provided depth map when the step aligns with the cadence."""

        if self.every <= 0:
            return
        if step % self.every != 0:
            return
        self._log(step, depth_map)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _lr_schedule(step: int, config: TrainConfig) -> float:
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    return config.lr


def evaluate(
    model: MoRModel,
    loader: DataLoader,
    device: torch.device,
    hist_logger: Optional[DepthHistogramLogger] = None,
) -> float:
    """Evaluate the model on the validation loader and optionally log histograms."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            outputs = model(batch[:, :-1], labels=batch)
            total_loss += outputs["loss"].item() * batch.size(0)
            total_tokens += batch.size(0)
            if hist_logger is not None:
                hist_logger.maybe_log(step, outputs["depth_map"].detach().cpu())
    model.train()
    return float(total_loss / max(total_tokens, 1))


def train(
    model_config: ModelConfig,
    router_config: RouterConfig,
    train_config: TrainConfig,
    kv_config: KVConfig | None = None,
) -> Tuple[MoRModel, float]:
    """Train a Mixture-of-Recursions model and return the model plus validation loss."""

    device = torch.device(train_config.device)
    if train_config.deterministic:
        _seed_everything(train_config.seed)
    model = MoRModel(model_config, router_config, kv_config)
    model.to(device)

    train_loader, val_loader = get_dataloaders(train_config.seq_len, model_config.vocab_size, train_config.batch_size)
    opt = optim.AdamW(model.parameters(), lr=train_config.lr)

    depth_logger = DepthHistogramLogger(
        max_depth=model_config.max_recursions,
        every=train_config.log_depth_hist_every,
        csv_path=train_config.depth_hist_path,
        label="train",
    )

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
        depth_logger.maybe_log(step + 1, outputs["depth_map"].detach().cpu())
        if (step + 1) % train_config.log_interval == 0:
            ppl = torch.exp(outputs["loss"]).item()
            avg_depth = float(outputs.get("router_avg_depth", torch.tensor(0.0)).item())
            active_tokens = outputs.get("router_active")
            if active_tokens is not None and active_tokens.numel() > 0:
                active_summary = ",".join(str(int(v.item())) for v in active_tokens)
            else:
                active_summary = "-"
            print(
                f"step {step+1}: loss={outputs['loss'].item():.4f} ppl={ppl:.2f} avg_depth={avg_depth:.2f} active={active_summary}"
            )

    eval_logger = None
    if train_config.log_depth_hist_every > 0:
        eval_logger = DepthHistogramLogger(
            max_depth=model_config.max_recursions,
            every=train_config.log_depth_hist_every,
            csv_path=None,
            label="eval",
        )
    val_loss = evaluate(model, val_loader, device, eval_logger)
    depth_logger.close()
    if eval_logger is not None:
        eval_logger.close()
    return model, val_loss

"""Central training baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.eval.evaluator import EvalConfig, evaluate_model
from src.utils.mlflow_utils import log_artifacts, log_metrics
from src.utils.plots import plot_metric_curves


@dataclass(frozen=True)
class CentralConfig:
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = False
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    eval_cfg: EvalConfig = EvalConfig()
    output_dir: str = "./outputs"


def train_central(
    model: nn.Module,
    train_loader: DataLoader,
    holdout_loaders: Dict[str, DataLoader],
    cfg: CentralConfig,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=cfg.mixed_precision)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = _move_batch(batch, device)
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(batch)
                logits = outputs["logits"]
                labels = batch["label"]
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            steps += 1

        metrics = {"epoch": float(epoch), "train_loss": total_loss / max(steps, 1)}

        holdout_metrics = {}
        for name, loader in holdout_loaders.items():
            eval_metrics, artifacts = evaluate_model(model, loader, cfg.eval_cfg)
            for key, value in eval_metrics.items():
                holdout_metrics[f"holdout_{name}_{key}"] = value
            log_artifacts(artifacts, artifact_path=f"eval/{name}")

        metrics.update(holdout_metrics)
        log_metrics(metrics, step=epoch)
        history.append(metrics)

    # plot metric curves
    keys = [k for k in history[-1].keys() if k not in {"epoch"}]
    plot_path = Path(cfg.output_dir) / "central_metrics.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metric_curves(history, keys, plot_path)
    log_artifacts([plot_path], artifact_path="analysis")

    return history


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

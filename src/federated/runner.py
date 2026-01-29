"""Federated training runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.uncertainty import reliability_diagram
from src.utils.mlflow_utils import log_artifacts, log_metrics
from src.utils.plots import (
    plot_attention_bar,
    plot_metric_curves,
    plot_pr,
    plot_reliability,
    plot_roc,
    plot_topk_slices_grid,
)


@dataclass(frozen=True)
class RunnerConfig:
    rounds: int = 3
    log_client_metrics: bool = False
    output_dir: str = "./outputs"
    save_best: bool = True
    specificity_target: float = 0.95


def run_federated(
    server: FederatedServer,
    clients: list[FederatedClient],
    holdout_loaders: dict[str, DataLoader],
    cfg: RunnerConfig,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    best_auroc = -1.0
    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for round_idx in range(cfg.rounds):
        client_states = []
        weights = []
        round_loss = []
        for client in clients:
            metrics = client.train()
            round_loss.append(metrics["train_loss"])
            if cfg.log_client_metrics:
                log_metrics(
                    {f"client_{client.client_id}_loss": metrics["train_loss"]},
                    step=round_idx,
                )
            client_states.append(client.get_shared_state())
            weights.append(len(client.loader.dataset))

            # save private checkpoints each round
            private_state = client.get_private_state()
            if private_state:
                torch.save(
                    private_state,
                    ckpt_dir / f"private_{client.client_id}_round{round_idx}.pt",
                )

        server.aggregate(client_states, weights)
        shared_state = server.get_shared_state()
        for client in clients:
            client.load_shared_state(shared_state)

        metrics = {"round": float(round_idx), "train_loss": float(np.mean(round_loss))}
        holdout_metrics, artifacts = evaluate_holdouts(
            server.model, holdout_loaders, cfg.specificity_target, output_dir
        )
        metrics.update(holdout_metrics)
        log_metrics(metrics, step=round_idx)

        # save global shared checkpoint
        torch.save(shared_state, ckpt_dir / f"shared_round{round_idx}.pt")

        if cfg.save_best and holdout_metrics.get("holdout_auroc", -1.0) > best_auroc:
            best_auroc = holdout_metrics.get("holdout_auroc", -1.0)
            torch.save(shared_state, ckpt_dir / "shared_best.pt")

        for artifact in artifacts:
            log_artifacts([artifact], artifact_path="analysis")

        history.append(metrics)

    # plot global metrics across rounds
    keys = [k for k in history[-1] if k not in {"round"}]
    plot_path = Path(cfg.output_dir) / "federated_metrics.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metric_curves(history, keys, plot_path)
    log_artifacts([plot_path], artifact_path="analysis")

    return history


def evaluate_holdouts(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    specificity_target: float,
    output_dir: Path,
) -> tuple[dict[str, float], list[Path]]:
    model.eval()
    artifacts: list[Path] = []
    y_true_all: list[float] = []
    y_prob_all: list[float] = []
    attention_all: list[np.ndarray] = []
    first_slices = None
    first_attn = None

    for _, loader in loaders.items():
        y_true: list[float] = []
        y_prob: list[float] = []
        attention: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, next(model.parameters()).device)
                outputs = model(batch)
                p_fail = outputs["p_fail"].detach().cpu().numpy().tolist()
                y_prob.extend(p_fail)
                y_true.extend(batch["label"].detach().cpu().numpy().tolist())
                attn = outputs["attention_weights"].detach().cpu().numpy()
                attention.append(attn)
                if first_slices is None:
                    first_slices = batch["slices"].detach().cpu().numpy()
                    first_attn = attn
        y_true_all.extend(y_true)
        y_prob_all.extend(y_prob)
        attention_all.extend(attention)

    auroc = _auroc(np.array(y_true_all), np.array(y_prob_all))
    sens = _sensitivity_at_specificity(
        np.array(y_true_all), np.array(y_prob_all), specificity_target
    )
    metrics = {"holdout_auroc": auroc, "holdout_sensitivity_at_spec": sens}

    # artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    rel = reliability_diagram(torch.tensor(y_true_all), torch.tensor(y_prob_all))
    rel_path = output_dir / "reliability.json"
    rel_path.write_text(json.dumps(rel, indent=2), encoding="utf-8")
    artifacts.append(rel_path)
    artifacts.append(
        plot_roc(np.array(y_true_all), np.array(y_prob_all), output_dir / "roc.png")
    )
    artifacts.append(
        plot_pr(np.array(y_true_all), np.array(y_prob_all), output_dir / "pr.png")
    )
    artifacts.append(plot_reliability(rel, output_dir / "reliability.png"))

    cm = _confusion_matrix(np.array(y_true_all), np.array(y_prob_all))
    cm_path = output_dir / "confusion_matrix.json"
    cm_path.write_text(json.dumps(cm, indent=2), encoding="utf-8")
    artifacts.append(cm_path)

    if attention_all:
        mean_attn = np.mean(np.concatenate(attention_all, axis=0), axis=0).tolist()
        attn_path = output_dir / "attention_summary.json"
        attn_path.write_text(
            json.dumps({"mean_attention": mean_attn}, indent=2), encoding="utf-8"
        )
        artifacts.append(attn_path)
    if first_attn is not None:
        attn_vec = first_attn[0]
        artifacts.append(plot_attention_bar(attn_vec, output_dir / "attention_bar.png"))
        if first_slices is not None:
            topk = np.argsort(-attn_vec)[: min(5, len(attn_vec))].tolist()
            artifacts.append(
                plot_topk_slices_grid(
                    first_slices[0], topk, output_dir / "topk_slices.png"
                )
            )

    return metrics, artifacts


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    positives = y_true.sum()
    negatives = len(y_true) - positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / max(positives, 1)
    fpr = fps / max(negatives, 1)
    return fpr, tpr


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = _roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def _sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, spec: float
) -> float:
    fpr, tpr = _roc_curve(y_true, y_score)
    specificity = 1 - fpr
    idx = np.where(specificity >= spec)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def _confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5
) -> dict[str, int]:
    pred = (y_prob >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

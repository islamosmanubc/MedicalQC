"""Evaluation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval.metrics import compute_metrics, confusion_matrix
from src.models.uncertainty import reliability_diagram
from src.utils.plots import plot_attention_bar, plot_pr, plot_reliability, plot_roc, plot_topk_slices_grid


@dataclass(frozen=True)
class EvalConfig:
    specificity_target: float = 0.95
    sensitivity_target: float = 0.8
    output_dir: str = "./outputs"


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, cfg: EvalConfig
) -> Tuple[Dict[str, float], List[Path]]:
    model.eval()
    y_true: List[float] = []
    y_prob: List[float] = []
    attention: List[np.ndarray] = []
    first_slices = None
    first_attn = None

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

    y_true_arr = np.array(y_true, dtype=np.float32)
    y_prob_arr = np.array(y_prob, dtype=np.float32)

    metrics = compute_metrics(
        y_true_arr,
        y_prob_arr,
        specificity_target=cfg.specificity_target,
        sensitivity_target=cfg.sensitivity_target,
    )

    artifacts: List[Path] = []
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true_arr, y_prob_arr)
    cm_path = output_dir / "confusion_matrix.json"
    cm_path.write_text(json.dumps(cm, indent=2), encoding="utf-8")
    artifacts.append(cm_path)

    rel = reliability_diagram(torch.tensor(y_true_arr), torch.tensor(y_prob_arr))
    rel_path = output_dir / "reliability.json"
    rel_path.write_text(json.dumps(rel, indent=2), encoding="utf-8")
    artifacts.append(rel_path)

    # plots
    artifacts.append(plot_roc(y_true_arr, y_prob_arr, output_dir / "roc.png"))
    artifacts.append(plot_pr(y_true_arr, y_prob_arr, output_dir / "pr.png"))
    artifacts.append(plot_reliability(rel, output_dir / "reliability.png"))

    if attention:
        mean_attn = np.mean(np.concatenate(attention, axis=0), axis=0).tolist()
        attn_path = output_dir / "attention_summary.json"
        attn_path.write_text(json.dumps({"mean_attention": mean_attn}, indent=2), encoding="utf-8")
        artifacts.append(attn_path)

    if first_attn is not None:
        # first sample attention plot
        attn_vec = first_attn[0]
        artifacts.append(plot_attention_bar(attn_vec, output_dir / "attention_bar.png"))
        if first_slices is not None:
            topk = np.argsort(-attn_vec)[: min(5, len(attn_vec))].tolist()
            artifacts.append(plot_topk_slices_grid(first_slices[0], topk, output_dir / "topk_slices.png"))

    return metrics, artifacts


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

"""Plotting utilities (matplotlib Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import pr_curve, roc_curve


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> Path:
    fpr, tpr = roc_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> Path:
    recall, precision = pr_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reliability(diagram: Dict[str, List[float]], path: Path) -> Path:
    bins = diagram["bin_edges"]
    acc = diagram["bin_acc"]
    conf = diagram["bin_conf"]
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(acc))]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.bar(centers, acc, width=(bins[1] - bins[0]) * 0.9, alpha=0.6, label="Accuracy")
    ax.plot(centers, conf, marker="o", label="Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_curves(history: Sequence[Dict[str, float]], keys: Iterable[str], path: Path) -> Path:
    fig, ax = plt.subplots()
    x = list(range(len(history)))
    for key in keys:
        y = [h.get(key) for h in history]
        if all(v is None for v in y):
            continue
        ax.plot(x, y, label=key)
    ax.set_xlabel("Round/Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_attention_bar(attn: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots()
    ax.bar(range(len(attn)), attn)
    ax.set_xlabel("Slice")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Attention Weights")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_topk_slices_grid(slices: np.ndarray, topk: Sequence[int], path: Path) -> Path:
    # slices: [S, C, H, W]
    if slices.ndim != 4:
        raise ValueError("slices must be [S, C, H, W]")
    k = len(topk)
    fig, axes = plt.subplots(1, k, figsize=(k * 2.5, 2.5))
    if k == 1:
        axes = [axes]
    for ax, idx in zip(axes, topk):
        img = slices[idx]
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.axis("off")
        ax.set_title(f"Slice {idx}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

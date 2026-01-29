"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    positives = y_true.sum()
    negatives = len(y_true) - positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / max(positives, 1)
    fpr = fps / max(negatives, 1)
    return fpr, tpr


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(y_true.sum(), 1)
    return recall, precision


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    recall, precision = pr_curve(y_true, y_score)
    return float(np.trapz(precision, recall))


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, specificity_target: float
) -> float:
    fpr, tpr = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    idx = np.where(specificity >= specificity_target)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def specificity_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, sensitivity_target: float
) -> float:
    fpr, tpr = roc_curve(y_true, y_score)
    idx = np.where(tpr >= sensitivity_target)[0]
    if len(idx) == 0:
        return 0.0
    return float(1 - fpr[idx[-1]])


def confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5
) -> dict[str, int]:
    pred = (y_prob >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity_target: float = 0.95,
    sensitivity_target: float = 0.8,
) -> dict[str, float]:
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "sensitivity_at_spec": sensitivity_at_specificity(
            y_true, y_score, specificity_target
        ),
        "specificity_at_sens": specificity_at_sensitivity(
            y_true, y_score, sensitivity_target
        ),
    }

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

from src.eval.metrics import (
    auprc,
    auroc,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)


def test_metrics_vs_sklearn():
    if roc_auc_score is None:
        return
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    assert np.isclose(auroc(y_true, y_score), roc_auc_score(y_true, y_score))
    assert np.isclose(auprc(y_true, y_score), average_precision_score(y_true, y_score))

    sens = sensitivity_at_specificity(y_true, y_score, 0.5)
    spec = specificity_at_sensitivity(y_true, y_score, 0.5)
    assert 0.0 <= sens <= 1.0
    assert 0.0 <= spec <= 1.0

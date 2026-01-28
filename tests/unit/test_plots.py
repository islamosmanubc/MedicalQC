import numpy as np

from src.utils.plots import (
    plot_attention_bar,
    plot_metric_curves,
    plot_pr,
    plot_reliability,
    plot_roc,
    plot_topk_slices_grid,
)


def test_plot_functions_save(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    roc_path = plot_roc(y_true, y_score, tmp_path / "roc.png")
    pr_path = plot_pr(y_true, y_score, tmp_path / "pr.png")

    rel = {
        "bin_edges": [0.0, 0.5, 1.0],
        "bin_acc": [0.2, 0.8],
        "bin_conf": [0.3, 0.7],
        "bin_count": [5, 5],
    }
    rel_path = plot_reliability(rel, tmp_path / "rel.png")

    history = [
        {"train_loss": 1.0, "holdout_auroc": 0.5},
        {"train_loss": 0.8, "holdout_auroc": 0.6},
    ]
    curve_path = plot_metric_curves(history, ["train_loss", "holdout_auroc"], tmp_path / "curve.png")

    attn = np.array([0.1, 0.2, 0.7])
    attn_path = plot_attention_bar(attn, tmp_path / "attn.png")

    slices = np.random.rand(3, 1, 8, 8)
    grid_path = plot_topk_slices_grid(slices, [2, 1], tmp_path / "grid.png")

    for p in [roc_path, pr_path, rel_path, curve_path, attn_path, grid_path]:
        assert p.exists()

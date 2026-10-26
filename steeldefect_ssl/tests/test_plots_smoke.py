"""The nine P4 figures render, and they show what their inputs say."""

import json

import numpy as np
import pytest

from defect_clustering.clustering.outlier_detection import detect_outliers_per_cluster
from steeldefect_ssl.eval import plots

# Per-cluster schema written by defect_clustering's detect_outliers_per_cluster.
OUTLIERS = {"per_cluster": {
    "0": {"n_outliers_combined": 4, "n_outliers_if": 3, "n_outliers_lof": 2, "size": 10, "skipped": False},
    "1": {"n_outliers_combined": 1, "n_outliers_if": 1, "n_outliers_lof": 0, "size": 10, "skipped": False},
}, "total_outliers": 5}


@pytest.fixture
def figures(monkeypatch):
    """Capture every figure handed to plots._save (placeholders included)."""
    captured = []
    original = plots._save

    def spy(fig, path):
        captured.append(fig)
        return original(fig, path)

    monkeypatch.setattr(plots, "_save", spy)
    return captured


def _is_placeholder(fig):
    return any("(n/a)" in t.get_text() for ax in fig.axes for t in ax.texts)


def test_the_nine_figures_render_without_placeholders(tmp_path, figures):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.repeat(np.arange(4), 10)
    is_novel = y == 3
    scores = rng.normal(size=40) + is_novel
    calls = {
        "ssl_loss_curve": lambda p: plots.plot_ssl_loss_curve(
            [{"method": "simclr", "loss_curve": [3.0, 2.5, 2.2], "collapse_curve": [0.5, 0.6, 0.6]}], p),
        "embedding": lambda p: plots.plot_embedding_umap(X, y, ["a", "b", "c", "d"], p),
        "probe_bar": lambda p: plots.plot_probe_bar(
            {"ssl": {"linear_probe_acc": 0.8, "knn_acc": 0.75},
             "random_init": {"linear_probe_acc": 0.4, "knn_acc": 0.35}}, p),
        "clustering_uplift": lambda p: plots.plot_clustering_uplift(
            [{"repr": "ssl", "nmi": 0.6, "nmi_ci_low": 0.55, "nmi_ci_high": 0.65, "ari": 0.5},
             {"repr": "raw_pixel_pca", "nmi": 0.4, "nmi_ci_low": 0.35, "nmi_ci_high": 0.45, "ari": 0.3}], p),
        "openset_hist": lambda p: plots.plot_openset_hist(scores, is_novel, p),
        "openset_roc_pr": lambda p: plots.plot_openset_roc_pr(
            {"energy": {"scores": scores, "is_novel": is_novel, "auroc": 0.8}}, p),
        "gcd": lambda p: plots.plot_gcd_umap(X, y, is_novel, p),
        "gcd_outliers": lambda p: plots.plot_gcd_outliers(OUTLIERS, p),
        "continual": lambda p: plots.plot_continual_forgetting(
            {"acc_old_before": 0.9, "acc_old_after": 0.88, "acc_new": 0.7, "forgetting_delta": 0.02}, p),
    }
    for name, call in calls.items():
        call(str(tmp_path / f"{name}.png"))
    assert len(list(tmp_path.glob("*.png"))) == 9
    assert not [fig for fig in figures if _is_placeholder(fig)]


def test_outlier_figure_shows_the_combined_counts(tmp_path, figures):
    plots.plot_gcd_outliers(OUTLIERS, str(tmp_path / "gcd_outliers.png"))
    (fig,) = figures
    assert not _is_placeholder(fig)
    assert [p.get_height() for p in fig.axes[0].patches] == [4, 1]


def test_outlier_figure_marks_the_unscored_clusters(tmp_path, figures):
    small = {"n_outliers_combined": 0, "n_outliers_if": 0, "n_outliers_lof": 0, "size": 7, "skipped": True}
    plots.plot_gcd_outliers({"per_cluster": {**OUTLIERS["per_cluster"], "2": small}},
                            str(tmp_path / "gcd_outliers.png"))
    (fig,) = figures
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2*"]
    assert ax.get_xlabel() == "cluster (* = too small, not scored)"


def test_outlier_figure_reads_what_the_producer_writes(tmp_path, figures):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(30, 4)), rng.normal(size=(30, 4)) + 6, rng.normal(size=(5, 4)) - 6])
    labels = np.repeat([0, 1, 2], [30, 30, 5])
    _, info = detect_outliers_per_cluster(X, labels, strategy="union", min_cluster_size=10, random_state=0)
    info = json.loads(json.dumps(info))   # render_plots reads it back from the metrics JSON
    plots.plot_gcd_outliers(info, str(tmp_path / "gcd_outliers.png"))
    (fig,) = figures
    ax = fig.axes[0]
    assert [p.get_height() for p in ax.patches] == [info["per_cluster"][k]["n_outliers_combined"]
                                                    for k in ("0", "1", "2")]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2*"]


def test_a_missing_outlier_count_draws_the_placeholder(tmp_path, figures):
    plots.plot_gcd_outliers({"per_cluster": {"0": {"size": 10, "skipped": False}}},
                            str(tmp_path / "gcd_outliers.png"))
    (fig,) = figures
    assert _is_placeholder(fig)


def test_embedding_figure_names_its_projection(tmp_path, figures):
    X = np.random.default_rng(1).normal(size=(20, 5))
    plots.plot_embedding_umap(X, np.repeat([0, 1], 10), None, str(tmp_path / "embedding.png"))
    (fig,) = figures
    assert fig.axes[0].get_title() == "SSL embedding (PCA)"

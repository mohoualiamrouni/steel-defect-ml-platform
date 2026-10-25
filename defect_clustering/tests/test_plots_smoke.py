"""Every public plotting function of defect_clustering renders a figure.

Headless (Agg) smoke tests on tiny synthetic inputs. They check that a file is
written, which is enough to catch API breaks such as matplotlib removing
``cm.get_cmap``: no test called these functions before, so CI stayed green
while ten plotting functions of the two clustering packages raised
AttributeError.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from defect_clustering.clustering import advanced_viz as viz
from defect_clustering.clustering import metrics, outlier_detection

N, D, K = 60, 6, 3


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(K), N // K)
    features = rng.normal(size=(N, D)) + labels[:, None] * 3.0
    groups = np.array(["scratch", "pit", "scale"])
    metadata = pd.DataFrame({
        "expert_group": groups[(labels + (rng.random(N) < 0.2)) % K],
        "timestamp": pd.date_range("2026-01-01", periods=N, freq="h"),
    })
    return features, labels, metadata


# plot_radar_chart and plot_feature_violin are aliases of the first two below.
FEATURE_PLOTS = [viz.plot_feature_discriminability, viz.plot_silhouette_distribution,
                 viz.plot_parallel_coordinates, viz.plot_cluster_distance_matrix]


@pytest.mark.parametrize("plot", FEATURE_PLOTS, ids=lambda f: f.__name__)
def test_feature_plots_render(plot, data, tmp_path):
    _, labels, _ = data
    # Group-based plots slice columns by advanced_viz._GROUPS (960 CNN + 48 morphology).
    width = viz._GROUPS[-1][1].stop
    features = np.random.default_rng(2).normal(size=(N, width)) + labels[:, None]
    out = tmp_path / "plot.png"
    plot(features, labels, str(out))
    assert out.stat().st_size > 0


def test_outlier_summary_renders(data, tmp_path):
    features, labels, _ = data
    mask, info = outlier_detection.detect_outliers_per_cluster(
        features, labels, contamination=0.1, min_cluster_size=10, random_state=0)
    out = tmp_path / "outliers.png"
    outlier_detection.plot_outlier_summary(features, labels, mask, info, str(out))
    assert out.stat().st_size > 0


def test_confusion_matrix_renders(data, tmp_path):
    _, labels, metadata = data
    out = tmp_path / "confusion.png"
    viz.plot_confusion_matrix(labels, metadata, str(out))
    assert out.stat().st_size > 0


def test_temporal_evolution_renders(data, tmp_path):
    _, labels, metadata = data
    out = tmp_path / "temporal.png"
    viz.plot_temporal_evolution(metadata, labels, str(out), n_bins=5)
    assert out.stat().st_size > 0


def test_alluvial_diagram_renders(data, tmp_path):
    _, labels, metadata = data
    out = tmp_path / "alluvial.png"
    viz.plot_alluvial_diagram(metadata, labels, output_path=str(out))
    assert out.stat().st_size > 0


def test_gap_statistic_renders(data, tmp_path):
    features, _, _ = data
    gap = metrics.gap_statistic(features, k_range=(2, 4), n_references=3, random_state=0, n_jobs=1)
    out = tmp_path / "gap.png"
    viz.plot_gap_statistic(gap, str(out))
    assert out.stat().st_size > 0


def test_elbow_renders(data, tmp_path):
    features, _, _ = data
    inertia = metrics.inertia_curve(features, k_range=(2, 4), random_state=0, n_jobs=1)
    out = tmp_path / "elbow.png"
    metrics.plot_elbow(inertia, str(out), highlight_k=3)
    assert out.stat().st_size > 0


def test_dendrogram_renders(data, tmp_path):
    features, _, _ = data
    out = tmp_path / "dendrogram.png"
    metrics.plot_dendrogram(features, str(out), n_clusters=K)
    assert out.stat().st_size > 0


def test_2d_projection_renders(data, tmp_path):
    features, labels, _ = data
    out = tmp_path / "projection.png"
    metrics.plot_2d_projection(features[:, :2], labels, ground_truth=labels, output_path=str(out))
    assert out.stat().st_size > 0


def test_algorithm_comparison_renders(data, tmp_path):
    features, _, _ = data
    comparison = metrics.compare_algorithms(features, k_range=(3, 4), random_state=0, n_jobs=1)
    out = tmp_path / "comparison.png"
    metrics.plot_algorithm_comparison(comparison, str(out), highlight_k=3)
    assert out.stat().st_size > 0


def test_umap_renders(data, tmp_path):
    features, labels, _ = data
    out = tmp_path / "umap.png"
    metrics.plot_umap(features, labels, str(out), n_neighbors=5, random_state=0)
    assert out.stat().st_size > 0


def test_defect_masks_grid_renders(data, tmp_path):
    _, labels, _ = data
    masks = (np.random.default_rng(1).random((N, 16, 16)) > 0.7).astype(np.uint8)
    out = tmp_path / "masks.png"
    metrics.plot_defect_masks_grid(masks, labels, n_per_cluster=2, output_path=str(out))
    assert out.stat().st_size > 0

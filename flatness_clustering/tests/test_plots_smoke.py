"""Every public plotting function of flatness_clustering renders a figure.

Headless (Agg) smoke tests on tiny synthetic inputs, including the PDF
clustering report on features extracted from generated profiles. They check
that a file is written, which is enough to catch API breaks such as matplotlib
removing ``cm.get_cmap``: no test called these functions before, so CI stayed
green while the training CLI failed.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from flatness_clustering.clustering import advanced_viz as viz
from flatness_clustering.clustering import metrics
from flatness_clustering.clustering.coil_clusterer import CoilClusterer
from flatness_clustering.clustering.report import generate_clustering_report
from flatness_clustering.data.generate_flatness_data import GENERATORS
from flatness_clustering.features.statistical_features import extract_batch

N, D, K = 48, 8, 4
PATTERNS = ["center_buckle", "edge_wave", "flat", "crossbow"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(K), N // K)
    features = rng.normal(size=(N, D)) + labels[:, None] * 3.0
    profiles = rng.normal(size=(N, 12, 40)) + labels[:, None, None]
    metadata = pd.DataFrame({
        "coil_id": [f"C{i:04d}" for i in range(N)],
        "pattern": np.array(PATTERNS)[labels],
        "class_id": labels,
        "grade": np.array(["304", "430"])[labels % 2],
        "customer": "customer_a",
        "timestamp": pd.date_range("2026-01-01", periods=N, freq="30min"),
    })
    return features, labels, metadata, profiles


# plot_radar_chart and plot_feature_violin are aliases of the first two below.
FEATURE_PLOTS = [viz.plot_feature_discriminability, viz.plot_silhouette_distribution,
                 viz.plot_parallel_coordinates, viz.plot_cluster_distance_matrix]


@pytest.mark.parametrize("plot", FEATURE_PLOTS, ids=lambda f: f.__name__)
def test_feature_plots_render(plot, real_coils, tmp_path):
    # Group-based plots need the extractor's 1105-column layout.
    features, _, metadata = real_coils
    out = tmp_path / "plot.png"
    plot(features, metadata["class_id"].to_numpy(), str(out))
    assert out.stat().st_size > 0


def test_confusion_matrix_renders(data, tmp_path):
    _, labels, metadata, _ = data
    out = tmp_path / "confusion.png"
    viz.plot_confusion_matrix(labels, metadata, str(out))
    assert out.stat().st_size > 0


def test_temporal_evolution_renders(data, tmp_path):
    _, labels, metadata, _ = data
    out = tmp_path / "temporal.png"
    viz.plot_temporal_evolution(metadata, labels, str(out), n_bins=5)
    assert out.stat().st_size > 0


def test_gap_statistic_renders(data, tmp_path):
    features, _, _, _ = data
    gap = metrics.gap_statistic(features, k_range=(2, 5), n_references=3, random_state=0)
    out = tmp_path / "gap.png"
    viz.plot_gap_statistic(gap, str(out))
    assert out.stat().st_size > 0


def test_elbow_renders(data, tmp_path):
    features, _, _, _ = data
    inertia = metrics.inertia_curve(features, k_range=(2, 5), random_state=0)
    out = tmp_path / "elbow.png"
    metrics.plot_elbow(inertia, str(out), highlight_k=K)
    assert out.stat().st_size > 0


def test_dendrogram_renders(data, tmp_path):
    features, _, _, _ = data
    out = tmp_path / "dendrogram.png"
    metrics.plot_dendrogram(features, str(out), n_clusters=K)
    assert out.stat().st_size > 0


def test_cluster_heatmaps_render(data, tmp_path):
    _, labels, _, profiles = data
    out = tmp_path / "heatmaps.png"
    metrics.plot_cluster_heatmaps(profiles, labels, n_samples=2, output_path=str(out),
                                  pattern_names=PATTERNS)
    assert out.stat().st_size > 0


def test_tsne_renders(data, tmp_path):
    features, labels, _, _ = data
    out = tmp_path / "tsne.png"
    metrics.plot_tsne(features[:, :2], labels, ground_truth=labels, output_path=str(out))
    assert out.stat().st_size > 0


def test_algorithm_comparison_renders(data, tmp_path):
    features, _, _, _ = data
    comparison = metrics.compare_algorithms(features, k_range=(3, 5), random_state=0)
    out = tmp_path / "comparison.png"
    metrics.plot_algorithm_comparison(comparison, str(out), highlight_k=K)
    assert out.stat().st_size > 0


def test_umap_renders(data, tmp_path):
    features, labels, _, _ = data
    out = tmp_path / "umap.png"
    metrics.plot_umap(features, labels, str(out), n_neighbors=5, random_state=0)
    assert out.stat().st_size > 0


@pytest.fixture(scope="module")
def real_coils():
    """Generated profiles of 4 pattern classes and their real 1105-column features."""
    rng = np.random.default_rng(0)
    names = list(GENERATORS)[:K]
    profiles = np.stack([GENERATORS[n](rng, width=40, length=300) for n in names for _ in range(6)])
    labels = np.repeat(np.arange(K), 6)
    metadata = pd.DataFrame({"pattern": np.array(names)[labels], "class_id": labels,
                             "grade": np.array(["304", "430"])[labels % 2]})
    return extract_batch(profiles), profiles, metadata


def test_clustering_report_renders(real_coils, tmp_path):
    features, profiles, metadata = real_coils
    clusterer = CoilClusterer(method="kmeans", n_clusters=K, random_state=0).fit(features)
    out = tmp_path / "report.pdf"
    generate_clustering_report(
        features, clusterer.cluster_labels_, metadata, output_path=str(out), profiles=profiles,
        clusterer=clusterer,
        inertia_data=metrics.inertia_curve(features, k_range=(2, 5), random_state=0),
        tsne_2d=features[:, :2], n_clusters=K, run_tag="smoke",
        gap_result=metrics.gap_statistic(features, k_range=(2, 5), n_references=3, random_state=0),
    )
    assert out.stat().st_size > 0

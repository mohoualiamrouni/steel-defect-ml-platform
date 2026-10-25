"""
Clustering Quality Metrics and Visualisations
==============================================

Functions for evaluating and visualising flatness profile clustering results.

Usage
-----
    from flatness_clustering.clustering.metrics import (
        compute_silhouette,
        inertia_curve,
        plot_elbow,
        cluster_summary,
        plot_cluster_heatmaps,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score as _sk_silhouette,
    davies_bouldin_score as _sk_davies_bouldin,
    calinski_harabasz_score as _sk_calinski_harabasz,
)


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score for a clustering result.  Range [-1, 1]; higher is better.

    Args:
        features: (N, F) float32 feature matrix.
        labels:   (N,) cluster label array.  Points with label -1 (noise) are excluded.

    Returns:
        Silhouette score (float).  Returns 0.0 if fewer than 2 clusters.
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return 0.0
    return float(_sk_silhouette(features[mask], labels[mask]))


def compute_davies_bouldin(features: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin Index for a clustering result.  Lower is better.

    Measures the average similarity between each cluster and its most similar
    neighbour (similarity = ratio of within-cluster scatter to centroid distance).

    Reference: Saas et al. (2017) — DB index used alongside silhouette for
    flatness defect cluster evaluation.

    Args:
        features: (N, F) feature matrix.
        labels:   (N,) cluster label array.  Points with label -1 (noise) excluded.

    Returns:
        Davies-Bouldin score (float).  Returns 0.0 if fewer than 2 clusters.
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return 0.0
    return float(_sk_davies_bouldin(features[mask], labels[mask]))


def compute_calinski_harabasz(features: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabász Score (Variance Ratio Criterion).  Higher is better.

    Ratio of between-cluster dispersion to within-cluster dispersion.
    Tends to favour convex, well-separated clusters.

    Reference: Saas et al. (2017) — CH score alongside silhouette and DB
    for flatness cluster selection.

    Args:
        features: (N, F) feature matrix.
        labels:   (N,) cluster label array.  Points with label -1 (noise) excluded.

    Returns:
        Calinski-Harabász score (float).  Returns 0.0 if fewer than 2 clusters.
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return 0.0
    return float(_sk_calinski_harabasz(features[mask], labels[mask]))


def inertia_curve(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 12),
    random_state: int = 42,
) -> Dict[int, Dict[str, float]]:
    """Compute KMeans inertia + three cluster quality metrics for a range of k.

    Implements the multi-metric k selection from Saas et al. (2017):
    evaluate inertia, silhouette (higher better), Davies-Bouldin (lower better),
    and Calinski-Harabász (higher better) across k to find the best k.

    Args:
        features:     (N, F) feature matrix.
        k_range:      (k_min, k_max) inclusive range.
        random_state: Random seed.

    Returns:
        Dict {k: {"inertia": float, "silhouette": float, "db": float, "ch": float}}
        for k in range(k_min, k_max + 1).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(features.astype(np.float64))
    result = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=200)
        labels = km.fit_predict(X)
        result[k] = {
            "inertia":   float(km.inertia_),
            "silhouette": compute_silhouette(X, labels),
            "db":         compute_davies_bouldin(X, labels),
            "ch":         compute_calinski_harabasz(X, labels),
        }
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_elbow(
    inertia_dict: Dict[int, Dict],
    output_path: Optional[str] = None,
    highlight_k: Optional[int] = None,
) -> None:
    """Plot KMeans inertia vs. k with silhouette overlay (Saas et al. 2017).

    Accepts either the legacy {k: float} format or the new
    {k: {"inertia": ..., "silhouette": ..., "db": ..., "ch": ...}} format.

    When the new format is used, shows silhouette score as a twin y-axis
    overlay so both the elbow and the quality peak are visible together.

    Args:
        inertia_dict: {k: float} or {k: dict} from inertia_curve().
        output_path:  Save path (.png).  If None, displays interactively.
        highlight_k:  If provided, marks the chosen k with a vertical line.
    """
    ks = sorted(inertia_dict.keys())
    # Support both legacy (float values) and new (dict values) format
    first_val = inertia_dict[ks[0]]
    new_format = isinstance(first_val, dict)

    inertia_vals = [
        inertia_dict[k]["inertia"] if new_format else inertia_dict[k]
        for k in ks
    ]
    sil_vals = [inertia_dict[k].get("silhouette", None) for k in ks] if new_format else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, inertia_vals, "o-", color="#2196F3", linewidth=2, markersize=7,
            label="Inertia")

    if sil_vals and any(v is not None for v in sil_vals):
        ax2 = ax.twinx()
        ax2.plot(ks, sil_vals, "s--", color="#FF5722", linewidth=1.5, markersize=6,
                 label="Silhouette")
        ax2.set_ylabel("Silhouette score (higher better)", color="#FF5722", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#FF5722")
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if highlight_k is not None and highlight_k in inertia_dict:
        ax.axvline(highlight_k, color="#E91E63", linestyle="--",
                   linewidth=1.5, alpha=0.8)
        ax.text(highlight_k, max(inertia_vals) * 0.95, f"  k={highlight_k}",
                color="#E91E63", fontsize=9)

    ax.set_xlabel("Number of clusters k", fontsize=12)
    ax.set_ylabel("KMeans inertia", fontsize=12)
    ax.set_title("Elbow Curve — Flatness Profile Clustering\n"
                 "(Saas et al. 2017: inertia + silhouette overlay)", fontsize=11)
    ax.set_xticks(ks)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_dendrogram(
    features_scaled: "np.ndarray",
    output_path: str,
    n_clusters: int = 5,
    max_display: int = 40,
) -> str:
    """Ward hierarchical clustering dendrogram with optimal flat-cut marker.

    Theoretical basis:
      - Saas et al. 2017 (1710.02268): Ward linkage minimises within-cluster
        variance and is recommended for dense, synchronised steel flatness profiles.
      - NeurIPS 2019 "Flattening a Hierarchical Clustering through Active Learning"
        (Vikram & Dasgupta): The dendrogram encodes all possible flat partitions;
        the horizontal cut at height Z[-(k-1), 2] is the flat partition into k
        clusters with maximum within-cluster homogeneity.

    The red dashed line marks the chosen cut (n_clusters). The gap between
    successive merge heights (largest gap = best k) is annotated.

    Args:
        features_scaled: (N, F) StandardScaler-normalised feature matrix.
        output_path:     Where to save the PNG.
        n_clusters:      Number of clusters — sets the cut-line position.
        max_display:     Truncate dendrogram to last `max_display` merges for readability.

    Returns:
        output_path string.
    """
    from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage
    from pathlib import Path as _Path

    Z = linkage(features_scaled.astype(np.float64), method="ward", metric="euclidean")

    # Identify the cut height for k clusters
    # Z[-(k-1), 2] is the merge distance at which we go from k+1 → k clusters
    cut_height = float(Z[-(n_clusters - 1), 2])

    fig, ax = plt.subplots(figsize=(12, 5))

    _dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=max_display,
        leaf_rotation=90.0,
        leaf_font_size=7,
        show_contracted=True,
        color_threshold=cut_height,
    )

    ax.axhline(
        y=cut_height, color="crimson", linestyle="--", lw=1.8,
        label=f"Flat cut → k = {n_clusters}  (height = {cut_height:.2f})",
    )
    ax.set_xlabel("Coil cluster (truncated to last merges)", fontsize=9)
    ax.set_ylabel("Ward linkage distance", fontsize=9)
    ax.set_title(
        "Ward Hierarchical Dendrogram — Flatness Profile Clustering\n"
        "Saas et al. 2017 + NeurIPS 2019 flat-cut criterion  "
        f"·  k = {n_clusters} selected",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] Dendrogram saved: {output_path}")
    return output_path


def cluster_summary(
    features: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-cluster summary DataFrame.

    Args:
        features:    (N, F) feature matrix.
        labels:      (N,) cluster label array.
        metadata_df: DataFrame with columns: coil_id, pattern, grade, customer.

    Returns:
        DataFrame with one row per cluster:
            cluster_id, size, dominant_pattern, pattern_entropy,
            top_grade, top_customer, mean_feature_norm
    """
    records = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        subset_meta = metadata_df[mask]
        subset_feat = features[mask]

        dominant_pattern = subset_meta["pattern"].value_counts().idxmax() \
            if len(subset_meta) > 0 else "unknown"

        # Shannon entropy of pattern distribution
        counts = subset_meta["pattern"].value_counts(normalize=True)
        pat_entropy = abs(float(-(counts * np.log(counts + 1e-9)).sum()))

        top_grade    = subset_meta["grade"].value_counts().idxmax() \
            if "grade" in subset_meta.columns and len(subset_meta) > 0 else "unknown"
        top_customer = subset_meta["customer"].value_counts().idxmax() \
            if "customer" in subset_meta.columns and len(subset_meta) > 0 else "unknown"

        records.append({
            "cluster_id":             int(cluster_id),
            "size":                   int(mask.sum()),
            "dominant_pattern":       dominant_pattern,
            "pattern_entropy":        round(pat_entropy, 3),
            "top_grade":              top_grade,
            "top_customer":           top_customer,
            "mean_norm_scaled":       round(float(np.linalg.norm(subset_feat, axis=1).mean()), 4),
        })

    return pd.DataFrame(records)


def plot_cluster_heatmaps(
    profiles: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 3,
    output_path: Optional[str] = None,
    pattern_names: Optional[List[str]] = None,
) -> None:
    """Show a grid of representative flatness profiles per cluster.

    Args:
        profiles:      (N, W, L) float32 flatness profiles.
        labels:        (N,) cluster label array.
        n_samples:     Number of sample profiles to show per cluster.
        output_path:   Save path (.png).  If None, displays interactively.
        pattern_names: Optional list to label each cluster row.
    """
    unique_clusters = sorted(c for c in set(labels) if c >= 0)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(
        n_clusters, n_samples,
        figsize=(n_samples * 4, n_clusters * 2.5),
        squeeze=False,
    )

    for row, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.where(labels == cluster_id)[0]
        sample_idx = cluster_mask[:n_samples]

        # Global color scale for this cluster
        cluster_profiles = profiles[cluster_mask]
        vmin = float(np.percentile(cluster_profiles, 2))
        vmax = float(np.percentile(cluster_profiles, 98))

        for col in range(n_samples):
            ax = axes[row][col]
            if col < len(sample_idx):
                profile = profiles[sample_idx[col]]
                # Show a downsampled version for speed (max 100 x 200 display)
                display = profile[::max(1, profile.shape[0]//100),
                                  ::max(1, profile.shape[1]//200)]
                im = ax.imshow(display, aspect="auto", cmap="RdBu_r",
                               vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    label = pattern_names[cluster_id] if pattern_names else f"Cluster {cluster_id}"
                    ax.set_ylabel(label, fontsize=9, labelpad=4)
            else:
                ax.set_visible(False)

    plt.suptitle("Representative Flatness Profiles per Cluster\n"
                 "(red = positive I-units / blue = negative)", fontsize=11)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=80)
        plt.close(fig)
    else:
        plt.show()


def plot_tsne(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE of Flatness Profiles",
    ground_truth: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> None:
    """Scatter plot of 2D t-SNE / UMAP embeddings coloured by cluster label.

    Args:
        embeddings_2d: (N, 2) array from TSNE/UMAP transform.
        labels:        (N,) cluster label array.
        title:         Plot title.
        ground_truth:  (N,) true pattern IDs — if provided, shows side-by-side.
        output_path:   Save path.
    """
    ncols = 2 if ground_truth is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 7, 6))
    if ncols == 1:
        axes = [axes]

    for ax, (lbl, title_suffix) in zip(
        axes,
        [(labels, "Cluster labels"), (ground_truth, "Ground truth patterns")]
        if ground_truth is not None
        else [(labels, "")],
    ):
        unique_lbl = sorted(set(lbl))
        n_colors = max(len(unique_lbl), 6)
        cmap = matplotlib.colormaps["tab10" if n_colors <= 10 else "tab20"].resampled(n_colors)
        for i, uid in enumerate(unique_lbl):
            mask = lbl == uid
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[cmap(i / n_colors)], s=40, alpha=0.80,
                linewidths=0.3, edgecolors="white",
                label=str(uid),
            )
        ax.set_title(f"{title}\n{title_suffix}", fontsize=10)
        ax.set_xlabel("t-SNE dim 1", fontsize=10)
        ax.set_ylabel("t-SNE dim 2", fontsize=10)
        ax.legend(title="Label", fontsize=7, markerscale=1.5,
                  bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Multi-algorithm comparison  (Saas 2017, Ren 2021, Asadi 2021)
# ---------------------------------------------------------------------------

def compare_algorithms(
    features: np.ndarray,
    k_range: Tuple[int, int] = (3, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit KMeans, GMM, Agglomerative (Ward), and MiniBatchKMeans at each k.

    Computes three complementary cluster quality metrics for each (method, k):
      - Silhouette     : [-1, 1], higher is better
      - Davies-Bouldin : [0, ∞), lower is better
      - Calinski-Harabász: [0, ∞), higher is better

    Spectral is excluded as it is O(n²) in memory and too slow on large datasets.

    Motivated by Saas et al. 2017 (1710.02268) for time-series clustering
    algorithm selection, and Ren et al. 2021 (DeepAL Survey) for GMM.

    Args:
        features:     (N, F) float32 feature matrix (unscaled).
        k_range:      (k_min, k_max) inclusive range of cluster counts to test.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: method, k, silhouette, davies_bouldin, calinski_harabasz.
        Rows are sorted by (method, k).
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(features.astype(np.float64))
    k_min, k_max = k_range
    rows = []

    methods = {
        "kmeans": lambda k: KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X),
        "gmm": lambda k: GaussianMixture(n_components=k, random_state=random_state, n_init=3).fit(X).predict(X),
        "agglomerative": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
        "minibatch_kmeans": lambda k: MiniBatchKMeans(
            n_clusters=k, random_state=random_state, batch_size=min(1024, len(X)), n_init=3
        ).fit_predict(X),
    }

    for method_name, fit_fn in methods.items():
        for k in range(k_min, k_max + 1):
            try:
                labels = fit_fn(k)
                sil = compute_silhouette(X, labels)
                db  = compute_davies_bouldin(X, labels)
                ch  = compute_calinski_harabasz(X, labels)
            except Exception as exc:
                print(f"[compare_algorithms] {method_name} k={k} failed: {exc}")
                sil, db, ch = 0.0, 0.0, 0.0
            rows.append({
                "method": method_name,
                "k": k,
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 1),
            })

    return pd.DataFrame(rows).sort_values(["method", "k"]).reset_index(drop=True)


def plot_algorithm_comparison(
    comparison_df: pd.DataFrame,
    output_path: str,
    highlight_k: int = 6,
) -> None:
    """Plot silhouette / Davies-Bouldin / Calinski-Harabász vs k, one line per algorithm.

    Produces a 3-row figure showing how each metric evolves with cluster count for
    each algorithm. A vertical dashed line marks the chosen k (highlight_k).

    Motivated by Saas et al. 2017 — multi-metric elbow analysis for algorithm selection.

    Args:
        comparison_df: Output of compare_algorithms().
        output_path:   Path to save the PNG.
        highlight_k:   k value to mark with a vertical dashed line.
    """
    methods = comparison_df["method"].unique()
    # Distinct markers per algorithm for print readability
    _markers = ["o", "s", "D", "^", "v", "P", "X"]
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))
    method_style = {
        m: {"color": c, "marker": _markers[i % len(_markers)]}
        for i, (m, c) in enumerate(zip(methods, colors))
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(hspace=0.45, top=0.90, bottom=0.10)
    fig.suptitle(
        "Clustering Algorithm Comparison\n"
        "KMeans / GMM / Agglomerative (Ward) / MiniBatchKMeans",
        fontsize=13, fontweight="bold",
    )

    metrics = [
        ("silhouette",        "Silhouette Score",        True,  "Higher is better"),
        ("davies_bouldin",    "Davies-Bouldin Index",    False, "Lower is better"),
        ("calinski_harabasz", "Calinski-Harabasz Score", True,  "Higher is better"),
    ]

    for ax, (col, ylabel, higher_better, note) in zip(axes, metrics):
        for method in methods:
            sub = comparison_df[comparison_df["method"] == method].sort_values("k")
            sty = method_style[method]
            ax.plot(sub["k"], sub[col], marker=sty["marker"], ms=7, lw=2.2,
                    color=sty["color"], label=method)
        ax.axvline(highlight_k, color="#E91E63", linestyle="--", lw=1.5, alpha=0.7,
                   label=f"k = {highlight_k}")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}  ({note})", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    # Single shared legend at top of figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods) + 1,
              fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.5, 0.955))

    axes[-1].set_xlabel("Number of clusters (k)", fontsize=11)
    k_vals = sorted(comparison_df["k"].unique())
    axes[-1].set_xticks(k_vals)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] Algorithm comparison saved: {output_path}")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for silhouette score  (McInnes 2018 style)
# ---------------------------------------------------------------------------

def bootstrap_silhouette(
    features: "np.ndarray",
    labels: "np.ndarray",
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> dict:
    """Bootstrap confidence interval for silhouette score.

    Repeatedly subsample the data and compute silhouette on each subsample.
    Returns mean, std, and 95% CI, giving a rigorous uncertainty estimate
    for the clustering quality metric.

    This matters because silhouette on a single sample can vary substantially
    with dataset size and cluster balance. Bootstrap CIs enable comparison
    between clustering methods with statistical confidence.

    Args:
        features:     (N, F) feature matrix.
        labels:       (N,) cluster label array.
        n_bootstrap:  Number of bootstrap resamples (default 100).
        sample_frac:  Fraction of data per resample (default 0.8 = 80%).
        random_state: RNG seed for reproducibility.

    Returns:
        Dict with keys: mean, std, ci_low, ci_high, scores_list.
        ci_low/ci_high are the 2.5th and 97.5th percentile (95% CI).
    """
    rng = np.random.default_rng(random_state)
    N = len(features)
    k = max(2, int(N * sample_frac))
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=k, replace=False)
        sub_labels = labels[idx]
        # Need at least 2 unique labels in subsample
        if len(set(sub_labels[sub_labels >= 0])) < 2:
            continue
        try:
            s = float(_sk_silhouette(features[idx][sub_labels >= 0],
                                     sub_labels[sub_labels >= 0]))
            scores.append(s)
        except Exception:
            pass

    if len(scores) < 3:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "scores_list": scores}

    arr = np.array(scores)
    ci = np.percentile(arr, [2.5, 97.5])
    return {
        "mean":        float(arr.mean()),
        "std":         float(arr.std()),
        "ci_low":      float(ci[0]),
        "ci_high":     float(ci[1]),
        "scores_list": scores,
    }


# ---------------------------------------------------------------------------
# UMAP visualisation  (McInnes et al. 2018)
# ---------------------------------------------------------------------------

def plot_umap(
    features: "np.ndarray",
    labels: "np.ndarray",
    output_path: str = "outputs/plots/umap_clusters.png",
    title: str = "UMAP Cluster Projection",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    label_names: "Optional[Dict[int, str]]" = None,
) -> str:
    """UMAP 2-D projection of cluster embeddings (McInnes et al. 2018).

    UMAP (Uniform Manifold Approximation and Projection) preserves global
    structure better than t-SNE and is faster for large datasets. It is
    recommended for production monitoring dashboards where you need to see
    how clusters evolve over time.

    Compared to t-SNE:
      - UMAP preserves inter-cluster distances (global structure)
      - t-SNE only preserves local neighbourhood structure
      - UMAP is 5-10x faster for N > 5000

    Reference: McInnes et al. (2018) "UMAP: Uniform Manifold Approximation
    and Projection for Dimension Reduction". arXiv:1802.03426.

    Args:
        features:     (N, F) feature matrix.
        labels:       (N,) cluster label array.
        output_path:  Path to save the PNG.
        title:        Plot title.
        n_neighbors:  UMAP neighbours parameter (default 15).
                      Smaller = more local structure preserved.
        min_dist:     UMAP minimum embedding distance (default 0.1).
                      Smaller = tighter clusters in projection.
        metric:       Distance metric (default 'euclidean').
        random_state: RNG seed.
        label_names:  Optional {label_id: name} dict for legend.

    Returns:
        Path to the saved PNG.
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "UMAP visualisation requires the umap-learn package: "
            "pip install umap-learn"
        )

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(features.astype(np.float32))

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    n_colors = max(10, len(unique_labels))
    colors = [cmap(i / n_colors) for i in range(len(unique_labels))]

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = (label_names or {}).get(lbl, f"Cluster {lbl}" if lbl >= 0 else "Noise")
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[idx]], label=name,
            alpha=0.6, s=12, edgecolors="none",
        )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(f"{title}\n"
                 f"n_neighbors={n_neighbors}  min_dist={min_dist}  metric={metric}",
                 fontsize=11)
    ax.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] UMAP projection saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Gap Statistic (Tibshirani, Walther & Hastie 2001)
# ---------------------------------------------------------------------------

def gap_statistic(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 12),
    n_references: int = 20,
    random_state: int = 42,
) -> Dict:
    """Gap Statistic for principled k selection (Tibshirani et al. 2001).

    Gap(k) = E*[log(W_k)] - log(W_k)
    where W_k = within-cluster dispersion (inertia), E* = expectation
    under a uniform reference distribution (SVD-aligned bounding box).

    Optimal k = smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    where s_k is the simulation standard error.

    Args:
        features:      (N, F) feature matrix (should be scaled).
        k_range:       (k_min, k_max) inclusive range.
        n_references:  Number of reference datasets (paper recommends 10-20).
        random_state:  Seed for reproducibility.

    Returns:
        Dict with keys:
            'gap_values':       list of Gap(k) for each k
            'sk_values':        list of s_k (standard error) for each k
            'optimal_k':        int — smallest k satisfying gap criterion
            'log_wk':           list of log(W_k) (observed)
            'expected_log_wk':  list of E*[log(W_k)] (reference mean)
            'k_range_tested':   list of k values tested
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(random_state)
    N, F = features.shape
    ks = list(range(k_range[0], k_range[1] + 1))

    # SVD-aligned bounding box for reference generation (Tibshirani 2001 Sec 4)
    # SVD-aligned bounding box (Tibshirani 2001, Section 3)
    # With N < F, SVD gives Vt of shape (N, F), so rotated is (N, N).
    # Generate reference data in the rotated space, then rotate back.
    _, _, Vt = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
    D = Vt.shape[0]  # min(N, F) — rank of the rotated space
    rotated = features @ Vt.T  # (N, D)
    col_min = rotated.min(axis=0)  # (D,)
    col_max = rotated.max(axis=0)  # (D,)

    log_wk = []
    expected_log_wk = []
    sk_values = []

    for k in ks:
        # Observed W_k
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
        km.fit(features)
        wk = km.inertia_
        log_wk.append(np.log(wk + 1e-300))

        # Reference W_k* (B reference datasets)
        ref_log_wks = []
        for b in range(n_references):
            ref_rotated = rng.uniform(col_min, col_max, size=(N, D))
            ref_data = ref_rotated @ Vt  # rotate back to original space (N, F)
            km_ref = KMeans(n_clusters=k, n_init=5, max_iter=200,
                            random_state=random_state + b)
            km_ref.fit(ref_data)
            ref_log_wks.append(np.log(km_ref.inertia_ + 1e-300))

        ref_log_wks = np.array(ref_log_wks)
        expected_log_wk.append(float(ref_log_wks.mean()))
        sk_values.append(float(ref_log_wks.std() * np.sqrt(1 + 1.0 / n_references)))

    gap_values = [e - o for e, o in zip(expected_log_wk, log_wk)]

    # Optimal k: smallest k where Gap(k) >= Gap(k+1) - s_{k+1}
    optimal_k = ks[-1]  # fallback
    for i in range(len(ks) - 1):
        if gap_values[i] >= gap_values[i + 1] - sk_values[i + 1]:
            optimal_k = ks[i]
            break

    print(f"[metrics] Gap Statistic (Tibshirani 2001): optimal k = {optimal_k}")
    return {
        "gap_values": gap_values,
        "sk_values": sk_values,
        "optimal_k": optimal_k,
        "log_wk": log_wk,
        "expected_log_wk": expected_log_wk,
        "k_range_tested": ks,
    }


# ---------------------------------------------------------------------------
# Consensus Clustering
# ---------------------------------------------------------------------------

def consensus_clustering(
    features: np.ndarray,
    n_clusters: int = 5,
    algorithms: Optional[List[str]] = None,
    n_runs: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Consensus clustering via co-association matrix.

    Combines results from multiple clustering algorithms and runs to produce
    a more robust final clustering.

    Algorithm:
        1. Run each algorithm M times with different random seeds
        2. Build co-association matrix C[i,j] = fraction of runs where
           samples i and j are assigned to the same cluster
        3. Apply Agglomerative clustering on (1 - C) as a distance matrix
        4. Return consensus labels

    Args:
        features:     (N, F) feature matrix.
        n_clusters:   Target number of clusters.
        algorithms:   List of algorithms to combine.
                      Supported: "kmeans", "agglomerative", "gmm".
                      Default: ["kmeans", "agglomerative", "gmm"].
        n_runs:       Number of runs per algorithm (default 50).
        random_state: Base seed.

    Returns:
        (consensus_labels, co_association_matrix)
        - consensus_labels: shape (N,) int array
        - co_association_matrix: shape (N, N) float in [0, 1]
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture

    if algorithms is None:
        algorithms = ["kmeans", "agglomerative", "gmm"]

    N = features.shape[0]
    co_assoc = np.zeros((N, N), dtype=np.float64)
    total_runs = 0

    for algo_name in algorithms:
        for _ in range(n_runs):
            seed = random_state + total_runs
            total_runs += 1

            if algo_name == "kmeans":
                model = KMeans(n_clusters=n_clusters, n_init=3, max_iter=200,
                               random_state=seed)
                labels = model.fit_predict(features)
            elif algo_name == "agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage="ward")
                labels = model.fit_predict(features)
            elif algo_name == "gmm":
                model = GaussianMixture(n_components=n_clusters,
                                        covariance_type="full",
                                        max_iter=150, n_init=1,
                                        random_state=seed)
                labels = model.fit_predict(features)
            else:
                continue

            # Update co-association: samples with same label get +1
            for c in range(n_clusters):
                members = np.where(labels == c)[0]
                if len(members) > 1:
                    idx = np.ix_(members, members)
                    co_assoc[idx] += 1

    co_assoc /= max(total_runs, 1)

    # Final clustering on the co-association distance matrix
    distance = 1.0 - co_assoc
    np.fill_diagonal(distance, 0.0)
    final_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    consensus_labels = final_model.fit_predict(distance)

    print(f"[metrics] Consensus clustering: {len(algorithms)} algorithms × "
          f"{n_runs} runs = {total_runs} total, k={n_clusters}")
    return consensus_labels, co_assoc

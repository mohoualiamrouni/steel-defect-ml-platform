"""
Clustering Quality Metrics and Visualisations for Defect Clustering
====================================================================

Functions for evaluating and visualising defect group clustering results.

Usage
-----
    from defect_clustering.clustering.metrics import (
        compute_silhouette,
        compute_davies_bouldin,
        compute_calinski_harabasz,
        inertia_curve,
        plot_elbow,
        plot_dendrogram,
        cluster_summary,
        plot_2d_projection,
        plot_algorithm_comparison,
        bootstrap_silhouette,
        plot_umap,
        gap_statistic,
        consensus_clustering,
        plot_defect_masks_grid,
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

def compute_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 8000,
    random_state: int = 42,
) -> float:
    """Silhouette score for a clustering result.  Range [-1, 1]; higher is better.

    For large datasets (N > max_samples), a stratified subsample is used
    to avoid O(N^2) memory for the pairwise distance matrix.

    Args:
        features:    (N, F) float32 feature matrix.
        labels:      (N,) cluster label array.  Points with label -1 (noise) are excluded.
        max_samples: Subsample to this many points if N exceeds it.
        random_state: Random seed for reproducible subsampling.

    Returns:
        Silhouette score (float).  Returns 0.0 if fewer than 2 clusters.
    """
    mask = labels >= 0
    X, y = features[mask], labels[mask]
    if len(X) < 2 or len(set(y)) < 2:
        return 0.0
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        # Ensure at least 2 clusters survive subsampling
        if len(set(y)) < 2:
            return 0.0
    return float(_sk_silhouette(X, y))


def compute_davies_bouldin(features: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin Index for a clustering result.  Lower is better.

    Measures the average similarity between each cluster and its most similar
    neighbour (similarity = ratio of within-cluster scatter to centroid distance).

    Reference: Saas et al. (2017) -- DB index used alongside silhouette for
    cluster evaluation.

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
    """Calinski-Harabasz Score (Variance Ratio Criterion).  Higher is better.

    Ratio of between-cluster dispersion to within-cluster dispersion.
    Tends to favour convex, well-separated clusters.

    Reference: Saas et al. (2017) -- CH score alongside silhouette and DB
    for cluster selection.

    Args:
        features: (N, F) feature matrix.
        labels:   (N,) cluster label array.  Points with label -1 (noise) excluded.

    Returns:
        Calinski-Harabasz score (float).  Returns 0.0 if fewer than 2 clusters.
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return 0.0
    return float(_sk_calinski_harabasz(features[mask], labels[mask]))


def _inertia_for_k(k, X, random_state):
    """Fit MiniBatchKMeans for one k and return metrics (for joblib)."""
    from sklearn.cluster import MiniBatchKMeans

    N = X.shape[0]
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3,
                         max_iter=100, batch_size=min(1024, N))
    labels = km.fit_predict(X)
    # Use smaller subsample (4000) to keep per-worker memory under ~120 MiB
    return k, {
        "inertia":    float(km.inertia_),
        "silhouette": compute_silhouette(X, labels, max_samples=2000),
        "db":         compute_davies_bouldin(X, labels),
        "ch":         compute_calinski_harabasz(X, labels),
    }


def inertia_curve(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 12),
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[int, Dict[str, float]]:
    """Compute MiniBatchKMeans inertia + three cluster quality metrics for a range of k.

    Implements the multi-metric k selection from Saas et al. (2017):
    evaluate inertia, silhouette (higher better), Davies-Bouldin (lower better),
    and Calinski-Harabasz (higher better) across k to find the best k.

    Args:
        features:     (N, F) feature matrix.
        k_range:      (k_min, k_max) inclusive range.
        random_state: Random seed.
        n_jobs:       Number of parallel jobs (-1 = all cores).

    Returns:
        Dict {k: {"inertia": float, "silhouette": float, "db": float, "ch": float}}
        for k in range(k_min, k_max + 1).
    """
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed

    # Keep float32 end-to-end: each worker pickles X, so float64 doubles RAM
    # and triggers ArrayMemoryError on 20K×900-D inputs.
    X = StandardScaler().fit_transform(features).astype(np.float32, copy=False)
    ks = list(range(k_range[0], k_range[1] + 1))

    # Cap workers: each computes silhouette pairwise distances (~60 MiB each
    # with float32) plus pickles X per process.  min(4, -1) is -1, so handle
    # the -1 case explicitly.
    if n_jobs == -1 or n_jobs > 4:
        _safe_jobs = 4
    else:
        _safe_jobs = n_jobs
    results = Parallel(n_jobs=_safe_jobs, prefer="threads")(
        delayed(_inertia_for_k)(k, X, random_state) for k in ks
    )
    return {k: metrics for k, metrics in results}


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
    ax.set_title("Elbow Curve -- Defect Group Clustering\n"
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
        variance and is recommended for dense feature representations.
      - NeurIPS 2019 "Flattening a Hierarchical Clustering through Active Learning"
        (Vikram & Dasgupta): The dendrogram encodes all possible flat partitions;
        the horizontal cut at height Z[-(k-1), 2] is the flat partition into k
        clusters with maximum within-cluster homogeneity.

    The red dashed line marks the chosen cut (n_clusters). The gap between
    successive merge heights (largest gap = best k) is annotated.

    Args:
        features_scaled: (N, F) StandardScaler-normalised feature matrix.
        output_path:     Where to save the PNG.
        n_clusters:      Number of clusters -- sets the cut-line position.
        max_display:     Truncate dendrogram to last `max_display` merges for readability.

    Returns:
        output_path string.
    """
    from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage
    from pathlib import Path as _Path

    _MAX_WARD = 5000
    if features_scaled.shape[0] > _MAX_WARD:
        rng = np.random.RandomState(42)
        idx = rng.choice(features_scaled.shape[0], size=_MAX_WARD, replace=False)
        features_for_linkage = features_scaled[idx]
    else:
        features_for_linkage = features_scaled

    Z = linkage(features_for_linkage.astype(np.float64), method="ward", metric="euclidean")

    # Identify the cut height for k clusters
    # Z[-(k-1), 2] is the merge distance at which we go from k+1 -> k clusters
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
        label=f"Flat cut -> k = {n_clusters}  (height = {cut_height:.2f})",
    )
    ax.set_xlabel("Defect sample cluster (truncated to last merges)", fontsize=9)
    ax.set_ylabel("Ward linkage distance", fontsize=9)
    ax.set_title(
        "Ward Hierarchical Dendrogram -- Defect Group Clustering\n"
        "Saas et al. 2017 + NeurIPS 2019 flat-cut criterion  "
        f"  k = {n_clusters} selected",
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
    """Build a per-cluster summary DataFrame for defect clustering.

    Args:
        features:    (N, F) feature matrix.
        labels:      (N,) cluster label array.
        metadata_df: DataFrame with columns: sample_id, defect_type, expert_group.

    Returns:
        DataFrame with one row per cluster:
            cluster_id, size, dominant_defect_type, defect_type_entropy,
            top_expert_group, mean_feature_norm
    """
    records = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        subset_meta = metadata_df[mask]
        subset_feat = features[mask]

        dominant_defect_type = (
            subset_meta["defect_type"].value_counts().idxmax()
            if "defect_type" in subset_meta.columns and len(subset_meta) > 0
            else "unknown"
        )

        # Shannon entropy of defect_type distribution
        if "defect_type" in subset_meta.columns and len(subset_meta) > 0:
            counts = subset_meta["defect_type"].value_counts(normalize=True)
            dtype_entropy = abs(float(-(counts * np.log(counts + 1e-9)).sum()))
        else:
            dtype_entropy = 0.0

        top_expert_group = (
            subset_meta["expert_group"].value_counts().idxmax()
            if "expert_group" in subset_meta.columns and len(subset_meta) > 0
            else "unknown"
        )

        records.append({
            "cluster_id":            int(cluster_id),
            "size":                  int(mask.sum()),
            "dominant_defect_type":  dominant_defect_type,
            "defect_type_entropy":   round(dtype_entropy, 3),
            "top_expert_group":      top_expert_group,
            "mean_norm_scaled":      round(float(np.linalg.norm(subset_feat, axis=1).mean()), 4),
        })

    return pd.DataFrame(records)


def plot_2d_projection(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "2-D Projection of Defect Features",
    ground_truth: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> None:
    """Scatter plot of 2D t-SNE / UMAP embeddings coloured by cluster label.

    Args:
        embeddings_2d: (N, 2) array from TSNE/UMAP transform.
        labels:        (N,) cluster label array.
        title:         Plot title.
        ground_truth:  (N,) true label IDs -- if provided, shows side-by-side.
        output_path:   Save path.
    """
    ncols = 2 if ground_truth is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 7, 6))
    if ncols == 1:
        axes = [axes]

    for ax, (lbl, title_suffix) in zip(
        axes,
        [(labels, "Cluster labels"), (ground_truth, "Ground truth (expert groups)")]
        if ground_truth is not None
        else [(labels, "")],
    ):
        unique_lbl = sorted(set(lbl))
        n_colors = max(len(unique_lbl), 6)
        cmap = plt.cm.get_cmap("tab10" if n_colors <= 10 else "tab20", n_colors)
        for i, uid in enumerate(unique_lbl):
            mask = lbl == uid
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[cmap(i / n_colors)], s=40, alpha=0.80,
                linewidths=0.3, edgecolors="white",
                label=str(uid),
            )
        ax.set_title(f"{title}\n{title_suffix}", fontsize=10)
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
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

def _compare_single(method_name, k, X, random_state):
    """Fit one (method, k) pair and return metrics (helper for parallelism)."""
    from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture

    try:
        if method_name == "kmeans":
            # Sweep n_init=3 (final fit in clusterer.py uses full n_init=10)
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=3).fit_predict(X)
        elif method_name == "gmm":
            X64 = X.astype(np.float64)
            labels = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-3,
                max_iter=300,
                n_init=3,
                random_state=random_state,
            ).fit(X64).predict(X64)
        elif method_name == "agglomerative":
            # Ward linkage needs O(N^2) memory — subsample for large datasets
            _MAX_AGGLO = 5000
            if len(X) > _MAX_AGGLO:
                rng = np.random.RandomState(random_state)
                idx = rng.choice(len(X), size=_MAX_AGGLO, replace=False)
                X_agglo = X[idx]
            else:
                X_agglo = X
            labels_agglo = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_agglo)
            sil = compute_silhouette(X_agglo, labels_agglo, max_samples=2000)
            db = compute_davies_bouldin(X_agglo, labels_agglo)
            ch = compute_calinski_harabasz(X_agglo, labels_agglo)
            return {"method": method_name, "k": k,
                    "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
                    "calinski_harabasz": round(ch, 1)}
        elif method_name == "minibatch_kmeans":
            labels = MiniBatchKMeans(
                n_clusters=k, random_state=random_state,
                batch_size=min(1024, len(X)), n_init=3,
            ).fit_predict(X)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        sil = compute_silhouette(X, labels, max_samples=2000)
        db = compute_davies_bouldin(X, labels)
        ch = compute_calinski_harabasz(X, labels)
    except Exception as exc:
        print(f"[compare_algorithms] {method_name} k={k} failed: {exc}")
        sil, db, ch = 0.0, 0.0, 0.0
    return {
        "method": method_name, "k": k,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 1),
    }


def compare_algorithms(
    features: np.ndarray,
    k_range: Tuple[int, int] = (3, 8),
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Fit KMeans, GMM, Agglomerative (Ward), and MiniBatchKMeans at each k.

    Computes three complementary cluster quality metrics for each (method, k):
      - Silhouette     : [-1, 1], higher is better
      - Davies-Bouldin : [0, inf), lower is better
      - Calinski-Harabasz: [0, inf), higher is better

    Uses joblib parallelism across all (method, k) pairs.

    Args:
        features:     (N, F) float32 feature matrix (unscaled).
        k_range:      (k_min, k_max) inclusive range of cluster counts to test.
        random_state: Random seed for reproducibility.
        n_jobs:       Number of parallel jobs (-1 = all cores).

    Returns:
        DataFrame with columns: method, k, silhouette, davies_bouldin, calinski_harabasz.
        Rows are sorted by (method, k).
    """
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(features).astype(np.float32, copy=False)

    # Subsample for comparison (ranking methods, not final eval)
    _MAX_COMPARE = 8000
    if X.shape[0] > _MAX_COMPARE:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=_MAX_COMPARE, replace=False)
        X = X[idx]
        print(f"[compare_algorithms] Subsampled to {_MAX_COMPARE} for comparison")

    k_min, k_max = k_range
    method_names = ["kmeans", "gmm", "agglomerative", "minibatch_kmeans"]

    # Run sequentially to avoid memory multiplication across workers
    rows = []
    for method_name in method_names:
        for k in range(k_min, k_max + 1):
            rows.append(_compare_single(method_name, k, X, random_state))

    return pd.DataFrame(rows).sort_values(["method", "k"]).reset_index(drop=True)


def plot_algorithm_comparison(
    comparison_df: pd.DataFrame,
    output_path: str,
    highlight_k: int = 6,
) -> None:
    """Plot silhouette / Davies-Bouldin / Calinski-Harabasz vs k, one line per algorithm.

    Produces a 3-row figure showing how each metric evolves with cluster count for
    each algorithm. A vertical dashed line marks the chosen k (highlight_k).

    Motivated by Saas et al. 2017 -- multi-metric elbow analysis for algorithm selection.

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

def _single_bootstrap(features, labels, sample_size, seed):
    """Run one bootstrap silhouette iteration (for joblib parallelism)."""
    rng = np.random.default_rng(seed)
    N = len(features)
    idx = rng.choice(N, size=sample_size, replace=False)
    sub_labels = labels[idx]
    if len(set(sub_labels[sub_labels >= 0])) < 2:
        return None
    try:
        return float(_sk_silhouette(features[idx][sub_labels >= 0],
                                    sub_labels[sub_labels >= 0]))
    except Exception:
        return None


def bootstrap_silhouette(
    features: "np.ndarray",
    labels: "np.ndarray",
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
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
        n_jobs:       Number of parallel jobs (-1 = all cores).

    Returns:
        Dict with keys: mean, std, ci_low, ci_high, scores_list.
        ci_low/ci_high are the 2.5th and 97.5th percentile (95% CI).
    """
    from joblib import Parallel, delayed

    N = len(features)
    # Cap subsample size at 8000 to avoid O(N^2) memory
    k = min(max(2, int(N * sample_frac)), 8000)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_bootstrap)(features, labels, k, random_state + i)
        for i in range(n_bootstrap)
    )
    scores = [s for s in results if s is not None]

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

    _UMAP_MAX_N = 5000
    if features.shape[0] > _UMAP_MAX_N:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(features.shape[0], size=_UMAP_MAX_N, replace=False)
        features = features[idx]
        labels = np.asarray(labels)[idx]

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=features.shape[0] > 10_000,
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

def _gap_for_k(
    k: int,
    features: np.ndarray,
    Vt: np.ndarray,
    col_min: np.ndarray,
    col_max: np.ndarray,
    N: int,
    D: int,
    n_references: int,
    random_state: int,
) -> Tuple[float, float, float]:
    """Compute Gap Statistic components for a single k value (helper for parallelism)."""
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(random_state + k * 1000)

    # Observed W_k
    km = MiniBatchKMeans(n_clusters=k, n_init=3, max_iter=100,
                         batch_size=min(1024, N), random_state=random_state)
    km.fit(features)
    log_wk = np.log(km.inertia_ + 1e-300)

    # Reference W_k* (B reference datasets)
    ref_log_wks = []
    for b in range(n_references):
        ref_rotated = rng.uniform(col_min, col_max, size=(N, D)).astype(np.float32, copy=False)
        ref_data = ref_rotated @ Vt  # rotate back to original space
        km_ref = MiniBatchKMeans(n_clusters=k, n_init=3, max_iter=100,
                                 batch_size=min(1024, N),
                                 random_state=random_state + b)
        km_ref.fit(ref_data)
        ref_log_wks.append(np.log(km_ref.inertia_ + 1e-300))

    ref_log_wks = np.array(ref_log_wks)
    expected = float(ref_log_wks.mean())
    sk = float(ref_log_wks.std() * np.sqrt(1 + 1.0 / n_references))
    return log_wk, expected, sk


def gap_statistic(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 12),
    n_references: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """Gap Statistic for principled k selection (Tibshirani et al. 2001).

    Gap(k) = E*[log(W_k)] - log(W_k)
    where W_k = within-cluster dispersion (inertia), E* = expectation
    under a uniform reference distribution (SVD-aligned bounding box).

    Optimal k = smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    where s_k is the simulation standard error.

    Uses MiniBatchKMeans and joblib parallelism across k values for speed.

    Args:
        features:      (N, F) feature matrix (should be scaled).
        k_range:       (k_min, k_max) inclusive range.
        n_references:  Number of reference datasets (paper recommends 10-20).
        random_state:  Seed for reproducibility.
        n_jobs:        Number of parallel jobs (-1 = all cores).

    Returns:
        Dict with keys:
            'gap_values':       list of Gap(k) for each k
            'sk_values':        list of s_k (standard error) for each k
            'optimal_k':        int -- smallest k satisfying gap criterion
            'log_wk':           list of log(W_k) (observed)
            'expected_log_wk':  list of E*[log(W_k)] (reference mean)
            'k_range_tested':   list of k values tested
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(random_state)
    N, _F = features.shape
    ks = list(range(k_range[0], k_range[1] + 1))

    # Subsample for large datasets to keep runtime manageable
    max_gap_samples = 10000
    if N > max_gap_samples:
        idx = rng.choice(N, size=max_gap_samples, replace=False)
        features = features[idx]
        N = max_gap_samples

    # Force float32 end-to-end: each joblib worker pickles `features` + `Vt`,
    # so float64 doubles per-worker RAM and triggers ArrayMemoryError.
    features = np.ascontiguousarray(features, dtype=np.float32)

    # SVD-aligned bounding box for reference generation (Tibshirani 2001 Sec 4)
    _, _, Vt = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
    Vt = Vt.astype(np.float32, copy=False)
    D = Vt.shape[0]
    rotated = features @ Vt.T
    col_min = rotated.min(axis=0)
    col_max = rotated.max(axis=0)

    # Cap workers: each pickles the full feature matrix + references.
    if n_jobs == -1 or n_jobs > 4:
        _safe_jobs = 4
    else:
        _safe_jobs = n_jobs

    # Parallel computation across k values
    results = Parallel(n_jobs=_safe_jobs, prefer="threads")(
        delayed(_gap_for_k)(k, features, Vt, col_min, col_max,
                            N, D, n_references, random_state)
        for k in ks
    )

    log_wk = [r[0] for r in results]
    expected_log_wk = [r[1] for r in results]
    sk_values = [r[2] for r in results]
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

def _consensus_single_run(algo_name, n_clusters, features, seed):
    """Fit one clustering run for consensus (for joblib parallelism)."""
    from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture

    N = features.shape[0]
    if algo_name == "kmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, max_iter=200,
                                batch_size=min(1024, N), random_state=seed)
        return model.fit_predict(features)
    elif algo_name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(features)
    elif algo_name == "gmm":
        try:
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                reg_covar=1e-3,
                max_iter=200,
                n_init=1,
                random_state=seed,
            )
            return model.fit_predict(features.astype(np.float64))
        except Exception:
            return None
    return None


def consensus_clustering(
    features: np.ndarray,
    n_clusters: int = 5,
    algorithms: Optional[List[str]] = None,
    n_runs: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Consensus clustering via co-association matrix.

    Combines results from multiple clustering algorithms and runs to produce
    a more robust final clustering.

    Algorithm:
        1. Run each algorithm M times with different random seeds (parallel)
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
        n_jobs:       Number of parallel jobs (-1 = all cores).

    Returns:
        (consensus_labels, co_association_matrix)
        - consensus_labels: shape (N,) int array
        - co_association_matrix: shape (N, N) float in [0, 1]
    """
    from sklearn.cluster import AgglomerativeClustering
    from joblib import Parallel, delayed

    if algorithms is None:
        algorithms = ["kmeans", "agglomerative", "gmm"]

    N = features.shape[0]

    # Subsample for large datasets to avoid O(N^2) memory in co-assoc matrix
    max_consensus_samples = 5000
    subsample_idx = None
    if N > max_consensus_samples:
        rng = np.random.default_rng(random_state)
        subsample_idx = rng.choice(N, size=max_consensus_samples, replace=False)
        features = features[subsample_idx]
        N = max_consensus_samples

    # Build task list: (algo_name, seed) for all runs
    tasks = []
    run_idx = 0
    for algo_name in algorithms:
        for _ in range(n_runs):
            tasks.append((algo_name, random_state + run_idx))
            run_idx += 1

    # Parallel fit all runs
    all_labels = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_consensus_single_run)(algo_name, n_clusters, features, seed)
        for algo_name, seed in tasks
    )

    # Build co-association matrix from results
    co_assoc = np.zeros((N, N), dtype=np.float32)
    total_runs = 0
    for labels in all_labels:
        if labels is None:
            continue
        total_runs += 1
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

    if subsample_idx is not None:
        print(f"[metrics] Consensus clustering (subsampled to {N}): "
              f"{len(algorithms)} algorithms x {n_runs} runs = {total_runs}")
    else:
        print(f"[metrics] Consensus clustering: {len(algorithms)} algorithms x "
              f"{n_runs} runs = {total_runs} total, k={n_clusters}")
    return consensus_labels, co_assoc


# ---------------------------------------------------------------------------
# Defect mask grid (replaces plot_cluster_heatmaps for defect context)
# ---------------------------------------------------------------------------

def plot_defect_masks_grid(
    masks: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int = 3,
    output_path: Optional[str] = None,
) -> None:
    """Show representative defect mask images per cluster in a grid layout.

    For each cluster, selects representative samples (closest to centroid by
    L2 norm of the flattened mask) and displays them as grayscale images.

    This replaces plot_cluster_heatmaps from the flatness pipeline, which
    showed 2-D flatness profiles. Here we show binary or grayscale defect
    masks that represent the spatial extent and shape of each defect.

    Args:
        masks:          (N, H, W) or (N, H, W, C) array of defect mask images.
                        Can be binary masks, grayscale, or RGB images.
        labels:         (N,) cluster label array.
        n_per_cluster:  Number of representative masks to show per cluster.
        output_path:    Save path (.png). If None, displays interactively.
    """
    unique_clusters = sorted(c for c in set(labels) if c >= 0)
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        print("[metrics] No valid clusters found -- skipping defect mask grid.")
        return

    fig, axes = plt.subplots(
        n_clusters, n_per_cluster,
        figsize=(n_per_cluster * 3.5, n_clusters * 2.8),
        squeeze=False,
    )

    for row, cluster_id in enumerate(unique_clusters):
        cluster_mask_indices = np.where(labels == cluster_id)[0]
        n_in_cluster = len(cluster_mask_indices)

        # Select representative samples: pick those closest to the cluster
        # mean in flattened mask space (simple but effective heuristic)
        if n_in_cluster > n_per_cluster:
            flat = masks[cluster_mask_indices].reshape(n_in_cluster, -1).astype(np.float64)
            centroid = flat.mean(axis=0)
            dists = np.linalg.norm(flat - centroid, axis=1)
            selected = cluster_mask_indices[np.argsort(dists)[:n_per_cluster]]
        else:
            selected = cluster_mask_indices[:n_per_cluster]

        # Determine display parameters for this cluster
        cluster_masks = masks[cluster_mask_indices]
        vmin = float(np.percentile(cluster_masks, 1))
        vmax = float(np.percentile(cluster_masks, 99))
        if vmax - vmin < 1e-6:
            vmin, vmax = 0.0, 1.0

        for col in range(n_per_cluster):
            ax = axes[row][col]
            if col < len(selected):
                img = masks[selected[col]]

                # Handle different mask formats
                if img.ndim == 2:
                    # Grayscale or binary mask
                    ax.imshow(img, aspect="auto", cmap="gray",
                              vmin=vmin, vmax=vmax)
                elif img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                    # RGB/RGBA or single-channel with trailing dim
                    if img.shape[-1] == 1:
                        ax.imshow(img[:, :, 0], aspect="auto", cmap="gray",
                                  vmin=vmin, vmax=vmax)
                    else:
                        # Normalize to [0, 1] if needed for RGB display
                        display_img = img.astype(np.float64)
                        if display_img.max() > 1.0:
                            display_img = display_img / 255.0
                        ax.imshow(np.clip(display_img, 0, 1), aspect="auto")
                else:
                    ax.imshow(img.reshape(img.shape[0], -1), aspect="auto",
                              cmap="gray")

                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    n_total = int((labels == cluster_id).sum())
                    ax.set_ylabel(f"Cluster {cluster_id}\n(n={n_total})",
                                  fontsize=9, labelpad=6)
                if row == 0:
                    ax.set_title(f"Sample {col + 1}", fontsize=8)
            else:
                ax.set_visible(False)

    plt.suptitle(
        "Representative Defect Masks per Cluster\n"
        "(selected by proximity to cluster centroid in mask space)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[metrics] Defect mask grid saved: {output_path}")
    else:
        plt.show()

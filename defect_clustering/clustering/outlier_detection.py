"""
Per-Cluster Outlier Detection (DECOR-style)
============================================

Implements the outlier detection strategy from:
    DECOR (AAAI 2026 Workshop) -- Deep Embedding Clustering with
    Orientation Robustness.

After clustering, each cluster is independently analysed for outliers
using two complementary methods:

1. **Isolation Forest** (Liu et al. 2008) -- tree-based anomaly detection.
   Outliers are points that require fewer splits to isolate.
   Works well for high-dimensional spaces without density assumptions.

2. **Local Outlier Factor (LOF)** (Breunig et al. 2000) -- density-based.
   Outliers are points whose local density is significantly lower than
   their neighbours'.  Catches outliers in varying-density clusters.

The combined approach flags a sample as an outlier if EITHER method
detects it (union strategy), or optionally only if BOTH agree
(intersection strategy for higher precision).

Usage:
    from defect_clustering.clustering.outlier_detection import (
        detect_outliers_per_cluster,
        plot_outlier_summary,
    )

    outlier_mask, outlier_info = detect_outliers_per_cluster(
        features, labels, contamination=0.05
    )
    # outlier_mask: (N,) boolean array
    # outlier_info: dict with per-cluster details

References:
    Liu, Ting & Zhou (2008) -- Isolation Forest (ICDM)
    Breunig et al. (2000) -- LOF (SIGMOD)
    DECOR (2025) -- arXiv:2510.03328
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np


def detect_outliers_per_cluster(
    features: np.ndarray,
    labels: np.ndarray,
    contamination: float = 0.05,
    strategy: Literal["union", "intersection"] = "union",
    min_cluster_size: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Detect outliers within each cluster using Isolation Forest + LOF.

    Parameters
    ----------
    features : ndarray of shape (N, D)
        Feature matrix (scaled).
    labels : ndarray of shape (N,)
        Cluster assignments.
    contamination : float
        Expected proportion of outliers per cluster (0 < c < 0.5).
    strategy : {'union', 'intersection'}
        How to combine IF and LOF results:
        - 'union': flagged by either method (higher recall)
        - 'intersection': flagged by both methods (higher precision)
    min_cluster_size : int
        Clusters smaller than this skip outlier detection.
    random_state : int
        Random seed.

    Returns
    -------
    outlier_mask : ndarray of shape (N,), dtype=bool
        True for outlier samples.
    info : dict
        Per-cluster details: n_outliers, methods, cluster sizes.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    N = len(features)
    outlier_mask = np.zeros(N, dtype=bool)
    if_outliers = np.zeros(N, dtype=bool)
    lof_outliers = np.zeros(N, dtype=bool)

    unique_labels = np.unique(labels)
    cluster_info = {}

    for k in unique_labels:
        mask = labels == k
        n_k = mask.sum()
        idx = np.where(mask)[0]

        if n_k < min_cluster_size:
            cluster_info[int(k)] = {
                "size": int(n_k),
                "n_outliers_if": 0,
                "n_outliers_lof": 0,
                "n_outliers_combined": 0,
                "skipped": True,
            }
            continue

        X_k = features[mask]

        # Isolation Forest
        n_neighbors_lof = min(20, n_k - 1)
        actual_contamination = min(contamination, (n_k - 1) / n_k - 0.001)
        actual_contamination = max(actual_contamination, 0.001)

        try:
            iso = IsolationForest(
                contamination=actual_contamination,
                random_state=random_state,
                n_jobs=-1,
            )
            if_pred = iso.fit_predict(X_k)  # -1 = outlier, 1 = inlier
            if_mask = if_pred == -1
        except Exception:
            if_mask = np.zeros(n_k, dtype=bool)

        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors_lof,
                contamination=actual_contamination,
            )
            lof_pred = lof.fit_predict(X_k)  # -1 = outlier, 1 = inlier
            lof_mask = lof_pred == -1
        except Exception:
            lof_mask = np.zeros(n_k, dtype=bool)

        # Combine
        if strategy == "union":
            combined = if_mask | lof_mask
        else:
            combined = if_mask & lof_mask

        # Apply to global arrays
        if_outliers[idx] = if_mask
        lof_outliers[idx] = lof_mask
        outlier_mask[idx] = combined

        cluster_info[int(k)] = {
            "size": int(n_k),
            "n_outliers_if": int(if_mask.sum()),
            "n_outliers_lof": int(lof_mask.sum()),
            "n_outliers_combined": int(combined.sum()),
            "skipped": False,
        }

    info = {
        "strategy": strategy,
        "contamination": contamination,
        "total_outliers": int(outlier_mask.sum()),
        "total_outliers_if": int(if_outliers.sum()),
        "total_outliers_lof": int(lof_outliers.sum()),
        "outlier_rate": float(outlier_mask.sum()) / N if N > 0 else 0.0,
        "per_cluster": cluster_info,
    }

    return outlier_mask, info


def plot_outlier_summary(
    features: np.ndarray,
    labels: np.ndarray,
    outlier_mask: np.ndarray,
    outlier_info: Dict,
    output_path: str,
) -> None:
    """Generate a summary plot of outlier detection results.

    Creates a 2x2 figure:
    - Top-left: UMAP with outliers highlighted
    - Top-right: Outlier counts per cluster (bar chart)
    - Bottom-left: IF vs LOF agreement per cluster
    - Bottom-right: Outlier rate by cluster size

    Parameters
    ----------
    features : ndarray (N, D)
    labels : ndarray (N,)
    outlier_mask : ndarray (N,) bool
    outlier_info : dict from detect_outliers_per_cluster
    output_path : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Per-Cluster Outlier Detection (DECOR-style)\n"
        f"Strategy: {outlier_info['strategy']} | "
        f"Total outliers: {outlier_info['total_outliers']} "
        f"({outlier_info['outlier_rate']:.1%})",
        fontsize=14, fontweight="bold",
    )

    # ── Top-left: UMAP with outliers ──────────────────────────────────
    ax = axes[0, 0]
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            random_state=42,
        )
        # Subsample for speed if needed
        N = len(features)
        if N > 8000:
            idx = np.random.RandomState(42).choice(N, 8000, replace=False)
            emb = reducer.fit_transform(features[idx])
            ol_sub = outlier_mask[idx]
            lab_sub = labels[idx]
        else:
            emb = reducer.fit_transform(features)
            ol_sub = outlier_mask
            lab_sub = labels

        # Plot inliers
        inlier = ~ol_sub
        ax.scatter(
            emb[inlier, 0], emb[inlier, 1],
            c=lab_sub[inlier], cmap="tab20", alpha=0.3, s=5,
            label="Inliers",
        )
        # Plot outliers
        ax.scatter(
            emb[ol_sub, 0], emb[ol_sub, 1],
            c="red", marker="x", s=30, alpha=0.8,
            label=f"Outliers ({ol_sub.sum()})",
        )
        ax.legend(fontsize=8)
    except ImportError:
        ax.text(0.5, 0.5, "Install umap-learn\nfor this plot",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("UMAP with Outliers Highlighted")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # ── Top-right: Outlier counts per cluster ─────────────────────────
    ax = axes[0, 1]
    per_cluster = outlier_info["per_cluster"]
    cluster_ids = sorted(per_cluster.keys())
    n_outliers = [per_cluster[k]["n_outliers_combined"] for k in cluster_ids]
    sizes = [per_cluster[k]["size"] for k in cluster_ids]

    x_pos = np.arange(len(cluster_ids))
    ax.bar(x_pos, n_outliers, color="tomato", alpha=0.8, label="Outliers")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{k}" for k in cluster_ids], rotation=45, fontsize=7)
    ax.set_ylabel("Number of Outliers")
    ax.set_title("Outliers per Cluster")
    ax.legend(fontsize=8)

    # ── Bottom-left: IF vs LOF agreement ──────────────────────────────
    ax = axes[1, 0]
    n_if = [per_cluster[k]["n_outliers_if"] for k in cluster_ids]
    n_lof = [per_cluster[k]["n_outliers_lof"] for k in cluster_ids]

    width = 0.35
    ax.bar(x_pos - width/2, n_if, width, label="Isolation Forest", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, n_lof, width, label="LOF", color="darkorange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{k}" for k in cluster_ids], rotation=45, fontsize=7)
    ax.set_ylabel("Outlier Count")
    ax.set_title("Isolation Forest vs LOF per Cluster")
    ax.legend(fontsize=8)

    # ── Bottom-right: Outlier rate vs cluster size ────────────────────
    ax = axes[1, 1]
    rates = [
        per_cluster[k]["n_outliers_combined"] / max(per_cluster[k]["size"], 1)
        for k in cluster_ids
    ]
    ax.scatter(sizes, rates, c="tomato", edgecolors="black", s=60, alpha=0.8)
    for i, k in enumerate(cluster_ids):
        ax.annotate(f"C{k}", (sizes[i], rates[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Outlier Rate")
    ax.set_title("Outlier Rate vs Cluster Size")
    ax.axhline(y=outlier_info["contamination"], color="gray",
               linestyle="--", alpha=0.5, label=f"Expected ({outlier_info['contamination']:.0%})")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

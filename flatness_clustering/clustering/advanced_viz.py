"""
Advanced Clustering Visualizations for Flatness Profiles
=========================================================

Six publication-grade visualizations that go beyond basic bar charts:

    plot_feature_discriminability() — ANOVA F-score per feature group (Fisher 1925)
    plot_confusion_matrix()        — cluster vs ground-truth pattern heatmap + NMI/ARI
    plot_parallel_coordinates()    — multi-axis feature profile, one line per coil
    plot_silhouette_distribution() — per-cluster silhouette violin (Rousseeuw 1987)
    plot_temporal_evolution()      — cluster mix stacked-area over production time
    plot_cluster_distance_matrix() — inter-cluster centroid distance heatmap

All functions write a PNG to `output_path` (matplotlib Agg backend, no display).
All functions are safe with N as small as 2 coils per cluster.

Usage:
    from flatness_clustering.clustering.advanced_viz import (
        plot_feature_discriminability, plot_confusion_matrix,
        plot_parallel_coordinates, plot_silhouette_distribution,
        plot_temporal_evolution, plot_cluster_distance_matrix,
    )
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from flatness_clustering.features.statistical_features import feature_group_slices


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CMAP   = "tab10"
_DARK   = "#212121"
_GREY   = "#546E7A"
_DPI    = 150

# Feature group slices, taken from the extractor so they follow its layout
# (30 + 50 + 15 + 10 + 1000 = 1105 columns).
_GROUP_NAMES = {"A": "Group A\nZonal", "B": "Group B\nSnapshot", "C": "Group C\nSpectral",
                "D": "Group D\nTemporal", "E": "Group E\nROCKET"}
_GROUPS = [(_GROUP_NAMES[g], cols) for g, cols in feature_group_slices()]


def _cluster_colors(k: int) -> np.ndarray:
    return matplotlib.colormaps[_CMAP].resampled(max(k, 2))(np.linspace(0, 1, k))


def _footer(fig) -> None:
    fig.text(0.5, 0.005,
             "Flatness Clustering — Advanced Analysis",
             ha="center", va="bottom", fontsize=7, color=_GREY)


# ---------------------------------------------------------------------------
# 1. Feature Discriminability (ANOVA F-score — Fisher 1925)
# ---------------------------------------------------------------------------

def plot_feature_discriminability(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Feature Discriminability — ANOVA F-score (Fisher 1925)",
) -> None:
    """Two-panel figure showing which features drive cluster separation.

    Left:  Mean ANOVA F-score per feature group (5 bars).
           F = MS_between / MS_within — high F = group drives separation.
    Right: Top 15 individual features by F-score (horizontal bar chart).

    Replaces the radar chart which becomes uninformative after StandardScaler
    (all group abs-means converge to ~0.8).

    Args:
        features:    (N, F) float32 feature matrix (StandardScaler space).
        labels:      (N,) int cluster label array.
        output_path: Where to save the PNG.
    """
    from scipy.stats import f_oneway

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    n_features = features.shape[1]

    # Compute per-feature ANOVA F-score
    f_scores = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        groups = [features[labels == cid, j] for cid in cluster_ids]
        # Need at least 2 samples per group for ANOVA
        if all(len(g) >= 2 for g in groups):
            f_val, _ = f_oneway(*groups)
            f_scores[j] = f_val if np.isfinite(f_val) else 0.0

    # Group-level mean F
    group_names = [g[0] for g in _GROUPS]
    group_f = []
    for _, slc in _GROUPS:
        indices = range(*slc.indices(n_features))
        group_f.append(float(f_scores[list(indices)].mean()) if len(list(indices)) > 0 else 0.0)

    # Top 15 individual features
    top_k = min(15, n_features)
    top_idx = np.argsort(f_scores)[::-1][:top_k]

    # Map feature index to group name
    def _feat_group(idx):
        for name, slc in _GROUPS:
            r = range(*slc.indices(n_features))
            if idx in r:
                return name.split("\n")[1] if "\n" in name else name
        return "?"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.35)

    # Left: group-level F-scores
    group_colors = ["#1565C0", "#00695C", "#E65100", "#C62828", "#7B1FA2"]
    axes[0].bar(group_names, group_f, color=group_colors,
                edgecolor="white", linewidth=1.2)
    for i, v in enumerate(group_f):
        axes[0].text(i, v + max(group_f) * 0.02, f"{v:.1f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Mean ANOVA F-score")
    axes[0].set_title("Cluster Discriminability by Feature Group\n"
                      "F = MS_between / MS_within  —  higher = more discriminative",
                      fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: top 15 individual features
    top_labels = [f"f{idx:03d} ({_feat_group(idx)})" for idx in top_idx]
    top_vals = f_scores[top_idx]
    y_pos = np.arange(top_k)
    bar_colors = [group_colors[
        next((gi for gi, (_, slc) in enumerate(_GROUPS)
              if idx in range(*slc.indices(n_features))), 0)]
        for idx in top_idx]
    axes[1].barh(y_pos, top_vals, color=bar_colors, edgecolor="white", height=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ANOVA F-score")
    axes[1].set_title(f"Top {top_k} Most Discriminative Features\n"
                      "Colour = feature group", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold", color=_DARK, y=0.98)
    _footer(fig)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Feature discriminability saved: {output_path}")


# Backward compatibility alias
plot_radar_chart = plot_feature_discriminability


# ---------------------------------------------------------------------------
# 2. Confusion matrix (cluster vs ground-truth pattern)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str,
    pattern_col: str = "pattern",
) -> None:
    """Heatmap of assigned cluster vs ground-truth pattern.

    Rows = assigned cluster, columns = true pattern.
    Annotates with NMI (Normalized Mutual Information) and
    ARI (Adjusted Rand Index) in the subtitle.

    Args:
        labels:      (N,) int cluster label array.
        metadata:    DataFrame aligned with labels, containing pattern_col.
        output_path: Where to save the PNG.
        pattern_col: Column in metadata containing ground-truth labels.
    """
    if pattern_col not in metadata.columns:
        print(f"[advanced_viz] '{pattern_col}' not in metadata - skipping confusion matrix.")
        return

    from sklearn.metrics import (
        adjusted_rand_score,
        confusion_matrix,
        normalized_mutual_info_score,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    true_labels = metadata[pattern_col].values[:len(labels)]
    unique_patterns = sorted(set(true_labels))
    unique_clusters = sorted(set(labels))

    # Build confusion matrix: rows = cluster, cols = pattern
    pattern_to_idx = {p: i for i, p in enumerate(unique_patterns)}
    cm = confusion_matrix(
        labels,
        [pattern_to_idx[p] for p in true_labels],
        labels=unique_clusters,
    )

    nmi = normalized_mutual_info_score(true_labels, labels)
    ari = adjusted_rand_score(true_labels, labels)

    fig, ax = plt.subplots(figsize=(max(6, len(unique_patterns) * 1.2),
                                     max(5, len(unique_clusters) * 0.9)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=unique_patterns,
        yticklabels=[f"Cluster {c}" for c in unique_clusters],
        ax=ax, linewidths=0.5, linecolor="#E0E0E0",
    )
    ax.set_xlabel("Ground-truth pattern", fontsize=10)
    ax.set_ylabel("Assigned cluster", fontsize=10)
    ax.set_title(
        f"Cluster vs Ground-Truth Pattern\n"
        f"NMI = {nmi:.3f}   ARI = {ari:.3f}   "
        f"(1.0 = perfect, 0 = random)",
        fontsize=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    _footer(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Confusion matrix saved: {output_path}  "
          f"(NMI={nmi:.3f}, ARI={ari:.3f})")


# ---------------------------------------------------------------------------
# 3. Parallel coordinates
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    n_sample: int = 50,
) -> None:
    """Parallel coordinates across 4 group-mean axes, one line per coil.

    Each coil is a semi-transparent line across 4 axes (one per feature group).
    The value on each axis = mean |feature| within that group, normalized to [0,1].
    Mean per cluster shown as full-opacity thick line.

    Args:
        features:    (N, F) float32 feature matrix.
        labels:      (N,) int cluster label array.
        output_path: Where to save the PNG.
        n_sample:    Max coils to draw (sampled randomly if N > n_sample).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)
    colors = _cluster_colors(k)
    n_groups = len(_GROUPS)
    group_names = [g[0].replace("\n", " ") for g in _GROUPS]

    # Compute per-coil group means
    group_vals = np.zeros((len(features), n_groups), dtype=np.float64)
    for j, (_, slc) in enumerate(_GROUPS):
        group_vals[:, j] = np.abs(features[:, slc]).mean(axis=1)

    # Normalize to [0, 1] per axis
    col_min = group_vals.min(axis=0)
    col_max = group_vals.max(axis=0)
    col_range = np.where((col_max - col_min) < 1e-10, 1.0, col_max - col_min)
    group_norm = (group_vals - col_min) / col_range

    # Sample if needed
    rng = np.random.default_rng(42)
    if len(features) > n_sample:
        idx = rng.choice(len(features), n_sample, replace=False)
    else:
        idx = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(9, 5))

    x_ticks = np.arange(n_groups)

    # Draw individual coil lines (semi-transparent)
    for i in idx:
        cid = labels[i]
        ci = cluster_ids.index(cid)
        ax.plot(x_ticks, group_norm[i], color=colors[ci], alpha=0.07, lw=0.6)

    # Draw cluster mean lines (full opacity, thick)
    legend_handles = []
    for ci, cid in enumerate(cluster_ids):
        mask = labels == cid
        mean_vals = group_norm[mask].mean(axis=0)
        ax.plot(x_ticks, mean_vals, color=colors[ci], lw=2.5,
                marker="o", markersize=6, zorder=5)
        legend_handles.append(
            mpatches.Patch(color=colors[ci], label=f"Cluster {cid} (n={mask.sum()})")
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Normalized group mean |feature|")
    ax.set_title(
        "Parallel Coordinates — Feature Group Profile per Cluster\n"
        "Thin lines = individual coils  ·  Thick lines = cluster mean",
        fontsize=10,
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Vertical separators at each axis
    for x in x_ticks:
        ax.axvline(x, color=_GREY, lw=0.6, alpha=0.5)

    _footer(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Parallel coordinates saved: {output_path}")


# ---------------------------------------------------------------------------
# 4. Per-Cluster Silhouette Distribution (Rousseeuw 1987)
# ---------------------------------------------------------------------------

def plot_silhouette_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Per-cluster violin of silhouette coefficients — gold standard diagnostic.

    s(i) = (b - a) / max(a, b)   where
        a = mean intra-cluster distance
        b = mean nearest-cluster distance
    s ∈ [-1, 1]:  s > 0 = well-assigned,  s < 0 = likely mis-assigned.

    One violin per cluster, coloured by cluster.  Red dashed line at s = 0.
    Annotated with n (size) and median silhouette per cluster.

    Replaces the L2-norm violin which is a crude single-number that
    does not diagnose cluster quality.

    Reference: Rousseeuw, P. J. (1987) "Silhouettes: a graphical aid to
    the interpretation and validation of cluster analysis".

    Args:
        features:    (N, F) feature matrix (StandardScaler space).
        labels:      (N,) int cluster label array.
        output_path: Where to save the PNG.
    """
    from sklearn.metrics import silhouette_samples

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)
    colors = _cluster_colors(k)

    sil_vals = silhouette_samples(features, labels)
    global_mean = float(sil_vals.mean())

    data = [sil_vals[labels == cid] for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(max(7, k * 1.4), 5.5))

    parts = ax.violinplot(data, positions=range(k), showmedians=True,
                          showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.65)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(_DARK)
            parts[key].set_linewidth(1.2)

    # s = 0 boundary and global mean
    ax.axhline(0, color=_DARK, linestyle="-", lw=1.0, alpha=0.6, label="s = 0 (boundary)")
    ax.axhline(global_mean, color="red", linestyle="--", lw=1.5, alpha=0.8,
               label=f"global mean = {global_mean:.3f}")

    # Annotate each violin
    for i, (cid, d) in enumerate(zip(cluster_ids, data)):
        med = float(np.median(d))
        n_neg = int((d < 0).sum())
        ax.text(i, max(d.max(), 0) + 0.04,
                f"n={len(d)}\nmed={med:.3f}" + (f"\n{n_neg} neg" if n_neg > 0 else ""),
                ha="center", fontsize=7.5, color=_GREY, va="bottom")

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
    ax.set_ylabel("Silhouette coefficient  s(i)")
    ax.set_title(
        "Per-Cluster Silhouette Distribution (Rousseeuw 1987)\n"
        "s(i) = (b−a)/max(a,b)  ·  s > 0 = well-assigned  ·  s < 0 = mis-assigned",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    _footer(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Silhouette distribution saved: {output_path}")


# Backward compatibility alias
plot_feature_violin = plot_silhouette_distribution


# ---------------------------------------------------------------------------
# 5. Temporal evolution (stacked area)
# ---------------------------------------------------------------------------

def plot_temporal_evolution(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    output_path: str,
    timestamp_col: str = "timestamp",
    n_bins: int = 20,
) -> None:
    """Stacked area chart showing cluster distribution over production time.

    Detects whether steel production drifts toward a different pattern over
    time (e.g. more edge_crack as roll wear increases).

    Args:
        metadata:      DataFrame aligned with labels (must have timestamp_col).
        labels:        (N,) int cluster label array.
        output_path:   Where to save the PNG.
        timestamp_col: Column containing timestamps (parseable by pd.to_datetime).
        n_bins:        Number of time windows to aggregate into.
    """
    if timestamp_col not in metadata.columns:
        print(f"[advanced_viz] '{timestamp_col}' not in metadata - skipping temporal evolution.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    df = metadata[: len(labels)].copy()
    df["_cluster"] = labels
    df["_time"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=["_time"])

    if len(df) < 2:
        print("[advanced_viz] Not enough valid timestamps - skipping temporal evolution.")
        return

    df = df.sort_values("_time")
    df["_bin"] = pd.cut(df["_time"], bins=n_bins, labels=False, include_lowest=True)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)
    colors = _cluster_colors(k)

    # Build count matrix: (n_bins, k)
    pivot = (
        df.groupby(["_bin", "_cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cluster_ids, fill_value=0)
    )

    # Time axis labels (bin midpoints) — use integer nanoseconds to avoid
    # Timestamp arithmetic issues across pandas/numpy versions
    bin_edges = pd.cut(df["_time"], bins=n_bins, retbins=True)[1]
    _ns = np.array([pd.Timestamp(t).value for t in bin_edges])
    bin_mids  = [pd.Timestamp(int((_ns[i] + _ns[i + 1]) // 2)) for i in range(len(_ns) - 1)]
    mid_labels = [t.strftime("%b %d") for t in bin_mids]

    xs = pivot.index.values
    ys = [pivot[cid].values for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(xs, ys, labels=[f"Cluster {c}" for c in cluster_ids],
                 colors=colors, alpha=0.85)

    tick_ids = pivot.index[::max(1, len(pivot) // 8)]
    ax.set_xticks(tick_ids)
    ax.set_xticklabels(
        [mid_labels[int(i)] for i in tick_ids if int(i) < len(mid_labels)],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_xlabel("Production date")
    ax.set_ylabel("Coil count per time window")
    ax.set_title(
        "Cluster Distribution Over Production Time\n"
        "Shift in dominant colour = drift in flatness pattern mix",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=min(k, 3))
    ax.grid(axis="y", alpha=0.2)

    _footer(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Temporal evolution saved: {output_path}")


# ---------------------------------------------------------------------------
# 6. Inter-cluster distance matrix
# ---------------------------------------------------------------------------

def plot_cluster_distance_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Heatmap of pairwise Euclidean distance between cluster centroids.

    Low distance between two clusters = they are close in feature space
    and may be candidates for merging (or k should be reduced).

    Args:
        features:    (N, F) feature matrix (raw, not necessarily scaled).
        labels:      (N,) int cluster label array.
        output_path: Where to save the PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)

    # Compute centroids
    centroids = np.array([features[labels == cid].mean(axis=0)
                          for cid in cluster_ids])

    # Pairwise Euclidean distances
    dist_matrix = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            dist_matrix[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))

    # Normalize to [0, 1] for readability
    max_d = dist_matrix.max()
    dist_norm = dist_matrix / (max_d + 1e-10)

    labels_str = [f"C{c}" for c in cluster_ids]

    # Show only the normalised matrix — raw Euclidean distances are data-scale-dependent
    # and produce unreadable annotations (e.g. 1e8) for unnormalised feature spaces.
    np.fill_diagonal(dist_norm, np.nan)
    min_val = float(np.nanmin(dist_norm))
    min_pos = np.unravel_index(np.nanargmin(dist_norm), dist_norm.shape)
    np.fill_diagonal(dist_norm, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        dist_norm, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=labels_str, yticklabels=labels_str,
        ax=ax, linewidths=0.8, linecolor="#E0E0E0",
        vmin=0, vmax=1, annot_kws={"fontsize": 10},
    )
    ax.set_title(
        f"Normalised inter-cluster distance  (0 = same centroid, 1 = max)\n"
        f"Closest pair: C{cluster_ids[min_pos[0]]} ↔ C{cluster_ids[min_pos[1]]} "
        f"(d_norm = {min_val:.2f})",
        fontsize=10,
    )
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)

    fig.suptitle("Inter-Cluster Centroid Distance Matrix", fontsize=12,
                 fontweight="bold", color=_DARK)
    _footer(fig)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Distance matrix saved: {output_path}")


# ---------------------------------------------------------------------------
# Gap Statistic plot (Tibshirani et al. 2001)
# ---------------------------------------------------------------------------

def plot_gap_statistic(
    gap_result: dict,
    output_path: str,
    title: str = "Gap Statistic (Tibshirani et al. 2001)",
) -> None:
    """Plot Gap(k) curve with error bars and optimal k annotation.

    Args:
        gap_result: Dict returned by metrics.gap_statistic().
        output_path: Path for the output PNG.
        title: Figure title.
    """
    ks = gap_result["k_range_tested"]
    gaps = gap_result["gap_values"]
    sks = gap_result["sk_values"]
    optimal_k = gap_result["optimal_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.85)

    # Left: Gap(k) with error bars
    ax1.errorbar(ks, gaps, yerr=sks, fmt="o-", color="#1976D2",
                 capsize=4, capthick=1.5, markersize=6, lw=2,
                 label="Gap(k) +/- s_k")
    ax1.axvline(optimal_k, color="#D32F2F", ls="--", lw=1.5,
                label=f"Optimal k = {optimal_k}")
    ax1.set_xlabel("Number of clusters (k)", fontsize=11)
    ax1.set_ylabel("Gap(k)", fontsize=11)
    ax1.set_title("Gap Statistic\nGap(k) = E*[log W_k] - log W_k", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: Observed vs Expected log(W_k)
    ax2.plot(ks, gap_result["log_wk"], "s-", color="#E65100",
             markersize=5, lw=1.5, label="Observed log(W_k)")
    ax2.plot(ks, gap_result["expected_log_wk"], "o--", color="#388E3C",
             markersize=5, lw=1.5, label="E*[log(W_k)] (reference)")
    ax2.axvline(optimal_k, color="#D32F2F", ls="--", lw=1.5)
    ax2.set_xlabel("Number of clusters (k)", fontsize=11)
    ax2.set_ylabel("log(W_k)", fontsize=11)
    ax2.set_title("Within-cluster dispersion\nObserved vs uniform reference", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", color=_DARK)
    _footer(fig)
    fig.tight_layout()

    from pathlib import Path as _Path
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[advanced_viz] Gap statistic plot saved: {output_path}")

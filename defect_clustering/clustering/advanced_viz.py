"""
Advanced Clustering Visualizations for Defect Groups
=====================================================

Seven publication-grade visualizations plus an alluvial diagram:

    plot_feature_discriminability() -- ANOVA F-score per feature group (Fisher 1925)
    plot_confusion_matrix()        -- cluster vs expert group heatmap + NMI/ARI
    plot_parallel_coordinates()    -- multi-axis feature profile, one line per sample
    plot_silhouette_distribution() -- per-cluster silhouette violin (Rousseeuw 1987)
    plot_temporal_evolution()       -- cluster mix stacked-area over time
    plot_cluster_distance_matrix() -- inter-cluster centroid distance heatmap
    plot_gap_statistic()           -- Gap(k) curve with error bars
    plot_alluvial_diagram()        -- Sankey-style flow: expert groups -> proposed clusters

All functions write a PNG to `output_path` (matplotlib Agg backend, no display).
All functions are safe with N as small as 2 samples per cluster.

Usage:
    from defect_clustering.clustering.advanced_viz import (
        plot_feature_discriminability, plot_confusion_matrix,
        plot_parallel_coordinates, plot_silhouette_distribution,
        plot_temporal_evolution, plot_cluster_distance_matrix,
        plot_gap_statistic, plot_alluvial_diagram,
    )
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CMAP   = "tab10"
_DARK   = "#212121"
_GREY   = "#546E7A"
_DPI    = 150

# Default feature group slices for defect CNN feature vectors.
# These correspond to typical CNN block outputs + hand-crafted features.
# Override by passing feature_groups= to the relevant functions.
_GROUPS = [
    ("CNN\nBlock 1",  slice(0, 320)),
    ("CNN\nBlock 2",  slice(320, 640)),
    ("CNN\nBlock 3",  slice(640, 960)),
    ("Shape",         slice(960, 984)),
    ("Texture",       slice(984, 1008)),
]


def _cluster_colors(k: int) -> np.ndarray:
    return matplotlib.colormaps[_CMAP].resampled(max(k, 2))(np.linspace(0, 1, k))


def _footer(fig) -> None:
    fig.text(0.5, 0.005,
             "Defect Clustering -- Advanced Analysis",
             ha="center", va="bottom", fontsize=7, color=_GREY)


# ---------------------------------------------------------------------------
# 1. Feature Discriminability (ANOVA F-score -- Fisher 1925)
# ---------------------------------------------------------------------------

def plot_feature_discriminability(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Feature Discriminability -- ANOVA F-score (Fisher 1925)",
    feature_groups: Optional[Sequence[Tuple[str, slice]]] = None,
) -> None:
    """Two-panel figure showing which features drive cluster separation.

    Left:  Mean ANOVA F-score per feature group (bars).
           F = MS_between / MS_within -- high F = group drives separation.
    Right: Top 15 individual features by F-score (horizontal bar chart).

    Args:
        features:       (N, F) float32 feature matrix (StandardScaler space).
        labels:         (N,) int cluster label array.
        output_path:    Where to save the PNG.
        title:          Plot title.
        feature_groups: List of (name, slice) tuples defining feature groups.
                        Falls back to module-level _GROUPS if not provided.
    """
    from scipy.stats import f_oneway

    groups = feature_groups if feature_groups is not None else _GROUPS
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    n_features = features.shape[1]

    # Compute per-feature ANOVA F-score
    f_scores = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        grps = [features[labels == cid, j] for cid in cluster_ids]
        # Need at least 2 samples per group for ANOVA
        if all(len(g) >= 2 for g in grps):
            f_val, _ = f_oneway(*grps)
            f_scores[j] = f_val if np.isfinite(f_val) else 0.0

    # Group-level mean F
    group_names = [g[0] for g in groups]
    group_f = []
    for _, slc in groups:
        indices = range(*slc.indices(n_features))
        group_f.append(float(f_scores[list(indices)].mean()) if len(list(indices)) > 0 else 0.0)

    # Top 15 individual features
    top_k = min(15, n_features)
    top_idx = np.argsort(f_scores)[::-1][:top_k]

    # Map feature index to group name
    def _feat_group(idx):
        for name, slc in groups:
            r = range(*slc.indices(n_features))
            if idx in r:
                return name.split("\n")[1] if "\n" in name else name
        return "?"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.35)

    # Left: group-level F-scores
    _default_colors = ["#1565C0", "#00695C", "#E65100", "#C62828", "#7B1FA2",
                       "#4527A0", "#00838F", "#558B2F", "#BF360C", "#37474F"]
    group_colors = _default_colors[:len(groups)]
    axes[0].bar(group_names, group_f, color=group_colors,
                edgecolor="white", linewidth=1.2)
    for i, v in enumerate(group_f):
        axes[0].text(i, v + max(group_f) * 0.02, f"{v:.1f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Mean ANOVA F-score")
    axes[0].set_title("Cluster Discriminability by Feature Group\n"
                      "F = MS_between / MS_within  --  higher = more discriminative",
                      fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: top 15 individual features
    top_labels = [f"f{idx:03d} ({_feat_group(idx)})" for idx in top_idx]
    top_vals = f_scores[top_idx]
    y_pos = np.arange(top_k)
    bar_colors = [group_colors[
        next((gi for gi, (_, slc) in enumerate(groups)
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
# 2. Confusion matrix (cluster vs expert group)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str,
    expert_col: str = "expert_group",
) -> None:
    """Heatmap of assigned cluster vs expert-labeled ground-truth group.

    Rows = assigned cluster, columns = expert group.
    Annotates with NMI (Normalized Mutual Information) and
    ARI (Adjusted Rand Index) in the subtitle.

    Args:
        labels:      (N,) int cluster label array.
        metadata:    DataFrame aligned with labels, containing expert_col.
        output_path: Where to save the PNG.
        expert_col:  Column in metadata containing ground-truth labels.
    """
    if expert_col not in metadata.columns:
        print(f"[advanced_viz] '{expert_col}' not in metadata -- skipping confusion matrix.")
        return

    from sklearn.metrics import (
        adjusted_rand_score,
        confusion_matrix,
        normalized_mutual_info_score,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    true_labels = metadata[expert_col].values[:len(labels)]
    unique_groups = sorted(set(true_labels))
    unique_clusters = sorted(set(labels))

    # Build confusion matrix: rows = cluster, cols = expert group
    group_to_idx = {p: i for i, p in enumerate(unique_groups)}
    cm = confusion_matrix(
        labels,
        [group_to_idx[p] for p in true_labels],
        labels=unique_clusters,
    )

    nmi = normalized_mutual_info_score(true_labels, labels)
    ari = adjusted_rand_score(true_labels, labels)

    fig, ax = plt.subplots(figsize=(max(6, len(unique_groups) * 1.2),
                                     max(5, len(unique_clusters) * 0.9)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=unique_groups,
        yticklabels=[f"Cluster {c}" for c in unique_clusters],
        ax=ax, linewidths=0.5, linecolor="#E0E0E0",
    )
    ax.set_xlabel("Expert group (ground truth)", fontsize=10)
    ax.set_ylabel("Assigned cluster", fontsize=10)
    ax.set_title(
        f"Cluster vs Expert Group\n"
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
    feature_groups: Optional[Sequence[Tuple[str, slice]]] = None,
) -> None:
    """Parallel coordinates across feature group axes, one line per sample.

    Each sample is a semi-transparent line across axes (one per feature group).
    The value on each axis = mean |feature| within that group, normalized to [0,1].
    Mean per cluster shown as full-opacity thick line.

    Args:
        features:       (N, F) float32 feature matrix.
        labels:         (N,) int cluster label array.
        output_path:    Where to save the PNG.
        n_sample:       Max samples to draw (sampled randomly if N > n_sample).
        feature_groups: List of (name, slice) tuples. Falls back to _GROUPS.
    """
    groups = feature_groups if feature_groups is not None else _GROUPS
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)
    colors = _cluster_colors(k)
    n_groups = len(groups)
    group_names = [g[0].replace("\n", " ") for g in groups]

    # Compute per-sample group means
    group_vals = np.zeros((len(features), n_groups), dtype=np.float64)
    for j, (_, slc) in enumerate(groups):
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

    # Draw individual sample lines (semi-transparent)
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
        "Parallel Coordinates -- Feature Group Profile per Cluster\n"
        "Thin lines = individual samples  /  Thick lines = cluster mean",
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
    """Per-cluster violin of silhouette coefficients -- gold standard diagnostic.

    s(i) = (b - a) / max(a, b)   where
        a = mean intra-cluster distance
        b = mean nearest-cluster distance
    s in [-1, 1]:  s > 0 = well-assigned,  s < 0 = likely mis-assigned.

    One violin per cluster, coloured by cluster.  Red dashed line at s = 0.
    Annotated with n (size) and median silhouette per cluster.

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
        "s(i) = (b-a)/max(a,b)  /  s > 0 = well-assigned  /  s < 0 = mis-assigned",
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
    """Stacked area chart showing cluster distribution over time.

    Detects whether defect production drifts toward a different pattern over
    time (e.g., more surface scratches as tool wear increases).

    Args:
        metadata:      DataFrame aligned with labels (must have timestamp_col).
        labels:        (N,) int cluster label array.
        output_path:   Where to save the PNG.
        timestamp_col: Column containing timestamps (parseable by pd.to_datetime).
        n_bins:        Number of time windows to aggregate into.
    """
    if timestamp_col not in metadata.columns:
        print(f"[advanced_viz] '{timestamp_col}' not in metadata -- skipping temporal evolution.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    df = metadata[: len(labels)].copy()
    df["_cluster"] = labels
    df["_time"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=["_time"])

    if len(df) < 2:
        print("[advanced_viz] Not enough valid timestamps -- skipping temporal evolution.")
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

    # Time axis labels (bin midpoints)
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
    ax.set_xlabel("Date")
    ax.set_ylabel("Sample count per time window")
    ax.set_title(
        "Cluster Distribution Over Time\n"
        "Shift in dominant colour = drift in defect type mix",
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

    # Show only the normalised matrix
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
        f"Closest pair: C{cluster_ids[min_pos[0]]} <-> C{cluster_ids[min_pos[1]]} "
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


# ---------------------------------------------------------------------------
# 8. Alluvial / Sankey Diagram (expert groups -> proposed clusters)
# ---------------------------------------------------------------------------

def plot_alluvial_diagram(
    metadata: pd.DataFrame,
    labels_proposed: np.ndarray,
    expert_col: str = "expert_group",
    output_path: Optional[str] = None,
) -> None:
    """Sankey-style alluvial flow from expert groups to proposed clusters.

    Draws flows using matplotlib only (no plotly dependency). Each flow band
    connects an expert group on the left to a proposed cluster on the right,
    with width proportional to the number of samples in that (group, cluster)
    pair.

    Args:
        metadata:        DataFrame with at least the expert_col column.
        labels_proposed: (N,) int array of proposed cluster labels, aligned
                         with metadata rows.
        expert_col:      Column in metadata for ground-truth groups.
        output_path:     Save path (.png). If None, displays interactively.
    """
    if expert_col not in metadata.columns:
        print(f"[advanced_viz] '{expert_col}' not in metadata -- skipping alluvial diagram.")
        return

    expert_labels = metadata[expert_col].values[:len(labels_proposed)]
    unique_experts = sorted(set(expert_labels))
    unique_clusters = sorted(set(labels_proposed))

    n_left = len(unique_experts)
    n_right = len(unique_clusters)

    # Build flow matrix: flow[i, j] = count of samples in expert_i -> cluster_j
    flow = np.zeros((n_left, n_right), dtype=np.float64)
    expert_to_idx = {e: i for i, e in enumerate(unique_experts)}
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    for ex, cl in zip(expert_labels, labels_proposed):
        flow[expert_to_idx[ex], cluster_to_idx[cl]] += 1

    total = flow.sum()
    if total < 1:
        print("[advanced_viz] No flows to draw -- skipping alluvial diagram.")
        return

    # Layout parameters
    fig_height = max(6, 0.6 * max(n_left, n_right))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    left_x = 0.0
    right_x = 1.0
    bar_width = 0.08
    gap_frac = 0.05  # fraction of total height for gaps

    # Compute vertical positions for left (expert) and right (cluster) bars
    left_sizes = flow.sum(axis=1)   # per expert group
    right_sizes = flow.sum(axis=0)  # per cluster

    def _compute_positions(sizes, gap_frac_val):
        """Compute (y_start, y_end) for each bar, normalized to [0, 1]."""
        total_size = sizes.sum()
        n = len(sizes)
        total_gap = gap_frac_val * n
        usable = 1.0 - total_gap
        positions = []
        y = 0.0
        for s in sizes:
            h = usable * (s / total_size) if total_size > 0 else usable / n
            positions.append((y, y + h))
            y += h + gap_frac_val
        return positions

    left_pos = _compute_positions(left_sizes, gap_frac)
    right_pos = _compute_positions(right_sizes, gap_frac)

    # Colors: one distinct color per expert group
    cmap_left = matplotlib.colormaps["Set2"].resampled(max(n_left, 3))
    cmap_right = matplotlib.colormaps["tab10"].resampled(max(n_right, 3))
    left_colors = [cmap_left(i / max(n_left, 1)) for i in range(n_left)]
    right_colors = [cmap_right(i / max(n_right, 1)) for i in range(n_right)]

    # Draw left bars (expert groups)
    for i, (name, (y0, y1)) in enumerate(zip(unique_experts, left_pos)):
        ax.barh(
            y=(y0 + y1) / 2, width=bar_width, height=y1 - y0,
            left=left_x - bar_width / 2,
            color=left_colors[i], edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        ax.text(left_x - bar_width / 2 - 0.02, (y0 + y1) / 2,
                f"{name} ({int(left_sizes[i])})",
                ha="right", va="center", fontsize=8, fontweight="bold")

    # Draw right bars (clusters)
    for j, (cid, (y0, y1)) in enumerate(zip(unique_clusters, right_pos)):
        ax.barh(
            y=(y0 + y1) / 2, width=bar_width, height=y1 - y0,
            left=right_x - bar_width / 2,
            color=right_colors[j], edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        ax.text(right_x + bar_width / 2 + 0.02, (y0 + y1) / 2,
                f"Cluster {cid} ({int(right_sizes[j])})",
                ha="left", va="center", fontsize=8, fontweight="bold")

    # Draw flow bands (curved ribbons from left bar to right bar)
    # Track cumulative offsets within each bar
    left_offsets = np.array([p[0] for p in left_pos], dtype=np.float64)
    right_offsets = np.array([p[0] for p in right_pos], dtype=np.float64)

    for i in range(n_left):
        for j in range(n_right):
            count = flow[i, j]
            if count < 0.5:
                continue

            # Height of this flow band
            left_total = left_sizes[i]
            right_total = right_sizes[j]
            left_h = (left_pos[i][1] - left_pos[i][0]) * (count / left_total) if left_total > 0 else 0
            right_h = (right_pos[j][1] - right_pos[j][0]) * (count / right_total) if right_total > 0 else 0

            # Start/end y positions
            y_left_bottom = left_offsets[i]
            y_left_top = y_left_bottom + left_h
            y_right_bottom = right_offsets[j]
            y_right_top = y_right_bottom + right_h

            left_offsets[i] = y_left_top
            right_offsets[j] = y_right_top

            # Create a smooth curved band using fill_between on a bezier-like path
            x_left_edge = left_x + bar_width / 2
            x_right_edge = right_x - bar_width / 2
            n_pts = 50
            xs = np.linspace(x_left_edge, x_right_edge, n_pts)

            # Sigmoid-like interpolation for smooth curves
            t = (xs - x_left_edge) / (x_right_edge - x_left_edge)
            # Smooth step: 3t^2 - 2t^3 (Hermite basis)
            s = 3 * t**2 - 2 * t**3

            y_bottom = y_left_bottom + s * (y_right_bottom - y_left_bottom)
            y_top = y_left_top + s * (y_right_top - y_left_top)

            # Use the expert group color with transparency
            ax.fill_between(xs, y_bottom, y_top,
                            color=left_colors[i], alpha=0.35, linewidth=0,
                            zorder=2)
            # Thin edge lines for readability
            ax.plot(xs, y_bottom, color=left_colors[i], alpha=0.5, lw=0.3, zorder=2)
            ax.plot(xs, y_top, color=left_colors[i], alpha=0.5, lw=0.3, zorder=2)

    # Formatting
    ax.set_xlim(-0.35, 1.35)
    y_max = max(p[1] for p in left_pos + right_pos) + 0.02
    ax.set_ylim(-0.02, y_max)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["Expert Groups", "Proposed Clusters"], fontsize=11,
                       fontweight="bold")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_title(
        "Alluvial Diagram: Expert Groups -> Proposed Clusters\n"
        "Band width proportional to sample count in each (group, cluster) pair",
        fontsize=11, fontweight="bold", color=_DARK, pad=15,
    )

    _footer(fig)
    fig.tight_layout()

    if output_path:
        from pathlib import Path as _Path
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[advanced_viz] Alluvial diagram saved: {output_path}")
    else:
        plt.show()

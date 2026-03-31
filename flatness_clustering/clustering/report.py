"""
Flatness Clustering PDF Report Generator
=========================================

Generates a multi-page PDF report summarising a complete flatness clustering
run (features, cluster quality, visualisations, per-cluster profiles).

Pages:
    1. Cover page         — title, run metadata, quality metric cards (incl. NMI/ARI/Purity)
    2. Cluster quality    — Silhouette / Davies-Bouldin / Calinski-Harabász
                            with elbow curve (inertia + silhouette overlay)
    3. Cluster heatmaps   — representative flatness profiles per cluster
    4. Discriminability   — feature importance + confusion patterns
    5. Ground truth confusion — pattern × cluster matrix with per-cluster purity
    6. Cluster summary    — per-cluster table (size, dominant pattern, entropy…)
    7. Feature overview   — Group A/B/C/D/E contribution bar + feature norm per cluster
    8. UMAP embedding     — cluster labels vs ground-truth patterns
    9. Silhouette distribution — per-cluster silhouette violin plots
   10. DEC soft-assignment heatmap — coil × cluster probability matrix
   11. Metric trend table — silhouette/DB/CH across tested k values
   12. Gap statistic      — Tibshirani et al. 2001
   13. Algorithm comparison — multi-method performance comparison

Uses only matplotlib (PdfPages) — no external PDF library required.

Usage
-----
    from flatness_clustering.clustering.report import generate_clustering_report

    generate_clustering_report(
        features=feature_matrix,       # (N, 125) float32
        labels=cluster_labels,         # (N,) int
        metadata=metadata_df,          # DataFrame with coil_id, pattern, …
        profiles=profiles_array,       # (N, W, L) float32  (optional)
        clusterer=coil_clusterer,      # fitted CoilClusterer  (optional)
        inertia_data=inertia_dict,     # {k: {inertia, silhouette, db, ch}}  (optional)
        tsne_2d=tsne_embeddings,       # (N, 2)  (optional)
        output_path="outputs/reports/clustering_report.pdf",
        n_clusters=6,
    )
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from flatness_clustering.clustering.metrics import (
    compute_calinski_harabasz,
    compute_davies_bouldin,
    compute_silhouette,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clustering_report(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str = "flatness_clustering/outputs/reports/clustering_report.pdf",
    profiles: Optional[np.ndarray] = None,
    clusterer=None,
    inertia_data: Optional[Dict] = None,
    tsne_2d: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    run_tag: str = "",
    algorithm_comparison_path: Optional[str] = None,
    gap_result: Optional[Dict] = None,
) -> str:
    """Generate a multi-page PDF report for a flatness clustering run.

    Args:
        features:      (N, F) feature matrix used for clustering.
        labels:        (N,) cluster label array.
        metadata:      DataFrame with columns: coil_id, pattern, grade, customer.
        output_path:   Where to save the PDF.
        profiles:      (N, W, L) raw flatness profiles — used for heatmap page.
        clusterer:     Fitted CoilClusterer — used for soft-assignment page.
        inertia_data:  {k: {"inertia", "silhouette", "db", "ch"}} from inertia_curve().
        tsne_2d:       (N, 2) pre-computed t-SNE embedding.
        n_clusters:    Number of clusters (derived from labels if not given).
        run_tag:       Optional tag added to the cover page (e.g. "run-2024-01").

    Returns:
        Absolute path to the saved PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    k = n_clusters or int(labels.max()) + 1

    # Compute StandardScaler-normalised features for all within-report metrics
    # and visualisations.  Raw `features` is kept only for predict_with_confidence
    # calls (clusterer scales internally to avoid double-scaling).
    if clusterer is not None and hasattr(clusterer, "_scaler"):
        features_sc = clusterer._scaler.transform(
            features.astype(np.float64)
        ).astype(np.float32)
    else:
        features_sc = features  # fallback: no scaler available

    sil = compute_silhouette(features_sc, labels)
    db  = compute_davies_bouldin(features_sc, labels)
    ch  = compute_calinski_harabasz(features_sc, labels)

    with PdfPages(output_path) as pdf:
        _page_cover(pdf, features_sc, labels, metadata, k, sil, db, ch, run_tag)
        _page_quality_metrics(pdf, features_sc, labels, sil, db, ch, inertia_data, k)
        if profiles is not None:
            _page_heatmaps(pdf, profiles, labels, metadata)
        _page_discriminability_confusion(pdf, features_sc, labels, metadata)
        if "class_id" in metadata.columns:
            _page_ground_truth_confusion(pdf, labels, metadata, k)
        _page_cluster_summary(pdf, features_sc, labels, metadata, sil, db, ch)
        _page_feature_overview(pdf, features, features_sc, labels, k)
        _page_umap(pdf, features_sc, labels, metadata)
        _page_silhouette_distribution(pdf, features_sc, labels)
        _page_parallel_distance(pdf, features_sc, labels)
        if clusterer is not None and hasattr(clusterer, "predict_with_confidence"):
            _page_soft_assignments(pdf, features, labels, clusterer, k)
        if inertia_data is not None:
            _page_metric_trends(pdf, inertia_data, k)
        if gap_result is not None:
            _page_gap_statistic(pdf, gap_result, k)
        if algorithm_comparison_path and os.path.exists(algorithm_comparison_path):
            _page_algorithm_comparison(pdf, algorithm_comparison_path)

        d = pdf.infodict()
        d["Title"]   = f"Flatness Clustering Report — k={k}"
        d["Subject"] = "Steel Flatness Profile Clustering"
        d["Creator"] = "flatness_clustering.clustering.report"

    print(f"[report] Clustering PDF saved: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE   = "#1565C0"
_TEAL   = "#00695C"
_RED    = "#C62828"
_ORANGE = "#E65100"
_AMBER  = "#F9A825"
_GREY   = "#546E7A"
_LIGHT  = "#ECEFF1"
_DARK   = "#212121"
_CLUSTER_CMAP = "tab10"


def _cluster_colors(k: int):
    return plt.cm.get_cmap(_CLUSTER_CMAP, max(k, 2))(np.linspace(0, 1, k))


def _header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=13, fontweight="bold", color=_DARK)
    if subtitle:
        fig.text(0.5, 0.942, subtitle, ha="center", va="top",
                 fontsize=8.5, color=_GREY)


def _footer(fig) -> None:
    fig.text(0.5, 0.008,
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
             "Flatness Clustering — Saas et al. (2017) + Asadi & Regan (2021)",
             ha="center", va="bottom", fontsize=7, color=_GREY)


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _page_cover(pdf, features, labels, metadata, k, sil, db, ch, run_tag):
    fig = plt.figure(figsize=(8.27, 11.69))
    _footer(fig)

    fig.text(0.5, 0.90, "Flatness Clustering Report",
             ha="center", fontsize=22, fontweight="bold", color=_DARK)
    fig.text(0.5, 0.855, "Steel Surface Flatness Profiles",
             ha="center", fontsize=13, color=_BLUE)
    if run_tag:
        fig.text(0.5, 0.825, run_tag, ha="center", fontsize=10, color=_GREY)
    fig.text(0.5, 0.800, datetime.now().strftime("%d %B %Y  •  %H:%M"),
             ha="center", fontsize=10, color=_GREY)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.77, 0.77],
                               transform=fig.transFigure, color=_GREY, lw=0.8))

    # 3 metric cards
    card_data = [
        ("Silhouette Score",       f"{sil:.4f}",
         "↑ higher is better", _TEAL if sil > 0.3 else _AMBER),
        ("Davies-Bouldin Index",   f"{db:.4f}",
         "↓ lower is better",  _TEAL if db < 1.5 else _AMBER),
        ("Calinski-Harabász",      f"{ch:.1f}",
         "↑ higher is better", _TEAL if ch > 100 else _AMBER),
    ]
    for i, (title, val, note, color) in enumerate(card_data):
        x = 0.08 + i * 0.30
        ax = fig.add_axes([x, 0.64, 0.24, 0.10])
        ax.set_facecolor(color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.text(0.5, 0.75, title, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.32, val, ha="center", va="center",
                fontsize=14, color="white", fontweight="bold", transform=ax.transAxes)
        fig.text(x + 0.12, 0.630, note, ha="center", fontsize=7.5, color=_GREY)

    # Ground-truth metrics (NMI, ARI, Purity) — second row of cards
    if "class_id" in metadata.columns:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        gt = metadata["class_id"].values
        nmi = normalized_mutual_info_score(gt, labels)
        ari = adjusted_rand_score(gt, labels)
        # Purity
        correct = 0
        for c in sorted(set(labels)):
            mask = labels == c
            if mask.sum() > 0:
                correct += np.bincount(gt[mask]).max()
        purity = correct / len(labels)

        gt_cards = [
            ("NMI",    f"{nmi:.3f}", _TEAL if nmi >= 0.8 else _AMBER if nmi >= 0.6 else _RED),
            ("ARI",    f"{ari:.3f}", _TEAL if ari >= 0.6 else _AMBER if ari >= 0.3 else _RED),
            ("Purity", f"{purity:.1%}", _TEAL if purity >= 0.7 else _AMBER if purity >= 0.5 else _RED),
        ]
        for i, (title, val, color) in enumerate(gt_cards):
            x = 0.08 + i * 0.30
            ax = fig.add_axes([x, 0.56, 0.24, 0.065])
            ax.set_facecolor(color)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.text(0.5, 0.72, title, ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.28, val, ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold", transform=ax.transAxes)

    # Dataset summary
    n = len(labels)
    unique_labels = sorted(set(labels))
    cluster_sizes = [int((labels == c).sum()) for c in unique_labels]

    y_summary = 0.50 if "class_id" in metadata.columns else 0.58
    fig.text(0.5, y_summary, "Dataset Summary",
             ha="center", fontsize=10, fontweight="bold", color=_DARK)
    summary = [
        ("Total coils",     str(n)),
        ("Feature dim",     str(features.shape[1])),
        ("Clusters (k)",    str(k)),
        ("Smallest cluster",f"{min(cluster_sizes)} coils"),
        ("Largest cluster", f"{max(cluster_sizes)} coils"),
        ("Patterns",        str(metadata["pattern"].nunique()) if "pattern" in metadata.columns else "N/A"),
    ]
    y_items = y_summary - 0.04
    for idx, (key, val) in enumerate(summary):
        col = idx % 3
        row = idx // 3
        fig.text(0.12 + col * 0.27, y_items - row * 0.04,
                 f"{key}:  {val}", ha="left", fontsize=9, color=_DARK)

    y_line = y_items - 0.10
    fig.add_artist(plt.Line2D([0.08, 0.92], [y_line, y_line],
                               transform=fig.transFigure, color=_GREY, lw=0.5))

    # Cluster size mini bar chart
    ax2 = fig.add_axes([0.10, 0.20, 0.80, 0.16])
    colors = _cluster_colors(k)
    ax2.bar([str(c) for c in unique_labels], cluster_sizes,
            color=colors, edgecolor="white", linewidth=1)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Coil count")
    ax2.set_title("Cluster Size Distribution", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Reference
    fig.text(0.5, 0.11,
             "Clustering quality: Saas et al. (2017) — silhouette, Davies-Bouldin, Calinski-Harabász\n"
             "Soft assignment:    Asadi & Regan (2021) — DEC Student-t kernel\n"
             "Features:           105 statistical + 1000 ROCKET (Group E, 500 kernels, Dempster 2020)",
             ha="center", fontsize=8, color=_GREY, linespacing=1.8, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — Quality metrics + elbow
# ---------------------------------------------------------------------------

def _page_quality_metrics(pdf, features, labels, sil, db, ch, inertia_data, k):
    has_elbow = inertia_data is not None and len(inertia_data) >= 2
    n_rows = 2 if has_elbow else 1
    fig, axes = plt.subplots(n_rows, 3 if not has_elbow else 1,
                              figsize=(8.27, 11.69 if has_elbow else 5.5))
    _header(fig, "Cluster Quality Metrics",
            "Saas et al. (2017) — three complementary cluster evaluation indices")
    _footer(fig)
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.40, wspace=0.35)

    if not has_elbow:
        ax_sil, ax_db, ax_ch = axes

        # Silhouette
        mask = labels >= 0
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            from sklearn.metrics import silhouette_samples
            sil_vals = silhouette_samples(features[mask], labels[mask])
            cluster_ids = sorted(set(labels[mask]))
            colors = _cluster_colors(len(cluster_ids))
            y_pos = 0
            for i, cid in enumerate(cluster_ids):
                c_vals = np.sort(sil_vals[labels[mask] == cid])
                ax_sil.barh(range(y_pos, y_pos + len(c_vals)), c_vals,
                            color=colors[i], alpha=0.8, height=1.0)
                y_pos += len(c_vals) + 2
            ax_sil.axvline(sil, color="red", linestyle="--", lw=1.5,
                           label=f"mean = {sil:.4f}")
        ax_sil.set_title(f"Silhouette\n{sil:.4f}  (↑ better)", fontsize=9)
        ax_sil.set_xlabel("Silhouette coefficient")
        ax_sil.legend(fontsize=8)

        # DB bar per cluster
        from sklearn.metrics import davies_bouldin_score
        ax_db.bar(["DB Index"], [db], color=_AMBER if db > 1.5 else _TEAL, width=0.4)
        ax_db.set_title(f"Davies-Bouldin\n{db:.4f}  (↓ better)", fontsize=9)
        ax_db.set_ylim(0, max(db * 1.5, 0.5))
        ax_db.grid(axis="y", alpha=0.3)

        # CH bar
        ax_ch.bar(["CH Score"], [ch], color=_TEAL if ch > 100 else _AMBER, width=0.4)
        ax_ch.set_title(f"Calinski-Harabász\n{ch:.1f}  (↑ better)", fontsize=9)
        ax_ch.grid(axis="y", alpha=0.3)

    else:
        # Top row: three metric boxes
        fig2, axes2 = plt.subplots(1, 3, figsize=(8.27, 2.5))
        for ax_m, (title, val, note, color) in zip(axes2, [
            ("Silhouette",       f"{sil:.4f}", "↑ better", _TEAL if sil > 0.3 else _AMBER),
            ("Davies-Bouldin",   f"{db:.4f}",  "↓ better", _TEAL if db < 1.5 else _AMBER),
            ("Calinski-Harabász",f"{ch:.1f}", "↑ better", _TEAL if ch > 100 else _AMBER),
        ]):
            ax_m.bar([title], [1], color=color, width=0.5)
            ax_m.set_ylim(0, 1.6)
            ax_m.set_yticks([])
            ax_m.text(0, 0.9, val, ha="center", va="bottom",
                      fontsize=14, fontweight="bold", color="white")
            ax_m.text(0, 1.25, note, ha="center", va="bottom",
                      fontsize=8, color=_GREY)
            ax_m.set_title(title, fontsize=9)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Elbow page
        _page_elbow_from_data(pdf, inertia_data, k, sil, db, ch)
        plt.close(fig)
        return

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_elbow_from_data(pdf, inertia_data, chosen_k, sil, db, ch):
    ks = sorted(inertia_data.keys())
    first = inertia_data[ks[0]]
    new_fmt = isinstance(first, dict)

    inertia_vals = [inertia_data[k]["inertia"] if new_fmt else inertia_data[k] for k in ks]
    sil_vals  = [inertia_data[k].get("silhouette", 0) for k in ks] if new_fmt else None
    db_vals   = [inertia_data[k].get("db", 0) for k in ks] if new_fmt else None
    ch_vals   = [inertia_data[k].get("ch", 0) for k in ks] if new_fmt else None

    has_extra = new_fmt and sil_vals is not None

    nrows = 2 if has_extra else 1
    fig, axes = plt.subplots(nrows, 1 if not has_extra else 2,
                              figsize=(8.27, 8 if has_extra else 4.5))
    _header(fig, "Elbow Curve + Multi-Metric k Selection",
            "Saas et al. (2017) — inertia (elbow) + silhouette / DB / CH overlays")
    _footer(fig)
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.45, wspace=0.40)

    if has_extra:
        ax_el, ax_sil = axes[0]
        ax_db, ax_ch  = axes[1]
    else:
        ax_el = axes if nrows == 1 else axes[0]

    # Inertia
    ax_el.plot(ks, inertia_vals, "o-", color=_BLUE, lw=2, markersize=6, label="Inertia")
    if chosen_k in inertia_data:
        ax_el.axvline(chosen_k, color=_RED, linestyle="--", lw=1.5,
                      label=f"chosen k={chosen_k}")
    ax_el.set_xlabel("k"); ax_el.set_ylabel("Inertia")
    ax_el.set_title("Elbow Curve"); ax_el.legend(fontsize=8)
    ax_el.set_xticks(ks); ax_el.grid(alpha=0.3)

    if has_extra:
        ax_sil.plot(ks, sil_vals, "s--", color=_TEAL, lw=1.5, markersize=5)
        if chosen_k in inertia_data:
            ax_sil.axvline(chosen_k, color=_RED, linestyle="--", lw=1)
        ax_sil.set_xlabel("k"); ax_sil.set_ylabel("Silhouette")
        ax_sil.set_title("Silhouette vs k  (↑ better)")
        ax_sil.set_xticks(ks); ax_sil.grid(alpha=0.3)

        ax_db.plot(ks, db_vals, "^-", color=_AMBER, lw=1.5, markersize=5)
        if chosen_k in inertia_data:
            ax_db.axvline(chosen_k, color=_RED, linestyle="--", lw=1)
        ax_db.set_xlabel("k"); ax_db.set_ylabel("Davies-Bouldin")
        ax_db.set_title("Davies-Bouldin vs k  (↓ better)")
        ax_db.set_xticks(ks); ax_db.grid(alpha=0.3)

        ax_ch.plot(ks, ch_vals, "D-", color=_BLUE, lw=1.5, markersize=5)
        if chosen_k in inertia_data:
            ax_ch.axvline(chosen_k, color=_RED, linestyle="--", lw=1)
        ax_ch.set_xlabel("k"); ax_ch.set_ylabel("Calinski-Harabász")
        ax_ch.set_title("Calinski-Harabász vs k  (↑ better)")
        ax_ch.set_xticks(ks); ax_ch.grid(alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3 — Cluster summary table
# ---------------------------------------------------------------------------

def _page_cluster_summary(pdf, features, labels, metadata, sil, db, ch):
    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Per-Cluster Summary Table",
            "Cluster composition, dominant patterns, and quality metrics")
    _footer(fig)

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78])
    ax.axis("off")

    cluster_ids = sorted(set(labels))
    colors = _cluster_colors(len(cluster_ids))

    # Header
    headers = ["Cluster", "Size", "% Total", "Dominant pattern",
               "Pattern entropy", "Top grade", "‖z‖₂ (scaled)"]
    col_x = [0.00, 0.09, 0.17, 0.27, 0.50, 0.63, 0.77]
    y = 0.96
    row_h = 0.058

    for hdr, x in zip(headers, col_x):
        ax.text(x, y, hdr, fontsize=8.5, fontweight="bold", color=_DARK,
                transform=ax.transAxes)
    y -= 0.01
    ax.plot([0, 1], [y, y], color=_GREY, lw=0.6, transform=ax.transAxes)
    y -= 0.02

    n = len(labels)
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        subset = metadata[mask] if len(metadata) == len(labels) else metadata.iloc[:0]

        size = int(mask.sum())
        pct  = size / n

        dominant = (subset["pattern"].value_counts().idxmax()
                    if "pattern" in subset.columns and len(subset) > 0 else "—")
        counts = (subset["pattern"].value_counts(normalize=True)
                  if "pattern" in subset.columns and len(subset) > 0
                  else pd.Series([1.0]))
        entropy = float(-(counts * np.log(counts + 1e-9)).sum())

        top_grade = (subset["grade"].value_counts().idxmax()
                     if "grade" in subset.columns and len(subset) > 0 else "—")

        feat_norm = float(np.linalg.norm(features[mask], axis=1).mean())

        row_vals = [str(cid), str(size), f"{pct:.1%}", dominant,
                    f"{entropy:.3f}", top_grade, f"{feat_norm:.2f}"]

        bg_color = colors[i]
        # colour swatch
        ax.add_patch(plt.Rectangle((col_x[0] - 0.005, y - 0.035), 0.008, 0.045,
                                    transform=ax.transAxes,
                                    facecolor=bg_color, edgecolor="none"))

        for val, x in zip(row_vals, col_x):
            ax.text(x + (0.010 if x == col_x[0] else 0), y, val,
                    fontsize=8, color=_DARK, transform=ax.transAxes, va="top")

        ax.plot([0, 1], [y - row_h + 0.015, y - row_h + 0.015], color="#E0E0E0", lw=0.4, transform=ax.transAxes)
        y -= row_h

    # Footer metrics
    y -= 0.02
    ax.plot([0, 1], [y + 0.01, y + 0.01], color=_GREY, lw=0.6, transform=ax.transAxes)
    ax.text(0.0, y - 0.02,
            f"Overall:  Silhouette = {sil:.4f}   "
            f"Davies-Bouldin = {db:.4f}   "
            f"Calinski-Harabász = {ch:.1f}",
            fontsize=8.5, color=_BLUE, transform=ax.transAxes, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4 — Feature group overview
# ---------------------------------------------------------------------------

def _page_feature_overview(pdf, features, features_sc, labels, k):  # noqa: ARG001
    """Feature overview page.

    Args:
        features:    Raw (N, F) feature matrix — used for group bar chart so
                     relative raw magnitudes are preserved across groups.
        features_sc: StandardScaler-normalised (N, F) — used for L2-norm per
                     cluster so norms are on a meaningful scale (~√F ≈ 11).
        labels:      (N,) cluster label array.
        k:           Number of clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 5.5))
    _header(fig, "Feature Overview",
            "Feature group contributions (raw magnitude) and per-cluster norms (scaled)")
    _footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.38)

    # Group importance: mean |feature| per group on raw features
    # Shows relative energy contribution of each feature group before normalisation
    groups = [
        ("Group A\nZonal stats",      slice(0, 30)),
        ("Group B\nSnapshots",        slice(30, 80)),
        ("Group C\nSpectral",         slice(80, 95)),
        ("Group D\nTemporal",         slice(95, 105)),
        ("Group E\nROCKET",           slice(105, 135)),
    ]
    group_names = [g[0] for g in groups]
    group_means = [float(np.abs(features[:, g[1]]).mean()) for g in groups]
    total = sum(group_means) + 1e-12
    group_pcts = [v / total * 100 for v in group_means]
    group_colors = [_BLUE, _TEAL, _ORANGE, _RED, "#7B1FA2"]

    axes[0].bar(group_names, group_pcts, color=group_colors,
                edgecolor="white", linewidth=1.2)
    for i, v in enumerate(group_pcts):
        axes[0].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].set_title("Feature Group Energy Share\n(raw magnitude, 5 groups / 125 dims)", fontsize=9)
    axes[0].set_ylabel("% of total mean |feature|")
    axes[0].set_ylim(0, max(group_pcts) * 1.2)
    axes[0].grid(axis="y", alpha=0.3)

    # Per-cluster feature norm on SCALED features (meaningful scale: ~√F ≈ 11)
    cluster_ids = sorted(set(labels))
    cluster_norms = [float(np.linalg.norm(features_sc[labels == c], axis=1).mean())
                     for c in cluster_ids]
    cluster_colors = _cluster_colors(len(cluster_ids))
    axes[1].bar([f"C{c}" for c in cluster_ids], cluster_norms,
                color=cluster_colors, edgecolor="white", linewidth=1)
    for i, v in enumerate(cluster_norms):
        axes[1].text(i, v + max(cluster_norms) * 0.02, f"{v:.2f}",
                     ha="center", fontsize=9)
    axes[1].set_title("Mean Feature L2-Norm per Cluster\n(StandardScaler space — expected ≈ √F)", fontsize=9)
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Mean ‖z‖₂  (scaled)")
    axes[1].grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page — UMAP embedding (McInnes et al. 2018)
# ---------------------------------------------------------------------------

def _page_umap(pdf, features_sc, labels, metadata):
    """Dual-panel UMAP: cluster labels (left) + ground truth (right)."""
    try:
        import umap
    except ImportError:
        # Skip page if umap-learn not installed
        return

    has_gt = "class_id" in metadata.columns or "pattern" in metadata.columns
    ncols = 2 if has_gt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8.27 * (ncols / 1.3), 5.5))
    _header(fig, "UMAP Embedding (McInnes et al. 2018)",
            "Preserves global + local structure  ·  StandardScaler feature space")
    _footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.35)

    if ncols == 1:
        axes = [axes]

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(features_sc)

    k = len(set(labels))
    cmap = plt.cm.get_cmap(_CLUSTER_CMAP, max(k, 2))

    axes[0].scatter(emb[:, 0], emb[:, 1], c=labels,
                    cmap=cmap, s=12, alpha=0.7, linewidths=0)
    axes[0].set_title("Cluster labels", fontsize=9)
    axes[0].set_xlabel("UMAP dim 1"); axes[0].set_ylabel("UMAP dim 2")

    if has_gt:
        gt_col = "class_id" if "class_id" in metadata.columns else "pattern"
        gt = metadata[gt_col].values[:len(emb)]
        if gt.dtype.kind in ("U", "O"):
            unique_gt = sorted(set(gt))
            gt_int = np.array([unique_gt.index(v) for v in gt])
        else:
            gt_int = gt.astype(int)
        k_gt = len(set(gt_int))
        cmap_gt = plt.cm.get_cmap("Set2", max(k_gt, 2))
        axes[1].scatter(emb[:, 0], emb[:, 1], c=gt_int,
                        cmap=cmap_gt, s=12, alpha=0.7, linewidths=0)
        axes[1].set_title(f"Ground truth ({gt_col})", fontsize=9)
        axes[1].set_xlabel("UMAP dim 1"); axes[1].set_ylabel("UMAP dim 2")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page — Silhouette Distribution (Rousseeuw 1987)
# ---------------------------------------------------------------------------

def _page_silhouette_distribution(pdf, features_sc, labels):
    """Per-cluster silhouette violin — gold standard cluster diagnostic."""
    from sklearn.metrics import silhouette_samples

    fig, ax = plt.subplots(figsize=(8.27, 5.5))
    _header(fig, "Per-Cluster Silhouette Distribution (Rousseeuw 1987)",
            "s(i) = (b−a)/max(a,b)  ·  s > 0 = well-assigned  ·  s < 0 = mis-assigned")
    _footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.14)

    cluster_ids = sorted(set(labels))
    k = len(cluster_ids)
    colors = _cluster_colors(k)

    sil_vals = silhouette_samples(features_sc, labels)
    global_mean = float(sil_vals.mean())

    data = [sil_vals[labels == cid] for cid in cluster_ids]
    parts = ax.violinplot(data, positions=range(k), showmedians=True, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.65)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(_DARK)
            parts[key].set_linewidth(1.2)

    ax.axhline(0, color=_DARK, linestyle="-", lw=1.0, alpha=0.6, label="s = 0")
    ax.axhline(global_mean, color="red", linestyle="--", lw=1.5,
               label=f"global mean = {global_mean:.3f}")

    for i, (cid, d) in enumerate(zip(cluster_ids, data)):
        med = float(np.median(d))
        ax.text(i, max(d.max(), 0) + 0.04,
                f"n={len(d)}\nmed={med:.3f}", ha="center", fontsize=7.5,
                color=_GREY, va="bottom")

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
    ax.set_ylabel("Silhouette coefficient s(i)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 6 — Cluster heatmaps
# ---------------------------------------------------------------------------

def _page_heatmaps(pdf, profiles, labels, metadata):
    unique_clusters = sorted(c for c in set(labels) if c >= 0)
    n_clusters = len(unique_clusters)
    n_samples   = 3

    fig, axes = plt.subplots(n_clusters, n_samples,
                              figsize=(8.27, max(3.0, n_clusters * 2.2)),
                              squeeze=False)
    _header(fig, "Representative Flatness Profiles per Cluster",
            "Red = positive I-units (stretched)  ·  Blue = negative (compressed)")
    _footer(fig)
    fig.subplots_adjust(top=0.90, bottom=0.05, hspace=0.5, wspace=0.05)

    pattern_map: dict = {}
    if "pattern" in metadata.columns and len(metadata) == len(labels):
        for cid in unique_clusters:
            mask = labels == cid
            vc = metadata.loc[mask, "pattern"].value_counts()
            pattern_map[cid] = vc.idxmax() if len(vc) > 0 else "?"

    for row, cid in enumerate(unique_clusters):
        cid_mask = np.where(labels == cid)[0]
        sample_idx = cid_mask[:n_samples]
        c_profiles = profiles[cid_mask]
        vmin = float(np.percentile(c_profiles, 2))
        vmax = float(np.percentile(c_profiles, 98))

        for col in range(n_samples):
            ax = axes[row][col]
            if col < len(sample_idx):
                p = profiles[sample_idx[col]]
                disp = p[::max(1, p.shape[0]//80), ::max(1, p.shape[1]//160)]
                ax.imshow(disp, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    label = pattern_map.get(cid, f"Cluster {cid}")
                    ax.set_ylabel(f"C{cid}\n{label}", fontsize=7, labelpad=3)
            else:
                ax.set_visible(False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 7 — DEC soft assignment heatmap
# ---------------------------------------------------------------------------

def _page_soft_assignments(pdf, features, labels, clusterer, k):
    n_show = min(40, len(features))
    idx = np.random.default_rng(42).choice(len(features), n_show, replace=False)
    idx = idx[np.argsort(labels[idx])]   # sort by cluster label

    soft_matrix = np.zeros((n_show, k), dtype=np.float32)
    for i, fi in enumerate(idx):
        r = clusterer.predict_with_confidence(features[fi])
        q = r.get("soft_assignments", np.zeros(k))
        soft_matrix[i, :len(q)] = q[:k]

    fig, ax = plt.subplots(figsize=(8.27, 7))
    _header(fig, "DEC Soft Assignment Probabilities",
            "Asadi & Regan (2021) — Student-t kernel   q_ij = (1+d²)^{-1} / Σ(1+d²)^{-1}")
    _footer(fig)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.92)

    im = ax.imshow(soft_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Soft assignment probability q_ij")

    ax.set_xlabel("Cluster j")
    ax.set_ylabel(f"Coil index (n={n_show}, sorted by assigned cluster)")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{j}" for j in range(k)], fontsize=9)
    ax.set_title(f"Soft assignment heat-map — {n_show} coils × {k} clusters\n"
                 "Each row sums to 1.0  (brighter = higher probability)", fontsize=9)

    # Overlay argmax cluster borders
    for i in range(n_show):
        best = int(soft_matrix[i].argmax())
        ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="navy", lw=0.8))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 8 — Metric trend table across k values
# ---------------------------------------------------------------------------

def _page_metric_trends(pdf, inertia_data, chosen_k):
    ks = sorted(inertia_data.keys())
    first = inertia_data[ks[0]]
    new_fmt = isinstance(first, dict)

    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Cluster Quality Trends Across k",
            "All metrics evaluated for each candidate k — Saas et al. (2017)")
    _footer(fig)

    ax = fig.add_axes([0.06, 0.10, 0.88, 0.74])
    ax.axis("off")

    headers = ["k", "Inertia", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabász ↑", "Chosen"]
    col_x   = [0.0, 0.10,      0.28,            0.46,                0.64,                   0.86]
    y = 0.96; row_h = 0.055

    for hdr, x in zip(headers, col_x):
        ax.text(x, y, hdr, fontsize=9, fontweight="bold", color=_DARK, transform=ax.transAxes)
    y -= 0.01
    ax.plot([0, 1], [y, y], color=_GREY, lw=0.6, transform=ax.transAxes)
    y -= 0.02

    for k in ks:
        entry = inertia_data[k]
        inertia_v = entry["inertia"] if new_fmt else entry
        sil_v  = entry.get("silhouette", 0) if new_fmt else "—"
        db_v   = entry.get("db", 0)         if new_fmt else "—"
        ch_v   = entry.get("ch", 0)         if new_fmt else "—"
        chosen = "✓  ←" if k == chosen_k else ""

        row_bg = _LIGHT if k == chosen_k else "white"
        vals = [str(k),
                f"{inertia_v:.1f}",
                f"{sil_v:.4f}" if isinstance(sil_v, float) else sil_v,
                f"{db_v:.4f}"  if isinstance(db_v, float)  else db_v,
                f"{ch_v:.1f}"  if isinstance(ch_v, float)  else ch_v,
                chosen]

        for val, x in zip(vals, col_x):
            color = _BLUE if k == chosen_k else _DARK
            fw    = "bold" if k == chosen_k else "normal"
            ax.text(x, y, val, fontsize=9, color=color, fontweight=fw,
                    transform=ax.transAxes, va="top")

        ax.plot([0, 1], [y - row_h + 0.012, y - row_h + 0.012], color="#E0E0E0", lw=0.4, transform=ax.transAxes)
        y -= row_h

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page — Feature Discriminability + Confusion Matrix
# ---------------------------------------------------------------------------

def _page_discriminability_confusion(pdf, features_sc, labels, metadata):
    """Embed ANOVA discriminability (left) and cluster confusion matrix (right)."""
    from flatness_clustering.clustering.advanced_viz import (
        plot_feature_discriminability,
        plot_confusion_matrix as adv_plot_confusion,
    )

    with tempfile.TemporaryDirectory() as tmp:
        disc_path = os.path.join(tmp, "discriminability.png")
        conf_path = os.path.join(tmp, "confusion.png")
        plot_feature_discriminability(features_sc, labels, output_path=disc_path)
        adv_plot_confusion(labels, metadata, output_path=conf_path)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.88, bottom=0.05, wspace=0.1)
        _header(fig, "Feature Discriminability (ANOVA F) & Cluster Purity",
                "Left: ANOVA F-score per feature group (Fisher 1925)  |  "
                "Right: cluster vs ground-truth pattern (NMI / ARI)")
        _footer(fig)

        for ax, img_path in zip(axes, [disc_path, conf_path]):
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img, aspect="equal")
            ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 10 — Parallel coordinates + cluster distance matrix
# ---------------------------------------------------------------------------

def _page_parallel_distance(pdf, features, labels):
    """Embed parallel coordinates (left) and inter-cluster distance matrix (right)."""
    from flatness_clustering.clustering.advanced_viz import (
        plot_parallel_coordinates,
        plot_cluster_distance_matrix,
    )

    with tempfile.TemporaryDirectory() as tmp:
        pc_path   = os.path.join(tmp, "parallel.png")
        dist_path = os.path.join(tmp, "distance.png")
        plot_parallel_coordinates(features, labels, output_path=pc_path)
        plot_cluster_distance_matrix(features, labels, output_path=dist_path)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.88, bottom=0.05, wspace=0.1)
        _header(fig, "Parallel Coordinates & Inter-cluster Distance Matrix",
                "Parallel coords: per-coil feature-group means, coloured by cluster  |  "
                "Distance matrix: centroid Euclidean distances")
        _footer(fig)

        for ax, img_path in zip(axes, [pc_path, dist_path]):
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img, aspect="equal")
            ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_gap_statistic(pdf, gap_result: Dict, chosen_k: int) -> None:
    """Gap Statistic page (Tibshirani, Walther & Hastie 2001)."""
    ks = gap_result["k_range_tested"]
    gaps = gap_result["gap_values"]
    sks = gap_result["sk_values"]
    optimal_k = gap_result["optimal_k"]

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.92, hspace=0.35)
    _header(fig, "Gap Statistic — Principled k Selection",
            "Tibshirani, Walther & Hastie (2001)  ·  "
            "Optimal k = smallest k where Gap(k) >= Gap(k+1) - s_{k+1}")
    _footer(fig)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.errorbar(ks, gaps, yerr=sks, fmt="o-", color=_BLUE,
                 capsize=4, capthick=1.5, markersize=6, lw=2,
                 label="Gap(k) +/- s_k")
    ax1.axvline(optimal_k, color=_RED, ls="--", lw=1.5,
                label=f"Gap optimal k = {optimal_k}")
    ax1.axvline(chosen_k, color=_ORANGE, ls=":", lw=1.5,
                label=f"Pipeline k = {chosen_k}")
    ax1.set_xlabel("Number of clusters (k)", fontsize=11)
    ax1.set_ylabel("Gap(k)", fontsize=11)
    ax1.set_title("Gap(k) = E*[log W_k] - log W_k", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ks, gap_result["log_wk"], "s-", color=_ORANGE,
             markersize=5, lw=1.5, label="Observed log(W_k)")
    ax2.plot(ks, gap_result["expected_log_wk"], "o--", color=_TEAL,
             markersize=5, lw=1.5, label="E*[log(W_k)] (uniform ref)")
    ax2.axvline(optimal_k, color=_RED, ls="--", lw=1.5)
    ax2.set_xlabel("Number of clusters (k)", fontsize=11)
    ax2.set_ylabel("log(W_k)", fontsize=11)
    ax2.set_title("Within-cluster dispersion: observed vs uniform reference", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_ground_truth_confusion(pdf, labels, metadata, k):
    """Confusion matrix: ground truth patterns vs cluster assignments + NMI/ARI/Purity."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Ground Truth Confusion Matrix",
            "Cluster assignments vs known pattern classes  ·  NMI / ARI / Purity")
    _footer(fig)

    gt = metadata["class_id"].values
    pattern_names = sorted(metadata["pattern"].unique()) if "pattern" in metadata.columns else [str(i) for i in sorted(set(gt))]
    pattern_ids = sorted(set(gt))
    cluster_ids = sorted(set(labels))

    # Build confusion matrix: rows = pattern, cols = cluster
    n_patterns = len(pattern_ids)
    n_clusters = len(cluster_ids)
    cm = np.zeros((n_patterns, n_clusters), dtype=int)
    for pi, pid in enumerate(pattern_ids):
        for ci, cid in enumerate(cluster_ids):
            cm[pi, ci] = int(((gt == pid) & (labels == cid)).sum())

    # Compute metrics
    nmi = normalized_mutual_info_score(gt, labels)
    ari = adjusted_rand_score(gt, labels)
    correct = 0
    cluster_purities = []
    for ci, cid in enumerate(cluster_ids):
        col = cm[:, ci]
        if col.sum() > 0:
            p = col.max() / col.sum()
            cluster_purities.append(p)
            correct += col.max()
        else:
            cluster_purities.append(0.0)
    purity = correct / max(len(labels), 1)

    # Plot confusion matrix
    ax_cm = fig.add_axes([0.12, 0.28, 0.78, 0.55])
    im = ax_cm.imshow(cm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax_cm, fraction=0.03, pad=0.02)

    # Annotate cells
    for pi in range(n_patterns):
        for ci in range(n_clusters):
            val = cm[pi, ci]
            if val > 0:
                txt_color = "white" if val > cm.max() * 0.5 else "black"
                ax_cm.text(ci, pi, str(val), ha="center", va="center",
                           fontsize=6 if n_patterns > 10 else 7, color=txt_color)

    ax_cm.set_xticks(range(n_clusters))
    ax_cm.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=7, rotation=45)
    ax_cm.set_yticks(range(n_patterns))
    ax_cm.set_yticklabels(pattern_names, fontsize=7)
    ax_cm.set_xlabel("Cluster", fontsize=9)
    ax_cm.set_ylabel("Ground Truth Pattern", fontsize=9)
    ax_cm.set_title("Samples per (pattern, cluster) cell", fontsize=9)

    # Per-cluster purity bar at top
    ax_pur = fig.add_axes([0.12, 0.84, 0.78, 0.06])
    colors_pur = [_TEAL if p >= 0.7 else _AMBER if p >= 0.5 else _RED for p in cluster_purities]
    ax_pur.bar(range(n_clusters), cluster_purities, color=colors_pur, edgecolor="white")
    ax_pur.set_xticks(range(n_clusters))
    ax_pur.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=6)
    ax_pur.set_ylabel("Purity", fontsize=7)
    ax_pur.set_ylim(0, 1.05)
    ax_pur.axhline(0.5, color=_RED, lw=0.8, ls="--", alpha=0.5)
    ax_pur.set_title("Per-cluster Purity (fraction of dominant pattern)", fontsize=8)
    for ci, p in enumerate(cluster_purities):
        flag = "LOW" if p < 0.5 else ""
        if flag:
            ax_pur.text(ci, p + 0.02, flag, ha="center", fontsize=5.5,
                        color=_RED, fontweight="bold")

    # Metrics summary
    fig.text(0.5, 0.22,
             f"NMI = {nmi:.4f}  {'(good)' if nmi >= 0.8 else '(moderate)' if nmi >= 0.6 else '(poor)'}    |    "
             f"ARI = {ari:.4f}  {'(good)' if ari >= 0.6 else '(moderate)' if ari >= 0.3 else '(poor)'}    |    "
             f"Purity = {purity:.1%}  {'(good)' if purity >= 0.7 else '(moderate)' if purity >= 0.5 else '(poor)'}",
             ha="center", fontsize=9, fontweight="bold",
             color=_TEAL if nmi >= 0.7 and ari >= 0.5 else _AMBER)

    # Low-confidence cluster warnings (P22)
    low_purity_clusters = [f"C{cluster_ids[ci]}" for ci, p in enumerate(cluster_purities) if p < 0.5]
    if low_purity_clusters:
        fig.text(0.5, 0.18,
                 f"LOW CONFIDENCE clusters (purity < 50%): {', '.join(low_purity_clusters)}\n"
                 "Manual review recommended for coils assigned to these clusters.",
                 ha="center", fontsize=8, color=_RED, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                           edgecolor=_RED, alpha=0.9))

    # Interpretation
    fig.text(0.5, 0.12,
             "NMI (Normalized Mutual Information): 1.0 = perfect match, 0.0 = random\n"
             "ARI (Adjusted Rand Index): 1.0 = perfect match, 0.0 = random, <0 = worse than random\n"
             "Purity: fraction of samples matching the dominant pattern in their cluster",
             ha="center", fontsize=7, color=_GREY, linespacing=1.6, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_algorithm_comparison(pdf, algorithm_comparison_path: str) -> None:
    """Embed the multi-algorithm comparison chart (silhouette / DB / CH vs k)."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
    _header(fig, "Clustering Algorithm Comparison",
            "KMeans / GMM / Agglomerative (Ward) / MiniBatchKMeans  "
            "(Saas 2017, Ren 2021, Asadi 2021)")
    _footer(fig)

    img = plt.imread(algorithm_comparison_path)
    ax.imshow(img, aspect="equal")
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

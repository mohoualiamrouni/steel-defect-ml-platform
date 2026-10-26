"""The 9-figure deliverable set for P4 (matplotlib Agg, dpi=150).

Each function is defensive: missing/degenerate inputs produce a small "n/a"
placeholder rather than crashing the pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _na(path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, f"{title}\n(n/a)", ha="center", va="center")
    ax.axis("off")
    _save(fig, path)


def _embed2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """2-D projection for scatter plots. Uses PCA, not UMAP: UMAP pulls in numba,
    whose OpenMP runtime conflicts with torch/MKL on this Windows stack and
    segfaults natively (uncatchable). PCA is robust and dependency-light."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] <= 2:
        return X[:, :2]
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def plot_ssl_loss_curve(histories: List[Dict], path: str) -> None:
    histories = [h for h in histories if h and h.get("loss_curve")]
    if not histories:
        return _na(path, "SSL loss")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    for h in histories:
        ep = range(1, len(h["loss_curve"]) + 1)
        ax1.plot(ep, h["loss_curve"], marker="o", ms=3, label=f"{h['method']} loss")
        if h.get("collapse_curve"):
            ax2.plot(ep, h["collapse_curve"], ls="--", alpha=.7,
                     label=f"{h['method']} collapse_std")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("SSL loss"); ax2.set_ylabel("collapse std")
    ax1.set_title("Self-supervised pretraining")
    ax1.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def plot_embedding_umap(X, y, class_names: Optional[Sequence[str]], path: str, seed=42) -> None:
    """2-D map of the SSL embedding. It uses PCA (see _embed2d); the function and
    file names predate the switch from UMAP."""
    if X is None or len(X) < 5:
        return _na(path, "Embedding (PCA)")
    emb = _embed2d(np.asarray(X), seed)
    fig, ax = plt.subplots(figsize=(6, 5))
    y = np.asarray(y)
    for c in np.unique(y):
        m = y == c
        lab = class_names[c] if (class_names is not None and 0 <= c < len(class_names)) else str(c)
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=.6, label=lab)
    ax.set_title("SSL embedding (PCA)"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=1.5, ncol=2, loc="best")
    _save(fig, path)


def plot_probe_bar(probe_results: Dict[str, Dict[str, float]], path: str) -> None:
    if not probe_results:
        return _na(path, "Probe accuracy")
    reps = list(probe_results)
    metrics = ["linear_probe_acc", "knn_acc"]
    x = np.arange(len(reps)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mname in enumerate(metrics):
        vals = [probe_results[r].get(mname, np.nan) for r in reps]
        ax.bar(x + (i - .5) * w, vals, w, label=mname.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(reps, rotation=15)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.set_title("Frozen-feature probes")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_clustering_uplift(rows: List[Dict], path: str) -> None:
    if not rows:
        return _na(path, "Clustering uplift")
    reps = [r["repr"] for r in rows]
    nmi = [r["nmi"] for r in rows]
    lo = [r["nmi"] - r.get("nmi_ci_low", r["nmi"]) for r in rows]
    hi = [r.get("nmi_ci_high", r["nmi"]) - r["nmi"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(reps, nmi, yerr=[np.abs(lo), np.abs(hi)], capsize=5, color="#4C72B0")
    for i, r in enumerate(rows):
        ax.text(i, nmi[i] + 0.01, f"ARI={r.get('ari', float('nan')):.2f}", ha="center", fontsize=8)
    ax.set_ylabel("NMI (95% CI)"); ax.set_title("Clustering uplift vs ground truth")
    _save(fig, path)


def plot_openset_hist(scores, is_novel, path: str) -> None:
    scores = np.asarray(scores); is_novel = np.asarray(is_novel).astype(bool)
    if len(scores) < 5 or is_novel.min() == is_novel.max():
        return _na(path, "Open-set scores")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[~is_novel], bins=40, alpha=.6, density=True, label="known")
    ax.hist(scores[is_novel], bins=40, alpha=.6, density=True, label="novel (held-out)")
    ax.set_xlabel("novelty score"); ax.set_ylabel("density")
    ax.set_title("Open-set novelty scores"); ax.legend()
    _save(fig, path)


def plot_openset_roc_pr(curves: Dict[str, Dict], path: str) -> None:
    """curves[name] = {'scores':, 'is_novel':, 'auroc':}."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    valid = {k: v for k, v in curves.items() if v is not None and v.get("scores") is not None}
    if not valid:
        return _na(path, "Open-set ROC/PR")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, v in valid.items():
        y = np.asarray(v["is_novel"]).astype(int); s = np.asarray(v["scores"])
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, s)
        a1.plot(fpr, tpr, label=f"{name} (AUROC={v.get('auroc', float('nan')):.2f})")
        pr, rc, _ = precision_recall_curve(y, s)
        a2.plot(rc, pr, label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=.7)
    a1.set_xlabel("FPR"); a1.set_ylabel("TPR"); a1.set_title("ROC"); a1.legend(fontsize=8)
    a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.set_title("PR"); a2.legend(fontsize=8)
    _save(fig, path)


def plot_gcd_umap(X, labels, is_novel, path: str, seed=42) -> None:
    """GCD clusters on a 2-D PCA map (see _embed2d); the name predates the switch
    from UMAP."""
    if X is None or len(X) < 5:
        return _na(path, "GCD clusters")
    emb = _embed2d(np.asarray(X), seed)
    labels = np.asarray(labels); is_novel = np.asarray(is_novel).astype(bool)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[~is_novel, 0], emb[~is_novel, 1], s=8, c=labels[~is_novel],
               cmap="tab20", alpha=.5)
    ax.scatter(emb[is_novel, 0], emb[is_novel, 1], s=22, marker="x", c="red",
               label="novel (held-out)")
    ax.set_title("Generalized category discovery"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_gcd_outliers(outlier_info: Dict, path: str) -> None:
    if not outlier_info:
        return _na(path, "GCD outliers")
    per = outlier_info.get("per_cluster") or outlier_info
    try:
        clusters = sorted(per.keys(), key=lambda x: int(x))
        # run_pipeline asks detect_outliers_per_cluster for the IF ∪ LOF union, stored
        # as n_outliers_combined; a missing key draws the placeholder, never a zero bar.
        counts = [int(per[c]["n_outliers_combined"]) if isinstance(per[c], dict) else int(per[c])
                  for c in clusters]
    except Exception:
        return _na(path, "GCD outliers")
    # Clusters below min_cluster_size are not scored; the producer stores 0 for them.
    skipped = [isinstance(per[c], dict) and bool(per[c].get("skipped")) for c in clusters]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{c}*" if s else str(c) for c, s in zip(clusters, skipped)], counts, color="#C44E52")
    ax.set_xlabel("cluster (* = too small, not scored)" if any(skipped) else "cluster")
    ax.set_ylabel("# outliers (IF ∪ LOF)")
    ax.set_title("Per-cluster outliers")
    _save(fig, path)


def plot_continual_forgetting(cont: Dict, path: str) -> None:
    if not cont:
        return _na(path, "Continual forgetting")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = {"old before": cont.get("acc_old_before"),
            "old after +novel": cont.get("acc_old_after"),
            "new (novel)": cont.get("acc_new")}
    bars = {k: v for k, v in bars.items() if v is not None}
    ax.bar(list(bars), list(bars.values()),
           color=["#4C72B0", "#55A868", "#DD8452"][:len(bars)])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    ax.set_title(f"Continual integration (forgetting Δ={cont.get('forgetting_delta', float('nan')):+.3f})")
    _save(fig, path)

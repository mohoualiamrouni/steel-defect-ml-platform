"""Compare proposed (data-driven) clustering against expert groupings.

This module is the core analytical output of the pipeline.  It quantifies
how well the proposed clusters align with existing expert groups, identifies
expert groups that are split across multiple clusters or that should be
merged, and flags individual defect types whose classification may need
revision.

All public functions are stateless and operate on plain numpy / pandas
objects so they can be reused across notebooks, scripts, and the automated
pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ========================================================================
# Low-level metrics
# ========================================================================


def _compute_purity(labels_proposed: np.ndarray, labels_true: np.ndarray) -> float:
    """Compute clustering purity.

    For each cluster, count how many samples belong to the most frequent
    true label, then sum and divide by the total number of samples.

    Parameters
    ----------
    labels_proposed : np.ndarray
        Predicted cluster labels, shape ``(N,)``.
    labels_true : np.ndarray
        Ground-truth labels (e.g. expert groups), shape ``(N,)``.

    Returns
    -------
    float
        Purity score in ``[0, 1]``.  Higher is better.
    """
    n = len(labels_proposed)
    if n == 0:
        return 0.0

    correct = 0
    for cluster_id in np.unique(labels_proposed):
        mask = labels_proposed == cluster_id
        cluster_true = labels_true[mask]
        if len(cluster_true) == 0:
            continue
        counts = Counter(cluster_true).most_common(1)
        if not counts:
            continue
        correct += counts[0][1]
    return correct / n


def compute_group_metrics(
    labels_proposed: np.ndarray,
    labels_expert: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive suite of clustering-vs-ground-truth metrics.

    Parameters
    ----------
    labels_proposed : np.ndarray
        Predicted cluster assignments, shape ``(N,)``.
    labels_expert : np.ndarray
        Expert-assigned group labels, shape ``(N,)``.

    Returns
    -------
    dict
        Keys: ``nmi``, ``ari``, ``purity``, ``v_measure``, ``homogeneity``,
        ``completeness``.
    """
    labels_proposed = np.asarray(labels_proposed)
    labels_expert = np.asarray(labels_expert)

    return {
        "nmi": float(normalized_mutual_info_score(labels_expert, labels_proposed)),
        "ari": float(adjusted_rand_score(labels_expert, labels_proposed)),
        "purity": float(_compute_purity(labels_proposed, labels_expert)),
        "v_measure": float(v_measure_score(labels_expert, labels_proposed)),
        "homogeneity": float(homogeneity_score(labels_expert, labels_proposed)),
        "completeness": float(completeness_score(labels_expert, labels_proposed)),
    }


# ========================================================================
# Cluster breakdown helpers
# ========================================================================


def _cluster_breakdown(
    labels_proposed: np.ndarray,
    metadata: pd.DataFrame,
    defect_type_col: str,
    expert_group_col: str,
) -> pd.DataFrame:
    """Build a per-cluster summary DataFrame.

    For each proposed cluster, compute:
      * ``cluster_id`` — cluster label.
      * ``size`` — number of samples.
      * ``dominant_type`` — most frequent defect type.
      * ``dominant_group`` — most frequent expert group.
      * ``purity`` — fraction of samples belonging to the dominant group.
      * ``n_types`` — number of distinct defect types.
      * ``n_groups`` — number of distinct expert groups.
    """
    records: List[Dict[str, Any]] = []
    for cid in sorted(np.unique(labels_proposed)):
        mask = labels_proposed == cid
        sub = metadata.loc[mask]
        size = int(mask.sum())

        type_counts = sub[defect_type_col].value_counts()
        group_counts = sub[expert_group_col].value_counts()

        records.append(
            {
                "cluster_id": cid,
                "size": size,
                "dominant_type": type_counts.index[0],
                "dominant_type_frac": float(type_counts.iloc[0]) / size,
                "dominant_group": group_counts.index[0],
                "dominant_group_frac": float(group_counts.iloc[0]) / size,
                "purity": float(group_counts.iloc[0]) / size,
                "n_types": int(type_counts.shape[0]),
                "n_groups": int(group_counts.shape[0]),
            }
        )
    return pd.DataFrame(records)


def _find_split_groups(
    labels_proposed: np.ndarray,
    expert_labels: np.ndarray,
    min_clusters: int = 2,
) -> List[Dict[str, Any]]:
    """Identify expert groups whose members are spread across many clusters.

    An expert group is *split* when its members appear in ``>= min_clusters``
    proposed clusters.

    Returns
    -------
    list of dict
        Each dict has keys: ``expert_group``, ``n_clusters``,
        ``cluster_distribution`` (Counter mapping cluster_id -> count).
    """
    group_to_clusters: Dict[Any, Counter] = defaultdict(Counter)
    for label, group in zip(labels_proposed, expert_labels):
        group_to_clusters[group][label] += 1

    splits: List[Dict[str, Any]] = []
    for group, distribution in sorted(group_to_clusters.items(), key=lambda x: x[0]):
        if len(distribution) >= min_clusters:
            splits.append(
                {
                    "expert_group": group,
                    "n_clusters": len(distribution),
                    "cluster_distribution": dict(distribution),
                }
            )
    return splits


def _find_merge_candidates(
    labels_proposed: np.ndarray,
    expert_labels: np.ndarray,
    overlap_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Find pairs of expert groups that overlap heavily in the same cluster.

    Two groups are *merge candidates* when a sizeable fraction of both groups
    lands in the same cluster.

    Parameters
    ----------
    overlap_threshold : float
        Minimum fraction of *either* group that must co-occur in a single
        cluster for the pair to be flagged.

    Returns
    -------
    list of dict
        Each dict has keys: ``group_a``, ``group_b``, ``shared_cluster``,
        ``frac_a_in_cluster``, ``frac_b_in_cluster``.
    """
    # Build: expert_group -> { cluster_id: count }
    group_to_clusters: Dict[Any, Counter] = defaultdict(Counter)
    group_sizes: Counter = Counter()
    for label, group in zip(labels_proposed, expert_labels):
        group_to_clusters[group][label] += 1
        group_sizes[group] += 1

    unique_groups = sorted(group_to_clusters.keys())
    candidates: List[Dict[str, Any]] = []
    seen_pairs: set = set()

    for i, ga in enumerate(unique_groups):
        for gb in unique_groups[i + 1 :]:
            shared_clusters = set(group_to_clusters[ga]) & set(
                group_to_clusters[gb]
            )
            for cid in shared_clusters:
                frac_a = group_to_clusters[ga][cid] / group_sizes[ga]
                frac_b = group_to_clusters[gb][cid] / group_sizes[gb]
                if frac_a >= overlap_threshold or frac_b >= overlap_threshold:
                    pair_key = (ga, gb, cid)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        candidates.append(
                            {
                                "group_a": ga,
                                "group_b": gb,
                                "shared_cluster": cid,
                                "frac_a_in_cluster": round(frac_a, 4),
                                "frac_b_in_cluster": round(frac_b, 4),
                            }
                        )
    return candidates


def _find_reclassification_candidates(
    labels_proposed: np.ndarray,
    metadata: pd.DataFrame,
    defect_type_col: str,
    expert_group_col: str,
    purity_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Flag defect types whose dominant cluster suggests a different group.

    A defect type is a *reclassification candidate* when:

    1. Most of its samples land in a single cluster, AND
    2. That cluster's dominant expert group differs from the defect type's
       current expert group.

    Parameters
    ----------
    purity_threshold : float
        Minimum fraction of the defect type that must land in the dominant
        cluster for it to be flagged.

    Returns
    -------
    list of dict
        Each dict: ``defect_type``, ``current_group``, ``suggested_group``,
        ``frac_in_dominant_cluster``.
    """
    candidates: List[Dict[str, Any]] = []
    df = metadata.copy()
    df["_proposed"] = labels_proposed

    for dtype, sub in df.groupby(defect_type_col):
        current_mode = sub[expert_group_col].mode()
        if current_mode.empty:
            continue
        current_group = current_mode.iloc[0]
        cluster_counts = sub["_proposed"].value_counts()
        if cluster_counts.empty:
            continue
        dominant_cluster = cluster_counts.index[0]
        frac = cluster_counts.iloc[0] / len(sub)

        if frac < purity_threshold:
            continue

        # What is the dominant expert group in that cluster?
        cluster_mask = df["_proposed"] == dominant_cluster
        cluster_group_mode = df.loc[cluster_mask, expert_group_col].mode()
        if cluster_group_mode.empty:
            continue
        cluster_group = cluster_group_mode.iloc[0]

        if cluster_group != current_group:
            candidates.append(
                {
                    "defect_type": dtype,
                    "current_group": current_group,
                    "suggested_group": cluster_group,
                    "frac_in_dominant_cluster": round(frac, 4),
                }
            )

    return candidates


# ========================================================================
# Public functions
# ========================================================================


def analyze_grouping(
    labels_proposed: np.ndarray,
    metadata: pd.DataFrame,
    features: np.ndarray,
    defect_type_col: str = "defect_type",
    expert_group_col: str = "expert_group",
) -> Dict[str, Any]:
    """Full analysis of proposed clustering vs expert grouping.

    This is the main entry point for post-clustering analysis.  It computes
    a rich set of metrics, breakdowns, and actionable recommendations.

    Parameters
    ----------
    labels_proposed : np.ndarray
        Proposed cluster labels, shape ``(N,)``.
    metadata : pd.DataFrame
        Must contain ``defect_type_col`` and ``expert_group_col``.
    features : np.ndarray
        Feature matrix ``(N, D)`` used for the silhouette score.
    defect_type_col : str
        Column name for the fine-grained defect type.
    expert_group_col : str
        Column name for the expert-assigned group.

    Returns
    -------
    dict
        ``'comparison_metrics'``
            Dict with ``nmi``, ``ari``, ``purity``, ``v_measure``,
            ``homogeneity``, ``completeness``, and ``silhouette``.
        ``'cluster_breakdown'``
            Per-cluster DataFrame (dominant type/group, purity, size, ...).
        ``'split_groups'``
            List of expert groups split across multiple clusters.
        ``'merge_candidates'``
            List of expert group pairs that cluster together.
        ``'reclassification_candidates'``
            List of defect types whose clustering suggests re-grouping.
        ``'mapping_table'``
            Detailed per-defect-type mapping DataFrame with recommendations.
    """
    labels_proposed = np.asarray(labels_proposed)
    expert_labels = metadata[expert_group_col].values

    # -- Metrics -----------------------------------------------------------
    comparison_metrics = compute_group_metrics(labels_proposed, expert_labels)

    # Silhouette score (may be expensive for large N, sample if needed).
    n = len(labels_proposed)
    n_unique = len(np.unique(labels_proposed))
    if n_unique > 1:
        if n > 50_000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, size=50_000, replace=False)
            sil = float(
                silhouette_score(features[sample_idx], labels_proposed[sample_idx])
            )
        else:
            sil = float(silhouette_score(features, labels_proposed))
    else:
        sil = 0.0
    comparison_metrics["silhouette"] = sil

    # -- Cluster breakdown -------------------------------------------------
    cluster_bkdn = _cluster_breakdown(
        labels_proposed, metadata, defect_type_col, expert_group_col
    )

    # -- Split / merge / reclassify ----------------------------------------
    split_groups = _find_split_groups(labels_proposed, expert_labels)
    merge_candidates = _find_merge_candidates(labels_proposed, expert_labels)
    reclass_candidates = _find_reclassification_candidates(
        labels_proposed, metadata, defect_type_col, expert_group_col
    )

    # -- Mapping table -----------------------------------------------------
    mapping_table = generate_mapping_table(
        labels_proposed, metadata, defect_type_col, expert_group_col
    )

    return {
        "comparison_metrics": comparison_metrics,
        "cluster_breakdown": cluster_bkdn,
        "split_groups": split_groups,
        "merge_candidates": merge_candidates,
        "reclassification_candidates": reclass_candidates,
        "mapping_table": mapping_table,
    }


def generate_mapping_table(
    labels_proposed: np.ndarray,
    metadata: pd.DataFrame,
    defect_type_col: str = "defect_type",
    expert_group_col: str = "expert_group",
) -> pd.DataFrame:
    """Create a detailed defect-type mapping table with recommendations.

    For every unique ``defect_type`` the table records:

    * ``defect_type`` — the type label.
    * ``count`` — number of samples of this type.
    * ``expert_group`` — the expert-assigned group.
    * ``dominant_cluster`` — the proposed cluster that contains most samples.
    * ``frac_in_dominant`` — fraction in the dominant cluster.
    * ``n_clusters`` — number of distinct proposed clusters the type spans.
    * ``recommendation`` — one of ``"keep"``, ``"merge"``, ``"split"``,
      or ``"reclassify"``.

    Decision logic for the recommendation:

    * **keep** — The defect type's expert group matches the dominant group
      in its dominant cluster, and most samples (>= 70 %) stay in one
      cluster.
    * **reclassify** — The dominant cluster's dominant expert group differs
      from the type's current group and >= 60 % of the type's samples are
      in that cluster.
    * **split** — The type's samples are scattered across >= 3 clusters
      with no single cluster holding >= 50 %.
    * **merge** — The type maps cleanly to one cluster but that cluster
      contains another expert group at >= 30 % representation.

    Parameters
    ----------
    labels_proposed : np.ndarray
        Proposed cluster labels, shape ``(N,)``.
    metadata : pd.DataFrame
        Must contain ``defect_type_col`` and ``expert_group_col``.
    defect_type_col : str
        Column for the defect type.
    expert_group_col : str
        Column for the expert group.

    Returns
    -------
    pd.DataFrame
        One row per defect type.
    """
    labels_proposed = np.asarray(labels_proposed)
    df = metadata.copy()
    df["_proposed"] = labels_proposed

    records: List[Dict[str, Any]] = []

    for dtype, sub in df.groupby(defect_type_col):
        count = len(sub)
        current_group = sub[expert_group_col].mode().iloc[0]

        cluster_counts = sub["_proposed"].value_counts()
        dominant_cluster = cluster_counts.index[0]
        frac_in_dominant = cluster_counts.iloc[0] / count
        n_clusters = len(cluster_counts)

        # Determine the dominant expert group in the dominant cluster.
        cluster_mask = df["_proposed"] == dominant_cluster
        cluster_group_counts = df.loc[cluster_mask, expert_group_col].value_counts()
        dominant_cluster_group = cluster_group_counts.index[0]

        # --- Recommendation logic -----------------------------------------
        if dominant_cluster_group != current_group and frac_in_dominant >= 0.6:
            recommendation = "reclassify"
        elif n_clusters >= 3 and frac_in_dominant < 0.5:
            recommendation = "split"
        elif frac_in_dominant >= 0.7 and dominant_cluster_group == current_group:
            # Check if another group is well-represented in the same cluster.
            if len(cluster_group_counts) > 1:
                second_group_frac = (
                    cluster_group_counts.iloc[1] / cluster_group_counts.sum()
                )
                if second_group_frac >= 0.3:
                    recommendation = "merge"
                else:
                    recommendation = "keep"
            else:
                recommendation = "keep"
        else:
            recommendation = "keep"

        records.append(
            {
                "defect_type": dtype,
                "count": count,
                "expert_group": current_group,
                "dominant_cluster": dominant_cluster,
                "frac_in_dominant": round(frac_in_dominant, 4),
                "n_clusters": n_clusters,
                "dominant_cluster_group": dominant_cluster_group,
                "recommendation": recommendation,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["recommendation", "defect_type"])
        .reset_index(drop=True)
    )


def _bootstrap_nmi_ci(
    labels_expert: np.ndarray,
    labels_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> Tuple[float, float]:
    """Percentile 95% bootstrap CI of NMI(expert, pred) by resampling rows.

    The clustering is fixed; we resample the (expert, pred) label pairs with
    replacement ``n_bootstrap`` times and recompute NMI, then take the
    2.5/97.5 percentiles.  This is the CI the report claims (``B=100``) but
    that the previous implementation never actually computed.
    """
    if n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    n = len(labels_expert)
    vals = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals[b] = normalized_mutual_info_score(labels_expert[idx], labels_pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def run_ablation(
    cnn_features: np.ndarray,
    seg_features: Optional[np.ndarray],
    labels_expert: np.ndarray,
    k: int,
    random_state: int = 42,
    n_bootstrap: int = 100,
    fusion_pca_variance: float = 0.95,
    cnn_pca_components: int = 50,
) -> pd.DataFrame:
    """Test hypothesis H2: do interpretable morphology descriptors add NMI?

    This is the **honest** ablation.  The previous version concatenated the
    z-scored 960-D CNN block with the 48-D segmentation block *without PCA*
    (``np.hstack``), so the 960 CNN dimensions swamped the 48 morphology
    dimensions and the "combined" row scored *below* CNN-only — the exact
    inverse of the report's claim.  That naïve path is kept here as the
    ``concat_naive`` row precisely to show *why* block-scaling + PCA matters;
    the ``fusion_block_pca`` row runs the production
    :func:`defect_clustering.features.feature_fusion.fuse_features`
    (``concat_scale`` + PCA) that the report actually describes, which is the
    only configuration in which H2 can be fairly tested.

    Rows produced (each clustered with the SAME ``KMeans(k)``):

    * ``cnn_only``         — ``StandardScaler(cnn)`` → PCA(``cnn_pca_components``)
    * ``seg_only``         — ``StandardScaler(seg)`` (48-D, no PCA)
    * ``concat_naive``     — ``hstack(z(cnn), z(seg))``, **no PCA** (the wrong path)
    * ``fusion_block_pca`` — ``fuse_features(concat_scale, pca_variance)``  ← H2 test
    * ``weighted_08_02``   — ``fuse_features(cnn_w=0.8, seg_w=0.2, pca)``

    H2 verdict (computed by the caller from this table): PASSED iff
    ``fusion_block_pca.nmi_ci_low > cnn_only.nmi`` **and**
    ``delta_nmi_vs_cnn >= 0.03``.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature_set, nmi, nmi_ci_low, nmi_ci_high, ari,
        silhouette, n_features, delta_nmi_vs_cnn``.  Numbers here are the
        single source of truth for the report's H2 table — never hand-edit.
    """
    from sklearn.decomposition import PCA  # local import: keeps module load light

    labels_expert = np.asarray(labels_expert)

    def _cluster_and_score(X: np.ndarray) -> Dict[str, Any]:
        X = np.ascontiguousarray(X, dtype=np.float32)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_pred = km.fit_predict(X)
        nmi = float(normalized_mutual_info_score(labels_expert, labels_pred))
        ari = float(adjusted_rand_score(labels_expert, labels_pred))
        sil = (
            float(silhouette_score(X, labels_pred))
            if len(np.unique(labels_pred)) > 1
            else 0.0
        )
        lo, hi = _bootstrap_nmi_ci(labels_expert, labels_pred, n_bootstrap, random_state)
        return {
            "nmi": nmi,
            "nmi_ci_low": lo,
            "nmi_ci_high": hi,
            "ari": ari,
            "silhouette": sil,
            "n_features": int(X.shape[1]),
        }

    # ---- Build each feature representation ----
    reps: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # cnn_only: standardise then PCA (comparable, dense representation)
    cnn_z = StandardScaler().fit_transform(cnn_features)
    n_comp = min(cnn_pca_components, cnn_z.shape[1], cnn_z.shape[0])
    reps["cnn_only"] = PCA(n_components=n_comp, random_state=random_state).fit_transform(cnn_z)

    if seg_features is not None:
        seg_z = StandardScaler().fit_transform(seg_features)
        reps["seg_only"] = seg_z

        # concat_naive: the WRONG path (no PCA) — proves block-scaling matters
        reps["concat_naive"] = np.hstack([cnn_z, seg_z])

        # fusion_block_pca: the production path the report describes (H2 test)
        from defect_clustering.features.feature_fusion import fuse_features

        fused, _ = fuse_features(
            cnn_features,
            seg_features,
            method="concat_scale",
            pca_variance=fusion_pca_variance,
            cnn_weight=1.0,
            seg_weight=1.0,
            random_state=random_state,
        )
        reps["fusion_block_pca"] = fused

        weighted, _ = fuse_features(
            cnn_features,
            seg_features,
            method="concat_scale",
            pca_variance=fusion_pca_variance,
            cnn_weight=0.8,
            seg_weight=0.2,
            random_state=random_state,
        )
        reps["weighted_08_02"] = weighted

    # ---- Cluster & score each ----
    rows: List[Dict[str, Any]] = []
    for name, X in reps.items():
        rec = {"feature_set": name}
        rec.update(_cluster_and_score(X))
        rows.append(rec)

    df = pd.DataFrame(rows)
    nmi_cnn = float(df.loc[df["feature_set"] == "cnn_only", "nmi"].iloc[0])
    df["delta_nmi_vs_cnn"] = (df["nmi"] - nmi_cnn).round(4)
    for col in ("nmi", "nmi_ci_low", "nmi_ci_high", "ari", "silhouette"):
        df[col] = df[col].round(4)
    return df

"""
Defect Group Clustering -- Training Pipeline
=============================================

Orchestrates the full defect group re-clustering workflow:
    1. Load CNN features from pickle + segmentation masks + metadata
    2. Extract segmentation features (regionprops + Hu moments + GLCM)
    3. Fuse CNN + segmentation features (StandardScaler per block + PCA)
    2b. (Optional) UMAP embedding for clustering (DECOR-style)
    4. Compute inertia curve (elbow) over k_range
    5. Gap Statistic on pre-scaled features for principled k selection
    6. Multi-algorithm comparison (KMeans, GMM, Ward, MiniBatch)
    7. Fit chosen clustering method at optimal k
       - NEW: DeepDPM / DPGMM for automatic K discovery (Ronen et al. 2022)
    7b. Per-cluster outlier detection (Isolation Forest + LOF, DECOR-style)
    8. Quality metrics: silhouette + bootstrap CI, DB, CH
    9. Consensus clustering comparison
    10. External comparison: NMI, ARI, Purity vs expert grouping
    11. Group analysis: splits, merges, reclassification candidates
    12. Ablation: CNN-only vs seg-only vs combined
    13. All visualizations (UMAP, dendrogram, confusion, outliers, etc.)
    14. Save all artifacts

References:
    Ronen et al. (2022) -- DeepDPM: automatic K discovery (CVPR), arXiv:2203.14309
    DECOR (2025) -- IF+LOF outlier detection per cluster, arXiv:2510.03328
    Zhang et al. (2023) -- DefChars morphological features, arXiv:2307.11643

Usage:
    # Baseline (KMeans + Gap Statistic)
    python defect_clustering/pipeline/train_pipeline.py
    python defect_clustering/pipeline/train_pipeline.py --method gmm --n_clusters 25

    # Advanced: DeepDPM with auto-K discovery
    python defect_clustering/pipeline/train_pipeline.py --method deep_dpm

    # Full pipeline: UMAP embed + DeepDPM + outlier detection
    python defect_clustering/pipeline/train_pipeline.py --method deep_dpm --umap-embed
"""

from __future__ import annotations

import argparse
import os
import shutil
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import multiprocessing
multiprocessing.freeze_support()   # Windows loky/spawn safety


def run_training(
    config: Optional["DefectClusteringConfig"] = None,
    **overrides,
) -> dict:
    """Run the complete defect group clustering training pipeline.

    Parameters
    ----------
    config : DefectClusteringConfig or None
        Pipeline configuration.  If ``None``, default config is created.
    **overrides
        Any field of :class:`DefectClusteringConfig` can be overridden as a
        keyword argument (e.g. ``method="gmm"``, ``n_clusters=25``).

    Returns
    -------
    dict
        Summary with keys: ``n_samples``, ``n_features``, ``n_clusters``,
        ``silhouette_score``, ``davies_bouldin_index``, ``calinski_harabasz``,
        ``nmi``, ``ari``, ``purity``, ``output_dir``, and more.
    """
    import json
    import sys

    # Ensure the repository root is importable
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── Imports from defect_clustering ────────────────────────────────────
    from defect_clustering.config import DefectClusteringConfig
    from defect_clustering.data.defect_dataset import DefectDataset
    from defect_clustering.data.generate_synthetic_data import (
        generate_synthetic_defect_data,
    )
    from defect_clustering.features.feature_fusion import fuse_features
    from defect_clustering.clustering.clusterer import DefectClusterer
    from defect_clustering.clustering.metrics import (
        cluster_summary,
        compare_algorithms,
        bootstrap_silhouette,
        compute_calinski_harabasz,
        compute_davies_bouldin,
        compute_silhouette,
        inertia_curve,
        gap_statistic,
        consensus_clustering,
        plot_algorithm_comparison,
        plot_elbow,
        plot_umap,
        plot_dendrogram,
        plot_defect_masks_grid,
    )
    from defect_clustering.clustering.advanced_viz import (
        plot_feature_discriminability,
        plot_confusion_matrix as plot_cluster_confusion,
        plot_parallel_coordinates,
        plot_silhouette_distribution,
        plot_temporal_evolution,
        plot_cluster_distance_matrix,
        plot_gap_statistic,
        plot_alluvial_diagram,
    )
    from defect_clustering.clustering.outlier_detection import (
        detect_outliers_per_cluster,
        plot_outlier_summary,
    )
    from defect_clustering.analysis.group_analysis import (
        compute_group_metrics,
        analyze_grouping,
        generate_mapping_table,
        run_ablation,
    )

    # ── Config ────────────────────────────────────────────────────────────
    if config is None:
        config = DefectClusteringConfig()
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"[train] WARNING: unknown config override '{key}' -- ignored")

    # ── Fast mode overrides ────────────────────────────────────────────────
    if config.fast_mode:
        print("[train] FAST MODE enabled — reducing iterations for speed")
        if config.gap_n_references > 10:
            config.gap_n_references = 10
        if config.deep_dpm_epochs > 80:
            config.deep_dpm_epochs = 80

    # ── Output directories ────────────────────────────────────────────────
    out = Path(config.output_dir)
    plots_dir = out / "plots"
    reports_dir = out / "reports"

    print("[train] Clearing previous outputs...")
    for subdir in (plots_dir, reports_dir):
        if subdir.exists():
            shutil.rmtree(subdir)
    for pattern in ("*.csv", "*.json", "*.npz", "*.joblib"):
        for f in out.glob(pattern):
            f.unlink()
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    
    # 1. Data Loading
    
    print(f"\n[train] === Step 1: Data Loading ===")
    print(f"[train] CNN features pickle : {config.cnn_features_pickle}")
    print(f"[train] Masks directory     : {config.masks_dir}")
    print(f"[train] Metadata CSV        : {config.metadata_csv}")

    # If synthetic data doesn't exist yet, generate it
    if not os.path.isfile(config.cnn_features_pickle):
        print("[train] CNN features pickle not found -- generating synthetic data...")
        syn_info = generate_synthetic_defect_data()
        print(f"[train] Synthetic data generated: {syn_info['n_images']} images, "
              f"{syn_info['n_types']} types, {syn_info['n_groups']} groups")

    dataset = DefectDataset(config)
    data = dataset.load()
    print(dataset.summary())

    cnn_features = data["cnn_features"]     # (N, D_cnn)
    seg_features = data["seg_features"]     # (N, D_seg) or None
    metadata = data["metadata"]             # DataFrame
    image_ids = data["image_ids"]           # list of str
    masks = data["masks"]                   # list of ndarray or None

    N = cnn_features.shape[0]
    D_cnn = cnn_features.shape[1]
    D_seg = seg_features.shape[1] if seg_features is not None else 0

    print(f"[train] Loaded {N} samples | CNN dim={D_cnn} | Seg dim={D_seg}")
    print(f"[train] Metadata columns: {list(metadata.columns)}")

    # ── Stratified subsample (cap once for every downstream step) ─────────
    cnn_features_full = cnn_features
    seg_features_full = seg_features
    metadata_full = metadata
    _subsample_applied = False
    _subsample_n = getattr(config, "subsample_n", 0) or 0
    if 0 < _subsample_n < N:
        strat_col = (
            config.expert_group_col
            if config.expert_group_col in metadata.columns
            else None
        )
        rng_ss = np.random.default_rng(config.random_state)
        if strat_col is not None:
            # Per-group cap: floor(subsample_n / n_groups), min 1 per group.
            groups = metadata[strat_col].to_numpy()
            uniq, counts = np.unique(groups, return_counts=True)
            base = max(1, _subsample_n // len(uniq))
            idx_parts = []
            for g, c in zip(uniq, counts):
                g_idx = np.where(groups == g)[0]
                take = min(base, c)
                idx_parts.append(rng_ss.choice(g_idx, size=take, replace=False))
            idx = np.concatenate(idx_parts)
            # Top-up with random extras until we hit _subsample_n.
            if len(idx) < _subsample_n:
                remaining = np.setdiff1d(np.arange(N), idx, assume_unique=False)
                extra = rng_ss.choice(
                    remaining,
                    size=min(_subsample_n - len(idx), len(remaining)),
                    replace=False,
                )
                idx = np.concatenate([idx, extra])
            idx = np.sort(idx)
        else:
            idx = np.sort(rng_ss.choice(N, size=_subsample_n, replace=False))

        cnn_features = cnn_features[idx]
        seg_features = seg_features[idx] if seg_features is not None else None
        metadata = metadata.iloc[idx].reset_index(drop=True)
        image_ids = [image_ids[i] for i in idx]
        if masks is not None:
            masks = [masks[i] for i in idx]
        N = cnn_features.shape[0]
        _subsample_applied = True
        _subsample_idx = idx
        print(f"[train] Subsampled to {N} rows "
              f"(stratified on {strat_col or 'uniform'})")


    # 2. Feature Fusion
    
    print(f"\n[train] === Step 2: Feature Fusion ===")

    if config.use_segmentation_features and seg_features is not None:
        fused_features, fusion_info = fuse_features(
            cnn_features=cnn_features,
            seg_features=seg_features,
            method=config.fusion_method,
            pca_components=config.pca_components,
            pca_variance=config.pca_variance,
            cnn_weight=config.cnn_weight,
            seg_weight=config.seg_weight,
            random_state=config.random_state,
        )
        print(f"[train] Fused features: {fused_features.shape} "
              f"(method={config.fusion_method})")
        if fusion_info.get("pca_variance_explained") is not None:
            print(f"[train] PCA variance explained: "
                  f"{fusion_info['pca_variance_explained']:.2%}")
    else:
        # CNN-only mode
        print("[train] Running in CNN-only mode (no segmentation features)")
        from sklearn.preprocessing import StandardScaler as _ScalerCNN
        _cnn_scaler = _ScalerCNN()
        fused_features = _cnn_scaler.fit_transform(
            cnn_features.astype(np.float64)
        ).astype(np.float32)

        # PCA reduction for high-dim CNN features (essential for Gap Statistic)
        pca_n = config.pca_components
        if pca_n is not None and fused_features.shape[1] > pca_n:
            # Explicit integer component count
            from sklearn.decomposition import PCA as _PCA_CNN
            pca = _PCA_CNN(n_components=pca_n, random_state=config.random_state)
            fused_features = pca.fit_transform(fused_features).astype(np.float32)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"[train] PCA {D_cnn} -> {pca_n} dims "
                  f"(variance retained: {var_exp:.2%})")
        elif pca_n is None and config.pca_variance < 1.0:
            # Variance-based auto-selection
            from sklearn.decomposition import PCA as _PCA_CNN
            pca = _PCA_CNN(n_components=config.pca_variance,
                           random_state=config.random_state)
            fused_features = pca.fit_transform(fused_features).astype(np.float32)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"[train] PCA {D_cnn} -> {pca.n_components_} dims "
                  f"(variance retained: {var_exp:.2%})")

        fusion_info = {
            "method": "cnn_only",
            "n_samples": N,
            "n_features": fused_features.shape[1],
            "cnn_dim": D_cnn,
            "seg_dim": 0,
        }
        print(f"[train] CNN-only features (scaled): {fused_features.shape}")

    # Save feature matrix
    feat_path = out / "feature_matrix.npz"
    np.savez_compressed(feat_path, features=fused_features)
    print(f"[train] Feature matrix saved: {feat_path}")

    # ==================================================================
    # 2b. UMAP Embedding for Clustering (DECOR / VAE-style)
    # ==================================================================
    if getattr(config, "umap_embed_for_clustering", False):
        n_comp = getattr(config, "umap_n_components", 50)
        print(f"\n[train] === Step 2b: UMAP Embedding for Clustering "
              f"(D={fused_features.shape[1]} -> {n_comp}) ===")
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_comp,
                n_neighbors=15,
                min_dist=0.0,
                metric="euclidean",
                random_state=config.random_state,
                low_memory=fused_features.shape[0] > 10_000,
            )
            fused_features_full = fused_features.copy()  # keep original for viz
            fused_features = reducer.fit_transform(
                fused_features.astype(np.float32, copy=False)
            ).astype(np.float32)
            print(f"[train] UMAP-embedded features: {fused_features.shape}")
        except ImportError:
            print("[train] UMAP embedding skipped (install umap-learn)")


    # 3. Inertia Curve (Elbow)
    
    k_min, k_max = config.k_range
    print(f"\n[train] === Step 3: Inertia Curve (k={k_min}..{k_max}) ===")
    inertia_dict = inertia_curve(
        fused_features,
        k_range=config.k_range,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    # 4. Gap Statistic (Tibshirani et al. 2001)
    
    print(f"\n[train] === Step 4: Gap Statistic (Tibshirani 2001) ===")
    from sklearn.preprocessing import StandardScaler as _TempScaler
    _features_pre = _TempScaler().fit_transform(
        fused_features.astype(np.float64)
    ).astype(np.float32)

    # Cap N for Gap to avoid OOM on (N, D) reference samples — the
    # Tibshirani estimator is statistically valid on a sub-sample.
    _GAP_MAX_N = 6000
    if _features_pre.shape[0] > _GAP_MAX_N:
        _rng_gap = np.random.default_rng(config.random_state)
        _gap_idx = _rng_gap.choice(
            _features_pre.shape[0], size=_GAP_MAX_N, replace=False,
        )
        _features_gap = _features_pre[_gap_idx]
        print(f"[train] Gap: subsampled {_GAP_MAX_N}/{_features_pre.shape[0]} "
              f"to bound memory")
    else:
        _features_gap = _features_pre
    gap_result = gap_statistic(
        _features_gap,
        k_range=config.k_range,
        n_references=config.gap_n_references,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    gap_optimal_k = gap_result["optimal_k"]
    print(f"[train] Gap Statistic optimal k = {gap_optimal_k}")

    # Determine k to use
    if config.n_clusters is not None:
        k = config.n_clusters
        print(f"[train] Using user-specified k = {k}")
        if gap_optimal_k != k:
            print(f"[train] WARNING: Gap Statistic recommends k={gap_optimal_k} "
                  f"but pipeline is using k={k}.")
            print(f"[train]   Note: Gap statistic can underestimate k in "
                  f"high-dimensional spaces (D={fused_features.shape[1]}).")
    else:
        k = gap_optimal_k
        print(f"[train] Using Gap Statistic optimal k = {k}")

    try:
        plot_gap_statistic(
            gap_result,
            output_path=str(plots_dir / "gap_statistic.png"),
        )
        print(f"[train] Gap statistic plot saved: {plots_dir / 'gap_statistic.png'}")
    except Exception:
        print("[train] Gap statistic plot failed:")
        traceback.print_exc()

    # Anchor-k silhouettes (independent of Gap optimum).
    # 28 = expert groups, 50 = midpoint of the extended k range.
    ANCHOR_KS = [28, 50]
    _sil_anchors: dict = {}
    try:
        from sklearn.cluster import MiniBatchKMeans as _MBK
        for _ak in ANCHOR_KS:
            if _ak == k:
                continue
            _lbl = _MBK(
                n_clusters=_ak,
                random_state=config.random_state,
                n_init=3,
                batch_size=min(1024, len(_features_pre)),
            ).fit_predict(_features_pre.astype(np.float64))
            _sil_anchors[_ak] = float(
                compute_silhouette(_features_pre, _lbl, max_samples=2000)
            )
            print(f"[train] Anchor k={_ak}: silhouette={_sil_anchors[_ak]:.4f}")
    except Exception:
        print("[train] Anchor-k evaluation failed:")
        traceback.print_exc()


    # 5. Algorithm Comparison
    
    k_compare_min = max(3, k_min)
    k_compare_max = max(k_max, k + 2)
    print(f"\n[train] === Step 5: Algorithm Comparison "
          f"(k={k_compare_min}..{k_compare_max}) ===")
    comparison_df = None
    try:
        comparison_df = compare_algorithms(
            fused_features,
            k_range=(k_compare_min, k_compare_max),
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        plot_algorithm_comparison(
            comparison_df,
            output_path=str(plots_dir / "algorithm_comparison.png"),
            highlight_k=k,
        )
        # Sort for deterministic row+column ordering across runs.
        comparison_sorted = comparison_df.reindex(
            columns=sorted(comparison_df.columns)
        )
        sort_keys = [c for c in ("algorithm", "k") if c in comparison_sorted.columns]
        if sort_keys:
            comparison_sorted = comparison_sorted.sort_values(sort_keys).reset_index(drop=True)
        comparison_sorted.to_csv(out / "algorithm_comparison.csv", index=False)
    except Exception:
        print("[train] Algorithm comparison failed:")
        traceback.print_exc()

    
    # 6. Fit Main Clusterer
    
    # For DeepDPM/DPGMM: K is auto-discovered, no need for Gap Statistic k
    is_nonparametric = config.method in ("deep_dpm", "dpgmm")

    if is_nonparametric:
        print(f"\n[train] === Step 6: Fit {config.method.upper()} "
              f"(non-parametric, K auto-discovered) ===")
        print(f"[train] Initial K hint = {k} (will be adjusted automatically)")
    else:
        if getattr(config, "target_k", None):
            k = int(config.target_k)
            print(f"[train] target_k override active -> k={k}")

        print(f"\n[train] === Step 6: Fit DefectClusterer "
              f"(method={config.method}, k={k}) ===")

    clusterer = DefectClusterer(
        method=config.method,
        n_clusters=k,
        random_state=config.random_state,
        distance_metric=config.distance_metric,
    )
    if hasattr(clusterer, "overcluster_factor"):
        clusterer.overcluster_factor = float(getattr(config, "overcluster_factor", 2.0))
    clusterer.fit(fused_features)
    labels = clusterer.cluster_labels_

    # Monster-cluster safety check — any cluster > 25% of N signals a cosine
    # collapse (Steinbach et al. 2000) or isotropic-noise regime.
    _uniq, _cnt = np.unique(labels[labels >= 0], return_counts=True)
    if _cnt.size:
        _max_frac = _cnt.max() / _cnt.sum()
        if _max_frac > 0.25:
            print(
                f"[train] WARNING: cluster {_uniq[_cnt.argmax()]} holds "
                f"{_max_frac:.1%} of samples — possible monster-cluster collapse."
            )

    # For non-parametric methods, update k to the discovered value
    if is_nonparametric:
        k = len(np.unique(labels[labels >= 0]))
        print(f"[train] {config.method.upper()} discovered K = {k} clusters")
        if hasattr(clusterer._model, 'split_merge_history_'):
            history = clusterer._model.split_merge_history_
            if history:
                print(f"[train] Split/merge history ({len(history)} operations):")
                for h in history[-5:]:
                    print(f"  - {h}")

    # Extract StandardScaler-transformed space for consistent metrics
    features_scaled = clusterer._scaler.transform(
        fused_features.astype(np.float64)
    ).astype(np.float32)

    print(f"[train] Clustering complete: {len(np.unique(labels))} clusters found")
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"[train] Cluster sizes: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")

    
    # 7. Quality Metrics
    
    print(f"\n[train] === Step 7: Quality Metrics (Saas et al. 2017) ===")

    sil_score = compute_silhouette(features_scaled, labels)
    db_score = compute_davies_bouldin(features_scaled, labels)
    ch_score = compute_calinski_harabasz(features_scaled, labels)

    # Bootstrap CI for silhouette (McInnes 2018 style)
    _n_boot = 30 if config.fast_mode else 100
    sil_bootstrap = bootstrap_silhouette(
        features_scaled, labels,
        n_bootstrap=_n_boot, sample_frac=0.8,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    print(f"\n[train] Cluster Quality Metrics (Saas et al. 2017):")
    print(f"  {'Metric':<28}  {'Value':>10}  {'Direction'}")
    print(f"  {'-' * 52}")
    print(f"  {'Silhouette Score':<28}  {sil_score:>10.4f}  higher is better")
    print(f"  {'  Bootstrap 95% CI':<28}  [{sil_bootstrap['ci_low']:.4f}, "
          f"{sil_bootstrap['ci_high']:.4f}]  (n={_n_boot} resamples)")
    print(f"  {'Davies-Bouldin Index':<28}  {db_score:>10.4f}  lower is better")
    print(f"  {'Calinski-Harabasz Score':<28}  {ch_score:>10.1f}  higher is better")

    # Quality assessment
    print(f"\n[train] === Quality Assessment ===")
    if sil_score < 0.20:
        print(f"  [WARN] Silhouette = {sil_score:.3f} (poor, < 0.20): "
              "clusters overlap significantly. Consider adjusting k.")
    elif sil_score < 0.35:
        print(f"  [INFO] Silhouette = {sil_score:.3f} (moderate): "
              "some overlap between clusters.")
    else:
        print(f"  [OK]   Silhouette = {sil_score:.3f} (good): "
              "clusters are reasonably well-separated.")

    # Cluster size imbalance
    imbalance_ratio = float(label_counts.max()) / max(float(label_counts.min()), 1)
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5.0:
        print(f"  [WARN] Imbalance ratio {imbalance_ratio:.1f}x > 5.0: "
              "largest cluster is much bigger than smallest. "
              "Rare defect types may be merged into large clusters.")

    # Near-merge warning -- check inter-centroid distances
    if clusterer.centroids_ is not None and len(clusterer.centroids_) > 1:
        try:
            from scipy.spatial.distance import pdist, squareform
            centroids = clusterer.centroids_
            dists = squareform(pdist(centroids))
            np.fill_diagonal(dists, np.inf)
            d_max = dists[dists < np.inf].max() if dists[dists < np.inf].size > 0 else 1.0
            norm_dists = dists / d_max
            for ci in range(len(centroids)):
                for cj in range(ci + 1, len(centroids)):
                    nd = norm_dists[ci, cj]
                    if nd < 0.4:
                        print(f"  [WARN] Clusters C{ci} and C{cj} are very close "
                              f"(normalised distance = {nd:.2f}). Consider merging.")
        except Exception:
            pass

    if gap_optimal_k != k:
        print(f"  [INFO] Gap Statistic suggests k={gap_optimal_k} vs chosen k={k}")

    # ==================================================================
    # 7b. Per-Cluster Outlier Detection (DECOR-style: IF + LOF)
    # ==================================================================
    outlier_mask = None
    outlier_info = None
    if getattr(config, "outlier_detection", True):
        print(f"\n[train] === Step 7b: Per-Cluster Outlier Detection "
              f"(DECOR-style) ===")
        print(f"[train] Methods: Isolation Forest + Local Outlier Factor")
        print(f"[train] Strategy: {getattr(config, 'outlier_strategy', 'union')} | "
              f"Contamination: {getattr(config, 'outlier_contamination', 0.05):.0%}")
        try:
            outlier_mask, outlier_info = detect_outliers_per_cluster(
                features=features_scaled,
                labels=labels,
                contamination=getattr(config, "outlier_contamination", 0.05),
                strategy=getattr(config, "outlier_strategy", "union"),
                random_state=config.random_state,
            )
            n_outliers = outlier_mask.sum()
            print(f"[train] Total outliers detected: {n_outliers} "
                  f"({n_outliers / N:.1%})")
            print(f"[train]   Isolation Forest: {outlier_info['total_outliers_if']}")
            print(f"[train]   LOF:              {outlier_info['total_outliers_lof']}")
            print(f"[train]   Combined:         {outlier_info['total_outliers']}")

            # Per-cluster breakdown for high-outlier clusters
            for ck, ci in outlier_info["per_cluster"].items():
                if ci.get("skipped"):
                    continue
                rate = ci["n_outliers_combined"] / max(ci["size"], 1)
                if rate > 0.10:
                    print(f"  [WARN] Cluster C{ck}: {ci['n_outliers_combined']} "
                          f"outliers ({rate:.0%}) — may contain mixed defect types")

            # Save outlier info
            import json as _json_ol
            with open(out / "outlier_detection.json", "w") as f:
                _json_ol.dump(outlier_info, f, indent=2)

        except Exception:
            print("[train] Outlier detection failed:")
            traceback.print_exc()


    # 8. Consensus Clustering Comparison
    
    print(f"\n[train] === Step 8: Consensus Clustering Comparison ===")
    cons_sil = None
    if not getattr(config, "enable_consensus", False):
        print("[train] Consensus clustering disabled (use --enable-consensus to turn on).")
    elif config.method != "consensus":
        try:
            print("[train] Computing consensus clustering for comparison...")
            _cons_runs = 10 if config.fast_mode else 20
            consensus_labels, _ = consensus_clustering(
                features_scaled, n_clusters=k,
                algorithms=["kmeans", "agglomerative", "gmm"],
                n_runs=_cons_runs, random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
            # consensus_clustering may subsample for large N — compute
            # silhouette on whatever length was returned
            n_cons = len(consensus_labels)
            if n_cons < len(features_scaled):
                cons_sil = compute_silhouette(
                    features_scaled[:n_cons], consensus_labels)
            else:
                cons_sil = compute_silhouette(features_scaled, consensus_labels)
            print(f"[train] Consensus silhouette: {cons_sil:.4f}  "
                  f"(vs {config.method} silhouette: {sil_score:.4f})")
            if cons_sil > sil_score + 0.01:
                print(f"[train] NOTE: Consensus clustering outperforms "
                      f"{config.method} by {cons_sil - sil_score:.4f} "
                      f"silhouette points")
        except Exception:
            print("[train] Consensus clustering failed:")
            traceback.print_exc()

    
    # 9. External Comparison with Expert Grouping
    
    expert_col = config.expert_group_col
    defect_col = config.defect_type_col
    group_metrics = None

    print(f"\n[train] === Step 9: External Comparison with Expert Grouping ===")
    if expert_col in metadata.columns:
        expert_labels = metadata[expert_col].values
        group_metrics = compute_group_metrics(labels, expert_labels)

        print(f"[train] External Comparison Metrics (vs '{expert_col}'):")
        print(f"  {'Metric':<28}  {'Value':>10}  {'Quality'}")
        print(f"  {'-' * 55}")

        nmi_val = group_metrics["nmi"]
        ari_val = group_metrics["ari"]
        purity_val = group_metrics["purity"]
        v_measure_val = group_metrics["v_measure"]
        homogeneity_val = group_metrics["homogeneity"]
        completeness_val = group_metrics["completeness"]

        def _quality(val, thresholds=(0.6, 0.8)):
            if val >= thresholds[1]:
                return "good"
            elif val >= thresholds[0]:
                return "moderate"
            return "poor"

        print(f"  {'NMI':<28}  {nmi_val:>10.4f}  {_quality(nmi_val)}")
        print(f"  {'ARI':<28}  {ari_val:>10.4f}  {_quality(ari_val, (0.3, 0.6))}")
        print(f"  {'Purity':<28}  {purity_val:>10.4f}  {_quality(purity_val, (0.5, 0.7))}")
        print(f"  {'V-measure':<28}  {v_measure_val:>10.4f}  {_quality(v_measure_val)}")
        print(f"  {'Homogeneity':<28}  {homogeneity_val:>10.4f}  {_quality(homogeneity_val)}")
        print(f"  {'Completeness':<28}  {completeness_val:>10.4f}  {_quality(completeness_val)}")

        # Adjusted Mutual Info — chance-corrected alternative to NMI that
        # handles imbalanced cluster sizes (Vinh, Epps, Bailey 2010).
        try:
            from sklearn.metrics import adjusted_mutual_info_score
            ami_val = float(adjusted_mutual_info_score(expert_labels, labels))
            group_metrics["ami"] = ami_val
            print(f"  {'AMI (chance-corrected)':<28}  {ami_val:>10.4f}  {_quality(ami_val)}")
        except Exception:
            pass
    else:
        print(f"[train] WARNING: '{expert_col}' column not found in metadata -- "
              f"skipping external comparison.")

    
    # 10. Group Analysis: splits, merges, reclassification candidates
    
    print(f"\n[train] === Step 10: Group Analysis ===")
    grouping_result = None
    mapping_table = None
    ablation_df = None

    if expert_col in metadata.columns and defect_col in metadata.columns:
        try:
            grouping_result = analyze_grouping(
                labels_proposed=labels,
                metadata=metadata,
                features=features_scaled,
                defect_type_col=defect_col,
                expert_group_col=expert_col,
            )

            # Print split groups
            split_groups = grouping_result["split_groups"]
            if split_groups:
                print(f"\n[train] Expert groups SPLIT across multiple clusters "
                      f"({len(split_groups)} groups):")
                for sg in split_groups[:10]:
                    print(f"  - {sg['expert_group']}: spread across "
                          f"{sg['n_clusters']} clusters")
            else:
                print("[train] No expert groups are significantly split.")

            # Print merge candidates
            merge_candidates = grouping_result["merge_candidates"]
            if merge_candidates:
                print(f"\n[train] MERGE candidates ({len(merge_candidates)} pairs):")
                for mc in merge_candidates[:10]:
                    print(f"  - {mc['group_a']} + {mc['group_b']} in "
                          f"cluster {mc['shared_cluster']} "
                          f"(frac_a={mc['frac_a_in_cluster']:.2f}, "
                          f"frac_b={mc['frac_b_in_cluster']:.2f})")
            else:
                print("[train] No merge candidates found.")

            # Print reclassification candidates
            reclass = grouping_result["reclassification_candidates"]
            if reclass:
                print(f"\n[train] RECLASSIFICATION candidates ({len(reclass)} types):")
                for rc in reclass[:10]:
                    print(f"  - {rc['defect_type']}: {rc['current_group']} -> "
                          f"{rc['suggested_group']} "
                          f"(frac={rc['frac_in_dominant_cluster']:.2f})")
            else:
                print("[train] No reclassification candidates found.")

            # Generate mapping table
            mapping_table = grouping_result["mapping_table"]
            mapping_table.to_csv(out / "mapping_table.csv", index=False)
            print(f"[train] Mapping table saved: {out / 'mapping_table.csv'}")

            # Print recommendation summary
            if "recommendation" in mapping_table.columns:
                rec_counts = mapping_table["recommendation"].value_counts()
                print(f"\n[train] Recommendation summary:")
                for rec, cnt in rec_counts.items():
                    print(f"  {rec:<15s}: {cnt:>4d} defect types")

        except Exception:
            print("[train] Group analysis failed:")
            traceback.print_exc()

        # Ablation: CNN-only vs seg-only vs combined
        print(f"\n[train] Running ablation study (CNN-only vs seg-only vs combined)...")
        try:
            expert_labels_arr = metadata[expert_col].values
            ablation_df = run_ablation(
                cnn_features=cnn_features,
                seg_features=seg_features,
                labels_expert=expert_labels_arr,
                k=k,
                random_state=config.random_state,
                n_bootstrap=getattr(config, "ablation_n_bootstrap", 100),
                fusion_pca_variance=getattr(config, "fusion_pca_variance", 0.95),
            )
            print(f"\n[train] Ablation Results:")
            print(ablation_df.to_string(index=False))
            ablation_df.to_csv(out / "ablation_results.csv", index=False)
            print(f"[train] Ablation results saved: {out / 'ablation_results.csv'}")
        except Exception:
            print("[train] Ablation study failed:")
            traceback.print_exc()
    else:
        print(f"[train] Skipping group analysis: need both '{defect_col}' and "
              f"'{expert_col}' columns in metadata.")

    
    # Save metrics to JSON
    
    metrics_dict = {
        "n_samples": N,
        "n_features": int(fused_features.shape[1]),
        "n_clusters": k,
        "method": config.method,
        "silhouette_score": round(sil_score, 6),
        "silhouette_bootstrap_mean": round(sil_bootstrap["mean"], 6),
        "silhouette_bootstrap_ci_low": round(sil_bootstrap["ci_low"], 6),
        "silhouette_bootstrap_ci_high": round(sil_bootstrap["ci_high"], 6),
        "davies_bouldin_index": round(db_score, 6),
        "calinski_harabasz": round(ch_score, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "gap_optimal_k": gap_optimal_k,
    }
    if cons_sil is not None:
        metrics_dict["consensus_silhouette"] = round(cons_sil, 6)
    if group_metrics is not None:
        for gm_key, gm_val in group_metrics.items():
            metrics_dict[gm_key] = round(gm_val, 6)
    if outlier_info is not None:
        metrics_dict["outlier_total"] = outlier_info["total_outliers"]
        metrics_dict["outlier_rate"] = round(outlier_info["outlier_rate"], 4)
        metrics_dict["outlier_strategy"] = outlier_info["strategy"]
    if is_nonparametric:
        metrics_dict["k_discovery_method"] = config.method
    if _sil_anchors:
        metrics_dict["anchor_silhouettes"] = {
            str(_kk): round(_vv, 6) for _kk, _vv in _sil_anchors.items()
        }

    metrics_path = out / "cluster_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[train] Metrics saved: {metrics_path}")

    if group_metrics is not None and expert_labels is not None:
        try:
            from scipy.optimize import linear_sum_assignment
            from sklearn.metrics import balanced_accuracy_score, confusion_matrix

            expert_arr = np.asarray(expert_labels)
            labels_arr = np.asarray(labels)
            uniq_expert, expert_idx = np.unique(expert_arr, return_inverse=True)
            uniq_cluster, cluster_idx = np.unique(labels_arr, return_inverse=True)
            cm = confusion_matrix(expert_idx, cluster_idx)
            row_ind, col_ind = linear_sum_assignment(-cm)
            mapping = {c: r for r, c in zip(row_ind, col_ind)}
            mapped = np.array(
                [mapping.get(c, -1) for c in cluster_idx], dtype=int
            )
            valid = mapped >= 0
            bal_acc = float(
                balanced_accuracy_score(expert_idx[valid], mapped[valid])
            ) if valid.any() else 0.0

            per_group_recall = {}
            for gi, gname in enumerate(uniq_expert):
                mask_g = expert_idx == gi
                n_g = int(mask_g.sum())
                if n_g == 0:
                    continue
                correct = int(((mapped == gi) & mask_g).sum())
                per_group_recall[str(gname)] = {
                    "n": n_g,
                    "recall": round(correct / n_g, 6),
                }

            imbalance_report = {
                "balanced_accuracy": round(bal_acc, 6),
                "purity": round(float(group_metrics["purity"]), 6),
                "n_expert_groups": int(len(uniq_expert)),
                "n_clusters": int(len(uniq_cluster)),
                "per_group_recall": per_group_recall,
            }
            imb_path = out / "imbalance_report.json"
            with open(imb_path, "w") as f:
                json.dump(imbalance_report, f, indent=2)
            print(f"[train] Imbalance report saved: {imb_path}  (bal_acc={bal_acc:.4f})")
        except Exception as e:
            print(f"[train] imbalance_report skipped: {e}")

    # Save model
    clusterer_path = out / "clusterer.joblib"
    clusterer.save(str(clusterer_path))

    
    # Cluster assignments CSV
    
    # When subsampled, predict on the full N so the CSV covers every defect.
    if _subsample_applied:
        try:
            if fusion_info.get("method") == "cnn_only":
                fused_full = fusion_info["cnn_scaler"].transform(
                    cnn_features_full.astype(np.float32)
                ).astype(np.float32)
            else:
                from defect_clustering.features.feature_fusion import (
                    transform_features as _transform_full,
                )
                fused_full = _transform_full(
                    cnn_features=cnn_features_full.astype(np.float32),
                    seg_features=seg_features_full.astype(np.float32),
                    info=fusion_info,
                )
            labels_full = clusterer.predict(fused_full)
            assignments_df = metadata_full.copy()
            assignments_df["cluster_id"] = labels_full
            print(f"[train] Predicted labels on full N={len(labels_full)}")
        except Exception:
            print("[train] Full-N predict failed, falling back to subsample:")
            traceback.print_exc()
            assignments_df = metadata.copy()
            assignments_df["cluster_id"] = labels
            if outlier_mask is not None:
                assignments_df["is_outlier"] = outlier_mask.astype(int)
    else:
        assignments_df = metadata.copy()
        assignments_df["cluster_id"] = labels
        if outlier_mask is not None:
            assignments_df["is_outlier"] = outlier_mask.astype(int)
    assignments_path = out / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"[train] Cluster assignments saved: {assignments_path}")

    
    # Cluster summary table
    
    summary_df = cluster_summary(features_scaled, labels, metadata)
    print("\n[train] === Cluster Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out / "cluster_summary.csv", index=False)

    
    # 11. Visualizations
    
    print(f"\n[train] === Step 11: Visualizations ===")

    # 11e. Confusion matrix (cluster vs expert group)
    try:
        plot_cluster_confusion(
            labels=labels,
            metadata=metadata,
            output_path=str(plots_dir / "adv_cluster_confusion.png"),
            expert_col=expert_col,
        )
    except Exception:
        print("[train] Confusion matrix plot failed:")
        traceback.print_exc()

    # 11g. Silhouette distribution
    try:
        plot_silhouette_distribution(
            features=features_scaled,
            labels=labels,
            output_path=str(plots_dir / "adv_silhouette_distribution.png"),
        )
    except Exception:
        print("[train] Silhouette distribution plot failed:")
        traceback.print_exc()

    # 11j. Alluvial diagram (expert -> proposed)
    try:
        plot_alluvial_diagram(
            metadata=metadata,
            labels_proposed=labels,
            expert_col=expert_col,
            output_path=str(plots_dir / "alluvial_diagram.png"),
        )
        print(f"[train] Alluvial diagram saved: {plots_dir / 'alluvial_diagram.png'}")
    except Exception:
        print("[train] Alluvial diagram failed:")
        traceback.print_exc()

    print(f"[train] All plots saved in: {plots_dir}")

    
    # 12. Final Summary
    
    print(f"\n[train] === Training Complete ===")
    print(f"  Samples             : {N}")
    print(f"  CNN features        : {D_cnn}")
    print(f"  Seg features        : {D_seg}")
    print(f"  Fused features      : {fused_features.shape[1]}")
    print(f"  Method              : {config.method}")
    print(f"  Clusters            : {k}")
    print(f"  Silhouette          : {sil_score:.4f}  "
          f"95% CI [{sil_bootstrap['ci_low']:.4f}, "
          f"{sil_bootstrap['ci_high']:.4f}]")
    print(f"  Davies-Bouldin      : {db_score:.4f}")
    print(f"  Calinski-Harabasz   : {ch_score:.1f}")
    if group_metrics is not None:
        print(f"  NMI (vs expert)     : {group_metrics['nmi']:.4f}")
        print(f"  ARI (vs expert)     : {group_metrics['ari']:.4f}")
        print(f"  Purity (vs expert)  : {group_metrics['purity']:.4f}")
    print(f"  Gap optimal k       : {gap_optimal_k}")
    if is_nonparametric:
        print(f"  K discovery         : {config.method} (auto)")
    if outlier_info is not None:
        print(f"  Outliers detected   : {outlier_info['total_outliers']} "
              f"({outlier_info['outlier_rate']:.1%})")
    print(f"  Artifacts           : {out}")

    # Return summary dict
    result = {
        "n_samples": N,
        "n_features": int(fused_features.shape[1]),
        "n_clusters": k,
        "method": config.method,
        "silhouette_score": sil_score,
        "silhouette_ci_low": sil_bootstrap["ci_low"],
        "silhouette_ci_high": sil_bootstrap["ci_high"],
        "davies_bouldin_index": db_score,
        "calinski_harabasz": ch_score,
        "gap_optimal_k": gap_optimal_k,
        "imbalance_ratio": imbalance_ratio,
        "output_dir": str(out),
    }
    if cons_sil is not None:
        result["consensus_silhouette"] = cons_sil
    if group_metrics is not None:
        result.update(group_metrics)
    if ablation_df is not None:
        result["ablation"] = ablation_df.to_dict(orient="records")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train defect group clustering model",
    )
    p.add_argument(
        "--output_dir", default="defect_clustering/outputs",
        help="Directory for saving artifacts and plots (default: defect_clustering/outputs)",
    )
    p.add_argument(
        "--n_clusters", type=int, default=None,
        help="Number of clusters. Default: auto from Gap Statistic.",
    )
    p.add_argument(
        "--k-range", type=int, nargs=2, metavar=("K_MIN", "K_MAX"),
        default=None, dest="k_range",
        help="Inclusive (k_min, k_max) range for Gap Statistic and comparison "
             "(default: from config, typically 10-60)",
    )
    p.add_argument(
        "--subsample-n", type=int, default=None, dest="subsample_n",
        help="Stratified cap applied once at Step 1 (0 = keep all). "
             "Default: 8000 from config.",
    )
    p.add_argument(
        "--enable-consensus", action="store_true", dest="enable_consensus",
        help="Enable Step 8 consensus clustering (costly; off by default).",
    )
    p.add_argument(
        "--full", action="store_true", dest="full_mode",
        help="Flip off fast_mode and keep all rows (paper-grade run).",
    )
    p.add_argument(
        "--method", default=None,
        choices=["kmeans", "hdbscan", "gmm", "spectral",
                 "agglomerative", "minibatch_kmeans", "dpgmm", "deep_dpm",
                 "bisecting_merge"],
        help="Clustering algorithm (default: kmeans). "
             "dpgmm/deep_dpm auto-discover K (Ronen et al. CVPR 2022)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--no-seg", action="store_true", dest="no_seg",
        help="Disable segmentation features (CNN-only mode)",
    )
    p.add_argument(
        "--no-pca", action="store_true", dest="no_pca",
        help="Disable PCA reduction after feature fusion",
    )
    p.add_argument(
        "--umap-embed", action="store_true", dest="umap_embed",
        help="Use UMAP embedding before clustering (DECOR-style)",
    )
    p.add_argument(
        "--umap-dim", type=int, default=50,
        help="UMAP target dimensionality (default: 50)",
    )
    p.add_argument(
        "--no-outliers", action="store_true", dest="no_outliers",
        help="Disable per-cluster outlier detection",
    )
    p.add_argument(
        "--outlier-contamination", type=float, default=0.05,
        help="Expected outlier fraction per cluster (default: 0.05)",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Enable fast mode (fewer iterations, ~2x faster)",
    )
    p.add_argument(
        "--n-jobs", type=int, default=4, dest="n_jobs",
        help="Number of parallel jobs (default: 4; each worker pickles the "
             "full feature matrix, so higher values cause memory blow-ups).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    overrides = {
        "output_dir": args.output_dir,
        "random_state": args.seed,
    }
    if args.method is not None:
        overrides["method"] = args.method
    if args.n_clusters is not None:
        overrides["n_clusters"] = args.n_clusters
    if args.k_range is not None:
        overrides["k_range"] = tuple(args.k_range)
    if args.no_seg:
        overrides["use_segmentation_features"] = False
    if args.no_pca:
        overrides["pca_components"] = None
    if args.umap_embed:
        overrides["umap_embed_for_clustering"] = True
        overrides["umap_n_components"] = args.umap_dim
    if args.no_outliers:
        overrides["outlier_detection"] = False
    overrides["outlier_contamination"] = args.outlier_contamination
    if args.fast:
        overrides["fast_mode"] = True
    if args.full_mode:
        overrides["fast_mode"] = False
        overrides["subsample_n"] = 0
    if args.subsample_n is not None:
        overrides["subsample_n"] = args.subsample_n
    if args.enable_consensus:
        overrides["enable_consensus"] = True
    overrides["n_jobs"] = args.n_jobs

    run_training(**overrides)

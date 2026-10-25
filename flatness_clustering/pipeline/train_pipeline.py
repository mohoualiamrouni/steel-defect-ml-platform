"""
Flatness Clustering — Training Pipeline
========================================

Orchestrates the full model-fitting workflow:
    1. Load all synthetic coil profiles via FlatnessDataset
    2. Extract 1105-dim features (extract_batch): 105 statistical ones, including
       Group D temporal, plus Group E ROCKET (500 kernels, 1000 features)
    3. Extract 50-dim spatial features for visualisation (SpatialFeatureExtractor)
    4. Fit CoilClusterer (KMeans by default)
    5. Compute quality metrics: silhouette + Davies-Bouldin + Calinski-Harabász
       (Saas et al. 2017) + inertia curve (elbow)
    6. Save artifacts: clusterer.joblib, spatial_pca.joblib, feature_matrix.npz,
       cluster_assignments.csv, cluster_metrics.json
    7. Generate plots: elbow curve (with silhouette overlay), UMAP
    8. Print a formatted metric summary table

Usage
-----
    python flatness_clustering/pipeline/train_pipeline.py

    # Or from Python:
    from flatness_clustering.pipeline.train_pipeline import run_training
    summary = run_training()
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def run_training(
    data_dir: str = "flatness_clustering/data/synthetic_profiles",
    output_dir: str = "flatness_clustering/outputs",
    n_clusters: int = 5,
    method: str = "kmeans",
    random_state: int = 42,
    rocket_variant: str = "rocket",
) -> dict:
    """Run the complete flatness clustering training pipeline.

    Args:
        data_dir:       Directory with NPZ coil profiles and metadata.csv.
        output_dir:     Directory for saving artifacts and plots.
        n_clusters:     Number of clusters for KMeans.
        method:         'kmeans' or 'hdbscan'.
        random_state:   Random seed.
        rocket_variant: 'rocket' (default, original Dempster 2020 random kernels)
            or 'minirocket' (Dempster 2020 deterministic 84-kernel variant).
            Default preserves backward-compatibility byte-for-byte.
    Returns:
        Summary dict with keys: n_coils, n_features, n_clusters,
        silhouette_score, output_dir.
    """
    import sys
    # Ensure the repository root is importable
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.clustering.metrics import (
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
        plot_cluster_heatmaps,
        plot_elbow,
        plot_umap,
    )
    from flatness_clustering.clustering.advanced_viz import (
        plot_feature_discriminability,
        plot_confusion_matrix as plot_cluster_confusion,
        plot_parallel_coordinates,
        plot_silhouette_distribution,
        plot_temporal_evolution,
        plot_cluster_distance_matrix,
        plot_gap_statistic,
    )
    from flatness_clustering.data.flatness_dataset import FlatnessDataset
    from flatness_clustering.features.spatial_features import SpatialFeatureExtractor
    from flatness_clustering.features.statistical_features import extract_batch
    import json

    out = Path(output_dir)
    plots_dir  = out / "plots"
    reports_dir = out / "reports"

    # Clear all outputs from previous runs so no stale files survive
    print("[train] Clearing previous outputs...")
    for subdir in (plots_dir, reports_dir):
        if subdir.exists():
            shutil.rmtree(subdir)
    for pattern in ("*.csv", "*.json", "*.npz", "*.joblib"):
        for f in out.glob(pattern):
            f.unlink()
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load profiles
    # ------------------------------------------------------------------
    print(f"\n[train] Loading profiles from: {data_dir}")
    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Data directory not found: '{data_dir}'\n"
            "  Run: python flatness_clustering/data/generate_flatness_data.py"
        )
    dataset = FlatnessDataset(data_dir)
    print(f"[train] Dataset: {dataset}")

    print("[train] Loading all profiles into memory...")
    profiles, metadata = dataset.load_all_profiles()   # (N, W, L)
    N, W, L = profiles.shape
    print(f"[train] Loaded {N} coils  |  shape: ({W}, {L})  |  {W*L/1e6:.1f}M pts/coil")

    # ------------------------------------------------------------------
    # 2. Statistical feature extraction
    # ------------------------------------------------------------------
    print(f"[train] Extracting statistical features (rocket_variant={rocket_variant})...")
    features = extract_batch(profiles, rocket_variant=rocket_variant)
    # (N, F) — F=1105 with Group D temporal + Group E ROCKET/MiniRocket (1000 kernels)
    print(f"[train] Feature matrix: {features.shape}  (dtype={features.dtype})")

    # Save feature matrix
    feat_path = out / "feature_matrix.npz"
    np.savez_compressed(feat_path, features=features)
    print(f"[train] Features saved: {feat_path}")

    # ------------------------------------------------------------------
    # 3. Spatial features for visualisation
    # ------------------------------------------------------------------
    print("[train] Extracting spatial features (for UMAP / PCA)...")
    spatial_extractor = SpatialFeatureExtractor(
        target_shape=(32, 64), n_components=50, random_state=random_state,
    )
    spatial_embeddings = spatial_extractor.fit_transform(profiles)   # (N, 50)
    spatial_extractor.save(str(out / "spatial_pca.joblib"))
    print(f"[train] Spatial embeddings: {spatial_embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Elbow curve
    # ------------------------------------------------------------------
    print("[train] Computing inertia curve (k=2..12)...")
    inertia_dict = inertia_curve(features, k_range=(2, 12), random_state=random_state)
    plot_elbow(
        inertia_dict,
        output_path=str(plots_dir / "elbow_curve.png"),
        highlight_k=n_clusters,
    )
    print(f"[train] Elbow plot saved: {plots_dir / 'elbow_curve.png'}")

    # ------------------------------------------------------------------
    # 4a. Gap Statistic (Tibshirani et al. 2001) — principled k selection
    # ------------------------------------------------------------------
    print("[train] Computing Gap Statistic (Tibshirani 2001) for principled k selection...")
    # Single scaler shared with the downstream clusterer (fitted below).
    # Gap Statistic and clusterer MUST see the same normalised feature space,
    # otherwise Gap may recommend a k that the clusterer ultimately fits on
    # a subtly different representation.
    from sklearn.preprocessing import StandardScaler as _SharedScaler
    _shared_scaler = _SharedScaler()
    _features_scaled_shared = _shared_scaler.fit_transform(
        features.astype(np.float64)
    ).astype(np.float32)
    gap_result = gap_statistic(_features_scaled_shared, k_range=(2, 12), n_references=20,
                               random_state=random_state)
    gap_optimal_k = gap_result["optimal_k"]
    print(f"[train] Gap Statistic optimal k = {gap_optimal_k}  "
          f"(pipeline k = {n_clusters})")
    if gap_optimal_k != n_clusters:
        print(f"[train] WARNING: Gap Statistic recommends k={gap_optimal_k} but "
              f"pipeline is using k={n_clusters}.")
        print(f"[train]   Note: Gap statistic can underestimate k in high-dimensional "
              f"spaces (D={features.shape[1]}).")
        print(f"[train]   Consider running with --n_clusters {gap_optimal_k} or "
              f"inspecting the UMAP + algorithm comparison.")
    plot_gap_statistic(
        gap_result,
        output_path=str(plots_dir / "gap_statistic.png"),
    )

    # ------------------------------------------------------------------
    # 4b. Multi-algorithm comparison (Saas 2017, Ren 2021, Asadi 2021)
    # ------------------------------------------------------------------
    k_compare_max = max(12, n_clusters + 2)
    print(f"[train] Comparing clustering algorithms (k=3..{k_compare_max}) - KMeans / GMM / Agglomerative / MiniBatchKMeans...")
    try:
        comparison_df = compare_algorithms(features, k_range=(3, k_compare_max), random_state=random_state)
        plot_algorithm_comparison(
            comparison_df,
            output_path=str(plots_dir / "algorithm_comparison.png"),
            highlight_k=n_clusters,
        )
        comparison_df.to_csv(out / "algorithm_comparison.csv", index=False)
        print(f"[train] Algorithm comparison saved: {plots_dir / 'algorithm_comparison.png'}")
    except Exception as _exc:
        import traceback
        print(f"[train] Algorithm comparison skipped: {_exc}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 5. Fit clustering model
    # ------------------------------------------------------------------
    print(f"[train] Fitting CoilClusterer (method={method}, k={n_clusters})...")
    clusterer = CoilClusterer(
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    # Persist the feature-extraction variant so inference uses the SAME one
    # (prevents the silent rocket/minirocket feature-dim mismatch).
    clusterer.rocket_variant = rocket_variant
    clusterer.fit(features)
    labels = clusterer.cluster_labels_

    # Extract the StandardScaler-transformed space that the clusterer actually used.
    # Every metric and visualisation below must use this so they are consistent
    # with how the model sees the data (Saas et al. 2017 recommend computing all
    # cluster quality indices in the same normalised space used for fitting).
    features_scaled = clusterer._scaler.transform(
        features.astype(np.float64)
    ).astype(np.float32)

    # --- Three cluster quality metrics (Saas et al. 2017) ---
    sil_score = compute_silhouette(features_scaled, labels)
    db_score  = compute_davies_bouldin(features_scaled, labels)
    ch_score  = compute_calinski_harabasz(features_scaled, labels)

    # Bootstrap CI for silhouette (McInnes 2018 style)
    sil_bootstrap = bootstrap_silhouette(features_scaled, labels, n_bootstrap=100,
                                         sample_frac=0.8, random_state=random_state)

    # --- Optional: Consensus clustering comparison ---
    if method != "consensus":
        try:
            print("[train] Computing consensus clustering for comparison...")
            consensus_labels, _ = consensus_clustering(
                features_scaled, n_clusters=n_clusters,
                algorithms=["kmeans", "agglomerative", "gmm"],
                n_runs=20, random_state=random_state,
            )
            cons_sil = compute_silhouette(features_scaled, consensus_labels)
            print(f"[train] Consensus silhouette: {cons_sil:.4f}  "
                  f"(vs {method} silhouette: {sil_score:.4f})")
            if cons_sil > sil_score + 0.01:
                print(f"[train] NOTE: Consensus clustering outperforms {method} "
                      f"by {cons_sil - sil_score:.4f} silhouette points")
        except Exception as _e:
            import traceback
            print(f"[train] Consensus clustering skipped: {_e}")
            traceback.print_exc()

    print(f"\n[train] Cluster Quality Metrics (Saas et al. 2017):")
    print(f"  {'Metric':<28}  {'Value':>10}  {'Direction'}")
    print(f"  {'-'*52}")
    print(f"  {'Silhouette Score':<28}  {sil_score:>10.4f}  higher is better")
    print(f"  {'  Bootstrap 95% CI':<28}  [{sil_bootstrap['ci_low']:.4f}, "
          f"{sil_bootstrap['ci_high']:.4f}]  (n=100 resamples)")
    print(f"  {'Davies-Bouldin Index':<28}  {db_score:>10.4f}  lower is better")
    print(f"  {'Calinski-Harabász Score':<28}  {ch_score:>10.1f}  higher is better")

    # ── Quality Assessment ──────────────────────────────────────────────
    print(f"\n[train] === Quality Assessment ===")
    # Silhouette interpretation
    if sil_score < 0.20:
        print(f"  [WARN] Silhouette = {sil_score:.3f} (poor, < 0.20): "
              "clusters overlap significantly. Consider increasing k.")
    elif sil_score < 0.35:
        print(f"  [INFO] Silhouette = {sil_score:.3f} (moderate): "
              "some overlap between clusters.")
    else:
        print(f"  [OK]   Silhouette = {sil_score:.3f} (good): "
              "clusters are reasonably well-separated.")

    # Cluster size imbalance (P10)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    imbalance_ratio = float(label_counts.max()) / max(float(label_counts.min()), 1)
    print(f"  Cluster sizes: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
    if imbalance_ratio > 2.0:
        print(f"  [WARN] Imbalance ratio {imbalance_ratio:.1f}x > 2.0: "
              "largest cluster is much bigger than smallest. "
              "Rare defect types may be merged into large clusters.")

    # Near-merge warning (P11) — check inter-centroid distances
    if hasattr(clusterer, '_model') and hasattr(clusterer._model, 'cluster_centers_'):
        from scipy.spatial.distance import pdist, squareform
        centroids = clusterer._model.cluster_centers_
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

    # Gap mismatch in metrics
    if gap_optimal_k != n_clusters:
        print(f"  [INFO] Gap Statistic suggests k={gap_optimal_k} vs chosen k={n_clusters}")

    # NMI and ARI (ground truth comparison) — P23
    nmi_score = ari_score = None
    if "class_id" in metadata.columns:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        gt_labels = metadata["class_id"].values
        nmi_score = float(normalized_mutual_info_score(gt_labels, labels))
        ari_score = float(adjusted_rand_score(gt_labels, labels))
        print(f"[train] Ground truth comparison:")
        print(f"  NMI  = {nmi_score:.4f}  "
              f"({'good' if nmi_score >= 0.8 else 'moderate' if nmi_score >= 0.6 else 'poor'})")
        print(f"  ARI  = {ari_score:.4f}  "
              f"({'good' if ari_score >= 0.6 else 'moderate' if ari_score >= 0.3 else 'poor'})")

    # Cluster purity (P22)
    purity = None
    if "class_id" in metadata.columns:
        gt_labels = metadata["class_id"].values
        correct = 0
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                most_common = np.bincount(gt_labels[mask]).argmax()
                correct += (gt_labels[mask] == most_common).sum()
        purity = float(correct / len(labels))
        print(f"  Purity = {purity:.1%}  "
              f"({'good' if purity >= 0.7 else 'moderate' if purity >= 0.5 else 'poor — review clustering'})")

    # Save metrics to JSON
    metrics_dict = {
        "n_clusters":                 n_clusters,
        "silhouette_score":           round(sil_score, 6),
        "silhouette_bootstrap_mean":  round(sil_bootstrap["mean"], 6),
        "silhouette_bootstrap_ci_low": round(sil_bootstrap["ci_low"], 6),
        "silhouette_bootstrap_ci_high": round(sil_bootstrap["ci_high"], 6),
        "davies_bouldin_index":       round(db_score, 6),
        "calinski_harabasz":          round(ch_score, 2),
        "imbalance_ratio":            round(imbalance_ratio, 2),
        "gap_optimal_k":              gap_optimal_k,
    }
    if nmi_score is not None:
        metrics_dict["nmi"] = round(nmi_score, 6)
        metrics_dict["ari"] = round(ari_score, 6)
    if purity is not None:
        metrics_dict["purity"] = round(purity, 4)
    metrics_path = out / "cluster_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[train] Metrics saved: {metrics_path}")

    # Save model
    clusterer_path = out / "clusterer.joblib"
    clusterer.save(str(clusterer_path))

    # ------------------------------------------------------------------
    # 6. Cluster assignments CSV
    # ------------------------------------------------------------------
    assignments_df = metadata.copy()
    assignments_df["cluster_id"] = labels
    assignments_path = out / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"[train] Cluster assignments saved: {assignments_path}")

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    summary_df = cluster_summary(features_scaled, labels, metadata)
    print("\n[train] === Cluster Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out / "cluster_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 7b. Ward dendrogram (Saas et al. 2017 + NeurIPS 2019 flat-cut)
    # ------------------------------------------------------------------
    print("[train] Generating Ward dendrogram...")
    try:
        from flatness_clustering.clustering.metrics import plot_dendrogram
        plot_dendrogram(
            features_scaled=features_scaled,
            output_path=str(plots_dir / "dendrogram.png"),
            n_clusters=n_clusters,
        )
        print(f"[train] Dendrogram saved: {plots_dir / 'dendrogram.png'}")
    except Exception as _exc:
        import traceback
        print(f"[train] Dendrogram skipped: {_exc}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 8. Cluster heatmaps
    # ------------------------------------------------------------------
    print("\n[train] Generating cluster heatmaps...")
    plot_cluster_heatmaps(
        profiles=profiles,
        labels=labels,
        n_samples=3,
        output_path=str(plots_dir / "cluster_heatmaps.png"),
    )
    print(f"[train] Heatmaps saved: {plots_dir / 'cluster_heatmaps.png'}")

    # ------------------------------------------------------------------
    # 9b. UMAP visualisation (McInnes et al. 2018)
    # ------------------------------------------------------------------
    print("[train] Computing UMAP embedding (preserves global structure)...")
    try:
        # Cluster assignment UMAP — legend shows "Cluster 0 … k-1"
        plot_umap(
            features=features_scaled,
            labels=labels,
            output_path=str(plots_dir / "umap_clusters.png"),
            title="UMAP — Flatness Profile Cluster Assignments (scaled features)",
            n_neighbors=15,
            min_dist=0.1,
            random_state=random_state,
            label_names=None,   # show "Cluster N" labels (not pattern names)
        )
        print(f"[train] UMAP cluster plot saved: {plots_dir / 'umap_clusters.png'}")

        # Ground truth UMAP — diagnostic: one colour per flatness pattern class
        if "class_id" in metadata.columns:
            _PATTERN_NAMES = [
                "center_wave", "edge_wave", "symmetric_bow", "quarter_buckle", "flat",
                "random_noise", "longitudinal_bow", "complex_wave",
                "herringbone_wave", "periodic_mark",
            ]
            gt_label_names = {i: n for i, n in enumerate(_PATTERN_NAMES)}
            plot_umap(
                features=features_scaled,
                labels=metadata["class_id"].values,
                output_path=str(plots_dir / "umap_ground_truth.png"),
                title="UMAP — Ground Truth Pattern Classes",
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state,
                label_names=gt_label_names,
            )
            print(f"[train] UMAP ground truth saved: {plots_dir / 'umap_ground_truth.png'}")
    except ImportError:
        print("[train] UMAP skipped (install umap-learn: pip install umap-learn)")

    # ------------------------------------------------------------------
    # 10. Advanced clustering visualisations
    # ------------------------------------------------------------------
    print("[train] Generating advanced visualisations...")

    plot_feature_discriminability(
        features=features_scaled,
        labels=labels,
        output_path=str(plots_dir / "adv_feature_discriminability.png"),
    )
    plot_cluster_confusion(
        labels=labels,
        metadata=metadata,
        output_path=str(plots_dir / "adv_cluster_confusion.png"),
    )
    plot_parallel_coordinates(
        features=features_scaled,
        labels=labels,
        output_path=str(plots_dir / "adv_parallel_coords.png"),
    )
    plot_silhouette_distribution(
        features=features_scaled,
        labels=labels,
        output_path=str(plots_dir / "adv_silhouette_distribution.png"),
    )
    plot_cluster_distance_matrix(
        features=features_scaled,
        labels=labels,
        output_path=str(plots_dir / "adv_distance_matrix.png"),
    )
    if "timestamp" in metadata.columns:
        plot_temporal_evolution(
            metadata=metadata,
            labels=labels,
            output_path=str(plots_dir / "adv_temporal_evolution.png"),
        )
    print(f"[train] Advanced plots saved in: {plots_dir}")

    # ------------------------------------------------------------------
    # 11. PDF report
    # ------------------------------------------------------------------
    print("[train] Generating PDF report...")
    try:
        from flatness_clustering.clustering.report import generate_clustering_report
        algo_cmp_path = str(plots_dir / "algorithm_comparison.png")
        report_path = generate_clustering_report(
            features=features,          # raw — report.py scales internally for metrics
            labels=labels,
            metadata=metadata,
            output_path=str(out / "reports" / "clustering_report.pdf"),
            profiles=profiles,
            clusterer=clusterer,
            inertia_data=inertia_dict,
            tsne_2d=None,
            n_clusters=n_clusters,
            run_tag=f"method={method}  seed={random_state}",
            algorithm_comparison_path=algo_cmp_path if os.path.exists(algo_cmp_path) else None,
            gap_result=gap_result,
        )
        print(f"[train] PDF report: {report_path}")
    except Exception as _exc:
        print(f"[train] PDF report skipped: {_exc}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n[train] === Training Complete ===")
    print(f"  Coils               : {N}")
    print(f"  Features            : {features.shape[1]}  (ROCKET Group E — 500 kernels)")
    print(f"  Clusters            : {n_clusters}")
    print(f"  Silhouette          : {sil_score:.4f}  "
          f"95% CI [{sil_bootstrap['ci_low']:.4f}, {sil_bootstrap['ci_high']:.4f}]")
    print(f"  Davies-Bouldin      : {db_score:.4f}")
    print(f"  Calinski-Harabász   : {ch_score:.1f}")
    print(f"  Artifacts           : {out}")

    return {
        "n_coils":              N,
        "n_features":           int(features.shape[1]),
        "n_clusters":           n_clusters,
        "silhouette_score":     sil_score,
        "davies_bouldin_index": db_score,
        "calinski_harabasz":    ch_score,
        "output_dir":           str(out),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train flatness clustering model")
    p.add_argument("--data_dir",   default="flatness_clustering/data/synthetic_profiles")
    p.add_argument("--output_dir", default="flatness_clustering/outputs")
    p.add_argument("--n_clusters", type=int, default=14)
    p.add_argument("--method",     default="kmeans",
                   choices=["kmeans", "hdbscan", "gmm", "spectral", "agglomerative", "minibatch_kmeans"])
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        method=args.method,
        random_state=args.seed,
    )

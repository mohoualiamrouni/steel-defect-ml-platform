"""
Databricks Script 2 — Clustering Training
==========================================

Reads the coil feature table produced by 01_feature_extraction.py,
fits a CoilClusterer, and logs everything to MLflow.

Input:
    steeldefect_flatness.coil_features   (Delta Lake)

Output:
    steeldefect_flatness.cluster_assignments  (Delta Lake)
    MLflow run with:
        params:  n_clusters, method, random_state
        metrics: silhouette_score, n_coils, inertia (KMeans)
        artifact: clusterer.joblib

Usage in Databricks:
    Run after 01_feature_extraction.py has completed.
    Set widgets for n_clusters, method, and mlflow_experiment.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def run_clustering_training(
    feature_table:     str = "steeldefect_flatness.coil_features",
    assignment_table:  str = "steeldefect_flatness.cluster_assignments",
    n_clusters:        int = 6,
    method:            str = "kmeans",
    random_state:      int = 42,
    mlflow_experiment: str = "/steeldefect/flatness_clustering",
    artifact_dir:      str = "/dbfs/steeldefect/flatness_clustering/models",
) -> dict:
    """Fit clustering model on coil features and log to MLflow.

    Args:
        feature_table:    Delta table with 1105 feature columns + metadata.
        assignment_table: Output Delta table for cluster assignments.
        n_clusters:       K for KMeans.
        method:           'kmeans' or 'hdbscan'.
        random_state:     Random seed.
        mlflow_experiment: MLflow experiment path.
        artifact_dir:     DBFS path to save clusterer.joblib.

    Returns:
        Summary dict.
    """
    # Databricks: spark is injected
    global spark  # noqa: F821

    import os
    repo_root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.clustering.metrics import compute_silhouette
    from flatness_clustering.features.statistical_features import feature_names

    feat_names = feature_names()

    # ------------------------------------------------------------------
    # 1. Load features from Delta
    # ------------------------------------------------------------------
    print(f"[clustering_training] Reading: {feature_table}")
    df_spark = spark.table(feature_table)          # noqa: F821
    df_pandas = df_spark.toPandas()

    meta_cols = ["coil_id", "grade", "customer", "timestamp"]
    features  = df_pandas[feat_names].values.astype(np.float32)
    metadata  = df_pandas[meta_cols].copy()
    n_coils   = len(features)
    print(f"[clustering_training] {n_coils} coils  |  {features.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Fit clusterer
    # ------------------------------------------------------------------
    print(f"[clustering_training] Fitting {method} (k={n_clusters})...")
    clusterer = CoilClusterer(
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    clusterer.fit(features)
    labels = clusterer.cluster_labels_

    features_scaled = clusterer._scaler.transform(features.astype(np.float64))
    silhouette = compute_silhouette(features_scaled, labels)
    print(f"[clustering_training] Silhouette: {silhouette:.4f}")

    # ------------------------------------------------------------------
    # 3. Write assignments to Delta
    # ------------------------------------------------------------------
    assignments_df = metadata.copy()
    assignments_df["cluster_id"]  = labels
    assignments_df["assigned_at"] = datetime.utcnow().isoformat()

    assignments_spark = spark.createDataFrame(assignments_df)  # noqa: F821
    (
        assignments_spark
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(assignment_table)
    )
    print(f"[clustering_training] Assignments written: {assignment_table}")

    # ------------------------------------------------------------------
    # 4. Log to MLflow
    # ------------------------------------------------------------------
    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=f"clustering_{method}_k{n_clusters}"):
            # Params
            mlflow.log_param("n_clusters",    n_clusters)
            mlflow.log_param("method",        method)
            mlflow.log_param("random_state",  random_state)
            mlflow.log_param("n_coils",       n_coils)

            # Metrics
            mlflow.log_metric("silhouette_score", silhouette)
            if method == "kmeans" and clusterer._model is not None:
                mlflow.log_metric("kmeans_inertia", float(clusterer._model.inertia_))

            # Artifact: serialise clusterer and upload
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            clusterer_path = f"{artifact_dir}/clusterer.joblib"
            clusterer.save(clusterer_path)
            mlflow.log_artifact(clusterer_path, artifact_path="models")

            # Register the full CoilClusterer (scaler, estimator and its nearest-
            # centroid predict), not the inner estimator fitted on scaled features:
            # a model loaded from the registry must work on raw features.
            mlflow.sklearn.log_model(
                clusterer,
                artifact_path="sklearn_model",
                registered_model_name="steeldefect_flatness_clusterer",
            )

            run_id = mlflow.active_run().info.run_id
            print(f"[clustering_training] MLflow run: {run_id}")

    return {
        "n_coils":          n_coils,
        "n_clusters":       n_clusters,
        "method":           method,
        "silhouette_score": silhouette,
        "assignment_table": assignment_table,
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

def _local_test() -> None:
    """Test clustering on local synthetic data — no Databricks needed."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.clustering.metrics import compute_silhouette
    from flatness_clustering.data.generate_flatness_data import (
        CLASS_NAMES, GENERATORS,
    )
    from flatness_clustering.features.statistical_features import extract_batch

    rng = np.random.default_rng(0)
    profiles = []
    labels_gt = []
    for cls_id, (cls_name, gen) in enumerate(GENERATORS.items()):
        for _ in range(20):
            profiles.append(gen(rng, width=150, length=2000))
            labels_gt.append(cls_id)

    profiles_arr = np.stack(profiles, axis=0)
    features = extract_batch(profiles_arr)
    print(f"Feature matrix: {features.shape}")

    clusterer = CoilClusterer(method="kmeans", n_clusters=6)
    clusterer.fit(features)
    features_scaled = clusterer._scaler.transform(features.astype(np.float64))
    sil = compute_silhouette(features_scaled, clusterer.cluster_labels_)
    print(f"Silhouette score: {sil:.4f}")
    print(f"Cluster sizes: {np.bincount(clusterer.cluster_labels_)}")


if __name__ == "__main__":
    _local_test()

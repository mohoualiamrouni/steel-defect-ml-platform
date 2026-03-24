"""
Flatness Clustering — Cold-Rolling Mill
===============================================

Clusters steel coil flatness profiles (2D heatmaps of height measurements)
so that coils with similar flatness patterns are grouped together.

Key modules:
    data.generate_flatness_data  — synthetic coil profile generator
    data.flatness_dataset        — dataset loader (NPZ files)
    features.statistical_features — zone stats + FFT fingerprinting (~95 dims)
    features.spatial_features    — downsampled heatmap + PCA (for visualization)
    clustering.coil_clusterer    — KMeans / HDBSCAN + nearest-cluster assignment
    clustering.metrics           — silhouette, elbow, cluster quality plots
    pipeline.train_pipeline      — full training orchestration
    pipeline.inference_pipeline  — assign new coil to nearest cluster
    databricks/                  — PySpark + MLflow + Delta Lake scripts

Usage:
    # Generate synthetic data
    python flatness_clustering/data/generate_flatness_data.py

    # Train clustering model
    python flatness_clustering/pipeline/train_pipeline.py

    # Assign a new coil
    from flatness_clustering.pipeline.inference_pipeline import assign_new_coil
    result = assign_new_coil(profile_array)
"""

__version__ = "0.1.0"

"""Configuration dataclass for the defect clustering pipeline.

Centralises every tuneable parameter: data paths, feature extraction,
feature fusion, clustering algorithm selection, Gap-Statistic settings,
evaluation columns, and random seed.  Supports YAML persistence with a
JSON fallback when PyYAML is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class DefectClusteringConfig:
    """Pipeline-wide configuration for defect clustering.

    Parameters
    ----------
    cnn_features_pickle : str
        Path to the pickle file containing pre-extracted CNN feature vectors.
    masks_dir : str or None
        Directory of binary segmentation masks (one per image).  Set to
        ``None`` to skip segmentation-based features.
    metadata_csv : str or None
        Path to a CSV that maps each ``image_id`` to ``defect_type`` and
        ``expert_group``.
    output_dir : str
        Directory where pipeline artefacts (plots, tables, models) are saved.
    use_cnn_features : bool
        Whether to include CNN features in the fused representation.
    use_segmentation_features : bool
        Whether to include segmentation-derived features.
    seg_aggregation : str
        Aggregation strategy for mask-level statistics.
        ``"mean"`` or ``"mean_max"``.
    fusion_method : str
        How CNN and segmentation features are combined.
        ``"concat_scale"`` | ``"pca"`` | ``"weighted"``.
    pca_components : int or None
        Fixed number of PCA components.  Ignored when *pca_variance* is set
        and *pca_components* is ``None``.
    pca_variance : float
        Cumulative explained-variance threshold for PCA (0 < v <= 1).
    cnn_weight : float
        Multiplicative weight applied to CNN features before fusion.
    seg_weight : float
        Multiplicative weight applied to segmentation features before fusion.
    k_range : tuple[int, int]
        Inclusive range ``(k_min, k_max)`` explored by the Gap Statistic.
    n_clusters : int or None
        Fixed cluster count.  ``None`` means the pipeline will select *k*
        automatically via the Gap Statistic.
    method : str
        Clustering algorithm.  One of ``"kmeans"`` | ``"gmm"`` |
        ``"agglomerative"`` | ``"hdbscan"`` | ``"spectral"`` |
        ``"minibatch_kmeans"``.
    distance_metric : str
        Distance metric — ``"euclidean"`` or ``"cosine"``.
    gap_n_references : int
        Number of reference datasets to generate for the Gap Statistic.
    expert_group_col : str
        Column name in the metadata CSV that stores the expert-assigned group.
    defect_type_col : str
        Column name in the metadata CSV that stores the defect type label.
    random_state : int
        Global random seed for reproducibility.
    """

    # -- Data paths ----------------------------------------------------------
    cnn_features_pickle: str = "defect_clustering/data/synthetic/cnn_features.pkl"
    masks_dir: Optional[str] = "defect_clustering/data/synthetic/masks"
    metadata_csv: Optional[str] = "defect_clustering/data/synthetic/metadata.csv"
    output_dir: str = "defect_clustering/outputs"

    # -- Feature extraction --------------------------------------------------
    use_cnn_features: bool = True
    use_segmentation_features: bool = True
    seg_aggregation: str = "mean_max"  # mean | mean_max

    # -- Feature fusion ------------------------------------------------------
    fusion_method: str = "concat_scale"  # concat_scale | pca | weighted | cnn_only
    pca_components: Optional[int] = 50  # PCA-50 front-end; None = use pca_variance
    pca_variance: float = 0.95
    cnn_weight: float = 1.0
    seg_weight: float = 1.0

    # -- Clustering ----------------------------------------------------------
    k_range: Tuple[int, int] = (20, 60)
    n_clusters: Optional[int] = None  # None = use Gap Statistic
    method: str = "minibatch_kmeans"  # bisecting_merge | kmeans | gmm | agglomerative | hdbscan | spectral | minibatch_kmeans | dpgmm | deep_dpm
    target_k: Optional[int] = 28  # Force final cluster count (skip Gap); None = use Gap-optimal
    overcluster_factor: float = 1.0  # k'=factor*target_k for bisecting_merge (1.0 = flat BisectingKMeans)
    distance_metric: str = "cosine"  # euclidean | cosine  (cosine suits ReLU-pool CNN embeddings)

    # -- DeepDPM (Ronen et al. CVPR 2022) ------------------------------------
    deep_dpm_epochs: int = 150
    deep_dpm_alpha: float = 10.0        # DP concentration (higher = more clusters)
    deep_dpm_use_mlp: bool = True       # False = pure DPGMM (faster)

    # -- UMAP embedding for clustering (Ref 2/4: DECOR / VAE-style) ---------
    umap_embed_for_clustering: bool = False  # Use UMAP-reduced features for clustering
    umap_n_components: int = 50              # Target dimensionality

    # -- Outlier detection (DECOR: Isolation Forest + LOF) -------------------
    outlier_detection: bool = True
    outlier_contamination: float = 0.05      # Expected fraction of outliers per cluster
    outlier_strategy: str = "union"          # union | intersection

    # -- Gap Statistic -------------------------------------------------------
    gap_n_references: int = 8

    # -- Evaluation ----------------------------------------------------------
    expert_group_col: str = "expert_group"
    defect_type_col: str = "defect_type"

    # -- H2 ablation + re-grouping proposal (Subject 2 deliverable) ----------
    fusion_pca_variance: float = 0.95   # block-scale + PCA variance for the H2 fusion row
    ablation_n_bootstrap: int = 100     # bootstrap resamples for the NMI CIs
    merge_affinity_thresh: float = 0.35 # min co-clustering affinity to propose a merge

    # -- Azure Databricks / OBT platform integration ------------------------
    obt_table: str = "steeldefect.obt_segmentations"
    mlflow_model_name: str = "steeldefect_defect_clusterer"

    # -- Performance ---------------------------------------------------------
    fast_mode: bool = True  # Default to fast path; flip off with --full for paper-grade
    subsample_n: int = 8000  # Cap applied once at Step 1 (stratified on expert_group). 0 = keep all.
    enable_consensus: bool = False  # Step 8 consensus clustering is costly; off by default.
    cnn_pca_components: int = 50    # PCA front-end for cnn_only fusion. 0 = no PCA.
    n_jobs: int = 4          # Capped default: loky pickles full feature matrix per worker;
                             # 4 workers is a safe upper bound on 16 GB RAM machines.
                             # Override with --n-jobs on high-memory boxes.

    # -- General -------------------------------------------------------------
    random_state: int = 42

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def _as_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation, converting tuples to lists."""
        from dataclasses import asdict

        d = asdict(self)
        # Tuples serialise poorly in JSON / YAML — store as lists.
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    # -- YAML / JSON persistence --------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "DefectClusteringConfig":
        """Load configuration from a YAML (preferred) or JSON file.

        Parameters
        ----------
        path : str
            Filesystem path to a ``.yaml`` / ``.yml`` or ``.json`` file.

        Returns
        -------
        DefectClusteringConfig
            A new config instance populated from the file contents.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        RuntimeError
            If PyYAML is required but not installed.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        ext = os.path.splitext(path)[1].lower()

        if ext in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError(
                    "PyYAML is required to load .yaml configs.  "
                    "Install it with: pip install pyyaml"
                )
            data: Dict[str, Any] = yaml.safe_load(raw) or {}
        elif ext == ".json":
            data = json.loads(raw)
        else:
            # Try YAML first, then JSON.
            if _HAS_YAML:
                data = yaml.safe_load(raw) or {}
            else:
                data = json.loads(raw)

        # Convert list back to tuple for k_range.
        if "k_range" in data and isinstance(data["k_range"], list):
            data["k_range"] = tuple(data["k_range"])

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Persist configuration to a YAML (preferred) or JSON file.

        If PyYAML is not installed the file is written as JSON instead,
        regardless of the requested extension.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self._as_dict()

        ext = os.path.splitext(path)[1].lower()

        if _HAS_YAML and ext in (".yaml", ".yml", ""):
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(
                    data,
                    fh,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"{self.__class__.__name__}("]
        for key, value in self._as_dict().items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)

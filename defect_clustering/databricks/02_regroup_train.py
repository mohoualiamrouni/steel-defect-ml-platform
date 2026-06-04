"""Databricks job (Subject 2) — train the re-grouping model + write proposal.

Reads CNN + morphology features from the OBT, fuses them (block-scale + PCA),
clusters to K≈28, computes the H2 ablation (NMI with vs without morphology),
builds the metallurgical re-grouping proposal, writes ``steeldefect.defect_regrouping``
(carrying ``nmi_cnn_only`` / ``nmi_fusion`` — the H2 evidence), and registers
``steeldefect_defect_clusterer`` in the MLflow Model Registry.

Spark injected by Databricks.   run:  main(spark, run_id="2026-05", target_k=28)
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

REGROUP_TABLE = "steeldefect.defect_regrouping"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def write_defect_regrouping(spark, run_id, meta, labels, proposal, ablation,
                            model_uri):
    """Append one row per segmentation to ``steeldefect.defect_regrouping``."""
    import pandas as pd

    nmi_cnn = float(ablation.set_index("feature_set").loc["cnn_only", "nmi"])
    nmi_fus = float(ablation.set_index("feature_set").loc["fusion_block_pca", "nmi"])
    ari = float(ablation.set_index("feature_set").loc["fusion_block_pca", "ari"])
    gmap = proposal["group_map"]
    fam_to_id = {f: i for i, f in enumerate(sorted(set(gmap.values())))}
    rows = pd.DataFrame({
        "run_id": run_id,
        "segmentation_id": meta["segmentation_id"].values,
        "expert_group": meta["expert_group"].values,
        "cluster_id": labels.astype(int),
        "proposed_group_id": [fam_to_id[gmap[g]] for g in meta["expert_group"]],
        "confidence": 1.0,
        "is_outlier": False,
        "nmi_cnn_only": nmi_cnn,
        "nmi_fusion": nmi_fus,
        "ari": ari,
        "fusion_method": "concat_scale+pca",
        "model_uri": model_uri,
        "created_ts": _dt.datetime.utcnow().isoformat(),
    })
    spark.createDataFrame(rows).write.format("delta").mode("append").saveAsTable(REGROUP_TABLE)


def main(spark, run_id: str, target_k: int = 28, random_state: int = 42) -> dict:
    _bootstrap_path()
    import mlflow
    from steeldefect_platform import mlflow_registry
    from sklearn.cluster import KMeans

    from defect_clustering.analysis.group_analysis import run_ablation
    from defect_clustering.analysis.regrouping_proposal import (
        propose_regrouping, write_proposal,
    )
    from defect_clustering.clustering.outlier_detection import (
        detect_outliers_per_cluster,
    )
    from defect_clustering.features.feature_fusion import fuse_features

    read = _load_sibling("01_feature_read.py")
    cnn, seg, meta = read.read_obt_features(spark, random_state=random_state)

    fused, _ = fuse_features(cnn, seg, method="concat_scale", pca_variance=0.95,
                             random_state=random_state)
    labels = KMeans(n_clusters=target_k, n_init=10,
                    random_state=random_state).fit_predict(fused)

    ablation = run_ablation(cnn, seg, meta["expert_group"].values, k=target_k,
                            random_state=random_state)
    proposal = propose_regrouping(labels, meta, expert_group_col="expert_group")
    write_proposal(proposal, os.environ.get("STEELDEFECT_OUTPUT_DIR", "defect_clustering/outputs"))

    with mlflow.start_run(run_name=f"regroup_{run_id}") as run:
        mlflow.log_metric("nmi_before", proposal["nmi_before"])
        mlflow.log_metric("nmi_after", proposal["nmi_after"])
        mlflow.log_metric("n_merges", proposal["n_merges"])
        from sklearn.base import BaseEstimator
        model = KMeans(n_clusters=target_k, n_init=10,
                       random_state=random_state).fit(fused)
        model_uri = mlflow_registry.register(run, model, mlflow_registry.DEFECT_CLUSTERER)

    write_defect_regrouping(spark, run_id, meta, labels, proposal, ablation, model_uri)
    print(f"[regroup_train] K {len(set(meta['expert_group']))}->{proposal['k_proposed']}, "
          f"{proposal['n_merges']} merges, NMI {proposal['nmi_before']}->{proposal['nmi_after']}")
    return {"model_uri": model_uri, "proposal": proposal["k_proposed"]}


def _load_sibling(filename: str):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

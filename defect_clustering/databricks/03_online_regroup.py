"""Databricks job (Subject 2) — online assignment of new OBT rows.

Structured-streaming counterpart of ``02_regroup_train``: as new segmentations
land in the OBT, assign each to its nearest re-grouping cluster using the
MLflow-registered ``steeldefect_defect_clusterer`` and append the assignment to
``steeldefect.defect_regrouping``.  Mirrors the streaming block of
``flatness_clustering/databricks/03_online_assignment.py`` (availableNow trigger,
foreachBatch, MLflow Model Registry load).

    run:  run_online_regroup(spark, mode="streaming")
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

OBT_TABLE = "steeldefect.obt_segmentations"
REGROUP_TABLE = "steeldefect.defect_regrouping"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def _assign_batch(spark, batch_df, model, run_id):
    import numpy as np
    import pandas as pd
    from steeldefect_platform.obt_schema import MORPHO_NAMES

    pdf = batch_df.select("segmentation_id", "expert_group", "cnn_vector",
                          *MORPHO_NAMES).toPandas()
    if pdf.empty:
        return
    cnn = np.stack(pdf["cnn_vector"].apply(lambda v: np.asarray(v, np.float32)))
    morpho = pdf[MORPHO_NAMES].to_numpy(np.float32)
    from defect_clustering.features.feature_fusion import fuse_features
    fused, _ = fuse_features(cnn, np.hstack([morpho, morpho]),
                             method="concat_scale", pca_variance=0.95)
    labels = model.predict(fused)
    out = pd.DataFrame({
        "run_id": run_id,
        "segmentation_id": pdf["segmentation_id"].values,
        "expert_group": pdf["expert_group"].values,
        "cluster_id": labels.astype(int),
        "created_ts": _dt.datetime.utcnow().isoformat(),
    })
    (spark.createDataFrame(out).write.format("delta")
        .mode("append").saveAsTable(REGROUP_TABLE))


def run_online_regroup(spark, mode: str = "batch",
                       checkpoint_path: str = None, run_id: str = "online") -> None:
    _bootstrap_path()
    from steeldefect_platform import mlflow_registry
    model = mlflow_registry.load_production(mlflow_registry.DEFECT_CLUSTERER)

    if mode == "batch":
        df = spark.table(OBT_TABLE).where("cnn_vector IS NOT NULL")
        _assign_batch(spark, df, model, run_id)
        print("[online_regroup] batch assignment complete")
    elif mode == "streaming":
        if checkpoint_path is None:
            root = os.environ.get("STEELDEFECT_DBFS_ROOT", "/dbfs/steeldefect/defect_clustering")
            checkpoint_path = f"{root}/checkpoints/regroup"
        stream = spark.readStream.format("delta").table(OBT_TABLE)
        q = (stream.writeStream
             .option("checkpointLocation", checkpoint_path)
             .foreachBatch(lambda bdf, bid: _assign_batch(spark, bdf, model, run_id))
             .trigger(availableNow=True).start())
        q.awaitTermination()
        print("[online_regroup] streaming assignment complete")
    else:
        raise ValueError(f"mode must be 'batch' or 'streaming', got '{mode}'")

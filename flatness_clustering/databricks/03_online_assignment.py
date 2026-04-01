"""
Databricks Script 3 — Online Cluster Assignment
================================================

Reads newly produced coil profiles from a Delta Lake landing table,
assigns each coil to the nearest cluster using the MLflow-registered
CoilClusterer, and appends results to the assignments table.

Supports both:
  - Batch mode    : process all rows in incoming_profiles at once
  - Streaming mode: Delta Lake structured streaming (auto-loader trigger)

Input:
    steeldefect_flatness.incoming_profiles   (Delta Lake, same schema as raw_profiles)
    MLflow Model Registry: "steeldefect_flatness_clusterer" (latest Production version)

Output:
    steeldefect_flatness.cluster_assignments  (Delta Lake, append mode)

Usage in Databricks:
    # Batch
    run_online_assignment(mode="batch")

    # Streaming (continuous)
    run_online_assignment(mode="streaming")
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load_clusterer(model_uri: str = "models:/steeldefect_flatness_clusterer/Production"):
    """Load the CoilClusterer from MLflow Model Registry."""
    try:
        import mlflow
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load clusterer from MLflow ({model_uri}): {exc}\n"
            "Ensure 02_clustering_training.py has been run and the model is registered."
        ) from exc


def _assign_partition(
    pdf: pd.DataFrame,
    clusterer_path: str,
) -> pd.DataFrame:
    """Pandas UDF inner function: assign one partition of new coils."""
    import os
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.features.statistical_features import (
        extract_statistical_features,
    )

    clusterer = CoilClusterer.load(clusterer_path)

    results = []
    for _, row in pdf.iterrows():
        blob   = row["profile_blob"]
        width  = int(row["profile_width"])
        length = int(row["profile_length"])

        expected_bytes = width * length * 4
        if len(blob) != expected_bytes:
            raise ValueError(
                f"Blob size mismatch: expected {expected_bytes} bytes "
                f"({width}x{length} float32), got {len(blob)} bytes"
            )
        arr = np.frombuffer(blob, dtype="<f4").reshape(width, length)
        feat = extract_statistical_features(arr)
        assignment = clusterer.predict_with_confidence(feat)

        results.append({
            "coil_id":           row["coil_id"],
            "grade":             row["grade"],
            "customer":          row["customer"],
            "timestamp":         row["timestamp"],
            "cluster_id":        int(assignment["cluster_id"]),
            "confidence":        float(assignment["confidence"]),
            "is_outlier":        bool(assignment["is_outlier"]),
            "assigned_at":       datetime.utcnow().isoformat(),
        })

    return pd.DataFrame(results)


def run_online_assignment(
    incoming_table:    str  = "steeldefect_flatness.incoming_profiles",
    assignment_table:  str  = "steeldefect_flatness.cluster_assignments",
    clusterer_path:    str  = None,
    mode:              str  = "batch",         # "batch" | "streaming"
    checkpoint_path:   str  = None,
) -> None:
    """Assign incoming coil profiles to clusters.

    Args:
        incoming_table:   Delta table with new profiles (profile_blob, width, length, ...).
        assignment_table: Destination Delta table for assignments.
        clusterer_path:   DBFS path to the CoilClusterer joblib file.
                          Defaults to env var STEELDEFECT_CLUSTERER_PATH or /dbfs/steeldefect/... .
        mode:             'batch' or 'streaming'.
        checkpoint_path:  Checkpoint path for streaming mode.
                          Defaults to env var STEELDEFECT_CHECKPOINT_PATH or /dbfs/steeldefect/... .
    """
    import os

    _dbfs_root = os.environ.get("STEELDEFECT_DBFS_ROOT", "/dbfs/steeldefect/flatness_clustering")
    if clusterer_path is None:
        clusterer_path = f"{_dbfs_root}/models/clusterer.joblib"
    if checkpoint_path is None:
        checkpoint_path = f"{_dbfs_root}/checkpoints/assignment"

    global spark  # noqa: F821 — injected by Databricks

    from pyspark.sql.types import (
        BooleanType, DoubleType, StringType, StructField, StructType, TimestampType,
    )

    assignment_schema = StructType([
        StructField("coil_id",    StringType(), True),
        StructField("grade",      StringType(), True),
        StructField("customer",   StringType(), True),
        StructField("timestamp",  StringType(), True),
        StructField("cluster_id", StringType(), True),
        StructField("confidence", DoubleType(), True),
        StructField("is_outlier", BooleanType(), True),
        StructField("assigned_at", StringType(), True),
    ])

    assign_fn = lambda pdf: _assign_partition(pdf, clusterer_path)  # noqa: E731

    if mode == "batch":
        print(f"[online_assignment] Batch mode: reading {incoming_table}")
        incoming_df = spark.table(incoming_table)                    # noqa: F821
        n = incoming_df.count()
        print(f"[online_assignment] Processing {n} new coils...")

        assigned_df = (
            incoming_df
            .repartition(max(1, n // 50))
            .groupby("coil_id")
            .applyInPandas(assign_fn, schema=assignment_schema)
        )

        (
            assigned_df
            .write
            .format("delta")
            .mode("append")
            .saveAsTable(assignment_table)
        )
        print(f"[online_assignment] {n} coils appended to {assignment_table}")

    elif mode == "streaming":
        print(f"[online_assignment] Streaming mode: listening to {incoming_table}")
        incoming_stream = (
            spark                                                     # noqa: F821
            .readStream
            .format("delta")
            .table(incoming_table)
        )

        def foreach_batch_fn(batch_df, batch_id):
            n = batch_df.count()
            if n == 0:
                return
            assigned = (
                batch_df
                .groupby("coil_id")
                .applyInPandas(assign_fn, schema=assignment_schema)
            )
            assigned.write.format("delta").mode("append").saveAsTable(assignment_table)
            print(f"[online_assignment] Batch {batch_id}: assigned {n} coils")

        query = (
            incoming_stream
            .writeStream
            .option("checkpointLocation", checkpoint_path)
            .foreachBatch(foreach_batch_fn)
            .trigger(availableNow=True)   # process all available, then stop
            .start()
        )
        query.awaitTermination()
        print("[online_assignment] Streaming complete.")

    else:
        raise ValueError(f"mode must be 'batch' or 'streaming', got '{mode}'")


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

def _local_test() -> None:
    """Demonstrate inference on a synthetic coil — no Spark needed."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.data.generate_flatness_data import GENERATORS
    from flatness_clustering.pipeline.inference_pipeline import assign_new_coil

    rng = np.random.default_rng(42)
    clusterer_path = "flatness_clustering/outputs/clusterer.joblib"

    if not Path(clusterer_path).exists():
        print(f"Clusterer not found at '{clusterer_path}'.")
        print("Run: python flatness_clustering/pipeline/train_pipeline.py")
        return

    for pattern_name, gen in GENERATORS.items():
        profile = gen(rng, width=150, length=2000)
        result = assign_new_coil(profile, clusterer_path=clusterer_path)
        print(
            f"  [{pattern_name:15s}]  "
            f"cluster={result['cluster_id']}  "
            f"confidence={result['confidence']:.2f}  "
            f"outlier={result['is_outlier']}"
        )


if __name__ == "__main__":
    print("Local inference demo (no Spark required):\n")
    _local_test()

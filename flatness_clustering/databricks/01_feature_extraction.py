"""
Databricks Script 1 — Feature Extraction
==========================================

Reads raw flatness profiles from a Delta Lake table, extracts the 1105-dim
statistical feature vector for each coil, and writes a feature table.

Delta Lake schema assumed:
    steeldefect_flatness.raw_profiles
        coil_id      STRING
        grade        STRING
        customer     STRING
        timestamp    TIMESTAMP
        profile_blob BINARY   -- compressed numpy array (W * L float32 values)
        profile_width  INT    -- W
        profile_length INT    -- L

Output table:
    steeldefect_flatness.coil_features
        coil_id      STRING
        grade        STRING
        customer     STRING
        timestamp    TIMESTAMP
        + 1105 feature columns (see statistical_features.feature_names())

MLflow:
    Logs row count, feature schema, extraction timestamp.

Usage in Databricks:
    Attach this file to a cluster and run as a notebook or job.
    The `spark` and `mlflow` objects are provided by the Databricks runtime.
"""

# ---------------------------------------------------------------------------
# This script is designed to run in the Databricks environment.
# For local testing, mock PySpark with pandas equivalents (see bottom of file).
# ---------------------------------------------------------------------------

import io
import struct
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd

# Databricks runtime provides: spark, dbutils, display
# mlflow is pre-installed on Databricks ML runtimes
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature extraction UDF
# ---------------------------------------------------------------------------

def _decode_profile(blob: bytes, width: int, length: int) -> np.ndarray:
    """Decode a BINARY blob back to a (width, length) float32 numpy array.

    The blob is assumed to be a raw little-endian float32 buffer.
    For compressed NPZ, replace with:  np.load(io.BytesIO(blob))['profile']
    """
    expected_bytes = width * length * 4  # float32 = 4 bytes
    if len(blob) != expected_bytes:
        raise ValueError(
            f"Blob size mismatch: expected {expected_bytes} bytes "
            f"({width}x{length} float32), got {len(blob)} bytes"
        )
    arr = np.frombuffer(blob, dtype="<f4")   # little-endian float32
    return arr.reshape(width, length)


def extract_features_pandas_udf(
    coil_ids:      pd.Series,
    grades:        pd.Series,
    customers:     pd.Series,
    timestamps:    pd.Series,
    profile_blobs: pd.Series,
    widths:        pd.Series,
    lengths:       pd.Series,
) -> pd.DataFrame:
    """Pandas UDF that extracts statistical features from raw profile blobs.

    Signature follows PySpark's pandas_udf (GROUPED_MAP / applyInPandas style).
    Each call receives a partition of rows as pandas Series.

    Returns a DataFrame with coil metadata + 1105 feature columns.
    """
    import sys
    from pathlib import Path
    # Databricks: add the repo root to sys.path
    import os
    repo_root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from flatness_clustering.features.statistical_features import (
        extract_statistical_features,
        feature_names,
    )

    feat_names = feature_names()
    rows = []

    for i in range(len(coil_ids)):
        try:
            profile = _decode_profile(
                profile_blobs.iloc[i],
                int(widths.iloc[i]),
                int(lengths.iloc[i]),
            )
            feat_vec = extract_statistical_features(profile)
        except Exception as exc:
            # On error, fill with zeros to avoid dropping the coil
            print(f"[feature_extraction] WARNING: coil {coil_ids.iloc[i]} failed: {exc}")
            feat_vec = np.zeros(len(feat_names), dtype=np.float32)

        row = {
            "coil_id":   coil_ids.iloc[i],
            "grade":     grades.iloc[i],
            "customer":  customers.iloc[i],
            "timestamp": timestamps.iloc[i],
        }
        for name, val in zip(feat_names, feat_vec):
            row[name] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main Databricks job
# ---------------------------------------------------------------------------

def run_feature_extraction(
    input_table:  str = "steeldefect_flatness.raw_profiles",
    output_table: str = "steeldefect_flatness.coil_features",
    mlflow_experiment: str = "/steeldefect/flatness_clustering",
) -> None:
    """Read raw profiles, extract features, write to Delta Lake.

    Designed to run inside a Databricks notebook or job.
    """
    # spark is injected by Databricks runtime
    global spark  # noqa: F821

    print(f"[feature_extraction] Reading: {input_table}")
    raw_df = spark.table(input_table)                          # noqa: F821
    raw_df.printSchema()
    n_rows = raw_df.count()
    print(f"[feature_extraction] {n_rows} profiles to process")

    # Import feature names to define output schema
    from flatness_clustering.features.statistical_features import feature_names
    from pyspark.sql.types import (
        DoubleType, StringType, StructField, StructType, TimestampType,
    )

    out_schema = StructType(
        [
            StructField("coil_id",   StringType(),    True),
            StructField("grade",     StringType(),    True),
            StructField("customer",  StringType(),    True),
            StructField("timestamp", TimestampType(), True),
        ]
        + [StructField(name, DoubleType(), True) for name in feature_names()]
    )

    # Apply feature extraction via applyInPandas (one partition per coil batch)
    feature_df = (
        raw_df
        .repartition(200)   # tune to cluster size
        .groupby("coil_id")
        .applyInPandas(
            lambda pdf: extract_features_pandas_udf(
                pdf["coil_id"], pdf["grade"], pdf["customer"],
                pdf["timestamp"], pdf["profile_blob"],
                pdf["profile_width"], pdf["profile_length"],
            ),
            schema=out_schema,
        )
    )

    # Write to Delta Lake
    (
        feature_df
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(output_table)
    )
    print(f"[feature_extraction] Written: {output_table}  ({n_rows} rows)")

    # Log to MLflow
    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="feature_extraction"):
            mlflow.log_param("input_table",  input_table)
            mlflow.log_param("output_table", output_table)
            mlflow.log_metric("n_coils",     n_rows)
            mlflow.log_metric("n_features",  len(feature_names()))
            mlflow.set_tag("run_timestamp",  datetime.utcnow().isoformat())
        print("[feature_extraction] Logged to MLflow.")


# ---------------------------------------------------------------------------
# Local test (no Spark required)
# ---------------------------------------------------------------------------

def _local_test() -> None:
    """Quick sanity test using synthetic profiles — no Databricks needed."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from flatness_clustering.data.generate_flatness_data import GENERATORS
    from flatness_clustering.features.statistical_features import (
        extract_statistical_features, feature_names,
    )

    rng = np.random.default_rng(0)
    profile = GENERATORS["center_wave"](rng, width=150, length=2000)
    feats = extract_statistical_features(profile)
    names = feature_names()

    print(f"Feature vector shape: {feats.shape}")
    print(f"First 10 features:")
    for n, v in list(zip(names, feats))[:10]:
        print(f"  {n:40s}: {v:.4f}")


if __name__ == "__main__":
    # Run local test if executed directly
    _local_test()

"""Inspection → OBT ingestion (batch + structured streaming).

In the target deployment, an on-prem edge job at the plant runs the inspection
segmenter and pushes only segmentation *summaries* (keys, label, 960-d vector, 24
morphology features, position/density, ADLS URIs) into a Delta landing table
``steeldefect_landing.segmentations``, so the raw images never leave the plant.
This repository has no plant feed: its demos and seed jobs use calibrated-synthetic
rows (``source_version = "synthetic-cal-v1"``).

This module lands those summaries into ``steeldefect.obt_segmentations`` using the
same ``applyInPandas`` UDF + ``sys.path``/``STEELDEFECT_WORKSPACE_ROOT`` idiom as
``flatness_clustering/databricks/01_feature_extraction.py``.  ``upsert_severity``
is the Delta ``MERGE`` used by the datamining severity job to write predictions
back into the OBT.

All Spark imports are lazy so the module loads in a plain-Python context.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from .obt_schema import (
    CNN_DIM, MORPHO_NAMES, OBT_TABLE, SOURCE_SYNTHETIC, validate_row,
)


def _ensure_workspace_on_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


#: Extra columns the UDF appends so the driver can route rows after validation.
_DQ_COLS = ("_dq_valid", "_dq_reason")


def _udf_struct_type():
    """OBT struct + two data-quality routing columns (``_dq_valid``, ``_dq_reason``)."""
    from pyspark.sql.types import BooleanType, StringType, StructField
    from .obt_schema import obt_struct_type
    return (obt_struct_type()
            .add(StructField("_dq_valid", BooleanType(), False))
            .add(StructField("_dq_reason", StringType(), True)))


def _segmentation_to_obt_udf(pdf, source_version: str = SOURCE_SYNTHETIC):
    """applyInPandas inner fn: one landing partition → OBT rows TAGGED for quality.

    Each row is kept and tagged (``_dq_valid`` / ``_dq_reason``) rather than
    silently dropped, so the driver can dead-letter the rejects and enforce a
    batch-level data-quality gate. When a ``mask_uri`` is present and morphology
    columns are absent, the 24 features are (re)computed with the SAME extractor
    the rest of the stack uses, so the OBT columns match ``BASE_FEATURE_NAMES``.
    """
    import pandas as pd  # worker-side

    _ensure_workspace_on_path()
    cols = [f.name for f in _udf_struct_type().fields]
    rows = []
    for _, r in pdf.iterrows():
        d = r.to_dict()
        d.setdefault("cnn_dim", CNN_DIM)
        d.setdefault("source_version", source_version)
        if d.get("area") is None and d.get("mask_uri"):  # morphology backfill
            try:
                from defect_clustering.features.segmentation_features import (
                    extract_segmentation_features,
                )
                import numpy as np
                from .io_utils import decode_mask_uri
                mask = decode_mask_uri(d["mask_uri"])
                # An empty mask has no morphology: its columns stay null.
                if mask is not None and np.count_nonzero(mask):
                    feats = extract_segmentation_features(mask, aggregation="mean")  # (24,)
                    for name, val in zip(MORPHO_NAMES, np.asarray(feats).reshape(-1)):
                        d[name] = float(val)
            except Exception:
                pass  # leave morphology null; not fatal for ingestion
        try:
            validate_row(d)
            d["_dq_valid"], d["_dq_reason"] = True, None
        except ValueError as exc:
            d["_dq_valid"], d["_dq_reason"] = False, str(exc)
        rows.append(d)
    # reindex to the exact schema order so the pandas→Spark mapping is by name
    return pd.DataFrame(rows).reindex(columns=cols)


def _write_obt(spark, valid_df, obt_table: str) -> None:
    """Idempotent insert into the OBT via Delta ``MERGE`` on ``segmentation_id``.

    Re-ingesting the same batch does NOT duplicate rows (the gap a plain
    ``append`` left). Falls back to an append if the table is not yet a
    registered Delta table (first load) or Delta is unavailable.
    """
    try:
        from delta.tables import DeltaTable
        if spark.catalog.tableExists(obt_table):
            (DeltaTable.forName(spark, obt_table).alias("t")
                .merge(valid_df.alias("s"), "t.segmentation_id = s.segmentation_id")
                .whenNotMatchedInsertAll()
                .execute())
            return
    except Exception:
        pass  # fall through to append (Delta/table not available)
    valid_df.write.format("delta").mode("append").saveAsTable(obt_table)


def _handle_batch(spark, raw_df, obt_table: str, dlq_table: str,
                  source_version: str, max_bad_frac: float) -> int:
    """Validate → route → gate one (micro)batch; return rows written to the OBT."""
    from pyspark.sql import functions as F
    from .logging_utils import get_logger
    log = get_logger("steeldefect.ingestion")

    tagged = (raw_df.groupby("coil_id")
              .applyInPandas(lambda pdf: _segmentation_to_obt_udf(pdf, source_version),
                             schema=_udf_struct_type())).cache()
    total = tagged.count()
    if total == 0:
        return 0
    valid = tagged.filter(F.col("_dq_valid")).drop(*_DQ_COLS)
    invalid = tagged.filter(~F.col("_dq_valid"))
    n_bad = invalid.count()
    n_ok = total - n_bad

    if n_bad:  # dead-letter the rejects instead of dropping them
        (invalid.withColumn("dlq_ts", F.current_timestamp())
            .write.format("delta").mode("append")
            .option("mergeSchema", "true").saveAsTable(dlq_table))
    _write_obt(spark, valid, obt_table)

    bad_frac = n_bad / total
    log.info("ingest batch: %d ok, %d dead-lettered (%.2f%%)", n_ok, n_bad, 100 * bad_frac)
    if bad_frac > max_bad_frac:  # loud data-quality gate
        raise ValueError(
            f"data-quality gate failed: {bad_frac:.1%} invalid rows "
            f"(> {max_bad_frac:.1%}); rejects in dead-letter table {dlq_table}"
        )
    tagged.unpersist()
    return n_ok


def ingest_steeldefect_batch(spark, landing_table: str = "steeldefect_landing.segmentations",
                       obt_table: str = OBT_TABLE,
                       source_version: str = SOURCE_SYNTHETIC,
                       max_bad_frac: float = 0.02) -> int:
    """Batch-ingest the landing table into the OBT (idempotent MERGE + DQ gate)."""
    return _handle_batch(spark, spark.table(landing_table), obt_table,
                         f"{obt_table}_deadletter", source_version, max_bad_frac)


def ingest_steeldefect_stream(spark, landing_table: str = "steeldefect_landing.segmentations",
                        obt_table: str = OBT_TABLE,
                        checkpoint_path: Optional[str] = None,
                        source_version: str = SOURCE_SYNTHETIC,
                        trigger: str = "availableNow",
                        max_bad_frac: float = 0.02) -> None:
    """Structured-streaming ingestion: per-microbatch idempotent MERGE + DQ gate."""
    if checkpoint_path is None:
        root = os.environ.get("STEELDEFECT_DBFS_ROOT", "/dbfs/steeldefect/platform")
        checkpoint_path = f"{root}/checkpoints/obt_ingest"
    dlq = f"{obt_table}_deadletter"
    stream = spark.readStream.format("delta").table(landing_table)

    def _foreach_batch(batch_df, batch_id):
        _handle_batch(spark, batch_df, obt_table, dlq, source_version, max_bad_frac)

    q = (
        stream.writeStream
        .option("checkpointLocation", checkpoint_path)
        .foreachBatch(_foreach_batch)
        .trigger(availableNow=(trigger == "availableNow"))
        .start()
    )
    q.awaitTermination()


def upsert_severity(spark, severity_df, obt_table: str = OBT_TABLE) -> int:
    """MERGE predicted severity[,_lo,_hi] back into the OBT on segmentation_id.

    *severity_df* is a Spark DataFrame with columns
    ``segmentation_id, severity[, severity_lo, severity_hi]``.
    """
    from delta.tables import DeltaTable

    severity_df.createOrReplaceTempView("_sev_updates")
    cols = set(severity_df.columns)
    set_clause = ["t.severity = s.severity"]
    if "severity_lo" in cols:
        set_clause.append("t.severity_lo = s.severity_lo")
    if "severity_hi" in cols:
        set_clause.append("t.severity_hi = s.severity_hi")
    spark.sql(
        f"MERGE INTO {obt_table} t USING _sev_updates s "
        f"ON t.segmentation_id = s.segmentation_id "
        f"WHEN MATCHED THEN UPDATE SET {', '.join(set_clause)}"
    )
    return severity_df.count()

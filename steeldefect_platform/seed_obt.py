"""Seed the SteelDefect OBT with calibrated-synthetic data for a runnable demo.

Purpose: give a *real* Databricks deployment something to show, so the proof
screenshots (Delta history, MLflow A/B, Jobs run) have genuine content. The data
is synthetic but *structured*, not noise:

* CNN embeddings are drawn around a per-expert-group centroid, so the 960-D space
  actually clusters by group (P2 re-grouping becomes meaningful);
* morphology descriptors separate the groups and *drive* a latent severity, so
  the P3 severity regressor has real signal;
* ``local`` defects are written with severity left NULL on purpose, then filled
  by a real ``MERGE`` upsert — exactly the production path.

Every row carries ``source_version = synthetic-cal-v1`` (the honesty flag).

The write is split into several Delta operations so ``DESCRIBE HISTORY`` shows a
realistic lifecycle::

    CREATE TABLE  ->  WRITE (batch 1)  ->  WRITE (batch 2, later days)
                  ->  MERGE (severity upsert)  ->  OPTIMIZE (Z-ORDER)

Run on Databricks (notebook)::

    import sys; sys.path.insert(0, "/Workspace/Repos/<you>/the plant")
    from steeldefect_platform.seed_obt import seed
    summary = seed(spark, n_coils=40, with_mlflow=True)

or as a one-off task::

    python steeldefect_platform/seed_obt.py --coils 40 --with-mlflow
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steeldefect_platform.obt_schema import (  # noqa: E402
    CNN_DIM, MORPHO_NAMES, N_EXPERT_GROUPS, OBT_TABLE, SOURCE_SYNTHETIC,
    create_table, obt_struct_type, optimize_obt, validate_row,
)

# A small metallurgical taxonomy so expert_group names read realistically.
_GROUP_NAMES = [
    "inclusion", "sliver", "lamination", "shell", "scale", "scratch",
    "roll_mark", "edge_crack", "coil_break", "weld_defect", "buckle",
    "camber", "corrugation", "pinhole", "blister", "seam", "gouge",
    "dent", "stain", "oil_spot", "rust", "burr", "wave", "chatter",
    "hole", "crack", "pit", "contamination",
][:N_EXPERT_GROUPS]

_LINES = ["L1", "L2"]
_CUSTOMERS = ["AERO", "AUTO", "ENERGY"]
_GRADES = ["A", "B", "C"]


def _generate_rows(n_coils: int, seed_rng: int) -> List[Dict[str, Any]]:
    """Build OBT row dicts with structure (clustered embeddings, driven severity)."""
    import numpy as np

    rng = np.random.default_rng(seed_rng)
    cols = obt_struct_type().fieldNames()

    # per-group centroids: embeddings (960-D) and morphology (24-D) separate groups
    emb_centroids = rng.normal(0, 1.0, size=(N_EXPERT_GROUPS, CNN_DIM)).astype("float32")
    morph_centroids = rng.normal(0, 1.0, size=(N_EXPERT_GROUPS, len(MORPHO_NAMES)))
    # severity weights over a few morphology dims + a per-group base
    sev_w = rng.normal(0, 1.0, size=len(MORPHO_NAMES))
    sev_w[6:] = 0.0  # only the first shape descriptors drive severity (sparse, learnable)
    grp_sev_base = rng.uniform(-1.0, 1.0, size=N_EXPERT_GROUPS)

    base_day = _dt.date(2026, 5, 4)
    rows: List[Dict[str, Any]] = []

    for c in range(n_coils):
        coil_id = f"C{c:04d}"
        line_id = _LINES[c % len(_LINES)]
        day = base_day + _dt.timedelta(days=int(rng.integers(0, 12)))  # spread => partitions
        date_key = day.isoformat()
        customer = _CUSTOMERS[int(rng.integers(0, len(_CUSTOMERS)))]
        grade = _GRADES[int(rng.integers(0, len(_GRADES)))]
        campaign_id = f"campaign-2026-Q{1 + (day.month - 1) // 3}"
        coil_len = float(rng.uniform(500, 1500))
        coil_w = float(rng.uniform(1.0, 2.0))
        n_def = int(rng.integers(25, 70))

        for d in range(n_def):
            g = int(rng.integers(0, N_EXPERT_GROUPS))
            kind = "local" if rng.random() < 0.7 else "group"
            emb = (emb_centroids[g] + rng.normal(0, 0.35, CNN_DIM)).astype("float32")
            morph = morph_centroids[g] + rng.normal(0, 0.4, len(MORPHO_NAMES))
            seg_id = f"{coil_id}-{d:03d}"
            ts = _dt.datetime.combine(day, _dt.time(int(rng.integers(0, 24)),
                                                    int(rng.integers(0, 60))))

            row: Dict[str, Any] = {k: None for k in cols}
            row.update(
                segmentation_id=seg_id, coil_id=coil_id, line_id=line_id,
                defect_id=f"D{d:03d}", image_id=f"{coil_id}_{d:03d}.png",
                event_ts=ts, date_key=date_key, grade=grade, customer=customer,
                campaign_id=campaign_id, defect_kind=kind,
                defect_type=_GROUP_NAMES[g], expert_group=_GROUP_NAMES[g],
                expert_group_id=g, steeldefect_class_conf=float(rng.uniform(0.6, 0.99)),
                cnn_dim=CNN_DIM, cnn_vector=emb.tolist(),
                cnn_model_uri="models:/steeldefect_segmenter/Production",
                coil_length_m=coil_len, coil_width_m=coil_w,
                crop_uri=f"abfss://steeldefect@datalake/crops/{seg_id}.png",
                mask_uri=f"abfss://steeldefect@datalake/masks/{seg_id}.png",
                is_labeled=bool(rng.random() < 0.15), al_cycle=0,
                ingest_ts=_dt.datetime.now(), source_version=SOURCE_SYNTHETIC,
            )
            for name, val in zip(MORPHO_NAMES, morph):
                row[name] = float(val)

            if kind == "local":
                row.update(
                    pos_x_m=float(rng.uniform(0, coil_len)),
                    pos_y_m=float(rng.uniform(0, coil_w)),
                    bbox_row_min=int(rng.integers(0, 900)),
                    bbox_col_min=int(rng.integers(0, 900)),
                    bbox_row_max=int(rng.integers(900, 1024)),
                    bbox_col_max=int(rng.integers(900, 1024)),
                    # severity left NULL on purpose -> filled by the MERGE step
                )
            else:
                row.update(
                    cell_w_cm=int(rng.choice([10, 20, 50])),
                    cell_l_m=int(rng.choice([1, 2, 5])),
                    defect_count=int(rng.integers(5, 200)),
                    density_per_m2=float(rng.uniform(0.1, 8.0)),
                    coverage_frac=float(rng.uniform(0.01, 0.4)),
                )

            # latent severity (for the MERGE step) — driven by morphology + group
            row["_latent_sev"] = float(
                1.0 / (1.0 + np.exp(-(morph @ sev_w + grp_sev_base[g] + rng.normal(0, 0.3))))
            )
            validate_row(row)
            rows.append(row)

    return rows


def _to_obt_df(spark, rows: List[Dict[str, Any]]):
    """Build the OBT DataFrame (drops the helper column, casts date_key to DATE)."""
    from pyspark.sql import functions as F

    clean = [{k: v for k, v in r.items() if k != "_latent_sev"} for r in rows]
    df = spark.createDataFrame(clean, schema=obt_struct_type())
    return df.withColumn("date_key", F.to_date("date_key"))


def _severity_updates_df(spark, rows: List[Dict[str, Any]]):
    """severity (+ conformal-style band) for LOCAL defects, for the MERGE upsert."""
    import numpy as np
    from pyspark.sql.types import (FloatType, StringType, StructField,
                                   StructType)

    margin = 0.12  # CQR-style fixed half-width for the demo
    recs = [
        (r["segmentation_id"], float(r["_latent_sev"]),
         float(np.clip(r["_latent_sev"] - margin, 0, 1)),
         float(np.clip(r["_latent_sev"] + margin, 0, 1)))
        for r in rows if r["defect_kind"] == "local"
    ]
    schema = StructType([
        StructField("segmentation_id", StringType(), False),
        StructField("severity", FloatType(), True),
        StructField("severity_lo", FloatType(), True),
        StructField("severity_hi", FloatType(), True),
    ])
    return spark.createDataFrame(recs, schema=schema)


def _seed_ab_mlflow(spark, rows: List[Dict[str, Any]], model_name: str = "steeldefect_segmenter") -> bool:
    """Best-effort: log an A/B run and register two model versions. Returns success."""
    try:
        import mlflow
        import numpy as np
        from mlflow.tracking import MlflowClient
        from sklearn.linear_model import LogisticRegression

        X = np.array([r["cnn_vector"] for r in rows], dtype="float32")
        y = np.array([r["expert_group_id"] for r in rows])
        n = len(y)
        idx = np.random.default_rng(0).permutation(n)
        tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]

        # incumbent A trained on half the labels; challenger B on all of them
        half = tr[: len(tr) // 2]
        acc = {}
        for tag, sub in (("A", half), ("B", tr)):
            with mlflow.start_run(run_name=f"segmenter_{tag}"):
                clf = LogisticRegression(max_iter=200)
                clf.fit(X[sub], y[sub])
                a = float((clf.predict(X[te]) == y[te]).mean())
                acc[tag] = a
                mlflow.log_param("n_train", int(len(sub)))
                mlflow.log_metric("val_accuracy", a)
                mlflow.sklearn.log_model(clf, "model", registered_model_name=model_name)

        # A/B verdict logged as a run (mirrors ab_testing.promotion semantics)
        with mlflow.start_run(run_name="ab_decision"):
            delta = acc["B"] - acc["A"]
            decision = "promote_B" if delta > 0.02 else "keep_A"
            mlflow.log_metric("acc_A", acc["A"])
            mlflow.log_metric("acc_B", acc["B"])
            mlflow.log_metric("delta_acc", delta)
            mlflow.set_tag("decision", decision)

        # incumbent A -> Production so the registry shows a real stage transition
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            oldest = min(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(model_name, oldest.version, "Production")
        return True
    except Exception as exc:  # mlflow/sklearn absent, or no tracking server
        print(f"[seed] MLflow A/B step skipped: {exc}")
        return False


def seed(spark, n_coils: int = 40, table: str = OBT_TABLE,
         with_mlflow: bool = False, rng_seed: int = 7) -> Dict[str, Any]:
    """Create the OBT and seed it across several Delta operations.

    Returns a small summary dict (row count, partitions, history depth).
    """
    print(f"[seed] creating {table} (+ companions) ...")
    create_table(spark)

    rows = _generate_rows(n_coils, rng_seed)
    cut = int(0.6 * len(rows))
    batch1, batch2 = rows[:cut], rows[cut:]

    print(f"[seed] WRITE batch 1: {len(batch1)} rows ...")
    _to_obt_df(spark, batch1).write.format("delta").mode("append").saveAsTable(table)
    print(f"[seed] WRITE batch 2 (later days): {len(batch2)} rows ...")
    _to_obt_df(spark, batch2).write.format("delta").mode("append").saveAsTable(table)

    print("[seed] MERGE: upsert severity into LOCAL defects ...")
    from steeldefect_platform.ingestion import upsert_severity
    n_merged = upsert_severity(spark, _severity_updates_df(spark, rows), table)
    print(f"[seed]   merged severity for {n_merged} local defects")

    print("[seed] OPTIMIZE ... ZORDER BY ...")
    try:
        optimize_obt(spark, table)
    except Exception as exc:
        print(f"[seed]   OPTIMIZE skipped on this runtime: {exc}")

    ml_ok = _seed_ab_mlflow(spark, rows) if with_mlflow else False

    # summary
    total = spark.table(table).count()
    parts = spark.table(table).select("date_key", "line_id").distinct().count()
    try:
        hist = spark.sql(f"DESCRIBE HISTORY {table}").count()
    except Exception:
        hist = -1
    summary = {"rows": total, "partitions": parts, "history_versions": hist,
               "mlflow_ab": ml_ok, "source_version": SOURCE_SYNTHETIC}
    print(f"[seed] DONE: {summary}")
    print("[seed] next: DESCRIBE HISTORY %s  (screenshot #3)" % table)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed the SteelDefect OBT with synthetic data")
    parser.add_argument("--coils", type=int, default=40)
    parser.add_argument("--table", default=OBT_TABLE)
    parser.add_argument("--with-mlflow", action="store_true",
                        help="also log an A/B run and register two model versions")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    seed(spark, n_coils=args.coils, table=args.table,
         with_mlflow=args.with_mlflow, rng_seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Databricks notebook source
# MAGIC %md
# MAGIC # SteelDefect OBT seeder — self-contained (serverless-safe)
# MAGIC Inlines the seed logic so it imports ONLY installed packages
# MAGIC (numpy/sklearn/mlflow/pyspark) and never reads a workspace `.py` file —
# MAGIC the workspace-files FUSE read fails on this serverless workspace with
# MAGIC `409 BlobCustomerSpecifiedEncryptionMismatch`. Faithful to
# MAGIC `steeldefect_platform/seed_obt.py`: clustered 960-D embeddings, morphology that
# MAGIC drives a latent severity, and CREATE → WRITE → WRITE → MERGE → OPTIMIZE,
# MAGIC plus a UC-registry MLflow A/B. Every row carries the honesty flag.

# COMMAND ----------

# MAGIC %pip install scikit-learn
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import numpy as np
from pyspark.sql import functions as F

TABLE = "steeldefect.obt_segmentations"        # default catalog main, schema steeldefect
SRC = "synthetic-cal-v1"            # honesty flag
CNN_DIM = 960
N_GROUPS = 28
MORPHO = [
    "area", "perimeter", "eccentricity", "solidity", "extent",
    "major_axis_length", "minor_axis_length", "orientation",
    "equivalent_diameter", "bbox_aspect_ratio",
    "hu_moment_0", "hu_moment_1", "hu_moment_2", "hu_moment_3",
    "hu_moment_4", "hu_moment_5", "hu_moment_6",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_ASM", "compactness",
]
GROUP_NAMES = [
    "inclusion", "sliver", "lamination", "shell", "scale", "scratch",
    "roll_mark", "edge_crack", "coil_break", "weld_defect", "buckle",
    "camber", "corrugation", "pinhole", "blister", "seam", "gouge",
    "dent", "stain", "oil_spot", "rust", "burr", "wave", "chatter",
    "hole", "crack", "pit", "contamination",
][:N_GROUPS]
NAN = float("nan")

# COMMAND ----------

def gen_rows(n_coils=40, seed=7):
    rng = np.random.default_rng(seed)
    emb_c = rng.normal(0, 1.0, (N_GROUPS, CNN_DIM)).astype("float32")
    morph_c = rng.normal(0, 1.0, (N_GROUPS, len(MORPHO)))
    w = rng.normal(0, 1.0, len(MORPHO)); w[6:] = 0.0      # sparse, learnable
    gb = rng.uniform(-1.0, 1.0, N_GROUPS)
    base = dt.date(2026, 5, 4)
    rows = []
    for c in range(n_coils):
        coil_id = f"C{c:04d}"; line_id = ["L1", "L2"][c % 2]
        day = base + dt.timedelta(days=int(rng.integers(0, 12)))
        date_key = day.isoformat()
        camp = f"campaign-2026-Q{1 + (day.month - 1)//3}"
        clen = float(rng.uniform(500, 1500)); cw = float(rng.uniform(1.0, 2.0))
        for d in range(int(rng.integers(25, 70))):
            g = int(rng.integers(0, N_GROUPS))
            kind = "local" if rng.random() < 0.7 else "group"
            emb = (emb_c[g] + rng.normal(0, 0.35, CNN_DIM)).astype("float32")
            morph = morph_c[g] + rng.normal(0, 0.4, len(MORPHO))
            sid = f"{coil_id}-{d:03d}"
            ts = dt.datetime.combine(day, dt.time(int(rng.integers(0, 24)), int(rng.integers(0, 60))))
            r = dict(
                segmentation_id=sid, coil_id=coil_id, line_id=line_id,
                event_ts=ts, date_key=date_key, campaign_id=camp,
                defect_kind=kind, defect_type=GROUP_NAMES[g], expert_group=GROUP_NAMES[g],
                expert_group_id=g, steeldefect_class_conf=float(rng.uniform(0.6, 0.99)),
                cnn_dim=CNN_DIM, cnn_vector=[float(x) for x in emb],
                pos_x_m=float(rng.uniform(0, clen)) if kind == "local" else NAN,
                pos_y_m=float(rng.uniform(0, cw)) if kind == "local" else NAN,
                density_per_m2=NAN if kind == "local" else float(rng.uniform(0.1, 8.0)),
                severity=NAN, severity_lo=NAN, severity_hi=NAN,     # filled by MERGE
                coil_length_m=clen, coil_width_m=cw,
                is_labeled=bool(rng.random() < 0.15),
                ingest_ts=dt.datetime.now(), source_version=SRC,
            )
            for name, val in zip(MORPHO, morph):
                r[name] = float(val)
            # latent severity driven by morphology + group (for the MERGE step)
            r["_sev"] = float(1.0 / (1.0 + np.exp(-(morph @ w + gb[g] + rng.normal(0, 0.3)))))
            rows.append(r)
    return rows

rows = gen_rows(40)
print(f"generated {len(rows)} rows")

# COMMAND ----------

def to_df(recs):
    clean = [{k: v for k, v in r.items() if k != "_sev"} for r in recs]
    return spark.createDataFrame(clean).withColumn("date_key", F.to_date("date_key"))

cut = int(0.6 * len(rows))
# CREATE + WRITE (batch 1)
(to_df(rows[:cut]).write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").partitionBy("date_key", "line_id")
    .saveAsTable(TABLE))
# WRITE (batch 2, later)
to_df(rows[cut:]).write.format("delta").mode("append").saveAsTable(TABLE)
print("written 2 batches")

# COMMAND ----------

# MERGE: upsert severity (+ conformal band) into LOCAL defects
from delta.tables import DeltaTable
m = 0.12
recs = [(r["segmentation_id"], float(r["_sev"]),
         float(np.clip(r["_sev"] - m, 0, 1)), float(np.clip(r["_sev"] + m, 0, 1)))
        for r in rows if r["defect_kind"] == "local"]
upd = spark.createDataFrame(
    recs, "segmentation_id string, severity double, severity_lo double, severity_hi double")
(DeltaTable.forName(spark, TABLE).alias("t")
    .merge(upd.alias("s"), "t.segmentation_id = s.segmentation_id")
    .whenMatchedUpdate(set={"severity": "s.severity",
                            "severity_lo": "s.severity_lo",
                            "severity_hi": "s.severity_hi"})
    .execute())
print(f"merged severity for {len(recs)} local defects")

# COMMAND ----------

# OPTIMIZE ... ZORDER BY (file compaction + data-skipping layout)
spark.sql(f"OPTIMIZE {TABLE} ZORDER BY (coil_id, expert_group_id, defect_kind)")
print("optimized")

# COMMAND ----------
# NOTE: the seed deliberately does NOT register models or log accuracy. The
# model registry (versions + champion/challenger) is handled by nb_ab as a pure
# MLOps-infrastructure demo, and real model performance lives in the H1-H4
# results. Accuracy on this synthetic data would not be a meaningful result.

# COMMAND ----------

total = spark.table(TABLE).count()
parts = spark.table(TABLE).select("date_key", "line_id").distinct().count()
hist = spark.sql(f"DESCRIBE HISTORY {TABLE}").count()
print({"rows": total, "partitions": parts, "history_versions": hist, "source_version": SRC})
display(spark.sql(f"DESCRIBE HISTORY {TABLE}").select("version", "timestamp", "operation"))

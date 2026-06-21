# Databricks notebook source
# MAGIC %md
# MAGIC # Shared helpers for the serverless steeldefect-obt-daily notebooks
# MAGIC `%run` from each task notebook. Self-contained (numpy + spark only) so it
# MAGIC never reads a workspace `.py` file (which fails with a 409 encryption
# MAGIC error on this serverless workspace). Mirrors `steeldefect_platform/seed_obt.py`.

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


def gen_rows(n_coils=10, seed=99, day0=None):
    """Synthetic OBT rows: clustered 960-D embeddings + morphology, severity NULL
    (filled later by the severity MERGE)."""
    rng = np.random.default_rng(seed)
    emb_c = rng.normal(0, 1.0, (N_GROUPS, CNN_DIM)).astype("float32")
    morph_c = rng.normal(0, 1.0, (N_GROUPS, len(MORPHO)))
    base = day0 or dt.date(2026, 5, 16)
    rows = []
    for c in range(n_coils):
        coil_id = f"D{seed:02d}{c:04d}"; line_id = ["L1", "L2"][c % 2]
        day = base + dt.timedelta(days=int(rng.integers(0, 10)))
        date_key = day.isoformat()
        camp = f"campaign-2026-Q{1 + (day.month - 1)//3}"
        clen = float(rng.uniform(500, 1500)); cw = float(rng.uniform(1.0, 2.0))
        for d in range(int(rng.integers(25, 70))):
            g = int(rng.integers(0, N_GROUPS))
            kind = "local" if rng.random() < 0.7 else "group"
            emb = (emb_c[g] + rng.normal(0, 0.35, CNN_DIM)).astype("float32")
            morph = morph_c[g] + rng.normal(0, 0.4, len(MORPHO))
            ts = dt.datetime.combine(day, dt.time(int(rng.integers(0, 24)), int(rng.integers(0, 60))))
            r = dict(
                segmentation_id=f"{coil_id}-{d:03d}", coil_id=coil_id, line_id=line_id,
                event_ts=ts, date_key=date_key, campaign_id=camp,
                defect_kind=kind, defect_type=GROUP_NAMES[g], expert_group=GROUP_NAMES[g],
                expert_group_id=g, steeldefect_class_conf=float(rng.uniform(0.6, 0.99)),
                cnn_dim=CNN_DIM, cnn_vector=[float(x) for x in emb],
                pos_x_m=float(rng.uniform(0, clen)) if kind == "local" else NAN,
                pos_y_m=float(rng.uniform(0, cw)) if kind == "local" else NAN,
                density_per_m2=NAN if kind == "local" else float(rng.uniform(0.1, 8.0)),
                severity=NAN, severity_lo=NAN, severity_hi=NAN,
                coil_length_m=clen, coil_width_m=cw,
                is_labeled=bool(rng.random() < 0.15),
                ingest_ts=dt.datetime.now(), source_version=SRC,
            )
            for name, val in zip(MORPHO, morph):
                r[name] = float(val)
            rows.append(r)
    return rows


def to_df(recs):
    return spark.createDataFrame(recs).withColumn("date_key", F.to_date("date_key"))

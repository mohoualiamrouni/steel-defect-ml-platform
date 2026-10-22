"""Single source of truth for the SteelDefect One Big Table (OBT) schema.

The OBT (``steeldefect.obt_segmentations``) is the shared Delta Lake substrate of the
three featured subjects of the SteelDefect ecosystem:

* Subject 1 (Active Learning + A/B) reads the unlabeled pool and writes
  selected candidates / promotion decisions;
* Subject 2 (defect re-grouping) reads the CNN + morphology features and writes
  the proposed taxonomy;
* Subject 3 (datamining + cutting) reads local defects, writes severity back,
  and produces the cutting plan.

Design decision (see report appendix ``app:obt_platform``): ONE physical table
with a ``defect_kind ∈ {'local','group'}`` discriminator, rather than two
tables.  ``local`` defects carry an exact ``(pos_x_m, pos_y_m)``; ``group`` /
background defects carry a per-cell density and leave the position null.  The
960-d CNN embedding and the 24 ``regionprops``/OpenCV morphology features are
shared columns (≈90 % of the schema), so a single table avoids duplicating the
schema and the Z-ORDER tuning, while the discriminator keeps predicate pushdown
clean.

This module imports no Spark at import time; :func:`obt_struct_type` and
:func:`create_table` import ``pyspark`` lazily so the constants and the DDL
string remain usable in a plain-Python (no-cluster) context — exactly the
local-fallback idiom used in ``flatness_clustering/databricks/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants — bound to the real code, not invented
# ---------------------------------------------------------------------------

OBT_TABLE: str = "steeldefect.obt_segmentations"

#: Embedding dimension of the in-line inspection CNN (960-D, as in the report).
#: Matches ``defect_clustering.data.generate_synthetic_data`` (cnn_dim=960). The
#: public NEU-DET run uses an ImageNet EfficientNet-B0 instead (GAP + L2, 1280-D).
CNN_DIM: int = 960

#: The 24 per-region morphology descriptors, in EXACTLY the order produced by
#: ``defect_clustering.features.segmentation_features.BASE_FEATURE_NAMES``
#: (10 shape + 7 log-Hu + 6 GLCM + 1 compactness).  Kept as a literal so this
#: module never imports the heavy feature stack; cross-checked at runtime by
#: :func:`assert_morpho_names_match` when that module is importable.
MORPHO_NAMES: List[str] = [
    # shape (10)
    "area", "perimeter", "eccentricity", "solidity", "extent",
    "major_axis_length", "minor_axis_length", "orientation",
    "equivalent_diameter", "bbox_aspect_ratio",
    # log-Hu moments (7)
    "hu_moment_0", "hu_moment_1", "hu_moment_2", "hu_moment_3",
    "hu_moment_4", "hu_moment_5", "hu_moment_6",
    # GLCM Haralick (6)
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_ASM",
    # compactness (1)
    "compactness",
]
N_MORPHO: int = len(MORPHO_NAMES)  # 24

#: Number of top-level expert groups (the clustering target K). The taxonomy is
#: a two-level hierarchy: 28 expert groups over ~100 individual subtypes. There
#: is no "56" — see the report's 28/56 reconciliation.
N_EXPERT_GROUPS: int = 28

DEFECT_KINDS = ("local", "group")

#: source_version values: the honesty flag distinguishing calibrated-synthetic
#: data (which the thesis uses) from a real production feed.
SOURCE_SYNTHETIC = "synthetic-cal-v1"
SOURCE_PROD_PREFIX = "steeldefect-prod-"


def assert_morpho_names_match() -> None:
    """Best-effort check that MORPHO_NAMES == the real BASE_FEATURE_NAMES.

    Raises ``AssertionError`` on mismatch; silently returns if the
    ``defect_clustering`` feature module is not importable (e.g. on a minimal
    Spark worker).
    """
    try:
        from defect_clustering.features.segmentation_features import (
            BASE_FEATURE_NAMES,
        )
    except Exception:
        return
    assert list(BASE_FEATURE_NAMES) == MORPHO_NAMES, (
        "OBT MORPHO_NAMES drifted from segmentation_features.BASE_FEATURE_NAMES"
    )


# ---------------------------------------------------------------------------
# Spark StructType (lazy pyspark import)
# ---------------------------------------------------------------------------

def obt_struct_type():
    """Return the ``StructType`` for ``steeldefect.obt_segmentations``.

    Lazily imports ``pyspark.sql.types`` so this module is importable without
    Spark.  The column order mirrors :data:`OBT_DDL`.
    """
    from pyspark.sql.types import (
        ArrayType, BooleanType, DoubleType, FloatType, IntegerType,
        StringType, StructField, StructType, TimestampType,
    )

    fields: List[Any] = [
        # identity / lineage
        StructField("segmentation_id", StringType(), False),
        StructField("coil_id", StringType(), False),
        StructField("line_id", StringType(), False),
        StructField("defect_id", StringType(), True),
        StructField("image_id", StringType(), True),
        # time / process
        StructField("event_ts", TimestampType(), False),
        StructField("date_key", StringType(), False),  # 'YYYY-MM-DD' (partition)
        StructField("grade", StringType(), True),
        StructField("customer", StringType(), True),
        StructField("campaign_id", StringType(), True),
        # taxonomy
        StructField("defect_kind", StringType(), False),
        StructField("defect_type", StringType(), True),
        StructField("expert_group", StringType(), True),
        StructField("expert_group_id", IntegerType(), True),
        StructField("steeldefect_class_conf", FloatType(), True),
        # CNN embedding (by value)
        StructField("cnn_dim", IntegerType(), True),
        StructField("cnn_vector", ArrayType(FloatType()), True),
        StructField("cnn_model_uri", StringType(), True),
    ]
    # 24 morphology columns
    fields += [StructField(name, DoubleType(), True) for name in MORPHO_NAMES]
    fields += [
        # LOCAL-only
        StructField("pos_x_m", DoubleType(), True),
        StructField("pos_y_m", DoubleType(), True),
        StructField("bbox_row_min", IntegerType(), True),
        StructField("bbox_col_min", IntegerType(), True),
        StructField("bbox_row_max", IntegerType(), True),
        StructField("bbox_col_max", IntegerType(), True),
        StructField("severity", FloatType(), True),
        StructField("severity_lo", FloatType(), True),
        StructField("severity_hi", FloatType(), True),
        # GROUP-only
        StructField("cell_w_cm", IntegerType(), True),
        StructField("cell_l_m", IntegerType(), True),
        StructField("defect_count", IntegerType(), True),
        StructField("density_per_m2", DoubleType(), True),
        StructField("coverage_frac", DoubleType(), True),
        # geometry / refs
        StructField("coil_length_m", DoubleType(), True),
        StructField("coil_width_m", DoubleType(), True),
        StructField("crop_uri", StringType(), True),
        StructField("mask_uri", StringType(), True),
        # AL / governance
        StructField("is_labeled", BooleanType(), True),
        StructField("al_cycle", IntegerType(), True),
        StructField("ingest_ts", TimestampType(), True),
        StructField("source_version", StringType(), True),
    ]
    return StructType(fields)


# ---------------------------------------------------------------------------
# DDL (string) — full listing lives in report appendix app:obt_platform
# ---------------------------------------------------------------------------

OBT_DDL: str = f"""
CREATE TABLE IF NOT EXISTS {OBT_TABLE} (
  segmentation_id   STRING   NOT NULL,
  coil_id           STRING   NOT NULL,
  line_id           STRING   NOT NULL,
  defect_id         STRING,
  image_id          STRING,
  event_ts          TIMESTAMP NOT NULL,
  date_key          DATE      NOT NULL,
  grade             STRING,
  customer          STRING,
  campaign_id       STRING,
  defect_kind       STRING   NOT NULL,          -- 'local' | 'group'
  defect_type       STRING,
  expert_group      STRING,
  expert_group_id   INT,                        -- 0..27
  steeldefect_class_conf  FLOAT,
  cnn_dim           INT,                         -- = 960
  cnn_vector        ARRAY<FLOAT>,
  cnn_model_uri     STRING,
  area DOUBLE, perimeter DOUBLE, eccentricity DOUBLE, solidity DOUBLE,
  extent DOUBLE, major_axis_length DOUBLE, minor_axis_length DOUBLE,
  orientation DOUBLE, equivalent_diameter DOUBLE, bbox_aspect_ratio DOUBLE,
  hu_moment_0 DOUBLE, hu_moment_1 DOUBLE, hu_moment_2 DOUBLE, hu_moment_3 DOUBLE,
  hu_moment_4 DOUBLE, hu_moment_5 DOUBLE, hu_moment_6 DOUBLE,
  glcm_contrast DOUBLE, glcm_dissimilarity DOUBLE, glcm_homogeneity DOUBLE,
  glcm_energy DOUBLE, glcm_correlation DOUBLE, glcm_ASM DOUBLE,
  compactness DOUBLE,
  pos_x_m DOUBLE, pos_y_m DOUBLE,
  bbox_row_min INT, bbox_col_min INT, bbox_row_max INT, bbox_col_max INT,
  severity FLOAT, severity_lo FLOAT, severity_hi FLOAT,
  cell_w_cm INT, cell_l_m INT, defect_count INT,
  density_per_m2 DOUBLE, coverage_frac DOUBLE,
  coil_length_m DOUBLE, coil_width_m DOUBLE,
  crop_uri STRING, mask_uri STRING,
  is_labeled BOOLEAN, al_cycle INT, ingest_ts TIMESTAMP, source_version STRING
)
USING DELTA
PARTITIONED BY (date_key, line_id)
TBLPROPERTIES (
  delta.autoOptimize.optimizeWrite   = true,
  delta.autoOptimize.autoCompact     = true,
  delta.dataSkippingNumIndexedCols   = 32,
  delta.logRetentionDuration         = 'interval 30 days',
  delta.deletedFileRetentionDuration = 'interval 7 days'
)
""".strip()

#: Companion tables written back by the pipelines (small, not "big").
COMPANION_DDL: Dict[str, str] = {
    "steeldefect.al_candidates": (
        "CREATE TABLE IF NOT EXISTS steeldefect.al_candidates ("
        " segmentation_id STRING, image_id STRING, al_cycle INT, strategy STRING,"
        " score_uncertainty DOUBLE, score_diversity DOUBLE, score_anomaly DOUBLE,"
        " score_total DOUBLE, rank INT, selected BOOLEAN, model_uri STRING,"
        " created_ts TIMESTAMP) USING DELTA PARTITIONED BY (al_cycle)"
    ),
    "steeldefect.ab_decisions": (
        "CREATE TABLE IF NOT EXISTS steeldefect.ab_decisions ("
        " ab_run_id STRING, model_a_uri STRING, model_b_uri STRING,"
        " eval_dataset_id STRING, metric STRING, welch_t DOUBLE, welch_p DOUBLE,"
        " mannwhitney_p DOUBLE, mcnemar_p DOUBLE, bh_reject BOOLEAN,"
        " cohen_d DOUBLE, ci_low DOUBLE, ci_high DOUBLE, srm_p DOUBLE,"
        " prob_b_better DOUBLE, decision STRING, promoted_model_uri STRING,"
        " decided_ts TIMESTAMP) USING DELTA"
    ),
    "steeldefect.defect_regrouping": (
        "CREATE TABLE IF NOT EXISTS steeldefect.defect_regrouping ("
        " run_id STRING, segmentation_id STRING, expert_group STRING,"
        " cluster_id INT, proposed_group_id INT, confidence DOUBLE,"
        " is_outlier BOOLEAN, nmi_cnn_only DOUBLE, nmi_fusion DOUBLE, ari DOUBLE,"
        " fusion_method STRING, model_uri STRING, created_ts TIMESTAMP)"
        " USING DELTA PARTITIONED BY (run_id)"
    ),
    "steeldefect.cutting_plans": (
        "CREATE TABLE IF NOT EXISTS steeldefect.cutting_plans ("
        " plan_id STRING, coil_id STRING, n_pieces INT, total_value DOUBLE,"
        " grade_A_frac DOUBLE, grade_B_frac DOUBLE, grade_C_frac DOUBLE,"
        " scrap_frac DOUBLE, yield_frac DOUBLE, n_uncertain_pieces INT,"
        " cqr_coverage DOUBLE, solver STRING, model_uri STRING,"
        " created_ts TIMESTAMP) USING DELTA"
    ),
    "steeldefect.cutting_plan_pieces": (
        "CREATE TABLE IF NOT EXISTS steeldefect.cutting_plan_pieces ("
        " plan_id STRING, coil_id STRING, piece_idx INT,"
        " x0_m DOUBLE, x1_m DOUBLE, y0_m DOUBLE, y1_m DOUBLE, area_m2 DOUBLE,"
        " grade STRING, piece_value DOUBLE, mean_severity DOUBLE,"
        " sev_lo DOUBLE, sev_hi DOUBLE, grade_uncertain BOOLEAN) USING DELTA"
    ),
    "steeldefect.novelty_candidates": (
        "CREATE TABLE IF NOT EXISTS steeldefect.novelty_candidates ("
        " segmentation_id STRING, image_id STRING, encoder_uri STRING, scorer STRING,"
        " novelty_score DOUBLE, is_novel_flag BOOLEAN, proposed_family_id INT,"
        " run_id STRING, source_version STRING, created_ts TIMESTAMP)"
        " USING DELTA PARTITIONED BY (run_id)"
    ),
}


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------

def validate_row(d: Dict[str, Any]) -> None:
    """Validate a single OBT row dict; raise ``ValueError`` on violation.

    Enforces the invariants the schema relies on: a 960-d CNN vector, a valid
    ``defect_kind``, ``local`` ⇒ position present, ``group`` ⇒ density present,
    and ``expert_group_id`` in ``[0, 27]`` when present.
    """
    kind = d.get("defect_kind")
    if kind not in DEFECT_KINDS:
        raise ValueError(f"defect_kind must be in {DEFECT_KINDS}, got {kind!r}")

    vec = d.get("cnn_vector")
    if vec is not None:
        if len(vec) != CNN_DIM:
            raise ValueError(f"cnn_vector length {len(vec)} != {CNN_DIM}")
        if d.get("cnn_dim") not in (None, CNN_DIM):
            raise ValueError(f"cnn_dim {d.get('cnn_dim')} != {CNN_DIM}")

    egid = d.get("expert_group_id")
    if egid is not None and not (0 <= int(egid) < N_EXPERT_GROUPS):
        raise ValueError(f"expert_group_id {egid} out of [0,{N_EXPERT_GROUPS - 1}]")

    if kind == "local":
        if d.get("pos_x_m") is None or d.get("pos_y_m") is None:
            raise ValueError("local defect requires pos_x_m and pos_y_m")
    elif kind == "group":
        if d.get("density_per_m2") is None:
            raise ValueError("group defect requires density_per_m2")


def create_table(spark, table: str = OBT_TABLE, with_companions: bool = True) -> None:
    """Create the OBT (and optionally the companion tables) if absent."""
    spark.sql(OBT_DDL if table == OBT_TABLE else OBT_DDL.replace(OBT_TABLE, table))
    if with_companions:
        for ddl in COMPANION_DDL.values():
            spark.sql(ddl)


#: Columns clustered by the periodic Z-ORDER maintenance pass. These are the
#: frequent point/range predicates of the three consumers (a coil at a time,
#: a given expert group, a given defect kind) — distinct from the partition
#: keys ``(date_key, line_id)``. Z-ORDER is a maintenance operation, never part
#: of the CREATE TABLE DDL.
ZORDER_COLS = ("coil_id", "expert_group_id", "defect_kind")


def optimize_obt(spark, table: str = OBT_TABLE, zorder_cols=ZORDER_COLS) -> None:
    """Run the periodic file-compaction + Z-ORDER maintenance pass on the OBT.

    Wired as a scheduled Databricks maintenance task (see ``jobs.py``): after the
    streaming ``ingest`` accumulates many small files, ``OPTIMIZE ... ZORDER BY``
    co-locates rows sharing the frequent predicate columns so that data-skipping
    prunes whole files at query time. Complements the table's
    ``delta.autoOptimize`` (optimize-write + auto-compact) which only handles
    write-time small files.
    """
    cols = ", ".join(zorder_cols)
    spark.sql(f"OPTIMIZE {table} ZORDER BY ({cols})")

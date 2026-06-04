"""Databricks job (Subject 2) — read defect features from the OBT.

Reads the CNN embedding (960-d) + the 24 regionprops/OpenCV morphology columns
from ``steeldefect.obt_segmentations`` and returns aligned numpy blocks for
clustering.  The 48-d ``mean_max`` segmentation vector used by the clusterer is
DERIVED at read time (the OBT stores the atomic 24 per-region descriptors, not
the aggregate), keeping the table honest about raw-vs-derived.

Mirrors ``flatness_clustering/databricks/01_feature_extraction.py``: stratified
Spark sample before ``toPandas()``, no per-row Python on the driver.
"""

from __future__ import annotations

import os
import sys

import numpy as np

OBT_TABLE = "steeldefect.obt_segmentations"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def read_obt_features(spark, obt_table: str = OBT_TABLE, subsample_n: int = 8000,
                      strat_col: str = "expert_group", random_state: int = 42):
    """Return ``(cnn_960, seg_48, meta_df)`` from the OBT.

    ``seg_48`` is the derived mean+max aggregate; for single-region crops
    max == mean, so it is ``hstack([morpho24, morpho24])`` — documented as a
    read-time view, not stored.
    """
    _bootstrap_path()
    from steeldefect_platform.obt_schema import CNN_DIM, MORPHO_NAMES

    cols = ["segmentation_id", "image_id", "expert_group", "expert_group_id",
            "defect_type", "cnn_vector"] + MORPHO_NAMES
    df = spark.table(obt_table).where(
        "defect_kind IN ('local','group') AND cnn_vector IS NOT NULL"
    ).select(*cols)

    # stratified sample by expert_group before collecting (mirrors flatness 02)
    n_total = df.count()
    if 0 < subsample_n < n_total:
        frac = min(1.0, float(subsample_n) / n_total)
        fractions = {r[strat_col]: frac
                     for r in df.select(strat_col).distinct().collect()}
        df = df.sampleBy(strat_col, fractions=fractions, seed=random_state)

    pdf = df.toPandas()
    cnn = np.stack(pdf["cnn_vector"].apply(lambda v: np.asarray(v, np.float32)))
    assert cnn.shape[1] == CNN_DIM, f"expected {CNN_DIM}-d CNN, got {cnn.shape[1]}"
    morpho = pdf[MORPHO_NAMES].to_numpy(np.float32)
    seg_48 = np.hstack([morpho, morpho])   # mean+max view (max==mean per crop)
    meta = pdf[["segmentation_id", "image_id", "expert_group",
                "expert_group_id", "defect_type"]].copy()
    return cnn, seg_48, meta

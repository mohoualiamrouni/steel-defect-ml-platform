"""Databricks job (Subject 3) — severity scoring (postprocessing of the inspection output).

Reads local defects from the OBT, predicts severity (and the conformal interval)
with the MLflow-registered ``steeldefect_severity_xgb`` bundle, and MERGEs the result
back into ``steeldefect.obt_segmentations``.  First task of the two-task datamining
postprocessing Job requested by the plant ("automatically treat the results from the
inspection systems as a postprocessing step").

Spark injected by Databricks.   run:  main(spark, since="2026-05-01")
"""

from __future__ import annotations

import os
import sys

OBT_TABLE = "steeldefect.obt_segmentations"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def _build_features(pdf, group_encoder):
    """Reconstruct the model's feature matrix: 24 morphology + one-hot group.

    Uses the SAME group encoder persisted with the model (bundled in one MLflow
    artifact) so the column order can never drift from training.
    """
    import numpy as np
    from steeldefect_platform.obt_schema import MORPHO_NAMES

    morpho = pdf[MORPHO_NAMES].to_numpy(np.float32)
    grp = group_encoder.transform(pdf[["expert_group"]])
    grp = grp.toarray() if hasattr(grp, "toarray") else np.asarray(grp)
    return np.hstack([morpho, grp]).astype(np.float32)


def main(spark, since: str = "2026-01-01",
         model_name: str = "steeldefect_severity_xgb",
         use_cqr: bool = True, cqr_alpha: float = 0.1) -> int:
    _bootstrap_path()
    import numpy as np
    from steeldefect_platform import mlflow_registry
    from steeldefect_platform.ingestion import upsert_severity
    from steeldefect_platform.obt_schema import MORPHO_NAMES
    from datamining.features.severity_model import predict_severity

    bundle = mlflow_registry.load_production(model_name)
    model = getattr(bundle, "model", bundle)
    encoder = getattr(bundle, "group_encoder", None)
    cqr = getattr(bundle, "cqr", None)

    cols = ["segmentation_id", "expert_group"] + MORPHO_NAMES
    pdf = (spark.table(OBT_TABLE)
           .where(f"defect_kind='local' AND date_key >= '{since}'")
           .select(*cols).toPandas())
    if pdf.empty:
        print("[severity_score] no local defects to score")
        return 0

    X = _build_features(pdf, encoder) if encoder is not None \
        else pdf[MORPHO_NAMES].to_numpy(np.float32)
    out = pdf[["segmentation_id"]].copy()
    out["severity"] = predict_severity(model, X).astype(np.float32)
    if use_cqr and cqr is not None:
        lo, hi = cqr.predict_interval(X)
        out["severity_lo"] = np.asarray(lo, np.float32)
        out["severity_hi"] = np.asarray(hi, np.float32)

    n = upsert_severity(spark, spark.createDataFrame(out), OBT_TABLE)
    print(f"[severity_score] scored + merged severity for {n} defects")
    return n

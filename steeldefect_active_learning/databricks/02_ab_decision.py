"""Databricks job 2 (Subject 1) — A/B model-promotion decision (the gate).

Compares a challenger model B (Staging) against the incumbent A (Production) on
a frozen, ground-truth-labelled evaluation slice, runs the full statistical
cascade (``compute_all_metrics``), applies the promotion gate
(``ab_testing.promotion.gate_and_register``), writes the audit row to
``steeldefect.ab_decisions``, and — only on ``promote_B`` — transitions B to
Production in the MLflow Model Registry.

No model reaches Production except through this job.  Spark is injected by
Databricks.

    run:  main(spark, eval_dataset_id="2026-05-eval", model_a_uri=..., model_b_uri=...)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

DECISIONS_TABLE = "steeldefect.ab_decisions"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def load_eval_slice(spark, eval_dataset_id: str):
    """Load the frozen A/B evaluation predictions table for this run.

    Expected columns: image_path/label, pred_a, pred_b, confidence_a,
    confidence_b (+ entropy/margin); the ``label`` column is REQUIRED so the
    paired McNemar / per-class guardrail actually fire.
    """
    return (
        spark.table("steeldefect.ab_eval_slices")
        .where(f"eval_dataset_id = '{eval_dataset_id}'")
        .toPandas()
    )


def _write_decision(spark, verdict: dict, all_metrics: dict,
                    eval_dataset_id: str, model_a_uri: str, model_b_uri: str) -> None:
    import datetime as _dt
    st = all_metrics.get("statistical", {})
    mc = all_metrics.get("mcnemar", {})
    srm = all_metrics.get("srm", {})
    acc = (all_metrics.get("bayesian", {}) or {}).get("accuracy", {})
    row = {
        "ab_run_id": eval_dataset_id,
        "model_a_uri": model_a_uri,
        "model_b_uri": model_b_uri,
        "eval_dataset_id": eval_dataset_id,
        "metric": st.get("test_used"),
        "welch_t": float(st.get("t_stat", float("nan"))),
        "welch_p": float(st.get("p_value", float("nan"))),
        "mannwhitney_p": float(st.get("p_value", float("nan")))
            if st.get("test_used") == "mann_whitney_u" else float("nan"),
        "mcnemar_p": float(mc.get("p_value", float("nan"))),
        "bh_reject": bool(any(r.get("reject", False)
                              for r in (all_metrics.get("per_class_mcnemar", {}) or {}).get("rows", []))),
        "cohen_d": float(st.get("cohens_d", float("nan"))),
        "ci_low": float((st.get("confidence_interval_95") or [float("nan"), float("nan")])[0]),
        "ci_high": float((st.get("confidence_interval_95") or [float("nan"), float("nan")])[1]),
        "srm_p": float(srm.get("p_value", float("nan"))),
        "prob_b_better": float(acc.get("p_b_gt_a", float("nan"))),
        "decision": verdict["decision"],
        "promoted_model_uri": verdict.get("promoted_model_uri"),
        "decided_ts": _dt.datetime.utcnow().isoformat(),
    }
    spark.createDataFrame([row]).write.format("delta").mode("append").saveAsTable(DECISIONS_TABLE)


def main(spark, eval_dataset_id: str, model_a_uri: str, model_b_uri: str,
         model_b_version: Optional[int] = None) -> Optional[str]:
    _bootstrap_path()
    from steeldefect_active_learning.ab_testing.metrics import compute_all_metrics
    from steeldefect_active_learning.ab_testing.promotion import gate_and_register

    df = load_eval_slice(spark, eval_dataset_id)
    metrics = compute_all_metrics(
        df, task="classification", alpha=0.05, correction="benjamini-hochberg",
        srm_threshold=1e-3, bootstrap_n=1000, bayesian=True,
        bayesian_decision_threshold=0.95, sequential="msprt",
    )
    verdict = gate_and_register(
        metrics, model_b_uri=model_b_uri, model_name="steeldefect_segmenter",
        model_b_version=model_b_version,
    )
    _write_decision(spark, verdict, metrics, eval_dataset_id, model_a_uri, model_b_uri)
    print(f"[ab_decision] decision={verdict['decision']} "
          f"promoted={verdict.get('promoted_model_uri')}")
    return verdict.get("promoted_model_uri")

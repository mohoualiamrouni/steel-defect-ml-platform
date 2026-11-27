"""Databricks job: A/B promotion decision.

Loads the frozen evaluation slice, runs ``compute_all_metrics`` and the promotion gate
with the thresholds of ``pipeline_config.yaml`` (``ab_testing.promotion``), appends the
audit row to ``steeldefect.ab_decisions``, and only on ``promote_B`` with a
``model_b_version`` and a ``model_a_version`` moves the ``@champion`` alias from A to B,
if it still points at A. The bundle does not schedule this job (its ab_decision task runs
nb_ab.py), and the dispatcher passes no model version, so it cannot promote yet. Spark is
injected by Databricks.

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
    srm = all_metrics.get("srm", {})
    # The confidence test ('metric' names it) fills its own columns; the other test's are NaN.
    welch = st.get("test_used") == "welch_t"
    mann_whitney = st.get("test_used") == "mann_whitney_u"
    # The gate's own one-sided exact McNemar p and paired posterior (None on invalid input).
    estimates = verdict.get("estimates") or {}
    row = {
        "ab_run_id": eval_dataset_id,
        "model_a_uri": model_a_uri,
        "model_b_uri": model_b_uri,
        "eval_dataset_id": eval_dataset_id,
        "metric": st.get("test_used"),
        "welch_t": float(st.get("t_stat", float("nan"))) if welch else float("nan"),
        "welch_p": float(st.get("p_value", float("nan"))) if welch else float("nan"),
        "mannwhitney_p": float(st.get("p_value", float("nan"))) if mann_whitney else float("nan"),
        "mcnemar_p": float(estimates.get("mcnemar_p_one_sided", float("nan"))),
        "bh_reject": bool(any(r.get("reject", False)
                              for r in (all_metrics.get("per_class_mcnemar", {}) or {}).get("rows", []))),
        "cohen_d": float(st.get("cohens_d", float("nan"))),
        "ci_low": float((st.get("confidence_interval_95") or [float("nan"), float("nan")])[0]),
        "ci_high": float((st.get("confidence_interval_95") or [float("nan"), float("nan")])[1]),
        "srm_p": float(srm.get("p_value", float("nan"))),
        "prob_b_better": float(estimates.get("p_b_better_posterior", float("nan"))),
        "decision": verdict["decision"],
        "promoted_model_uri": verdict.get("promoted_model_uri"),
        "decided_ts": _dt.datetime.utcnow().isoformat(),
    }
    spark.createDataFrame([row]).write.format("delta").mode("append").saveAsTable(DECISIONS_TABLE)


def main(spark, eval_dataset_id: str, model_a_uri: str, model_b_uri: str,
         model_b_version: Optional[int] = None,
         config_path: Optional[str] = None,
         model_a_version: Optional[int] = None) -> Optional[str]:
    _bootstrap_path()
    from steeldefect_active_learning.ab_testing.metrics import compute_all_metrics
    from steeldefect_active_learning.ab_testing.promotion import gate_and_register
    from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config

    cfg = load_config(config_path or SHIPPED_CONFIG_PATH).ab_testing
    df = load_eval_slice(spark, eval_dataset_id)
    metrics = compute_all_metrics(
        df, task="classification", alpha=cfg.alpha,
        correction=cfg.multiple_testing_correction, srm_threshold=cfg.srm_threshold,
        expected_ratio=cfg.expected_arm_ratio, bootstrap_n=cfg.bootstrap_n, bayesian=True,
        bayesian_decision_threshold=cfg.bayesian_decision_threshold,
        sequential=cfg.sequential,
    )
    verdict = gate_and_register(
        metrics, model_b_uri=model_b_uri, model_name="steeldefect_segmenter",
        model_b_version=model_b_version, model_a_version=model_a_version,
        **cfg.gate_kwargs(),
    )
    _write_decision(spark, verdict, metrics, eval_dataset_id, model_a_uri, model_b_uri)
    for reason in verdict["reasons"]:
        print(f"[ab_decision] {'PASS' if reason['passed'] else 'FAIL'} "
              f"{reason['criterion']}: {reason['detail']}")
    print(f"[ab_decision] decision={verdict['decision']} "
          f"promoted={verdict.get('promoted_model_uri')}")
    return verdict.get("promoted_model_uri")

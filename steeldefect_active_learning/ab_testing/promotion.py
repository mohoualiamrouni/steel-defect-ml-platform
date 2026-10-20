"""A/B promotion gate — the single ship/no-ship decision for inspection models.

No model reaches the MLflow ``Production`` stage except through this gate.
:func:`decide_promotion` consumes the full cascade produced by
:func:`steeldefect_active_learning.ab_testing.metrics.compute_all_metrics` and returns
a structured verdict; :func:`gate_and_register` applies it and, only on
``promote_B``, calls :func:`steeldefect_platform.mlflow_registry.promote`.

Promotion rule (challenger B promoted over incumbent A iff ALL hold):

* SRM not detected                      — the evaluation split is valid;
* statistical test significant (p<α)     — a real difference exists;
* |Cohen's d| ≥ ``min_cohens_d``         — the difference is not trivially small;
* analysis adequately powered (≥0.80);
* Bayesian P(B more accurate than A) ≥ ``bayesian_threshold``;
* paired McNemar significant AND B wins more discordant pairs than A (c ≥ b);
* **no class significantly regresses** under B (BH-corrected per-class McNemar).

The last clause is the industrially decisive guardrail: a challenger that wins
on average but tanks a safety-critical rare class (e.g. ``coil_break``) must not
ship.  The realistic outcome on a *close* challenger is ``keep_A`` globally with
a non-empty ``per_class_promote`` list — i.e. promote B only on the classes it
was retrained for, via a routing rule, not a wholesale swap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _crit(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "detail": detail}


def decide_promotion(
    all_metrics: Dict[str, Any],
    *,
    alpha: float = 0.05,
    min_cohens_d: float = 0.2,
    bayesian_threshold: float = 0.95,
    require_no_class_regression: bool = True,
) -> Dict[str, Any]:
    """Return a structured ship/no-ship verdict from a cascade dict.

    Robust to missing sub-blocks: a criterion whose evidence is absent is
    treated as NOT satisfied (fail-closed), since promotion must be earned.
    """
    st = all_metrics.get("statistical", {}) or {}
    srm = all_metrics.get("srm", {}) or {}
    power = all_metrics.get("power", {}) or {}
    mc = all_metrics.get("mcnemar", {}) or {}
    bayes = all_metrics.get("bayesian", {}) or {}
    pcm = all_metrics.get("per_class_mcnemar", {}) or {}

    reasons: List[Dict[str, Any]] = []

    # 1. valid randomisation / split
    srm_ok = not bool(srm.get("srm_detected", True))
    reasons.append(_crit("srm_valid", srm_ok,
                         f"srm_detected={srm.get('srm_detected')}, p={srm.get('p_value')}"))

    # 2. significant difference
    sig = bool(st.get("significant", False)) and float(st.get("p_value", 1.0)) < alpha
    reasons.append(_crit("significant", sig,
                         f"test={st.get('test_used')}, p={st.get('p_value')}"))

    # 3. effect size not trivial
    d = abs(float(st.get("cohens_d", 0.0)))
    d_ok = d >= min_cohens_d
    reasons.append(_crit("effect_size", d_ok, f"|cohens_d|={d:.3f} ≥ {min_cohens_d}"))

    # 4. adequately powered
    pw_ok = bool(power.get("adequately_powered", False))
    reasons.append(_crit("powered", pw_ok,
                         f"achieved_power={power.get('achieved_power')}"))

    # 5. Bayesian P(B more accurate than A)
    acc_block = bayes.get("accuracy") or bayes.get("agreement") or {}
    p_b_gt_a = float(acc_block.get("p_b_gt_a", 0.0))
    bayes_ok = p_b_gt_a >= bayesian_threshold
    reasons.append(_crit("bayesian_prob", bayes_ok,
                         f"P(B>A)={p_b_gt_a:.3f} ≥ {bayesian_threshold}"))

    # 6. paired accuracy: McNemar significant AND B wins more discordant pairs
    b = int(mc.get("b", 0))  # A correct, B wrong
    c = int(mc.get("c", 0))  # A wrong, B correct
    mc_ok = (float(mc.get("p_value", 1.0)) < alpha) and (c >= b)
    reasons.append(_crit("mcnemar_paired", mc_ok,
                         f"p={mc.get('p_value')}, b={b}, c={c} (B better iff c≥b)"))

    # 7. no class significantly regresses; collect per-class wins/regressions.
    # Rows come from per_class_mcnemar: b = A correct and B wrong, c = A wrong
    # and B correct, so acc_b - acc_a = (c - b) / n on that class.
    per_class_promote: List[str] = []
    per_class_regress: List[str] = []
    for row in pcm.get("rows", []) or []:
        reject = bool(row.get("reject", False))
        if row.get("b") is not None and row.get("c") is not None:
            delta = (int(row["c"]) - int(row["b"])) / max(int(row.get("n", 0)), 1)
        else:
            delta = float(row.get("delta", 0.0))  # acc_b - acc_a, if supplied directly
        name = str(row.get("class_name", row.get("class", row.get("class_id", "?"))))
        if reject and delta > 0:
            per_class_promote.append(name)
        elif reject and delta < 0:
            per_class_regress.append(name)
    no_regression = (len(per_class_regress) == 0)
    if require_no_class_regression:
        reasons.append(_crit("no_class_regression", no_regression,
                             f"regressing classes={per_class_regress or 'none'}"))

    gate_passed = all(r["passed"] for r in reasons)
    decision = "promote_B" if gate_passed else "keep_A"

    return {
        "decision": decision,
        "gate_passed": gate_passed,
        "reasons": reasons,
        "per_class_promote": per_class_promote,
        "per_class_regress": per_class_regress,
        "thresholds": {
            "alpha": alpha,
            "min_cohens_d": min_cohens_d,
            "bayesian_threshold": bayesian_threshold,
        },
    }


def gate_and_register(
    all_metrics: Dict[str, Any],
    model_b_uri: str,
    model_name: str = "steeldefect_segmenter",
    model_b_version: Optional[int] = None,
    *,
    alpha: float = 0.05,
    min_cohens_d: float = 0.2,
    bayesian_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Run the gate; if ``promote_B``, promote model B in the MLflow registry.

    Returns the verdict dict augmented with ``promoted_model_uri`` (the URI when
    promoted, else ``None``).  Promotion is a no-op if MLflow is unavailable or
    ``model_b_version`` is not supplied — the verdict is still returned so the
    caller can persist the ``steeldefect.ab_decisions`` audit row.
    """
    verdict = decide_promotion(
        all_metrics,
        alpha=alpha,
        min_cohens_d=min_cohens_d,
        bayesian_threshold=bayesian_threshold,
    )
    promoted_uri = None
    if verdict["decision"] == "promote_B" and model_b_version is not None:
        try:
            from steeldefect_platform.mlflow_registry import promote
            promote(model_name, model_b_version, to_stage="Production")
            promoted_uri = model_b_uri
        except Exception:
            promoted_uri = None  # registry unavailable; verdict still returned
    verdict["promoted_model_uri"] = promoted_uri
    verdict["model_name"] = model_name
    verdict["model_b_uri"] = model_b_uri
    return verdict

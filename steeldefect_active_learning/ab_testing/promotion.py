"""A/B promotion gate: does challenger B replace incumbent A?

The gate reads a paired offline evaluation: both models scored the same labelled images.
:func:`decide_promotion` reads only the ``paired_counts`` block of
:func:`steeldefect_active_learning.ab_testing.metrics.compute_all_metrics` (how many images
both models get right, only A, only B, or neither; overall, per true class and per
predicted class) and returns ``promote_B`` only if every criterion holds:

1. ``valid_input``: the counts satisfy the contract of ``paired.paired_counts``, and the
   gate is configured (the class taxonomy, and the class prior at which accuracy is
   compared);
2. ``sample_size``: the slice meets the prospective plan, fixed by configuration before
   any outcome is seen (effective size >= planned size);
3. ``superiority``: B wins significantly more discordant images than A on the slice
   (one-sided exact McNemar), and the lower bound of acc_B - acc_A at the deployment
   class prior exceeds ``min_gain``;
4. ``no_class_regression``: on every class of the taxonomy, the lower bound of
   recall_B - recall_A exceeds minus the class margin;
5. ``no_false_alarm_increase``: no class shows, at level alpha, an increase of B's false
   alarms into it larger than its false-alarm margin.

Every per-class, slice and false-alarm bound is Tango's (1998) score bound with a one-count
continuity adjustment (``paired.score_lower_bound``); the bound at the deployment prior
combines the per-class bounds by MOVER. Requiring every clause is an intersection-union test
(Berger 1982): no multiplicity correction is needed, and the probability of promoting a B
for which one of these claims is false is at most the largest clause size. At
alpha = 0.05 that size stayed at or below 0.05 in every configuration checked with up to
40 % of the images discordant (exact enumeration for the per-class and slice bounds,
simulation for the prior-weighted bound), and reached 0.0503 only when every image is
discordant. alpha must lie in [0.025, 0.5): at small alpha, the slice bound tested above
a positive ``min_gain`` is too skewed for the adjustment. When B never loses and gains
exactly min_gain, its exact size reaches 1.12 alpha at alpha = 0.01 and 1.001 alpha at
0.02, over every n from 20 to 6,000 (0.993 alpha at 0.0225, and 0.998 alpha with n up to
30,000). From 0.025 to 0.45, the exact sizes checked stayed at or below alpha, again
except when every image of a class is discordant (up to 1.009 alpha). Missing or invalid
evidence fails its criterion. Model confidence plays no part.

The gate assumes what it cannot check: the images are independent (one per coil, or a
cluster column with one image per cluster), the labels of the images on which A and B
disagree were checked without seeing the model outputs, the slice was fixed before the
models were scored, and ``class_prior`` describes the deployment stream. False alarms are
checked with a harm veto: it fires on evidence that B adds false alarms into a class
beyond its margin, and does not certify that B adds none.

The gate decides. :func:`plan_promotion` and :func:`apply_promotion` carry a ``promote_B``
to the model registry by a compare-and-set on ``@champion`` (a read, then a write: MLflow
has no atomic one, see ``mlflow_registry.promote_if_champion``): B replaces only the version
it was compared with, and the outcome (applied, refused, failed, unknown) is reported,
never swallowed. :func:`gate_and_register` runs the three in turn.

References:
    Berger (1982) "Multiparameter hypothesis testing and acceptance sampling",
        Technometrics 24:295-300 (intersection-union tests).
    Tango (1998) "Equivalence test and confidence interval for the difference in
        proportions for the paired-sample design", Statistics in Medicine 17:891-908.
"""

from __future__ import annotations

import logging
import numbers
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Union

from ..config import check_gate_thresholds
from .paired import (
    CELLS,
    CONTINUITY_ADJUSTMENT,
    mcnemar_one_sided_p,
    mover_bounds,
    n_effective,
    noninferiority_floor,
    paired_counts_errors,
    planning_n,
    posterior_b_better,
    score_lower_bound,
    score_upper_bound,
)

VERDICT_SCHEMA = "promotion_verdict/2"

logger = logging.getLogger(__name__)


def _crit(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "detail": detail}


def _label(row: Mapping[str, Any]) -> str:
    """A class shown by name when the slice has one."""
    return str(row["name"] if row.get("name") is not None else row["class"])


def _listed(details: List[str], limit: int = 5) -> str:
    return "; ".join(details[:limit]) + ("; ..." if len(details) > limit else "")


def decide_promotion(
    all_metrics: Mapping[str, Any],
    *,
    alpha: float = 0.05,
    min_gain: float = 0.0,
    noninferiority_margin: float = 0.05,
    class_margins: Optional[Mapping[Hashable, float]] = None,
    expected_classes: Optional[Sequence[Hashable]] = None,
    class_prior: Union[None, str, Mapping[Hashable, float]] = None,
    false_alarm_margin: Optional[float] = 0.005,
    class_false_alarm_margins: Optional[Mapping[Hashable, float]] = None,
    min_detectable_gain: float = 0.02,
    planning_discordance: float = 0.10,
    power: float = 0.80,
    require_no_class_regression: bool = True,
) -> Dict[str, Any]:
    """Return the promotion verdict for a cascade dict.

    Args:
        all_metrics:          output of ``compute_all_metrics``; only its
                              ``paired_counts`` block is read.
        alpha:                one-sided level of every clause, in [0.025, 0.5).
        min_gain:             superiority: the lower bound of acc_B - acc_A must exceed it.
        noninferiority_margin, class_margins: per-class margin m_k; a class passes when the
                              lower bound of recall_B - recall_A exceeds -m_k.
        expected_classes:     the class taxonomy; no label or prediction may lie outside it.
                              A class absent from the slice fails no_class_regression, and
                              leaves the prior-weighted estimand not estimable when its prior
                              share is positive. None: not configured.
        class_prior:          deployment class shares (renormalised) at which accuracy is
                              compared, or "slice" for the slice's own class mix. None: not
                              configured.
        false_alarm_margin, class_false_alarm_margins: veto a class when B adds false alarms
                              into it, at level alpha, beyond this rate; None disables it.
        min_detectable_gain, planning_discordance, power: the prospective plan. The slice
                              must have at least the effective size at which the superiority
                              clause alone reaches ``power`` against a B
                              ``min_detectable_gain`` more accurate on every class, with
                              ``planning_discordance`` discordant images. Each class needs its
                              own planned size (``per_class[].n_planned``, split so that all
                              classes together reach ``power``); those are warned about, not
                              enforced, so the gate's power at the plan is lower: close to 0
                              when the rare classes are as rare in the slice as in the
                              deployment prior (a few images each).
        require_no_class_regression: enforce the per-class clause.

    Returns:
        A ``promotion_verdict/2`` dict: ``decision`` ("promote_B" | "keep_A"),
        ``gate_passed``, ``conclusion``, ``reasons`` (one entry per enabled criterion, in
        order), ``warnings``, ``n_images``, ``id_col``, ``counts``, ``estimates``,
        ``plan``, ``per_class``, ``per_predicted_class``, ``failing_classes``,
        ``vetoed_classes``, ``thresholds`` and ``disabled_criteria``. Every float is finite
        or None, so the verdict is strict JSON.

    Raises ValueError for out-of-range thresholds; bad data never raises, it fails
    ``valid_input``. ``conclusion`` is descriptive: "evidence_against_B" means that at
    least one one-sided bound excludes the promotion region, "inconclusive" that the
    evidence is too thin either way; it is not a multiplicity-controlled test.
    """
    check_gate_thresholds(
        alpha=alpha, min_gain=min_gain, noninferiority_margin=noninferiority_margin,
        class_margins=class_margins, expected_classes=expected_classes,
        class_prior=class_prior, false_alarm_margin=false_alarm_margin,
        class_false_alarm_margins=class_false_alarm_margins,
        min_detectable_gain=min_detectable_gain, planning_discordance=planning_discordance,
        power=power, require_no_class_regression=require_no_class_regression,
    )
    margins = dict(class_margins or {})
    fa_margins = dict(class_false_alarm_margins or {})
    disabled = []
    if not require_no_class_regression:
        disabled.append("no_class_regression")
    if false_alarm_margin is None:
        disabled.append("no_false_alarm_increase")
    criteria = ["valid_input", "sample_size", "superiority"]
    criteria += [c for c in ("no_class_regression", "no_false_alarm_increase") if c not in disabled]
    verdict: Dict[str, Any] = {
        "schema": VERDICT_SCHEMA, "decision": "keep_A", "gate_passed": False,
        "conclusion": "invalid_input", "reasons": [], "warnings": [],
        "n_images": None, "id_col": None, "counts": None, "estimates": None, "plan": None,
        "per_class": [], "per_predicted_class": [], "failing_classes": [], "vetoed_classes": [],
        "thresholds": {
            "alpha": alpha, "min_gain": min_gain, "noninferiority_margin": noninferiority_margin,
            "class_margins": [{"class": k, "margin": float(m)} for k, m in margins.items()],
            "expected_classes": None if expected_classes is None else list(expected_classes),
            "class_prior": (class_prior if class_prior is None or class_prior == "slice"
                            else [{"class": k, "weight": float(w)} for k, w in class_prior.items()]),
            "false_alarm_margin": false_alarm_margin,
            "class_false_alarm_margins": [{"class": k, "margin": float(m)}
                                          for k, m in fa_margins.items()],
            "min_detectable_gain": min_detectable_gain,
            "planning_discordance": planning_discordance, "power": power,
            "require_no_class_regression": require_no_class_regression,
            "continuity_adjustment": CONTINUITY_ADJUSTMENT,
        },
        "disabled_criteria": disabled,
    }
    reasons = verdict["reasons"]

    # ---- 1. valid input, configured gate
    pc = all_metrics.get("paired_counts") if isinstance(all_metrics, Mapping) else None
    errors = paired_counts_errors(pc)
    if expected_classes is None:
        errors.append("expected_classes is not configured: an absent class could not be detected")
    if class_prior is None:
        errors.append("class_prior is not configured: set the deployment class prior, or "
                      "'slice' to certify the slice's own class mix")
    if not errors:
        universe = set(expected_classes)
        outside = [r["class"] for r in pc["per_class"] if r["class"] not in universe]
        for r in pc.get("per_predicted_class") or []:
            if r["class"] not in universe and r["class"] not in outside:
                outside.append(r["class"])
        if outside:
            more = f" and {len(outside) - 10} more" if len(outside) > 10 else ""
            errors.append(f"classes outside expected_classes in the slice: {outside[:10]}{more}")
        if false_alarm_margin is not None and not pc.get("predictions_available"):
            errors.append("the false-alarm clause needs pred_a and pred_b (correct_* columns "
                          "alone cannot feed it)")
    if errors:
        reasons.append(_crit("valid_input", False, "invalid input: " + "; ".join(errors) + ": fail"))
        reasons.extend(_crit(c, False, "not evaluated: invalid input") for c in criteria[1:])
        return verdict

    n, n11 = pc["n"], pc["both_correct"]
    n10, n01 = pc["only_a_correct"], pc["only_b_correct"]
    reasons.append(_crit("valid_input", True, f"{n} unique images ({pc.get('id_col')}), "
                                              f"{len(pc['per_class'])} classes: pass"))
    rows = {r["class"]: r for r in pc["per_class"]}
    order = list(expected_classes)
    n_classes = len(order)
    if class_prior == "slice":
        weights = {k: rows[k]["n"] / n if k in rows else 0.0 for k in order}
    else:
        total = float(sum(class_prior.values()))
        weights = {k: float(class_prior[k]) / total for k in order}
    per_class_power = power ** (1.0 / n_classes)

    # ---- per-class table: status of every expected class
    per_class = []
    for k in order:
        m_k = float(margins.get(k, noninferiority_margin))
        row = {"class": k, "name": None, "n": 0, **{c: 0 for c in CELLS},
               "delta": None, "lower": None, "upper": None, "margin": m_k,
               "n_floor": noninferiority_floor(m_k, alpha),
               "n_planned": planning_n(planning_discordance, min_detectable_gain, -m_k,
                                       alpha, per_class_power),
               "weight": weights[k], "status": "absent"}
        r = rows.get(k)
        if r is not None:
            nk, ak, bk = r["n"], r["only_a_correct"], r["only_b_correct"]
            lower = score_lower_bound(ak, bk, nk, alpha)
            upper = score_upper_bound(ak, bk, nk, alpha)
            if lower > -m_k:
                status = "pass"
            elif upper < -m_k:
                status = "regressed"
            elif nk < row["n_floor"]:
                status = "insufficient_n"
            else:
                status = "inconclusive"
            row.update({"name": r.get("name"), "n": nk, **{c: r[c] for c in CELLS},
                        "delta": (bk - ak) / nk, "lower": lower, "upper": upper,
                        "status": status})
        per_class.append(row)
    verdict["per_class"] = per_class

    # ---- estimates: the slice, and the estimand at the class prior
    p_one_sided = mcnemar_one_sided_p(n10, n01)
    slice_lower = score_lower_bound(n10, n01, n, alpha)
    slice_upper = score_upper_bound(n10, n01, n, alpha)
    weighted = [r for r in per_class if weights[r["class"]] > 0]
    absent_weighted = [r["class"] for r in weighted if r["status"] == "absent"]
    if class_prior == "slice":
        estimand = "slice"
        est, est_lower, est_upper = (n01 - n10) / n, slice_lower, slice_upper
    else:
        estimand = "deployment_prior"
        if absent_weighted:
            est = est_lower = est_upper = None
        else:
            est, est_lower, est_upper = mover_bounds(
                [r["delta"] for r in weighted], [r["lower"] for r in weighted],
                [r["upper"] for r in weighted], [weights[r["class"]] for r in weighted])
    if class_prior == "slice":
        n_eff = float(n)    # the identity, which floating point can miss by one ulp
    else:
        n_eff = n_effective([weights[r["class"]] for r in weighted], [r["n"] for r in weighted])
    n_planned = planning_n(planning_discordance, min_detectable_gain, min_gain, alpha, power)
    discordance = (n10 + n01) / n
    verdict.update(n_images=n, id_col=pc.get("id_col"), counts={**{c: pc[c] for c in CELLS}, "n": n})
    verdict["estimates"] = {
        "acc_a": (n11 + n10) / n, "acc_b": (n11 + n01) / n, "delta": (n01 - n10) / n,
        "delta_lower": slice_lower, "delta_upper": slice_upper,
        "mcnemar_p_one_sided": p_one_sided, "n_discordant": n10 + n01,
        "discordance_rate": discordance, "estimand": estimand, "estimand_delta": est,
        "estimand_lower": est_lower, "estimand_upper": est_upper,
        "p_b_better_posterior": posterior_b_better(n10, n01),
    }
    verdict["plan"] = {
        "n_planned": n_planned, "n_effective": n_eff,
        "planning_discordance": planning_discordance,
        "min_detectable_gain": min_detectable_gain, "power": power,
        "per_class_power": per_class_power,
    }
    if discordance > planning_discordance:
        verdict["warnings"].append(
            f"observed discordance {discordance:.4f} exceeds planning_discordance "
            f"{planning_discordance}: the slice is less powerful than planned")
    under = [r["class"] for r in per_class if r["n"] < r["n_planned"]]
    if under:
        verdict["warnings"].append(f"classes below their planned size: {under}")
    if class_prior == "slice":
        verdict["warnings"].append("estimand: the slice's own class mix (class_prior='slice')")

    # ---- 2. sample size: configuration and class sizes only, never the outcomes
    size_ok = n_eff >= n_planned
    if absent_weighted:
        detail = (f"classes {absent_weighted} with a positive weight are absent: "
                  f"n_effective = 0 < n_planned = {n_planned}: fail")
    else:
        detail = (f"n_effective = {n_eff:.1f} {'>=' if size_ok else '<'} n_planned = {n_planned} "
                  f"(gain {min_detectable_gain}, discordance {planning_discordance}, "
                  f"power {power}): {'pass' if size_ok else 'fail'}")
    reasons.append(_crit("sample_size", size_ok, detail))

    # ---- 3. superiority
    p_ok = p_one_sided < alpha
    if est_lower is None:
        bound_ok = False
        bound_detail = (f"classes {absent_weighted} absent: the prior-weighted difference is "
                        "not estimable")
    else:
        bound_ok = est_lower > min_gain
        bound_detail = (f"{estimand} lower bound {est_lower:+.4f} "
                        f"{'>' if bound_ok else '<='} {min_gain}")
    reasons.append(_crit(
        "superiority", p_ok and bound_ok,
        f"one-sided exact McNemar p = {p_one_sided:.4g} {'<' if p_ok else '>='} {alpha}; "
        f"{bound_detail}: {'pass' if p_ok and bound_ok else 'fail'}"))

    # ---- 4. per-class non-inferiority
    failing = [r for r in per_class if r["status"] != "pass"]
    verdict["failing_classes"] = [r["class"] for r in failing]
    if require_no_class_regression:
        if failing:
            details = []
            for r in failing:
                if r["status"] == "absent":
                    details.append(f"{_label(r)}: absent")
                else:
                    reason = r["status"]
                    if reason == "insufficient_n":
                        reason += f": n = {r['n']} < zero-discordance floor {r['n_floor']}"
                    details.append(f"{_label(r)}: lower bound {r['lower']:+.3f} <= "
                                   f"{-r['margin']:+.3f} ({reason})")
            reasons.append(_crit("no_class_regression", False,
                                 f"{len(failing)} of {n_classes} classes fail: "
                                 f"{_listed(details)}: fail"))
        else:
            reasons.append(_crit("no_class_regression", True,
                                 f"all {n_classes} classes: lower bound > -margin: pass"))

    # ---- 5. false alarms (harm veto)
    vetoed = []
    if false_alarm_margin is not None:
        alarm_rows = {r["class"]: r for r in pc["per_predicted_class"]}
        for k in order:
            mu = float(fa_margins.get(k, false_alarm_margin))
            r = alarm_rows.get(k)
            # No row: the class is never labelled and never predicted (paired_counts_errors
            # rejects any other gap). n_other = 0: every image is of class k.
            if r is None or r["n_other"] == 0:
                verdict["per_predicted_class"].append({
                    "class": k, "name": None, "n_other": 0 if r is None else r["n_other"],
                    "only_a_alarm": 0, "only_b_alarm": 0, "fa_delta": 0.0, "fa_lower": None,
                    "fa_margin": mu, "status": "pass"})
                continue
            n_other, fa_a, fa_b = r["n_other"], r["only_a_alarm"], r["only_b_alarm"]
            lower = score_lower_bound(fa_a, fa_b, n_other, alpha)
            status = "veto" if lower > mu else "pass"
            if status == "veto":
                vetoed.append(k)
            verdict["per_predicted_class"].append({
                "class": k, "name": r.get("name"), "n_other": n_other, "only_a_alarm": fa_a,
                "only_b_alarm": fa_b, "fa_delta": (fa_b - fa_a) / n_other, "fa_lower": lower,
                "fa_margin": mu, "status": status})
        verdict["vetoed_classes"] = vetoed
        if vetoed:
            details = [f"{_label(r)}: lower bound {r['fa_lower']:+.4f} > {r['fa_margin']}"
                       for r in verdict["per_predicted_class"] if r["status"] == "veto"]
            reasons.append(_crit("no_false_alarm_increase", False, _listed(details) + ": fail"))
        else:
            reasons.append(_crit("no_false_alarm_increase", True,
                                 "no class: lower bound of the false-alarm increase > margin: pass"))

    verdict["gate_passed"] = all(r["passed"] for r in reasons)
    verdict["decision"] = "promote_B" if verdict["gate_passed"] else "keep_A"
    if verdict["gate_passed"]:
        verdict["conclusion"] = "promote_B"
    elif (any(r["status"] == "regressed" for r in per_class) or vetoed
          or (est_upper is not None and est_upper <= min_gain)):
        verdict["conclusion"] = "evidence_against_B"
    else:
        verdict["conclusion"] = "inconclusive"
    return verdict


def _version(value: Any) -> Optional[int]:
    """A registry version: a positive integer (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        return None
    return int(value)


def version_uri(model_name: str, version: int) -> str:
    """The registry URI of one model version."""
    return f"models:/{model_name}/{version}"


def plan_promotion(
    verdict: Mapping[str, Any],
    model_name: str,
    model_b_uri: str,
    model_b_version: Optional[int],
    model_a_version: Optional[int],
    promote: bool = True,
) -> Dict[str, Any]:
    """What applying a verdict to the registry takes, checked before any registry call.

    ``status`` is one of:

    - ``not_applicable``: the gate kept A;
    - ``not_requested``: promote_B, but ``promote`` is False or no ``model_b_version``
      was given: the decision only;
    - ``refused``: promote_B, but the promotion cannot be tied to what the gate compared:
      ``model_b_uri`` does not name ``model_b_version`` of ``model_name`` (the URI recorded
      must be the version moved), or ``model_a_version`` is missing or invalid (without
      it, the champion B was compared with cannot be checked);
    - ``pending``: :func:`apply_promotion` may call the registry.

    Returns ``{"status", "detail", "champion_before", "champion_after",
    "promoted_model_uri"}``, the last three None.
    """
    def outcome(status: str, detail: str) -> Dict[str, Any]:
        return {"status": status, "detail": detail, "champion_before": None,
                "champion_after": None, "promoted_model_uri": None}

    if verdict.get("decision") != "promote_B":
        return outcome("not_applicable", f"decision {verdict.get('decision')}: nothing to apply")
    if not promote:
        return outcome("not_requested", "promote_B with promote=False: decision only")
    if model_b_version is None:
        return outcome("not_requested", "promote_B without model_b_version: decision only")
    b, a = _version(model_b_version), _version(model_a_version)
    if b is None:
        return outcome("refused", f"model_b_version {model_b_version!r} is not a positive "
                                  f"integer")
    if model_b_uri != version_uri(model_name, b):
        return outcome("refused", f"model_b_uri {model_b_uri!r} does not name "
                                  f"{version_uri(model_name, b)!r}, the version to promote")
    if a is None:
        return outcome("refused", f"model_a_version {model_a_version!r} is not a positive "
                                  f"integer: the champion B was compared with is unknown")
    return outcome("pending", f"promote {model_name} v{b} over v{a}")


def apply_promotion(
    plan: Mapping[str, Any],
    model_name: str,
    model_b_version: Optional[int],
    model_a_version: Optional[int],
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Carry out a ``pending`` plan with ``mlflow_registry.promote_if_champion``.

    Any other plan is returned as it is. A pending plan becomes ``applied`` (``@champion``
    points at B) or ``refused`` (``@champion`` no longer points at A: B was not compared
    with the model it would replace). A registry error is logged with its traceback, and
    the alias is read again, since the error may have come after the write: ``applied``
    if it reads B, ``failed`` if it reads anything else (with what it reads), ``unknown``
    if it cannot be read. Never raises: the caller records the outcome and decides how to
    end. ``client`` is an ``MlflowClient`` (default: the configured registry).
    """
    if plan.get("status") != "pending":
        return dict(plan)
    from steeldefect_platform.mlflow_registry import alias_version, promote_if_champion
    out = {"status": "failed", "detail": "", "champion_before": None, "champion_after": None,
           "promoted_model_uri": None}
    try:
        cas = promote_if_champion(model_name, model_b_version,
                                  expected_champion=model_a_version, client=client)
    except Exception as exc:  # noqa: BLE001 - recorded and logged, not swallowed
        logger.exception("promotion of %s v%s failed", model_name, model_b_version)
        error = f"registry error: {type(exc).__name__}: {exc}"
        try:
            now = alias_version(model_name, client=client)
        except Exception as again:  # noqa: BLE001 - the outcome is unknown, and says so
            out.update(status="unknown",
                       detail=f"{error}; @champion could not be read back "
                              f"({type(again).__name__}): check it by hand")
            return out
        out["champion_after"] = now
        if now == model_b_version:
            out.update(status="applied", detail=f"{error}; @champion reads v{now}",
                       promoted_model_uri=version_uri(model_name, now))
        else:
            out["detail"] = (f"{error}; @champion reads "
                             f"{'unset' if now is None else f'v{now}'}")
        return out
    out.update(champion_before=cas["champion_before"], champion_after=cas["champion_after"],
               detail=cas["detail"])
    if cas["applied"]:
        out.update(status="applied",
                   promoted_model_uri=version_uri(model_name, cas["champion_after"]))
    else:
        out["status"] = "refused"
    return out


def gate_and_register(
    all_metrics: Mapping[str, Any],
    model_b_uri: str,
    model_name: str = "steeldefect_segmenter",
    model_b_version: Optional[int] = None,
    model_a_version: Optional[int] = None,
    **thresholds: Any,
) -> Dict[str, Any]:
    """Run the gate, then carry a ``promote_B`` to the registry by a compare-and-set.

    Calls ``decide_promotion(all_metrics, **thresholds)``, so a misspelt or retired
    threshold raises TypeError, then :func:`plan_promotion` and :func:`apply_promotion`:
    on promote_B with a ``model_b_version``, ``@champion`` moves to B only if
    ``model_b_uri`` names that version and ``@champion`` still points at
    ``model_a_version``, the model B was compared with.

    Returns the verdict augmented with ``promotion`` (status, detail, champion before and
    after), ``promoted_model_uri`` (the version moved, else None), ``model_name`` and
    ``model_b_uri``. A registry error is logged and reported with what ``@champion`` reads
    afterwards (``applied``, ``failed`` or ``unknown``), never as no promotion.
    """
    verdict = decide_promotion(all_metrics, **thresholds)
    plan = plan_promotion(verdict, model_name, model_b_uri, model_b_version, model_a_version)
    verdict["promotion"] = apply_promotion(plan, model_name, model_b_version, model_a_version)
    verdict["promoted_model_uri"] = verdict["promotion"]["promoted_model_uri"]
    verdict["model_name"] = model_name
    verdict["model_b_uri"] = model_b_uri
    return verdict

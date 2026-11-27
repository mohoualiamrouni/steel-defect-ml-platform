"""Databricks job: A/B promotion decision on a frozen, paired evaluation slice.

The slice, ``steeldefect.ab_eval_slices`` (one row per image; its schema is in
``steeldefect_platform.obt_schema.COMPANION_DDL``), holds for one ``eval_dataset_id`` the
true label of each image, its coil, the predictions of champion A and challenger B, and the
registry versions that produced them. The job:

1. loads it with a parameterised query;
2. refuses to decide twice on one challenger for one slice (the same id, the same images, or
   any image or coil in common), except to finish a promotion that a crash or a registry
   error left unsettled, when a run asks for that same promotion again;
3. with ``promote``, checks that ``@champion`` still points at A before it records anything;
4. checks the slice (its columns, the versions compared, one sampling frame, string ids, the
   paired design) and that neither model was trained on its images or coils
   (``steeldefect.model_training_images``). A slice that fails is invalid input: the gate
   keeps A, the row keeps the reasons and no diagnostic, and it does not count as a decision;
5. runs ``compute_all_metrics`` and the promotion gate with the thresholds of
   ``pipeline_config.yaml`` (``ab_testing.promotion``);
6. records the slice's images, then the decision, in ``steeldefect.ab_decision_images`` and
   ``steeldefect.ab_decisions``, before any registry write;
7. on ``promote_B``, moves ``@champion`` from A to B by a compare-and-set
   (``promotion.plan_promotion`` and ``apply_promotion``: a read, then a write, which MLflow
   cannot make atomic), and records the outcome as a second row.

Spark is injected by Databricks.

    run:  main(spark, eval_dataset_id="2026-05-eval", model_a_version=6, model_b_version=7,
               model_name="main.steeldefect.steeldefect_segmenter", registry_uri="databricks-uc")
"""

from __future__ import annotations

import hashlib
import numbers
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SLICES_TABLE = "steeldefect.ab_eval_slices"
DECISIONS_TABLE = "steeldefect.ab_decisions"
DECISION_IMAGES_TABLE = "steeldefect.ab_decision_images"
MANIFEST_TABLE = "steeldefect.model_training_images"
#: The session view of the slice's (image_id, coil_id) pairs that the job's queries join.
SLICE_VIEW = "ab_slice_images"
#: The registered model. On Unity Catalog a name is catalog.schema.model, and a short name
#: is completed with the session's current catalog and schema: pass the full name there.
MODEL_NAME = "steeldefect_segmenter"

#: The slice's identifier and cluster columns: one row per image, one image per coil.
ID_COL, CLUSTER_COL = "image_id", "coil_id"
#: Columns the job needs; the cascade also reads confidence_a / confidence_b when present.
REQUIRED_COLUMNS = ("image_id", "coil_id", "sampling_frame", "label",
                    "model_a_version", "pred_a", "model_b_version", "pred_b")


def _repo_root() -> str:
    """The checkout to import the package from.

    ``STEELDEFECT_WORKSPACE_ROOT`` when set (as ``jobs._load_script`` reads it), else the
    checkout that holds this script. It used to default to a hard-coded /Workspace/Repos
    path, placed ahead of the deployed checkout, so that any copy there shadowed the code
    deployed with the job.
    """
    return (os.environ.get("STEELDEFECT_WORKSPACE_ROOT")
            or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _bootstrap_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def as_version(value: Any, name: str) -> int:
    """A registry version from a job parameter: a positive integer (numpy's too, not a
    bool), or a string of ASCII digits (Databricks job parameters are strings)."""
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def as_flag(value: Any, name: str) -> bool:
    """A boolean from a job parameter: True or False, or "true" or "false" in any case.
    Any other value raises: the string "false" is truthy in Python."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def load_eval_slice(spark, eval_dataset_id: str):
    """The rows of one evaluation slice, as a pandas DataFrame.

    The id is a query parameter, never part of the SQL text: an id holding a quote used
    to select other slices.
    """
    _name(eval_dataset_id, "eval_dataset_id")
    return spark.sql(f"SELECT * FROM {SLICES_TABLE} WHERE eval_dataset_id = :eval_dataset_id",
                     args={"eval_dataset_id": eval_dataset_id}).toPandas()


def slice_errors(df, model_a_version: int, model_b_version: int) -> List[str]:
    """Why a slice cannot stand for the comparison of these two versions ([] when it can).

    The cascade checks the paired design (labels, one row per image, one image per coil);
    this checks what only the job knows: the required columns, string image and coil ids
    (the provenance checks compare them as strings), that every row was scored by the
    versions compared, and that the slice was drawn from a single sampling frame.
    """
    import pandas as pd

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"slice: missing column(s) {missing}"]
    errors = []
    for col in (ID_COL, CLUSTER_COL):
        not_str = ~df[col].map(lambda v: isinstance(v, str))
        if not_str.any():
            kinds = sorted({type(v).__name__ for v in df.loc[not_str, col]})
            errors.append(f"slice: {int(not_str.sum())} value(s) of {col} are not strings "
                          f"({kinds})")
    for col, want, model in (("model_a_version", model_a_version, "A"),
                             ("model_b_version", model_b_version, "B")):
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() | (values != want)
        if bad.any():
            found = sorted({str(v) for v in df.loc[bad, col].unique()})[:5]
            errors.append(f"slice: {int(bad.sum())} row(s) were scored by another version of "
                          f"{model} than v{want} ({col}: {found})")
    frames = set(df["sampling_frame"].dropna().astype(str))
    n_missing = int(df["sampling_frame"].isna().sum())
    if len(frames) != 1 or n_missing:
        errors.append(f"slice: sampling_frame must hold one value on every row "
                      f"({len(frames)} distinct, {n_missing} missing)")
    return errors


def slice_digest(df) -> Optional[str]:
    """SHA-256 of the slice's sorted image ids, one per line: the slice's identity, whatever
    its eval_dataset_id (None without image ids)."""
    if ID_COL not in df.columns or df.empty:
        return None
    ids = sorted(df[ID_COL].astype(str))
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def register_slice_view(spark, df) -> None:
    """The slice's (image_id, coil_id) pairs, as strings, in the session view the
    provenance queries join."""
    pairs = ([(str(i), str(c)) for i, c in zip(df[ID_COL], df[CLUSTER_COL])]
             if {ID_COL, CLUSTER_COL} <= set(df.columns) else [])
    (spark.createDataFrame(pairs, schema="image_id STRING, coil_id STRING")
     .createOrReplaceTempView(SLICE_VIEW))


def training_overlap_errors(spark, model_name: str, model_a_version: int,
                            model_b_version: int) -> List[str]:
    """Why the slice may hold images, or coils, that A or B was trained on ([] when not).

    Reads ``steeldefect.model_training_images``, which the training job must write (one row
    per training image, with its coil). The coil is the gate's unit of independence: an
    image of a coil that was partly used in training is not independent of the training
    set. A version with no row there cannot be checked, and fails closed.
    """
    found = spark.sql(
        f"SELECT m.model_version AS model_version,"
        f" COUNT(DISTINCT s.image_id) AS n_images, COUNT(DISTINCT c.coil_id) AS n_coils"
        f" FROM {MANIFEST_TABLE} m"
        f" LEFT JOIN {SLICE_VIEW} s ON m.image_id = s.image_id"
        f" LEFT JOIN {SLICE_VIEW} c ON m.coil_id = c.coil_id"
        f" WHERE m.model_name = :model_name AND m.model_version IN (:version_a, :version_b)"
        f" GROUP BY m.model_version",
        args={"model_name": model_name, "version_a": model_a_version,
              "version_b": model_b_version}).toPandas()
    overlap = {int(r.model_version): (int(r.n_images), int(r.n_coils))
               for r in found.itertuples()}
    errors = []
    for version, model in ((model_a_version, "A"), (model_b_version, "B")):
        if version not in overlap:
            errors.append(f"provenance: no training manifest for {model} ({model_name} "
                          f"v{version}) in {MANIFEST_TABLE}: the slice cannot be checked "
                          f"against its training images")
            continue
        n_images, n_coils = overlap[version]
        if n_images:
            errors.append(f"provenance: {n_images} image(s) of the slice are in the training "
                          f"manifest of {model} (v{version})")
        if n_coils:
            errors.append(f"provenance: {n_coils} coil(s) of the slice have images in the "
                          f"training manifest of {model} (v{version})")
    return errors


def prior_decisions(spark, model_name: str, model_b_version: int, eval_dataset_id: str,
                    digest: Optional[str]) -> List[Dict[str, Any]]:
    """The decisions already taken on this challenger that share the slice: the same
    eval_dataset_id or digest, or any image or coil of its recorded images.

    A decision on invalid input does not count: the gate looked at no evidence, and its row
    keeps none. Each one comes with its slice, its gate version and the statuses of its
    promotion rows.
    """
    decided = spark.sql(
        f"SELECT DISTINCT d.decision_id AS decision_id, d.model_a_version AS model_a_version,"
        f" d.eval_dataset_id AS eval_dataset_id, d.slice_sha256 AS slice_sha256,"
        f" d.gate_version AS gate_version, d.promotion_status AS promotion_status"
        f" FROM {DECISIONS_TABLE} d"
        f" LEFT JOIN {DECISION_IMAGES_TABLE} i ON i.decision_id = d.decision_id"
        f" LEFT JOIN {SLICE_VIEW} s ON s.image_id = i.image_id OR s.coil_id = i.coil_id"
        f" WHERE d.event = 'decision' AND d.model_name = :model_name"
        f" AND d.model_b_version = :version_b AND d.conclusion <> 'invalid_input'"
        f" AND (d.eval_dataset_id = :eval_dataset_id OR d.slice_sha256 = :slice_sha256"
        f" OR s.image_id IS NOT NULL)",
        args={"model_name": model_name, "version_b": model_b_version,
              "eval_dataset_id": eval_dataset_id, "slice_sha256": digest}).toPandas()
    promotions = spark.sql(
        f"SELECT decision_id, promotion_status FROM {DECISIONS_TABLE}"
        f" WHERE event = 'promotion' AND model_name = :model_name"
        f" AND model_b_version = :version_b",
        args={"model_name": model_name, "version_b": model_b_version}).toPandas()
    outcomes: Dict[str, List[str]] = {}
    for r in promotions.itertuples():
        outcomes.setdefault(str(r.decision_id), []).append(str(r.promotion_status))
    return [{"decision_id": str(r.decision_id), "model_a_version": int(r.model_a_version),
             "eval_dataset_id": r.eval_dataset_id, "slice_sha256": r.slice_sha256,
             "gate_version": r.gate_version, "promotion_status": r.promotion_status,
             "promotion_outcomes": outcomes.get(str(r.decision_id), [])}
            for r in decided.itertuples()]


def _resumable(prior: Dict[str, Any], model_a_version: int, eval_dataset_id: str,
               digest: Optional[str]) -> bool:
    """The same promotion, never settled: a promote_B on this very slice (its id and its
    images) against the same A, with no outcome yet (a crash between the two rows) or a
    registry error (failed, unknown). Refused and applied are final."""
    return (prior["promotion_status"] == "pending"
            and prior["model_a_version"] == model_a_version
            and prior["eval_dataset_id"] == eval_dataset_id
            and prior["slice_sha256"] == digest
            and not {"applied", "refused"} & set(prior["promotion_outcomes"]))


def ensure_tables(spark) -> None:
    """Create, when absent, the tables the job writes and the manifest table it reads
    (empty, it makes the provenance check fail closed, with its reason; the slice table is
    the training job's to create), and add to an older decisions table the columns it
    lacks, whose rows keep NULL there (``gate_version`` NULL marks them)."""
    from steeldefect_platform.obt_schema import COMPANION_DDL, ddl_columns

    for table in (DECISIONS_TABLE, DECISION_IMAGES_TABLE, MANIFEST_TABLE):
        spark.sql(COMPANION_DDL[table])
    have = {c.lower() for c in spark.table(DECISIONS_TABLE).columns}
    missing = [f"{name} {kind}" for name, kind in ddl_columns(DECISIONS_TABLE)
               if name.lower() not in have]
    if missing:
        spark.sql(f"ALTER TABLE {DECISIONS_TABLE} ADD COLUMNS ({', '.join(missing)})")


def append_rows(spark, table: str, rows: List[Dict[str, Any]]) -> None:
    """Append rows with the table's own schema: Spark does not infer the type of a column
    that is None on every row, and a TIMESTAMP column takes a datetime."""
    from steeldefect_platform.obt_schema import ddl_columns, ddl_schema

    names = [name for name, _ in ddl_columns(table)]
    data = [tuple(row.get(name) for name in names) for row in rows]
    (spark.createDataFrame(data, schema=ddl_schema(table))
     .write.format("delta").mode("append").saveAsTable(table))


def _number(value: Any) -> Optional[float]:
    from steeldefect_active_learning.ab_testing.serialization import strict
    value = strict(value)
    return None if value is None else float(value)


def _count(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _event(decision_id: str, event: str, gate_version: Optional[str],
           job: Dict[str, Any]) -> Dict[str, Any]:
    """The columns every event of a decision carries."""
    return {"decision_id": decision_id, "event": event,
            "event_ts": datetime.now(timezone.utc), "gate_version": gate_version,
            **{k: job[k] for k in ("model_name", "model_a_version", "model_a_uri",
                                   "model_b_version", "model_b_uri", "eval_dataset_id")}}


def decision_row(decision_id: str, verdict: Dict[str, Any], metrics: Optional[Dict[str, Any]],
                 job: Dict[str, Any], sampling_frame: Optional[str],
                 plan: Dict[str, Any]) -> Dict[str, Any]:
    """The audit row of a decision: what was compared, on which slice, with what outcome
    and why; the verdict, the diagnostics (none on invalid input) and the job's parameters
    whole, as strict JSON."""
    from steeldefect_active_learning.ab_testing.serialization import dumps

    est = verdict.get("estimates") or {}
    counts = verdict.get("counts") or {}
    failing = [r["criterion"] for r in verdict.get("reasons", []) if not r.get("passed")
               and not str(r.get("detail", "")).startswith("not evaluated")]
    return {
        **_event(decision_id, "decision", verdict.get("schema"), job),
        "sampling_frame": sampling_frame, "slice_sha256": job.get("slice_sha256"),
        "n_images": _count(verdict.get("n_images")),
        "decision": verdict["decision"], "conclusion": verdict.get("conclusion"),
        "failing_criteria": ",".join(failing) or None,
        "estimand": est.get("estimand"), "estimand_delta": _number(est.get("estimand_delta")),
        "estimand_lower": _number(est.get("estimand_lower")),
        "mcnemar_p": _number(est.get("mcnemar_p_one_sided")),
        "prob_b_better": _number(est.get("p_b_better_posterior")),
        **{cell: _count(counts.get(cell))
           for cell in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong")},
        "promotion_status": plan["status"], "promotion_detail": plan["detail"],
        "verdict_json": dumps(verdict),
        "diagnostics_json": (None if metrics is None else
                             dumps({k: v for k, v in metrics.items() if k != "paired_counts"})),
        "job_json": dumps(job),
    }


def promotion_row(decision_id: str, gate_version: Optional[str], job: Dict[str, Any],
                  outcome: Dict[str, Any]) -> Dict[str, Any]:
    """The audit row of a registry call: applied, refused, failed or unknown, and the
    champion before and after."""
    return {**_event(decision_id, "promotion", gate_version, job),
            "promotion_status": outcome["status"], "promotion_detail": outcome["detail"],
            "champion_before": _count(outcome["champion_before"]),
            "champion_after": _count(outcome["champion_after"]),
            "promoted_model_uri": outcome["promoted_model_uri"]}


def _settle(outcome: Dict[str, Any], decision_id: str) -> Optional[str]:
    """The URI promoted; raise when a promote_B was not applied, or may have been."""
    status, detail = outcome["status"], outcome["detail"]
    if status in ("refused", "failed"):
        raise RuntimeError(f"promote_B was not applied ({status}): {detail} "
                           f"(decision {decision_id})")
    if status == "unknown":
        raise RuntimeError(f"promote_B may have been applied (unknown): {detail} "
                           f"(decision {decision_id})")
    return outcome["promoted_model_uri"]


def main(spark, eval_dataset_id: str, model_a_version: Any, model_b_version: Any, *,
         model_name: str = MODEL_NAME, promote: Any = True,
         config_path: Optional[str] = None,
         registry_uri: Optional[str] = None) -> Optional[str]:
    """Decide on challenger ``model_b_version`` against champion ``model_a_version``.

    With ``promote`` False the decision is recorded and the registry is left alone (it is
    still the decision of this challenger on this slice). ``registry_uri`` (e.g.
    ``databricks-uc``) selects the model registry. A promote_B that is refused, fails or
    has an unknown outcome raises RuntimeError after its rows are written, so that the task
    fails. Returns the URI of the version promoted, else None.
    """
    a = as_version(model_a_version, "model_a_version")
    b = as_version(model_b_version, "model_b_version")
    if a == b:
        raise ValueError(f"model_a_version and model_b_version are both v{a}")
    promote = as_flag(promote, "promote")
    _name(eval_dataset_id, "eval_dataset_id")
    _name(model_name, "model_name")
    if registry_uri is not None:
        _name(registry_uri, "registry_uri")
    _bootstrap_path()
    from steeldefect_active_learning.ab_testing.metrics import compute_all_metrics
    from steeldefect_active_learning.ab_testing.paired import paired_counts
    from steeldefect_active_learning.ab_testing.promotion import (
        apply_promotion, decide_promotion, plan_promotion, version_uri,
    )
    from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config
    from steeldefect_platform.mlflow_registry import alias_version, use_registry

    if registry_uri is not None:
        use_registry(registry_uri)
    config_path = config_path or SHIPPED_CONFIG_PATH
    cfg = load_config(config_path).ab_testing
    cascade = dict(alpha=cfg.alpha, correction=cfg.multiple_testing_correction,
                   srm_threshold=cfg.srm_threshold, expected_ratio=cfg.expected_arm_ratio,
                   bootstrap_n=cfg.bootstrap_n,
                   bayesian_decision_threshold=cfg.bayesian_decision_threshold,
                   sequential=cfg.sequential)
    job = {"eval_dataset_id": eval_dataset_id, "model_name": model_name,
           "model_a_version": a, "model_a_uri": version_uri(model_name, a),
           "model_b_version": b, "model_b_uri": version_uri(model_name, b),
           "promote": promote, "config_path": config_path, "registry_uri": registry_uri,
           "id_col": ID_COL, "cluster_col": CLUSTER_COL, "cascade": cascade}
    ensure_tables(spark)
    df = load_eval_slice(spark, eval_dataset_id)
    register_slice_view(spark, df)
    job["slice_sha256"] = slice_digest(df)

    # One decision per challenger per slice: deciding again, under other thresholds, under
    # another id, or on a slice that shares images or coils, until B passes is a multiple
    # test that the gate's level does not cover. The one exception finishes a promotion
    # that a crash or a registry error left unsettled, when a run asks for it again: the
    # same slice (id and images), the same A and B, with promote.
    priors = prior_decisions(spark, model_name, b, eval_dataset_id, job["slice_sha256"])
    if priors:
        if (promote and len(priors) == 1
                and _resumable(priors[0], a, eval_dataset_id, job["slice_sha256"])):
            decision_id = priors[0]["decision_id"]
            plan = {"status": "pending", "detail": f"resume the promotion of {decision_id}",
                    "champion_before": None, "champion_after": None, "promoted_model_uri": None}
            outcome = apply_promotion(plan, model_name, b, a)
            append_rows(spark, DECISIONS_TABLE, [promotion_row(
                decision_id, priors[0]["gate_version"], job, outcome)])
            print(f"[ab_decision] resumed {decision_id}: promotion {outcome['status']}: "
                  f"{outcome['detail']}")
            return _settle(outcome, decision_id)
        raise RuntimeError(f"{model_name} v{b} already has a decision on this slice "
                           f"({', '.join(p['decision_id'] for p in priors)}): one decision "
                           f"per challenger per slice")

    # A wrong model name, an unreachable registry or a champion that moved stops the job
    # here, before it records anything: the slice stays available.
    if promote:
        champion = alias_version(model_name)
        if champion != a:
            raise RuntimeError(f"@champion of {model_name} is "
                               f"{'unset' if champion is None else f'v{champion}'}, but the "
                               f"slice was scored by v{a}: score it with the current champion")

    problems = slice_errors(df, a, b)
    if {ID_COL, CLUSTER_COL} <= set(df.columns):
        problems += training_overlap_errors(spark, model_name, a, b)
    pc = paired_counts(df, id_col=ID_COL, cluster_col=CLUSTER_COL)
    if problems or not pc["valid"]:
        # Invalid input: no diagnostic is computed, so that the row reveals nothing of the
        # outcome before the slice is fixed and the decision taken.
        pc = {**pc, "valid": False, "errors": problems + list(pc.get("errors") or [])}
        metrics, verdict = None, decide_promotion({"paired_counts": pc}, **cfg.gate_kwargs())
    else:
        metrics = compute_all_metrics(df, task="classification", bayesian=True,
                                      id_col=ID_COL, cluster_col=CLUSTER_COL, **cascade)
        verdict = decide_promotion(metrics, **cfg.gate_kwargs())
        if verdict.get("conclusion") == "invalid_input":   # the gate's configuration
            metrics = None
    frames = (set(df["sampling_frame"].dropna().astype(str))
              if "sampling_frame" in df.columns else set())

    plan = plan_promotion(verdict, model_name, job["model_b_uri"], b, a, promote=promote)
    decision_id = uuid.uuid4().hex
    if metrics is not None:   # an evaluation that counts: record the images it used
        append_rows(spark, DECISION_IMAGES_TABLE,
                    [{"decision_id": decision_id, "image_id": str(i), "coil_id": str(c)}
                     for i, c in zip(df[ID_COL], df[CLUSTER_COL])])
    append_rows(spark, DECISIONS_TABLE, [decision_row(
        decision_id, verdict, metrics, job, next(iter(frames)) if len(frames) == 1 else None,
        plan)])
    outcome = apply_promotion(plan, model_name, b, a)
    if plan["status"] == "pending":
        append_rows(spark, DECISIONS_TABLE, [promotion_row(decision_id, verdict.get("schema"),
                                                           job, outcome)])

    for reason in verdict["reasons"]:
        print(f"[ab_decision] {'PASS' if reason['passed'] else 'FAIL'} "
              f"{reason['criterion']}: {reason['detail']}")
    print(f"[ab_decision] decision {decision_id}: {verdict['decision']} "
          f"({verdict.get('conclusion')}); promotion {outcome['status']}: {outcome['detail']}")
    return _settle(outcome, decision_id)

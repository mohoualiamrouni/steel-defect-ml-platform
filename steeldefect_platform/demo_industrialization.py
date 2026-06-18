"""Live, local demonstration of the platform's *operational* guarantees.

Industrialisation is not measured in model accuracy (that is H1-H4) but in
**operational properties**: does the pipeline reject bad data, refuse to
duplicate on re-run, detect drift, and promote/roll-back models safely?  This
script exercises those properties by calling the **real** platform code and
prints a metrics table you can run live:

    python -m steeldefect_platform.demo_industrialization

What is real vs. emulated (kept honest):
  * Data-quality gate / dead-letter  -> REAL  ``obt_schema.validate_row`` + the
    exact ``bad_frac > max_bad_frac`` arithmetic of ``ingestion._handle_batch``.
  * Idempotent ingest                -> the MERGE *key-dedup* semantics
    (``whenNotMatchedInsertAll`` on ``segmentation_id``) emulated in-memory;
    the production write is a Delta ``MERGE`` (needs a cluster -- runbook).
  * Drift detection power            -> REAL  ``drift.mmd_drift`` /
    ``drift.ADWINState`` over Monte-Carlo trials.
  * Model registry round-trip        -> REAL  ``mlflow_registry.register /
    promote / load_production`` against a local **sqlite** MLflow registry.
  * Jobs DAG                         -> REAL  ``jobs.topological_order``.

No Databricks, no GPU.  All numbers are produced at run time, never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a binomial rate k/n (honest small-sample interval)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steeldefect_platform import drift, jobs, obt_schema  # noqa: E402
from steeldefect_platform.obt_schema import validate_row  # noqa: E402


# ---------------------------------------------------------------------------
# helpers to synthesise OBT rows
# ---------------------------------------------------------------------------

def _valid_row(i: int) -> Dict:
    return {"segmentation_id": f"seg-{i:06d}", "coil_id": f"coil-{i % 50}",
            "defect_kind": "local", "pos_x_m": 0.1, "pos_y_m": 0.2}


_INVALID_MAKERS = [
    lambda i: {"segmentation_id": f"seg-{i:06d}", "defect_kind": "spurious"},               # bad kind
    lambda i: {"segmentation_id": f"seg-{i:06d}", "defect_kind": "local"},                  # local w/o position
    lambda i: {"segmentation_id": f"seg-{i:06d}", "defect_kind": "group"},                  # group w/o density
    lambda i: {"segmentation_id": f"seg-{i:06d}", "defect_kind": "local",
               "pos_x_m": 0, "pos_y_m": 0, "expert_group_id": 99},                          # bad group id
    lambda i: {"segmentation_id": f"seg-{i:06d}", "defect_kind": "local",
               "pos_x_m": 0, "pos_y_m": 0, "cnn_vector": [0.0] * 10},                       # bad vector len
]


def _make_batch(n: int, bad_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_bad = int(round(n * bad_frac))
    rows = [_valid_row(i) for i in range(n - n_bad)]
    rows += [_INVALID_MAKERS[i % len(_INVALID_MAKERS)](10_000 + i) for i in range(n_bad)]
    rng.shuffle(rows)
    return rows, n_bad


# ---------------------------------------------------------------------------
# 1. Data-quality gate + dead-letter  (REAL validate_row + real gate arithmetic)
# ---------------------------------------------------------------------------

def _route_and_gate(rows: List[Dict], max_bad_frac: float):
    """Faithful local replay of ``ingestion._handle_batch``: tag each row with
    the REAL ``validate_row``, route rejects to a dead-letter list, then apply
    the loud batch gate with the SAME ``bad_frac > max_bad_frac`` ``raise`` as
    production (ingestion.py:135). Returns (accepted, dead_letter, bad_frac);
    raises ``ValueError`` exactly as the production gate does. Note: production
    writes valid rows + dead-letters BEFORE this raise — the gate is a loud
    tripwire, not a write barrier."""
    accepted, dead_letter = [], []
    for r in rows:
        try:
            validate_row(r)               # production code path
            accepted.append(r)
        except ValueError as exc:
            dead_letter.append((r.get("segmentation_id"), str(exc)))
    total = len(rows)
    bad_frac = len(dead_letter) / total if total else 0.0
    if bad_frac > max_bad_frac:            # identical to ingestion.py:135-139
        raise ValueError(
            f"data-quality gate failed: {bad_frac:.1%} invalid rows "
            f"(> {max_bad_frac:.1%}); rejects in dead-letter table"
        )
    return accepted, dead_letter, bad_frac


def demo_dq_gate() -> Dict:
    out = {}
    for label, frac in [("clean (1% bad)", 0.01), ("dirty (5% bad)", 0.05)]:
        rows, n_bad = _make_batch(1000, frac, seed=1)
        gate_raised, raised_msg, acc, dlq, bad_frac = False, None, [], [], frac
        try:
            acc, dlq, bad_frac = _route_and_gate(rows, max_bad_frac=0.02)
        except ValueError as exc:          # the loud gate actually fired
            gate_raised, raised_msg = True, str(exc)
            # recompute routing for reporting (the raise short-circuits returns)
            for r in rows:
                try:
                    validate_row(r); acc.append(r)
                except ValueError as e:
                    dlq.append((r.get("segmentation_id"), str(e)))
            bad_frac = len(dlq) / len(rows)
        misrouted = abs(len(dlq) - n_bad)  # 0 here (deterministic, pre-labelled)
        reasons = {}
        for _, why in dlq:
            key = why.split(",")[0].split(" got")[0][:40]
            reasons[key] = reasons.get(key, 0) + 1
        out[label] = {"total": len(rows), "accepted": len(acc),
                      "dead_lettered": len(dlq), "bad_frac": round(bad_frac, 4),
                      "gate_raised": gate_raised, "raised_msg": raised_msg,
                      "misrouted_rows": misrouted, "reason_breakdown": reasons}
    return out


# ---------------------------------------------------------------------------
# 2. Idempotent ingest  (MERGE whenNotMatchedInsertAll on segmentation_id)
# ---------------------------------------------------------------------------

def _merge_insert(store: Dict, rows: List[Dict]) -> int:
    inserted = 0
    for r in rows:
        k = r["segmentation_id"]
        if k not in store:           # == whenNotMatchedInsertAll
            store[k] = r
            inserted += 1
    return inserted


def demo_idempotency() -> Dict:
    rows, _ = _make_batch(1000, 0.0, seed=2)            # 1000 unique, all valid
    store: Dict = {}
    first = _merge_insert(store, rows)
    second = _merge_insert(store, rows)                 # re-run the SAME batch
    append_total = len(rows) * 2                        # what a naive append yields
    return {"batch_rows": len(rows), "after_first_ingest": len(store),
            "inserted_on_reingest": second,
            "emulated_merge_duplication_rate": round(second / len(rows), 4),
            "naive_append_duplication_rate": round((append_total - len(rows)) / len(rows), 4),
            "note": "in-memory key-dedup; production write is a Delta MERGE (cluster)"}


# ---------------------------------------------------------------------------
# 3. Drift detection power  (REAL mmd_drift + ADWIN)
# ---------------------------------------------------------------------------

def _rate(k: int, n: int) -> Dict:
    """A binomial rate reported honestly: point estimate + Wilson 95% CI."""
    return {"rate": round(k / n, 3) if n else None, "k": k, "n": n,
            "ci95": _wilson_ci(k, n)}


def demo_drift(n=150, d=8, trials=40, n_perm=199, alpha=0.05, noise_sd=0.01) -> Dict:
    # MMD power curve: detection rate (with CI) vs per-dim mean shift.
    shifts = [0.0, 0.1, 0.2, 0.3, 0.5]
    mmd_curve = {}
    for s in shifts:
        flags = 0
        for t in range(trials):
            rng = np.random.default_rng(1000 + t)
            ref = rng.normal(0.0, 1.0, size=(n, d))
            cur = rng.normal(s, 1.0, size=(n, d))
            _, p = drift.mmd_drift(ref, cur, n_permutations=n_perm, random_state=t)
            flags += int(p < alpha)
        mmd_curve[f"shift={s}"] = _rate(flags, trials)

    # ADWIN: include near-threshold jumps (1-2 sigma) so the delay/miss tradeoff
    # is non-trivial, not only the easy 5-20 sigma steps. SNR = jump / noise_sd.
    adwin = {}
    for jump in [0.01, 0.02, 0.05, 0.10, 0.20]:
        delays = []
        for t in range(trials):
            rng = np.random.default_rng(2000 + t)
            pre = list(0.90 + rng.normal(0, noise_sd, 40))
            post = list(0.90 - jump + rng.normal(0, noise_sd, 40))
            st = drift.ADWINState()
            delay = None
            for k, v in enumerate(pre + post):
                if st.update(v) and k >= 40 and delay is None:
                    delay = k - 40
            delays.append(delay)
        detected = [x for x in delays if x is not None]
        adwin[f"jump={jump}"] = {
            "snr_sigma": round(jump / noise_sd, 1),
            "detection": _rate(len(detected), trials),
            "median_delay_steps": (round(float(np.median(detected)), 1) if detected else None),
            "p90_delay_steps": (int(np.percentile(detected, 90)) if detected else None),
            "max_delay_steps": (max(detected) if detected else None)}
    # false alarms on a stationary stream
    fa = 0
    for t in range(trials):
        rng = np.random.default_rng(3000 + t)
        st = drift.ADWINState()
        fired = any(st.update(v) for v in 0.90 + rng.normal(0, noise_sd, 80))
        fa += int(fired)
    adwin["stationary_false_alarm"] = _rate(fa, trials)

    return {"mmd_detection_rate": mmd_curve,
            "mmd_false_positive@shift0": mmd_curve["shift=0.0"],
            "mmd_nominal_alpha": alpha, "adwin": adwin,
            "interpretation": ("MMD power rises with effect size; FP rate at shift 0 "
                               "is consistent with nominal alpha (see CI). ADWIN rows "
                               "report SNR so the easy (>=5 sigma) vs hard (1-2 sigma) "
                               "regimes are distinguishable. Rates carry Wilson 95% CIs; "
                               "{} trials only bound them loosely.".format(trials)),
            "config": {"n": n, "d": d, "trials": trials, "n_perm": n_perm,
                       "noise_sd": noise_sd}}


# ---------------------------------------------------------------------------
# 4. Model-registry round-trip  (REAL register / promote / load_production)
# ---------------------------------------------------------------------------

def demo_registry() -> Dict:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:                                   # pragma: no cover
        return {"skipped": f"mlflow/sklearn unavailable: {exc}"}

    from steeldefect_platform import mlflow_registry as reg

    tmp = tempfile.mkdtemp(prefix="steeldefect_mlflow_")
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    exp_id = mlflow.create_experiment("steeldefect_demo",
                                      artifact_location=f"file:///{tmp.replace(os.sep, '/')}/art")
    mlflow.set_experiment(experiment_id=exp_id)
    client = MlflowClient()
    NAME = "steeldefect_segmenter"
    probe = np.array([[2.0, 2.0]])

    # two distinguishable models: A predicts 1 on probe, B predicts 0
    XA = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]); yA = [1, 1, 0, 0]
    XB = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]); yB = [0, 0, 1, 1]
    modelA = LogisticRegression().fit(XA, yA)
    modelB = LogisticRegression().fit(XB, yB)

    def _version_after(register_fn):
        register_fn()
        vs = [int(mv.version) for mv in client.search_model_versions(f"name='{NAME}'")]
        return max(vs)

    with mlflow.start_run() as run:
        vA = _version_after(lambda: reg.register(run, modelA, NAME))
    with mlflow.start_run() as run:
        vB = _version_after(lambda: reg.register(run, modelB, NAME))

    def _timed(fn):
        t0 = time.perf_counter(); r = fn(); return r, (time.perf_counter() - t0) * 1e3

    steps = []
    for desc, ver, expect in [("promote vA -> champion", vA, int(modelA.predict(probe)[0])),
                              ("promote vB -> champion", vB, int(modelB.predict(probe)[0])),
                              ("rollback to vA",          vA, int(modelA.predict(probe)[0]))]:
        _, t_promote = _timed(lambda: reg.promote(NAME, ver, alias=reg.PROD_ALIAS))
        alias_v = int(client.get_model_version_by_alias(NAME, reg.PROD_ALIAS).version)
        model, t_load = _timed(lambda: reg.load_production(NAME))
        got = int(model.predict(probe)[0])
        # identity/routing check (NOT accuracy): does @champion serve the exact
        # artifact we promoted? probe-label differs per version by construction.
        steps.append({"step": desc, "alias_resolves_to": f"v{alias_v}",
                      "alias_correct": alias_v == ver,
                      "serves_expected_artifact": got == expect,
                      "promote_ms": round(t_promote, 1), "load_ms": round(t_load, 1)})

    ok = all(s["alias_correct"] and s["serves_expected_artifact"] for s in steps)
    return {"versions_registered": [vA, vB], "round_trip_ok": ok, "steps": steps}


# ---------------------------------------------------------------------------
# 5. Jobs DAG topological validity
# ---------------------------------------------------------------------------

def demo_dag() -> Dict:
    order = jobs.topological_order()
    pos = {n: i for i, n in enumerate(order)}
    valid = all(pos[d] < pos[n] for n, deps in jobs.JOBS_DAG.items() for d in deps)
    return {"nodes": len(order), "edges": sum(len(v) for v in jobs.JOBS_DAG.values()),
            "topological_order_valid": valid, "order": order}


# ---------------------------------------------------------------------------

def main(trials: int = 40, n_perm: int = 199) -> Dict:
    t0 = time.perf_counter()
    results = {
        "dq_gate": demo_dq_gate(),
        "idempotency": demo_idempotency(),
        "drift": demo_drift(trials=trials, n_perm=n_perm),
        "registry": demo_registry(),
        "dag": demo_dag(),
        "obt_constants": {"CNN_DIM": obt_schema.CNN_DIM,
                          "N_MORPHO": obt_schema.N_MORPHO,
                          "N_EXPERT_GROUPS": obt_schema.N_EXPERT_GROUPS},
        "wall_clock_s": None,
    }
    results["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    _print(results)
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "industrialization_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[written] {path}")
    return results


def _print(r: Dict) -> None:
    p = print
    p("\n" + "=" * 70)
    p("  STEELDEFECT PLATFORM - LIVE INDUSTRIALISATION METRICS (local, synthetic)")
    p("=" * 70)

    def _ci(x):  # "(CI 0-9%)"
        lo, hi = x["ci95"]
        return f"[CI {lo:.0%}-{hi:.0%}]"

    p("\n[1] DATA-QUALITY GATE + DEAD-LETTER  (REAL validate_row + gate arithmetic from _handle_batch)")
    for k, v in r["dq_gate"].items():
        verdict = "FAILED (raised loudly)" if v["gate_raised"] else "PASSED"
        p(f"  {k:16s}: {v['accepted']}/{v['total']} accepted, "
          f"{v['dead_lettered']} dead-lettered ({v['bad_frac']:.1%}), "
          f"gate {verdict}, misrouted={v['misrouted_rows']}")
        if v["raised_msg"]:
            p(f"                    raise -> {v['raised_msg']}")

    i = r["idempotency"]
    p("\n[2] IDEMPOTENT INGEST  (key-dedup semantics, EMULATED in-memory; prod = Delta MERGE)")
    p(f"  re-ingesting an identical {i['batch_rows']}-row batch inserted "
      f"{i['inserted_on_reingest']} rows  ->  emulated-MERGE dup-rate "
      f"{i['emulated_merge_duplication_rate']:.0%}  vs  naive-append "
      f"{i['naive_append_duplication_rate']:.0%}  (Delta MERGE itself needs a cluster)")

    d = r["drift"]
    cfg = d["config"]
    p(f"\n[3] DRIFT DETECTION  (REAL MMD permutation test + ADWIN, on synthetic shifts; "
      f"{cfg['trials']} trials, Wilson 95% CI)")
    p(f"  MMD detection rate vs per-dim mean-shift  (FP at shift 0 should ~ alpha={d['mmd_nominal_alpha']}):")
    for k, v in d["mmd_detection_rate"].items():
        p(f"      {k:12s} -> {v['rate']:.0%} {_ci(v)}  (k={v['k']}/{v['n']})")
    sfa = d["adwin"]["stationary_false_alarm"]
    p(f"  ADWIN stationary false-alarm: {sfa['rate']:.0%} {_ci(sfa)}")
    for k, v in d["adwin"].items():
        if k.startswith("jump"):
            det = v["detection"]
            p(f"      {k:10s} ({v['snr_sigma']:>4}sigma) -> detect {det['rate']:.0%} {_ci(det)}, "
              f"median/p90/max delay {v['median_delay_steps']}/{v['p90_delay_steps']}/"
              f"{v['max_delay_steps']} steps")

    g = r["registry"]
    p("\n[4] MODEL-REGISTRY ROUND-TRIP  (REAL MLflow alias promote/rollback; identity check, not accuracy)")
    if g.get("skipped"):
        p(f"  skipped: {g['skipped']}")
    else:
        p(f"  registered versions {g['versions_registered']}, round-trip OK: {g['round_trip_ok']}")
        for s in g["steps"]:
            p(f"      {s['step']:24s} -> {s['alias_resolves_to']}, "
              f"serves-expected-artifact={s['serves_expected_artifact']}, "
              f"promote {s['promote_ms']}ms / load {s['load_ms']}ms")

    dag = r["dag"]
    p("\n[5] JOBS DAG")
    p(f"  {dag['nodes']} nodes, {dag['edges']} edges, topological order valid: "
      f"{dag['topological_order_valid']}")
    p(f"\n  (total wall-clock {r['wall_clock_s']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live local industrialisation metrics.")
    ap.add_argument("--trials", type=int, default=40,
                    help="Monte-Carlo trials per drift point (more = tighter CIs, slower)")
    ap.add_argument("--n-perm", type=int, default=199,
                    help="MMD permutations (more = finer p-value resolution)")
    a = ap.parse_args()
    main(trials=a.trials, n_perm=a.n_perm)

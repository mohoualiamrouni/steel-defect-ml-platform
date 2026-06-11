"""Databricks Jobs orchestration for the SteelDefect platform.

``JOBS_DAG`` declares the task dependency graph; the ``job_*`` functions are the
entrypoints each Databricks task calls.  The heavy lifting lives in the
per-subject ``databricks/`` modules (created under each pipeline package); these
wrappers exist so the DAG is declared in one place and the platform owns the
schedule.

DAG (see report appendix ``app:mlops``)::

    ingest (stream/availableNow)
      ├─ al_pool_build (weekly) ─ [expert label] ─ MERGE is_labeled
      │     └─ segmenter_train ─ ab_decision ─(promote_B?)─ promote(steeldefect_segmenter)
      ├─ defect_regroup_train (per release) ─ register(steeldefect_defect_clusterer)
      │     └─ online_regroup (stream)
      └─ severity_score (per-coil µbatch) ─ upsert severity ─ cutting_plan
    drift_monitor (hourly) ─(ADWIN/MMD alarm)─ triggers segmenter_train + regroup_train

No Spark import at module load.
"""

from __future__ import annotations

import os
from typing import Dict, List

#: task -> list of upstream task names it depends on.
JOBS_DAG: Dict[str, List[str]] = {
    "ingest": [],
    "al_pool_build": ["ingest"],
    "segmenter_train": ["al_pool_build"],
    "ab_decision": ["segmenter_train"],
    "defect_regroup_train": ["ingest"],
    "online_regroup": ["defect_regroup_train"],
    "severity_score": ["ingest"],
    "cutting_plan": ["severity_score"],
    "drift_monitor": ["ingest"],
    "optimize_obt": ["ingest"],  # periodic OPTIMIZE ... ZORDER maintenance
    # P4 self-supervised perception (release-triggered, off the daily cron):
    "ssl_pretrain": ["ingest"],          # SSL foundation encoder on the unlabeled stream
    "embedding_refresh": ["ssl_pretrain"],  # re-embed OBT + probe uplift
    "openset_score": ["ingest"],         # stream novelty scoring -> steeldefect.novelty_candidates
}


def _spark():
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()


def _load_script(rel_path: str):
    """Import a numeric-prefixed Databricks job script by file path.

    Job scripts (e.g. ``01_al_pool_build.py``) follow the flatness pilot naming
    and are not importable as ``module.name``; in Databricks a Job task points
    at the notebook path directly.  Here we load them via importlib so the
    orchestrator can call their ``main()`` uniformly.
    """
    import importlib.util
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, rel_path)
    spec = importlib.util.spec_from_file_location(
        rel_path.replace("/", ".").replace(".py", ""), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_ingest(**kw) -> None:
    from .ingestion import ingest_steeldefect_stream
    ingest_steeldefect_stream(_spark(), **kw)


def job_al_pool_build(cycle_start: str, budget: int = 200, **kw) -> None:
    m = _load_script("steeldefect_active_learning/databricks/01_al_pool_build.py")
    m.main(_spark(), cycle_start=cycle_start, budget=budget, **kw)


def job_ab_decision(eval_dataset_id: str, model_a_uri: str, model_b_uri: str) -> None:
    m = _load_script("steeldefect_active_learning/databricks/02_ab_decision.py")
    m.main(_spark(), eval_dataset_id, model_a_uri, model_b_uri)


def job_defect_regroup(run_id: str, target_k: int = 28) -> None:
    m = _load_script("defect_clustering/databricks/02_regroup_train.py")
    m.main(_spark(), run_id=run_id, target_k=target_k)


def job_severity(since: str = "2026-01-01") -> None:
    m = _load_script("datamining/databricks/01_severity_score.py")
    m.main(_spark(), since=since)


def job_cutting(since: str = "2026-01-01") -> None:
    m = _load_script("datamining/databricks/02_cutting_plan.py")
    m.main(_spark(), since=since)


def job_drift_monitor(reference_campaign: str, current_campaign: str,
                      obt_table: str = "steeldefect.obt_segmentations") -> Dict:
    from .drift import cohort_features, mmd_drift
    spark = _spark()
    ref = cohort_features(spark, obt_table, reference_campaign)
    cur = cohort_features(spark, obt_table, current_campaign)
    mmd2, p = mmd_drift(ref, cur)
    return {"mmd2": mmd2, "p_value": p, "drift": bool(p < 0.05)}


def job_optimize_obt(table: str = "steeldefect.obt_segmentations") -> None:
    """Periodic file-compaction + Z-ORDER maintenance pass on the OBT."""
    from .obt_schema import optimize_obt
    optimize_obt(_spark(), table=table)


def job_ssl_pretrain(run_id: str, epochs: int = 20, **kw) -> None:
    m = _load_script("steeldefect_ssl/databricks/01_ssl_pretrain.py")
    m.main(_spark(), run_id=run_id, epochs=epochs, **kw)


def job_ssl_refresh(run_id: str, **kw) -> None:
    m = _load_script("steeldefect_ssl/databricks/02_embedding_refresh.py")
    m.main(_spark(), run_id=run_id, **kw)


def job_ssl_openset(since: str = "2026-01-01", run_id: str = "ssl-latest", **kw) -> Dict:
    m = _load_script("steeldefect_ssl/databricks/03_openset_score.py")
    return m.main(_spark(), since=since, run_id=run_id, **kw)


def topological_order() -> List[str]:
    """Return a valid execution order of JOBS_DAG (Kahn's algorithm)."""
    indeg = {k: len(v) for k, v in JOBS_DAG.items()}
    order: List[str] = []
    ready = [k for k, d in indeg.items() if d == 0]
    while ready:
        n = ready.pop()
        order.append(n)
        for m, deps in JOBS_DAG.items():
            if n in deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    return order

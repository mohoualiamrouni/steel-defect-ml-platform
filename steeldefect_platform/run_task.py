"""Uniform task dispatcher for the SteelDefect platform Databricks job.

A Databricks Asset Bundle (``databricks.yml``) runs each DAG node as a
``spark_python_task`` that points at THIS file and passes the task name as the
first parameter::

    python run_task.py <task> --params '{"cycle_start": "2026-06-01"}'

The dispatcher maps a ``JOBS_DAG`` task name to its platform entrypoint in
:mod:`steeldefect_platform.jobs`, so the whole pipeline is invoked through one
deployable, testable surface instead of a pile of ad-hoc notebooks.

Conceptual nodes intentionally absent from :data:`TASKS` (declared as such in
the bundle and the runbook): ``segmenter_train`` (external GPU training, runs
off-cluster) and ``online_regroup`` (a long-running structured-streaming task
deployed as a continuous job, not a scheduled batch task).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# spark_python_task runs this as a FILE, so make the package importable
# regardless of the working directory the Databricks task starts in.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steeldefect_platform import jobs  # noqa: E402

#: task name (matches steeldefect_platform.jobs.JOBS_DAG) -> entrypoint callable.
TASKS = {
    "ingest": jobs.job_ingest,
    "al_pool_build": jobs.job_al_pool_build,
    "ab_decision": jobs.job_ab_decision,
    "defect_regroup_train": jobs.job_defect_regroup,
    "severity_score": jobs.job_severity,
    "cutting_plan": jobs.job_cutting,
    "drift_monitor": jobs.job_drift_monitor,
    "optimize_obt": jobs.job_optimize_obt,
    "ssl_pretrain": jobs.job_ssl_pretrain,
    "embedding_refresh": jobs.job_ssl_refresh,
    "openset_score": jobs.job_ssl_openset,
}


def main(argv=None) -> int:
    import time

    from steeldefect_platform.logging_utils import get_logger

    parser = argparse.ArgumentParser(description="SteelDefect platform task runner")
    parser.add_argument("task", choices=sorted(TASKS),
                        help="JOBS_DAG node to execute")
    parser.add_argument("--params", default="{}",
                        help="JSON object of keyword arguments for the task")
    args = parser.parse_args(argv)

    log = get_logger("steeldefect.run_task")
    kwargs = json.loads(args.params)
    log.info("task '%s' started (params=%s)", args.task, kwargs)
    t0 = time.time()
    try:
        result = TASKS[args.task](**kwargs)
    except Exception:
        # Log a full traceback before failing the Databricks task (loud failure).
        log.exception("task '%s' FAILED after %.1fs", args.task, time.time() - t0)
        raise
    log.info("task '%s' done in %.1fs", args.task, time.time() - t0)
    if result is not None:  # e.g. drift_monitor returns a verdict dict
        print(json.dumps(result, default=str))  # machine-readable result on stdout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

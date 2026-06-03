"""steeldefect_platform — shared Azure Databricks / Delta "One Big Table" spine.

This package generalises the production deployment pattern proven by the
flatness-clustering pilot (``flatness_clustering/databricks/``: Delta landing →
applyInPandas UDF → Delta feature table → MLflow Model Registry → batch /
structured-streaming assignment) into a single substrate on which the three
featured subjects sit:

* Subject 1 — Active Learning + A/B testing  (steeldefect_active_learning/databricks/)
* Subject 2 — Defect-group re-clustering     (defect_clustering/databricks/)
* Subject 3 — Datamining + cutting plan       (datamining/databricks/)

Modules
-------
* :mod:`obt_schema`      — the OBT Delta schema (single source of truth) + DDL.
* :mod:`io_utils`        — array<->blob and crop/mask decoding helpers.
* :mod:`mlflow_registry` — load / register / promote helpers over MLflow.
* :mod:`drift`           — ADWIN + MMD drift monitors over OBT cohorts.
* :mod:`ingestion`       — inspection landing → OBT (batch + streaming) + upsert.
* :mod:`jobs`            — Databricks Jobs DAG + task entrypoints.

Nothing here imports Spark or MLflow at module load; those are imported lazily
inside the functions that need them, so the package is usable (schema, DDL,
drift math, IO helpers) in a plain-Python / no-cluster context.
"""

from __future__ import annotations

from . import drift, io_utils, mlflow_registry, obt_schema
from .obt_schema import (
    CNN_DIM,
    MORPHO_NAMES,
    N_EXPERT_GROUPS,
    OBT_TABLE,
    SOURCE_SYNTHETIC,
    obt_struct_type,
    validate_row,
)

__all__ = [
    "obt_schema",
    "io_utils",
    "mlflow_registry",
    "drift",
    "OBT_TABLE",
    "CNN_DIM",
    "MORPHO_NAMES",
    "N_EXPERT_GROUPS",
    "SOURCE_SYNTHETIC",
    "obt_struct_type",
    "validate_row",
]

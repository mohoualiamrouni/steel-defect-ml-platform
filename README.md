# SteelDefect: ML and MLOps platform for steel surface defect analytics

Coordinated machine learning pipelines on one shared Azure Databricks / Delta Lake
platform (the *One Big Table*, `steeldefect.obt_segmentations`), built around an in-line
steel surface inspection system. They turn its output (a 960-d CNN embedding and 24
morphology features per defect) into annotation savings, a data-driven defect taxonomy,
an optimized cutting plan, coil flatness families and a self-improving perception core.

| Pipeline | What it does | Code |
|---|---|---|
| **P1: Active learning and A/B** | select the most informative images to label; decide model promotion with a fail-closed statistical gate | `steeldefect_active_learning/` |
| **P2: Defect regrouping** | fuse CNN and morphology features, discover defect families (DEC, DeepDPM) | `defect_clustering/` |
| **P3: Datamining and cutting** | severity regression (XGBoost with conformalized quantile regression), then a guillotine cutting-plan optimizer | `datamining/` |
| **P4: Self-supervised core** | contrastive encoder (NT-Xent), open-set detection, generalized category discovery, continual NCM | `steeldefect_ssl/` |
| **Flatness clustering** | coil flatness profiles: statistical and ROCKET / MiniRocket features, DEC-style clustering | `flatness_clustering/` |
| **Platform** | OBT schema, jobs DAG, MLflow registry, ingestion, drift monitoring | `steeldefect_platform/`, `databricks.yml` |

## Data honesty

All synthetic data is **calibrated-synthetic**, and every row carries
`source_version = "synthetic-cal-v1"`. This validates the *architecture and methods*,
not production numbers; a production feed would carry `steeldefect-prod-*` instead.

## Results on public real data (NEU-DET)

P1 and P2 were also evaluated on NEU-DET, a public dataset of 1,800 hot-rolled steel
strip images in six defect classes (`run_neu_p1p2.py`, summary in
`outputs_neu_p1p2_summary.json`). Neither hypothesis is confirmed on this dataset, and
both results are reported as measured:

- **P2, feature fusion**: fusing morphology features with the CNN embedding gives an
  NMI of 0.8776 (95% bootstrap CI 0.8643 to 0.8963, 100 resamples), against 0.8765 for
  the CNN embedding alone. The +0.0011 difference is not significant.
- **P1, active learning**: TypiClust reaches a macro-F1 of 0.9034 with 100 labels, and
  random sampling matches it at the same budget: no labeling savings on this dataset.

To reproduce, download NEU-DET and place its class folders under
`steeldefect_ssl/data/neu_det/raw/`; the split manifests in that folder list the images
used.

## Quickstart

```bash
pip install -r requirements.txt   # core ML deps (pyspark/delta come from the Databricks Runtime)
pytest -q                         # unit tests across the pipelines and the platform
python -m steeldefect_platform.demo_industrialization
```

The last command replays the platform's operational guarantees on synthetic data:
data-quality gate and dead-letter, idempotent ingest (Delta MERGE semantics emulated
in memory), drift detection (MMD, ADWIN), MLflow promotion and rollback against a local
registry, and the jobs DAG ordering. No Databricks workspace and no GPU are needed.

To deploy on Databricks, see [`DEPLOY_RUNBOOK.md`](DEPLOY_RUNBOOK.md)
(`databricks bundle validate / deploy / run`). CI runs a syntax check and the test
suites on every push (`.github/workflows/ci.yml`).

## Status

Production-grade architecture and MLOps idioms (idempotent MERGE ingestion with
dead-letter and data-quality gate, jobs DAG, MLflow alias-based registry, drift
monitoring, A/B gate, autoscaling bundle, CI), validated on synthetic data and on CPU.
At-scale Spark/Delta runs, real-data and GPU training, Unity Catalog governance and live
dashboards are the next steps.

## Layout

```
steeldefect_active_learning/  P1   defect_clustering/  P2   datamining/  P3
steeldefect_ssl/              P4   flatness_clustering/     coil flatness clustering
steeldefect_platform/  OBT schema, jobs DAG, MLflow registry, ingestion, drift, tests
databricks.yml         Databricks Asset Bundle (jobs, schedules, clusters)
DEPLOY_RUNBOOK.md      deploy and operate
```

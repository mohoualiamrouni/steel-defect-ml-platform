# SteelDefect platform — deploy & capture proof (runbook)

Goal: deploy `databricks.yml` on a real workspace and capture **three screenshots**
(synthetic data: this shows that the **architecture** runs, not production numbers).

> **No workspace handy?** `python -m steeldefect_platform.demo_industrialization`
> runs the *operational* guarantees locally in ~70 s (no cluster, no GPU):
> the real data-quality gate (it raises loudly on a dirty batch), drift
> MMD/ADWIN power with confidence intervals, and a real MLflow alias
> promote→rollback round-trip. It prints a metrics table and writes
> `steeldefect_platform/outputs/industrialization_metrics.json`. The Delta
> `MERGE`/`OPTIMIZE` parts below still need a real workspace.

## What you'll end up with
1. **Jobs run-graph** — the `steeldefect_obt_daily` DAG (`ingest → severity_score →
   cutting_plan`, plus `drift_monitor` and `optimize_obt`) showing green tasks.
2. **MLflow** — the registered model versions and their `champion` / `challenger`
   aliases (the bundle logs no A/B comparison).
3. **Delta history** — `DESCRIBE HISTORY steeldefect.obt_segmentations` showing the
   `WRITE` / `MERGE` / `OPTIMIZE` operations (time travel is real).

---

## 0. Prerequisites (local machine)
- Python 3.10+ and this repo.
- The **Databricks CLI v0.2x** (the new one, not the legacy `pip install databricks-cli`):
  - Windows: `winget install Databricks.DatabricksCLI` — then `databricks version`.

## 1. Get a free workspace

**Path A — Azure free trial (recommended: the platform targets Azure Databricks).**
Azure gives a 30-day / credit trial. Create an **Azure Databricks** resource
(Premium tier needed for Unity Catalog/MLflow Registry), open the workspace.
The bundle runs on serverless compute (no job clusters), so it deploys as-is here.

**Path B — Databricks Free Edition (zero cost, fastest).**
Sign up at the Databricks Free Edition. It is **serverless-only**, which is what the
bundle uses, so no change is needed. Jobs, Unity Catalog and MLflow work.

## 2. Authenticate the CLI
```bash
databricks auth login --host https://adb-XXXX.azuredatabricks.net
# opens a browser (OAuth). Confirm:
databricks current-user me
```

## 3. Point the bundle at your workspace
The bundle does not pin a workspace: the CLI targets the workspace of the profile
you authenticated in step 2, or the one in `DATABRICKS_HOST`. To use another
profile, pass `-p <profile>` or set `DATABRICKS_CONFIG_PROFILE`. Then validate:
```bash
databricks bundle validate -t dev
```
This parses the bundle and resolves every task — a green validate already proves
the infra-as-code is well-formed.

## 4. Seed the OBT with synthetic data (so MLflow/Delta have something to show)
The repo ships a ready-made seeder, `steeldefect_platform/seed_obt.py`. It creates the
tables and writes structured synthetic data across several Delta operations
(`WRITE → WRITE → MERGE → OPTIMIZE`), so the history screenshot is rich, and
(optionally) logs an MLflow A/B with two registered model versions.

**Easiest — run the bundle's seed job** (after step 5's deploy). It runs
`seed_notebook.py`, a self-contained version of this seeder without the MLflow step:
```bash
databricks bundle run steeldefect_seed -t dev
```

**Or in a workspace notebook** (handy if you want to watch the prints):
```python
import sys; sys.path.insert(0, "/Workspace/Repos/<you>/steel-defect-ml-platform")  # repo root
from steeldefect_platform.seed_obt import seed
summary = seed(spark, n_coils=40, with_mlflow=True)
summary   # -> {'rows': ..., 'partitions': ..., 'history_versions': ..., 'mlflow_ab': True}
```
On the serverless workspace used for this project, importing workspace files this way
failed with the 409 error described in the caveats; the bundle's seed job avoids it.
The embeddings cluster by expert group, morphology drives a latent severity (so
the regressor has signal), and `local` defects get their severity via a real
`MERGE`. The honesty flag is automatic: every seeded row carries
`source_version = "synthetic-cal-v1"`.

## 5. Deploy the bundle
```bash
databricks bundle deploy -t dev
```
`dev` mode auto-pauses schedules and prefixes everything with your username, so
nothing runs unattended. Open **Workflows** in the UI — the six jobs appear.

## 6. Run the daily pipeline → screenshot #1
```bash
databricks bundle run steeldefect_obt_daily -t dev
```
Open the run in **Workflows → steeldefect-obt-daily → the run** and screenshot the
**task graph** (ingest → severity_score → cutting_plan, + drift_monitor,
optimize_obt). → **figure 1**.

## 7. MLflow → screenshot #2
Run `databricks bundle run steeldefect_ab -t dev`. Its `nb_ab.py` registers two
segmenter versions and sets the `champion` / `challenger` aliases; it logs no accuracy
and runs no A/B test. In **Models**, screenshot the registered versions and their
aliases. → **figure 2**.

## 8. Delta history → screenshot #3
In a SQL cell or notebook:
```sql
DESCRIBE HISTORY steeldefect.obt_segmentations;
```
Screenshot the operations column (`WRITE`, `MERGE`, `OPTIMIZE`). This shows that the
severity `MERGE` and the Z-ORDER maintenance pass are real; the stand-in ingest is a
plain append, so it does not demonstrate idempotent ingestion. → **figure 3**.

## 9. Keep the screenshots
Keep the three PNGs with the deployment notes, captioned as a deployment on
calibrated-synthetic data (Jobs graph, MLflow registry, Delta history), not as
production results.

---

## Caveats (keep it honest)
- **Data is synthetic.** The screenshots prove the platform *runs and is
  deployable*, not any production metric. Say so wherever the screenshots are shown.
- **The seed, A/B and daily OBT jobs run stand-in notebooks.** On the serverless
  workspace used, reading workspace `.py` files failed with
  `409 BlobCustomerSpecifiedEncryptionMismatch`, so the 7 tasks of these jobs run
  simplified, self-contained notebooks (`seed_notebook.py`, `nb_*.py`): for example a
  least-squares severity model with a fixed band instead of XGBoost + CQR, an append
  of a fixed synthetic batch instead of a `MERGE` at ingest, and a registry step with
  no A/B test. The AL, regroup and SSL jobs call the tested package through
  `steeldefect_platform/run_task.py`, which also imports workspace `.py` files, so
  they are exposed to the same error on that workspace.
- **`segmenter_train` / `online_regroup`** are intentionally not in the bundle
  (external GPU training; continuous streaming) — documented in `databricks.yml`.
- **State files**: `.gitignore` already excludes `.databricks/` and `*.tfstate*`, the
  local deploy state the bundle writes.

---

## Operations (observe · respond · recover)

The data-quality gate, dead-letter table, dispatcher traceback and idempotent `MERGE`
below belong to the tested package (`steeldefect_platform/ingestion.py`,
`run_task.py`). The bundle's stand-in daily job has none of them: its `nb_ingest.py`
appends a fixed synthetic batch, so re-running it duplicates rows.

### SLOs & what to monitor
| Signal | Target / SLO | Where |
|---|---|---|
| OBT freshness | rows landed within 1 h of the edge push | `DESCRIBE HISTORY steeldefect.obt_segmentations` (last `MERGE`) |
| Data quality | dead-letter rate < **2 %** per batch (ingestion gate) | `steeldefect.obt_segmentations_deadletter`; the gate raises above 2 % |
| Drift | MMD `p < 0.05` ⇒ retrain trigger | `drift_monitor` task output; `email_notifications` |
| Job success | scheduled jobs green; ≤ 2 auto-retries | Workflows run history (per-task `max_retries`) |
| Model serving | `@champion` alias resolves for every served model | MLflow Models page |

Every job has `email_notifications.on_failure`; the critical `ingest` / `severity_score`
tasks auto-retry (`max_retries: 2`). The ingestion **data-quality gate** fails loudly
(rejects → dead-letter table) rather than silently dropping rows.

### Incident response
| Symptom | First check | Action |
|---|---|---|
| Job task failed | run logs (the dispatcher logs a full traceback) | rely on auto-retry; if persistent, fix + re-run the single task |
| DQ gate raised (`data-quality gate failed`) | `steeldefect.obt_segmentations_deadletter` (`_dq_reason`) | fix the upstream landing schema/feed; re-ingest (MERGE is idempotent) |
| Drift alarm | the `mmd2` / `p_value` verdict | trigger `segmenter_train` (external) → A/B gate; or `defect_regroup_train` |
| Streaming stalled | the checkpoint dir + stream status | restart the stream; Delta checkpoint resumes exactly-once |

### Rollback
- **Model**: re-point the alias to the previous version —
  `MlflowClient().set_registered_model_alias("main.steeldefect.steeldefect_segmenter", "champion", <prev>)`
  (in the package, the A/B gate's `keep_A` is the *forward* safeguard; this is the
  *backward* one).
- **Data**: Delta time-travel — `RESTORE TABLE steeldefect.obt_segmentations TO VERSION AS OF <n>`,
  possible while that version's data files survive `VACUUM` (7 days by default; the
  log keeps 30 days).
- **Deployment**: redeploy the previous bundle revision —
  `databricks bundle deploy -t prod --var="service_principal=<application-id>"` from the
  prior commit; the jobs are declarative (serverless, no clusters), so the rollback is
  the diff.

```

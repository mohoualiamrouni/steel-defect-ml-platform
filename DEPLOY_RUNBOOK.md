# SteelDefect platform — deploy & capture proof (runbook)

Goal: turn the *asserted* "Azure Databricks platform" into **three screenshots**
a jury can't argue with, by actually deploying `databricks.yml` and running it on
a real workspace (synthetic data — that's fine; this proves the **architecture**,
not production numbers).

> **No workspace handy?** `python -m steeldefect_platform.demo_industrialization`
> runs the *operational* guarantees locally in ~70 s (no cluster, no GPU):
> the real data-quality gate (it raises loudly on a dirty batch), drift
> MMD/ADWIN power with confidence intervals, and a real MLflow alias
> promote→rollback round-trip. It prints a metrics table and writes
> `steeldefect_platform/outputs/industrialization_metrics.json`. The Delta
> `MERGE`/streaming/`OPTIMIZE` parts below still need a real workspace.

## What you'll end up with (paste these into the report appendix)
1. **Jobs run-graph** — the `steeldefect_obt_daily` DAG (`ingest → severity_score →
   cutting_plan`, plus `drift_monitor` and `optimize_obt`) showing green tasks.
2. **MLflow** — the experiment with the A/B comparison and the registered model
   versions (incumbent A vs challenger B).
3. **Delta history** — `DESCRIBE HISTORY steeldefect.obt_segmentations` showing the
   `STREAMING UPDATE` / `MERGE` / `OPTIMIZE` operations (time travel is real).

---

## 0. Prerequisites (local machine)
- Python 3.10+ and this repo.
- The **Databricks CLI v0.2x** (the new one, not the legacy `pip install databricks-cli`):
  - Windows: `winget install Databricks.DatabricksCLI` — then `databricks version`.

## 1. Get a free workspace

**Path A — Azure free trial (recommended, most faithful to the report).**
Azure gives a 30-day / credit trial. Create an **Azure Databricks** resource
(Premium tier needed for Unity Catalog/MLflow Registry), open the workspace.
The `new_cluster` job clusters in `databricks.yml` work as-is here.

**Path B — Databricks Free Edition (zero cost, fastest).**
Sign up at the Databricks Free Edition. Caveat: it is **serverless-only**, so the
`job_clusters:` (`new_cluster`) blocks aren't allowed — see the one-line switch
in the Caveats section. Everything else (Jobs, Unity Catalog, MLflow) works.

## 2. Authenticate the CLI
```bash
databricks auth login --host https://adb-XXXX.azuredatabricks.net
# opens a browser (OAuth). Confirm:
databricks current-user me
```

## 3. Point the bundle at your workspace
Edit the two `workspace.host:` lines in `databricks.yml` (the `dev` and `prod`
targets) to your workspace URL. Then validate:
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

**Easiest — run it as a bundle job** (after step 5's deploy):
```bash
databricks bundle run steeldefect_seed -t dev
```

**Or in a workspace notebook** (handy if you want to watch the prints):
```python
import sys; sys.path.insert(0, "/Workspace/Repos/<you>/the plant")  # repo root
from steeldefect_platform.seed_obt import seed
summary = seed(spark, n_coils=40, with_mlflow=True)
summary   # -> {'rows': ..., 'partitions': ..., 'history_versions': ..., 'mlflow_ab': True}
```
The embeddings cluster by expert group, morphology drives a latent severity (so
the regressor has signal), and `local` defects get their severity via a real
`MERGE`. The honesty flag is automatic: every seeded row carries
`source_version = "synthetic-cal-v1"`.

## 5. Deploy the bundle
```bash
databricks bundle deploy -t dev
```
`dev` mode auto-pauses schedules and prefixes everything with your username, so
nothing runs unattended. Open **Workflows** in the UI — the three jobs appear.

## 6. Run the daily pipeline → screenshot #1
```bash
databricks bundle run steeldefect_obt_daily -t dev
```
Open the run in **Workflows → steeldefect-obt-daily → the run** and screenshot the
**task graph** (ingest → severity_score → cutting_plan, + drift_monitor,
optimize_obt). → **figure 1**.

## 7. MLflow → screenshot #2
Run the A/B step with two model versions (after you've logged an incumbent and a
challenger), or open the experiment created by `ab_decision`. In **Experiments**,
open the A/B run and the **Models** registry page (versions + stage). Screenshot
the comparison + the registered versions. → **figure 2**.

## 8. Delta history → screenshot #3
In a SQL cell or notebook:
```sql
DESCRIBE HISTORY steeldefect.obt_segmentations;
```
Screenshot the operations column (`STREAMING UPDATE`, `MERGE`, `OPTIMIZE`). This
demonstrates idempotent writes + the Z-ORDER maintenance pass are real. → **figure 3**.

## 9. Put them in the report
Save the three PNGs into `tn-report/template_tex/figures/` (e.g.
`db_jobs_graph.png`, `db_mlflow_ab.png`, `db_delta_history.png`) and add a figure
block in the platform appendix next to Table `tab:steeldefect:evidence`, captioned
"Plateforme déployée sur Azure Databricks (données synthétiques calibrées) :
graphe de Jobs, registre MLflow, et historique Delta." That single figure turns
the evidence table from *traceable* into *demonstrated*.

---

## Caveats (keep it honest)
- **Data is synthetic.** The screenshots prove the platform *runs and is
  deployable*, not any production metric. Keep the §"Limites" framing.
- **Free Edition is serverless-only.** If you use Path B, remove the
  `job_clusters:` blocks and the per-task `job_cluster_key: steeldefect` lines, and add
  at job level:
  ```yaml
      # serverless: no job_clusters; tasks run on serverless compute
  ```
  (On Azure Path A, leave the bundle as written.)
- **`segmenter_train` / `online_regroup`** are intentionally not in the bundle
  (external GPU training; continuous streaming) — documented in `databricks.yml`.
- **State files**: add `.databricks/` and `*.tfstate*` to `.gitignore` — the
  bundle writes local deploy state you don't want to commit.

---

## Operations (observe · respond · recover)

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
  `MlflowClient().set_registered_model_alias("steeldefect_segmenter", "champion", <prev>)`
  (the A/B gate's `keep_A` is the *forward* safeguard; this is the *backward* one).
- **Data**: Delta time-travel — `RESTORE TABLE steeldefect.obt_segmentations TO VERSION AS OF <n>`
  (history is retained 30 days).
- **Deployment**: redeploy the previous bundle revision — `databricks bundle deploy -t prod`
  from the prior commit; jobs/clusters are declarative, so the rollback is the diff.

```

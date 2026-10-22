# Databricks notebook source
# MAGIC %md
# MAGIC # SteelDefect platform — live demo (Free Edition / serverless)
# MAGIC
# MAGIC Run-all produces, on a real Databricks workspace:
# MAGIC 1. the OBT + companion Delta tables, seeded with calibrated-synthetic data;
# MAGIC 2. a real **Delta history** (WRITE / MERGE / OPTIMIZE)  -> screenshot;
# MAGIC 3. live **H3 numerical metrics** (grade fractions, yield) from the real
# MAGIC    guillotine cutting optimiser  -> screenshot (better than a pytest dump);
# MAGIC 4. an **MLflow A/B** run with accuracies measured on the synthetic data  -> screenshot.
# MAGIC
# MAGIC Data is synthetic (`source_version = synthetic-cal-v1`); this proves the
# MAGIC architecture runs, not production numbers.

# COMMAND ----------
# MAGIC %md ### 0. Point Python at the synced repo, create the schema
# MAGIC The repository root comes from `STEELDEFECT_WORKSPACE_ROOT` when it is set,
# MAGIC otherwise it is derived from this notebook's own location.

# COMMAND ----------
import os, sys, shutil

# Serverless cannot reliably `import` .py directly from /Workspace (the FUSE mount
# doesn't support the mmap the import machinery uses -> [Errno 5]). Fix: copy the
# packages to local disk once, then import from there.
DST = "/tmp/steeldefect"


def _repo_root_in_workspace(local_copy: str) -> str:
    """Where this repository lives in the workspace.

    STEELDEFECT_WORKSPACE_ROOT wins when set (ignoring the local copy this cell
    points it at), then the root derived from this notebook's own path
    (<root>/steeldefect_platform/notebooks/<name>), then the Repos convention.
    """
    env_root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT")
    if env_root and env_root != local_copy:
        return env_root
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
        root = ctx.notebookPath().get().rsplit("/", 3)[0]
        return root if root.startswith("/Workspace") else "/Workspace" + root
    except Exception:
        return "/Workspace/Repos/steel-defect-ml-platform"


SRC = _repo_root_in_workspace(DST)
for pkg in ("steeldefect_platform", "datamining"):
    src = f"{SRC}/{pkg}"
    if os.path.isdir(src):
        shutil.copytree(src, f"{DST}/{pkg}", dirs_exist_ok=True)
if DST not in sys.path:
    sys.path.insert(0, DST)
os.environ["STEELDEFECT_WORKSPACE_ROOT"] = DST     # jobs._load_script reads from here

spark.sql("CREATE SCHEMA IF NOT EXISTS steeldefect")
print("copied to", DST,
      "| steeldefect_platform:", os.path.isdir(DST + "/steeldefect_platform"),
      "| datamining:", os.path.isdir(DST + "/datamining"))

# COMMAND ----------
# MAGIC %md ### 1. Seed the OBT (WRITE x2 -> MERGE severity -> OPTIMIZE/ZORDER)

# COMMAND ----------
from steeldefect_platform.seed_obt import seed

summary = seed(spark, n_coils=40, with_mlflow=False)   # MLflow done separately below
summary

# COMMAND ----------
# MAGIC %md ### 2. Delta history  → **screenshot this** (idempotent MERGE + OPTIMIZE are real)

# COMMAND ----------
display(spark.sql("DESCRIBE HISTORY steeldefect.obt_segmentations"))

# COMMAND ----------
# MAGIC %md A quick look at the seeded substrate (one shared table, two defect kinds):

# COMMAND ----------
display(spark.sql("""
  SELECT defect_kind, count(*) AS n,
         round(avg(severity), 3) AS avg_severity,
         count(DISTINCT expert_group_id) AS expert_groups,
         count(DISTINCT date_key) AS days
  FROM steeldefect.obt_segmentations
  GROUP BY defect_kind
"""))

# COMMAND ----------
# MAGIC %md ### 3. Run the real cutting optimiser → live **H3 metrics**  → **screenshot this**
# MAGIC `cutting_plan` is self-contained: it reads the severity the MERGE wrote and
# MAGIC produces area-weighted grade fractions + yield per coil.

# COMMAND ----------
try:
    from steeldefect_platform.jobs import job_cutting
    job_cutting(since="2026-01-01")          # writes steeldefect.cutting_plans
    print("cutting plans written")
except Exception as exc:
    print("[demo] cutting_plan step failed (see trace):", exc)
    raise

# COMMAND ----------
# Per-coil plan with the H3 evidence columns:
display(spark.sql("""
  SELECT coil_id, n_pieces,
         round(grade_A_frac, 3) AS grade_A,
         round(grade_B_frac, 3) AS grade_B,
         round(scrap_frac, 4)   AS scrap,
         round(yield_frac, 3)   AS yield_frac
  FROM steeldefect.cutting_plans
  ORDER BY coil_id
  LIMIT 20
"""))

# COMMAND ----------
# MAGIC %md **The H3 headline numbers, computed live** (the kind of table worth putting in the report):

# COMMAND ----------
display(spark.sql("""
  SELECT count(*)                       AS coils,
         round(avg(grade_A_frac), 3)    AS mean_grade_A,
         round(avg(scrap_frac), 4)      AS mean_scrap,
         round(max(scrap_frac), 4)      AS worst_coil_scrap,
         round(avg(yield_frac), 3)      AS mean_yield
  FROM steeldefect.cutting_plans
"""))

# COMMAND ----------
# MAGIC %md ### 4. MLflow A/B with measured accuracies (synthetic data)  → **screenshot the Experiment**
# MAGIC Trains two segmenter heads on the seeded embeddings (A on half the labels,
# MAGIC B on all) and logs the comparison. No model-registry write, so it works on
# MAGIC Free Edition's Unity-Catalog registry without 3-level naming.

# COMMAND ----------
try:
    import mlflow, numpy as np
    from sklearn.linear_model import LogisticRegression

    pdf = (spark.table("steeldefect.obt_segmentations")
           .select("cnn_vector", "expert_group_id").toPandas())
    X = np.array(pdf["cnn_vector"].tolist(), dtype="float32")
    y = pdf["expert_group_id"].values
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); tr, te = idx[:int(.7*len(y))], idx[int(.7*len(y)):]
    half = tr[:len(tr)//2]

    mlflow.set_experiment(f"{REPO.rsplit('/the plant',1)[0]}/steeldefect_ab_demo")
    acc = {}
    for tag, sub in (("A", half), ("B", tr)):
        with mlflow.start_run(run_name=f"segmenter_{tag}"):
            clf = LogisticRegression(max_iter=200).fit(X[sub], y[sub])
            a = float((clf.predict(X[te]) == y[te]).mean())
            acc[tag] = a
            mlflow.log_param("n_train", int(len(sub)))
            mlflow.log_metric("val_accuracy", a)
    with mlflow.start_run(run_name="ab_decision"):
        delta = acc["B"] - acc["A"]
        mlflow.log_metric("acc_A", acc["A"]); mlflow.log_metric("acc_B", acc["B"])
        mlflow.log_metric("delta_acc", delta)
        mlflow.set_tag("decision", "promote_B" if delta > 0.02 else "keep_A")
    print("A/B logged:", acc)
except Exception as exc:
    print("[demo] MLflow step skipped:", exc)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Screenshots to capture
# MAGIC - **#3 Delta history** (cell 2) — `WRITE`, `MERGE`, `OPTIMIZE` operations.
# MAGIC - **Numerical H3 metrics** (cells 3) — the cutting-plan grade/scrap/yield tables.
# MAGIC - **MLflow A/B** — open **Experiments → steeldefect_ab_demo**.
# MAGIC - **Catalog** — open **Catalog → steeldefect → obt_segmentations** (schema + sample).
# MAGIC - (Optional **Jobs graph**: create a Job from `steeldefect_platform/run_task.py`,
# MAGIC   or deploy the serverless bundle — see the chat for the one-line edit.)

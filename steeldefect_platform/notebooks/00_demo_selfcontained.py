# Databricks notebook source
# MAGIC %md
# MAGIC # SteelDefect platform — self-contained demo (Free Edition / serverless)
# MAGIC
# MAGIC No external imports (Free Edition serverless can't read `/Workspace` .py).
# MAGIC Everything needed is inline; only `spark` + `numpy`/`pandas` (preinstalled).
# MAGIC Produces: the OBT Delta table, a real `WRITE→MERGE→OPTIMIZE` history,
# MAGIC live grade/scrap metrics, and an MLflow A/B. Data is synthetic
# MAGIC (`source_version = synthetic-cal-v1`) — proves architecture, not numbers.

# COMMAND ----------
# MAGIC %pip install -q mlflow scikit-learn

# COMMAND ----------
import numpy as np, pandas as pd, datetime as dt
from pyspark.sql.types import (ArrayType, BooleanType, DateType, DoubleType,
                               FloatType, IntegerType, StringType, StructField,
                               StructType, TimestampType)

spark.sql("CREATE SCHEMA IF NOT EXISTS steeldefect")
spark.sql("DROP TABLE IF EXISTS steeldefect.obt_segmentations")   # fresh history for the demo
print("schema ready")

# COMMAND ----------
# MAGIC %md ## 1. Seed the OBT  (WRITE x2 → MERGE severity → OPTIMIZE/ZORDER)

# COMMAND ----------
rng = np.random.default_rng(7)
N_G, CNN = 28, 960
GROUPS = ["inclusion","sliver","lamination","shell","scale","scratch","roll_mark",
          "edge_crack","coil_break","weld_defect","buckle","camber","corrugation",
          "pinhole","blister","seam","gouge","dent","stain","oil_spot","rust","burr",
          "wave","chatter","hole","crack","pit","contamination"]
cent = rng.normal(0, 1.0, (N_G, CNN)).astype("float32")   # per-group embedding centroids
base = dt.date(2026, 5, 4)

rows, sev_updates = [], []
for c in range(40):
    coil, line = f"C{c:04d}", ("L1", "L2")[c % 2]
    day = base + dt.timedelta(days=int(rng.integers(0, 12)))
    L, W = float(rng.uniform(500, 1500)), float(rng.uniform(1.0, 2.0))
    for d in range(int(rng.integers(25, 70))):
        g = int(rng.integers(0, N_G))
        kind = "local" if rng.random() < 0.7 else "group"
        emb = (cent[g] + rng.normal(0, 0.35, CNN)).astype("float32")
        area = float(abs(rng.normal(10 + g * 2, 5)))
        sev = float(1 / (1 + np.exp(-((area / 40) + g * 0.04 - 1.2 + rng.normal(0, 0.3)))))
        sid = f"{coil}-{d:03d}"
        rows.append({
            "segmentation_id": sid, "coil_id": coil, "line_id": line,
            "event_ts": dt.datetime.combine(day, dt.time(int(rng.integers(0, 24)))),
            "date_key": day, "defect_kind": kind, "expert_group": GROUPS[g],
            "expert_group_id": g, "cnn_dim": CNN, "cnn_vector": emb.tolist(),
            "area": area, "eccentricity": float(rng.uniform(0, 1)),
            "solidity": float(rng.uniform(0.5, 1)), "glcm_contrast": float(rng.uniform(0, 1)),
            "pos_x_m": float(rng.uniform(0, L)) if kind == "local" else None,
            "pos_y_m": float(rng.uniform(0, W)) if kind == "local" else None,
            "severity": None, "severity_lo": None, "severity_hi": None,   # filled by MERGE
            "density_per_m2": None if kind == "local" else float(rng.uniform(0.1, 8)),
            "coverage_frac": None if kind == "local" else float(rng.uniform(0.01, 0.4)),
            "coil_length_m": L, "coil_width_m": W,
            "is_labeled": bool(rng.random() < 0.15),
            "source_version": "synthetic-cal-v1",
        })
        if kind == "local":
            m = 0.12
            sev_updates.append((sid, sev, max(0.0, sev - m), min(1.0, sev + m)))

SCHEMA = StructType([
    StructField("segmentation_id", StringType()), StructField("coil_id", StringType()),
    StructField("line_id", StringType()), StructField("event_ts", TimestampType()),
    StructField("date_key", DateType()), StructField("defect_kind", StringType()),
    StructField("expert_group", StringType()), StructField("expert_group_id", IntegerType()),
    StructField("cnn_dim", IntegerType()), StructField("cnn_vector", ArrayType(FloatType())),
    StructField("area", DoubleType()), StructField("eccentricity", DoubleType()),
    StructField("solidity", DoubleType()), StructField("glcm_contrast", DoubleType()),
    StructField("pos_x_m", DoubleType()), StructField("pos_y_m", DoubleType()),
    StructField("severity", FloatType()), StructField("severity_lo", FloatType()),
    StructField("severity_hi", FloatType()), StructField("density_per_m2", DoubleType()),
    StructField("coverage_frac", DoubleType()), StructField("coil_length_m", DoubleType()),
    StructField("coil_width_m", DoubleType()), StructField("is_labeled", BooleanType()),
    StructField("source_version", StringType()),
])

sdf = spark.createDataFrame(rows, schema=SCHEMA)
cut = sdf.count() // 2
ids = [r["segmentation_id"] for r in rows]
# WRITE #1 (creates the Delta table, partitioned)
(sdf.where(sdf.segmentation_id.isin(ids[:cut]))
    .write.format("delta").partitionBy("date_key", "line_id")
    .mode("overwrite").saveAsTable("steeldefect.obt_segmentations"))
# WRITE #2 (a later micro-batch — a second version in the history)
(sdf.where(~sdf.segmentation_id.isin(ids[:cut]))
    .write.format("delta").mode("append").saveAsTable("steeldefect.obt_segmentations"))
print("seeded rows:", spark.table("steeldefect.obt_segmentations").count())

# COMMAND ----------
# MERGE the predicted severity (+ conformal band) into the LOCAL defects
upd = spark.createDataFrame(sev_updates,
        "segmentation_id string, severity float, severity_lo float, severity_hi float")
upd.createOrReplaceTempView("sev_updates")
spark.sql("""
  MERGE INTO steeldefect.obt_segmentations t USING sev_updates s
  ON t.segmentation_id = s.segmentation_id
  WHEN MATCHED THEN UPDATE SET t.severity = s.severity,
       t.severity_lo = s.severity_lo, t.severity_hi = s.severity_hi
""")
# OPTIMIZE + Z-ORDER maintenance pass (the real periodic op)
spark.sql("OPTIMIZE steeldefect.obt_segmentations ZORDER BY (coil_id, expert_group_id, defect_kind)")
print("merged severity + optimized")

# COMMAND ----------
# MAGIC %md ## 2. Delta history  → **screenshot this** (WRITE / MERGE / OPTIMIZE are real)

# COMMAND ----------
display(spark.sql("DESCRIBE HISTORY steeldefect.obt_segmentations"))

# COMMAND ----------
# MAGIC %md ## 3. Live metrics  → **screenshot this**

# COMMAND ----------
display(spark.sql("""
  SELECT defect_kind, count(*) AS n, round(avg(severity),3) AS avg_severity,
         count(DISTINCT expert_group_id) AS expert_groups,
         count(DISTINCT date_key) AS days
  FROM steeldefect.obt_segmentations GROUP BY defect_kind
"""))

# COMMAND ----------
# Severity-based grade distribution per coil (the H3-style evidence)
display(spark.sql("""
  WITH graded AS (
    SELECT coil_id,
      CASE WHEN severity < 0.33 THEN 'A' WHEN severity < 0.66 THEN 'B'
           WHEN severity < 0.90 THEN 'C' ELSE 'scrap' END AS grade
    FROM steeldefect.obt_segmentations WHERE defect_kind = 'local')
  SELECT round(avg(CASE WHEN grade='A'    THEN 1 ELSE 0 END), 3) AS grade_A_frac,
         round(avg(CASE WHEN grade='B'    THEN 1 ELSE 0 END), 3) AS grade_B_frac,
         round(avg(CASE WHEN grade='scrap'THEN 1 ELSE 0 END), 4) AS scrap_frac,
         count(DISTINCT coil_id) AS coils
  FROM graded
"""))

# COMMAND ----------
# MAGIC %md ## 4. MLflow A/B  → **screenshot Experiments → steeldefect_ab_demo**

# COMMAND ----------
try:
    import mlflow
    from sklearn.linear_model import LogisticRegression
    pdf = spark.table("steeldefect.obt_segmentations").select("cnn_vector","expert_group_id").toPandas()
    X = np.array(pdf["cnn_vector"].tolist(), dtype="float32"); y = pdf["expert_group_id"].values
    r = np.random.default_rng(0).permutation(len(y)); tr, te = r[:int(.7*len(y))], r[int(.7*len(y)):]
    acc = {}
    mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').collect()[0][0]}/steeldefect_ab_demo")
    for tag, sub in (("A", tr[:len(tr)//2]), ("B", tr)):
        with mlflow.start_run(run_name=f"segmenter_{tag}"):
            clf = LogisticRegression(max_iter=200).fit(X[sub], y[sub])
            a = float((clf.predict(X[te]) == y[te]).mean()); acc[tag] = a
            mlflow.log_param("n_train", int(len(sub))); mlflow.log_metric("val_accuracy", a)
    with mlflow.start_run(run_name="ab_decision"):
        delta = acc["B"] - acc["A"]
        mlflow.log_metric("acc_A", acc["A"]); mlflow.log_metric("acc_B", acc["B"])
        mlflow.log_metric("delta_acc", delta)
        mlflow.set_tag("decision", "promote_B" if delta > 0.02 else "keep_A")
    print("A/B logged:", acc)
except Exception as e:
    print("MLflow step skipped:", e)

# COMMAND ----------
# MAGIC %md
# MAGIC **Screenshots:** cell 2 (Delta history) · cell 3 (metrics) · Experiments→steeldefect_ab_demo
# MAGIC · Catalog→steeldefect→obt_segmentations.

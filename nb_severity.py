# Databricks notebook source
# MAGIC %md # Task: severity_score — fit morphology->severity, MERGE predictions back
# MAGIC Mirrors the P3 datamining step (here a numpy least-squares stand-in for the
# MAGIC XGBoost+CQR regressor) and writes via an idempotent Delta MERGE.

# COMMAND ----------
# MAGIC %run ./nb_common

# COMMAND ----------

from delta.tables import DeltaTable

# train on local defects that already have a severity label
train = (spark.table(TABLE).where("defect_kind='local' AND severity IS NOT NULL")
         .select("severity", *MORPHO).toPandas())
Xtr = np.c_[np.ones(len(train)), train[MORPHO].fillna(0.0).values]
beta, *_ = np.linalg.lstsq(Xtr, train["severity"].values.astype(float), rcond=None)

# predict for ALL local defects (incl. the freshly-ingested, severity-NULL ones)
alld = spark.table(TABLE).where("defect_kind='local'").select("segmentation_id", *MORPHO).toPandas()
pred = np.clip(np.c_[np.ones(len(alld)), alld[MORPHO].fillna(0.0).values] @ beta, 0, 1)
m = 0.12  # conformal-style band
recs = list(zip(alld["segmentation_id"].tolist(), pred.astype(float).tolist(),
                np.clip(pred - m, 0, 1).astype(float).tolist(),
                np.clip(pred + m, 0, 1).astype(float).tolist()))
upd = spark.createDataFrame(
    recs, "segmentation_id string, severity double, severity_lo double, severity_hi double")

(DeltaTable.forName(spark, TABLE).alias("t")
    .merge(upd.alias("s"), "t.segmentation_id = s.segmentation_id")
    .whenMatchedUpdate(set={"severity": "s.severity",
                            "severity_lo": "s.severity_lo",
                            "severity_hi": "s.severity_hi"})
    .execute())
print("severity MERGE upserted", len(recs), "local defects")

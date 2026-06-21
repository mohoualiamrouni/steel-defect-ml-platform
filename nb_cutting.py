# Databricks notebook source
# MAGIC %md # Task: cutting_plan — per-coil yield/scrap summary -> steeldefect.cutting_plans
# MAGIC Mirrors the P3 cutting step: aggregates the (now scored) severities per coil
# MAGIC into a plan table the downstream consumers read.

# COMMAND ----------
# MAGIC %run ./nb_common

# COMMAND ----------

plan = (spark.table(TABLE).where("defect_kind='local'")
        .groupBy("coil_id", "line_id")
        .agg(F.count("*").alias("n_defects"),
             F.avg("severity").alias("mean_sev"),
             F.expr("percentile_approx(severity, 0.9)").alias("p90_sev"),
             F.first("coil_length_m").alias("coil_length_m")))
plan = (plan
        .withColumn("scrap_frac", F.least(F.lit(1.0), F.coalesce("mean_sev", F.lit(0.0))))
        .withColumn("yield_frac", F.lit(1.0) - F.col("scrap_frac"))
        .withColumn("created_ts", F.current_timestamp()))
(plan.write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable("steeldefect.cutting_plans"))
print("cutting_plans rows", spark.table("steeldefect.cutting_plans").count())

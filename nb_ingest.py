# Databricks notebook source
# MAGIC %md # Task: ingest — append a fresh daily batch to the OBT (severity NULL)

# COMMAND ----------
# MAGIC %run ./nb_common

# COMMAND ----------

rows = gen_rows(n_coils=10, seed=99, day0=dt.date(2026, 5, 16))
to_df(rows).write.format("delta").mode("append").saveAsTable(TABLE)
print("ingested", len(rows), "rows; OBT total now", spark.table(TABLE).count())

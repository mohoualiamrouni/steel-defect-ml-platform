# Databricks notebook source
# MAGIC %md # Task: optimize_obt — file compaction + Z-ORDER (data-skipping layout)

# COMMAND ----------

TABLE = "steeldefect.obt_segmentations"
spark.sql(f"OPTIMIZE {TABLE} ZORDER BY (coil_id, expert_group_id, defect_kind)")
print("OPTIMIZE ZORDER complete on", TABLE)

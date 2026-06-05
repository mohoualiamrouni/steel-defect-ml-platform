# Databricks notebook source
# MAGIC %md
# MAGIC # SteelDefect A/B → MLflow (standalone, no Spark)
# MAGIC
# MAGIC Logs the segmenter A/B comparison to MLflow with **zero Spark** (so the
# MAGIC mlflow install can't conflict with PySpark's protobuf). Run all, then open
# MAGIC **Experiments → steeldefect_ab_demo**. Synthetic embeddings, same structure as the OBT.

# COMMAND ----------
# MAGIC %pip install -q mlflow-skinny scikit-learn

# COMMAND ----------
dbutils.library.restartPython()   # load the freshly-installed packages cleanly

# COMMAND ----------
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression

# synthetic embeddings that cluster by expert group (same idea as the real OBT)
rng = np.random.default_rng(7)
N_G, CNN, N = 28, 256, 2000
cent = rng.normal(0, 1.0, (N_G, CNN))
y = rng.integers(0, N_G, N)
X = np.stack([cent[g] + rng.normal(0, 0.35, CNN) for g in y])

r = rng.permutation(N)
tr, te = r[: int(0.7 * N)], r[int(0.7 * N):]

acc = {}
mlflow.set_experiment("/Users/user@example.com/steeldefect_ab_demo")
# incumbent A trained on half the labels; challenger B on all of them
for tag, sub in (("A", tr[: len(tr) // 2]), ("B", tr)):
    with mlflow.start_run(run_name=f"segmenter_{tag}"):
        clf = LogisticRegression(max_iter=200).fit(X[sub], y[sub])
        a = float((clf.predict(X[te]) == y[te]).mean())
        acc[tag] = a
        mlflow.log_param("n_train", int(len(sub)))
        mlflow.log_metric("val_accuracy", a)

with mlflow.start_run(run_name="ab_decision"):
    delta = acc["B"] - acc["A"]
    mlflow.log_metric("acc_A", acc["A"])
    mlflow.log_metric("acc_B", acc["B"])
    mlflow.log_metric("delta_acc", delta)
    mlflow.set_tag("decision", "promote_B" if delta > 0.02 else "keep_A")

print("A/B logged:", acc, "-> open Experiments -> steeldefect_ab_demo")

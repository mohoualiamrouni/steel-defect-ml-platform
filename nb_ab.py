# Databricks notebook source
# MAGIC %md
# MAGIC # Segmenter model registry — versioning + champion/challenger (MLOps wiring)
# MAGIC Registers two versions of the inspection segmenter in Unity Catalog and assigns
# MAGIC the production aliases. This screenshot proves the **registry / promotion
# MAGIC infrastructure** on real Azure Databricks — nothing more.
# MAGIC
# MAGIC Deliberately **no accuracy metric**: the models are trained on
# MAGIC calibrated-synthetic embeddings, so any accuracy here is a property of the
# MAGIC data generator, not a result. The promotion *decision* is governed by the
# MAGIC 7-criterion A/B gate (see report `sec:res_h1`), and real model performance
# MAGIC is reported in H1-H4 by the actual pipelines. The only thing logged here is
# MAGIC the factual `n_train` difference between the two versions.

# COMMAND ----------
# MAGIC %pip install scikit-learn
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import json
import numpy as np
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

MODEL = "main.steeldefect.steeldefect_segmenter"
N_GROUPS, CNN_DIM, N = 28, 960, 2500

# synthetic embedding feature space (clearly not production data)
rng = np.random.default_rng(7)
cen = rng.normal(0, 1, (N_GROUPS, CNN_DIM))
y = rng.integers(0, N_GROUPS, N)
X = (cen[y] + rng.normal(0, 0.35, (N, CNN_DIM))).astype("float32")

mlflow.set_registry_uri("databricks-uc")
# incumbent = fewer labels, challenger = full set: the real, factual difference
for role, n_train in (("incumbent", 600), ("challenger", N)):
    with mlflow.start_run(run_name=f"segmenter_{role}"):
        clf = LogisticRegression(max_iter=300).fit(X[:n_train], y[:n_train])
        mlflow.log_param("n_train", int(n_train))
        mlflow.set_tag("role", role)
        mlflow.set_tag("source_version", "synthetic-cal-v1")   # honesty flag
        mlflow.sklearn.log_model(clf, "model", registered_model_name=MODEL,
                                 signature=infer_signature(X[:2], clf.predict(X[:2])))

client = MlflowClient()
vers = sorted(int(v.version) for v in client.search_model_versions(f"name='{MODEL}'"))
client.set_registered_model_alias(MODEL, "champion", vers[0])      # current production
client.set_registered_model_alias(MODEL, "challenger", vers[-1])   # candidate under gate
out = {"versions": vers, "champion": vers[0], "challenger": vers[-1]}
print(out)
dbutils.notebook.exit(json.dumps(out))

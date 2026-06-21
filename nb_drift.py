# Databricks notebook source
# MAGIC %md # Task: drift_monitor — embedding-cohort distribution check
# MAGIC Mirrors the MMD drift monitor: compares two embedding cohorts and reports a
# MAGIC distance with a permutation p-value (subsampled for speed).

# COMMAND ----------
# MAGIC %run ./nb_common

# COMMAND ----------

emb = np.array(spark.table(TABLE).select("cnn_vector").limit(600).toPandas()["cnn_vector"].tolist(),
               dtype=float)
a, b = emb[:len(emb) // 2], emb[len(emb) // 2:]


def mmd2(x, y):
    g = 1.0 / (np.median(np.sum((x[:80, None] - x[None, :80]) ** 2, -1)) + 1e-9)
    k = lambda u, v: np.exp(-g * (np.sum(u * u, 1)[:, None] + np.sum(v * v, 1)[None, :] - 2 * u @ v.T))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


obs = mmd2(a, b)
rng = np.random.default_rng(0)
pooled = np.vstack([a, b]); n = len(a); cnt = 0
for _ in range(100):
    rng.shuffle(pooled)
    if mmd2(pooled[:n], pooled[n:]) >= obs:
        cnt += 1
p = (cnt + 1) / 101
print(f"drift MMD^2={obs:.4f}, p={p:.3f} -> {'DRIFT (retrain)' if p < 0.05 else 'no significant drift'}")

"""Drift monitoring over OBT cohorts.

Two complementary monitors, both already cited in the report bibliography:

* **ADWIN** (Bifet & Gavaldà 2007) — an adaptive sliding window over a stream
  of scalar statistics (e.g. the per-batch mean inspection confidence) that flags a
  change when two sub-windows differ beyond a Hoeffding bound.
* **MMD** (Gretton et al.; "Failing Loudly", Rabanser 2019) — a kernel
  two-sample test on the CNN embedding cohorts (reference campaign vs current)
  that detects distributional dataset shift.

A firing monitor sets a flag that the Jobs DAG turns into a retraining trigger
for ``steeldefect_segmenter`` (which then re-enters the A/B promotion gate) and a
re-clustering trigger for ``steeldefect_defect_clusterer``.

Pure numpy; no Spark import.  ``cohort_features`` is the only Spark-touching
helper and imports lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ADWIN (compact, exact-enough implementation for a scalar score stream)
# ---------------------------------------------------------------------------

@dataclass
class ADWINState:
    """Adaptive windowing change detector over a scalar stream.

    Not a full bucket-exponential ADWIN; a clear O(n) window-split detector
    sufficient for monitoring a per-batch confidence stream.  ``delta`` is the
    confidence parameter (smaller → fewer false alarms).
    """

    delta: float = 0.002
    window: List[float] = field(default_factory=list)
    drift_detected: bool = False
    last_change_index: int = -1
    _t: int = 0

    def update(self, value: float) -> bool:
        """Add a value; return True if a change point is detected this step."""
        self.window.append(float(value))
        self._t += 1
        self.drift_detected = False
        n = len(self.window)
        if n < 8:
            return False
        arr = np.asarray(self.window, dtype=float)
        # try every split; flag if the sub-window means differ beyond the bound
        for cut in range(2, n - 1):
            w0, w1 = arr[:cut], arr[cut:]
            n0, n1 = len(w0), len(w1)
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            var = float(arr.var()) + 1e-12
            eps = math.sqrt(2.0 / m * var * math.log(2.0 / self.delta))
            if abs(w0.mean() - w1.mean()) > eps:
                self.drift_detected = True
                self.last_change_index = self._t
                # drop the stale prefix (adaptive shrink)
                self.window = arr[cut:].tolist()
                return True
        return False


def adwin_monitor(stream_of_scores, delta: float = 0.002) -> ADWINState:
    """Run ADWIN over an iterable of scalar scores; return the final state."""
    state = ADWINState(delta=delta)
    for v in stream_of_scores:
        state.update(v)
    return state


# ---------------------------------------------------------------------------
# MMD (quadratic-time MMD^2: unbiased U-statistic diagonal terms + V-statistic
# cross term, RBF kernel) for embedding cohorts. Validity of the reported
# p-value comes from the permutation test, not from the estimator's bias.
# ---------------------------------------------------------------------------

def _rbf_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Median-heuristic bandwidth for the RBF kernel."""
    z = np.vstack([x, y])
    n = min(len(z), 512)
    idx = np.linspace(0, len(z) - 1, n).astype(int)
    zz = z[idx]
    d2 = np.sum((zz[:, None, :] - zz[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return 1.0 / (med + 1e-12)


def mmd_drift(ref: np.ndarray, cur: np.ndarray,
              n_permutations: int = 200, random_state: int = 42
              ) -> Tuple[float, float]:
    """Kernel MMD² statistic + permutation p-value between two embedding sets.

    Returns ``(mmd2, p_value)``.  A small p-value indicates the current cohort's
    embedding distribution has drifted from the reference.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    gamma = _rbf_gamma(ref, cur)

    def _k(a, b):
        d2 = np.sum(a * a, 1)[:, None] + np.sum(b * b, 1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * np.maximum(d2, 0.0))

    def _mmd2(a, b):
        m, n = len(a), len(b)
        kaa = (_k(a, a).sum() - m) / (m * (m - 1) + 1e-12)
        kbb = (_k(b, b).sum() - n) / (n * (n - 1) + 1e-12)
        kab = _k(a, b).mean()
        return kaa + kbb - 2 * kab

    obs = _mmd2(ref, cur)
    rng = np.random.default_rng(random_state)
    pooled = np.vstack([ref, cur])
    m = len(ref)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if _mmd2(pooled[:m], pooled[m:]) >= obs:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return float(obs), float(p)


def cohort_features(spark, obt_table: str, campaign_id: str,
                    limit: int = 5000) -> np.ndarray:
    """Pull a CNN-embedding cohort for one campaign from the OBT (lazy Spark).

    Filters with the DataFrame API (bound column + literal) rather than an
    f-string ``WHERE`` clause, so a ``campaign_id`` arriving from job params
    cannot be used for SQL injection.
    """
    from pyspark.sql import functions as F

    df = (
        spark.table(obt_table)
        .where((F.col("campaign_id") == campaign_id) & F.col("cnn_vector").isNotNull())
        .select("cnn_vector")
        .limit(limit)
    )
    rows = df.toPandas()["cnn_vector"].tolist()
    from .io_utils import stack_cnn_vectors
    return stack_cnn_vectors(rows)

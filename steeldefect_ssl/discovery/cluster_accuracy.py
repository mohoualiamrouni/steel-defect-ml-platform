"""Hungarian (optimal-assignment) clustering accuracy for category discovery."""

from __future__ import annotations

import numpy as np


def hungarian_cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best label-permutation accuracy via the Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    ut = {c: i for i, c in enumerate(np.unique(y_true))}
    up = {c: i for i, c in enumerate(np.unique(y_pred))}
    n = max(len(ut), len(up))
    w = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[up[p], ut[t]] += 1
    row, col = linear_sum_assignment(-w)
    return float(w[row, col].sum()) / float(len(y_true))

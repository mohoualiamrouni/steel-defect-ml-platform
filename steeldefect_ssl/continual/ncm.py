"""Nearest-Class-Mean classifier (Mensink 2013 / iCaRL, Rebuffi 2017).

Prototypes are class means in the (L2-normalised) SSL feature space. Adding a
newly-discovered family is purely additive — old prototypes are untouched — so
catastrophic forgetting is structurally near-zero. ``forgetting`` quantifies the
only residual effect: a new prototype occasionally "stealing" old-class test
points. Optional herding (``herding_buffer``) selects exemplars approximating a
class mean for a small replay buffer.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


class NCMClassifier:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.protos_: Dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NCMClassifier":
        X = _l2(X) if self.normalize else np.asarray(X)
        self.protos_ = {int(c): X[y == c].mean(axis=0) for c in np.unique(y)}
        return self

    def add_class(self, X_new: np.ndarray, label: int) -> None:
        X_new = _l2(X_new) if self.normalize else np.asarray(X_new)
        self.protos_[int(label)] = X_new.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _l2(X) if self.normalize else np.asarray(X)
        labels = np.array(sorted(self.protos_))
        P = np.stack([self.protos_[int(c)] for c in labels])      # (C, d)
        d = ((X[:, None, :] - P[None, :, :]) ** 2).sum(-1)        # (N, C)
        return labels[d.argmin(axis=1)]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    def forgetting(self, X_old: np.ndarray, y_old: np.ndarray,
                   acc_before: float) -> Dict[str, float]:
        acc_after = self.accuracy(X_old, y_old)
        return {"acc_old_before": float(acc_before),
                "acc_old_after": float(acc_after),
                "forgetting_delta": float(acc_before - acc_after)}


def herding_buffer(X: np.ndarray, m: int) -> np.ndarray:
    """Greedy herding (iCaRL): indices of ``m`` exemplars whose running mean
    best approximates the class mean."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    chosen, running = [], np.zeros_like(mu)
    m = int(min(m, len(X)))
    for t in range(1, m + 1):
        scores = X @ (mu * t - running)
        if chosen:
            scores[chosen] = -np.inf      # without replacement (unique exemplars)
        idx = int(np.argmax(scores))
        chosen.append(idx)
        running += X[idx]
    return np.asarray(chosen, dtype=int)

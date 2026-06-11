"""Open-set novelty scorers on SSL features (higher score = more novel).

SimCLR features live on a hypersphere (cosine / NT-Xent objective), so the
scorers default to **L2-normalised** features (``l2_normalize=True``) — the
geometry the OOD literature (SSD, kNN-OOD) uses for contrastive features —
rather than per-dim z-scoring.

* :class:`MahalanobisScorer` — tied-covariance Gaussian per known class (DDU).
* :class:`KNNScorer` — mean distance to the k nearest known features (cosine).
* :class:`CosineProtoScorer` — 1 - max cosine to the known class prototypes.
* :class:`EnergyScorer` — free energy ``-logsumexp(logits)`` of a linear head.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


class _Prep:
    """Feature pre-processing shared by the distance scorers: L2-normalise
    (hypersphere) or per-dim standardise."""

    def __init__(self, l2_normalize: bool = True):
        self.l2 = bool(l2_normalize)

    def fit(self, X):
        self.scaler_ = None if self.l2 else StandardScaler().fit(X)
        return self

    def __call__(self, X):
        return _l2(X) if self.l2 else self.scaler_.transform(X)


class MahalanobisScorer:
    def __init__(self, shrinkage: float = 1e-3, l2_normalize: bool = True):
        self.shrinkage = float(shrinkage)
        self.prep = _Prep(l2_normalize)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisScorer":
        Xs = self.prep.fit(X)(X)
        classes = np.unique(y)
        d = Xs.shape[1]
        means, pooled = [], np.zeros((d, d), dtype=np.float64)
        for c in classes:
            Xc = Xs[y == c]
            mu = Xc.mean(axis=0)
            means.append(mu)
            diff = Xc - mu
            pooled += diff.T @ diff
        pooled /= max(1, (len(Xs) - len(classes)))
        pooled += self.shrinkage * np.eye(d)
        try:
            self.cov_inv_ = np.linalg.inv(pooled)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(pooled)
        self.means_ = np.asarray(means)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.prep(X)
        out = np.empty(len(Xs), dtype=np.float64)
        for i, x in enumerate(Xs):
            diffs = self.means_ - x
            d = np.einsum("cd,de,ce->c", diffs, self.cov_inv_, diffs)
            out[i] = float(d.min())
        return out


class KNNScorer:
    def __init__(self, k: int = 5, l2_normalize: bool = True):
        self.k = int(k)
        self.prep = _Prep(l2_normalize)

    def fit(self, X: np.ndarray, y=None) -> "KNNScorer":
        from sklearn.neighbors import NearestNeighbors
        Xs = self.prep.fit(X)(X)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(Xs))).fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self.nn_.kneighbors(self.prep(X))
        return dist.mean(axis=1)


class CosineProtoScorer:
    """Novelty = 1 - max cosine similarity to the known class prototypes
    (class means on the unit hypersphere)."""

    def __init__(self, l2_normalize: bool = True):
        self.l2 = True  # prototypes are always on the sphere

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CosineProtoScorer":
        Xn = _l2(X)
        self.protos_ = _l2(np.stack([Xn[y == c].mean(axis=0) for c in np.unique(y)]))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - (_l2(X) @ self.protos_.T).max(axis=1)


class EnergyScorer:
    def __init__(self, random_state: int = 42, l2_normalize: bool = True):
        self.random_state = int(random_state)
        self.prep = _Prep(l2_normalize)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnergyScorer":
        Xs = self.prep.fit(X)(X)
        self.clf_ = LogisticRegression(max_iter=2000, random_state=self.random_state)
        self.clf_.fit(Xs, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp
        logits = self.clf_.decision_function(self.prep(X))
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        return -logsumexp(logits, axis=1)


def build_scorer(name: str, shrinkage: float = 1e-3, knn_k: int = 5,
                 random_state: int = 42, l2_normalize: bool = True):
    if name == "mahalanobis":
        return MahalanobisScorer(shrinkage=shrinkage, l2_normalize=l2_normalize)
    if name == "knn":
        return KNNScorer(k=knn_k, l2_normalize=l2_normalize)
    if name == "cos_proto":
        return CosineProtoScorer(l2_normalize=l2_normalize)
    if name == "energy":
        return EnergyScorer(random_state=random_state, l2_normalize=l2_normalize)
    raise ValueError(f"unknown open-set scorer: {name}")

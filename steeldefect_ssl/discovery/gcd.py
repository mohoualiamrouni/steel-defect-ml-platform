"""Generalized Category Discovery (Vaze et al., 2022) on SSL features.

* :func:`select_k` — choose the number of clusters: reuse P2's ``gap_statistic``
  (Tibshirani) or ``DeepDPMClustering`` (non-parametric, CPU mode), or a fixed K;
  always with a robust silhouette fallback.
* :func:`semi_supervised_kmeans` — seeded k-means whose first centroids are the
  labeled known-class means (Basu 2002 seeding ~ GCD), the rest k-means++ from
  the unlabeled pool.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _silhouette_k(X: np.ndarray, k_range, random_state: int = 42) -> int:
    lo, hi = int(k_range[0]), int(k_range[1])
    best_k, best_s = lo, -1.0
    n = len(X)
    for k in range(lo, hi + 1):
        if k >= n:
            break
        lbl = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit_predict(X)
        if len(np.unique(lbl)) < 2:
            continue
        s = silhouette_score(X, lbl, sample_size=min(2000, n), random_state=random_state)
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def select_k(X: np.ndarray, cfg_discovery, random_state: int = 42) -> int:
    """Return the chosen number of clusters for the GCD pool."""
    mode = cfg_discovery.k_select
    if mode == "fixed" and cfg_discovery.fixed_k:
        return int(cfg_discovery.fixed_k)
    if mode == "gap":
        try:
            from defect_clustering.clustering.metrics import gap_statistic
            res = gap_statistic(X, k_range=tuple(cfg_discovery.k_range),
                                n_references=8, random_state=random_state, n_jobs=1)
            return int(res["optimal_k"])
        except Exception as exc:
            print(f"[gcd] gap_statistic unavailable ({exc}); silhouette fallback")
            return _silhouette_k(X, cfg_discovery.k_range, random_state)
    if mode == "deep_dpm":
        try:
            from defect_clustering.clustering.deep_dpm import DeepDPMClustering
            lo, hi = cfg_discovery.k_range
            dpm = DeepDPMClustering(init_k=int((lo + hi) // 2), use_mlp=False,
                                    random_state=random_state)
            dpm.fit(X)
            return int(getattr(dpm, "n_clusters_", _silhouette_k(X, cfg_discovery.k_range, random_state)))
        except Exception as exc:
            print(f"[gcd] DeepDPM unavailable ({exc}); silhouette fallback")
            return _silhouette_k(X, cfg_discovery.k_range, random_state)
    return _silhouette_k(X, cfg_discovery.k_range, random_state)


def semi_supervised_kmeans(X: np.ndarray, y: np.ndarray, mask_known: np.ndarray,
                           k: int, random_state: int = 42) -> np.ndarray:
    """Seeded k-means: known-class means seed the first centroids; extras via
    k-means++ on the unlabeled rows. ``y`` holds known labels (>=0) where
    ``mask_known`` is True, else ignored."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    known_classes = np.unique(y[mask_known]) if mask_known.any() else np.array([], dtype=int)
    seeds = [X[mask_known & (y == c)].mean(axis=0) for c in known_classes]
    n_extra = max(0, k - len(seeds))

    pool = X[~mask_known] if (~mask_known).any() else X
    if n_extra > 0:
        # k-means++ style: iteratively pick farthest-from-current-seeds points
        chosen = []
        if not seeds:
            chosen.append(pool[rng.integers(len(pool))])
        cur = np.asarray(seeds + chosen) if (seeds or chosen) else pool[:1]
        while len(seeds) + len(chosen) < k:
            d2 = ((pool[:, None, :] - cur[None, :, :]) ** 2).sum(-1).min(axis=1)
            probs = d2 / (d2.sum() + 1e-12)
            chosen.append(pool[rng.choice(len(pool), p=probs)])
            cur = np.asarray(seeds + chosen)
        init = np.asarray(seeds + chosen)
    else:
        init = np.asarray(seeds[:k])

    km = KMeans(n_clusters=k, init=init, n_init=1, random_state=random_state)
    return km.fit_predict(X)

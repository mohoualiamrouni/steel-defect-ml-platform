"""
Diversity-based clustering for stratified active learning.

Clusters the CNN embedding space so that the candidate selector can pick
top-uncertain images from *each* cluster — ensuring the labeling budget
covers diverse surface defect types rather than redundant similar patches.

Supported methods:
    kmeans      — fast, deterministic for known k (default)
    hdbscan     — density-based, no fixed k, robust to noise

Advanced selection strategies (do not cluster — directly select a budget):
    coreset     — Greedy k-center (Sener & Savarese, ICLR 2018)
                  Minimizes the covering radius δ(S) = max_{x∉S} min_{s∈S} d(x,s).
                  Selects the most geometrically diverse subset.
    sensitivity — Clustering-based sensitivity sampling (Axiotis et al., 2024)
                  s(x_i) = d(x_i, μ_{C(i)})² / Φ_k + 1 / |C(i)|
                  Provably preserves the average loss up to (1±ε) factor.

References:
    Sener & Savarese (2018) "Active Learning for CNNs: A Core-Set Approach" ICLR.
    Axiotis et al. (2024) "Data-Efficient Learning via Clustering-Based Sensitivity
        Sampling: Foundation Models and Beyond" NeurIPS.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np

ClusterMethod = Literal["kmeans", "hdbscan"]


def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    n_clusters: int = 10,
    method: ClusterMethod = "kmeans",
    random_state: int = 42,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
) -> Dict[str, int]:
    """Cluster image embeddings for diversity-based sampling.

    Args:
        embeddings:       Dict {image_path: embedding_vector}.
        n_clusters:       Target number of clusters (KMeans) or used to derive
                          min_cluster_size for HDBSCAN if not provided.
        method:           'kmeans' or 'hdbscan'.
        random_state:     Random seed for reproducibility.
        min_cluster_size: HDBSCAN param. Defaults to len(embeddings) // n_clusters.
        min_samples:      HDBSCAN param. Defaults to min_cluster_size // 2.

    Returns:
        Dict mapping image_path -> cluster_id (int).
        HDBSCAN assigns -1 to noise points; treat them as their own cluster.
    """
    paths = list(embeddings.keys())
    X = np.stack([embeddings[p] for p in paths]).astype(np.float32)  # (N, D)

    if method == "kmeans":
        labels = _kmeans(X, n_clusters, random_state)
    elif method == "hdbscan":
        n = len(X)
        if min_cluster_size is None:
            min_cluster_size = max(2, n // n_clusters)
        if min_samples is None:
            min_samples = max(1, min_cluster_size // 2)
        labels = _hdbscan(X, min_cluster_size, min_samples)
    else:
        raise ValueError(f"Unknown clustering method '{method}'. Choose: kmeans, hdbscan")

    return {path: int(label) for path, label in zip(paths, labels)}


# ---------------------------------------------------------------------------
# Greedy k-center Core-Set  (Sener & Savarese ICLR 2018)
# ---------------------------------------------------------------------------

def greedy_k_center(
    embeddings: Dict[str, np.ndarray],
    budget: int,
    already_labeled: Optional[List[str]] = None,
) -> List[str]:
    """Greedy k-center core-set selection.

    Minimizes the covering radius δ(S) = max_{x∉S} min_{s∈S} d(x,s) by
    iteratively adding the unlabeled point farthest from its nearest labeled
    center.  This is a 2-approximation to the optimal k-center problem
    (Gonzalez 1985) and bounds the generalization gap as per Theorem 1 in
    Sener & Savarese (2018).

    Args:
        embeddings:      {path: embedding} for all unlabeled candidates.
        budget:          Number of points to select.
        already_labeled: Paths already in the labeled set (act as initial centers).
                         If None, the first unlabeled point is used as seed.

    Returns:
        List of `budget` paths selected as the most geometrically diverse subset.
    """
    paths = list(embeddings.keys())
    X = np.stack([embeddings[p] for p in paths]).astype(np.float32)   # (N, D)
    N = len(paths)
    budget = min(budget, N)

    # min_dist[i] = distance to the nearest already-selected center
    if already_labeled:
        labeled_paths = [p for p in already_labeled if p in embeddings]
        if labeled_paths:
            X_labeled = np.stack([embeddings[p] for p in labeled_paths]).astype(np.float32)
            # For each unlabeled point, find min distance to any labeled point
            min_dist = np.full(N, np.inf, dtype=np.float32)
            for c in X_labeled:
                d = np.linalg.norm(X - c, axis=1)
                min_dist = np.minimum(min_dist, d)
        else:
            min_dist = np.full(N, np.inf, dtype=np.float32)
    else:
        # Seed: first point in the pool
        min_dist = np.linalg.norm(X - X[0], axis=1)

    selected_indices: List[int] = []
    for _ in range(budget):
        # Select the point with maximum min-distance to current centers
        idx = int(np.argmax(min_dist))
        selected_indices.append(idx)
        # Update min_dist with distance to newly added center
        d_to_new = np.linalg.norm(X - X[idx], axis=1)
        min_dist = np.minimum(min_dist, d_to_new)
        min_dist[idx] = 0.0   # already selected

    return [paths[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Sensitivity Sampling  (Axiotis et al. 2024 Theorem 1)
# ---------------------------------------------------------------------------

def sensitivity_sample(
    embeddings: Dict[str, np.ndarray],
    n_clusters: int,
    budget: int,
    random_state: int = 42,
) -> List[str]:
    """Clustering-based sensitivity sampling.

    Computes per-point sensitivity scores:
        s(x_i) = d(x_i, μ_{C(i)})² / Φ_k  +  1 / |C(i)|
    where:
        μ_{C(i)} = centroid of x_i's cluster
        Φ_k      = total k-means cost = Σ_i d(x_i, μ_{C(i)})²
        |C(i)|   = size of x_i's cluster

    Samples `budget` points proportional to s(x_i) without replacement.
    This provably preserves the average loss of the full dataset up to a
    multiplicative (1±ε) factor (Axiotis et al. 2024 Theorem 1).

    Args:
        embeddings:   {path: embedding} for all candidates.
        n_clusters:   Number of k-means clusters for sensitivity computation.
        budget:       Number of points to select.
        random_state: Random seed.

    Returns:
        List of `budget` paths selected proportional to sensitivity.
    """
    from sklearn.cluster import KMeans

    paths = list(embeddings.keys())
    X = np.stack([embeddings[p] for p in paths]).astype(np.float32)
    N = len(paths)
    budget = min(budget, N)

    k = min(n_clusters, N)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_   # (k, D)

    # Squared distances to assigned centroid
    assigned_centroids = centroids[labels]          # (N, D)
    sq_dist = np.sum((X - assigned_centroids) ** 2, axis=1)   # (N,)

    # k-means cost
    phi_k = sq_dist.sum()
    if phi_k < 1e-12:
        phi_k = 1e-12   # avoid division by zero for constant embeddings

    # Cluster sizes
    cluster_sizes = np.bincount(labels, minlength=k)   # (k,)
    point_cluster_sizes = cluster_sizes[labels]         # (N,)

    # Sensitivity: outlier-ness + rarity
    sensitivity = sq_dist / phi_k + 1.0 / point_cluster_sizes.astype(np.float32)

    # Normalize to probability distribution
    prob = sensitivity / sensitivity.sum()

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(N, size=budget, replace=False, p=prob)

    return [paths[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X)


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def _hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        return clusterer.fit_predict(X)
    except ImportError:
        from sklearn.cluster import HDBSCAN as skHDBSCAN
        clusterer = skHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        return clusterer.fit_predict(X)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_cluster_counts(cluster_ids: Dict[str, int]) -> Dict[int, int]:
    """Return {cluster_id: count} sorted by cluster_id."""
    from collections import Counter
    return dict(sorted(Counter(cluster_ids.values()).items()))

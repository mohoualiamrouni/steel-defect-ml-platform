"""
CoilClusterer — Cluster Coil Flatness Feature Vectors
======================================================

Fits KMeans, HDBSCAN, GMM, Spectral, Agglomerative, or MiniBatchKMeans on the
1105-dim statistical feature vectors produced
by statistical_features.extract_batch().  Provides:

  - fit()        — train on historical coil library
  - predict()    — assign one or more new coils to nearest cluster
  - distance_to_all_clusters() — inspect confidence of assignment
  - predict_with_confidence()  — DEC soft assignment + outlier flag
  - save() / load() — persist model artifacts

The scaler (StandardScaler) is fitted and serialised together with the
clustering model so that the same normalisation is applied at inference.

DEC Soft Assignment (Asadi & Regan 2021)
-----------------------------------------
Confidence is computed using the Student-t kernel from Deep Embedded Clustering:

    q_{ij} = (1 + d(z_i, μ_j)² / α)^{-(α+1)/2}
             ────────────────────────────────────────
             Σ_{j'} (1 + d(z_i, μ_{j'})² / α)^{-(α+1)/2}

with α=1 (one degree of freedom).  This gives soft cluster membership
probabilities summing to 1, with higher values for nearby cluster centres.

Reference:
    Asadi & Regan (2021) "Self-supervised Deep Embedded Clustering for
    Surface Defect Detection" — soft t-distribution cluster assignment.

Distance Metric
---------------
    "euclidean" (default) — standard L2 distance in scaled feature space.
    "cosine"              — cosine distance = 1 - cosine_similarity.
                            Better for high-dimensional feature spaces.

Outlier Detection
-----------------
    "percentile"   (default) — outlier if d_nearest > training_percentile.
    "mahalanobis"            — outlier if Mahalanobis distance > χ²(0.95, df=F)
                               threshold per cluster.

Usage
-----
    from flatness_clustering.clustering.coil_clusterer import CoilClusterer

    clusterer = CoilClusterer(method="kmeans", n_clusters=6)
    clusterer.fit(feature_matrix)      # (N, 125) float32

    cluster_id = clusterer.predict(new_feature)      # int
    distances  = clusterer.distance_to_all_clusters(new_feature)  # (K,)
    result     = clusterer.predict_with_confidence(new_feature)
    # result["soft_assignments"]  → (K,) probability vector  (DEC)
    # result["confidence"]        → max(soft_assignments)
    # result["is_outlier"]        → bool

    clusterer.save("flatness_clustering/outputs/clusterer.joblib")
    clusterer2 = CoilClusterer.load("flatness_clustering/outputs/clusterer.joblib")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


OutlierMethod = Literal["percentile", "mahalanobis"]
DistanceMetric = Literal["euclidean", "cosine"]


class CoilClusterer:
    """Cluster steel coil flatness profiles using KMeans or HDBSCAN.

    Args:
        method:             'kmeans', 'hdbscan', 'gmm', 'spectral',
                            'agglomerative', or 'minibatch_kmeans'.
        n_clusters:         Number of clusters (ignored for HDBSCAN).
        random_state:       Random seed for reproducibility.
        outlier_percentile: Percentile of within-cluster distances above which a
                            point is flagged as an outlier (percentile method only).
        distance_metric:    'euclidean' (default) or 'cosine'.
        outlier_method:     'percentile' (default) or 'mahalanobis'.
        dec_alpha:          Student-t degrees of freedom for DEC soft assignment (default 1.0).
    """

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 6,
        random_state: int = 42,
        outlier_percentile: float = 95.0,
        distance_metric: DistanceMetric = "euclidean",
        outlier_method: OutlierMethod = "percentile",
        dec_alpha: float = 1.0,
    ) -> None:
        _VALID_METHODS = ("kmeans", "hdbscan", "gmm", "spectral", "agglomerative", "minibatch_kmeans")
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.outlier_percentile = outlier_percentile
        self.distance_metric = distance_metric
        self.outlier_method = outlier_method
        self.dec_alpha = dec_alpha
        # Feature-extraction variant the clusterer was TRAINED with. Persisted
        # in the joblib so inference uses the same variant and can never
        # silently mismatch the training feature dimension.
        self.rocket_variant: Optional[str] = None

        self._scaler: StandardScaler = StandardScaler()
        self._model = None                   # fitted sklearn estimator
        self.cluster_labels_: Optional[np.ndarray] = None   # (N,)
        self.centroids_: Optional[np.ndarray] = None        # (K, F)
        self._outlier_threshold: float = 0.0
        # Mahalanobis: per-cluster inverse covariance and chi2 threshold
        self._inv_covs_: Optional[list] = None
        self._mahal_thresholds_: Optional[np.ndarray] = None
        self._fitted: bool = False

    @property
    def expected_n_features(self) -> Optional[int]:
        """Feature dimension seen at fit time (``None`` before fitting)."""
        return getattr(self._scaler, "n_features_in_", None)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "CoilClusterer":
        """Fit clustering on the training feature matrix.

        Standardises features first (zero mean, unit variance), then fits
        the selected clustering algorithm.

        Args:
            features: float32 array of shape (N, F) — one row per coil.

        Returns:
            self (for chaining)
        """
        X = self._scaler.fit_transform(features.astype(np.float64))

        if self.method == "kmeans":
            self._model = self._fit_kmeans(X)
            self.cluster_labels_ = self._model.labels_
            self.centroids_ = self._model.cluster_centers_         # (K, F)
        elif self.method == "minibatch_kmeans":
            self._model = self._fit_minibatch_kmeans(X)
            self.cluster_labels_ = self._model.labels_
            self.centroids_ = self._model.cluster_centers_
        elif self.method == "gmm":
            self._model = self._fit_gmm(X)
            self.cluster_labels_ = self._model.predict(X)
            self.centroids_ = self._model.means_                   # (K, F)
        elif self.method == "spectral":
            self._model = self._fit_spectral(X)
            self.cluster_labels_ = self._model.labels_
            self.centroids_ = _compute_centroids(X, self.cluster_labels_)
        elif self.method == "agglomerative":
            self._model = self._fit_agglomerative(X)
            self.cluster_labels_ = self._model.labels_
            self.centroids_ = _compute_centroids(X, self.cluster_labels_)
        else:  # hdbscan
            self._model = self._fit_hdbscan(X)
            self.cluster_labels_ = self._model.labels_
            # Compute centroids as mean of each cluster (ignoring noise -1)
            unique_ids = sorted(set(self.cluster_labels_))
            if -1 in unique_ids:
                unique_ids.remove(-1)
            if unique_ids:
                self.centroids_ = np.stack([
                    X[self.cluster_labels_ == c].mean(axis=0)
                    for c in unique_ids
                ])
            else:
                self.centroids_ = X[:1]   # fallback

        # Compute outlier threshold: distance distribution within each cluster
        distances = self._distances_to_nearest_centroid(X)
        self._outlier_threshold = float(
            np.percentile(distances, self.outlier_percentile)
        )

        # Mahalanobis outlier threshold: per-cluster covariance + chi2 cutoff
        if self.outlier_method == "mahalanobis":
            self._fit_mahalanobis(X)

        self._fitted = True
        return self

    def _fit_kmeans(self, X: np.ndarray):
        from sklearn.cluster import KMeans
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        model.fit(X)
        return model

    def _fit_hdbscan(self, X: np.ndarray):
        n = len(X)
        min_cluster_size = max(2, n // (self.n_clusters * 2))
        try:
            import hdbscan
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
            )
        except ImportError:
            from sklearn.cluster import HDBSCAN
            model = HDBSCAN(min_cluster_size=min_cluster_size)
        model.fit(X)
        return model

    def _fit_gmm(self, X: np.ndarray):
        """Gaussian Mixture Model — soft probabilistic cluster assignments.

        From Ren et al. 2021 (DeepAL Survey): GMM provides density-aware
        clustering with per-cluster covariance, useful when clusters have
        varying shape and scale.
        """
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200,
            n_init=3,
        )
        model.fit(X)
        return model

    def _fit_spectral(self, X: np.ndarray):
        """Spectral Clustering — finds non-convex cluster shapes via graph Laplacian.

        Uses RBF affinity; assign_labels='kmeans' for fast final assignment.
        Note: no native centroids — computed as cluster means post-fit.
        """
        from sklearn.cluster import SpectralClustering
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            assign_labels="kmeans",
            affinity="rbf",
            n_jobs=-1,
        )
        model.fit(X)
        return model

    def _fit_agglomerative(self, X: np.ndarray):
        """Agglomerative Clustering with Ward linkage — minimizes within-cluster variance.

        From Saas et al. 2017 (1710.02268): Ward's method is recommended for
        time-series clustering with dense, synchronized profiles. Produces
        a hierarchical tree; use plot_dendrogram() to visualize.
        """
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="ward",
            metric="euclidean",
        )
        model.fit(X)
        return model

    def _fit_minibatch_kmeans(self, X: np.ndarray):
        """MiniBatch KMeans — fast approximation for large datasets.

        Useful for large steel production datasets (N > 10,000 coils) where
        standard KMeans is slow. Produces comparable quality to KMeans.
        """
        from sklearn.cluster import MiniBatchKMeans
        model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=min(1024, len(X)),
            max_iter=300,
            n_init=3,
        )
        model.fit(X)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign coil(s) to the nearest cluster.

        Uses Euclidean distance to cluster centroids in the scaled feature space.

        Args:
            features: float32 array of shape (F,) for one coil or (N, F) for a batch.

        Returns:
            int array of shape () or (N,) — cluster IDs.
        """
        self._check_fitted()
        single = features.ndim == 1
        X = self._scaler.transform(
            features.reshape(1, -1).astype(np.float64) if single
            else features.astype(np.float64)
        )
        dists = self._distances_to_all_centroids(X)   # (N, K)
        labels = dists.argmin(axis=1)
        return int(labels[0]) if single else labels

    def distance_to_all_clusters(self, feature: np.ndarray) -> np.ndarray:
        """Return Euclidean distance from one coil to every cluster centroid.

        Args:
            feature: float32 array of shape (F,).

        Returns:
            Distance array of shape (K,).
        """
        self._check_fitted()
        X = self._scaler.transform(feature.reshape(1, -1).astype(np.float64))
        return self._distances_to_all_centroids(X)[0]

    def predict_with_confidence(self, feature: np.ndarray) -> Dict:
        """Assign a coil and return rich assignment metadata using DEC soft assignment.

        Confidence is computed via the DEC Student-t kernel (Asadi & Regan 2021):
            q_{ij} = (1 + d²/α)^{-(α+1)/2} / Σ_{j'} (1 + d²/α)^{-(α+1)/2}
        where d is the distance to centroid j and α = dec_alpha (default 1.0).

        Args:
            feature: float32 array of shape (F,).

        Returns:
            dict with keys:
                cluster_id       : int — nearest cluster (argmax of soft_assignments)
                cluster_distances: np.ndarray (K,) — distance to each centroid
                soft_assignments : np.ndarray (K,) — DEC soft membership probabilities,
                                   sums to 1.0 (Asadi & Regan 2021 Eq. 3)
                confidence       : float in [0, 1] — max(soft_assignments)
                is_outlier       : bool — True if flagged as outlier
        """
        self._check_fitted()
        X = self._scaler.transform(feature.reshape(1, -1).astype(np.float64))
        distances = self._distances_to_all_centroids(X)[0]   # (K,)

        # GMM provides native soft assignments via predict_proba
        if self.method == "gmm":
            soft_assignments = self._model.predict_proba(X)[0].astype(np.float32)
        else:
            # DEC soft assignment (Asadi & Regan 2021) for all other methods
            soft_assignments = _soft_assignment(distances, self.dec_alpha)

        nearest_id = int(soft_assignments.argmax())
        confidence = float(soft_assignments.max())

        # Outlier detection
        if self.outlier_method == "mahalanobis" and self._inv_covs_ is not None:
            is_outlier = self._mahalanobis_outlier(X[0], nearest_id)
        else:
            is_outlier = bool(float(distances.min()) > self._outlier_threshold)

        return {
            "cluster_id":        nearest_id,
            "cluster_distances": distances,
            "soft_assignments":  soft_assignments,
            "confidence":        confidence,
            "is_outlier":        is_outlier,
        }

    def uncertainty_score(self, feature: np.ndarray) -> float:
        """Return the clustering uncertainty of a single coil profile.

        Computes the entropy of the soft cluster assignment vector:
            H(q) = -Σ_j q_j * log(q_j)

        High entropy → the profile is ambiguous between several clusters
        (e.g., sits between two flatness pattern types).
        Low entropy  → the profile clearly belongs to one cluster.

        This bridges the flatness pipeline into the active learning framework:
        flatness profiles with high uncertainty_score are the best candidates
        for human labeling or expert review.

        Args:
            feature: float32 array of shape (F,) — one coil's feature vector.

        Returns:
            Entropy in nats (float >= 0). Maximum is log(n_clusters).
        """
        result = self.predict_with_confidence(feature)
        q = result["soft_assignments"].astype(np.float64)
        q_clipped = np.clip(q, 1e-12, 1.0)
        return float(-np.sum(q_clipped * np.log(q_clipped)))

    def dec_target_distribution(self, Q: np.ndarray) -> np.ndarray:
        """Compute DEC target distribution P from soft assignments Q.

        Used for iterative self-training refinement (Xie et al. 2016 Eq. 4):

            p_{ij} = (q_{ij}² / f_j) / Σ_{j'} (q_{ij'}² / f_{j'})
            where f_j = Σ_i q_{ij}  (soft cluster frequency)

        P sharpens Q: it upweights high-confidence assignments and
        downweights uncertain ones, effectively self-labeling the data.
        Use KL(P || Q) as the DEC refinement loss.

        Args:
            Q: Soft assignment matrix (N, K) — output of predict_with_confidence
               called over the full training set.

        Returns:
            Target distribution P, shape (N, K), rows sum to 1.
        """
        Q = np.asarray(Q, dtype=np.float64)
        f = Q.sum(axis=0)  # (K,) — soft cluster frequencies
        f = np.maximum(f, 1e-12)
        numerator = Q ** 2 / f[np.newaxis, :]  # (N, K)
        return (numerator / numerator.sum(axis=1, keepdims=True)).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _distances_to_all_centroids(self, X: np.ndarray) -> np.ndarray:
        """Distance from each row in X to each centroid. Shape: (N, K)

        Uses Euclidean or cosine distance depending on self.distance_metric.
        For cosine: features and centroids are L2-normalized before computing
        Euclidean distance in the unit-sphere space (equivalent to cosine dist).
        """
        if self.distance_metric == "cosine":
            # Normalize to unit sphere
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            C_norm = self.centroids_ / (
                np.linalg.norm(self.centroids_, axis=1, keepdims=True) + 1e-12
            )
            # cosine distance = 1 - cosine similarity
            sim = X_norm @ C_norm.T          # (N, K)
            return 1.0 - np.clip(sim, -1.0, 1.0)
        else:
            diff = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]  # (N, K, F)
            return np.sqrt((diff ** 2).sum(axis=2))                          # (N, K)

    def _distances_to_nearest_centroid(self, X: np.ndarray) -> np.ndarray:
        """Minimum distance to any centroid for each point. Shape: (N,)"""
        return self._distances_to_all_centroids(X).min(axis=1)

    def _fit_mahalanobis(self, X: np.ndarray) -> None:
        """Fit per-cluster covariance matrices for Mahalanobis outlier detection.

        Uses regularized covariance (Ledoit-Wolf shrinkage via sklearn) to
        handle high-dimensional feature spaces.  Threshold = χ²(0.95, df=F).
        """
        from scipy.stats import chi2

        K = len(self.centroids_)
        F = X.shape[1]
        self._inv_covs_ = []
        self._mahal_thresholds_ = np.full(K, np.inf)

        for k in range(K):
            mask = self.cluster_labels_ == k
            X_k = X[mask]
            if len(X_k) < F + 1:
                # Not enough points: fall back to identity
                self._inv_covs_.append(np.eye(F))
            else:
                try:
                    from sklearn.covariance import LedoitWolf
                    lw = LedoitWolf().fit(X_k)
                    self._inv_covs_.append(lw.precision_)
                except Exception:
                    self._inv_covs_.append(np.eye(F))

            self._mahal_thresholds_[k] = chi2.ppf(0.95, df=F)

    def _mahalanobis_outlier(self, x: np.ndarray, cluster_id: int) -> bool:
        """Return True if x is an outlier w.r.t. cluster_id's Mahalanobis distance."""
        if self._inv_covs_ is None or cluster_id >= len(self._inv_covs_):
            return False
        mu = self.centroids_[cluster_id]
        inv_cov = self._inv_covs_[cluster_id]
        diff = x - mu
        mahal_sq = float(diff @ inv_cov @ diff)
        return mahal_sq > self._mahal_thresholds_[cluster_id]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CoilClusterer is not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise to disk with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"CoilClusterer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "CoilClusterer":
        """Load a previously saved clusterer."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CoilClusterer, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        k = len(self.centroids_) if self.centroids_ is not None else "?"
        return (
            f"CoilClusterer(method={self.method}, n_clusters={k}, "
            f"distance_metric={self.distance_metric!r}, "
            f"outlier_method={self.outlier_method!r}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute cluster centroids as mean of points in each cluster.

    Used for methods that don't natively expose centroids (Spectral, Agglomerative).

    Args:
        X:      (N, F) scaled feature matrix.
        labels: (N,) integer cluster label array.

    Returns:
        (K, F) centroid matrix where K = number of unique labels.
    """
    unique_ids = sorted(set(labels))
    centroids = []
    for c in unique_ids:
        mask = labels == c
        if mask.sum() > 0:
            centroids.append(X[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(X.shape[1]))
    return np.stack(centroids)


# ---------------------------------------------------------------------------
# DEC soft assignment  (Asadi & Regan 2021)
# ---------------------------------------------------------------------------

def _soft_assignment(distances: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """DEC Student-t kernel soft cluster assignment (Asadi & Regan 2021).

    Computes soft membership probabilities over K clusters using the
    Student-t distribution kernel:

        q_j = (1 + d_j² / α)^{-(α+1)/2}
              ───────────────────────────────────────
              Σ_{j'} (1 + d_{j'}² / α)^{-(α+1)/2}

    where d_j = distance to centroid j, α = degrees of freedom (default 1).

    With α=1 this is the Cauchy kernel; higher α approaches a Gaussian.
    Soft assignments sum to 1.0 and can be interpreted as cluster
    membership probabilities.

    Reference:
        Asadi & Regan (2021) Eq. 3 — self-supervised DEC soft assignment.
        Xie et al. (2016) "Unsupervised Deep Embedding for Clustering Analysis"
        (original DEC formulation).

    Args:
        distances: (K,) array of distances to each cluster centroid.
        alpha:     Student-t degrees of freedom (default 1.0).

    Returns:
        (K,) float32 array of soft membership probabilities summing to 1.
    """
    d_sq = distances.astype(np.float64) ** 2
    numerator = (1.0 + d_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    total = numerator.sum()
    if total < 1e-12:
        # Fallback: uniform assignment
        return np.ones(len(distances), dtype=np.float32) / len(distances)
    return (numerator / total).astype(np.float32)

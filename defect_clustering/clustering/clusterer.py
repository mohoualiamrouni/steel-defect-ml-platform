"""
DefectClusterer -- Cluster Defect Group Feature Vectors
=======================================================

Fits KMeans, HDBSCAN, GMM, Spectral, Agglomerative, or MiniBatchKMeans on the
feature vectors produced by the defect feature extraction pipeline.  Provides:

  - fit()        -- train on historical defect library
  - predict()    -- assign one or more new defect samples to nearest cluster
  - distance_to_all_clusters() -- inspect confidence of assignment
  - predict_with_confidence()  -- DEC soft assignment + outlier flag
  - save() / load() -- persist model artifacts

The scaler (StandardScaler) is fitted and serialised together with the
clustering model so that the same normalisation is applied at inference.

DEC Soft Assignment (Asadi & Regan 2021)
-----------------------------------------
Confidence is computed using the Student-t kernel from Deep Embedded Clustering:

    q_{ij} = (1 + d(z_i, mu_j)^2 / alpha)^{-(alpha+1)/2}
             -----------------------------------------------
             Sum_{j'} (1 + d(z_i, mu_{j'})^2 / alpha)^{-(alpha+1)/2}

with alpha=1 (one degree of freedom).  This gives soft cluster membership
probabilities summing to 1, with higher values for nearby cluster centres.

Reference:
    Asadi & Regan (2021) "Self-supervised Deep Embedded Clustering for
    Surface Defect Detection" -- soft t-distribution cluster assignment.

Distance Metric
---------------
    "euclidean" (default) -- standard L2 distance in scaled feature space.
    "cosine"              -- cosine distance = 1 - cosine_similarity.
                            Better for high-dimensional feature spaces.

Outlier Detection
-----------------
    "percentile"   (default) -- outlier if d_nearest > training_percentile.
    "mahalanobis"            -- outlier if Mahalanobis distance > chi^2(0.95, df=F)
                               threshold per cluster.

Usage
-----
    from defect_clustering.clustering.clusterer import DefectClusterer

    clusterer = DefectClusterer(method="kmeans", n_clusters=6)
    clusterer.fit(feature_matrix)      # (N, F) float32

    cluster_id = clusterer.predict(new_feature)      # int
    distances  = clusterer.distance_to_all_clusters(new_feature)  # (K,)
    result     = clusterer.predict_with_confidence(new_feature)
    # result["soft_assignments"]  -> (K,) probability vector  (DEC)
    # result["confidence"]        -> max(soft_assignments)
    # result["is_outlier"]        -> bool

    clusterer.save("defect_clustering/outputs/clusterer.joblib")
    clusterer2 = DefectClusterer.load("defect_clustering/outputs/clusterer.joblib")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


OutlierMethod = Literal["percentile", "mahalanobis"]
DistanceMetric = Literal["euclidean", "cosine"]


class DefectClusterer:
    """Cluster defect group feature vectors using configurable algorithms.

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
        _VALID_METHODS = (
            "kmeans", "hdbscan", "gmm", "spectral", "agglomerative",
            "minibatch_kmeans", "dpgmm", "deep_dpm", "bisecting_merge",
        )
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.outlier_percentile = outlier_percentile
        self.distance_metric = distance_metric
        self.outlier_method = outlier_method
        self.dec_alpha = dec_alpha
        self.overcluster_factor = 2.0

        self._scaler: StandardScaler = StandardScaler()
        self._model = None                   # fitted sklearn estimator
        self.cluster_labels_: Optional[np.ndarray] = None   # (N,)
        self.centroids_: Optional[np.ndarray] = None        # (K, F)
        self._outlier_threshold: float = 0.0
        # Mahalanobis: per-cluster inverse covariance and chi2 threshold
        self._inv_covs_: Optional[list] = None
        self._mahal_thresholds_: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "DefectClusterer":
        """Fit clustering on the training feature matrix.

        Standardises features first (zero mean, unit variance), then fits
        the selected clustering algorithm.

        Args:
            features: float32 array of shape (N, F) -- one row per defect sample.

        Returns:
            self (for chaining)
        """
        X = self._scaler.fit_transform(features.astype(np.float64))
        if self.distance_metric == "cosine":
            from sklearn.preprocessing import normalize as _l2norm
            X = _l2norm(X, norm="l2", axis=1)

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
        elif self.method == "dpgmm":
            self._model = self._fit_dpgmm(X)
            self.cluster_labels_ = self._model.predict(X)
            self.centroids_ = self._model.means_[
                np.unique(self.cluster_labels_)
            ]
            # Remap labels to contiguous 0..K-1
            unique = np.unique(self.cluster_labels_)
            remap = {old: new for new, old in enumerate(unique)}
            self.cluster_labels_ = np.array([remap[l] for l in self.cluster_labels_])
            self.centroids_ = _compute_centroids(X, self.cluster_labels_)
        elif self.method == "bisecting_merge":
            self._model = None
            self.cluster_labels_, self.centroids_ = self._fit_bisecting_merge(X)
        elif self.method == "deep_dpm":
            self._model = self._fit_deep_dpm(X)
            self.cluster_labels_ = self._model.labels_
            self.centroids_ = self._model.cluster_centers_
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
        """Gaussian Mixture Model -- soft probabilistic cluster assignments.

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
        """Spectral Clustering -- finds non-convex cluster shapes via graph Laplacian.

        Uses RBF affinity; assign_labels='kmeans' for fast final assignment.
        Note: no native centroids -- computed as cluster means post-fit.
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
        """Agglomerative Clustering with Ward linkage -- minimizes within-cluster variance.

        From Saas et al. 2017 (1710.02268): Ward's method is recommended for
        clustering dense feature representations. Produces a hierarchical tree;
        use plot_dendrogram() to visualize.
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
        """MiniBatch KMeans -- fast approximation for large datasets.

        Useful for large defect datasets (N > 10,000 samples) where standard
        KMeans is slow. Produces comparable quality to KMeans.
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

    def _fit_bisecting_merge(self, X: np.ndarray):
        """Over-cluster with BisectingKMeans then Ward-merge centroids.

        Steinbach et al. 2000 (Bisecting KMeans) + Bai 2017 / Fred & Jain 2005
        (centroid-level agglomerative merge) + Caron 2018 (over-cluster by 2x).
        Fixes monster-cluster imbalance while preserving NMI.
        """
        from sklearn.cluster import AgglomerativeClustering, BisectingKMeans

        target_k = int(self.n_clusters)
        over_k = max(target_k + 1, int(round(target_k * self.overcluster_factor)))
        over_k = min(over_k, X.shape[0] - 1)

        bkm = BisectingKMeans(
            n_clusters=over_k,
            random_state=self.random_state,
            bisecting_strategy="largest_cluster",
            n_init=3,
        )
        bkm.fit(X)
        sub_labels = bkm.labels_
        sub_centroids = bkm.cluster_centers_

        if over_k <= target_k:
            return sub_labels, sub_centroids

        agg = AgglomerativeClustering(n_clusters=target_k, linkage="ward")
        merge_map = agg.fit_predict(sub_centroids)
        final_labels = merge_map[sub_labels]

        final_centroids = np.stack([
            X[final_labels == c].mean(axis=0) for c in range(target_k)
        ])
        return final_labels.astype(int), final_centroids

    def _fit_dpgmm(self, X: np.ndarray):
        """Bayesian Gaussian Mixture with Dirichlet Process prior.

        Variational inference approximation to the Dirichlet Process
        (Blei & Jordan 2006).  Automatically prunes unused components,
        effectively discovering K from the data.

        Related to DeepDPM (Ronen et al. 2022) — uses the same underlying
        Dirichlet Process model but with variational inference instead of
        the MLP + split/merge approach.
        """
        import warnings
        from sklearn.mixture import BayesianGaussianMixture

        max_components = max(self.n_clusters * 2, 50)
        model = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=10.0,
            max_iter=300,
            n_init=3,
            random_state=self.random_state,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        return model

    def _fit_deep_dpm(self, X: np.ndarray):
        """DeepDPM — Non-parametric deep clustering (Ronen et al. CVPR 2022).

        Uses a Bayesian GMM initialisation followed by an MLP cluster network
        with split/merge via Hastings ratio (Normal-Inverse-Wishart prior).
        Automatically discovers K without user specification.
        """
        from defect_clustering.clustering.deep_dpm import DeepDPMClustering

        model = DeepDPMClustering(
            init_k=self.n_clusters,
            max_epochs=150,
            split_merge_every=25,
            start_split_merge=40,
            alpha=10.0,
            hidden_dims=[128, 64],
            lr=0.002,
            batch_size=256,
            min_cluster_size=10,
            random_state=self.random_state,
            use_mlp=True,
            verbose=True,
        )
        model.fit(X)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign defect sample(s) to the nearest cluster.

        Uses the configured distance metric in the scaled feature space.

        Args:
            features: float32 array of shape (F,) for one sample or (N, F) for a batch.

        Returns:
            int array of shape () or (N,) -- cluster IDs.
        """
        self._check_fitted()
        single = features.ndim == 1
        X = self._scaler.transform(
            features.reshape(1, -1).astype(np.float64) if single
            else features.astype(np.float64)
        )
        if self.distance_metric == "cosine":
            from sklearn.preprocessing import normalize as _l2norm
            X = _l2norm(X, norm="l2", axis=1)
        dists = self._distances_to_all_centroids(X)   # (N, K)
        labels = dists.argmin(axis=1)
        return int(labels[0]) if single else labels

    def distance_to_all_clusters(self, feature: np.ndarray) -> np.ndarray:
        """Return distance from one sample to every cluster centroid.

        Args:
            feature: float32 array of shape (F,).

        Returns:
            Distance array of shape (K,).
        """
        self._check_fitted()
        X = self._scaler.transform(feature.reshape(1, -1).astype(np.float64))
        if self.distance_metric == "cosine":
            from sklearn.preprocessing import normalize as _l2norm
            X = _l2norm(X, norm="l2", axis=1)
        return self._distances_to_all_centroids(X)[0]

    def predict_with_confidence(self, feature: np.ndarray) -> Dict:
        """Assign a defect sample and return rich assignment metadata using DEC soft assignment.

        Confidence is computed via the DEC Student-t kernel (Asadi & Regan 2021):
            q_{ij} = (1 + d^2/alpha)^{-(alpha+1)/2} / Sum_{j'} (1 + d^2/alpha)^{-(alpha+1)/2}
        where d is the distance to centroid j and alpha = dec_alpha (default 1.0).

        Args:
            feature: float32 array of shape (F,).

        Returns:
            dict with keys:
                cluster_id       : int -- nearest cluster (argmax of soft_assignments)
                cluster_distances: np.ndarray (K,) -- distance to each centroid
                soft_assignments : np.ndarray (K,) -- DEC soft membership probabilities,
                                   sums to 1.0 (Asadi & Regan 2021 Eq. 3)
                confidence       : float in [0, 1] -- max(soft_assignments)
                is_outlier       : bool -- True if flagged as outlier
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
        """Return the clustering uncertainty of a single defect sample.

        Computes the entropy of the soft cluster assignment vector:
            H(q) = -Sum_j q_j * log(q_j)

        High entropy -> the sample is ambiguous between several clusters
        (e.g., sits between two defect types).
        Low entropy  -> the sample clearly belongs to one cluster.

        This bridges the defect pipeline into the active learning framework:
        defect samples with high uncertainty_score are the best candidates
        for human labeling or expert review.

        Args:
            feature: float32 array of shape (F,) -- one sample's feature vector.

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

            p_{ij} = (q_{ij}^2 / f_j) / Sum_{j'} (q_{ij'}^2 / f_{j'})
            where f_j = Sum_i q_{ij}  (soft cluster frequency)

        P sharpens Q: it upweights high-confidence assignments and
        downweights uncertain ones, effectively self-labeling the data.
        Use KL(P || Q) as the DEC refinement loss.

        Args:
            Q: Soft assignment matrix (N, K) -- output of predict_with_confidence
               called over the full training set.

        Returns:
            Target distribution P, shape (N, K), rows sum to 1.
        """
        Q = np.asarray(Q, dtype=np.float64)
        f = Q.sum(axis=0)  # (K,) -- soft cluster frequencies
        f = np.maximum(f, 1e-12)
        numerator = Q ** 2 / f[np.newaxis, :]  # (N, K)
        return (numerator / numerator.sum(axis=1, keepdims=True)).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _distances_to_all_centroids(self, X: np.ndarray) -> np.ndarray:
        """Distance from each row in X to each centroid. Shape: (N, K).

        Delegates to sklearn's batched ``pairwise_distances``; both branches
        avoid the (N, K, F) broadcast that OOMs on wide CNN features.
        """
        from sklearn.metrics import pairwise_distances
        return pairwise_distances(
            X, self.centroids_, metric=self.distance_metric,
        )

    def _distances_to_nearest_centroid(self, X: np.ndarray) -> np.ndarray:
        """Minimum distance to any centroid for each point. Shape: (N,)."""
        from sklearn.metrics import pairwise_distances_argmin_min
        _, dmin = pairwise_distances_argmin_min(
            X, self.centroids_, metric=self.distance_metric,
        )
        return dmin

    def _fit_mahalanobis(self, X: np.ndarray) -> None:
        """Fit per-cluster covariance matrices for Mahalanobis outlier detection.

        Uses regularized covariance (Ledoit-Wolf shrinkage via sklearn) to
        handle high-dimensional feature spaces.  Threshold = chi^2(0.95, df=F).
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
            raise RuntimeError("DefectClusterer is not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise to disk with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"DefectClusterer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "DefectClusterer":
        """Load a previously saved clusterer."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DefectClusterer, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        k = len(self.centroids_) if self.centroids_ is not None else "?"
        return (
            f"DefectClusterer(method={self.method}, n_clusters={k}, "
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

        q_j = (1 + d_j^2 / alpha)^{-(alpha+1)/2}
              -----------------------------------------------
              Sum_{j'} (1 + d_{j'}^2 / alpha)^{-(alpha+1)/2}

    where d_j = distance to centroid j, alpha = degrees of freedom (default 1).

    With alpha=1 this is the Cauchy kernel; higher alpha approaches a Gaussian.
    Soft assignments sum to 1.0 and can be interpreted as cluster
    membership probabilities.

    Reference:
        Asadi & Regan (2021) Eq. 3 -- self-supervised DEC soft assignment.
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

"""
Spatial Feature Extraction (for Visualization Only)
====================================================

Downsamples each flatness profile to a fixed-size grid, then applies PCA
to produce a compact embedding (50 dims) used for t-SNE / UMAP plots.

This module is NOT used for clustering — the statistical features from
statistical_features.py are used for that.  The spatial embedding preserves
more visual structure and produces nicer cluster plots.

Downsampling method (controlled by `transform_method`):
    "paa"  (default) — Piecewise Aggregate Approximation (Saas et al. 2017)
                       Divides the profile into (target_h × target_w) non-overlapping
                       blocks and computes the block mean.  More shape-preserving than
                       interpolation: avoids artificial smoothing artefacts.
    "zoom"           — scipy.ndimage.zoom bilinear interpolation (legacy).

Reference:
    Saas et al. (2017) "Flatness Defect Classification" — PAA is more robust for
    preserving peak-to-valley patterns in flatness profiles than interpolation.

Usage
-----
    from flatness_clustering.features.spatial_features import SpatialFeatureExtractor

    extractor = SpatialFeatureExtractor(target_shape=(32, 64), n_components=50)
    embeddings = extractor.fit_transform(profiles)   # (N, 50) — uses PAA by default
    extractor.save("flatness_clustering/outputs/spatial_pca.joblib")

    # Legacy zoom mode:
    extractor = SpatialFeatureExtractor(target_shape=(32, 64), transform_method="zoom")

    # For new data:
    extractor2 = SpatialFeatureExtractor.load("flatness_clustering/outputs/spatial_pca.joblib")
    new_emb = extractor2.transform(new_profiles)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import joblib
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TransformMethod = Literal["paa", "zoom"]


class SpatialFeatureExtractor:
    """Downsampled-heatmap + PCA embedding for flatness profiles.

    Args:
        target_shape:     (rows, cols) to downsample each profile to.
                          Default (32, 64) produces 2048-dim flattened vectors.
        n_components:     Number of PCA components to keep.
        transform_method: "paa" (default) uses block-mean PAA downsampling
                          (Saas et al. 2017) which preserves peak-to-valley
                          patterns.  "zoom" uses bilinear scipy interpolation
                          (legacy behavior).
    """

    def __init__(
        self,
        target_shape: Tuple[int, int] = (32, 64),
        n_components: int = 50,
        transform_method: TransformMethod = "paa",
        random_state: int = 42,
    ) -> None:
        self.target_shape = target_shape
        self.n_components = n_components
        self.transform_method = transform_method
        self.random_state = random_state
        self._scaler: StandardScaler = StandardScaler()
        self._pca: PCA = PCA(n_components=n_components, random_state=random_state)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    def _downsample(self, profile: np.ndarray) -> np.ndarray:
        """Resample profile from (W, L) to target_shape."""
        if self.transform_method == "paa":
            return _paa_downsample(profile.astype(np.float32), *self.target_shape)
        else:
            h, w = profile.shape
            zoom_h = self.target_shape[0] / h
            zoom_w = self.target_shape[1] / w
            return zoom(profile.astype(np.float32), (zoom_h, zoom_w), order=1)

    def _downsample_batch(self, profiles: np.ndarray) -> np.ndarray:
        """Downsample (N, W, L) -> (N, target_h * target_w)."""
        flat_dim = self.target_shape[0] * self.target_shape[1]
        result = np.zeros((len(profiles), flat_dim), dtype=np.float32)
        for i, p in enumerate(profiles):
            result[i] = self._downsample(p).ravel()
        return result

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, profiles: np.ndarray) -> "SpatialFeatureExtractor":
        """Fit scaler and PCA on a batch of profiles.

        Args:
            profiles: float32 array shape (N, W, L).
        """
        X = self._downsample_batch(profiles)
        X_scaled = self._scaler.fit_transform(X)
        self._pca.fit(X_scaled)
        self._fitted = True
        explained = float(self._pca.explained_variance_ratio_.sum())
        print(f"SpatialFeatureExtractor: PCA variance explained = {explained:.1%}")
        return self

    def transform(self, profiles: np.ndarray) -> np.ndarray:
        """Transform profiles to PCA space.

        Args:
            profiles: float32 array shape (N, W, L).

        Returns:
            Embeddings of shape (N, n_components) float32.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before transform().")
        X = self._downsample_batch(profiles)
        X_scaled = self._scaler.transform(X)
        return self._pca.transform(X_scaled).astype(np.float32)

    def fit_transform(self, profiles: np.ndarray) -> np.ndarray:
        """Fit then transform."""
        return self.fit(profiles).transform(profiles)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise to disk with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SpatialFeatureExtractor":
        """Load a previously saved extractor."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SpatialFeatureExtractor, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"SpatialFeatureExtractor("
            f"target_shape={self.target_shape}, "
            f"n_components={self.n_components}, "
            f"transform_method={self.transform_method!r}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# PAA downsampling  (Saas et al. 2017)
# ---------------------------------------------------------------------------

def _paa_downsample(
    profile: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Piecewise Aggregate Approximation (PAA) downsampling for 2-D profiles.

    Divides the (H, W) profile into a (target_h × target_w) grid of
    non-overlapping blocks and computes the block mean.

    Advantages over scipy.ndimage.zoom (Saas et al. 2017):
    - Computes block means → preserves peak-to-valley amplitude.
    - No interpolation artefacts at sharp defect edges.
    - Deterministic: same result every call for the same input.

    Args:
        profile:  float32 array of shape (H, W).
        target_h: Number of row blocks (height of output grid).
        target_w: Number of column blocks (width of output grid).

    Returns:
        float32 array of shape (target_h, target_w).
    """
    H, W = profile.shape
    result = np.zeros((target_h, target_w), dtype=np.float32)
    row_splits = np.array_split(np.arange(H), target_h)
    col_splits = np.array_split(np.arange(W), target_w)
    for r, row_idx in enumerate(row_splits):
        for c, col_idx in enumerate(col_splits):
            block = profile[np.ix_(row_idx, col_idx)]
            result[r, c] = block.mean()
    return result

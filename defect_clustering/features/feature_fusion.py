"""Fuse CNN features and segmentation features into a unified representation.

The key design principle -- learned from flatness clustering where ROCKET
time-series features dominated the low-dimensional segmentation features --
is to **scale each feature block independently** before concatenation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Recognised fusion methods
_VALID_METHODS = {"concat_scale", "pca", "weighted", "cnn_only"}


def fuse_features(
    cnn_features: np.ndarray,
    seg_features: np.ndarray,
    method: str = "concat_scale",
    pca_components: Optional[int] = None,
    pca_variance: float = 0.95,
    cnn_weight: float = 1.0,
    seg_weight: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, dict]:
    """Fuse CNN and segmentation features.

    Parameters
    ----------
    cnn_features : np.ndarray
        ``(N, D_cnn)`` array, typically ``(N, 960)`` from EfficientNet.
    seg_features : np.ndarray
        ``(N, D_seg)`` array, typically ``(N, 48)`` segmentation statistics.
    method : str
        Fusion strategy.  One of:

        * ``"concat_scale"`` -- scale each block, optionally weight, concatenate,
          then optionally PCA-reduce.
        * ``"pca"`` -- alias for ``"concat_scale"`` with PCA always enabled.
        * ``"weighted"`` -- alias for ``"concat_scale"`` using explicit
          *cnn_weight* / *seg_weight* and **no** PCA.
    pca_components : int or None
        Number of PCA components to keep after concatenation.  When ``None``
        and ``pca_variance < 1.0``, variance-based auto-selection is used
        (recommended).  Set to an explicit integer to force a fixed count.
    pca_variance : float
        Target cumulative explained-variance ratio.  When *pca_components*
        is ``None``, this value is passed directly to
        ``PCA(n_components=pca_variance)`` for automatic component selection.
        When *pca_components* is an integer, used only as a warning threshold.
    cnn_weight : float
        Multiplicative weight applied to the scaled CNN block.
    seg_weight : float
        Multiplicative weight applied to the scaled segmentation block.
    random_state : int
        Random seed for PCA.

    Returns
    -------
    fused : np.ndarray
        ``(N, F)`` float32 fused feature matrix.
    info : dict
        Metadata for reproducibility and downstream inference.  Keys:

        * ``method`` -- the fusion method used.
        * ``n_samples`` -- number of samples.
        * ``n_features`` -- final feature dimension *F*.
        * ``cnn_dim`` -- original CNN feature dimension.
        * ``seg_dim`` -- original segmentation feature dimension.
        * ``cnn_weight`` -- weight applied to CNN block.
        * ``seg_weight`` -- weight applied to segmentation block.
        * ``cnn_scaler`` -- fitted ``StandardScaler`` for CNN block.
        * ``seg_scaler`` -- fitted ``StandardScaler`` for segmentation block.
        * ``pca`` -- fitted ``PCA`` object (or ``None``).
        * ``pca_variance_explained`` -- cumulative explained variance (or ``None``).
    """
    # ------------------------------------------------------------------
    # Short-circuit: cnn_only skips seg block entirely
    # ------------------------------------------------------------------
    if method == "cnn_only":
        cnn_scaler_only = StandardScaler()
        scaled = cnn_scaler_only.fit_transform(
            cnn_features.astype(np.float64)
        ).astype(np.float32)
        info = {
            "method": "cnn_only",
            "n_samples": cnn_features.shape[0],
            "n_features": int(scaled.shape[1]),
            "cnn_dim": int(cnn_features.shape[1]),
            "seg_dim": int(seg_features.shape[1]) if seg_features is not None else 0,
            "cnn_weight": 1.0,
            "seg_weight": 0.0,
            "cnn_scaler": cnn_scaler_only,
            "seg_scaler": None,
            "pca": None,
            "pca_variance_explained": None,
        }
        logger.info(
            "Fusion (cnn_only): (%d, %d)", scaled.shape[0], scaled.shape[1],
        )
        return scaled, info

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    _validate_inputs(cnn_features, seg_features, method)

    n_samples = cnn_features.shape[0]
    cnn_dim = cnn_features.shape[1]
    seg_dim = seg_features.shape[1]

    logger.info(
        "Fusing features: CNN (%d, %d) + Seg (%d, %d) | method=%s",
        n_samples, cnn_dim, n_samples, seg_dim, method,
    )

    # ------------------------------------------------------------------
    # Resolve method aliases
    # ------------------------------------------------------------------
    apply_pca: bool
    if method == "pca":
        apply_pca = True
        if pca_components is None:
            # Pick a sensible default: min(N, total_dim) / 2 but at least 50
            pca_components = max(50, min(n_samples, cnn_dim + seg_dim) // 2)
            logger.info(
                "method='pca' with pca_components=None: auto-selected %d components",
                pca_components,
            )
    elif method == "weighted":
        apply_pca = False
    else:  # concat_scale
        apply_pca = (pca_components is not None) or (pca_variance is not None and pca_variance < 1.0)

    # ------------------------------------------------------------------
    # Step 1: Independent StandardScaler on each block
    # ------------------------------------------------------------------
    cnn_scaler = StandardScaler()
    seg_scaler = StandardScaler()

    # Keep float32 throughout — StandardScaler handles float32 natively and
    # float64 upcast doubles the memory footprint of the fused matrix.
    cnn_scaled = cnn_scaler.fit_transform(cnn_features.astype(np.float32, copy=False))
    seg_scaled = seg_scaler.fit_transform(seg_features.astype(np.float32, copy=False))

    logger.info(
        "Scaled CNN block: mean=%.4f std=%.4f | Scaled Seg block: mean=%.4f std=%.4f",
        cnn_scaled.mean(), cnn_scaled.std(),
        seg_scaled.mean(), seg_scaled.std(),
    )

    # ------------------------------------------------------------------
    # Step 2: Apply per-block weights
    # ------------------------------------------------------------------
    if cnn_weight != 1.0:
        cnn_scaled = cnn_scaled * cnn_weight
        logger.info("Applied cnn_weight=%.3f", cnn_weight)
    if seg_weight != 1.0:
        seg_scaled = seg_scaled * seg_weight
        logger.info("Applied seg_weight=%.3f", seg_weight)

    # ------------------------------------------------------------------
    # Step 3: Concatenate
    # ------------------------------------------------------------------
    fused = np.concatenate([cnn_scaled, seg_scaled], axis=1)
    logger.info("Concatenated shape: (%d, %d)", fused.shape[0], fused.shape[1])

    # ------------------------------------------------------------------
    # Step 4: Optional PCA reduction
    # ------------------------------------------------------------------
    pca_model: Optional[PCA] = None
    pca_variance_explained: Optional[float] = None

    if apply_pca:
        if pca_components is not None:
            # Explicit integer component count (backward-compatible path)
            max_components = min(n_samples, fused.shape[1])
            if pca_components > max_components:
                logger.warning(
                    "Requested %d PCA components but only %d are possible "
                    "(min(n_samples=%d, n_features=%d)). Clamping.",
                    pca_components, max_components, n_samples, fused.shape[1],
                )
                pca_components = max_components
            pca_model = PCA(
                n_components=pca_components,
                random_state=random_state,
            )
        else:
            # Variance-based auto-selection: sklearn PCA accepts float in (0,1)
            pca_model = PCA(
                n_components=pca_variance,
                random_state=random_state,
            )

        fused = pca_model.fit_transform(fused)
        actual_components = pca_model.n_components_

        pca_variance_explained = float(
            np.sum(pca_model.explained_variance_ratio_)
        )
        logger.info(
            "PCA: %d -> %d components (%.2f%% variance explained)",
            cnn_dim + seg_dim,
            actual_components,
            pca_variance_explained * 100,
        )

        # Warn whenever the retained variance is materially below target,
        # regardless of whether components were chosen explicitly or via variance.
        if pca_variance_explained < (pca_variance - 0.02):
            logger.warning(
                "PCA retained only %.2f%% variance, below target %.2f%% "
                "(components=%d). Consider relaxing pca_variance or increasing pca_components.",
                pca_variance_explained * 100,
                pca_variance * 100,
                actual_components,
            )

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    fused = fused.astype(np.float32)

    info = {
        "method": method,
        "n_samples": n_samples,
        "n_features": fused.shape[1],
        "cnn_dim": cnn_dim,
        "seg_dim": seg_dim,
        "cnn_weight": cnn_weight,
        "seg_weight": seg_weight,
        "cnn_scaler": cnn_scaler,
        "seg_scaler": seg_scaler,
        "pca": pca_model,
        "pca_variance_explained": pca_variance_explained,
    }

    logger.info(
        "Fusion complete: (%d, %d) %s features",
        fused.shape[0], fused.shape[1], method,
    )
    return fused, info


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def transform_features(
    cnn_features: np.ndarray,
    seg_features: np.ndarray,
    info: dict,
) -> np.ndarray:
    """Apply a previously fitted fusion pipeline to new data.

    Use the scalers and PCA stored in *info* (returned by :func:`fuse_features`)
    so that inference data undergoes the exact same transformation as training
    data.

    Parameters
    ----------
    cnn_features : np.ndarray
        ``(M, D_cnn)`` new CNN features.
    seg_features : np.ndarray
        ``(M, D_seg)`` new segmentation features.
    info : dict
        The *info* dict returned by :func:`fuse_features`.

    Returns
    -------
    np.ndarray
        ``(M, F)`` float32 transformed features.
    """
    _validate_inputs(cnn_features, seg_features, info["method"])

    if cnn_features.shape[1] != info["cnn_dim"]:
        raise ValueError(
            f"CNN feature dimension mismatch: expected {info['cnn_dim']}, "
            f"got {cnn_features.shape[1]}"
        )
    if seg_features.shape[1] != info["seg_dim"]:
        raise ValueError(
            f"Seg feature dimension mismatch: expected {info['seg_dim']}, "
            f"got {seg_features.shape[1]}"
        )

    cnn_scaled = info["cnn_scaler"].transform(
        cnn_features.astype(np.float32, copy=False)
    )
    seg_scaled = info["seg_scaler"].transform(
        seg_features.astype(np.float32, copy=False)
    )

    if info["cnn_weight"] != 1.0:
        cnn_scaled = cnn_scaled * info["cnn_weight"]
    if info["seg_weight"] != 1.0:
        seg_scaled = seg_scaled * info["seg_weight"]

    fused = np.concatenate([cnn_scaled, seg_scaled], axis=1)

    if info["pca"] is not None:
        fused = info["pca"].transform(fused)

    return fused.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    cnn_features: np.ndarray,
    seg_features: np.ndarray,
    method: str,
) -> None:
    """Raise on invalid inputs."""
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown fusion method '{method}'. Choose from {_VALID_METHODS}."
        )

    if cnn_features.ndim != 2:
        raise ValueError(
            f"cnn_features must be 2-D, got shape {cnn_features.shape}"
        )
    if seg_features.ndim != 2:
        raise ValueError(
            f"seg_features must be 2-D, got shape {seg_features.shape}"
        )

    if cnn_features.shape[0] != seg_features.shape[0]:
        raise ValueError(
            f"Sample count mismatch: CNN has {cnn_features.shape[0]} rows, "
            f"segmentation has {seg_features.shape[0]} rows."
        )

    if cnn_features.shape[0] == 0:
        raise ValueError("Feature arrays are empty (0 samples).")

    # Warn about NaN
    cnn_nan = int(np.isnan(cnn_features).sum())
    seg_nan = int(np.isnan(seg_features).sum())
    if cnn_nan:
        logger.warning("CNN features contain %d NaN values", cnn_nan)
    if seg_nan:
        logger.warning("Segmentation features contain %d NaN values", seg_nan)

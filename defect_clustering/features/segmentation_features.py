"""Extract shape and texture features from binary defect segmentation masks.

This module provides functions to compute per-region shape descriptors,
Hu moment invariants, and GLCM texture features from binary or labeled
segmentation masks, with optional intensity images.  Results can be
aggregated across regions with configurable strategies (mean, mean+max).

Features per region (24 total)
------------------------------
* **Shape (10):** area, perimeter, eccentricity, solidity, extent,
  major_axis_length, minor_axis_length, orientation, equivalent_diameter,
  bbox_aspect_ratio.
* **Hu moments (7):** log-transformed Hu moment invariants.
* **GLCM texture (6):** contrast, dissimilarity, homogeneity, energy,
  correlation, ASM.
* **Compactness (1):** 4 * pi * area / perimeter**2.

References
----------
.. [1] Hu, M.-K. (1962). "Visual Pattern Recognition by Moment
   Invariants." *IRE Transactions on Information Theory*, 8(2), 179--187.
.. [2] Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). "Textural
   Features for Image Classification." *IEEE Transactions on Systems,
   Man, and Cybernetics*, SMC-3(6), 610--621.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, moments_hu, regionprops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHAPE_NAMES: List[str] = [
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "equivalent_diameter",
    "bbox_aspect_ratio",
]

_HU_NAMES: List[str] = [f"hu_moment_{i}" for i in range(7)]

_GLCM_NAMES: List[str] = [
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_ASM",
]

_COMPACTNESS_NAMES: List[str] = ["compactness"]

BASE_FEATURE_NAMES: List[str] = (
    _SHAPE_NAMES + _HU_NAMES + _GLCM_NAMES + _COMPACTNESS_NAMES
)

N_BASE_FEATURES: int = len(BASE_FEATURE_NAMES)  # 24

_VALID_AGGREGATIONS = {"mean", "mean_max"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return *numerator / denominator*, or 0.0 when *denominator* is zero."""
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _log_transform_hu(hu: np.ndarray) -> np.ndarray:
    """Apply sign-preserving log transform to Hu moments for stability.

    Hu moments span many orders of magnitude.  The transform
    ``-sign(h) * log10(|h|)`` compresses the range while preserving
    the sign, following common practice in the literature [1]_.

    Parameters
    ----------
    hu : np.ndarray, shape (7,)
        Raw Hu moment invariants.

    Returns
    -------
    np.ndarray, shape (7,)
        Log-transformed Hu moments.  Zero-valued moments map to 0.0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            hu != 0,
            -np.sign(hu) * np.log10(np.abs(hu)),
            0.0,
        )
    return result


def _compute_glcm_features(
    patch: np.ndarray,
) -> np.ndarray:
    """Compute 6 GLCM texture features from an 8-bit image patch.

    Parameters
    ----------
    patch : np.ndarray
        2-D image patch (uint8, values in [0, 255]).

    Returns
    -------
    np.ndarray, shape (6,)
        [contrast, dissimilarity, homogeneity, energy, correlation, ASM].
    """
    # Degenerate patches (single pixel, uniform) produce trivial GLCMs.
    if patch.size < 2 or patch.ndim != 2 or min(patch.shape) < 2:
        return np.zeros(6, dtype=np.float64)

    # Quantise to 64 grey levels to keep the GLCM tractable.
    levels = 64
    quantised = (patch / 256.0 * levels).astype(np.uint8)
    quantised = np.clip(quantised, 0, levels - 1)

    # Average over 4 orientations (0, 45, 90, 135 degrees).
    glcm = graycomatrix(
        quantised,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    props = []
    for prop_name in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]:
        vals = graycoprops(glcm, prop_name)  # shape (1, 4)
        props.append(np.nanmean(vals))

    return np.array(props, dtype=np.float64)


def _region_to_vector(
    region,
    intensity_image: Optional[np.ndarray],
) -> np.ndarray:
    """Convert a single ``regionprops`` region to a 24-element feature vector.

    Parameters
    ----------
    region : skimage.measure._regionprops.RegionProperties
        A region produced by ``skimage.measure.regionprops``.
    intensity_image : np.ndarray or None
        Optional greyscale image aligned with the label map.  If *None*,
        GLCM features are computed from the binary mask.

    Returns
    -------
    np.ndarray, shape (24,)
    """
    # -- Shape features (10) ------------------------------------------------
    area = float(region.area)
    perimeter = float(region.perimeter)
    eccentricity = float(region.eccentricity)
    solidity = float(region.solidity)
    extent = float(region.extent)
    major_axis = float(region.axis_major_length)
    minor_axis = float(region.axis_minor_length)
    orientation = float(region.orientation)
    equiv_diam = float(region.equivalent_diameter_area)

    min_row, min_col, max_row, max_col = region.bbox
    bbox_h = max_row - min_row
    bbox_w = max_col - min_col
    bbox_aspect = _safe_divide(float(bbox_h), float(bbox_w))

    shape_feats = np.array(
        [
            area,
            perimeter,
            eccentricity,
            solidity,
            extent,
            major_axis,
            minor_axis,
            orientation,
            equiv_diam,
            bbox_aspect,
        ],
        dtype=np.float64,
    )

    # -- Hu moments (7) -----------------------------------------------------
    hu_raw = moments_hu(region.moments_central)
    hu_feats = _log_transform_hu(hu_raw)

    # -- GLCM texture (6) ---------------------------------------------------
    if intensity_image is not None:
        patch = intensity_image[min_row:max_row, min_col:max_col].copy()
    else:
        # Fall back to the binary mask within the bounding box.
        patch = region.image.astype(np.float64) * 255.0

    # Ensure uint8 for GLCM computation.
    if patch.dtype != np.uint8:
        patch_min = patch.min()
        patch_max = patch.max()
        if patch_max - patch_min > 0:
            patch = (
                (patch - patch_min) / (patch_max - patch_min) * 255.0
            ).astype(np.uint8)
        else:
            patch = np.zeros_like(patch, dtype=np.uint8)

    glcm_feats = _compute_glcm_features(patch)

    # -- Compactness (1) ----------------------------------------------------
    compactness = _safe_divide(4.0 * math.pi * area, perimeter ** 2)

    return np.concatenate(
        [shape_feats, hu_feats, glcm_feats, np.array([compactness])]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_region_features(
    mask: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract features from every connected region in a segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary (bool, 0/1 or 0/255) or integer-labeled mask.  If binary,
        connected components (8-connectivity) are computed via
        ``skimage.measure.label``, one region each.  The values decide, not
        the dtype: a label image whose only label is 1 or 255 counts as
        binary.
    intensity_image : np.ndarray or None, optional
        Greyscale intensity image of the same spatial dimensions as *mask*.
        Used for GLCM texture computation.  When *None*, the binary mask is
        used instead.

    Returns
    -------
    np.ndarray, shape (n_regions, 24)
        Feature matrix with one row per region.  Returns an empty array with
        shape ``(0, 24)`` when the mask contains no foreground pixels.

    Raises
    ------
    ValueError
        If *mask* is not 2-D or if *intensity_image* dimensions do not match.

    Notes
    -----
    Shape features follow scikit-image ``regionprops`` conventions.  Hu
    moments are computed from ``moments_central`` and log-transformed for
    numerical stability [1]_.  GLCM features are averaged over four
    orientations at distance 1 following Haralick et al. [2]_.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-D, got {mask.ndim}-D array with shape {mask.shape}"
        )

    if intensity_image is not None:
        if intensity_image.ndim != 2:
            raise ValueError(
                f"intensity_image must be 2-D, got {intensity_image.ndim}-D"
            )
        if intensity_image.shape != mask.shape:
            raise ValueError(
                f"intensity_image shape {intensity_image.shape} does not match "
                f"mask shape {mask.shape}"
            )

    # Binary masks (bool, 0/1, or the 0/255 of OpenCV thresholds and the
    # synthetic mask drawers) are split into connected components; any other
    # integer mask is read as a label image.
    values = set(np.unique(mask).tolist())
    if mask.dtype == bool or values <= {0, 1} or values <= {0, 255}:
        labeled = label(mask > 0, connectivity=2)
    else:
        labeled = mask

    regions = regionprops(labeled)
    if len(regions) == 0:
        return np.empty((0, N_BASE_FEATURES), dtype=np.float64)

    feature_rows = []
    for region in regions:
        vec = _region_to_vector(region, intensity_image)
        feature_rows.append(vec)

    return np.stack(feature_rows, axis=0)


def extract_segmentation_features(
    mask: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    aggregation: str = "mean_max",
) -> np.ndarray:
    """Extract aggregated segmentation features for a single image.

    Computes per-region features via :func:`extract_region_features` and
    aggregates them across regions according to *aggregation*.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary or labeled mask.
    intensity_image : np.ndarray or None, optional
        Greyscale intensity image (same size as *mask*).
    aggregation : {"mean_max", "mean"}
        Aggregation strategy:

        * ``"mean_max"`` -- concatenation of the region-wise mean and max
          vectors, yielding 48 features.
        * ``"mean"`` -- region-wise mean only, yielding 24 features.

    Returns
    -------
    np.ndarray, shape (48,) or (24,)
        Aggregated feature vector.  A zero vector is returned when the mask
        contains no foreground regions.

    Raises
    ------
    ValueError
        If *aggregation* is not one of the supported strategies.
    """
    if aggregation not in _VALID_AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Choose from {sorted(_VALID_AGGREGATIONS)}."
        )

    region_feats = extract_region_features(mask, intensity_image)

    n_out = N_BASE_FEATURES * 2 if aggregation == "mean_max" else N_BASE_FEATURES

    if region_feats.shape[0] == 0:
        return np.zeros(n_out, dtype=np.float64)

    mean_vec = np.nanmean(region_feats, axis=0)

    if aggregation == "mean":
        return mean_vec

    # aggregation == "mean_max"
    max_vec = np.nanmax(region_feats, axis=0)
    return np.concatenate([mean_vec, max_vec])


def extract_batch(
    masks: List[np.ndarray],
    intensity_images: Optional[List[np.ndarray]] = None,
    aggregation: str = "mean_max",
    n_jobs: int = -1,
) -> np.ndarray:
    """Batch extraction of segmentation features for multiple images.

    Parameters
    ----------
    masks : list of np.ndarray
        List of 2-D binary or labeled masks.
    intensity_images : list of np.ndarray or None, optional
        Corresponding greyscale intensity images.  If provided, must be the
        same length as *masks*.  Pass *None* to skip intensity-based GLCM.
    aggregation : {"mean_max", "mean"}
        Aggregation strategy (see :func:`extract_segmentation_features`).
    n_jobs : int
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    np.ndarray, shape (N, 48) or (N, 24)
        Feature matrix with one row per image.

    Raises
    ------
    ValueError
        If *intensity_images* is provided and its length differs from *masks*.
    """
    from joblib import Parallel, delayed

    if intensity_images is not None and len(intensity_images) != len(masks):
        raise ValueError(
            f"Length mismatch: {len(masks)} masks vs "
            f"{len(intensity_images)} intensity images."
        )

    def _extract_one(idx):
        intensity = intensity_images[idx] if intensity_images is not None else None
        return extract_segmentation_features(masks[idx], intensity, aggregation)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_one)(i) for i in range(len(masks))
    )

    return np.stack(results, axis=0)


def feature_names(aggregation: str = "mean_max") -> List[str]:
    """Return the ordered list of feature names for a given aggregation.

    Parameters
    ----------
    aggregation : {"mean_max", "mean"}
        Aggregation strategy.

    Returns
    -------
    list of str
        Feature names.  Length 48 for ``"mean_max"``, 24 for ``"mean"``.

    Raises
    ------
    ValueError
        If *aggregation* is not one of the supported strategies.
    """
    if aggregation not in _VALID_AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Choose from {sorted(_VALID_AGGREGATIONS)}."
        )

    mean_names = [f"mean_{n}" for n in BASE_FEATURE_NAMES]

    if aggregation == "mean":
        return mean_names

    max_names = [f"max_{n}" for n in BASE_FEATURE_NAMES]
    return mean_names + max_names

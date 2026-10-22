"""Load and validate CNN feature vectors from pickle files.

Supports multiple serialization formats (dict, list-of-tuples, DataFrame)
and provides utilities for validation and L2 normalization.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



# Loading


def load_cnn_features(pickle_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load CNN features from a pickle file.

    Expected formats
    ----------------
    * ``Dict[str, np.ndarray]``  -- ``{image_id: feature_vector, ...}``
    * ``List[Tuple[str, np.ndarray]]`` -- ``[(image_id, feature_vector), ...]``
    * ``pandas.DataFrame`` with an ``image_id`` column and numeric feature columns

    Parameters
    ----------
    pickle_path : str
        Path to the ``.pkl`` file containing CNN features.

    Returns
    -------
    features : np.ndarray
        ``(N, D)`` float32 array where *D* is the feature dimension
        (960 for the inspection CNN embedding; 1280 for the ImageNet
        EfficientNet-B0 used on NEU-DET).
    image_ids : List[str]
        List of *N* image identifiers in the same row order as *features*.

    Raises
    ------
    FileNotFoundError
        If *pickle_path* does not exist.
    ValueError
        If the data format is unrecognised or features cannot be stacked.
    """
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    logger.info("Loading CNN features from %s", path)
    with open(path, "rb") as fh:
        raw = pickle.load(fh)

    features, image_ids = _parse_raw(raw)

    # Ensure float32 for downstream efficiency
    features = features.astype(np.float32, copy=False)

    logger.info(
        "Loaded %d feature vectors of dimension %d",
        features.shape[0],
        features.shape[1],
    )
    return features, image_ids


def _parse_raw(raw: Any) -> Tuple[np.ndarray, List[str]]:
    """Dispatch to the correct parser based on the object type."""

    # --- Dict[str, np.ndarray] ------------------------------------------------
    if isinstance(raw, dict):
        return _parse_dict(raw)

    # --- List / tuple of (id, vector) pairs -----------------------------------
    if isinstance(raw, (list, tuple)):
        return _parse_list_of_tuples(raw)

    # --- pandas DataFrame -----------------------------------------------------
    if isinstance(raw, pd.DataFrame):
        return _parse_dataframe(raw)

    raise ValueError(
        f"Unsupported data type in pickle: {type(raw).__name__}. "
        "Expected dict, list of tuples, or DataFrame."
    )


def _parse_dict(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    if not data:
        raise ValueError("Feature dictionary is empty.")

    image_ids = list(data.keys())
    try:
        features = np.stack([data[k] for k in image_ids])
    except ValueError as exc:
        raise ValueError(
            "Could not stack feature vectors -- inconsistent shapes."
        ) from exc

    image_ids = [str(k) for k in image_ids]
    return features, image_ids


def _parse_list_of_tuples(
    data: Union[List[Tuple[str, np.ndarray]], Tuple],
) -> Tuple[np.ndarray, List[str]]:
    if not data:
        raise ValueError("Feature list is empty.")

    # Validate structure: each element should be a 2-element sequence
    sample = data[0]
    if not (isinstance(sample, (list, tuple)) and len(sample) == 2):
        raise ValueError(
            "Expected list of (image_id, feature_vector) tuples, "
            f"but first element has type {type(sample).__name__} "
            f"with length {len(sample) if hasattr(sample, '__len__') else 'N/A'}."
        )

    image_ids = [str(item[0]) for item in data]
    try:
        features = np.stack([np.asarray(item[1]) for item in data])
    except ValueError as exc:
        raise ValueError(
            "Could not stack feature vectors -- inconsistent shapes."
        ) from exc

    return features, image_ids


def _parse_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if df.empty:
        raise ValueError("Feature DataFrame is empty.")

    if "image_id" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'image_id' column. "
            f"Found columns: {list(df.columns[:20])}"
        )

    image_ids = df["image_id"].astype(str).tolist()
    feature_cols = [c for c in df.columns if c != "image_id"]
    if not feature_cols:
        raise ValueError("DataFrame contains no feature columns besides 'image_id'.")

    features = df[feature_cols].values
    return features, image_ids



# Validation


def validate_features(features: np.ndarray) -> dict:
    """Validate a CNN feature matrix and return a diagnostics dictionary.

    Parameters
    ----------
    features : np.ndarray
        ``(N, D)`` feature array.

    Returns
    -------
    dict
        Keys: ``n_samples``, ``n_features``, ``dtype``, ``has_nan``,
        ``nan_count``, ``zero_vector_count``, ``is_normalized``,
        ``l2_norm_mean``, ``l2_norm_std``.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D array, got shape {features.shape}"
        )

    n_samples, n_features = features.shape
    nan_mask = np.isnan(features)
    nan_count = int(nan_mask.sum())
    has_nan = nan_count > 0

    if has_nan:
        logger.warning(
            "%d NaN values detected across %d / %d samples",
            nan_count,
            int(nan_mask.any(axis=1).sum()),
            n_samples,
        )

    # L2 norms (treat NaN rows as norm=0 for safety)
    norms = np.linalg.norm(
        np.nan_to_num(features, nan=0.0), axis=1
    )
    zero_vector_count = int((norms == 0.0).sum())
    if zero_vector_count:
        logger.warning(
            "%d zero-norm vectors detected (will cause issues in cosine similarity)",
            zero_vector_count,
        )

    # Considered "normalised" if all non-zero norms are within 1 +/- 1e-4
    non_zero_norms = norms[norms > 0]
    if non_zero_norms.size > 0:
        is_normalized = bool(np.allclose(non_zero_norms, 1.0, atol=1e-4))
    else:
        is_normalized = False

    if not is_normalized:
        logger.warning(
            "Features are NOT L2-normalised (mean norm=%.4f, std=%.4f). "
            "Consider calling normalize_features() before clustering.",
            float(norms.mean()),
            float(norms.std()),
        )

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "dtype": str(features.dtype),
        "has_nan": has_nan,
        "nan_count": nan_count,
        "zero_vector_count": zero_vector_count,
        "is_normalized": is_normalized,
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_std": float(norms.std()),
    }



# Normalisation


def normalize_features(features: np.ndarray) -> np.ndarray:
    """L2-normalise each row of the feature matrix in-place-safe fashion.

    Zero-norm rows are left as zeros and a warning is emitted.

    Parameters
    ----------
    features : np.ndarray
        ``(N, D)`` array.

    Returns
    -------
    np.ndarray
        ``(N, D)`` float32 array with unit-norm rows.
    """
    features = np.array(features, dtype=np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero_mask = (norms.squeeze() == 0.0)

    if zero_mask.any():
        logger.warning(
            "Cannot normalise %d zero-norm vectors; they will remain zeros.",
            int(zero_mask.sum()),
        )
        # Avoid division by zero: set zero norms to 1 (row stays all-zero)
        norms[norms == 0.0] = 1.0

    return features / norms

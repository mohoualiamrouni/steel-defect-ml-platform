"""I/O helpers for the SteelDefect platform.

Pure-Python / numpy utilities shared by the ingestion and pipeline jobs:
* convert a CNN embedding between a numpy array and a Spark ``ARRAY<FLOAT>``;
* decode a crop / mask referenced by an ADLS URI (the raw bytes live on
  the plant-private storage; only the URI travels in the OBT row).

No Spark import here — these run inside ``applyInPandas`` UDFs on the workers.
"""

from __future__ import annotations

import io
from typing import List, Optional

import numpy as np

from .obt_schema import CNN_DIM


def vector_to_array(vec: np.ndarray) -> List[float]:
    """Numpy 960-d embedding → plain python list[float] for an ARRAY<FLOAT>."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != CNN_DIM:
        raise ValueError(f"expected {CNN_DIM}-d CNN vector, got {arr.shape[0]}")
    return arr.tolist()


def array_to_vector(arr: Optional[List[float]]) -> Optional[np.ndarray]:
    """Spark ARRAY<FLOAT> (list) → numpy float32 array (None-safe)."""
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


def stack_cnn_vectors(arrays: List[Optional[List[float]]]) -> np.ndarray:
    """Stack a column of ARRAY<FLOAT> values into an ``(N, 960)`` matrix.

    Rows that are ``None`` are filled with zeros (and should be filtered by the
    caller via a ``cnn_vector IS NOT NULL`` predicate upstream).
    """
    out = np.zeros((len(arrays), CNN_DIM), dtype=np.float32)
    for i, a in enumerate(arrays):
        if a is not None:
            out[i] = np.asarray(a, dtype=np.float32)
    return out


def decode_mask_uri(uri: str, loader=None) -> Optional[np.ndarray]:
    """Decode a binary mask from an ADLS URI to a uint8 array.

    The actual byte fetch is delegated to ``loader(uri) -> bytes`` so this stays
    storage-agnostic and unit-testable (in production ``loader`` wraps the ADLS
    / dbutils client; locally it can read from disk).  Supports ``.npy`` and
    common image formats (PNG/JPG) via Pillow when available.
    """
    if loader is None:
        return None
    raw = loader(uri)
    if raw is None:
        return None
    if uri.endswith(".npy"):
        return np.load(io.BytesIO(raw), allow_pickle=False)
    try:
        from PIL import Image  # optional dependency
    except Exception:
        return None
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def to_date_key(event_ts) -> str:
    """Return the 'YYYY-MM-DD' partition key for a timestamp-like value."""
    # Accepts datetime, pandas Timestamp, or ISO string.
    s = str(event_ts)
    return s[:10]

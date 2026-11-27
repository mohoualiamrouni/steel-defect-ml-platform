"""Strict JSON for the A/B outputs: numpy types become Python values, NaN and infinities null.

``json.dumps`` writes NaN and Infinity, which strict JSON parsers reject; the CLI's JSON
files and the decision job's audit columns go through :func:`strict` first.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def strict(obj: Any) -> Any:
    """``obj`` with numpy scalars and arrays as Python values, non-finite floats as None,
    and mapping keys as strings."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return [strict(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict(v) for v in obj]
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps(strict(obj))``, refusing any NaN or infinity left."""
    return json.dumps(strict(obj), allow_nan=False, **kwargs)

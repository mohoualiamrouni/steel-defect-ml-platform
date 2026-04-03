"""Change-point detection on 1D quality profiles.

Wraps the *ruptures* library (Truong, Oudre & Vayer, 2020) to find
natural quality transitions along the length or width of a coil.
These change points serve as candidate cut positions for the
guillotine optimiser.
"""

from __future__ import annotations

from typing import List

import numpy as np
import ruptures


def detect_change_points(
    signal: np.ndarray,
    penalty: float = 1.0,
    model: str = "rbf",
    min_size: int = 2,
) -> List[int]:
    """Detect change points in a 1D signal using PELT.

    Parameters
    ----------
    signal : 1-D array
        Quality profile (e.g. mean severity per bin along one axis).
    penalty : float
        Penalty value for the PELT algorithm.  Higher values produce
        fewer change points.
    model : str
        Cost model for *ruptures* — ``"rbf"`` (default), ``"l2"``,
        ``"l1"``, or ``"normal"``.
    min_size : int
        Minimum segment length between change points (in bins).

    Returns
    -------
    list of int
        Change-point indices (exclusive end positions, as returned by
        *ruptures*).  The last element is always ``len(signal)`` and
        should typically be ignored.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    algo = ruptures.Pelt(model=model, min_size=min_size).fit(signal)
    change_points = algo.predict(pen=penalty)
    return change_points


def detect_length_change_points(
    quality_profile_1d: np.ndarray,
    penalty: float = 1.0,
    model: str = "rbf",
) -> List[int]:
    """Find quality transitions along the coil length.

    Parameters
    ----------
    quality_profile_1d : 1-D array
        Mean severity per length bin (obtained by averaging the quality
        heatmap along the width axis).
    penalty, model
        Forwarded to :func:`detect_change_points`.

    Returns
    -------
    list of int
        Change-point bin indices along the length axis.
    """
    return detect_change_points(quality_profile_1d, penalty=penalty, model=model)


def detect_width_change_points(
    quality_profile_1d: np.ndarray,
    penalty: float = 1.0,
    model: str = "rbf",
) -> List[int]:
    """Find quality transitions along the coil width.

    Typically produces fewer change points than the length direction
    since coils are much narrower than they are long.
    """
    return detect_change_points(quality_profile_1d, penalty=penalty, model=model)

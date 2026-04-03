"""Build 2D quality heatmaps from defect severity data.

Given a set of defects with (x, y) positions and severity scores on a
steel coil, this module creates a continuous 2D quality profile using
Gaussian kernel density estimation, then discretises it into quality
grades (A / B / C / scrap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def build_quality_heatmap(
    defects_df: pd.DataFrame,
    coil_length: float,
    coil_width: float,
    grid_shape: tuple[int, int] = (200, 40),
    kernel_bandwidth: float = 2.0,
    x_col: str = "x",
    y_col: str = "y",
    severity_col: str = "severity",
) -> np.ndarray:
    """Create a 2D severity heatmap on a regular grid.

    Each defect contributes its severity score to the nearest grid cell,
    then a Gaussian filter smooths the result to produce a continuous
    quality landscape.

    Parameters
    ----------
    defects_df : DataFrame
        Must contain columns for x position, y position, and severity.
    coil_length, coil_width : float
        Physical dimensions of the coil (metres).
    grid_shape : (int, int)
        (n_length_bins, n_width_bins) for the output grid.
    kernel_bandwidth : float
        Sigma of the Gaussian kernel in grid-cell units.
    severity_col : str
        Column containing predicted severity scores in [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape *grid_shape* with smoothed severity values.
    """
    n_rows, n_cols = grid_shape
    heatmap = np.zeros((n_rows, n_cols), dtype=np.float64)

    if len(defects_df) == 0:
        return heatmap

    # Map physical coordinates to grid indices
    xs = defects_df[x_col].values
    ys = defects_df[y_col].values
    sevs = defects_df[severity_col].values

    row_idx = np.clip(
        (xs / coil_length * n_rows).astype(int), 0, n_rows - 1
    )
    col_idx = np.clip(
        (ys / coil_width * n_cols).astype(int), 0, n_cols - 1
    )

    # Accumulate severity into grid cells
    np.add.at(heatmap, (row_idx, col_idx), sevs)

    # Smooth with Gaussian kernel
    if kernel_bandwidth > 0:
        heatmap = gaussian_filter(heatmap, sigma=kernel_bandwidth)

    # Normalise to [0, 1] so that quality thresholds are scale-invariant.
    # Without this, the Gaussian smoothing dilutes severity over thousands
    # of cells and the raw values stay far below any meaningful threshold.
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max

    return heatmap


def assign_quality_grades(
    heatmap: np.ndarray,
    thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> np.ndarray:
    """Discretise a severity heatmap into quality grades.

    Parameters
    ----------
    heatmap : np.ndarray
        Continuous severity values (output of :func:`build_quality_heatmap`).
    thresholds : tuple of float
        Sorted ascending boundaries between grades.  With 3 thresholds
        and 4 grades: severity < t0 → grade 0 (A), t0 ≤ sev < t1 → 1 (B),
        t1 ≤ sev < t2 → 2 (C), sev ≥ t2 → 3 (scrap).

    Returns
    -------
    np.ndarray
        Integer array (same shape as *heatmap*) with grade indices.
    """
    grades = np.zeros_like(heatmap, dtype=np.int32)
    for i, t in enumerate(thresholds):
        grades[heatmap >= t] = i + 1
    return grades

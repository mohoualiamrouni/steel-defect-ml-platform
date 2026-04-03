"""Coil-level spatial features derived from per-defect position data.

Each function accepts raw defect coordinates (in metres) together with the
physical dimensions of the coil and returns summary statistics that capture
where defects concentrate on the strip surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. Density map
# ---------------------------------------------------------------------------

def compute_density_map(
    defect_x: np.ndarray,
    defect_y: np.ndarray,
    coil_length: float,
    coil_width: float,
    grid_shape: tuple[int, int] = (100, 20),
) -> np.ndarray:
    """2-D histogram of defect counts normalised to defects / m².

    Parameters
    ----------
    defect_x : array of shape (n,)
        Longitudinal positions of defects (metres along coil length).
    defect_y : array of shape (n,)
        Transverse positions of defects (metres across coil width).
    coil_length : float
        Total coil length in metres.
    coil_width : float
        Total coil width in metres.
    grid_shape : (n_rows, n_cols)
        Number of bins along length and width respectively.

    Returns
    -------
    density : ndarray of shape *grid_shape*
        Each cell holds the defect density in defects / m² for that bin.
    """
    n_length_bins, n_width_bins = grid_shape

    counts, _, _ = np.histogram2d(
        defect_x,
        defect_y,
        bins=[n_length_bins, n_width_bins],
        range=[[0.0, coil_length], [0.0, coil_width]],
    )

    # Area of a single cell in m²
    cell_area = (coil_length / n_length_bins) * (coil_width / n_width_bins)

    density = counts / cell_area
    return density


# ---------------------------------------------------------------------------
# 2. Zone-level aggregated features
# ---------------------------------------------------------------------------

def compute_zone_features(
    defects_df: pd.DataFrame,
    coil_length: float,
    coil_width: float,
    n_length_zones: int = 5,
    n_width_zones: int = 3,
    severity_col: str = "severity_gt",
) -> np.ndarray:
    """Divide the coil surface into rectangular zones and compute per-zone stats.

    The DataFrame must contain columns ``"x"`` and ``"y"`` for defect
    positions (metres) and a severity column whose name is given by
    *severity_col*.

    Parameters
    ----------
    defects_df : DataFrame
        Must contain columns ``"x"``, ``"y"``, and *severity_col*.
    coil_length, coil_width : float
        Physical coil dimensions in metres.
    n_length_zones, n_width_zones : int
        Number of zones along each axis.
    severity_col : str
        Column name for the defect severity value.

    Returns
    -------
    features : ndarray of shape ``(n_length_zones * n_width_zones * 4,)``
        For every zone (row-major order): count, mean_severity,
        max_severity, density (defects / m²).  Zones with no defects
        receive 0 for all four statistics.
    """
    n_zones = n_length_zones * n_width_zones
    zone_length = coil_length / n_length_zones
    zone_width = coil_width / n_width_zones
    zone_area = zone_length * zone_width

    # Assign each defect to a zone index along each axis.
    length_idx = np.clip(
        (defects_df["x"].values / zone_length).astype(int),
        0,
        n_length_zones - 1,
    )
    width_idx = np.clip(
        (defects_df["y"].values / zone_width).astype(int),
        0,
        n_width_zones - 1,
    )
    zone_id = length_idx * n_width_zones + width_idx

    severity = defects_df[severity_col].values

    # Pre-allocate: 4 features per zone
    features = np.zeros(n_zones * 4, dtype=np.float64)

    for z in range(n_zones):
        mask = zone_id == z
        count = int(mask.sum())
        if count == 0:
            continue
        sev = severity[mask]
        offset = z * 4
        features[offset + 0] = count
        features[offset + 1] = sev.mean()
        features[offset + 2] = sev.max()
        features[offset + 3] = count / zone_area

    return features


# ---------------------------------------------------------------------------
# 3. Nearest-neighbour spatial statistics
# ---------------------------------------------------------------------------

def compute_spatial_stats(
    defect_x: np.ndarray,
    defect_y: np.ndarray,
    coil_length: float,
    coil_width: float,
) -> dict:
    """Nearest-neighbour distance statistics and Clark-Evans clustering ratio.

    Parameters
    ----------
    defect_x, defect_y : arrays of shape (n,)
        Defect positions in metres.
    coil_length, coil_width : float
        Physical coil dimensions in metres.

    Returns
    -------
    stats : dict with keys
        ``nn_mean``   – mean nearest-neighbour distance (m)
        ``nn_std``    – standard deviation of NN distances
        ``nn_min``    – minimum NN distance
        ``nn_max``    – maximum NN distance
        ``clark_evans_ratio`` – observed mean NN / expected mean NN under a
            homogeneous Poisson process.  Values < 1 indicate clustering,
            values > 1 indicate regularity.

    Notes
    -----
    When fewer than 2 defects are present every distance statistic is set to
    ``0.0`` and the Clark-Evans ratio to ``np.nan``.
    """
    n = len(defect_x)

    if n < 2:
        return {
            "nn_mean": 0.0,
            "nn_std": 0.0,
            "nn_min": 0.0,
            "nn_max": 0.0,
            "clark_evans_ratio": np.nan,
        }

    coords = np.column_stack([defect_x, defect_y])
    tree = cKDTree(coords)

    # k=2 because the closest point to each query is the point itself (dist 0)
    distances, _ = tree.query(coords, k=2)
    nn_distances = distances[:, 1]

    observed_mean = nn_distances.mean()

    # Expected mean NN distance for a 2-D Poisson process:
    #   E[r] = 1 / (2 * sqrt(intensity))
    area = coil_length * coil_width
    intensity = n / area  # points per m²
    expected_mean = 1.0 / (2.0 * np.sqrt(intensity))

    clark_evans = observed_mean / expected_mean

    return {
        "nn_mean": float(observed_mean),
        "nn_std": float(nn_distances.std()),
        "nn_min": float(nn_distances.min()),
        "nn_max": float(nn_distances.max()),
        "clark_evans_ratio": float(clark_evans),
    }

"""Generate synthetic steel coil data with defects placed at (x, y) positions.

Each coil has random dimensions and a variable number of defects scattered
across its surface.  Defect masks are generated via the existing
``defect_clustering`` helpers, and 24 segmentation features are extracted
per defect.  Ground-truth severity labels are derived from defect type and
area.

Three CSV files are written:

* ``coils_metadata.csv`` -- one row per coil.
* ``defects.csv`` -- one row per defect with position, type, severity, and
  24 shape/texture features.
* ``customer_requirements.csv`` -- fictional quality constraints per grade.

Usage::

    python -m datamining.data.generate_synthetic_coils

Or from Python::

    from datamining.data.generate_synthetic_coils import generate_synthetic_coils
    info = generate_synthetic_coils()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from defect_clustering.data.generate_synthetic_data import (
    EXPERT_GROUPS,
    _GROUP_SHAPE_FAMILY,
    _generate_mask,
)
from defect_clustering.features.segmentation_features import (
    BASE_FEATURE_NAMES,
    extract_region_features,
)

# ---------------------------------------------------------------------------
# Spatial placement patterns
# ---------------------------------------------------------------------------

# Which expert groups use which spatial pattern on the coil surface.
_SPATIAL_PATTERN: dict[str, str] = {
    "edge_crack": "edge",
    "camber": "edge",
    "roll_mark": "periodic",
    "corrugation": "periodic",
    "pit": "clustered",
    "inclusion": "clustered",
}
# Everything not listed above defaults to "uniform".

# ---------------------------------------------------------------------------
# Base severity per expert group (higher = more critical)
# ---------------------------------------------------------------------------

_BASE_SEVERITY: dict[str, float] = {
    "scratch": 0.4,
    "pit": 0.3,
    "roll_mark": 0.5,
    "edge_crack": 0.8,
    "inclusion": 0.35,
    "scale": 0.45,
    "streak": 0.3,
    "sliver": 0.55,
    "orange_peel": 0.25,
    "coil_break": 0.7,
    "dent": 0.5,
    "stain": 0.2,
    "weld_defect": 0.6,
    "lamination": 0.65,
    "gouge": 0.6,
    "buckle": 0.55,
    "pinhole": 0.4,
    "heat_tint": 0.3,
    "emulsion_residue": 0.2,
    "zinc_defect": 0.45,
    "bruise": 0.35,
    "coating_defect": 0.5,
    "impression": 0.3,
    "corrugation": 0.45,
    "shell": 0.5,
    "roughness": 0.25,
    "camber": 0.6,
    "contamination": 0.35,
}

# ---------------------------------------------------------------------------
# Defect placement helpers
# ---------------------------------------------------------------------------


def _place_uniform(
    rng: np.random.Generator,
    n: int,
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place defects uniformly at random across the coil surface."""
    xs = rng.uniform(0.0, length, size=n)
    ys = rng.uniform(0.0, width, size=n)
    return xs, ys


def _place_edge(
    rng: np.random.Generator,
    n: int,
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place defects concentrated near the coil edges (y ~ 0 or y ~ width)."""
    xs = rng.uniform(0.0, length, size=n)
    # Beta distribution skewed toward 0; randomly flip some to far edge
    ys_raw = rng.beta(0.5, 5.0, size=n) * (width / 2.0)
    flip = rng.random(size=n) > 0.5
    ys = np.where(flip, width - ys_raw, ys_raw)
    return xs, ys


def _place_periodic(
    rng: np.random.Generator,
    n: int,
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place defects at roughly periodic intervals along the coil length."""
    period = length / max(n, 1)
    xs = np.arange(n) * period + rng.normal(0.0, period * 0.1, size=n)
    xs = np.clip(xs, 0.0, length)
    ys = rng.uniform(0.0, width, size=n)
    return xs, ys


def _place_clustered(
    rng: np.random.Generator,
    n: int,
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place defects in Gaussian clusters on the coil surface."""
    n_clusters = rng.integers(2, max(3, n // 10 + 1))
    cx = rng.uniform(0.0, length, size=n_clusters)
    cy = rng.uniform(0.0, width, size=n_clusters)
    sigma_x = length * 0.05
    sigma_y = width * 0.15

    assignments = rng.integers(0, n_clusters, size=n)
    xs = cx[assignments] + rng.normal(0.0, sigma_x, size=n)
    ys = cy[assignments] + rng.normal(0.0, sigma_y, size=n)
    xs = np.clip(xs, 0.0, length)
    ys = np.clip(ys, 0.0, width)
    return xs, ys


_PLACEMENT_FN = {
    "uniform": _place_uniform,
    "edge": _place_edge,
    "periodic": _place_periodic,
    "clustered": _place_clustered,
}

# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------


def _features_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return a single 24-element feature vector from a binary mask.

    If the mask contains multiple regions, features are averaged across
    regions.  If the mask is empty, a zero vector is returned.
    """
    region_feats = extract_region_features(mask)
    if region_feats.shape[0] == 0:
        return np.zeros(24, dtype=np.float64)
    return np.nanmean(region_feats, axis=0)


def _inject_scrap_band(
    rng: np.random.Generator,
    coil_defects: list,
    length: float,
    band_len_range: tuple,
    band_sev_range: tuple,
) -> None:
    """Raise the severity of defects in a random length band to a high value.

    Creates a contiguous, multi-metre scrap region wider than the minimum
    piece length, so the guillotine optimiser cannot sub-cut around it.  If the
    band is sparse, the two defects nearest its centre are promoted, ensuring a
    non-empty scrap zone.  This is what makes scrap non-zero and
    physically-meaningful (vs. the flat generator's unrealistic "0 rebut").
    """
    band_len = float(rng.uniform(*band_len_range))
    x0 = float(rng.uniform(0.0, max(1e-6, length - band_len)))
    x1 = x0 + band_len
    in_band = [d for d in coil_defects if x0 <= d["x"] <= x1]
    if len(in_band) < 2:
        centre = (x0 + x1) / 2.0
        in_band = sorted(coil_defects, key=lambda d: abs(d["x"] - centre))[:2]
    for d in in_band:
        d["severity_gt"] = round(float(rng.uniform(*band_sev_range)), 4)


# ---------------------------------------------------------------------------
# Customer requirements generation
# ---------------------------------------------------------------------------


def _generate_customer_requirements(rng: np.random.Generator) -> pd.DataFrame:
    """Create a small table of fictional quality grades and constraints."""
    grades = [
        {
            "grade": "premium_automotive",
            "max_mean_severity": 0.10,
            "max_defect_density": 0.5,
            "value_per_m2": 120.0,
        },
        {
            "grade": "standard_automotive",
            "max_mean_severity": 0.25,
            "max_defect_density": 2.0,
            "value_per_m2": 85.0,
        },
        {
            "grade": "appliance",
            "max_mean_severity": 0.40,
            "max_defect_density": 5.0,
            "value_per_m2": 55.0,
        },
        {
            "grade": "construction",
            "max_mean_severity": 0.60,
            "max_defect_density": 15.0,
            "value_per_m2": 30.0,
        },
        {
            "grade": "general_purpose",
            "max_mean_severity": 0.80,
            "max_defect_density": 50.0,
            "value_per_m2": 18.0,
        },
    ]
    return pd.DataFrame(grades)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_synthetic_coils(
    output_dir: str = "datamining/data/synthetic",
    n_coils: int = 50,
    coil_length_range: tuple[float, float] = (500.0, 1500.0),
    coil_width_range: tuple[float, float] = (1.0, 2.0),
    defects_per_coil_range: tuple[int, int] = (20, 200),
    mask_size: int = 64,
    random_state: int = 42,
    p_bad_coil: float = 0.15,
    bad_band_len_range: tuple[float, float] = (8.0, 25.0),
    bad_band_sev_range: tuple[float, float] = (0.65, 0.85),
) -> dict:
    """Generate synthetic steel coil data with spatially placed defects.

    Parameters
    ----------
    output_dir:
        Directory where the three CSV files are written.
    n_coils:
        Number of coils to generate.
    coil_length_range:
        Min and max coil length in metres.
    coil_width_range:
        Min and max coil width in metres.
    defects_per_coil_range:
        Min and max number of defects per coil.
    mask_size:
        Height/width of binary segmentation masks.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``'coils_csv'``, ``'defects_csv'``, ``'requirements_csv'``,
        ``'n_coils'``, ``'n_defects_total'``.
    """
    rng = np.random.default_rng(random_state)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Flatten group -> subtypes for sampling
    group_names = list(EXPERT_GROUPS.keys())
    all_subtypes: list[tuple[str, str]] = []  # (group, subtype)
    for g in group_names:
        for st in EXPERT_GROUPS[g]:
            all_subtypes.append((g, st))

    coil_records: list[dict[str, Any]] = []
    defect_records: list[dict[str, Any]] = []
    defect_counter = 0

    for coil_idx in range(n_coils):
        coil_id = f"coil_{coil_idx:04d}"
        length = rng.uniform(*coil_length_range)
        width = rng.uniform(*coil_width_range)
        n_defects = int(rng.integers(*defects_per_coil_range))

        coil_records.append(
            {
                "coil_id": coil_id,
                "length": round(length, 2),
                "width": round(width, 3),
                "n_defects": n_defects,
            }
        )

        # Sample defect types for this coil (with replacement)
        type_indices = rng.integers(0, len(all_subtypes), size=n_defects)

        # Group defects by spatial pattern so positions can be generated
        # in bulk per pattern.
        pattern_groups: dict[str, list[int]] = {}
        for local_i, ti in enumerate(type_indices):
            group, _subtype = all_subtypes[ti]
            pattern = _SPATIAL_PATTERN.get(group, "uniform")
            pattern_groups.setdefault(pattern, []).append(local_i)

        # Pre-allocate position arrays for the coil
        xs = np.empty(n_defects, dtype=np.float64)
        ys = np.empty(n_defects, dtype=np.float64)

        for pattern, local_indices in pattern_groups.items():
            n_pat = len(local_indices)
            px, py = _PLACEMENT_FN[pattern](rng, n_pat, length, width)
            for k, li in enumerate(local_indices):
                xs[li] = px[k]
                ys[li] = py[k]

        # Generate mask, features, and severity for each defect
        coil_defects: list[dict[str, Any]] = []
        for local_i in range(n_defects):
            ti = type_indices[local_i]
            group, subtype = all_subtypes[ti]
            defect_id = f"defect_{defect_counter:07d}"

            # Generate mask and extract 24 features
            mask = _generate_mask(group, rng, mask_size)
            features = _features_from_mask(mask)

            # Derive ground-truth severity
            # The formula uses the full base severity (up to 0.8) scaled by
            # a wider area multiplier so that the resulting distribution
            # spans [0, 1] with meaningful variance across grades.
            base_sev = _BASE_SEVERITY.get(group, 0.4)
            mask_area = np.count_nonzero(mask)
            max_area = mask_size * mask_size
            normalized_area = mask_area / max_area if max_area > 0 else 0.0
            # Noise scale calibrated against the signal: with base_sev * mult
            # contributing ~0.09 std to severity, sigma=0.04 keeps the
            # signal-to-noise ratio comfortably above one and lets a
            # well-specified regressor reach R^2 ~ 0.5 (was 0.12 originally,
            # which capped achievable R^2 at ~0.05).
            noise = rng.normal(0.0, 0.04)
            severity_gt = base_sev * (0.5 + 0.8 * normalized_area) + noise
            # Occasionally inject high-severity outliers (~5% of defects).
            if rng.random() < 0.05:
                severity_gt += rng.uniform(0.10, 0.25)
            severity_gt = float(np.clip(severity_gt, 0.0, 1.0))

            row: dict[str, Any] = {
                "coil_id": coil_id,
                "defect_id": defect_id,
                "x": round(float(xs[local_i]), 4),
                "y": round(float(ys[local_i]), 4),
                "defect_type": subtype,
                "expert_group": group,
                "severity_gt": round(severity_gt, 4),
            }
            # Append the 24 feature columns
            for fname, fval in zip(BASE_FEATURE_NAMES, features):
                row[fname] = fval

            coil_defects.append(row)
            defect_counter += 1

        # With probability p_bad_coil, inject a contiguous high-severity band
        # so the cutting optimiser cannot sub-cut around it -> guaranteed,
        # physically-meaningful scrap (replaces the unrealistic "0 rebut").
        if rng.random() < p_bad_coil and coil_defects:
            _inject_scrap_band(rng, coil_defects, length,
                               bad_band_len_range, bad_band_sev_range)
        defect_records.extend(coil_defects)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    coils_csv = out / "coils_metadata.csv"
    defects_csv = out / "defects.csv"
    requirements_csv = out / "customer_requirements.csv"

    pd.DataFrame(coil_records).to_csv(str(coils_csv), index=False)
    pd.DataFrame(defect_records).to_csv(str(defects_csv), index=False)
    _generate_customer_requirements(rng).to_csv(
        str(requirements_csv), index=False
    )

    return {
        "coils_csv": str(coils_csv),
        "defects_csv": str(defects_csv),
        "requirements_csv": str(requirements_csv),
        "n_coils": n_coils,
        "n_defects_total": defect_counter,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _print_summary(info: dict) -> None:
    """Print a human-readable generation summary."""
    print("=" * 64)
    print("  Synthetic Coil Data -- Generation Summary")
    print("=" * 64)
    print(f"  Coils generated       : {info['n_coils']}")
    print(f"  Total defects         : {info['n_defects_total']:,}")
    print(f"  Coils metadata CSV    : {info['coils_csv']}")
    print(f"  Defects CSV           : {info['defects_csv']}")
    print(f"  Customer requirements : {info['requirements_csv']}")
    print("=" * 64)

    coils_df = pd.read_csv(info["coils_csv"])
    defects_df = pd.read_csv(info["defects_csv"])

    print(f"\n  Coil length range     : "
          f"{coils_df['length'].min():.1f} -- {coils_df['length'].max():.1f} m")
    print(f"  Coil width range      : "
          f"{coils_df['width'].min():.2f} -- {coils_df['width'].max():.2f} m")
    print(f"  Defects per coil      : "
          f"{coils_df['n_defects'].min()} -- {coils_df['n_defects'].max()}")

    print("\n  Defects per expert group:")
    group_counts = defects_df["expert_group"].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"    {group:<25s} {count:>6,}")

    print("\n  Spatial patterns used:")
    for group in sorted(set(defects_df["expert_group"])):
        pattern = _SPATIAL_PATTERN.get(group, "uniform")
        print(f"    {group:<25s} {pattern}")

    print("=" * 64)


if __name__ == "__main__":
    info = generate_synthetic_coils()
    _print_summary(info)

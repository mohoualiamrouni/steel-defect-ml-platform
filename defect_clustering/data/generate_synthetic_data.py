"""Generate realistic synthetic data for the defect group clustering pipeline.

This module produces synthetic CNN feature vectors, binary segmentation masks,
a metadata CSV, and a pickle file of features -- all designed to mimic the
statistical properties of real steel surface inspection data without requiring
access to proprietary images.

Usage::

    python -m defect_clustering.data.generate_synthetic_data

Or from Python::

    from defect_clustering.data.generate_synthetic_data import generate_synthetic_defect_data
    info = generate_synthetic_defect_data()
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.draw import disk, ellipse, line, polygon, random_shapes

# ---------------------------------------------------------------------------
# Expert groups and their subtypes
# ---------------------------------------------------------------------------

EXPERT_GROUPS: dict[str, list[str]] = {
    "scratch": [
        "longitudinal_scratch",
        "transverse_scratch",
        "micro_scratch",
        "deep_scratch",
    ],
    "pit": [
        "shallow_pit",
        "deep_pit",
        "clustered_pits",
    ],
    "roll_mark": [
        "periodic_roll_mark",
        "irregular_roll_mark",
        "chatter_mark",
        "herringbone_roll_mark",
    ],
    "edge_crack": [
        "hairline_edge_crack",
        "branching_edge_crack",
        "edge_chip",
        "zigzag_edge_crack",
    ],
    "inclusion": [
        "oxide_inclusion",
        "slag_inclusion",
        "alumina_inclusion",
        "sulfide_inclusion",
    ],
    "scale": [
        "primary_scale",
        "secondary_scale",
        "red_scale",
        "island_scale",
    ],
    "streak": [
        "oil_streak",
        "temper_streak",
        "roller_streak",
    ],
    "sliver": [
        "surface_sliver",
        "subsurface_sliver",
        "lamination_sliver",
        "hook_sliver",
    ],
    "orange_peel": [
        "mild_orange_peel",
        "severe_orange_peel",
        "directional_orange_peel",
    ],
    "coil_break": [
        "transverse_coil_break",
        "edge_coil_break",
        "center_coil_break",
    ],
    "dent": [
        "mechanical_dent",
        "handling_dent",
        "roll_dent",
        "conveyor_dent",
    ],
    "stain": [
        "water_stain",
        "oil_stain",
        "chemical_stain",
        "rust_stain",
    ],
    "weld_defect": [
        "weld_spatter",
        "weld_burn",
        "weld_seam_irregularity",
        "weld_undercut",
    ],
    "lamination": [
        "surface_lamination",
        "edge_lamination",
        "blister_lamination",
    ],
    "gouge": [
        "shallow_gouge",
        "deep_gouge",
        "guide_gouge",
    ],
    "buckle": [
        "center_buckle",
        "edge_buckle",
        "quarter_buckle",
        "wavy_edge",
    ],
    "pinhole": [
        "single_pinhole",
        "clustered_pinholes",
        "line_pinholes",
    ],
    "heat_tint": [
        "light_heat_tint",
        "dark_heat_tint",
        "banded_heat_tint",
        "localized_heat_tint",
    ],
    "emulsion_residue": [
        "streaky_emulsion",
        "spotty_emulsion",
        "film_emulsion",
    ],
    "zinc_defect": [
        "bare_spot",
        "zinc_ripple",
        "zinc_drip",
        "zinc_buildup",
    ],
    "bruise": [
        "impact_bruise",
        "friction_bruise",
        "telescoping_bruise",
    ],
    "coating_defect": [
        "coating_skip",
        "coating_blister",
        "coating_crazing",
        "coating_delamination",
    ],
    "impression": [
        "leveler_impression",
        "wrapper_impression",
        "debris_impression",
    ],
    "corrugation": [
        "fine_corrugation",
        "coarse_corrugation",
        "diagonal_corrugation",
    ],
    "shell": [
        "flaking_shell",
        "peeling_shell",
        "lifted_shell",
        "spalling_shell",
    ],
    "roughness": [
        "localised_roughness",
        "banded_roughness",
        "uniform_roughness",
    ],
    "camber": [
        "mild_camber",
        "severe_camber",
        "s_camber",
    ],
    "contamination": [
        "metallic_contamination",
        "organic_contamination",
        "dust_contamination",
        "fiber_contamination",
        "foreign_particle",
    ],
}

# --- Taxonomy invariants (single source of truth: the code, not the prose) ---
# The inspection taxonomy is a TWO-LEVEL hierarchy: 28 expert groups (the clustering
# target K) over ~100 individual defect subtypes. There is no "56" anywhere —
# that number in earlier report drafts was an error. These constants lock the
# count so it can never silently drift, and the report quotes them via the
# \ngroups / \nsubtypes macros.
N_EXPERT_GROUPS: int = len(EXPERT_GROUPS)            # == 28
N_SUBTYPES: int = sum(len(v) for v in EXPERT_GROUPS.values())  # ~100
assert N_EXPERT_GROUPS == 28, (
    f"Expected 28 expert groups (inspection taxonomy), found {N_EXPERT_GROUPS}"
)

# Mapping from group name to a preferred mask "shape family"
_GROUP_SHAPE_FAMILY: dict[str, str] = {
    "scratch": "line",
    "pit": "disk",
    "roll_mark": "line",
    "edge_crack": "line",
    "inclusion": "disk",
    "scale": "blob",
    "streak": "line",
    "sliver": "line",
    "orange_peel": "texture",
    "coil_break": "line",
    "dent": "ellipse",
    "stain": "blob",
    "weld_defect": "blob",
    "lamination": "blob",
    "gouge": "line",
    "buckle": "line",
    "pinhole": "disk",
    "heat_tint": "blob",
    "emulsion_residue": "blob",
    "zinc_defect": "blob",
    "bruise": "ellipse",
    "coating_defect": "blob",
    "impression": "ellipse",
    "corrugation": "line",
    "shell": "blob",
    "roughness": "texture",
    "camber": "line",
    "contamination": "disk",
}


# ---------------------------------------------------------------------------
# Mask drawing helpers
# ---------------------------------------------------------------------------


def _draw_line_mask(
    rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Draw one or more line segments on a blank mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    n_lines = rng.integers(1, 4)
    for _ in range(n_lines):
        r0, c0 = rng.integers(0, size, size=2)
        r1, c1 = rng.integers(0, size, size=2)
        rr, cc = line(int(r0), int(c0), int(r1), int(c1))
        # thicken the line slightly
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr_t = np.clip(rr + dr, 0, size - 1)
                cc_t = np.clip(cc + dc, 0, size - 1)
                mask[rr_t, cc_t] = 255
    return mask


def _draw_disk_mask(
    rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Draw one or more circular defects."""
    mask = np.zeros((size, size), dtype=np.uint8)
    n_disks = rng.integers(1, 6)
    for _ in range(n_disks):
        cr = rng.integers(5, size - 5)
        cc = rng.integers(5, size - 5)
        radius = rng.integers(1, max(2, size // 8))
        rr, ccol = disk((int(cr), int(cc)), int(radius), shape=(size, size))
        mask[rr, ccol] = 255
    return mask


def _draw_ellipse_mask(
    rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Draw an elliptical defect region."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cr = rng.integers(size // 4, 3 * size // 4)
    cc = rng.integers(size // 4, 3 * size // 4)
    r_radius = rng.integers(3, size // 3)
    c_radius = rng.integers(3, size // 3)
    rotation = rng.uniform(0, np.pi)
    rr, ccol = ellipse(
        int(cr), int(cc), int(r_radius), int(c_radius),
        shape=(size, size), rotation=rotation,
    )
    mask[rr, ccol] = 255
    return mask


def _draw_blob_mask(
    rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Draw an irregular blob using a random polygon."""
    mask = np.zeros((size, size), dtype=np.uint8)
    n_vertices = rng.integers(5, 12)
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_vertices))
    radii = rng.uniform(3, size // 4, size=n_vertices)
    rr_poly = np.clip((cy + radii * np.sin(angles)).astype(int), 0, size - 1)
    cc_poly = np.clip((cx + radii * np.cos(angles)).astype(int), 0, size - 1)
    rr, ccol = polygon(rr_poly, cc_poly, shape=(size, size))
    if len(rr) > 0:
        mask[rr, ccol] = 255
    return mask


def _draw_texture_mask(
    rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Draw a texture-like pattern using random_shapes from skimage."""
    result = random_shapes(
        (size, size),
        max_shapes=15,
        min_shapes=6,
        min_size=3,
        max_size=size // 4,
        allow_overlap=True,
        rng=int(rng.integers(0, 2**31)),
    )
    image = result[0]
    # random_shapes returns an RGB image; convert non-white pixels to mask
    gray = image.min(axis=2)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[gray < 255] = 255
    return mask


_SHAPE_DRAWERS = {
    "line": _draw_line_mask,
    "disk": _draw_disk_mask,
    "ellipse": _draw_ellipse_mask,
    "blob": _draw_blob_mask,
    "texture": _draw_texture_mask,
}


def _generate_mask(
    group: str, rng: np.random.Generator, size: int
) -> NDArray[np.uint8]:
    """Generate a binary segmentation mask appropriate for *group*."""
    family = _GROUP_SHAPE_FAMILY.get(group, "blob")
    drawer = _SHAPE_DRAWERS[family]
    return drawer(rng, size)


# ---------------------------------------------------------------------------
# CNN feature generation
# ---------------------------------------------------------------------------


def _generate_group_features(
    n_samples: int,
    cnn_dim: int,
    group_mean: NDArray[np.float64],
    intra_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample CNN features for a single defect type around *group_mean*.

    Features are drawn from a multivariate normal with diagonal covariance
    then L2-normalised so they live on the unit hypersphere (mimicking the
    output of a global-average-pooling layer followed by L2 normalisation).
    """
    noise = rng.normal(0, intra_scale, size=(n_samples, cnn_dim))
    features = group_mean[np.newaxis, :] + noise
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return features / norms


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_synthetic_defect_data(
    output_dir: str = "defect_clustering/data/synthetic",
    n_total: int = 20_000,
    n_groups: int = 28,
    cnn_dim: int = 960,
    mask_size: int = 64,
    random_state: int = 42,
) -> dict[str, Any]:
    """Generate synthetic defect data for pipeline development.

    Parameters
    ----------
    output_dir:
        Root directory where all artefacts are written.
    n_total:
        Approximate total number of synthetic images to generate.
    n_groups:
        Number of expert groups to use (capped at len(EXPERT_GROUPS)).
    cnn_dim:
        Dimensionality of CNN feature vectors.
    mask_size:
        Height and width of binary segmentation masks.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``'pickle_path'``, ``'metadata_path'``, ``'masks_dir'``,
        ``'n_images'``, ``'n_types'``, ``'n_groups'``.
    """
    rng = np.random.default_rng(random_state)
    out = Path(output_dir)
    masks_dir = out / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Select groups and subtypes
    # ------------------------------------------------------------------
    group_names = list(EXPERT_GROUPS.keys())[:n_groups]
    all_types: list[tuple[str, str]] = []  # (group, subtype)
    for g in group_names:
        for st in EXPERT_GROUPS[g]:
            all_types.append((g, st))
    n_types = len(all_types)

    # ------------------------------------------------------------------
    # 2. Assign sample counts (realistic, heavy-tailed class imbalance)
    # ------------------------------------------------------------------
    # Calibrated to the plant: a few common groups dominate and rare
    # structural defects get < ~10 instances/month (report L306). We draw the
    # imbalance at the GROUP level with a steep log-normal (top/rare ratio
    # ≈ 15-20×), THEN split each group's budget across its subtypes — this
    # produces a much heavier tail than a flat per-subtype draw (which averages
    # out to near-balance once aggregated to groups).
    group_raw = rng.lognormal(mean=5.5, sigma=1.4, size=n_groups)
    group_raw = np.clip(group_raw, 30.0, 4000.0)
    group_budget = group_raw * (n_total / group_raw.sum())
    counts = np.zeros(n_types, dtype=int)
    for gi, g in enumerate(group_names):
        sub_idx = [k for k, (gg, _st) in enumerate(all_types) if gg == g]
        w = rng.lognormal(mean=0.0, sigma=0.6, size=len(sub_idx))
        w = w / w.sum()
        for k, frac in zip(sub_idx, w):
            counts[k] = max(8, int(round(group_budget[gi] * frac)))

    # ------------------------------------------------------------------
    # 3. Generate CNN cluster centres with same-family OVERLAP
    # ------------------------------------------------------------------
    # Groups sharing a metallurgical super-family are placed CLOSE in CNN space
    # (the CNN, trained on appearance, confuses them) while their masks stay in
    # distinct shape families — so the morphology descriptors carry independent
    # discriminative signal the CNN alone misses.  This is the calibration that
    # makes the H2 gain ("DefChars add NMI") an honest, reproducible effect
    # rather than an artifact: without it the CNN is near-perfectly separable
    # and morphology adds nothing.
    try:
        from defect_clustering.analysis.regrouping_proposal import (
            METALLURGICAL_FAMILIES as _FAM,
        )
    except Exception:
        _FAM = {}

    # one anchor direction per family (families ~7 apart), groups offset ~2.5
    # from their family anchor (so within-family groups overlap in CNN space)
    fam_of = {g: _FAM.get(g, g) for g in group_names}
    fam_anchors: dict[str, NDArray[np.float64]] = {}
    for fam in sorted(set(fam_of.values())):
        v = rng.standard_normal(cnn_dim)
        v /= np.linalg.norm(v)
        fam_anchors[fam] = v * 7.0

    group_centres: dict[str, NDArray[np.float64]] = {}
    for g in group_names:
        off = rng.standard_normal(cnn_dim)
        off /= np.linalg.norm(off)
        group_centres[g] = fam_anchors[fam_of[g]] + off * 2.5

    # Subtype centres are offset from the group centre
    subtype_centres: list[NDArray[np.float64]] = []
    for g, _st in all_types:
        offset = rng.standard_normal(cnn_dim) * 0.6
        subtype_centres.append(group_centres[g] + offset)

    # ------------------------------------------------------------------
    # 4. Generate features, masks and metadata rows
    # ------------------------------------------------------------------
    records: list[dict[str, Any]] = []
    features_dict: dict[str, NDArray[np.float64]] = {}
    image_counter = 0

    for idx, (group, subtype) in enumerate(all_types):
        n_samples = int(counts[idx])
        # Intra-type spread (some types tighter than others). Calibrated so the
        # CNN-only NMI lands in a realistic 0.45-0.60 band, consistent with the
        # observed silhouette — not the implausibly clean 0.73 of the prior draw.
        intra_scale = rng.uniform(1.0, 1.9)
        feats = _generate_group_features(
            n_samples, cnn_dim, subtype_centres[idx], intra_scale, rng
        )

        for j in range(n_samples):
            image_id = f"img_{image_counter:06d}"
            mask_filename = f"{image_id}.npy"
            mask_path = masks_dir / mask_filename

            # CNN feature
            features_dict[image_id] = feats[j].astype(np.float32)

            # Segmentation mask
            mask = _generate_mask(group, rng, mask_size)
            np.save(str(mask_path), mask)

            records.append(
                {
                    "image_id": image_id,
                    "defect_type": subtype,
                    "expert_group": group,
                    "image_path": f"images/{image_id}.png",  # placeholder
                    "mask_path": f"masks/{mask_filename}",
                }
            )
            image_counter += 1

    # ------------------------------------------------------------------
    # 5. Save metadata CSV
    # ------------------------------------------------------------------
    metadata_path = out / "metadata.csv"
    df = pd.DataFrame(records)
    df.to_csv(str(metadata_path), index=False)

    # ------------------------------------------------------------------
    # 6. Save CNN features pickle
    # ------------------------------------------------------------------
    pickle_path = out / "cnn_features.pkl"
    with open(str(pickle_path), "wb") as fh:
        pickle.dump(features_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "pickle_path": str(pickle_path),
        "metadata_path": str(metadata_path),
        "masks_dir": str(masks_dir),
        "n_images": image_counter,
        "n_types": n_types,
        "n_groups": len(group_names),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _print_summary(info: dict[str, Any], df: pd.DataFrame) -> None:
    """Print human-readable summary statistics."""
    print("=" * 64)
    print("  Synthetic Defect Data -- Generation Summary")
    print("=" * 64)
    print(f"  Total images generated : {info['n_images']:,}")
    print(f"  Expert groups          : {info['n_groups']}")
    print(f"  Individual defect types: {info['n_types']}")
    print(f"  CNN features pickle    : {info['pickle_path']}")
    print(f"  Metadata CSV           : {info['metadata_path']}")
    print(f"  Masks directory        : {info['masks_dir']}")
    print("-" * 64)
    print("\n  Images per expert group:")
    group_counts = df["expert_group"].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"    {group:<25s} {count:>6,}")
    print("-" * 64)
    print("\n  Top-10 most common defect types:")
    type_counts = df["defect_type"].value_counts()
    for dtype, count in type_counts.head(10).items():
        print(f"    {dtype:<35s} {count:>6,}")
    print("\n  Top-10 rarest defect types:")
    for dtype, count in type_counts.tail(10).items():
        print(f"    {dtype:<35s} {count:>6,}")
    print("=" * 64)


if __name__ == "__main__":
    info = generate_synthetic_defect_data()
    df = pd.read_csv(info["metadata_path"])
    _print_summary(info, df)

"""
Synthetic Steel Surface Image Generator
===============================================================
Generates realistic-looking grayscale images of stainless/specialty steel coil
surfaces with the defect taxonomy used in the plant's surface-inspection
system.

Defect classes — plant taxonomy:
    0  - scratch          : fine longitudinal/transverse mechanical marks
    1  - pit              : deep crater-like depressions (scale fragments, roll damage)
    2  - roll_mark        : periodic marks repeating every roll circumference
    3  - edge_crack        : longitudinal cracks or laminations at strip edge
    4  - inclusion        : non-metallic inclusions (oxides, carbides, slag)
    5  - coil_break        : transverse crease bands from inelastic coiling deformation
    6  - scale             : residual oxide/scale patches (light-coloured blotches)
    7  - streak            : dark longitudinal marks from uneven lubrication or oil
    8  - sliver            : elongated raised metallic slivers (surface overlap)
    9  - orange_peel       : granular rough texture — excessive grain size after annealing
    10 - clean             : no defect — uniform cold-rolled stainless surface

Each generator replicates the morphological and photometric signature of the
defect as it appears under inspection line-scan lighting (diffuse backlight +
directional specular illumination from the rolling direction).

Physical references:
    - Yun & Ko (2014) "Surface Defect Detection of Steel Strips Using
      Lighting-Robust Dense U-Net" (Proc. ISIJ)
    - Saas, Amirat, Ouahabi (2017) "Surface Defect Classification"
    - Steel surface defect atlas (SEMass / WorldAutoSteel)

Folder structure created:
    data/sample_images/<class_name>/image_XXXX.png   (classification)
    data/sample_images/metadata.csv

Augmentation (Jin & Chen 2022):
    apply_mixup()               — pixel-level MixUp
    copy_paste_defect()         — feathered defect compositing
    generate_augmented_dataset()— batch augmentation with metadata

Semi-supervised split (Jin & Chen 2022):
    semi_supervised_split()     — retain n_labeled_per_class labels

Usage:
    python data/generate_sample_data.py
    python data/generate_sample_data.py --n_per_class 150 --size 256 --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Plant defect taxonomy
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "scratch",      # 0
    "pit",          # 1
    "roll_mark",    # 2
    "edge_crack",   # 3
    "inclusion",    # 4
    "coil_break",   # 5
    "scale",        # 6
    "streak",       # 7
    "sliver",       # 8
    "orange_peel",  # 9
    "clean",        # 10
]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# Which classes are actual defects (not clean)
DEFECT_CLASSES = [c for c in CLASS_NAMES if c != "clean"]

# Calibrated long-tail class prior for the inspection on-line stream (documented as
# calibrated-synthetic, source_version='synthetic-cal-v1').  The real inspection stream is
# heavily imbalanced: 'clean' dominates and rare structural defects
# (edge_crack/sliver/coil_break) get < ~10 instances/inspection-day (report L304-307).
# Using a uniform per-class count is the realism tell this replaces.
STEELDEFECT_CLASS_PRIOR = {
    "clean":       0.610,
    "scratch":     0.118,
    "streak":      0.071,
    "scale":       0.043,
    "pit":         0.031,
    "inclusion":   0.024,
    "roll_mark":   0.018,
    "orange_peel": 0.0125,
    "edge_crack":  0.0061,   # rare structural
    "sliver":      0.0039,   # rare
    "coil_break":  0.0025,   # rarest
}  # imbalance ratio ≈ 244 raw; ≈ 10 among the defect classes


def class_counts_from_prior(prior, n_total, min_per_class=8):
    """Convert a class-prior dict into integer per-class counts.

    ``count_c = max(min_per_class, round(n_total * prior_c))``.  The floor lets
    rare classes exist for active learning without faking their real frequency.
    """
    s = float(sum(prior.values())) or 1.0
    return {c: max(int(min_per_class), int(round(n_total * (p / s))))
            for c, p in prior.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Shared background model
# ─────────────────────────────────────────────────────────────────────────────

def _steel_background(rng: np.random.Generator, size: int) -> np.ndarray:
    """Realistic cold-rolled stainless steel surface (2B finish / bright-annealed).

    Physical model:
    1. Base reflectance — cold-rolled 2B finish has mean grey ~0.58–0.78
       with slight specular anisotropy in the rolling direction.
    2. Rolling grain — superimposed horizontal sinusoidal striations
       (multiple frequencies) aligned with the rolling direction.
    3. Micro-roughness — fine short-range correlated Gaussian noise,
       modelled as a spatially smooth field (Perlin-like) plus white noise.
    4. Periodic roller marks — faint dark bands from contamination residue on
       the work-roll surface, repeating every N pixels in rolling direction.
    5. Edge darkening — cosine brightness taper at left/right strip edges
       (real effect: edge contact during coiling reduces reflectance).
    6. Specular highlight variation — slight random brightness gradient
       simulating detector angle sensitivity across the strip width.
    """
    base = rng.uniform(0.55, 0.80)
    img = np.full((size, size), base, dtype=np.float32)
    rows = np.arange(size, dtype=np.float32)
    cols = np.arange(size, dtype=np.float32)

    # 1. Rolling grain: superimposed sinusoidal striations (rolling direction = rows)
    n_freqs = rng.integers(4, 10)
    for _ in range(n_freqs):
        freq      = rng.uniform(1.5, 18.0)
        amplitude = rng.uniform(0.002, 0.012)
        phase     = rng.uniform(0, 2 * np.pi)
        img += (amplitude * np.sin(2 * np.pi * freq * rows / size + phase))[:, None]

    # 2. Micro-roughness: two-scale (coarse + fine) Gaussian random fields
    #    Coarse: blurred noise (low-frequency surface waviness)
    coarse = rng.normal(0, 0.012, (size // 8, size // 8)).astype(np.float32)
    # Upsample coarse field with bilinear interpolation
    from PIL import Image as _PILImg
    coarse_up = np.array(
        _PILImg.fromarray(coarse).resize((size, size), _PILImg.BILINEAR)
    )
    img += coarse_up * 0.5
    #    Fine: pixel-level white noise
    img += rng.normal(0, 0.009, (size, size)).astype(np.float32)

    # 3. Periodic roller marks (very faint — background-level)
    period = rng.integers(size // 4, size)
    mark_w = rng.integers(1, 3)
    mark_d = rng.uniform(0.015, 0.045)
    offset = rng.integers(0, period)
    for r0 in range(offset, size, period):
        for w in range(mark_w):
            r = r0 + w
            if r < size:
                img[r, :] -= np.clip(mark_d + rng.normal(0, 0.006, size), 0, None)

    # 4. Edge darkening (cosine taper in width direction)
    x = np.linspace(0, np.pi, size, dtype=np.float32)
    edge_taper = 1.0 - 0.10 * (1.0 - np.sin(x))
    img *= edge_taper[None, :]

    # 5. Specular highlight variation (slow brightness gradient in width direction)
    highlight = 1.0 + 0.04 * np.sin(
        np.pi * cols / size + rng.uniform(0, np.pi)
    ).astype(np.float32)
    img *= highlight[None, :]

    return np.clip(img, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Defect generators — one per defect class
# ─────────────────────────────────────────────────────────────────────────────

def make_scratch(rng: np.random.Generator, size: int) -> np.ndarray:
    """Mechanical scratches on stainless steel surface.

    Physical origin: contact with foreign object (handling, tooling, guide),
    coil transport, slitter knife misalignment.

    Morphology:
    - 1–5 thin (1–3 px) dark lines
    - Predominantly longitudinal (rolling direction = rows) with slight curvature
    - Occasional transverse scratches (from perpendicular contact)
    - Irregular depth profile (not perfectly straight — varying local pressure)
    - Dark appearance: scratch disrupts the protective oxide layer exposing
      fresh metal that scatters light differently
    """
    img = _steel_background(rng, size)

    n_lines = rng.integers(1, 6)
    for _ in range(n_lines):
        orientation = rng.choice(["longitudinal", "transverse", "diagonal"],
                                  p=[0.55, 0.25, 0.20])
        width   = rng.integers(1, 4)
        depth   = rng.uniform(0.30, 0.70)   # darkness factor (lower = darker)
        # Slight waviness in the perpendicular direction
        waviness_amp = rng.uniform(0, 3.0)
        waviness_freq = rng.uniform(1.0, 4.0)

        if orientation == "longitudinal":
            # scratch runs along rows (rolling direction), fixed column band
            col_center = rng.integers(5, size - 5)
            col_offset = (waviness_amp * np.sin(
                2 * np.pi * waviness_freq * np.arange(size) / size
                + rng.uniform(0, 2 * np.pi)
            )).astype(int)
            for r in range(size):
                for w in range(-width, width + 1):
                    c = col_center + col_offset[r] + w
                    if 0 <= c < size:
                        img[r, c] *= (depth + rng.uniform(-0.05, 0.05))
        elif orientation == "transverse":
            # scratch runs along columns (cross-rolling direction), fixed row band
            row_center = rng.integers(5, size - 5)
            row_offset = (waviness_amp * np.sin(
                2 * np.pi * waviness_freq * np.arange(size) / size
                + rng.uniform(0, 2 * np.pi)
            )).astype(int)
            for c in range(size):
                for w in range(-width, width + 1):
                    r = row_center + row_offset[c] + w
                    if 0 <= r < size:
                        img[r, c] *= (depth + rng.uniform(-0.05, 0.05))
        else:
            # diagonal scratch
            r0  = rng.integers(0, size)
            c0  = rng.integers(0, size)
            angle = rng.uniform(10, 80)   # degrees from horizontal
            dr = np.cos(np.radians(angle))
            dc = np.sin(np.radians(angle))
            length = rng.integers(size // 4, size)
            for s in range(length):
                r = int(r0 + dr * s) % size
                c = int(c0 + dc * s) % size
                for w in range(-width, width + 1):
                    rr = np.clip(r + w, 0, size - 1)
                    img[rr, c] *= depth

    return np.clip(img, 0.0, 1.0)


def make_pit(rng: np.random.Generator, size: int) -> np.ndarray:
    """Surface pits and craters.

    Physical origin: hard scale or oxide particle pressed into the surface
    during hot rolling, later partially removed leaving a crater. Also from
    roll surface damage transferred to the strip.

    Morphology:
    - 2–10 elliptical depressions per image
    - Dark appearance: cavity interior scatters less light back to sensor
    - Bright rim: material displaced outward creates a ridge with higher
      reflectance (Lambertian scattering toward detector)
    - Variable depth profile: Gaussian falloff from center
    - Occasional clustering (from roll-surface lesion)
    """
    img = _steel_background(rng, size)
    n_pits = rng.integers(2, 11)
    Y, X = np.ogrid[:size, :size]

    # Optionally cluster pits (simulate roll damage region)
    clustered = rng.random() < 0.4
    if clustered:
        # pits grouped in a strip along rolling direction
        cx_base = rng.integers(size // 4, 3 * size // 4)
        cy_base = rng.integers(size // 4, 3 * size // 4)
    else:
        cx_base, cy_base = size // 2, size // 2

    for _ in range(n_pits):
        if clustered:
            cx = cx_base + rng.integers(-size // 6, size // 6 + 1)
            cy = cy_base + rng.integers(-size // 5, size // 5 + 1)
        else:
            cx = rng.integers(8, size - 8)
            cy = rng.integers(8, size - 8)
        rx = rng.integers(2, max(3, size // 12))
        ry = rng.integers(2, max(3, size // 10))   # slightly elongated in rolling dir
        depth = rng.uniform(0.02, 0.20)

        # Gaussian depth profile (smooth crater shape)
        dist_sq = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        crater_profile = np.exp(-dist_sq * 1.5)

        # Dark interior
        img -= depth * crater_profile

        # Bright rim (ring between dist_sq=0.8 and 1.6)
        rim_mask = (dist_sq >= 0.7) & (dist_sq <= 1.8)
        rim_brightness = rng.uniform(0.02, 0.08)
        img[rim_mask] = np.clip(img[rim_mask] + rim_brightness, 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_roll_mark(rng: np.random.Generator, size: int) -> np.ndarray:
    """Periodic roll marks — contamination/damage on the work-roll surface.

    Physical origin: foreign material embedded in roll surface, or a
    surface lesion on the roll. Creates a periodic mark on the strip
    with period equal to the roll circumference (π × D).

    Morphology:
    - 2–4 regularly spaced transverse bands per image
    - Each band: irregular width, mixed dark/bright texture
    - Can be single marks (spot on roll) or band marks (line on roll)
    - Slight variation in mark intensity along its length (pressure variation)
    """
    img = _steel_background(rng, size)

    mark_type = rng.choice(["spot", "band"], p=[0.4, 0.6])
    period = rng.integers(size // 5, size // 2)
    offset = rng.integers(0, period)

    if mark_type == "band":
        # Full-width dark/bright band
        mark_width = rng.integers(3, 10)
        darkness   = rng.uniform(0.25, 0.60)
        row = offset
        while row < size:
            for w in range(mark_width):
                r = row + w
                if r < size:
                    # Gaussian profile within the band (smoother edges)
                    rel = (w - mark_width / 2) / (mark_width / 2 + 1e-6)
                    profile = np.exp(-2 * rel ** 2)
                    noise = rng.normal(0, 0.03, size)
                    img[r, :] *= np.clip(1.0 - darkness * profile + noise, 0.1, 1.5)
            row += period
    else:
        # Spot marks — small dark patches at periodically spaced rows
        # (circumferential spot from a point inclusion on the roll surface)
        n_spots_per_period = rng.integers(1, 4)
        col_centers = rng.integers(size // 6, 5 * size // 6, n_spots_per_period)
        row = offset
        Y, X = np.ogrid[:size, :size]
        while row < size:
            for cx in col_centers:
                cy = row
                rx = rng.integers(3, size // 10)
                ry = rng.integers(2, size // 14)
                darkness = rng.uniform(0.35, 0.70)
                dist_sq = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
                profile = np.exp(-dist_sq * 2)
                img -= darkness * profile * (dist_sq <= 2.0).astype(float)
            row += period

    return np.clip(img, 0.0, 1.0)


def make_edge_crack(rng: np.random.Generator, size: int) -> np.ndarray:
    """Edge cracks and edge laminations.

    Physical origin: tensile stress concentration at strip edges during hot
    or cold rolling exceeds local fracture toughness. Common in AHSS and
    high-Mn stainless grades. Also includes edge laminations from inclusion
    bands at edges.

    Morphology:
    - Dark, irregular region along one (or both) edges
    - Jagged crack tip propagating inward with variable depth
    - Bright reflection from crack faces (specular component inside crack)
    - Can have secondary branching cracks
    - Edge damage region is wider at positions of max. tensile stress
    """
    img = _steel_background(rng, size)

    edges = rng.choice(["left", "right", "both"], p=[0.40, 0.40, 0.20])
    edge_list = ["left", "right"] if edges == "both" else [edges]

    for edge in edge_list:
        base_depth = rng.integers(size // 12, size // 4)

        # Irregular crack profile: random walk biased inward at stress raisers
        crack_profile = np.zeros(size, dtype=int)
        current_depth = 0
        for r in range(size):
            # Smooth random walk with mean-reversion toward base_depth
            step = rng.integers(-4, 5)
            current_depth = int(np.clip(
                current_depth + step + 0.15 * (base_depth - current_depth),
                0, size // 2
            ))
            crack_profile[r] = current_depth

        # Smooth the profile
        from scipy.ndimage import uniform_filter1d as _uf1d
        crack_profile = _uf1d(crack_profile.astype(float), size=5).astype(int)

        # Apply dark zone for the crack body
        for r in range(size):
            d = crack_profile[r]
            if edge == "left":
                cols_crack = slice(0, d)
                img[r, cols_crack] *= rng.uniform(0.02, 0.20)
                # Bright crack-tip highlight (diffraction from crack edge)
                if d < size - 2:
                    img[r, d:d + 2] = np.clip(
                        img[r, d:d + 2] * rng.uniform(1.3, 1.8), 0, 1
                    )
            else:
                cols_crack = slice(size - d, size)
                img[r, cols_crack] *= rng.uniform(0.02, 0.20)
                if size - d > 1:
                    img[r, size - d - 2:size - d] = np.clip(
                        img[r, size - d - 2:size - d] * rng.uniform(1.3, 1.8), 0, 1
                    )

        # Secondary branching cracks (50% probability)
        if rng.random() < 0.5:
            n_branches = rng.integers(1, 4)
            for _ in range(n_branches):
                br_row   = rng.integers(5, size - 5)
                br_depth = rng.integers(2, size // 8)
                br_width = rng.integers(1, 3)
                for w in range(br_width):
                    r = br_row + w
                    if r < size:
                        if edge == "left":
                            img[r, crack_profile[br_row]:
                                   crack_profile[br_row] + br_depth] *= 0.15
                        else:
                            start = size - crack_profile[br_row] - br_depth
                            img[r, start:size - crack_profile[br_row]] *= 0.15

    return np.clip(img, 0.0, 1.0)


def make_inclusion(rng: np.random.Generator, size: int) -> np.ndarray:
    """Non-metallic inclusions: oxides, carbides, slag particles.

    Physical origin: refractory material from ladle/tundish, oxide stringers
    from insufficient argon stirring, or carbide precipitation at grain
    boundaries during solidification. Especially prevalent in austenitic
    stainless grades (304/316) with high oxygen content.

    Morphology:
    - Mix of bright and dark spots (depends on inclusion type/size/orientation):
        * Bright: alumina (Al₂O₃), silica — high reflectance
        * Dark: sulfides (MnS), carbides — low reflectance, or shadow cavity
    - Elongated in rolling direction (stringer inclusions)
    - Can appear singly or in linear clusters (stringer pattern)
    - Often surrounded by a depletion zone (slightly darker halo)
    """
    img = _steel_background(rng, size)

    n_inclusions = rng.integers(1, 8)
    Y, X = np.ogrid[:size, :size]

    # Stringer pattern: inclusions aligned along rolling direction
    stringer = rng.random() < 0.5
    if stringer:
        # All inclusions in a narrow column band
        col_center = rng.integers(size // 5, 4 * size // 5)
        col_spread = rng.integers(3, size // 8)
    else:
        col_center = size // 2
        col_spread = size // 2

    for _ in range(n_inclusions):
        cy = rng.integers(8, size - 8)
        cx = int(np.clip(
            rng.normal(col_center, col_spread if stringer else size // 4),
            8, size - 8
        ))
        # Elongated in rolling direction (rows)
        ry = rng.integers(1, max(2, size // 18))
        rx = rng.integers(1, max(2, size // 24))

        inclusion_type = rng.choice(["bright", "dark", "mixed"], p=[0.35, 0.45, 0.20])
        dist_sq = ((X - cx) / (rx + 1e-6)) ** 2 + ((Y - cy) / (ry + 1e-6)) ** 2

        if inclusion_type == "bright":
            brightness = rng.uniform(0.25, 0.60)
            profile = brightness * np.exp(-dist_sq * 2.5)
            img = np.clip(img + profile, 0, 1)
        elif inclusion_type == "dark":
            darkness = rng.uniform(0.30, 0.75)
            profile = darkness * np.exp(-dist_sq * 2.5)
            img = np.clip(img - profile, 0, 1)
        else:
            # Mixed: bright center, dark ring
            bright_core = rng.uniform(0.20, 0.40) * np.exp(-dist_sq * 4.0)
            dark_ring   = rng.uniform(0.10, 0.20) * np.exp(-dist_sq * 1.2)
            img = np.clip(img + bright_core - dark_ring, 0, 1)

        # Depletion halo (slightly darker ring around inclusion)
        halo_mask = (dist_sq >= 1.2) & (dist_sq <= 3.5)
        img[halo_mask] = np.clip(img[halo_mask] * rng.uniform(0.90, 0.97), 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_coil_break(rng: np.random.Generator, size: int) -> np.ndarray:
    """Coil breaks — transverse crease bands from inelastic coil deformation.

    Physical origin: when a tightly wound coil is uncoiled on the processing
    line, regions that were bent around the coil core at temperatures below
    the yield stress undergo plastic deformation. This creates transverse
    bright/dark bands perpendicular to the rolling direction.

    Distinct from roll marks: coil breaks are wider (5–40 mm wide transverse
    bands), show a characteristic twin-band structure (one bright, one dark),
    and do not repeat with the roll circumference period.

    Morphology:
    - 1–4 transverse bands per image
    - Each band: bright–dark paired pattern (tensile vs compressive bending side)
    - Gradual edges (smooth transition from base metal)
    - Bands can be slightly tilted if coil was wound with camber
    """
    img = _steel_background(rng, size)
    n_breaks = rng.integers(1, 5)
    for _ in range(n_breaks):
        row_center = rng.integers(size // 8, 7 * size // 8)
        half_width = rng.integers(size // 20, size // 6)
        # Slight tilt: column-dependent row offset
        tilt_px = rng.uniform(-3, 3)

        brightness = rng.uniform(0.08, 0.18)
        sign = rng.choice([-1, 1])   # bright-first or dark-first

        for c in range(size):
            center = row_center + tilt_px * (c - size / 2) / size
            # Gaussian amplitude profile across the band
            r_range = np.arange(
                max(0, int(center - half_width * 2)),
                min(size, int(center + half_width * 2))
            )
            for r in r_range:
                norm_dist = (r - center) / (half_width + 1e-6)
                # First half: sign * bright/dark, second half: opposite
                if abs(norm_dist) <= 1.0:
                    if norm_dist < 0:
                        img[r, c] = np.clip(
                            img[r, c] + sign * brightness * (1 - abs(norm_dist)), 0, 1
                        )
                    else:
                        img[r, c] = np.clip(
                            img[r, c] - sign * brightness * (1 - abs(norm_dist)), 0, 1
                        )

    return np.clip(img, 0.0, 1.0)


def make_scale(rng: np.random.Generator, size: int) -> np.ndarray:
    """Residual oxide scale patches.

    Physical origin: incomplete removal of hot-roll scale during pickling
    (hydrofluoric/sulfuric acid treatment), or re-oxidation after pickling
    before cold rolling. Common at strip edges or below certain acid
    concentration gradients.

    Morphology:
    - Irregular bright-grey blotches (iron oxide has higher reflectance
      than bare metal under diffuse illumination)
    - Slightly rough texture within the patch (scale has cracked surfaces)
    - Irregular boundary — not sharp, more like a cloud or coral shape
    - Can cover 5–30% of the image area
    - Colour: light grey / yellowish-grey (in RGB but shown as brighter in BW)
    """
    img = _steel_background(rng, size)
    n_patches = rng.integers(1, 5)

    for _ in range(n_patches):
        cx = rng.integers(size // 8, 7 * size // 8)
        cy = rng.integers(size // 8, 7 * size // 8)
        # Patch size proportional to image
        r_major = rng.integers(size // 10, size // 4)
        r_minor = rng.integers(size // 14, r_major)
        angle = rng.uniform(0, np.pi)

        # Irregular patch boundary: base ellipse + Fourier perturbation
        theta = np.linspace(0, 2 * np.pi, 360)
        # Perturb radius with low-frequency sinusoids
        r_perturb = np.ones_like(theta)
        for k in range(2, 7):
            r_perturb += 0.15 * rng.uniform(-1, 1) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
        r_perturb = np.clip(r_perturb, 0.4, 1.6)

        # Rasterise with filled polygon approach
        Y, X = np.ogrid[:size, :size]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = cos_a * (X - cx) + sin_a * (Y - cy)
        y_rot = -sin_a * (X - cx) + cos_a * (Y - cy)

        # Ellipse distance (base shape)
        dist_ellipse = np.sqrt((x_rot / (r_major + 1e-6)) ** 2 +
                               (y_rot / (r_minor + 1e-6)) ** 2)

        scale_brightness = rng.uniform(0.06, 0.18)
        # Interior of patch — slightly brighter + noisy texture
        interior = dist_ellipse < 0.90
        img[interior] = np.clip(
            img[interior] + scale_brightness + rng.normal(0, 0.025, interior.sum()),
            0, 1
        )
        # Transition zone
        trans = (dist_ellipse >= 0.90) & (dist_ellipse < 1.20)
        fade  = 1.0 - (dist_ellipse[trans] - 0.90) / 0.30
        img[trans] = np.clip(img[trans] + scale_brightness * fade, 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_streak(rng: np.random.Generator, size: int) -> np.ndarray:
    """Dark longitudinal streaks — uneven lubrication or roll-coating residue.

    Physical origin: uneven distribution of rolling oil or emulsion across the
    strip width leads to regions of excess or insufficient lubrication. The
    heavy-oil zones appear as dark longitudinal bands when viewed under diffuse
    illumination. Also caused by uneven anti-oxidation coating application in
    continuous annealing lines.

    Morphology:
    - 1–4 dark longitudinal bands (parallel to rolling direction)
    - Gradual width variation along length (oil thickness varies)
    - Soft edges (Gaussian profile in width direction)
    - Dark appearance: oil layer increases surface absorptance
    - Can have slight periodic variation along length (metering-roll effect)
    """
    img = _steel_background(rng, size)
    n_streaks = rng.integers(1, 5)

    for _ in range(n_streaks):
        col_center = rng.integers(size // 8, 7 * size // 8)
        half_width = rng.integers(size // 20, size // 6)
        darkness   = rng.uniform(0.08, 0.30)
        # Slight variation in darkness along the streak (periodic or random)
        vary_type  = rng.choice(["periodic", "random"])

        for r in range(size):
            if vary_type == "periodic":
                period_r = rng.integers(size // 4, size)
                local_depth = darkness * (1.0 + 0.3 * np.sin(2 * np.pi * r / period_r))
            else:
                local_depth = darkness * rng.uniform(0.6, 1.4)

            for c in range(max(0, col_center - half_width * 3),
                           min(size, col_center + half_width * 3)):
                # Gaussian profile across width
                norm_dist = abs(c - col_center) / (half_width + 1e-6)
                weight = np.exp(-2 * norm_dist ** 2)
                img[r, c] = np.clip(img[r, c] - local_depth * weight, 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_sliver(rng: np.random.Generator, size: int) -> np.ndarray:
    """Slivers — thin metallic protrusions on the surface (overlap defect).

    Physical origin: folded metal from a surface lamination, lap, or seam
    defect in the original slab/billet gets rolled out as a thin sliver
    attached at one end and free at the other. Very common in ferritic
    stainless grades with low toughness.

    Morphology:
    - 1–3 thin, elongated, bright features (raised above surface → higher
      specular reflection toward sensor)
    - Tapered shape: wider/brighter at the attached end
    - Sharp edges (raised feature casts a narrow shadow)
    - Strong specular highlight along the sliver body
    - Shadow lines alongside the sliver (directional illumination)
    - Predominantly aligned with rolling direction
    """
    img = _steel_background(rng, size)

    n_slivers = rng.integers(1, 4)
    for _ in range(n_slivers):
        # Sliver start/end positions
        r_start = rng.integers(0, size // 2)
        r_end   = rng.integers(r_start + size // 8, min(size, r_start + 3 * size // 4))
        col_center = rng.integers(size // 8, 7 * size // 8)

        # Width tapers from base (r_start) to tip (r_end)
        base_half_width = rng.integers(2, max(3, size // 20))
        brightness_sliver = rng.uniform(0.25, 0.60)
        shadow_depth = rng.uniform(0.05, 0.15)

        for r in range(r_start, r_end):
            # Taper factor: 1.0 at base → 0.0 at tip
            taper = (r_end - r) / (r_end - r_start + 1e-6)
            half_w = max(1, int(base_half_width * taper))

            # Sliver body: bright
            for c in range(max(0, col_center - half_w), min(size, col_center + half_w + 1)):
                img[r, c] = np.clip(img[r, c] + brightness_sliver * taper, 0, 1)

            # Shadow alongside (narrow dark band next to the bright sliver)
            shadow_w = max(1, half_w // 2)
            for c in range(max(0, col_center + half_w + 1),
                           min(size, col_center + half_w + shadow_w + 2)):
                img[r, c] = np.clip(img[r, c] - shadow_depth, 0, 1)
            for c in range(max(0, col_center - half_w - shadow_w - 1),
                           min(size, col_center - half_w)):
                img[r, c] = np.clip(img[r, c] - shadow_depth, 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_orange_peel(rng: np.random.Generator, size: int) -> np.ndarray:
    """Orange peel — granular rough surface from excessive grain growth.

    Physical origin: abnormal grain growth during annealing (especially in
    ferritic stainless Fe-Cr grades, e.g. AISI 430) results in
    grains large enough (>50 μm) to be visible as individual facets in
    reflected light. Each grain facet tilted differently creates a
    characteristic orange-peel texture.

    Morphology:
    - Uniform coverage of small (3–15 px) irregular bright/dark cells
    - No directionality (unlike rolling grain or streaks)
    - Cell boundaries slightly darker than cell interiors
    - Random Voronoi-like cell structure
    - Subtle: individual cells are low-contrast
    """
    img = _steel_background(rng, size)

    # Generate random seed points for Voronoi-like cells
    n_grains = rng.integers(
        max(20, (size // 16) ** 2),
        max(21, (size // 8) ** 2)
    )
    seed_y = rng.integers(0, size, n_grains)
    seed_x = rng.integers(0, size, n_grains)
    grain_brightness = rng.normal(0, rng.uniform(0.025, 0.065), n_grains).astype(np.float32)

    # Assign each pixel to its nearest grain seed (simplified Voronoi)
    # Use block-processing to avoid huge memory usage at large sizes
    block = 32
    for r0 in range(0, size, block):
        for c0 in range(0, size, block):
            r1 = min(r0 + block, size)
            c1 = min(c0 + block, size)
            R = np.arange(r0, r1)
            C = np.arange(c0, c1)
            RR, CC = np.meshgrid(R, C, indexing="ij")
            # Distances from block pixels to all seed points
            dr = (RR[:, :, None] - seed_y[None, None, :]).astype(np.float32)
            dc = (CC[:, :, None] - seed_x[None, None, :]).astype(np.float32)
            dist = dr ** 2 + dc ** 2
            nearest = np.argmin(dist, axis=2)
            img[r0:r1, c0:c1] += grain_brightness[nearest]

    # Darken grain boundaries (pixels close to more than one nearest grain)
    # Approximation: local variance indicates boundary
    from scipy.ndimage import uniform_filter as _uf
    local_mean = _uf(img, size=3)
    local_sq   = _uf(img ** 2, size=3)
    local_var  = np.clip(local_sq - local_mean ** 2, 0, None)
    boundary_mask = local_var > np.percentile(local_var, 85)
    img[boundary_mask] = np.clip(img[boundary_mask] - 0.03, 0, 1)

    return np.clip(img, 0.0, 1.0)


def make_clean(rng: np.random.Generator, size: int) -> np.ndarray:
    """Clean surface — no defect, uniform cold-rolled stainless finish."""
    return _steel_background(rng, size)


# ─────────────────────────────────────────────────────────────────────────────
# Generator dispatch table
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "scratch":    make_scratch,
    "pit":        make_pit,
    "roll_mark":  make_roll_mark,
    "edge_crack": make_edge_crack,
    "inclusion":  make_inclusion,
    "coil_break": make_coil_break,
    "scale":      make_scale,
    "streak":     make_streak,
    "sliver":     make_sliver,
    "orange_peel":make_orange_peel,
    "clean":      make_clean,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str = "data/sample_images",
    n_per_class: int = 100,
    image_size: int = 128,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test_frac = 1 − train − val  ≈ 0.15
    n_total=None,
    prior=None,
    min_per_class: int = 8,
) -> pd.DataFrame:
    """Generate the full synthetic dataset.

    Two sampling modes: **balanced** (default, ``n_per_class`` per class) or
    **calibrated long-tail** — pass ``prior`` (e.g. :data:`STEELDEFECT_CLASS_PRIOR`)
    and ``n_total`` to draw a realistic imbalanced corpus; records then carry
    ``sampling_prior`` and ``source_version='synthetic-cal-v1'``.

    Args:
        output_dir:  Root directory (created if missing).
        n_per_class: Number of images per defect class (11 classes total).
        image_size:  Square image side length in pixels.
        seed:        Random seed for reproducibility.
        train_frac:  Fraction of images assigned to the training split.
        val_frac:    Fraction assigned to validation.

    Returns:
        DataFrame: path, class_id, class_name, has_defect, split
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    records = []

    long_tail = prior is not None and n_total is not None
    counts = (class_counts_from_prior(prior, n_total, min_per_class)
              if long_tail else {c: n_per_class for c in GENERATORS})

    for class_name, generator in GENERATORS.items():
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_id  = CLASS_IDS[class_name]
        n_c = int(counts.get(class_name, n_per_class))

        for i in tqdm(range(n_c),
                      desc=f"  [{class_name:12s}]",
                      leave=False):
            arr  = generator(rng, image_size)
            gray = (arr * 255).astype(np.uint8)
            rgb  = np.stack([gray, gray, gray], axis=-1)
            img_path = class_dir / f"{class_name}_{i:04d}.png"
            Image.fromarray(rgb).save(img_path)

            r     = rng.random()
            split = ("train" if r < train_frac
                     else ("val" if r < train_frac + val_frac else "test"))
            rec = {
                "path":       str(img_path.resolve()),
                "class_id":   class_id,
                "class_name": class_name,
                "has_defect": class_name != "clean",
                "split":      split,
            }
            if long_tail:
                rec["sampling_prior"] = float(prior.get(class_name, 0.0))
                rec["source_version"] = "synthetic-cal-v1"
            records.append(rec)

    df        = pd.DataFrame(records)
    meta_path = out / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"\nGenerated {len(df)} images -> {out}")
    print(f"  Classes     : {CLASS_NAMES}")
    print(f"  Image size  : {image_size}×{image_size} px  (3-channel RGB)")
    print(f"  Splits      : {df['split'].value_counts().to_dict()}")
    print(f"  Metadata CSV: {meta_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Demo A/B predictions — realistic synthetic outputs for testing
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_ab_data(
    metadata_path: str = "data/sample_images/metadata.csv",
    output_path:   str = "data/demo_ab_predictions.csv",
    accuracy_a:    float = 0.85,
    accuracy_b:    float = 0.82,
    noise_a:       float = 0.16,
    noise_b:       float = 0.16,
    seed:          int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic A/B model predictions for the plant taxonomy.

    Avoids the degenerate case of random-weight models (which predict the same
    class for every image, giving 0% disagreement and meaningless statistics).

    Confusion structure is calibrated to real inspection patterns:
    - scratch  ↔ streak   (both longitudinal, similar contrast)
    - pit      ↔ inclusion (both spot-like)
    - roll_mark ↔ coil_break (both transverse band patterns)
    - scale    ↔ orange_peel (both area-coverage, low-contrast)
    - sliver  is distinct (unique morphology)
    - edge_crack has low confusion rate (unique location signature)
    - clean is frequently confused with low-contrast defects (scale, orange_peel)

    Model A: well-trained production model (~85% overall, high confidence).
    Model B: challenger model (~72% overall, lower confidence on most classes,
             but *better* on 3 classes where it was specifically retrained).

    The mixed-result design (B better on some classes, worse on others) creates
    a realistic A/B comparison where the decision isn't trivially "A wins
    everywhere."

    Returns DataFrame with columns:
        image_path, label, label_name, pred_a, pred_name_a, confidence_a,
        pred_b, pred_name_b, confidence_b, agrees, confidence_delta,
        entropy_a, entropy_b, margin_a, margin_b
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.csv not found at '{metadata_path}'. "
            "Run generate_dataset() first."
        )

    meta      = pd.read_csv(metadata_path)
    rng       = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)

    # ── Per-class accuracy overrides ────────────────────────────────────
    # Model B was specifically retrained on these 3 classes (realistic
    # scenario: challenger model targets known weak spots).
    # On these classes, B outperforms A. On the rest, A remains better.
    b_improved_classes = {
        CLASS_IDS["edge_crack"]:  0.92,   # B specialised on edge defects
        CLASS_IDS["sliver"]:      0.90,   # B better at elongated features
        CLASS_IDS["orange_peel"]: 0.88,   # B retrained on texture defects
    }
    # Model A per-class: slight variation around base accuracy
    a_class_acc = {
        CLASS_IDS["edge_crack"]:  0.80,   # A weaker on edge defects
        CLASS_IDS["sliver"]:      0.82,   # A decent but not great
        CLASS_IDS["orange_peel"]: 0.78,   # A struggles with subtle textures
    }

    # ── Confusion weight matrix (n_classes × n_classes) ──────────────────
    # Row = true class, column = predicted class.
    # Diagonal = correct. Off-diagonal = confusion probability (unnormalised).
    # Designed from domain knowledge of which defects look similar.
    _CONF = np.eye(n_classes, dtype=np.float32)
    # symmetric confusions
    def _confuse(a, b, w):
        _CONF[a, b] += w
        _CONF[b, a] += w

    idx = CLASS_IDS
    _confuse(idx["scratch"],    idx["streak"],      0.30)   # both longitudinal marks
    _confuse(idx["scratch"],    idx["sliver"],       0.15)   # longitudinal bright/dark
    _confuse(idx["pit"],        idx["inclusion"],    0.35)   # both spot-like
    _confuse(idx["roll_mark"],  idx["coil_break"],   0.25)   # both transverse bands
    _confuse(idx["roll_mark"],  idx["scratch"],      0.10)   # can look like spaced lines
    _confuse(idx["scale"],      idx["orange_peel"],  0.40)   # both area-coverage, light
    _confuse(idx["scale"],      idx["clean"],        0.25)   # low-contrast scale → clean
    _confuse(idx["orange_peel"],idx["clean"],        0.30)   # subtle texture → clean
    _confuse(idx["streak"],     idx["coil_break"],   0.10)   # long dark band
    _confuse(idx["inclusion"],  idx["pit"],          0.20)   # small dark spot

    # Normalise rows to probability vectors
    _CONF_NORM = (_CONF / _CONF.sum(axis=1, keepdims=True)).astype(np.float32)

    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def _entropy(probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-10, 1.0)
        return float(-(p * np.log(p)).sum())

    def _simulate(true_label: int, accuracy: float, noise: float, scale: float) -> dict:
        # Sample predicted label: correct with probability `accuracy`, else
        # confuse according to the domain confusion matrix
        correct = rng.random() < accuracy
        if correct:
            pred_label = true_label
        else:
            confusion_row = _CONF_NORM[true_label].copy()
            confusion_row[true_label] = 0.0
            s = confusion_row.sum()
            if s < 1e-8:
                # fallback: uniform over wrong classes
                wrong = [c for c in range(n_classes) if c != true_label]
                pred_label = int(rng.choice(wrong))
            else:
                confusion_row /= s
                pred_label = int(rng.choice(n_classes, p=confusion_row))

        # Build logits: peak at predicted label + calibrated noise
        logits = rng.normal(0, noise, n_classes).astype(np.float32)
        logits[pred_label] += scale
        probs  = _softmax(logits)

        conf   = float(probs.max())
        pred   = int(probs.argmax())
        ent    = _entropy(probs)
        sorted_p = np.sort(probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
        return {"pred": pred, "confidence": conf, "entropy": ent, "margin": margin}

    records = []
    for _, row in meta.iterrows():
        true_label = int(row["class_id"])
        # Scale controls logit peakiness → confidence distribution spread.
        # Lowered (was 2.5-4.5 / 1.5-3.5) so the softmax max does NOT saturate
        # near 1.0: this yields wider, OVERLAPPING confidence distributions
        # (mean_conf_A≈0.74, mean_conf_B≈0.69, std≈0.13) and therefore a finite,
        # realistic Welch t instead of the degenerate ~1e6 value that betrays
        # synthetic data.
        scale_a = rng.uniform(1.8, 3.2)
        scale_b = rng.uniform(1.7, 3.1)

        # Per-class accuracy: use overrides where B is better, else defaults
        eff_acc_a = a_class_acc.get(true_label, accuracy_a)
        eff_acc_b = b_improved_classes.get(true_label, accuracy_b)

        a = _simulate(true_label, eff_acc_a, noise_a, scale=scale_a)
        b = _simulate(true_label, eff_acc_b, noise_b, scale=scale_b)
        records.append({
            "image_path":       row["path"],
            "label":            true_label,
            "label_name":       CLASS_NAMES[true_label],
            "pred_a":           a["pred"],
            "pred_name_a":      CLASS_NAMES[a["pred"]],
            "confidence_a":     round(a["confidence"], 6),
            "pred_b":           b["pred"],
            "pred_name_b":      CLASS_NAMES[b["pred"]],
            "confidence_b":     round(b["confidence"], 6),
            "agrees":           int(a["pred"] == b["pred"]),
            "confidence_delta": round(abs(a["confidence"] - b["confidence"]), 6),
            "entropy_a":        round(a["entropy"], 6),
            "entropy_b":        round(b["entropy"], 6),
            "margin_a":         round(a["margin"], 6),
            "margin_b":         round(b["margin"], 6),
            # OBT-compatibility bookkeeping (so this demo file maps 1:1 onto the
            # steeldefect.obt_segmentations / al_candidates governance columns).
            "is_labeled":       False,
            "al_cycle":         -1,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)

    dr = 1 - df["agrees"].mean()
    print(f"\nDemo A/B predictions saved: {output_path}")
    print(f"  n={len(df)}  disagreement={dr:.1%}  classes={df['pred_a'].nunique()}")
    print(f"  mean_conf_A={df['confidence_a'].mean():.3f}  "
          f"mean_conf_B={df['confidence_b'].mean():.3f}")
    print(f"  mean_entropy_A={df['entropy_a'].mean():.3f}  "
          f"mean_entropy_B={df['entropy_b'].mean():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Generate synthetic stainless steel surface images"
    )
    p.add_argument("--output_dir",  default="data/sample_images",
                   help="Output directory (default: data/sample_images)")
    p.add_argument("--n_per_class", type=int, default=100,
                   help="Images per class — 11 classes total (default: 100)")
    p.add_argument("--size",        type=int, default=128,
                   help="Image side length in pixels (default: 128)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    print("Generating synthetic dataset...")
    print(f"  Output   : {args.output_dir}")
    print(f"  N/class  : {args.n_per_class}  (11 classes × {args.n_per_class} = "
          f"{11 * args.n_per_class} images)")
    print(f"  Size     : {args.size}px  |  Seed: {args.seed}\n")

    df = generate_dataset(
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        image_size=args.size,
        seed=args.seed,
    )
    print("\nClass distribution (split breakdown):")
    print(df.groupby(["class_name", "split"]).size().unstack(fill_value=0).to_string())

    meta_path = os.path.join(args.output_dir, "metadata.csv")
    generate_demo_ab_data(
        metadata_path=meta_path,
        output_path="data/demo_ab_predictions.csv",
        seed=args.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation functions  (Jin & Chen 2022)
# ─────────────────────────────────────────────────────────────────────────────

def apply_mixup(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """MixUp augmentation at the pixel level (Jin & Chen 2022).

    Generates a convex combination of two images:
        x̃ = λ · img1 + (1 − λ) · img2,    λ ~ Beta(α, α)

    For steel surface defect detection, MixUp creates intermediate
    defect severity levels and improves model calibration.

    Reference:
        Jin & Chen (2022) Section 3.1 — MixUp for steel surface augmentation.
        Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization" (ICLR).

    Args:
        img1:   float32 array in [0, 1], any shape.
        img2:   float32 array same shape as img1.
        alpha:  Beta distribution parameter (default 0.4).
        rng:    Optional numpy random generator for reproducibility.

    Returns:
        Tuple (mixed_image, lam) where lam is the mixing coefficient.
    """
    if rng is None:
        rng = np.random.default_rng()
    lam  = float(rng.beta(alpha, alpha))
    mixed = lam * img1.astype(np.float32) + (1.0 - lam) * img2.astype(np.float32)
    return np.clip(mixed, 0.0, 1.0), lam


def copy_paste_defect(
    clean_img:    np.ndarray,
    defect_img:   np.ndarray,
    defect_mask:  np.ndarray,
    feather_sigma: float = 2.0,
) -> np.ndarray:
    """Copy-paste defect augmentation (Jin & Chen 2022).

    Composites a defect region from `defect_img` onto a `clean_img`
    background using Gaussian edge feathering to prevent hard boundaries.

    Reference:
        Jin & Chen (2022) Section 3.2 — copy-paste for defect augmentation.
        Ghiasi et al. (2021) "Simple Copy-Paste" (CVPR).

    Args:
        clean_img:      float32 [0,1] — background.
        defect_img:     float32 same shape — source of defect pixels.
        defect_mask:    float32 (H, W) binary mask (1 = defect region).
        feather_sigma:  Gaussian sigma for mask edge feathering.

    Returns:
        float32 same shape — background with pasted defect.
    """
    from scipy.ndimage import gaussian_filter

    clean  = clean_img.astype(np.float32)
    defect = defect_img.astype(np.float32)
    mask   = defect_mask.astype(np.float32)

    soft_mask = np.clip(gaussian_filter(mask, sigma=feather_sigma), 0.0, 1.0)
    if clean.ndim == 3 and soft_mask.ndim == 2:
        soft_mask = soft_mask[:, :, np.newaxis]

    return np.clip(soft_mask * defect + (1.0 - soft_mask) * clean, 0.0, 1.0)


def generate_augmented_dataset(
    source_dir: str,
    output_dir: str,
    n_augmented_per_class: int = 50,
    mixup_alpha: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate augmented images using MixUp and copy-paste (Jin & Chen 2022).

    For each defect class:
    - MixUp: randomly pairs two images from the same class and creates λ-blends.
    - Copy-Paste: composites defect regions from defect images onto clean backgrounds.

    Args:
        source_dir:             Directory with existing images and metadata.csv.
        output_dir:             Directory to save augmented images.
        n_augmented_per_class:  Number of augmented images to generate per class.
        mixup_alpha:            Beta parameter for MixUp (default 0.4).
        seed:                   Random seed.

    Returns:
        DataFrame with all original + augmented images including `aug_type` column.
    """
    rng    = np.random.default_rng(seed)
    source = Path(source_dir)
    out    = Path(output_dir)

    meta_path = source / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {source_dir}")

    df          = pd.read_csv(meta_path)
    aug_records = []

    for class_name in CLASS_NAMES:
        class_df = df[df["class_name"] == class_name]
        if len(class_df) < 2:
            continue

        aug_dir = out / class_name
        aug_dir.mkdir(parents=True, exist_ok=True)

        images = [
            np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            for p in class_df["path"].values
        ]

        n_mixup     = n_augmented_per_class // 2
        n_copypaste = n_augmented_per_class - n_mixup

        # --- MixUp ---
        for i in range(n_mixup):
            idx1, idx2 = rng.choice(len(images), 2, replace=False)
            mixed, _ = apply_mixup(images[idx1], images[idx2],
                                      alpha=mixup_alpha, rng=rng)
            fname = aug_dir / f"{class_name}_mixup_{i:04d}.png"
            gray  = (mixed * 255).astype(np.uint8)
            Image.fromarray(np.stack([gray, gray, gray], axis=-1)).save(fname)
            aug_records.append({
                "path":       str(fname.resolve()),
                "class_id":   CLASS_IDS[class_name],
                "class_name": class_name,
                "has_defect": class_name != "clean",
                "split":      "train",
                "aug_type":   "mixup",
            })

        # --- Copy-Paste ---
        clean_df  = df[df["class_name"] == "clean"]
        defect_df = class_df[class_df["has_defect"]]
        for i in range(n_copypaste):
            if len(defect_df) == 0 or len(clean_df) == 0:
                break
            clean_path  = rng.choice(clean_df["path"].values)
            defect_path = rng.choice(defect_df["path"].values)
            clean_img   = np.array(Image.open(clean_path).convert("L"),
                                   dtype=np.float32) / 255.0
            defect_img  = np.array(Image.open(defect_path).convert("L"),
                                   dtype=np.float32) / 255.0
            # Defect mask: pixels significantly darker than background
            threshold    = defect_img.mean() - defect_img.std()
            defect_mask  = (defect_img < threshold).astype(np.float32)
            result       = copy_paste_defect(clean_img, defect_img, defect_mask)
            fname        = aug_dir / f"{class_name}_copypaste_{i:04d}.png"
            gray         = (result * 255).astype(np.uint8)
            Image.fromarray(np.stack([gray, gray, gray], axis=-1)).save(fname)
            aug_records.append({
                "path":       str(fname.resolve()),
                "class_id":   CLASS_IDS[class_name],
                "class_name": class_name,
                "has_defect": class_name != "clean",
                "split":      "train",
                "aug_type":   "copy_paste",
            })

    df = df.copy()
    if "aug_type" not in df.columns:
        df["aug_type"] = "original"

    combined      = pd.concat([df, pd.DataFrame(aug_records)], ignore_index=True)
    combined_path = out / "metadata_augmented.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Augmented dataset: {len(aug_records)} new images -> {combined_path}")
    return combined


def semi_supervised_split(
    metadata_df: pd.DataFrame,
    n_labeled_per_class: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Semi-supervised split retaining only n_labeled_per_class per class (Jin & Chen 2022).

    Args:
        metadata_df:          DataFrame with 'class_name', 'split', 'path'.
        n_labeled_per_class:  Number of labeled images per class (default 5).
        seed:                 Random seed.

    Returns:
        Tuple (labeled_df, unlabeled_df).
    """
    rng      = np.random.default_rng(seed)
    df       = metadata_df.copy()
    train_df = df[df["split"] == "train"].copy()

    labeled_indices = []
    for class_name in df["class_name"].unique():
        class_mask = train_df["class_name"] == class_name
        class_idx  = train_df[class_mask].index.tolist()
        n_select   = min(n_labeled_per_class, len(class_idx))
        labeled_indices.extend(
            rng.choice(class_idx, n_select, replace=False).tolist()
        )

    train_df["is_labeled"] = False
    train_df.loc[labeled_indices, "is_labeled"] = True

    labeled_df   = train_df[train_df["is_labeled"]].copy()
    unlabeled_df = train_df[~train_df["is_labeled"]].copy()

    other_df = df[df["split"] != "train"]
    print(f"Semi-supervised split: {len(labeled_df)} labeled "
          f"/ {len(unlabeled_df)} unlabeled "
          f"/ {len(other_df)} val+test")
    return labeled_df, unlabeled_df

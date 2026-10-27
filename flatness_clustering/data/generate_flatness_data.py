"""
Synthetic Flatness Profile Generator — Cold-Rolling Mill
================================================================

Generates realistic flatness profiles for 14 pattern classes observed in
cold-rolled steel coils at the plant.  Each profile is a 2D array of shape
(WIDTH, LENGTH) where:

    WIDTH  = number of measurement points across the strip (cross-coil direction)
    LENGTH = number of measurement points along the coil (rolling direction)

Units: I-units  (1 I-unit = 10^-5 relative elongation difference)
  - Ideal (flat) coil: values near 0
  - Problematic coil:  peak values ±20 to ±50 I-units

Pattern classes
---------------
center_wave   — Gaussian bell in center width, sinusoidal along length
                Physics: insufficient tension at strip center during rolling
edge_wave     — Twin Gaussian bells at left/right edges
                Physics: edge elongation from work-roll edge contact
symmetric_bow — Parabolic (bowl/dome) curvature across full width
                Physics: thermal gradient — center cools slower than edges
quarter_buckle— Single off-center bell at ~25% or ~75% of width
                Physics: asymmetric roll-force distribution
flat          — Near-zero uniform profile with noise
                Physics: correctly rolled coil (reference)
random_noise  — Low-frequency 2D correlated noise without clear pattern
                Physics: mixed or unclassifiable defects

Output
------
flatness_clustering/data/synthetic_profiles/<class_name>/coil_XXXX.npz
    keys: profile (W, L) float32, coil_id, pattern, class_id, grade, customer, timestamp

flatness_clustering/data/synthetic_profiles/metadata.csv
    columns: coil_id, pattern, class_id, grade, customer, timestamp, filepath

Usage
-----
    python flatness_clustering/data/generate_flatness_data.py
    python flatness_clustering/data/generate_flatness_data.py \\
        --n_per_class 50 --width 150 --length 2000 --seed 42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "center_wave",
    "edge_wave",
    "symmetric_bow",
    "quarter_buckle",
    "flat",
    "random_noise",
    "longitudinal_bow",    # slow parabolic bow along rolling direction
    "complex_wave",        # mixed center+edge wave (hardest class)
    "herringbone_wave",    # diagonal alternating wave (uneven roll reduction)
    "periodic_mark",       # repeating roll-mark stamp along length
    # --- NEW harder classes (v2) ---
    "transient_buckle",    # localized buckle that appears/disappears along length
    "asymmetric_edge",     # one-sided edge wave with severity gradient
    "coil_set_curl",       # residual curvature from coiling — bow + periodic component
    "crossbow",            # combination: symmetric bow + quarter buckle superimposed
]
CLASS_IDS: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Representative steel grades and customers (synthetic). The customer codes are
# fictitious (suffix: NA = North America, Aero = aerospace). rng.choice draws by
# position: reordering a list changes the labels that a given seed produces.
GRADES = ["DC04", "DP600", "DP800", "HC220", "DX54D"]
CUSTOMERS = ["OEM_C_NA", "OEM_B_NA", "OEM_A_Aero", "OEM_D_Aero", "OEM_E_NA"]


# ---------------------------------------------------------------------------
# Pattern generators
# Each returns float32 array of shape (width, length) in I-units
# ---------------------------------------------------------------------------

def _make_center_wave(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Center tension defect: Gaussian bell in center, sinusoidal along length.

    Physics: insufficient rolling force at the strip center causes the center
    to elongate more than the edges, creating a waveliness pattern concentrated
    in the middle third of the strip width.
    """
    w_norm = np.linspace(-1.0, 1.0, width)   # normalized width [-1, 1]

    # Cross-width envelope: narrow or wide Gaussian bell centered at 0
    sigma_w = rng.uniform(0.15, 0.30)
    amplitude = rng.uniform(20.0, 45.0)
    width_envelope = amplitude * np.exp(-0.5 * (w_norm / sigma_w) ** 2)  # (W,)

    # Length modulation: slow sinusoid (period = 50-200 measurement points)
    period_pts = rng.integers(50, 201)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    length_mod = np.sin(2 * np.pi * l_axis / period_pts + phase)          # (L,)

    profile = width_envelope[:, np.newaxis] * length_mod[np.newaxis, :]   # (W, L)
    profile += rng.normal(0.0, 1.5, (width, length))                       # micro-texture
    return profile.astype(np.float32)


def _make_edge_wave(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Edge elongation: twin Gaussian bells at left and/or right edges.

    Physics: excessive roll force at the strip edges causes the edge material
    to elongate faster than the center, producing edge buckling.  Can be
    symmetric (both edges) or asymmetric (one edge only).
    """
    w_coord = np.linspace(0.0, 1.0, width)   # normalized [0, 1]
    sigma_e = rng.uniform(0.05, 0.14)

    # Left edge (can be absent — 25% chance)
    amp_left = rng.uniform(15.0, 40.0) if rng.random() > 0.25 else 0.0
    left_env = amp_left * np.exp(-0.5 * ((w_coord - 0.04) / sigma_e) ** 2)

    # Right edge (can be absent — 25% chance)
    amp_right = rng.uniform(15.0, 40.0) if rng.random() > 0.25 else 0.0
    right_env = amp_right * np.exp(-0.5 * ((w_coord - 0.96) / sigma_e) ** 2)

    # Ensure at least one edge is active
    if amp_left == 0.0 and amp_right == 0.0:
        amp_left = rng.uniform(20.0, 40.0)
        left_env = amp_left * np.exp(-0.5 * ((w_coord - 0.04) / sigma_e) ** 2)

    width_envelope = left_env + right_env    # (W,)

    period_pts = rng.integers(100, 401)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    length_mod = np.sin(2 * np.pi * l_axis / period_pts + phase)

    profile = width_envelope[:, np.newaxis] * length_mod[np.newaxis, :]
    profile += rng.normal(0.0, 1.5, (width, length))
    return profile.astype(np.float32)


def _make_symmetric_bow(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Parabolic cross-width curvature (bowl or dome shape).

    Physics: thermal gradient across the strip width — the center cools slower
    than the edges after rolling, creating a symmetric parabolic elongation
    difference.  Can be concave (center low) or convex (center high).
    """
    w_norm = np.linspace(-1.0, 1.0, width)   # normalized [-1, 1]

    # Parabolic cross-section: a * w^2  (a > 0 = concave, a < 0 = convex)
    sign = rng.choice([-1.0, 1.0])
    curvature_amp = sign * rng.uniform(15.0, 35.0)
    width_profile = curvature_amp * w_norm ** 2   # (W,)

    # Slow length-wise drift (long waves, period 500-2000 pts)
    period_pts = rng.integers(500, 2001)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    drift_amp = rng.uniform(0.3, 0.7)
    length_mod = 1.0 + drift_amp * np.sin(2 * np.pi * l_axis / period_pts + phase)

    profile = width_profile[:, np.newaxis] * length_mod[np.newaxis, :]
    profile += rng.normal(0.0, 1.5, (width, length))
    return profile.astype(np.float32)


def _make_quarter_buckle(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Off-center buckling at ~25% or ~75% of strip width.

    Physics: asymmetric roll-force distribution (e.g., roll crown imbalance)
    causes one quarter of the strip to elongate more than the rest.
    """
    w_coord = np.linspace(0.0, 1.0, width)

    # Bell center at 25% or 75% of width
    center_w = rng.choice([0.25, 0.75])
    sigma_w = rng.uniform(0.08, 0.20)
    amplitude = rng.uniform(18.0, 40.0)
    width_envelope = amplitude * np.exp(-0.5 * ((w_coord - center_w) / sigma_w) ** 2)

    period_pts = rng.integers(80, 251)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    length_mod = np.sin(2 * np.pi * l_axis / period_pts + phase)

    profile = width_envelope[:, np.newaxis] * length_mod[np.newaxis, :]
    profile += rng.normal(0.0, 1.5, (width, length))
    return profile.astype(np.float32)


def _make_flat(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Ideal flat profile: near-zero flatness with realistic measurement noise.

    Physics: correctly tension-leveled coil with minimal residual stress.
    The small noise represents measurement uncertainty (~1-2 I-units).
    """
    profile = rng.normal(0.0, 1.2, (width, length))
    # Very slight slow roll (large period, tiny amplitude) for realism
    period_pts = rng.integers(800, 2001)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    slow_wave = rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * l_axis / period_pts + phase)
    profile += slow_wave[np.newaxis, :]
    return profile.astype(np.float32)


def _make_random_noise(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Low-frequency 2D correlated noise (mixed/unclassifiable defect pattern).

    Simulates coils where multiple overlapping defects produce a complex
    flatness signature that does not belong to any single named pattern.
    Implemented as a superposition of several low-frequency 2D sinusoids.
    """
    profile = np.zeros((width, length), dtype=np.float32)
    n_components = rng.integers(4, 9)
    for _ in range(n_components):
        # Random spatial frequency in both directions
        freq_w = rng.uniform(0.5, 4.0)
        freq_l = rng.uniform(0.5, 8.0)
        amp = rng.uniform(5.0, 20.0)
        phase_w = rng.uniform(0.0, 2 * np.pi)
        phase_l = rng.uniform(0.0, 2 * np.pi)

        w_axis = np.linspace(0, 2 * np.pi * freq_w, width)
        l_axis = np.linspace(0, 2 * np.pi * freq_l, length)
        component = amp * np.outer(
            np.sin(w_axis + phase_w),
            np.sin(l_axis + phase_l),
        )
        profile += component

    profile += rng.normal(0.0, 2.0, (width, length))
    return profile.astype(np.float32)


def _make_longitudinal_bow(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Longitudinal bow: parabolic flatness variation along the rolling direction.

    Physics: uneven heating/cooling or tension gradients during hot-rolling
    cause the coil to bow along its length — material elongates more in the
    middle of the coil than at the head/tail ends.

    Profile is approximately uniform across the width but follows a
    parabolic arc along the length axis. May be concave or convex.

    Reference: Saas et al. 2017 — longitudinal bow identified as a distinct
    temporal pattern class in time-series clustering.
    """
    l_norm = np.linspace(-1.0, 1.0, length)
    # Parabolic shape along length: amplitude 10-35 I-units
    amplitude = rng.uniform(10.0, 35.0)
    sign = rng.choice([-1, 1])
    bow_length = sign * amplitude * (1.0 - l_norm ** 2)   # (L,) peak at centre

    # Slight width modulation (nearly flat across width)
    sigma_w = rng.uniform(0.8, 1.5)    # very broad Gaussian ≈ uniform
    w_norm = np.linspace(-1.0, 1.0, width)
    width_mod = np.exp(-0.5 * (w_norm / sigma_w) ** 2)   # (W,)

    profile = width_mod[:, np.newaxis] * bow_length[np.newaxis, :]  # (W, L)
    profile += rng.normal(0.0, 1.5, (width, length))
    return profile.astype(np.float32)


def _make_complex_wave(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Complex wave: linear superposition of center_wave and edge_wave defects.

    This is the hardest class to cluster — it looks like a blended pattern
    that sits between the centre and edge classes in feature space.  The
    mixing coefficient alpha is randomised to avoid exact repetition.

    Physics: coils that start with edge elongation and develop centre tension
    during annealing can exhibit both signatures simultaneously.

    Reference: Saas et al. 2017 — hard mixed-mode patterns stress the
    cluster algorithm's discriminative ability.
    """
    alpha = rng.uniform(0.30, 0.70)   # mixing weight for center_wave
    center = _make_center_wave(rng, width, length)
    edge   = _make_edge_wave(rng, width, length)
    profile = alpha * center + (1.0 - alpha) * edge
    return profile.astype(np.float32)


def _make_herringbone_wave(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Herringbone / chevron wave: diagonal alternating flatness pattern.

    Physics: uneven reduction across roll width during cold rolling — the strip
    elongates asymmetrically left vs. right, producing a wave that switches
    direction periodically along the rolling direction.  Typical of tandem-mill
    interstand tension imbalance or roll-bending asymmetry.

    Reference: Saas et al. 2017 — complex mode patterns stress cluster algorithms.
    """
    w_norm = np.linspace(-1.0, 1.0, width, dtype=np.float32)    # [-1, 1]
    l_norm = np.linspace(0.0, 1.0, length, dtype=np.float32)     # [0, 1]

    amplitude   = rng.uniform(8.0, 25.0)
    lambda_w    = rng.uniform(0.6, 1.2)    # wave half-period in width units
    period_l    = rng.uniform(0.08, 0.20)  # direction-switch period (fraction of length)
    noise_sigma = rng.uniform(0.8, 2.0)

    # Phase along width that alternates sign every period_l in length
    phase_shift = np.pi * np.sign(np.sin(2 * np.pi * l_norm / period_l))   # (L,)
    # Profile: sin(2π * w / λ_w + φ(l))
    profile = amplitude * np.sin(
        2 * np.pi * w_norm[None, :] / lambda_w + phase_shift[:, None]
    )   # (L, W) → transpose to (W, L)
    profile = profile.T

    profile += rng.normal(0.0, noise_sigma, profile.shape).astype(np.float32)
    return profile.astype(np.float32)


def _make_periodic_mark(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Periodic roll-mark stamp: Gaussian bump repeating every roll circumference.

    Physics: a damaged or contaminated roll surface embosses a mark onto the strip
    at every revolution.  The mark's position across the width (w0) and its period
    along the length are determined by roll geometry.

    Reference: MVTec AD (Bergmann 2019) — periodic texture defects; inspection
    roll-mark class in the plant taxonomy.
    """
    w_norm = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    l_norm = np.arange(length, dtype=np.float32)

    amplitude   = rng.uniform(10.0, 30.0)
    sigma_w     = rng.uniform(0.05, 0.20)   # width spread (fraction of half-width)
    w0          = rng.uniform(-0.7, 0.7)    # lateral position of mark center
    period      = rng.uniform(0.05, 0.15) * length   # pixels between stamps
    mark_width_l = rng.uniform(3.0, 10.0)   # longitudinal spread in pixels

    # Gaussian envelope in width direction
    width_env = np.exp(-0.5 * ((w_norm - w0) / sigma_w) ** 2)   # (W,)

    # Comb of Gaussians along length direction
    length_env = np.zeros(length, dtype=np.float32)
    offset = rng.uniform(0, period)
    k = offset
    while k < length:
        length_env += np.exp(-0.5 * ((l_norm - k) / mark_width_l) ** 2)
        k += period

    profile = amplitude * width_env[:, None] * length_env[None, :]   # (W, L)

    noise_sigma = rng.uniform(0.5, 1.5)
    profile += rng.normal(0.0, noise_sigma, profile.shape).astype(np.float32)
    return profile.astype(np.float32)


def _make_transient_buckle(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Transient buckle: localised wave that appears and disappears along length.

    Physics: temporary tension loss during speed change or coil splice causes
    a buckle that occupies only 15-40% of the coil length. The rest is flat.
    This is hard to cluster because global statistics dilute the local defect.

    Reference: Dempster (2020) ROCKET — time-series classifiers should detect
    localised events via convolutional kernels.
    """
    profile = rng.normal(0.0, 1.2, (width, length)).astype(np.float32)

    # Defect region: random start/end covering 15-40% of length
    frac = rng.uniform(0.15, 0.40)
    start = rng.integers(0, int(length * (1.0 - frac)))
    end = start + int(length * frac)

    # Inside defect region: center_wave-like pattern
    w_norm = np.linspace(-1.0, 1.0, width)
    sigma_w = rng.uniform(0.15, 0.35)
    amplitude = rng.uniform(25.0, 50.0)
    width_env = amplitude * np.exp(-0.5 * (w_norm / sigma_w) ** 2)

    period_pts = rng.integers(30, 100)
    l_axis = np.arange(end - start, dtype=np.float32)
    phase = rng.uniform(0.0, 2 * np.pi)
    length_mod = np.sin(2 * np.pi * l_axis / period_pts + phase)

    # Smooth onset/offset via Tukey-style window
    window = np.ones(end - start, dtype=np.float32)
    taper = min(50, (end - start) // 4)
    window[:taper] = np.linspace(0, 1, taper)
    window[-taper:] = np.linspace(1, 0, taper)

    defect = width_env[:, None] * (length_mod * window)[None, :]
    profile[:, start:end] += defect
    return profile


def _make_asymmetric_edge(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Asymmetric edge wave with severity gradient along the coil length.

    Physics: progressive roll wear causes the edge defect to worsen from
    head to tail of the coil. The amplitude ramps linearly from mild to
    severe. Only one edge is affected (left or right).

    This is hard because the feature statistics (mean, variance) differ from
    standard edge_wave — the severity gradient breaks the stationarity
    assumption that most global features rely on.
    """
    w_coord = np.linspace(0.0, 1.0, width)
    sigma_e = rng.uniform(0.05, 0.14)

    # Only one edge
    side = rng.choice(["left", "right"])
    if side == "left":
        edge_center = 0.04
    else:
        edge_center = 0.96
    amp_base = rng.uniform(5.0, 15.0)     # mild at coil head
    amp_max = rng.uniform(30.0, 55.0)     # severe at coil tail

    width_env = np.exp(-0.5 * ((w_coord - edge_center) / sigma_e) ** 2)

    # Severity ramp along length
    l_ramp = np.linspace(amp_base, amp_max, length)
    period_pts = rng.integers(80, 300)
    phase = rng.uniform(0.0, 2 * np.pi)
    l_axis = np.arange(length, dtype=np.float32)
    length_mod = np.sin(2 * np.pi * l_axis / period_pts + phase)

    profile = width_env[:, None] * (l_ramp * length_mod)[None, :]
    profile += rng.normal(0.0, 1.5, (width, length))
    return profile.astype(np.float32)


def _make_coil_set_curl(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Coil-set curl: residual curvature from coiling + periodic unwinding memory.

    Physics: after uncoiling, the strip retains a curvature memory from the
    coil radius. This creates a parabolic cross-width bow that oscillates
    with the coil winding period. Combines symmetric_bow + periodic features.

    Hard class: sits between symmetric_bow and periodic_mark in feature space,
    testing the pipeline's ability to distinguish compound defects.
    """
    w_norm = np.linspace(-1.0, 1.0, width)

    # Base parabolic bow (like symmetric_bow but weaker)
    curvature_amp = rng.uniform(8.0, 20.0)
    sign = rng.choice([-1.0, 1.0])
    width_bow = sign * curvature_amp * w_norm ** 2

    # Periodic modulation from coil winding (period = roll circumference)
    period = rng.uniform(80, 200)
    l_axis = np.arange(length, dtype=np.float32)
    phase = rng.uniform(0.0, 2 * np.pi)

    # The bow amplitude oscillates with period
    osc_amplitude = rng.uniform(0.4, 0.8)
    length_mod = 1.0 + osc_amplitude * np.cos(2 * np.pi * l_axis / period + phase)

    # Additional decaying envelope (curl relaxes towards tail)
    decay = np.exp(-rng.uniform(0.5, 2.0) * np.linspace(0, 1, length))
    length_mod *= decay

    profile = width_bow[:, None] * length_mod[None, :]
    profile += rng.normal(0.0, 1.0, (width, length))
    return profile.astype(np.float32)


def _make_crossbow(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    """Crossbow: symmetric bow + quarter buckle superimposed.

    Physics: simultaneous thermal crown (symmetric bow) and asymmetric
    roll-bending (quarter buckle). The resulting shape is a tilted parabola
    with a localized bump. In feature space this creates a distinct signature
    that shares features with both parent classes.

    Hard class: confusable with both symmetric_bow and quarter_buckle.
    """
    bow = _make_symmetric_bow(rng, width, length)
    buckle = _make_quarter_buckle(rng, width, length)
    alpha = rng.uniform(0.4, 0.6)
    profile = alpha * bow + (1.0 - alpha) * buckle
    return profile.astype(np.float32)


GENERATORS: Dict[str, Callable] = {
    "center_wave":      _make_center_wave,
    "edge_wave":        _make_edge_wave,
    "symmetric_bow":    _make_symmetric_bow,
    "quarter_buckle":   _make_quarter_buckle,
    "flat":             _make_flat,
    "random_noise":     _make_random_noise,
    "longitudinal_bow": _make_longitudinal_bow,
    "complex_wave":     _make_complex_wave,
    "herringbone_wave": _make_herringbone_wave,
    "periodic_mark":    _make_periodic_mark,
    # v2 harder classes
    "transient_buckle": _make_transient_buckle,
    "asymmetric_edge":  _make_asymmetric_edge,
    "coil_set_curl":    _make_coil_set_curl,
    "crossbow":         _make_crossbow,
}


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate_flatness_dataset(
    output_dir: str = "flatness_clustering/data/synthetic_profiles",
    n_per_class: int = 50,
    width: int = 150,
    length: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic flatness profiles for all 10 pattern classes.

    Args:
        output_dir:   Root directory.  Class subdirectories are created inside.
        n_per_class:  Number of coil profiles to generate per class.
        width:        Number of width measurement points (cross-coil direction).
        length:       Number of length measurement points (rolling direction).
        seed:         Random seed for reproducibility.

    Returns:
        Metadata DataFrame with columns:
            coil_id, pattern, class_id, grade, customer, timestamp, filepath
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)

    # Synthetic timestamp start (one coil produced every ~30 min)
    t_start = datetime(2024, 1, 1, 6, 0, 0)

    records = []
    coil_counter = 0

    for class_name, generator in GENERATORS.items():
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_id = CLASS_IDS[class_name]

        for i in tqdm(range(n_per_class),
                      desc=f"  [{class_name:15s}]",
                      leave=False):
            coil_id = f"C{coil_counter:05d}"
            profile = generator(rng, width, length)

            grade    = rng.choice(GRADES)
            customer = rng.choice(CUSTOMERS)
            timestamp = (t_start + timedelta(minutes=30 * coil_counter)).isoformat()

            npz_path = class_dir / f"{coil_id}.npz"
            np.savez_compressed(
                npz_path,
                profile=profile,
                coil_id=np.array(coil_id),
                pattern=np.array(class_name),
                class_id=np.array(class_id),
                grade=np.array(grade),
                customer=np.array(customer),
                timestamp=np.array(timestamp),
            )

            records.append({
                "coil_id":   coil_id,
                "pattern":   class_name,
                "class_id":  class_id,
                "grade":     grade,
                "customer":  customer,
                "timestamp": timestamp,
                "filepath":  str(npz_path.resolve()),
            })
            coil_counter += 1

    df = pd.DataFrame(records)
    meta_path = out / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"\nGenerated {len(df)} coil profiles -> {out}")
    print(f"  Classes   : {CLASS_NAMES}")
    print(f"  Shape     : ({width}, {length})  ({width * length / 1e6:.1f}M points/coil)")
    print(f"  Grades    : {GRADES}")
    print(f"  Customers : {CUSTOMERS}")
    print(f"  Metadata  : {meta_path}")

    print("\nClass distribution:")
    print(df.groupby(["pattern", "class_id"]).size()
            .rename("count").reset_index().to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic coil flatness profiles"
    )
    p.add_argument("--output_dir",  default="flatness_clustering/data/synthetic_profiles")
    p.add_argument("--n_per_class", type=int, default=50)
    p.add_argument("--width",       type=int, default=150,
                   help="Width measurement points (cross-coil direction)")
    p.add_argument("--length",      type=int, default=2000,
                   help="Length measurement points (rolling direction)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    print("Generating flatness profiles...")
    print(f"  Output    : {args.output_dir}")
    print(f"  N/class   : {args.n_per_class}")
    print(f"  Shape     : ({args.width}, {args.length})")
    print(f"  Seed      : {args.seed}\n")

    generate_flatness_dataset(
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        width=args.width,
        length=args.length,
        seed=args.seed,
    )

"""
Statistical Feature Extraction for Flatness Profiles
=====================================================

Converts a raw flatness profile (W x L float32 array) into a compact
feature vector (1105 dimensions) that captures the spatial structure of
the flatness pattern.

Five feature groups are computed:

Group A — Zonal statistics (30 features)
    The strip width is divided into 5 zones (left_edge, left_quarter,
    center, right_quarter, right_edge).  For each zone 6 statistical
    descriptors are computed: mean, std, peak-to-valley, RMS roughness,
    skewness, kurtosis.  Result: 5 x 6 = 30 features.

Group B — Cross-width shape snapshots (50 features)
    At 10 evenly-spaced positions along the coil length, a cross-width
    summary is computed: mean of left 20%, center 20%, right 20%;
    lateral symmetry index; linear fit slope.  Result: 5 x 10 = 50 features.

Group C — Spectral content (15 features)
    FFT of the mean width profile and mean length profile.  Power in
    frequency bands, dominant frequency, total variation, and
    autocorrelation length in each direction.  Result: ~15 features.

Group D — Temporal evolution features (10 features)  [Saas et al. 2017 + Asadi & Regan 2021]
    Captures longitudinal evolution patterns along the coil length using
    autocorrelation, DTW self-similarity, spectral entropy, PAA shape
    distance, and wavelet energy analysis.

    References:
        Saas et al. (2017) "Flatness Defect Classification" — DTW, PAA, temporal analysis.
        Asadi & Regan (2021) "DEC: Deep Embedded Clustering" — temporal pattern representation.

    Feature list (all computed on mean_length = mean over width axis):
    1.  length_autocorr_lag10      — normalized autocorr of mean-length-profile at lag 10
    2.  length_autocorr_lag50      — same at lag 50
    3.  length_autocorr_lag100     — same at lag 100
    4.  self_similarity_dtw        — DTW distance between first-third and last-third of profile
    5.  amplitude_trend_slope      — linear slope of peak-to-valley in sliding windows
    6.  spectral_entropy           — Shannon entropy of normalized FFT power spectrum
    7.  wavelet_energy_ratio       — energy in high-freq half vs total (via np.diff proxy)
    8.  temporal_std_of_std        — std of per-window std along length axis
    9.  edge_cross_correlation     — Pearson r between left-edge and right-edge column means
    10. paa_shape_distance         — L2 dist between PAA(first half) and PAA(second half)

Group E — ROCKET random kernel features (1000 features)  [Dempster et al. 2020]
    Applies 500 random convolutional kernels to the mean_length profile.
    Each kernel extracts 2 summary statistics:
      - max_val : maximum feature-map value (captures presence of any activation)
      - ppv     : proportion of positive values (captures how often activated)
    Kernel parameters (weights, dilation, padding) are random but fixed with seed=42,
    so results are deterministic across runs.

    Dempster et al. (2020) recommend 10,000 kernels for SOTA; 500 provides an
    excellent accuracy-to-cost tradeoff for 14-class discrimination (accuracy
    scales logarithmically with kernel count per paper Table 2).

    Reference:
        Dempster et al. (2020) "ROCKET: Exceptionally fast and accurate time series
        classification using random convolutional kernels". arXiv:1910.13051

Total: 1105 features per coil (30 + 50 + 15 + 10 + 1000).

Usage
-----
    from flatness_clustering.features.statistical_features import (
        extract_statistical_features,
        extract_batch,
    )

    feat = extract_statistical_features(profile)    # (1105,)
    feats = extract_batch(profiles_array)           # (N, 1105)
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Zone definitions (as fractions of strip width)
# ---------------------------------------------------------------------------

ZONES = [
    ("left_edge",     0.00, 0.10),
    ("left_quarter",  0.10, 0.40),
    ("center",        0.40, 0.60),
    ("right_quarter", 0.60, 0.90),
    ("right_edge",    0.90, 1.00),
]
N_ZONES = len(ZONES)                   # 5
N_ZONE_STATS = 6                       # mean, std, p2v, rms, skew, kurt
N_FEATURES_A = N_ZONES * N_ZONE_STATS  # 30

N_SNAPSHOTS = 10                       # cross-width samples along length
N_SNAPSHOT_FEATS = 5                   # left_mean, center_mean, right_mean, sym, slope
N_FEATURES_B = N_SNAPSHOTS * N_SNAPSHOT_FEATS  # 50

N_FEATURES_C = 15                      # spectral + variation + autocorr
N_FEATURES_D = 10                      # temporal evolution (Saas 2017 + Asadi 2021)
N_KERNELS_E  = 500                     # ROCKET random kernels (Dempster 2020); 500 for 14-class discrimination
N_FEATURES_E = N_KERNELS_E * 2        # max_val + ppv per kernel = 1000

N_FEATURES_TOTAL = (
    N_FEATURES_A + N_FEATURES_B + N_FEATURES_C + N_FEATURES_D + N_FEATURES_E
)  # 1105 (30 + 50 + 15 + 10 + 1000)


def feature_group_slices() -> List[tuple]:
    """(group letter, column slice) of the groups A to E, in column order.

    The single source of the column layout for code that reads features by
    group (plots and reports), so it cannot drift from the extractor.
    """
    out, start = [], 0
    for group, n in (("A", N_FEATURES_A), ("B", N_FEATURES_B), ("C", N_FEATURES_C),
                     ("D", N_FEATURES_D), ("E", N_FEATURES_E)):
        out.append((group, slice(start, start + n)))
        start += n
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_statistical_features(
    profile: np.ndarray,
    rocket_variant: str = "rocket",
) -> np.ndarray:
    """Extract the full 1105-dim feature vector from a single flatness profile.

    Args:
        profile: float32 array of shape (width, length) in I-units.

    Returns:
        Feature vector of shape (N_FEATURES_TOTAL,) = (1105,) float32.
        NaN-safe: invalid FFT or stat values are replaced with 0.
    """
    profile = profile.astype(np.float32)
    width, length = profile.shape

    feats_a = _group_a_zonal_stats(profile, width)
    feats_b = _group_b_cross_width_snapshots(profile, width, length)
    feats_c = _group_c_spectral(profile, width, length)
    feats_d = _group_d_temporal(profile, width, length)
    if rocket_variant == "minirocket":
        from flatness_clustering.features.rocket_minirocket import extract_minirocket_features
        feats_e = extract_minirocket_features(profile, n_features=N_FEATURES_E)
    else:
        feats_e = _group_e_rocket(profile)

    vec = np.concatenate([feats_a, feats_b, feats_c, feats_d, feats_e]).astype(np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def extract_batch(
    profiles: np.ndarray,
    rocket_variant: str = "rocket",
) -> np.ndarray:
    """Extract features for a batch of profiles.

    Args:
        profiles: float32 array of shape (N, width, length).
        rocket_variant: "rocket" (default, original behaviour) or
            "minirocket" (deterministic MiniRocket kernels, opt-in).

    Returns:
        Feature matrix of shape (N, 1105) float32.
    """
    return np.stack([extract_statistical_features(p, rocket_variant=rocket_variant)
                     for p in profiles], axis=0)


def feature_names() -> List[str]:
    """Return ordered feature names matching extract_statistical_features output."""
    names = []

    # Group A
    for zone_name, _, _ in ZONES:
        for stat in ["mean", "std", "peak2valley", "rms", "skewness", "kurtosis"]:
            names.append(f"zone_{zone_name}_{stat}")

    # Group B
    for snap_idx in range(N_SNAPSHOTS):
        pos_pct = int(snap_idx / (N_SNAPSHOTS - 1) * 100)
        for feat in ["left_mean", "center_mean", "right_mean", "symmetry", "slope"]:
            names.append(f"snap{pos_pct:03d}pct_{feat}")

    # Group C
    for feat in [
        "width_power_dc", "width_power_low", "width_power_mid", "width_power_high",
        "length_power_vlow", "length_power_low", "length_power_mid", "length_power_high",
        "width_peak_freq", "length_peak_freq",
        "width_total_variation", "length_total_variation",
        "width_autocorr_length", "length_autocorr_length",
        "profile_rms_global",
    ]:
        names.append(f"spectral_{feat}")

    # Group D — Temporal evolution (Saas 2017 + Asadi 2021)
    for feat in [
        "length_autocorr_lag10",
        "length_autocorr_lag50",
        "length_autocorr_lag100",
        "self_similarity_dtw",
        "amplitude_trend_slope",
        "spectral_entropy",
        "wavelet_energy_ratio",
        "temporal_std_of_std",
        "edge_cross_correlation",
        "paa_shape_distance",
    ]:
        names.append(f"temporal_{feat}")

    # Group E — ROCKET random kernels (Dempster 2020)
    for k in range(N_KERNELS_E):
        names.append(f"rocket_k{k:02d}_max")
        names.append(f"rocket_k{k:02d}_ppv")

    assert len(names) == N_FEATURES_TOTAL, f"Expected {N_FEATURES_TOTAL}, got {len(names)}"
    return names


# ---------------------------------------------------------------------------
# Group A — Zonal statistics
# ---------------------------------------------------------------------------

def _group_a_zonal_stats(profile: np.ndarray, width: int) -> np.ndarray:
    feats = []
    for _, z_start, z_end in ZONES:
        row_start = int(z_start * width)
        row_end   = max(row_start + 1, int(z_end * width))
        zone_data = profile[row_start:row_end, :].ravel()

        # NaN-aware stats: zones with partial NaN still produce a valid value.
        # All-NaN zones still return NaN and get zeroed by the final nan_to_num.
        with np.errstate(invalid="ignore"):
            mean   = float(np.nanmean(zone_data))
            std    = float(np.nanstd(zone_data))
            p2v    = float(np.nanmax(zone_data) - np.nanmin(zone_data))
            rms    = float(np.sqrt(np.nanmean(zone_data ** 2)))
        skew   = float(sp_stats.skew(zone_data, nan_policy="omit"))
        kurt   = float(sp_stats.kurtosis(zone_data, nan_policy="omit"))

        feats.extend([mean, std, p2v, rms, skew, kurt])

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Group B — Cross-width shape snapshots
# ---------------------------------------------------------------------------

def _group_b_cross_width_snapshots(
    profile: np.ndarray,
    width: int,
    length: int,
) -> np.ndarray:
    feats = []
    w20 = max(1, int(0.20 * width))

    snap_positions = np.linspace(0, length - 1, N_SNAPSHOTS, dtype=int)

    for l_pos in snap_positions:
        cross = profile[:, l_pos]

        with np.errstate(invalid="ignore"):
            left_mean   = float(np.nanmean(cross[:w20]))
            right_mean  = float(np.nanmean(cross[-w20:]))
            c_start = width // 2 - w20 // 2
            center_mean = float(np.nanmean(cross[c_start:c_start + w20]))
            sigma = float(np.nanstd(cross)) + 1e-8
        symmetry = abs(left_mean - right_mean) / sigma

        x = np.linspace(-1.0, 1.0, width)
        # polyfit rejects NaN: mask them before fitting so the slope is defined.
        valid = np.isfinite(cross)
        if valid.sum() >= 2:
            slope = float(np.polyfit(x[valid], cross[valid], 1)[0])
        else:
            slope = 0.0

        feats.extend([left_mean, center_mean, right_mean, symmetry, slope])

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Group C — Spectral content
# ---------------------------------------------------------------------------

def _group_c_spectral(
    profile: np.ndarray,
    width: int,
    length: int,
) -> np.ndarray:
    # NaN-aware axis reductions: a single NaN would otherwise poison the FFT.
    with np.errstate(invalid="ignore"):
        mean_width  = np.nanmean(profile, axis=1)
        mean_length = np.nanmean(profile, axis=0)
    mean_width  = np.nan_to_num(mean_width,  nan=0.0, posinf=0.0, neginf=0.0)
    mean_length = np.nan_to_num(mean_length, nan=0.0, posinf=0.0, neginf=0.0)

    feats = []

    fft_w = np.abs(np.fft.rfft(mean_width))
    freqs_w = np.fft.rfftfreq(width)
    feats += _band_powers(fft_w, freqs_w, [(0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5)])
    feats.append(float(freqs_w[np.argmax(fft_w[1:]) + 1]) if len(fft_w) > 1 else 0.0)

    fft_l = np.abs(np.fft.rfft(mean_length))
    freqs_l = np.fft.rfftfreq(length)
    feats += _band_powers(fft_l, freqs_l, [(0, 0.002), (0.002, 0.01), (0.01, 0.05), (0.05, 0.5)])
    feats.append(float(freqs_l[np.argmax(fft_l[1:]) + 1]) if len(fft_l) > 1 else 0.0)

    feats.append(float(np.abs(np.diff(mean_width)).sum()))
    feats.append(float(np.abs(np.diff(mean_length)).sum()))

    feats.append(_autocorr_length(mean_width))
    feats.append(_autocorr_length(mean_length))

    feats.append(float(np.sqrt(np.nanmean(profile ** 2))))

    assert len(feats) == N_FEATURES_C, f"Expected {N_FEATURES_C} spectral feats, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Group D — Temporal evolution features  (Saas et al. 2017 + Asadi & Regan 2021)
# ---------------------------------------------------------------------------

def _group_d_temporal(
    profile: np.ndarray,
    width: int,
    length: int,
) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mean_length = np.nanmean(profile, axis=0)
    mean_length = np.nan_to_num(mean_length, nan=0.0, posinf=0.0, neginf=0.0)

    feats = []

    feats.append(_autocorr_at_lag(mean_length, 10))
    feats.append(_autocorr_at_lag(mean_length, 50))
    feats.append(_autocorr_at_lag(mean_length, 100))

    third = max(1, length // 3)
    feats.append(_simple_dtw(mean_length[:third], mean_length[-third:]))

    n_windows = 10
    win_size = max(1, length // n_windows)
    pv_values = []
    for w in range(n_windows):
        seg = mean_length[w * win_size: (w + 1) * win_size]
        if len(seg) > 0:
            pv_values.append(float(seg.max() - seg.min()))
    if len(pv_values) >= 2:
        slope = float(np.polyfit(np.arange(len(pv_values), dtype=np.float32), pv_values, 1)[0])
    else:
        slope = 0.0
    feats.append(slope)

    fft_mag = np.abs(np.fft.rfft(mean_length))
    power = fft_mag ** 2
    total_power = power.sum()
    if total_power > 1e-12:
        p_norm = power / total_power
        p_norm = p_norm[p_norm > 1e-12]
        spectral_entropy = float(-(p_norm * np.log(p_norm)).sum())
    else:
        spectral_entropy = 0.0
    feats.append(spectral_entropy)

    total_energy = float((mean_length ** 2).sum()) + 1e-12
    feats.append(float((np.diff(mean_length) ** 2).sum()) / total_energy)

    std_values = []
    for w in range(n_windows):
        seg = mean_length[w * win_size: (w + 1) * win_size]
        if len(seg) > 1:
            std_values.append(float(seg.std()))
    feats.append(float(np.std(std_values)) if len(std_values) > 1 else 0.0)

    edge_w = max(1, int(0.05 * width))
    left_col  = profile[:edge_w, :].mean(axis=0)
    right_col = profile[-edge_w:, :].mean(axis=0)
    if left_col.std() > 1e-8 and right_col.std() > 1e-8:
        r = float(np.corrcoef(left_col, right_col)[0, 1])
    else:
        r = 0.0
    feats.append(r)

    n_paa = 8
    half = length // 2
    feats.append(float(np.linalg.norm(
        _paa(mean_length[:half], n_paa) - _paa(mean_length[half:2 * half], n_paa)
    )))

    assert len(feats) == N_FEATURES_D, f"Expected {N_FEATURES_D} temporal feats, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Group E — ROCKET random kernel features  (Dempster et al. 2020)
# ---------------------------------------------------------------------------

_ROCKET_SEED = 42
_ROCKET_LENGTHS = [7, 9, 11]
_rocket_kernel_cache: dict = {}


def _build_rocket_kernels(n_kernels: int, signal_length: int, seed: int = 42):
    """Generate fixed ROCKET kernel parameters (Dempster et al. 2020)."""
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(n_kernels):
        k_len = int(rng.choice(_ROCKET_LENGTHS))
        w = rng.standard_normal(k_len).astype(np.float32)
        w -= w.mean()
        bias = float(rng.uniform(-1.0, 1.0))
        max_exp = max(0.0, np.log2((signal_length - 1) / (k_len - 1 + 1e-8)))
        dilation = max(1, int(2 ** float(rng.uniform(0.0, max_exp))))
        pad = bool(rng.choice([True, False]))
        kernels.append({"weights": w, "bias": bias, "dilation": dilation, "pad": pad})
    return kernels


def _convolve1d_rocket(
    signal: np.ndarray,
    weights: np.ndarray,
    dilation: int,
    pad: bool,
    bias: float,
) -> np.ndarray:
    """Vectorised dilated 1-D convolution for ROCKET kernels."""
    k = len(weights)
    eff_len = (k - 1) * dilation + 1
    if pad:
        p = eff_len - 1
        sig = np.pad(signal, (p // 2, p - p // 2), mode="constant")
    else:
        sig = signal
    out_len = max(0, len(sig) - eff_len + 1)
    if out_len <= 0:
        return np.array([bias], dtype=np.float32)
    offsets = np.arange(k) * dilation
    starts = np.arange(out_len)
    windows = sig[starts[:, None] + offsets[None, :]]   # (out_len, k)
    return (windows @ weights.astype(np.float64)).astype(np.float32) + bias


def _group_e_rocket(
    profile: np.ndarray,
    n_kernels: int = N_KERNELS_E,
    seed: int = _ROCKET_SEED,
) -> np.ndarray:
    """ROCKET Group E: 2*n_kernels random-kernel features (Dempster et al. 2020).

    Each kernel → max_val (presence) + ppv (proportion of positive values).
    Kernels are random but fixed (seed=42) → fully deterministic.

    Reference: Dempster et al. (2020) arXiv:1910.13051
    """
    signal = profile.mean(axis=0).astype(np.float32)
    L = len(signal)
    cache_key = (n_kernels, L, seed)
    if cache_key not in _rocket_kernel_cache:
        _rocket_kernel_cache[cache_key] = _build_rocket_kernels(n_kernels, L, seed)
    kernels = _rocket_kernel_cache[cache_key]

    feats = []
    for kern in kernels:
        fmap = _convolve1d_rocket(
            signal, kern["weights"], kern["dilation"], kern["pad"], kern["bias"]
        )
        feats.append(float(fmap.max()))
        feats.append(float((fmap > 0).mean()))
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag <= 0 or lag >= n:
        return 0.0
    a, b = x[:n - lag], x[lag:]
    std_a, std_b = a.std(), b.std()
    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (std_a * std_b))


def _paa(x: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation."""
    n = len(x)
    if n == 0 or n_segments <= 0:
        return np.zeros(n_segments, dtype=np.float32)
    n_segments = min(n_segments, n)
    indices = np.array_split(np.arange(n), n_segments)
    return np.array([x[idx].mean() for idx in indices], dtype=np.float32)


def _simple_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance (DP), normalized by path length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m]) / (n + m)


def _band_powers(fft_magnitudes: np.ndarray, frequencies: np.ndarray, bands: list) -> list:
    powers = []
    for f_low, f_high in bands:
        mask = (frequencies >= f_low) & (frequencies < f_high)
        powers.append(float((fft_magnitudes[mask] ** 2).sum()) if mask.any() else 0.0)
    return powers


def _autocorr_length(signal: np.ndarray) -> float:
    """Return lag at which autocorrelation first crosses zero."""
    n = len(signal)
    if n < 4:
        return 0.0
    sig = signal - signal.mean()
    norm = float((sig ** 2).sum()) + 1e-10
    for lag in range(1, n):
        if float((sig[:n - lag] * sig[lag:]).sum()) / norm <= 0.0:
            return float(lag)
    return float(n)

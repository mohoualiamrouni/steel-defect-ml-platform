r"""MiniRocket Group E -- deterministic kernel features for time-series clustering.

Pure-numpy implementation of MiniRocket (Dempster, Schmidt, Webb 2020,
arXiv:2012.08791) used as a drop-in replacement for the random-kernel
ROCKET features in ``statistical_features._group_e_rocket``.

Key algorithmic differences vs.\ classical ROCKET (1910.13051):

1. **Kernels are deterministic** -- exactly 84 nine-bit binary patterns
   with three "high" positions out of nine, weights in $\{-1, 2\}$.
   $\binom{9}{3} = 84$, all kernels sum to zero by construction.
2. **No random weights, no random kernel lengths** -- the only source
   of variability is the choice of dilation per kernel.
3. **Bias values are data-dependent** -- median of the per-coil
   convolution output, instead of a fixed random bias.
4. **Output dimension is preserved at 1000** for the smoke-test
   constraint that downstream PCA / clustering see no shape change.
5. **Vectorised**: all 84 kernels are applied in one matmul per dilation
   so the pure-numpy variant is significantly faster than the per-kernel
   loop in the random-ROCKET reference.

Public API:
    extract_minirocket_features(profile_2d, n_features=1000, seed=42)
        -> ndarray of shape (n_features,) float32
"""
from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# 84 deterministic kernels (Dempster et al. 2020, eq. 3.1 / Table 1)
# ---------------------------------------------------------------------------
# Each kernel is a 9-element vector with three '2' entries and six '-1'.
# All possible such patterns: C(9, 3) = 84.  Sum of each kernel is zero.

def _build_84_kernels() -> np.ndarray:
    """Return all 84 nine-bit MiniRocket kernels as a (84, 9) float32 array."""
    kernels = np.full((84, 9), -1.0, dtype=np.float32)
    for i, idx in enumerate(combinations(range(9), 3)):
        kernels[i, list(idx)] = 2.0
    return kernels


_MINIROCKET_KERNELS = _build_84_kernels()
_KERNEL_LEN = 9


# ---------------------------------------------------------------------------
# Dilation schedule
# ---------------------------------------------------------------------------

def _select_dilations(signal_length: int, n_dilations: int = 6) -> List[int]:
    """Geometric dilation schedule per Dempster 2020 §3.3.

    Dilations span 1 .. (L-1)/(K-1), spaced exponentially.  Six dilations
    is the smallest schedule that keeps the (84 * 6 * 2) = 1008 feature
    count close to ROCKET's 1000.
    """
    if signal_length <= _KERNEL_LEN:
        return [1]
    max_exp = np.log2((signal_length - 1) / (_KERNEL_LEN - 1))
    raw = 2.0 ** np.linspace(0.0, max_exp, n_dilations)
    out = sorted({max(1, int(round(d))) for d in raw})
    return out


# ---------------------------------------------------------------------------
# Vectorised convolution: ALL 84 kernels in one matmul per dilation
# ---------------------------------------------------------------------------

def _dilated_conv_all_kernels(signal: np.ndarray, dilation: int) -> np.ndarray:
    """Apply all 84 MiniRocket kernels at one dilation in a single matmul.

    Parameters
    ----------
    signal : ndarray, shape (L,)
    dilation : int

    Returns
    -------
    ndarray of shape (84, out_len), where out_len = max(1, L - 8*dilation).
    """
    eff_len = (_KERNEL_LEN - 1) * dilation + 1
    out_len = len(signal) - eff_len + 1
    if out_len <= 0:
        return np.zeros((84, 1), dtype=np.float32)
    offsets = np.arange(_KERNEL_LEN) * dilation
    starts = np.arange(out_len)
    windows = signal[starts[:, None] + offsets[None, :]]   # (out_len, 9)
    # Single matmul: (84, 9) @ (9, out_len) -> (84, out_len)
    return (_MINIROCKET_KERNELS @ windows.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

def extract_minirocket_features(
    profile: np.ndarray,
    n_features: int = 1000,
    seed: int = 42,  # accepted for API symmetry; MiniRocket is deterministic
) -> np.ndarray:
    """MiniRocket features on the length-mean of a 2D profile.

    Parameters
    ----------
    profile : ndarray, shape (W, L)
        Coil profile, same input format as ``_group_e_rocket``.
    n_features : int
        Target feature count.  Default 1000 to match ROCKET dimension.
    seed : int
        Unused (MiniRocket is deterministic); kept for API parity.

    Returns
    -------
    ndarray of shape (n_features,) float32, NaN-safe.
    """
    signal = profile.mean(axis=0).astype(np.float32)
    L = len(signal)
    dilations = _select_dilations(L, n_dilations=6)

    # Pre-allocate: 84 kernels * len(dilations) * 2 stats per (kernel, dilation).
    n_pairs = 84 * len(dilations)
    out = np.empty(n_pairs * 2, dtype=np.float32)

    write = 0
    for dil in dilations:                                  # ~6 dilations
        fmaps = _dilated_conv_all_kernels(signal, dil)     # (84, out_len)
        biases = np.median(fmaps, axis=1, keepdims=True)   # (84, 1)
        ppv = (fmaps > biases).mean(axis=1).astype(np.float32)  # (84,)
        mx = fmaps.max(axis=1).astype(np.float32)               # (84,)
        # Interleave (ppv, max) to keep the layout used by ROCKET features.
        out[write : write + 168 : 2] = ppv
        out[write + 1 : write + 168 : 2] = mx
        write += 168

    # Truncate or pad to exactly n_features (parity with ROCKET).
    if out.size >= n_features:
        out = out[:n_features]
    else:
        out = np.concatenate([out, np.zeros(n_features - out.size, dtype=np.float32)])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

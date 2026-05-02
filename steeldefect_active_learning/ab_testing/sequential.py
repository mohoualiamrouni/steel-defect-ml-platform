"""
Sequential / always-valid A/B testing.

The fixed-horizon p-values produced by `metrics.statistical_significance_test`
are only valid if you commit to a sample size up front and look at the test
exactly once. In the plant's continuous-monitoring use case (the new inspection model
runs in shadow mode and is compared against the incumbent on every batch), the
analyst will look at the data many times — each peek inflates the type-I error.

Two correct approaches are implemented:

  1. **mSPRT** (Mixture Sequential Probability Ratio Test, Robbins 1970;
     Johari, Pekelis, Walsh 2017 "Always Valid Inference"). Returns an
     "always-valid p-value" that can be checked at any time.

  2. **O'Brien-Fleming** group-sequential boundaries (1979). Pre-defined
     interim looks; rejection threshold tightens at early peeks so the overall
     type-I error stays at α.

Both expose:
    - sequential_p_value(t):  returns the always-valid p at time t (t = current n)
    - boundary(t, alpha):     critical value at time t
    - decide(t, alpha):       'continue' | 'stop_reject' | 'stop_accept'

References:
    Robbins (1970) "Statistical methods related to the law of the iterated
        logarithm", Ann. Math. Stat.
    Johari, Pekelis, Walsh (2017) "Always Valid Inference: Bringing Sequential
        Analysis to A/B Testing", arXiv:1512.04922.
    O'Brien & Fleming (1979) "A multiple testing procedure for clinical
        trials", Biometrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# mSPRT (always-valid p-values)
# ---------------------------------------------------------------------------

@dataclass
class MSPRTState:
    """Sufficient statistics of a two-sample mSPRT for the difference of means.

    Tracks running counts and sums for each arm. The mixing prior on the effect
    size has variance tau_sq; smaller values concentrate prior mass near 0
    (better for detecting tiny effects), larger values are more agnostic.
    """
    tau_sq: float = 1.0
    sigma_sq: float = 0.025         # variance of confidence-style metrics ~ 0.15²

    n_a: int = 0
    n_b: int = 0
    sum_a: float = 0.0
    sum_b: float = 0.0
    sum_sq_a: float = 0.0
    sum_sq_b: float = 0.0

    history: List[Dict] = field(default_factory=list)

    def update(self, x_a: Optional[float] = None, x_b: Optional[float] = None) -> None:
        """Add one observation to each arm (either may be None to be skipped)."""
        if x_a is not None:
            self.n_a += 1
            self.sum_a += float(x_a)
            self.sum_sq_a += float(x_a) ** 2
        if x_b is not None:
            self.n_b += 1
            self.sum_b += float(x_b)
            self.sum_sq_b += float(x_b) ** 2

    def update_batch(self, xs_a: np.ndarray, xs_b: np.ndarray) -> None:
        """Vectorised update from two equal- or different-sized arrays."""
        xs_a = np.asarray(xs_a, dtype=float)
        xs_b = np.asarray(xs_b, dtype=float)
        self.n_a += len(xs_a)
        self.n_b += len(xs_b)
        self.sum_a += float(xs_a.sum())
        self.sum_b += float(xs_b.sum())
        self.sum_sq_a += float((xs_a ** 2).sum())
        self.sum_sq_b += float((xs_b ** 2).sum())

    # ------------------------------------------------------------------
    def mean_diff(self) -> float:
        if self.n_a == 0 or self.n_b == 0:
            return 0.0
        return self.sum_b / self.n_b - self.sum_a / self.n_a

    def empirical_variance(self) -> float:
        """Pooled variance estimate; falls back to self.sigma_sq if too few obs."""
        if self.n_a < 2 or self.n_b < 2:
            return self.sigma_sq
        var_a = (self.sum_sq_a - self.sum_a ** 2 / self.n_a) / max(1, self.n_a - 1)
        var_b = (self.sum_sq_b - self.sum_b ** 2 / self.n_b) / max(1, self.n_b - 1)
        return float(max(1e-12, (var_a + var_b) / 2.0))

    # ------------------------------------------------------------------
    def always_valid_p(self) -> float:
        """Mixture-SPRT always-valid p-value (two-sided test of μ_A = μ_B).

        Closed form for the Gaussian likelihood with a Gaussian mixing prior on
        the effect (Johari et al. 2017, eq. 7):
            Λ(n) = sqrt(σ² / (σ² + n_eff τ²)) ·
                   exp( n_eff² τ² (Δ̄)² / (2 σ² (σ² + n_eff τ²)) )
            p_av(n) = min(1, 1 / Λ(n))

        where n_eff = (n_a · n_b) / (n_a + n_b) is the effective sample size for
        the difference of means.

        The p-value is monotone non-increasing only in expectation — at any
        finite n the value can fluctuate, but P(min_n p_av < α) ≤ α under H0
        for all n simultaneously, which is what "always valid" means.
        """
        if self.n_a < 1 or self.n_b < 1:
            return 1.0
        n_eff = (self.n_a * self.n_b) / (self.n_a + self.n_b)
        sigma_sq = self.empirical_variance()
        tau_sq = self.tau_sq
        delta = self.mean_diff()
        denom = sigma_sq + n_eff * tau_sq
        log_lambda = (
            0.5 * math.log(sigma_sq / denom)
            + (n_eff ** 2) * tau_sq * (delta ** 2) / (2.0 * sigma_sq * denom)
        )
        p = float(math.exp(-log_lambda))
        return float(min(1.0, p))

    def decide(self, alpha: float = 0.05) -> Dict:
        """Return current decision: 'continue' or 'stop_reject_h0'.

        Records the current state in `self.history` so a wall-time series of
        p-values can be plotted alongside the cumulative effect size.
        """
        p_av = self.always_valid_p()
        decision = "stop_reject_h0" if p_av < alpha else "continue"
        rec = {
            "n_a":      self.n_a,
            "n_b":      self.n_b,
            "delta":    self.mean_diff(),
            "p_av":     p_av,
            "decision": decision,
        }
        self.history.append(rec)
        return rec


# ---------------------------------------------------------------------------
# O'Brien-Fleming group-sequential boundaries
# ---------------------------------------------------------------------------

def obrien_fleming_boundary(
    look: int,
    n_looks: int,
    alpha: float = 0.05,
) -> float:
    """Two-sided O'Brien-Fleming critical z-value at interim look ``look``.

    Boundary at look k of K looks: z_k = z_{α/2} · sqrt(K / k).

    Args:
        look:    1-indexed look number (1 ≤ look ≤ n_looks).
        n_looks: total number of planned looks.
        alpha:   overall family-wise error rate.

    Returns:
        Two-sided critical z-value. Reject H0 if |z_obs| ≥ z_k.
    """
    if not (1 <= look <= n_looks):
        raise ValueError(f"look must be in [1, {n_looks}], got {look}")
    from scipy.stats import norm
    z_alpha_2 = norm.ppf(1 - alpha / 2)
    return float(z_alpha_2 * math.sqrt(n_looks / look))


def obrien_fleming_decide(
    z_observed: float,
    look: int,
    n_looks: int,
    alpha: float = 0.05,
) -> Dict:
    """Decision at one O'Brien-Fleming interim look.

    Returns a dict with: critical_z, |z|, decision (`continue` or
    `stop_reject_h0`), and the look index. To pre-register the boundary at all
    looks, iterate this for k = 1..K.
    """
    z_crit = obrien_fleming_boundary(look, n_looks, alpha)
    decision = "stop_reject_h0" if abs(z_observed) >= z_crit else "continue"
    return {
        "look":       int(look),
        "n_looks":    int(n_looks),
        "z_observed": float(z_observed),
        "z_critical": z_crit,
        "alpha":      float(alpha),
        "decision":   decision,
    }


# ---------------------------------------------------------------------------
# Anytime-valid confidence sequence (paired with mSPRT)
# ---------------------------------------------------------------------------

def confidence_sequence(
    state: MSPRTState,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Anytime-valid confidence interval for μ_B − μ_A at the current time.

    Inverts the mSPRT: CS_n = { Δ : p_av(Δ; n) ≥ α }. Closed form for the
    Gaussian-mixture mSPRT — see Johari et al. 2017, Theorem 4.

    Returns (lower, upper) such that with probability ≥ 1 − α the true
    difference is contained at *every* time n simultaneously.
    """
    if state.n_a < 1 or state.n_b < 1:
        return (float("-inf"), float("inf"))
    n_eff = (state.n_a * state.n_b) / (state.n_a + state.n_b)
    sigma_sq = state.empirical_variance()
    tau_sq = state.tau_sq
    delta = state.mean_diff()
    denom = sigma_sq + n_eff * tau_sq
    # Derived from p_av < α  ⇔  (Δ̄ − Δ)² < (2 σ² denom / n_eff²) · log(1/α · sqrt(σ²/denom))
    log_term = math.log(1.0 / alpha) + 0.5 * math.log(sigma_sq / denom)
    if log_term <= 0:
        return (float("-inf"), float("inf"))
    half = math.sqrt(2.0 * sigma_sq * denom * log_term) / n_eff
    return (delta - half, delta + half)

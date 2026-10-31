"""
Bayesian diagnostics for the inspection model-replacement decision.

Three posterior models are exposed:
    - Paired Dirichlet for accuracy (``paired_accuracy_posterior``). Both models scored
      the same images, so the posterior is put on the four cells of the paired table:
      with a uniform Dirichlet(1, 1, 1, 1) prior, P(acc_B > acc_A) = P(theta > 1/2)
      with theta ~ Beta(1 + n01, 1 + n10), computed exactly; it equals 1 minus the
      one-sided mid-p McNemar value. The credible interval and the expected losses
      come from Dirichlet draws.

    - Beta-Binomial for two independent binomial arms (``beta_binomial_decision``).
      Each arm's success rate has a Beta prior; the posterior is
      Beta(α + successes, β + failures), and P(p_B > p_A) is computed via Monte-Carlo.
      The agreement block uses it to compare the disagreement rate with the agreement
      rate.

    - Normal-Normal for continuous outcomes (e.g., the confidences). With a
      vague prior the posterior on the mean reduces to a Student-t centred at
      the sample mean — for n in the hundreds the t is essentially Gaussian,
      so we use a Gaussian posterior with σ̂² / n variance; the two arms are
      treated as independent.

Every block reports:
    - posterior probability P(B > A)
    - expected loss E[max(0, μ_A − μ_B)] (regret if you ship B and B is worse)
    - credible interval at the requested level
    - a ``decision`` field (ship B iff P(B > A) ≥ threshold): a diagnostic label,
      since the promotion gate thresholds no posterior

References:
    Carpenter (2008) "Bayesian A/B testing".
    Stucchio (2015) "Bayesian A/B Testing at VWO" (Beta-Binomial expected loss).
    Kruschke (2018) "Rejecting or accepting parameter values in Bayesian
        estimation".
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .paired import paired_counts, paired_counts_errors, posterior_b_better


# ---------------------------------------------------------------------------
# Beta-Binomial: P(p_B > p_A), credible interval, expected loss
# ---------------------------------------------------------------------------

def beta_binomial_decision(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 200_000,
    cred_level: float = 0.95,
    decision_threshold: float = 0.95,
    seed: Optional[int] = 0,
) -> Dict:
    """Posterior decision rule for two binomial arms (Beta priors).

    Args:
        successes_a, trials_a: arm A counts
        successes_b, trials_b: arm B counts
        prior_alpha, prior_beta: shared symmetric Beta(α,β) prior. Default
            (1, 1) is uniform = Bayes-Laplace.
        n_samples: Monte-Carlo samples from each posterior.
        cred_level: credible-interval level for p_B − p_A.
        decision_threshold: P(B > A) cutoff for ship/no-ship.
        seed: RNG seed.

    Returns:
        Dict with: p_b_gt_a, expected_loss_ship_b, expected_loss_keep_a,
        cred_interval_diff, posterior_mean_a, posterior_mean_b, decision,
        n_samples.
    """
    rng = np.random.default_rng(seed)
    a_post_a = prior_alpha + successes_a
    b_post_a = prior_beta + (trials_a - successes_a)
    a_post_b = prior_alpha + successes_b
    b_post_b = prior_beta + (trials_b - successes_b)

    pa = rng.beta(a_post_a, b_post_a, size=n_samples)
    pb = rng.beta(a_post_b, b_post_b, size=n_samples)

    diff = pb - pa
    p_b_gt_a = float(np.mean(diff > 0))
    # Expected loss if we ship B (regret = max(0, p_A − p_B) given posterior).
    loss_ship_b = float(np.mean(np.maximum(0.0, pa - pb)))
    # Expected loss if we keep A (regret = max(0, p_B − p_A)).
    loss_keep_a = float(np.mean(np.maximum(0.0, pb - pa)))

    lo, hi = _credible_interval(diff, cred_level)
    decision = "ship_b" if p_b_gt_a >= decision_threshold else "keep_a"

    return {
        "model":                    "beta_binomial",
        "p_b_gt_a":                 p_b_gt_a,
        "expected_loss_ship_b":     loss_ship_b,
        "expected_loss_keep_a":     loss_keep_a,
        "cred_interval_diff":       (float(lo), float(hi)),
        "cred_level":               float(cred_level),
        "posterior_mean_a":         float(a_post_a / (a_post_a + b_post_a)),
        "posterior_mean_b":         float(a_post_b / (a_post_b + b_post_b)),
        "decision":                 decision,
        "decision_threshold":       float(decision_threshold),
        "n_samples":                int(n_samples),
    }


# ---------------------------------------------------------------------------
# Normal-Normal: P(μ_B > μ_A), credible interval, expected loss
# ---------------------------------------------------------------------------

def normal_decision(
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int = 200_000,
    cred_level: float = 0.95,
    decision_threshold: float = 0.95,
    seed: Optional[int] = 0,
) -> Dict:
    """Posterior decision rule for two Gaussian arms (vague prior).

    Each arm's mean has posterior N(μ̂, σ̂² / n). For confidence-style metrics
    (n ≳ 100) the t-correction is negligible. Uses Monte-Carlo on the
    posterior so the joint distribution of μ_B − μ_A is exact.

    Args:
        a, b: 1-D arrays of observations from arm A and arm B.
        n_samples: Monte-Carlo samples.
        cred_level: credible-interval level for the difference.
        decision_threshold: P(B > A) cutoff for ship/no-ship.
        seed: RNG seed.

    Returns:
        Same keys as beta_binomial_decision plus mu_a, mu_b, sigma_a, sigma_b, n_a, n_b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {"model": "normal", "applicable": False, "reason": "n < 2"}

    mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
    sd_a = float(np.std(a, ddof=1)) / math.sqrt(n_a)
    sd_b = float(np.std(b, ddof=1)) / math.sqrt(n_b)

    rng = np.random.default_rng(seed)
    post_a = rng.normal(mu_a, max(sd_a, 1e-12), size=n_samples)
    post_b = rng.normal(mu_b, max(sd_b, 1e-12), size=n_samples)
    diff = post_b - post_a
    p_b_gt_a = float(np.mean(diff > 0))
    loss_ship_b = float(np.mean(np.maximum(0.0, post_a - post_b)))
    loss_keep_a = float(np.mean(np.maximum(0.0, post_b - post_a)))

    lo, hi = _credible_interval(diff, cred_level)
    decision = "ship_b" if p_b_gt_a >= decision_threshold else "keep_a"

    return {
        "model":                "normal",
        "applicable":           True,
        "p_b_gt_a":             p_b_gt_a,
        "expected_loss_ship_b": loss_ship_b,
        "expected_loss_keep_a": loss_keep_a,
        "cred_interval_diff":   (float(lo), float(hi)),
        "cred_level":           float(cred_level),
        "mu_a":                 mu_a,
        "mu_b":                 mu_b,
        "sigma_a":              float(np.std(a, ddof=1)),
        "sigma_b":              float(np.std(b, ddof=1)),
        "n_a":                  n_a,
        "n_b":                  n_b,
        "decision":             decision,
        "decision_threshold":   float(decision_threshold),
        "n_samples":            int(n_samples),
    }


# ---------------------------------------------------------------------------
# Convenience: Bayesian decision from an A/B DataFrame
# ---------------------------------------------------------------------------

def paired_accuracy_posterior(
    counts: Mapping[str, Any],
    decision_threshold: float = 0.95,
    cred_level: float = 0.95,
    n_samples: int = 200_000,
    seed: Optional[int] = 0,
) -> Dict:
    """Posterior of acc_B - acc_A from the paired 2x2 counts.

    With a uniform Dirichlet(1, 1, 1, 1) prior on the four cells, the posterior is
    Dirichlet(1 + n11, 1 + n10, 1 + n01, 1 + n00), and
    P(acc_B > acc_A) = P(p01 > p10) = P(theta > 1/2) with theta ~ Beta(1 + n01, 1 + n10):
    computed exactly, it equals 1 minus the one-sided mid-p McNemar value. The credible
    interval and the expected losses come from Dirichlet draws.

    Args:
        counts: a valid ``paired_counts`` block (``paired.paired_counts``).

    Returns:
        Dict with: model, prior, p_b_gt_a, posterior_mean_a, posterior_mean_b,
        cred_interval_diff, expected_loss_ship_b, expected_loss_keep_a, cred_level,
        n_samples, n_discordant, decision (a diagnostic label), decision_threshold.
    """
    n11, n10 = counts["both_correct"], counts["only_a_correct"]
    n01, n00 = counts["only_b_correct"], counts["both_wrong"]
    n = n11 + n10 + n01 + n00
    p_b_gt_a = posterior_b_better(n10, n01)
    draws = np.random.default_rng(seed).dirichlet([1 + n11, 1 + n10, 1 + n01, 1 + n00],
                                                  size=n_samples)
    diff = draws[:, 2] - draws[:, 1]          # p01 - p10 = acc_B - acc_A
    lo, hi = _credible_interval(diff, cred_level)
    return {
        "model":                "dirichlet_paired",
        "prior":                "Dirichlet(1,1,1,1) on (n11, n10, n01, n00)",
        "p_b_gt_a":             p_b_gt_a,
        "posterior_mean_a":     float((2 + n11 + n10) / (n + 4)),
        "posterior_mean_b":     float((2 + n11 + n01) / (n + 4)),
        "cred_interval_diff":   (float(lo), float(hi)),
        "expected_loss_ship_b": float(np.mean(np.maximum(0.0, -diff))),
        "expected_loss_keep_a": float(np.mean(np.maximum(0.0, diff))),
        "cred_level":           float(cred_level),
        "n_samples":            int(n_samples),
        "n_discordant":         int(n10 + n01),
        "decision":             "ship_b" if p_b_gt_a >= decision_threshold else "keep_a",
        "decision_threshold":   float(decision_threshold),
    }


def bayesian_summary(
    df: pd.DataFrame,
    decision_threshold: float = 0.95,
    cred_level: float = 0.95,
    n_samples: int = 200_000,
    seed: Optional[int] = 0,
    *,
    counts: Optional[Mapping[str, Any]] = None,
) -> Dict:
    """Run the Bayesian diagnostic blocks on the A/B df.

    Selects the models based on what is available:
      - If 'agrees' is present, runs Beta-Binomial on the agreement rate.
      - If the paired counts are valid, runs the exact paired posterior on accuracy
        (:func:`paired_accuracy_posterior`). ``counts`` is the ``paired_counts`` block;
        when None it is computed from df, whose rows must then satisfy the contract
        of ``paired.paired_counts``.
      - If 'confidence_a' / 'confidence_b' are present, runs Normal-Normal on
        the confidence difference (interpreted as B − A).

    Returns a dict with up to three sub-dicts: 'agreement', 'accuracy',
    'confidence'. Their ``decision`` fields are diagnostic labels: the promotion
    gate thresholds no posterior.
    """
    out: Dict = {}

    if "agrees" in df.columns:
        # Symmetric formulation: B "wins" each pair where it matches the
        # majority/oracle. Use simple agreement rate as a proxy decision metric;
        # the question is whether disagreement rate is significantly different.
        s = int(df["agrees"].sum())
        n = int(len(df))
        out["agreement"] = beta_binomial_decision(
            successes_a=s, trials_a=n,
            successes_b=n - s, trials_b=n,
            decision_threshold=decision_threshold,
            cred_level=cred_level,
            n_samples=n_samples,
            seed=seed,
        )

    if counts is None:
        counts = paired_counts(df)
    if not paired_counts_errors(counts):
        out["accuracy"] = paired_accuracy_posterior(
            counts,
            decision_threshold=decision_threshold,
            cred_level=cred_level,
            n_samples=n_samples,
            seed=seed,
        )

    if "confidence_a" in df.columns and "confidence_b" in df.columns:
        a = df["confidence_a"].dropna().values.astype(float)
        b = df["confidence_b"].dropna().values.astype(float)
        out["confidence"] = normal_decision(
            a, b,
            decision_threshold=decision_threshold,
            cred_level=cred_level,
            n_samples=n_samples,
            seed=seed,
        )

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _credible_interval(samples: np.ndarray, level: float) -> Tuple[float, float]:
    """Equal-tailed credible interval at the given level."""
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))

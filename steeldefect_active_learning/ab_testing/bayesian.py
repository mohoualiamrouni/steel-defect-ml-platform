"""
Bayesian A/B testing for the inspection model-replacement decision.

Two posterior models are exposed:
    - Beta-Binomial for binary outcomes (e.g., per-class accuracy, agreement
      rate). Each arm's success rate has a Beta prior; the posterior is
      Beta(α + successes, β + failures). The decision metric is
      P(p_B > p_A) = Pr[Beta(α_B + s_B, β_B + f_B) > Beta(α_A + s_A, β_A + f_A)],
      computed via Monte-Carlo from the posterior.

    - Normal-Normal for continuous outcomes (e.g., confidence_delta). With a
      vague prior the posterior on the mean reduces to a Student-t centred at
      the sample mean — for n in the hundreds the t is essentially Gaussian,
      so we use a Gaussian posterior with σ̂² / n variance.

Both models report:
    - posterior probability P(B > A)
    - expected loss E[max(0, μ_A − μ_B)] (regret if you ship B and B is worse)
    - credible interval at the requested level
    - decision: ship B iff P(B > A) ≥ threshold (default 0.95)

References:
    Carpenter (2008) "Bayesian A/B testing".
    Stucchio (2015) "Bayesian A/B Testing at VWO" (Beta-Binomial expected loss).
    Kruschke (2018) "Rejecting or accepting parameter values in Bayesian
        estimation".
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


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

def bayesian_summary(
    df: pd.DataFrame,
    decision_threshold: float = 0.95,
    cred_level: float = 0.95,
    n_samples: int = 200_000,
    seed: Optional[int] = 0,
) -> Dict:
    """Run Beta-Binomial (agreement) and Normal (confidence) on the A/B df.

    Selects the model based on what columns are present:
      - If 'agrees' is present, runs Beta-Binomial on agreement rate.
      - If 'confidence_a' / 'confidence_b' are present, runs Normal-Normal on
        the confidence difference (interpreted as B − A).
      - If 'correct_a' / 'correct_b' (or 'pred_a','pred_b','label') are present,
        runs Beta-Binomial on accuracy.

    Returns a dict with up to three sub-dicts: 'agreement', 'accuracy',
    'confidence', each containing a full decision summary. Empty if none of
    the required columns are present.
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

    correct_a, correct_b = _resolve_correctness(df)
    if correct_a is not None:
        s_a, s_b = int(correct_a.sum()), int(correct_b.sum())
        n_a = n_b = int(len(df))
        out["accuracy"] = beta_binomial_decision(
            successes_a=s_a, trials_a=n_a,
            successes_b=s_b, trials_b=n_b,
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


def _resolve_correctness(df: pd.DataFrame):
    """Same logic as metrics._resolve_correctness — duplicated to avoid an
    import cycle when this module is loaded standalone in notebooks."""
    if "correct_a" in df.columns and "correct_b" in df.columns:
        return df["correct_a"].astype(bool).values, df["correct_b"].astype(bool).values
    if "pred_a" in df.columns and "pred_b" in df.columns and "label" in df.columns:
        ca = (df["pred_a"].values == df["label"].values).astype(bool)
        cb = (df["pred_b"].values == df["label"].values).astype(bool)
        return ca, cb
    return None, None

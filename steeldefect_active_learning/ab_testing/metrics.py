"""
A/B testing metrics, statistical significance, and visualisation.

Implements the 5-step decision cascade described in example-report.tex §3:
    1. Shapiro-Wilk normality check on paired confidence distributions
    2. Welch's t-test (parametric) or Mann-Whitney U (non-parametric fallback)
    3. McNemar's test on paired classifications (Yates / exact binomial)
    4. Cohen's d with Hedges' small-sample correction J(ν) = 1 − 3/(4ν−1)
    5. Family-wise / FDR correction across the 28 defect classes
       (Bonferroni | Holm | Benjamini-Hochberg)

Classic metrics:
    disagreement_rate         — % of images where Model A and Model B disagree
    confidence_delta_summary  — mean / std / p90 of |conf_a - conf_b|
    per_class_agreement       — agreement rate broken down by predicted class

Advanced metrics:
    confidence_by_band        — disagreement rate + entropy per confidence band
    entropy_comparison        — mean/std of prediction entropy per model
    margin_comparison         — mean margin score (top-1 minus top-2 softmax prob)

Statistical testing (Quin & Weyns 2023 — AutoPABS):
    statistical_significance_test — Welch's t-test + Cohen's d (+ Hedges) effect size
    mcnemar_test                  — paired-classifier McNemar (Yates / exact)
    per_class_mcnemar             — per-class McNemar suite with multiple-testing correction
    srm_check                     — sample-ratio-mismatch χ² gate (industry standard)
    correct_pvalues               — Bonferroni / Holm / Benjamini-Hochberg
    required_sample_size          — n needed for target power/effect
    power_analysis                — achieved power given current sample

Summary:
    compute_all_metrics       — single call returning all metrics as one dict

Visualisation:
    plot_comparison           — 4 standard diagnostic PNGs
    plot_advanced_comparison  — 4 additional advanced PNGs
        calibration_curve     — reliability diagram (A vs B)
        entropy_distribution  — histogram of prediction entropy per model
        confidence_bands      — disagreement rate per confidence band
        margin_vs_disagreement— margin score vs disagreement scatter

References:
    Quin & Weyns (2023) "Automating Pipelines of A/B Tests with Population
        Split Using Self-Adaptation and Machine Learning", IEEE.
    McNemar (1947) "Note on the sampling error of the difference between
        correlated proportions or percentages", Psychometrika 12(2).
    Hedges (1981) "Distribution theory for Glass's estimator of effect size".
    Benjamini & Hochberg (1995) "Controlling the false discovery rate", JRSS-B.
    Holm (1979) "A simple sequentially rejective multiple test procedure".
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classic scalar metrics
# ---------------------------------------------------------------------------

def disagreement_rate(df: pd.DataFrame) -> float:
    """Fraction of images where A and B predictions disagree."""
    return 1.0 - float(df["agrees"].mean())


def confidence_delta_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics of |confidence_a - confidence_b|."""
    delta = df["confidence_delta"]
    return {
        "mean":   float(delta.mean()),
        "median": float(delta.median()),
        "std":    float(delta.std()),
        "max":    float(delta.max()),
        "p90":    float(delta.quantile(0.9)),
        "p75":    float(delta.quantile(0.75)),
    }


def per_class_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement rate and count per predicted class (Model A)."""
    if "pred_a" not in df.columns:
        raise ValueError("per_class_agreement requires 'pred_a' column.")
    return (
        df.groupby("pred_a")["agrees"]
        .agg(agreement_rate="mean", count="count")
        .reset_index()
        .rename(columns={"pred_a": "class_id"})
        .sort_values("class_id")
    )


def per_class_iou_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """IoU statistics per image (segmentation only)."""
    if "iou" not in df.columns:
        raise ValueError("per_class_iou_comparison requires 'iou' column.")
    return df[["image_path", "iou", "agrees"]].describe()


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------

def entropy_comparison(df: pd.DataFrame) -> Dict[str, float]:
    """Compare prediction entropy between models.

    Requires 'entropy_a' and 'entropy_b' columns (added by ab_pipeline.py
    when run in classification mode).

    Returns:
        Dict with mean/std entropy per model and the entropy delta.
    """
    out: Dict[str, float] = {}
    for m in ("a", "b"):
        col = f"entropy_{m}"
        if col in df.columns:
            out[f"mean_entropy_{m}"]   = float(df[col].mean())
            out[f"std_entropy_{m}"]    = float(df[col].std())
            out[f"median_entropy_{m}"] = float(df[col].median())
    if "entropy_a" in df.columns and "entropy_b" in df.columns:
        delta = df["entropy_a"] - df["entropy_b"]
        out["mean_entropy_delta"] = float(delta.mean())
        out["entropy_winner"] = "a" if out.get("mean_entropy_a", 0) < out.get("mean_entropy_b", 0) else "b"
    return out


def margin_comparison(df: pd.DataFrame) -> Dict[str, float]:
    """Compare margin scores between models.

    Margin = softmax_top1 - softmax_top2.  Higher margin = more confident.
    Requires 'margin_a' and 'margin_b' columns.
    """
    out: Dict[str, float] = {}
    for m in ("a", "b"):
        col = f"margin_{m}"
        if col in df.columns:
            out[f"mean_margin_{m}"]   = float(df[col].mean())
            out[f"std_margin_{m}"]    = float(df[col].std())
    if "margin_a" in df.columns and "margin_b" in df.columns:
        out["margin_winner"] = "a" if out.get("mean_margin_a", 0) > out.get("mean_margin_b", 0) else "b"
    return out


def disagreement_by_entropy_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Disagreement rate within entropy quartiles of Model A.

    High disagreement in low-entropy (confident) images signals a
    systematic behavioural difference — not just noise.

    Returns:
        DataFrame with columns: quartile, entropy_threshold, n, disagreement_rate.
    """
    if "entropy_a" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["quartile"] = pd.qcut(df2["entropy_a"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    return (
        df2.groupby("quartile", observed=True)["agrees"]
        .agg(n="count", disagreement_rate=lambda x: 1 - x.mean())
        .reset_index()
    )


# ---------------------------------------------------------------------------
# CUPED variance reduction (Deng et al. 2013)
# ---------------------------------------------------------------------------

def apply_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
) -> pd.DataFrame:
    """Apply CUPED (Controlled-experiment Using Pre-Experiment Data) to a metric.

    Replaces the metric Y with Y_cuped = Y − θ · (X − E[X]) where
    θ = Cov(Y, X) / Var(X). Under the assumption that X is **pre-treatment**
    (uncorrelated with the assignment), Y_cuped is unbiased for E[Y] and has
    variance Var(Y) · (1 − ρ²) — typically ≥ 30% reduction at ρ ≈ 0.5, which
    shrinks MDE by ~15% at the same sample size.

    Critical assumption: covariate_col MUST be pre-treatment. In the plant's
    context that means (a) the previous-batch confidence on the same image,
    (b) a held-out reference confidence from the incumbent model, or (c) any
    feature computed before the new model was trained. Using a post-treatment
    covariate biases the estimate.

    Args:
        df:             A/B comparison DataFrame.
        metric_col:     Name of the response column (e.g., 'confidence_delta').
        covariate_col:  Name of the pre-treatment covariate column.

    Returns:
        New DataFrame with an added column ``{metric_col}_cuped``. The original
        column is preserved so vanilla and CUPED-adjusted CIs can be compared.

    References:
        Deng, Xu, Kohavi, Walker (2013) "Improving the Sensitivity of Online
        Controlled Experiments by Utilizing Pre-Experiment Data", WSDM.
    """
    if metric_col not in df.columns:
        raise ValueError(f"metric column {metric_col!r} not found in DataFrame")
    if covariate_col not in df.columns:
        raise ValueError(f"covariate column {covariate_col!r} not found in DataFrame")

    y = df[metric_col].astype(float).values
    x = df[covariate_col].astype(float).values
    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-12:
        # Degenerate covariate (constant) — CUPED is a no-op.
        theta = 0.0
    else:
        theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)

    out = df.copy()
    out[f"{metric_col}_cuped"] = y - theta * (x - x.mean())
    out.attrs[f"{metric_col}_cuped_theta"] = theta
    rho = 0.0
    if var_x > 1e-12 and np.var(y, ddof=1) > 1e-12:
        rho = float(np.corrcoef(y, x)[0, 1])
    out.attrs[f"{metric_col}_cuped_rho"] = rho
    out.attrs[f"{metric_col}_cuped_variance_ratio"] = float(1.0 - rho ** 2)
    return out


# ---------------------------------------------------------------------------
# Bootstrap CIs (percentile + BCa) for Cohen's d and metric mean
# ---------------------------------------------------------------------------

def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic: str = "cohens_d",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: Optional[int] = 0,
) -> Dict:
    """Bootstrap CI for two-sample Cohen's d or mean difference.

    Args:
        a, b:        1-D float arrays (Model A and Model B observations).
        statistic:   "cohens_d" or "mean_diff".
        n_resamples: Number of bootstrap resamples (default 10_000).
        alpha:       Significance level → CI level = 1 − alpha.
        method:      "percentile" or "bca" (bias-corrected and accelerated).
        seed:        RNG seed for reproducibility.

    Returns:
        Dict with: estimate, ci_low, ci_high, alpha, method, n_resamples.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {
            "estimate": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "alpha": alpha, "method": method,
            "n_resamples": 0,
        }

    def _stat(xa: np.ndarray, xb: np.ndarray) -> float:
        if statistic == "cohens_d":
            sp = math.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2.0 + 1e-12)
            return float((np.mean(xa) - np.mean(xb)) / sp)
        if statistic == "mean_diff":
            return float(np.mean(xa) - np.mean(xb))
        raise ValueError(f"unknown statistic {statistic!r}")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sa = a[rng.integers(0, n_a, size=n_a)]
        sb = b[rng.integers(0, n_b, size=n_b)]
        boot[i] = _stat(sa, sb)

    estimate = _stat(a, b)
    if method == "percentile":
        ci_low  = float(np.quantile(boot, alpha / 2))
        ci_high = float(np.quantile(boot, 1 - alpha / 2))
    elif method == "bca":
        from scipy.stats import norm
        # Bias-correction
        z0 = norm.ppf(np.mean(boot < estimate) + 1e-12)
        # Acceleration via jackknife on the pooled sample
        pooled = np.concatenate([a, b])
        n = len(pooled)
        jack = np.empty(n, dtype=float)
        for i in range(n):
            mask = np.arange(n) != i
            ja = pooled[mask][: n_a - (i < n_a)]
            jb = pooled[mask][n_a - (i < n_a):]
            if len(ja) >= 2 and len(jb) >= 2:
                jack[i] = _stat(ja, jb)
            else:
                jack[i] = estimate
        jbar = jack.mean()
        num = np.sum((jbar - jack) ** 3)
        den = 6.0 * (np.sum((jbar - jack) ** 2)) ** 1.5 + 1e-12
        accel = float(num / den)
        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)
        a_lo = norm.cdf(z0 + (z0 + z_lo) / max(1e-12, 1 - accel * (z0 + z_lo)))
        a_hi = norm.cdf(z0 + (z0 + z_hi) / max(1e-12, 1 - accel * (z0 + z_hi)))
        ci_low  = float(np.quantile(boot, a_lo))
        ci_high = float(np.quantile(boot, a_hi))
    else:
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")

    return {
        "estimate":    float(estimate),
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "alpha":       float(alpha),
        "method":      method,
        "n_resamples": int(n_resamples),
    }


# ---------------------------------------------------------------------------
# Multiple-testing correction (Bonferroni | Holm | Benjamini-Hochberg)
# ---------------------------------------------------------------------------

_VALID_CORRECTIONS = {"bonferroni", "holm", "benjamini-hochberg", "bh", "fdr_bh", "none", None}


def correct_pvalues(
    pvals: Sequence[float],
    method: Optional[str] = "benjamini-hochberg",
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Apply a family-wise / FDR correction to a list of p-values.

    Args:
        pvals:  Sequence of raw p-values (per-class or per-subgroup tests).
        method: One of {"bonferroni", "holm", "benjamini-hochberg", "none"}.
                Aliases "bh" and "fdr_bh" map to Benjamini-Hochberg.
        alpha:  Significance level for the corrected reject decision.

    Returns:
        Dict with three np.ndarray of shape (k,):
            "p_corrected" — adjusted p-values.
            "reject"      — boolean mask of corrected rejections at level alpha.
            "method"      — canonical name of the method used.

    Notes:
        Uses statsmodels.stats.multitest.multipletests when available (preferred),
        and falls back to a hand-rolled implementation if statsmodels is missing.
        Hand-rolled BH and Holm are exact; hand-rolled Bonferroni is trivial.
    """
    if method not in _VALID_CORRECTIONS:
        raise ValueError(
            f"correction method must be one of {_VALID_CORRECTIONS}, got {method!r}"
        )

    raw = np.asarray(list(pvals), dtype=float)
    k = len(raw)
    if k == 0:
        return {"p_corrected": raw, "reject": np.zeros(0, dtype=bool), "method": str(method)}

    canonical = {"bh": "benjamini-hochberg", "fdr_bh": "benjamini-hochberg"}.get(method, method)
    if canonical is None:
        canonical = "none"

    if canonical == "none":
        return {"p_corrected": raw.copy(), "reject": raw < alpha, "method": "none"}

    try:
        from statsmodels.stats.multitest import multipletests
        sm_method = {
            "bonferroni":         "bonferroni",
            "holm":               "holm",
            "benjamini-hochberg": "fdr_bh",
        }[canonical]
        reject, p_corr, _, _ = multipletests(raw, alpha=alpha, method=sm_method)
        return {"p_corrected": p_corr, "reject": reject, "method": canonical}
    except ImportError:
        logger.debug("statsmodels not available; using hand-rolled %s correction.", canonical)

    # Fallback implementations
    if canonical == "bonferroni":
        p_corr = np.minimum(raw * k, 1.0)
        reject = p_corr < alpha
    elif canonical == "holm":
        order = np.argsort(raw)
        ranked = raw[order]
        p_sorted_adj = np.empty(k)
        prev = 0.0
        for i in range(k):
            prev = max(prev, (k - i) * ranked[i])
            p_sorted_adj[i] = min(prev, 1.0)
        p_corr = np.empty(k)
        p_corr[order] = p_sorted_adj
        reject = p_corr < alpha
    else:  # benjamini-hochberg
        order = np.argsort(raw)
        ranked = raw[order]
        ranks = np.arange(1, k + 1)
        bh = ranked * k / ranks
        # enforce monotonicity from the right
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.minimum(bh, 1.0)
        p_corr = np.empty(k)
        p_corr[order] = bh
        reject = p_corr < alpha

    return {"p_corrected": p_corr, "reject": reject, "method": canonical}


# ---------------------------------------------------------------------------
# Sample-ratio-mismatch (SRM) gate
# ---------------------------------------------------------------------------

def srm_check(
    n_a: int,
    n_b: int,
    expected_ratio: float = 0.5,
    threshold: float = 1e-3,
) -> Dict:
    """χ² goodness-of-fit test for sample ratio mismatch.

    SRM is the single most common silent A/B failure mode: a bug in the
    randomiser produces unequal arm sizes, which invalidates downstream
    inference even when the test statistic looks fine. Industry default is
    to refuse to interpret an experiment when SRM p < 1e-3.

    Args:
        n_a, n_b:        Observed sample sizes for arm A and arm B.
        expected_ratio:  Expected fraction in arm A (default 0.5 = 50/50).
        threshold:       p-value cutoff below which SRM is flagged.

    Returns:
        Dict with: n_a, n_b, expected_ratio, observed_ratio_a, chi2, p_value,
        srm_detected (True ⇒ refuse to publish results).
    """
    from scipy import stats
    n = n_a + n_b
    if n == 0:
        return {
            "n_a": 0, "n_b": 0, "expected_ratio": expected_ratio,
            "observed_ratio_a": float("nan"), "chi2": float("nan"),
            "p_value": 1.0, "srm_detected": False,
        }
    observed = np.array([n_a, n_b], dtype=float)
    expected = np.array([n * expected_ratio, n * (1 - expected_ratio)], dtype=float)
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    return {
        "n_a":              int(n_a),
        "n_b":              int(n_b),
        "expected_ratio":   float(expected_ratio),
        "observed_ratio_a": float(n_a / n),
        "chi2":             float(chi2),
        "p_value":          float(p_value),
        "srm_detected":     bool(p_value < threshold),
    }


# ---------------------------------------------------------------------------
# Hedges' small-sample correction for Cohen's d
# ---------------------------------------------------------------------------

def hedges_correction(d: float, n_a: int, n_b: int) -> float:
    """Multiply Cohen's d by Hedges' J(ν) = 1 − 3 / (4ν − 1).

    Removes positive bias in d for small samples (ν = n_a + n_b − 2).
    For n ≳ 50 the correction is negligible; for n ≲ 20 it matters.
    """
    nu = n_a + n_b - 2
    if nu < 2:
        return float("nan")
    j = 1.0 - 3.0 / (4.0 * nu - 1.0)
    return float(d * j)


# ---------------------------------------------------------------------------
# McNemar's test for paired classifiers (cascade step 3)
# ---------------------------------------------------------------------------

def mcnemar_test(df: pd.DataFrame, exact_threshold: int = 25) -> Dict:
    """McNemar's test on paired predictions of two classifiers.

    Requires either:
      - 'correct_a' and 'correct_b' columns (per-row 0/1), preferred; OR
      - 'pred_a', 'pred_b', and 'label' columns (we derive correct_x = (pred_x == label)).

    Yates' continuity correction is applied when discordant pairs (b + c) ≥ exact_threshold;
    otherwise an exact binomial test on the discordant pairs is used (statsmodels-style).

    Returns:
        Dict with: b (a_correct ∧ b_wrong), c (a_wrong ∧ b_correct),
        statistic, p_value, test_used ('mcnemar_yates' | 'mcnemar_exact'),
        n_discordant, n_concordant, applicable.
    """
    correct_a, correct_b = _resolve_correctness(df)
    if correct_a is None:
        return {"applicable": False, "reason": "no ground truth or correctness columns"}

    # b: A correct, B wrong; c: A wrong, B correct  (discordant pairs)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c
    n_concordant = int(len(correct_a)) - n_discordant

    if n_discordant == 0:
        return {
            "applicable": True, "b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
            "test_used": "mcnemar_degenerate",
            "n_discordant": 0, "n_concordant": n_concordant,
        }

    if n_discordant >= exact_threshold:
        from scipy.stats import chi2 as _chi2
        # Yates-corrected χ² with df=1
        statistic = (abs(b - c) - 1.0) ** 2 / float(n_discordant)
        p_value = float(1.0 - _chi2.cdf(statistic, df=1))
        test_used = "mcnemar_yates"
    else:
        # Exact binomial on discordant pairs (k=min(b,c), n=b+c, p=0.5, two-sided)
        from scipy.stats import binom
        k = min(b, c)
        # two-sided exact: 2 * P(X ≤ k) clipped at 1
        p_value = float(min(1.0, 2.0 * binom.cdf(k, n_discordant, 0.5)))
        statistic = float(k)
        test_used = "mcnemar_exact"

    return {
        "applicable":   True,
        "b":            b,
        "c":            c,
        "statistic":    float(statistic),
        "p_value":      float(p_value),
        "test_used":    test_used,
        "n_discordant": n_discordant,
        "n_concordant": n_concordant,
    }


def per_class_mcnemar(
    df: pd.DataFrame,
    correction: Optional[str] = "benjamini-hochberg",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run McNemar per defect class and apply multiple-testing correction.

    Returns a DataFrame with one row per class:
        class_id, n, b, c, statistic, p_value, p_corrected, reject, test_used.

    Empty if ground truth ('label' column or correct_a/correct_b) is unavailable —
    we cannot do per-class McNemar without knowing which model is correct.
    """
    correct_a, correct_b = _resolve_correctness(df)
    if correct_a is None or "label" not in df.columns:
        return pd.DataFrame(columns=[
            "class_id", "n", "b", "c", "statistic", "p_value",
            "p_corrected", "reject", "test_used",
        ])

    rows: List[Dict] = []
    for cls, sub in df.groupby("label"):
        ca = correct_a[sub.index]
        cb = correct_b[sub.index]
        if len(ca) == 0:
            continue
        sub_df = pd.DataFrame({"correct_a": ca.astype(int), "correct_b": cb.astype(int)})
        res = mcnemar_test(sub_df)
        if not res.get("applicable", False):
            continue
        rows.append({
            "class_id":  int(cls) if not isinstance(cls, str) else cls,
            "n":         int(len(sub)),
            "b":         int(res["b"]),
            "c":         int(res["c"]),
            "statistic": float(res["statistic"]),
            "p_value":   float(res["p_value"]),
            "test_used": res["test_used"],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "class_id", "n", "b", "c", "statistic", "p_value",
            "p_corrected", "reject", "test_used",
        ])

    out = pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)
    corr = correct_pvalues(out["p_value"].values, method=correction, alpha=alpha)
    out["p_corrected"] = corr["p_corrected"]
    out["reject"]      = corr["reject"]
    out.attrs["correction_method"] = corr["method"]
    return out


def _resolve_correctness(df: pd.DataFrame):
    """Resolve per-row correctness for both arms.

    Prefers explicit 'correct_a'/'correct_b' columns; otherwise derives them from
    ('pred_a','pred_b','label'). Returns (correct_a, correct_b) as boolean arrays
    indexed identically to df, or (None, None) if not derivable.
    """
    if "correct_a" in df.columns and "correct_b" in df.columns:
        return df["correct_a"].astype(bool).values, df["correct_b"].astype(bool).values
    if "pred_a" in df.columns and "pred_b" in df.columns and "label" in df.columns:
        ca = (df["pred_a"].values == df["label"].values).astype(bool)
        cb = (df["pred_b"].values == df["label"].values).astype(bool)
        return ca, cb
    return None, None


# ---------------------------------------------------------------------------
# Statistical significance testing (Quin & Weyns 2023)
# ---------------------------------------------------------------------------

def statistical_significance_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict:
    """Welch's t-test comparing Model A vs Model B confidence distributions.

    Implements the hypothesis test framework from Quin & Weyns (2023):
    - H0: μ_A = μ_B  (no difference between models)
    - H1: μ_A ≠ μ_B  (two-tailed)

    Uses Welch's t-test (unequal variance) which is more robust than
    Student's t-test when sample variances differ between groups.

    Effect size is Cohen's d:
        d = (μ_A − μ_B) / σ_pooled
        where σ_pooled = √((σ²_A + σ²_B) / 2)

    Cohen's d interpretation:
        |d| < 0.2  → negligible
        0.2 ≤ |d| < 0.5 → small
        0.5 ≤ |d| < 0.8 → medium
        |d| ≥ 0.8  → large

    Args:
        df:    A/B comparison DataFrame with 'confidence_a', 'confidence_b'.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with keys:
            t_stat, p_value, significant, cohens_d, effect_magnitude,
            mean_a, mean_b, std_a, std_b, n,
            confidence_interval_95 (tuple: (lower, upper) of μ_A - μ_B)
    """
    from scipy import stats

    a = df["confidence_a"].dropna().values.astype(float)
    b = df["confidence_b"].dropna().values.astype(float)
    n_a, n_b = len(a), len(b)

    # Normality check (Shapiro-Wilk, capped at 5000 samples)
    n_sw = min(n_a, n_b, 5000)
    _, p_norm_a = stats.shapiro(a[:n_sw])
    _, p_norm_b = stats.shapiro(b[:n_sw])
    use_parametric = (p_norm_a >= 0.05 and p_norm_b >= 0.05)

    if use_parametric:
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        test_used = "welch_t"
    else:
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        t_stat = float(u_stat)  # store U-stat for backward compat
        test_used = "mann_whitney_u"

    # Cohen's d (always computed — meaningful regardless of test)
    pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    # Degenerate-variance guard: with near-constant confidences the denominator
    # collapses and Welch t / Cohen's d explode to absurd values (the ~1e6
    # "t-stat" that betrays synthetic, over-confident data). Flag it so callers
    # can report Mann-Whitney + effect size instead of a meaningless t.
    degenerate_variance = bool(pooled_std < 1e-3)
    cohens_d = (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)

    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    # 95% confidence interval for (μ_A - μ_B) using Welch-Satterthwaite df
    diff = np.mean(a) - np.mean(b)
    if use_parametric:
        se = math.sqrt(np.var(a, ddof=1) / n_a + np.var(b, ddof=1) / n_b)
        df_welch = (
            (np.var(a, ddof=1) / n_a + np.var(b, ddof=1) / n_b) ** 2
            / (
                (np.var(a, ddof=1) / n_a) ** 2 / (n_a - 1)
                + (np.var(b, ddof=1) / n_b) ** 2 / (n_b - 1)
            )
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
        ci = (diff - t_crit * se, diff + t_crit * se)
    else:
        ci = (float("nan"), float("nan"))  # CI not applicable for Mann-Whitney

    return {
        "t_stat":                 float(t_stat),
        "p_value":                float(p_value),
        "significant":            bool(p_value < alpha),
        "cohens_d":               float(cohens_d),
        "cohens_d_hedges":        hedges_correction(float(cohens_d), n_a, n_b),
        "effect_magnitude":       magnitude,
        "test_used":              test_used,
        "degenerate_variance":    degenerate_variance,
        "shapiro_p_a":            float(p_norm_a),
        "shapiro_p_b":            float(p_norm_b),
        "mean_a":                 float(np.mean(a)),
        "mean_b":                 float(np.mean(b)),
        "std_a":                  float(np.std(a, ddof=1)),
        "std_b":                  float(np.std(b, ddof=1)),
        "n":                      int(min(n_a, n_b)),
        "confidence_interval_95": (float(ci[0]), float(ci[1])),
    }


def required_sample_size(
    effect_size: float,
    sigma: float = 0.15,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum sample size to detect a given effect at target power.

    Formula (two-sample, two-tailed):
        n = 2 * (z_{α/2} + z_{β})² * σ² / Δ²

    Args:
        effect_size: Desired detectable mean difference (e.g. 0.05).
        sigma:       Expected standard deviation per group (default 0.15).
        alpha:       Significance level (default 0.05).
        power:       Desired statistical power (default 0.80).

    Returns:
        Required sample size per group (int).
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta  = norm.ppf(power)
    if effect_size <= 0:
        return int(1e6)
    n = 2 * ((z_alpha + z_beta) ** 2) * (sigma ** 2) / (effect_size ** 2)
    return max(2, math.ceil(n))


def power_analysis(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """Compute achieved statistical power given the current A/B data.

    Args:
        df:    A/B comparison DataFrame.
        alpha: Significance level.

    Returns:
        Dict with: achieved_power, observed_effect, n, required_n_for_80pct_power,
                   sigma, adequately_powered.
    """
    from scipy.stats import norm

    a = df["confidence_a"].dropna().values.astype(float)
    b = df["confidence_b"].dropna().values.astype(float)

    n = min(len(a), len(b))
    observed_effect = abs(np.mean(a) - np.mean(b))
    sigma = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)

    z_alpha = norm.ppf(1 - alpha / 2)
    se  = sigma * math.sqrt(2.0 / n) if n > 0 else 1.0
    ncp = observed_effect / (se + 1e-12)
    achieved_power = float(norm.cdf(ncp - z_alpha) + norm.cdf(-ncp - z_alpha))
    achieved_power = max(0.0, min(1.0, achieved_power))

    req_n = required_sample_size(
        effect_size=observed_effect, sigma=sigma, alpha=alpha, power=0.80
    ) if observed_effect > 1e-6 else int(1e6)

    return {
        "achieved_power":       achieved_power,
        "observed_effect":      float(observed_effect),
        "sigma":                float(sigma),
        "n":                    n,
        "required_n_for_80pct": req_n,
        "adequately_powered":   achieved_power >= 0.80,
    }


# ---------------------------------------------------------------------------
# Unified summary
# ---------------------------------------------------------------------------

def compute_all_metrics(
    df: pd.DataFrame,
    task: str = "classification",
    alpha: float = 0.05,
    correction: Optional[str] = "benjamini-hochberg",
    srm_threshold: float = 1e-3,
    expected_ratio: float = 0.5,
    cuped_covariate: Optional[str] = None,
    bootstrap_n: int = 0,
    bayesian: bool = False,
    bayesian_decision_threshold: float = 0.95,
    sequential: Optional[str] = None,
) -> Dict:
    """Compute the full A/B cascade in a single call.

    Args:
        df:              A/B comparison DataFrame.
        task:            "classification" or "segmentation".
        alpha:           Significance level (default 0.05).
        correction:      Multiple-testing correction across the per-class McNemar
                         suite. One of {"bonferroni", "holm", "benjamini-hochberg",
                         "none"}. Reads from cfg.ab_testing.multiple_testing_correction.
        srm_threshold:   p-value below which SRM gate flips. Default 1e-3 (industry
                         standard).
        expected_ratio:  Expected fraction of arm A in an SRM check. Default 0.5.

    Returns:
        Nested dict containing all five cascade steps:
            "srm":          sample-ratio-mismatch χ² gate
            "disagreement": disagreement rate + n
            "delta":        confidence-delta summary
            "statistical":  Welch / Mann-Whitney + Cohen's d (+ Hedges)
            "mcnemar":      paired-classifier McNemar (if labels available)
            "per_class_mcnemar": per-class McNemar w/ multiple-testing correction
            "power":        achieved power + required n at 80%
            "entropy":      entropy comparison (if entropy columns present)
            "margin":       margin comparison (if margin columns present)
            "segmentation": IoU / Dice metrics (if task='segmentation')
            "per_class":    per-class agreement breakdown
    """
    out: Dict = {}

    n_total = len(df)
    n_arm_a = int(df["arm"].eq("a").sum()) if "arm" in df.columns else n_total // 2
    n_arm_b = n_total - n_arm_a if "arm" in df.columns else n_total - n_arm_a
    out["srm"] = srm_check(n_arm_a, n_arm_b, expected_ratio=expected_ratio,
                           threshold=srm_threshold)

    out["disagreement"] = {"rate": disagreement_rate(df), "n": n_total}
    out["delta"]        = confidence_delta_summary(df)
    out["statistical"]  = statistical_significance_test(df, alpha=alpha)
    out["power"]        = power_analysis(df, alpha=alpha)

    # CUPED variance reduction on confidence_delta when a pre-treatment covariate
    # is supplied. Reports both vanilla and adjusted CIs so the gain is visible.
    if cuped_covariate and cuped_covariate in df.columns and "confidence_delta" in df.columns:
        try:
            df_cuped = apply_cuped(df, "confidence_delta", cuped_covariate)
            theta = df_cuped.attrs.get("confidence_delta_cuped_theta", 0.0)
            rho   = df_cuped.attrs.get("confidence_delta_cuped_rho", 0.0)
            v_ratio = df_cuped.attrs.get("confidence_delta_cuped_variance_ratio", 1.0)
            cuped_vals = df_cuped["confidence_delta_cuped"].values
            out["cuped"] = {
                "covariate":       cuped_covariate,
                "theta":           float(theta),
                "rho":             float(rho),
                "variance_ratio":  float(v_ratio),
                "mean":            float(np.mean(cuped_vals)),
                "std":             float(np.std(cuped_vals, ddof=1)),
                "var_reduction":   float(1.0 - v_ratio),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUPED skipped: %s", exc)

    # Bootstrap CI for Cohen's d on the confidence distributions, if requested.
    if bootstrap_n > 0 and "confidence_a" in df.columns and "confidence_b" in df.columns:
        a = df["confidence_a"].dropna().values.astype(float)
        b = df["confidence_b"].dropna().values.astype(float)
        out["bootstrap"] = {
            "cohens_d":  bootstrap_ci(a, b, "cohens_d",  n_resamples=bootstrap_n, alpha=alpha),
            "mean_diff": bootstrap_ci(a, b, "mean_diff", n_resamples=bootstrap_n, alpha=alpha),
        }

    # Bayesian decision (Beta-Binomial on agreement/accuracy + Normal on confidence).
    if bayesian:
        try:
            from .bayesian import bayesian_summary
            out["bayesian"] = bayesian_summary(
                df,
                decision_threshold=bayesian_decision_threshold,
                cred_level=1.0 - alpha,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bayesian summary skipped: %s", exc)

    # Sequential / always-valid testing (mSPRT or O'Brien-Fleming).
    if sequential and "confidence_a" in df.columns and "confidence_b" in df.columns:
        try:
            from .sequential import (
                MSPRTState,
                confidence_sequence,
                obrien_fleming_decide,
            )
            a = df["confidence_a"].dropna().values.astype(float)
            b = df["confidence_b"].dropna().values.astype(float)
            if sequential == "msprt":
                state = MSPRTState()
                state.update_batch(a, b)
                cs_lo, cs_hi = confidence_sequence(state, alpha=alpha)
                out["sequential"] = {
                    "method":     "msprt",
                    "p_av":       state.always_valid_p(),
                    "delta":      state.mean_diff(),
                    "cs_low":     cs_lo,
                    "cs_high":    cs_hi,
                    "decision":   "stop_reject_h0" if state.always_valid_p() < alpha else "continue",
                }
            elif sequential == "obrien_fleming":
                # Treat the current run as the final (and only) look — a sane
                # default; users with multi-look plans can call the helper directly.
                z = out["statistical"]["t_stat"]
                out["sequential"] = obrien_fleming_decide(
                    z_observed=z, look=1, n_looks=1, alpha=alpha,
                )
                out["sequential"]["method"] = "obrien_fleming"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sequential summary skipped: %s", exc)

    mc = mcnemar_test(df)
    if mc.get("applicable", False):
        out["mcnemar"] = mc
        per_cls_mc = per_class_mcnemar(df, correction=correction, alpha=alpha)
        if not per_cls_mc.empty:
            out["per_class_mcnemar"] = {
                "method": per_cls_mc.attrs.get("correction_method", correction),
                "alpha":  alpha,
                "rows":   per_cls_mc.to_dict(orient="records"),
            }

    if "entropy_a" in df.columns:
        out["entropy"] = entropy_comparison(df)
    if "margin_a" in df.columns:
        out["margin"]  = margin_comparison(df)

    if task == "segmentation":
        seg: Dict = {}
        if "iou" in df.columns:
            seg["mean_iou"]    = float(df["iou"].mean())
            seg["median_iou"]  = float(df["iou"].median())
            seg["pct_high_iou"]= float((df["iou"] >= 0.75).mean())
        if "dice" in df.columns:
            seg["mean_dice"]   = float(df["dice"].mean())
        if "boundary_iou" in df.columns:
            seg["mean_boundary_iou"] = float(df["boundary_iou"].mean())
        out["segmentation"] = seg

    if "pred_a" in df.columns:
        out["per_class"] = per_class_agreement(df).to_dict(orient="records")

    return out


# ---------------------------------------------------------------------------
# Visualisation — standard plots
# ---------------------------------------------------------------------------

def plot_comparison(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    task: str = "classification",
    prefix: str = "ab",
) -> None:
    """Generate and save the standard 4 A/B diagnostic PNGs.

    Saved files:
        {prefix}_disagreement_rate.png
        {prefix}_confidence_delta_dist.png
        {prefix}_per_class_agreement.png   (classification only)
        {prefix}_iou_distribution.png      (segmentation only)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _plot_disagreement_rate(df, output_dir, prefix)
    _plot_confidence_delta(df, output_dir, prefix)

    if task == "classification" and "pred_a" in df.columns:
        _plot_per_class_agreement(df, output_dir, prefix)
    if task == "segmentation" and "iou" in df.columns:
        _plot_iou_distribution(df, output_dir, prefix)

    plt.close("all")


def plot_advanced_comparison(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    prefix: str = "ab",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> None:
    """Generate 4 advanced A/B diagnostic PNGs.

    Saved files:
        {prefix}_calibration_curve.png            — reliability diagram
        {prefix}_entropy_distribution.png         — prediction entropy A vs B + KL divergence
        {prefix}_confidence_bands.png             — disagreement rate by conf band
        {prefix}_per_class_accuracy_delta.png     — acc(B) − acc(A) per class
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _plot_calibration_curve(df, output_dir, prefix, model_a_name, model_b_name)

    if "entropy_a" in df.columns:
        _plot_entropy_distribution(df, output_dir, prefix, model_a_name, model_b_name)
        _plot_confidence_bands(df, output_dir, prefix)

    if "pred_a" in df.columns:
        _plot_per_class_accuracy_delta(df, output_dir, prefix, model_a_name, model_b_name)

    plt.close("all")


# ---------------------------------------------------------------------------
# Standard plot helpers
# ---------------------------------------------------------------------------

def _plot_disagreement_rate(df, output_dir, prefix):
    stat = statistical_significance_test(df)
    agree    = int(df["agrees"].sum())
    disagree = len(df) - agree

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Agree", "Disagree"], [agree, disagree],
                  color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, [agree, disagree]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{count}\n({count/len(df):.1%})", ha="center", va="bottom", fontsize=10)

    p_str  = f"p={stat['p_value']:.4f}" if stat["p_value"] >= 0.0001 else "p<0.0001"
    d_str  = f"d={stat['cohens_d']:.3f} ({stat['effect_magnitude']})"
    sig_str = "significant" if stat["significant"] else "not significant"
    ax.set_title(f"A/B Agreement  |  disagreement={disagree/len(df):.1%}\n"
                 f"{p_str}  {d_str}  [{sig_str}]", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(agree, disagree) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_disagreement_rate.png"), dpi=150)
    plt.close(fig)


def _plot_confidence_delta(df, output_dir, prefix):
    stat = statistical_significance_test(df)
    pwr  = power_analysis(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["confidence_delta"], bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(df["confidence_delta"].mean(), color="red", linestyle="--",
               label=f"mean={df['confidence_delta'].mean():.3f}")
    ax.axvline(df["confidence_delta"].quantile(0.9), color="#FF9800", linestyle=":",
               label=f"p90={df['confidence_delta'].quantile(0.9):.3f}")

    p_str = f"p={stat['p_value']:.4f}" if stat["p_value"] >= 0.0001 else "p<0.0001"
    box = (f"Welch t={stat['t_stat']:.2f}, {p_str}\n"
           f"Cohen's d={stat['cohens_d']:.3f} ({stat['effect_magnitude']})\n"
           f"Power={pwr['achieved_power']:.1%}  n={pwr['n']}")
    ax.text(0.98, 0.95, box, transform=ax.transAxes, va="top", ha="right",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.4", alpha=0.1, edgecolor="#607D8B"))
    ax.set_xlabel("|confidence_a − confidence_b|")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Delta Distribution — A/B Testing (Quin & Weyns 2023)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_confidence_delta_dist.png"), dpi=150)
    plt.close(fig)


_STEELDEFECT_CLASS_NAMES = [
    "scratch", "pit", "roll_mark", "edge_crack", "inclusion",
    "coil_break", "scale", "streak", "sliver", "orange_peel", "clean",
]

def _class_label(class_id: int) -> str:
    try:
        return _STEELDEFECT_CLASS_NAMES[int(class_id)]
    except (IndexError, ValueError, TypeError):
        return str(class_id)


def _plot_per_class_agreement(df, output_dir, prefix):
    stats = per_class_agreement(df)
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 1.1), 5))
    colors = ["#4CAF50" if r >= 0.8 else "#FF9800" if r >= 0.5 else "#F44336"
              for r in stats["agreement_rate"]]
    x_labels = [_class_label(c) for c in stats["class_id"]]
    ax.bar(x_labels, stats["agreement_rate"],
           color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.7, label="80% threshold")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Defect class (predicted by Model A)")
    ax.set_ylabel("Agreement rate")
    ax.set_title("Per-class Agreement Rate — A vs B")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    for i, (r, n) in enumerate(zip(stats["agreement_rate"], stats["count"])):
        ax.text(i, r + 0.02, f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_per_class_agreement.png"), dpi=150)
    plt.close(fig)


def _plot_iou_distribution(df, output_dir, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["iou"], bins=40, color="#9C27B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["iou"].mean(), color="red", linestyle="--",
                    label=f"mean IoU={df['iou'].mean():.3f}")
    if "dice" in df.columns:
        axes[0].axvline(df["dice"].mean(), color="#FF9800", linestyle=":",
                        label=f"mean Dice={df['dice'].mean():.3f}")
    axes[0].set_xlabel("Score (A vs B)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mask IoU / Dice Distribution")
    axes[0].legend(fontsize=9)

    if "mask_defect_fraction_a" in df.columns and "mask_defect_fraction_b" in df.columns:
        axes[1].scatter(df["mask_defect_fraction_a"], df["mask_defect_fraction_b"],
                        alpha=0.4, s=10, color="#FF5722")
        lim = max(df["mask_defect_fraction_a"].max(), df["mask_defect_fraction_b"].max()) * 1.05
        axes[1].plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect agreement")
        axes[1].set_xlabel("Defect fraction — Model A")
        axes[1].set_ylabel("Defect fraction — Model B")
        axes[1].set_title("Defect Coverage: A vs B")
        axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_iou_distribution.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Advanced plot helpers
# ---------------------------------------------------------------------------

def _plot_calibration_curve(df, output_dir, prefix, name_a, name_b):
    """Reliability diagram: mean confidence vs fraction correct per confidence bin.

    Only works when a 'label' column (ground truth) is present.
    Falls back to a confidence distribution comparison otherwise.
    """
    has_gt = "label" in df.columns and "pred_a" in df.columns

    fig, ax = plt.subplots(figsize=(6, 5))

    if has_gt:
        n_bins = 10
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for model, color, name in [("a", "#2196F3", name_a), ("b", "#FF9800", name_b)]:
            conf_col = f"confidence_{model}"
            pred_col = f"pred_{model}"
            confs  = df[conf_col].values
            correct = (df[pred_col].values == df["label"].values).astype(float)
            bin_confs, bin_accs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (confs > lo) & (confs <= hi)
                if mask.sum() > 0:
                    bin_confs.append(float(confs[mask].mean()))
                    bin_accs.append(float(correct[mask].mean()))
            ax.plot(bin_confs, bin_accs, "o-", color=color, label=name, markersize=5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean confidence (per bin)")
        ax.set_ylabel("Fraction correct")
        ax.set_title("Reliability Diagram (Calibration Curve)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    else:
        # No labels: compare confidence distributions, zoomed to actual data range
        all_conf = pd.concat([df["confidence_a"], df["confidence_b"]])
        x_lo = max(0.0, float(all_conf.quantile(0.01)) - 0.05)
        x_hi = min(1.0, float(all_conf.quantile(0.999)) + 0.02)
        bins = np.linspace(x_lo, x_hi, 35)
        ax.hist(df["confidence_a"], bins=bins, alpha=0.6, color="#2196F3",
                label=f"{name_a}  μ={df['confidence_a'].mean():.3f}", density=True)
        ax.hist(df["confidence_b"], bins=bins, alpha=0.6, color="#FF9800",
                label=f"{name_b}  μ={df['confidence_b'].mean():.3f}", density=True)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Density")
        ax.set_title("Confidence Distribution — A vs B\n(no ground-truth labels; zoomed to data range)")

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_calibration_curve.png"), dpi=150)
    plt.close(fig)


def _plot_entropy_distribution(df, output_dir, prefix, name_a, name_b):
    """Overlay histogram of prediction entropy for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram overlay
    bins = 35
    axes[0].hist(df["entropy_a"], bins=bins, alpha=0.65, color="#2196F3",
                 label=f"{name_a}  μ={df['entropy_a'].mean():.3f}", density=True)
    axes[0].hist(df["entropy_b"], bins=bins, alpha=0.65, color="#FF9800",
                 label=f"{name_b}  μ={df['entropy_b'].mean():.3f}", density=True)
    axes[0].set_xlabel("Prediction entropy (nats)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction Entropy Distribution")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # KL divergence annotation (Kullback & Leibler 1951)
    try:
        all_ent = pd.concat([df["entropy_a"].dropna(), df["entropy_b"].dropna()])
        bin_edges = np.linspace(all_ent.min(), all_ent.max(), 51)
        hist_a, _ = np.histogram(df["entropy_a"].dropna(), bins=bin_edges, density=True)
        hist_b, _ = np.histogram(df["entropy_b"].dropna(), bins=bin_edges, density=True)
        eps = 1e-10
        ha = hist_a + eps; hb = hist_b + eps
        ha = ha / ha.sum(); hb = hb / hb.sum()
        kl_ab = float(np.sum(ha * np.log(ha / hb)))
        if not np.isfinite(kl_ab) or kl_ab > 50:
            kl_text = r"D_KL(A||B) $\gg$ 0"
            kl_interp = "Non-overlapping distributions"
        elif kl_ab < 0.05:
            kl_text = f"D_KL(A||B) = {kl_ab:.4f}"
            kl_interp = "Similar uncertainty"
        else:
            kl_text = f"D_KL(A||B) = {kl_ab:.4f}"
            kl_interp = "Different uncertainty profiles"
        axes[0].text(0.97, 0.97,
                     f"{kl_text}\n{kl_interp}",
                     transform=axes[0].transAxes, va="top", ha="right",
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="white", alpha=0.85,
                                           edgecolor="#607D8B"))
    except Exception:
        pass  # graceful fallback

    # Entropy scatter: A vs B per image, coloured by agreement
    agrees_mask = df["agrees"] == 1
    axes[1].scatter(df.loc[agrees_mask, "entropy_a"],
                    df.loc[agrees_mask, "entropy_b"],
                    alpha=0.25, s=8, color="#4CAF50", label="Agree")
    axes[1].scatter(df.loc[~agrees_mask, "entropy_a"],
                    df.loc[~agrees_mask, "entropy_b"],
                    alpha=0.5, s=8, color="#F44336", label="Disagree")
    lim = max(df["entropy_a"].max(), df["entropy_b"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    axes[1].set_xlabel(f"Entropy — {name_a}")
    axes[1].set_ylabel(f"Entropy — {name_b}")
    axes[1].set_title("Entropy Scatter: A vs B (by agreement)")
    axes[1].legend(fontsize=9, markerscale=2)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_entropy_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_confidence_bands(df, output_dir, prefix):
    """Disagreement rate per confidence band of Model A."""
    from steeldefect_active_learning.ab_testing.ab_pipeline import confidence_by_band
    band_df = confidence_by_band(df, n_bands=5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Disagreement rate per band
    colors = ["#4CAF50" if r < 0.05 else "#FF9800" if r < 0.15 else "#F44336"
              for r in band_df["disagreement_rate"]]
    bars = axes[0].bar(band_df["band"], band_df["disagreement_rate"],
                       color=colors, edgecolor="white", linewidth=1.2)
    for bar, row in zip(bars, band_df.itertuples()):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"n={row.n}", ha="center", fontsize=8)
    axes[0].axhline(0.10, color="#F44336", linestyle="--", alpha=0.6, label="10% threshold")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlabel("Confidence band (Model A)")
    axes[0].set_ylabel("Disagreement rate")
    axes[0].set_title("Disagreement Rate by Confidence Band\n"
                      "High disagreement at high confidence = systematic difference")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Entropy per band (if available)
    if "mean_entropy_a" in band_df.columns:
        axes[1].bar(band_df["band"], band_df["mean_entropy_a"],
                    color="#2196F3", alpha=0.7, label="Model A", edgecolor="white")
        if "mean_entropy_b" in band_df.columns:
            axes[1].bar(band_df["band"], band_df["mean_entropy_b"],
                        color="#FF9800", alpha=0.6, label="Model B", edgecolor="white")
        axes[1].set_xlabel("Confidence band (Model A)")
        axes[1].set_ylabel("Mean prediction entropy (nats)")
        axes[1].set_title("Mean Entropy by Confidence Band")
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "entropy not available\n(run with classification task)",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=10, color="#607D8B")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_confidence_bands.png"), dpi=150)
    plt.close(fig)


def _plot_per_class_accuracy_delta(df, output_dir, prefix, name_a, name_b):
    """Per-class accuracy delta: acc(B) − acc(A) for each class.

    Green bars = B improved.  Red bars = B degraded.
    THE most actionable viz for deployment: "will any class suffer if we switch?"

    Requires: pred_a, pred_b, label columns.
    Falls back gracefully if 'label' column is missing.
    """
    n_classes = len(df["label"].unique()) if "label" in df.columns else 5
    fig_h = max(5, 0.5 * n_classes + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.18)

    if "label" not in df.columns or "pred_a" not in df.columns or "pred_b" not in df.columns:
        ax.text(0.5, 0.5, "Per-class accuracy delta requires\npred_a, pred_b, label columns",
                ha="center", va="center", fontsize=11, color="#607D8B",
                transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle("Per-class Accuracy Delta — data unavailable", fontsize=10)
        fig.savefig(os.path.join(output_dir, f"{prefix}_per_class_accuracy_delta.png"), dpi=150)
        plt.close(fig)
        return

    classes = sorted(df["label"].unique())
    deltas = []
    acc_a_list = []
    acc_b_list = []
    for c in classes:
        mask = df["label"] == c
        acc_a = float((df.loc[mask, "pred_a"] == c).mean())
        acc_b = float((df.loc[mask, "pred_b"] == c).mean())
        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        deltas.append(acc_b - acc_a)

    # --- C1: Multiple testing correction (Benjamini-Hochberg) ---
    sig_df = per_class_significance_test(df, alpha=0.05, correction="benjamini-hochberg")
    sig_map = {}  # class_name → (p_adj, significant)
    if not sig_df.empty:
        for _, row in sig_df.iterrows():
            sig_map[row["class_name"]] = (row["p_value_adjusted"], row["significant"])

    # --- C2: Bootstrap CIs for per-class accuracy delta ---
    boot_df = bootstrap_per_class_delta(df, n_bootstrap=1000, ci=0.95)
    ci_map = {}  # class_name → (ci_low, ci_high)
    if not boot_df.empty:
        for _, row in boot_df.iterrows():
            ci_map[row["class_name"]] = (row["ci_low"], row["ci_high"])

    x_labels = [_class_label(c) for c in classes]
    bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]

    # Draw bars with bootstrap error bars
    ci_errors_low = []
    ci_errors_high = []
    for i, lbl in enumerate(x_labels):
        if lbl in ci_map:
            ci_low, ci_high = ci_map[lbl]
            ci_errors_low.append(deltas[i] - ci_low)
            ci_errors_high.append(ci_high - deltas[i])
        else:
            ci_errors_low.append(0)
            ci_errors_high.append(0)

    bars = ax.barh(range(len(classes)), deltas, color=bar_colors,
                   edgecolor="white", height=0.6,
                   xerr=[ci_errors_low, ci_errors_high],
                   error_kw=dict(capsize=3, capthick=1, ecolor="#555555", lw=1))
    ax.axvline(0, color="#212121", lw=1.2)

    # Annotate with actual accuracies + significance stars
    x_min = min(deltas) if deltas else -0.1
    x_max = max(deltas) if deltas else 0.1
    x_range = max(abs(x_min), abs(x_max), 0.05)

    for i, (d, a_a, a_b) in enumerate(zip(deltas, acc_a_list, acc_b_list)):
        n_cls = int((df["label"] == classes[i]).sum())
        tag = "DEGRADED " if d < -0.05 else ("IMPROVED " if d > 0.05 else "")
        label = f"{tag}A:{a_a:.0%}\u2192B:{a_b:.0%} (n={n_cls})"
        # Add significance stars (BH-corrected)
        lbl = x_labels[i]
        if lbl in sig_map:
            p_adj, is_sig = sig_map[lbl]
            if p_adj < 0.001:
                label += " ***"
            elif p_adj < 0.01:
                label += " **"
            elif p_adj < 0.05:
                label += " *"
        # Adaptive label placement: inside bar if bar is long
        bar_fraction = abs(d) / x_range if x_range > 0 else 0
        if bar_fraction > 0.6 and abs(d) > 0.08:
            x_pos = d * 0.5
            ha = "center"
            txt_color = "white"
            fw = "bold"
        else:
            x_pos = d + (0.005 if d >= 0 else -0.005)
            ha = "left" if d >= 0 else "right"
            txt_color = "#C62828" if d < -0.05 else "#212121"
            fw = "normal"
        ax.text(x_pos, i, label, ha=ha, va="center", fontsize=7,
                color=txt_color, fontweight=fw)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Accuracy delta  (acc_B − acc_A)")
    ax.set_title("Per-class Accuracy Delta  (BH-corrected, 95% bootstrap CI)\n"
                 "Green = B improved  ·  Red = B degraded  ·  "
                 "* p<.05  ** p<.01  *** p<.001", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle(f"Per-class Accuracy: {name_b} vs {name_a}",
                 fontsize=11, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(output_dir, f"{prefix}_per_class_accuracy_delta.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Advanced visualisation v2: KDE heatmap, violin, entropy CDF, decomposition
# ---------------------------------------------------------------------------

def plot_advanced_comparison_v2(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    prefix: str = "ab",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> None:
    """Generate 4 deeper A/B diagnostic PNGs.

    Saved files:
        {prefix}_kde_heatmap.png            — 2D KDE density of conf_a vs conf_b
        {prefix}_violin_per_class.png       — confidence violin per class (A vs B)
        {prefix}_mcnemar_test.png           — McNemar per-class significance test
        {prefix}_disagreement_decomposition.png — systematic/borderline/random split
    """
    os.makedirs(output_dir, exist_ok=True)

    _plot_kde_heatmap(df, output_dir, prefix, model_a_name, model_b_name)

    if "pred_a" in df.columns:
        _plot_violin_per_class(df, output_dir, prefix, model_a_name, model_b_name)

    if "pred_a" in df.columns:
        _plot_mcnemar_test(df, output_dir, prefix, model_a_name, model_b_name)

    if "margin_a" in df.columns:
        _plot_disagreement_decomposition(df, output_dir, prefix)

    plt.close("all")


def _plot_kde_heatmap(df, output_dir, prefix, name_a, name_b):
    """2D KDE density contour: confidence_a vs confidence_b."""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.95)

    ca = df["confidence_a"].values.astype(float)
    cb = df["confidence_b"].values.astype(float)

    # KDE on a grid
    grid_res = 80
    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    XX, YY = np.meshgrid(xi, yi)
    positions = np.vstack([XX.ravel(), YY.ravel()])
    try:
        kde = gaussian_kde(np.vstack([ca, cb]), bw_method="silverman")
        ZZ = kde(positions).reshape(grid_res, grid_res)
        ax.contourf(XX, YY, ZZ, levels=14, cmap="YlOrRd", alpha=0.85)
        ax.contour(XX, YY, ZZ, levels=6, colors="white", linewidths=0.5, alpha=0.5)
    except Exception:
        print("[metrics] WARNING: KDE contour failed — showing scatter only")

    # Scatter overlay coloured by agreement
    agrees = df["agrees"].values
    ax.scatter(ca[agrees == 1], cb[agrees == 1],
               s=6, alpha=0.25, color="#4CAF50", label="Agree")
    ax.scatter(ca[agrees == 0], cb[agrees == 0],
               s=6, alpha=0.45, color="#F44336", label="Disagree")

    # Auto-zoom to where data actually is (with a small margin)
    margin = 0.04
    x_lo = max(0.0, float(np.percentile(ca, 0.5)) - margin)
    x_hi = min(1.0, float(np.percentile(ca, 99.5)) + margin)
    y_lo = max(0.0, float(np.percentile(cb, 0.5)) - margin)
    y_hi = min(1.0, float(np.percentile(cb, 99.5)) + margin)
    ax.plot([x_lo, x_hi], [x_lo, x_hi], "w--", alpha=0.6, lw=1.2, label="Equal confidence")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(f"Confidence — {name_a}", fontsize=9)
    ax.set_ylabel(f"Confidence — {name_b}", fontsize=9)
    ax.set_title(
        "2D KDE Density — Confidence Space\n"
        "Hotspots show where most images fall; scatter = agreement/disagreement",
        fontsize=9,
    )
    ax.legend(fontsize=8, markerscale=2)

    fig.savefig(os.path.join(output_dir, f"{prefix}_kde_heatmap.png"), dpi=150)
    plt.close(fig)


def _plot_violin_per_class(df, output_dir, prefix, name_a, name_b):
    """Violin plot of confidence per predicted class (A vs B side-by-side)."""
    classes = sorted(df["pred_a"].unique())
    k = len(classes)

    fig, ax = plt.subplots(figsize=(max(7, k * 1.6), 5))
    fig.subplots_adjust(top=0.86, bottom=0.15)

    width = 0.35
    offsets = {"a": -width / 2, "b": width / 2}
    colors  = {"a": "#2196F3", "b": "#FF9800"}
    labels  = {"a": name_a,    "b": name_b}

    for side, col in [("a", "confidence_a"), ("b", "confidence_b")]:
        data = [df.loc[df["pred_a"] == c, col].values for c in classes]
        data = [d if len(d) > 1 else np.array([0.5, 0.5]) for d in data]
        positions = [i + offsets[side] for i in range(k)]
        parts = ax.violinplot(data, positions=positions,
                              widths=width * 0.9,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[side])
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("#212121")
        parts["cmedians"].set_linewidth(1.5)

    # Agreement rate annotation per class
    for i, c in enumerate(classes):
        sub = df[df["pred_a"] == c]
        ag  = sub["agrees"].mean() if len(sub) > 0 else 0
        n   = len(sub)
        ax.text(i, 1.04, f"n={n}\n{ag:.0%}", ha="center",
                fontsize=7.5, color="#546E7A", va="bottom")

    ax.set_xticks(range(k))
    ax.set_xticklabels([_class_label(c) for c in classes], fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Confidence score")
    ax.set_title(
        "Confidence Distribution per Class — A vs B\n"
        "Width = density  ·  Bar = median  ·  n / agreement rate shown above",
        fontsize=9,
    )
    legend_patches = [
        mpatches.Patch(color=colors["a"], alpha=0.7, label=name_a),
        mpatches.Patch(color=colors["b"], alpha=0.7, label=name_b),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(output_dir, f"{prefix}_violin_per_class.png"), dpi=150)
    plt.close(fig)


def _plot_mcnemar_test(df, output_dir, prefix, name_a, name_b):
    """McNemar's paired classifier test (McNemar 1947).

    Two-panel figure:
        Left:  2×2 contingency heatmap (A-correct/B-correct counts).
        Right: Statistical results (χ², p-value, interpretation).

    Requires columns: pred_a, pred_b, label (ground truth).
    Falls back to a placeholder if 'label' column is missing.
    """
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(top=0.85, bottom=0.12, wspace=0.35)

    if "label" not in df.columns or "pred_a" not in df.columns or "pred_b" not in df.columns:
        for ax in axes:
            ax.text(0.5, 0.5, "McNemar's test requires\npred_a, pred_b, label columns",
                    ha="center", va="center", fontsize=11, color="#607D8B",
                    transform=ax.transAxes)
            ax.axis("off")
        fig.suptitle("McNemar's Paired Classifier Test — data unavailable", fontsize=10)
        fig.savefig(os.path.join(output_dir, f"{prefix}_mcnemar_test.png"), dpi=150)
        plt.close(fig)
        return

    # Build 2×2 contingency table
    a_correct = (df["pred_a"] == df["label"]).values
    b_correct = (df["pred_b"] == df["label"]).values
    n00 = int((a_correct & b_correct).sum())       # both right
    n01 = int((a_correct & ~b_correct).sum())      # A right, B wrong
    n10 = int((~a_correct & b_correct).sum())      # B right, A wrong
    n11 = int((~a_correct & ~b_correct).sum())     # both wrong
    n_total = len(df)

    # McNemar test statistic
    discordant = n01 + n10
    if discordant >= 25:
        chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = 1.0 - sp_stats.chi2.cdf(chi2, df=1)
        test_type = "χ² (continuity-corrected)"
    elif discordant > 0:
        p_value = float(sp_stats.binomtest(n01, discordant, 0.5).pvalue)
        chi2 = float("nan")
        test_type = "Exact binomial"
    else:
        chi2, p_value = 0.0, 1.0
        test_type = "No discordant pairs"

    # Left panel: 2×2 heatmap
    table = np.array([[n00, n01], [n10, n11]])
    ax = axes[0]
    im = ax.imshow(table, cmap="RdYlGn", vmin=0, vmax=max(n_total // 2, 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{name_b}\ncorrect", f"{name_b}\nwrong"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{name_a}\ncorrect", f"{name_a}\nwrong"], fontsize=9)
    for i in range(2):
        for j in range(2):
            pct = table[i, j] / max(n_total, 1) * 100
            ax.text(j, i, f"{table[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if table[i, j] > n_total * 0.3 else "black")
    ax.set_title("McNemar Contingency Table\n(rows = A, cols = B)", fontsize=9)

    # Right panel: statistical results
    ax2 = axes[1]
    ax2.axis("off")
    verdict = "Models differ significantly" if p_value < 0.05 else "No significant difference"
    verdict_color = "#F44336" if p_value < 0.05 else "#4CAF50"

    lines = [
        ("Test type:", test_type),
        ("χ² statistic:", f"{chi2:.4f}" if np.isfinite(chi2) else "N/A (exact test)"),
        ("p-value:", f"{p_value:.6f}"),
        ("", ""),
        ("Discordant pairs:", f"{discordant}  (n₀₁={n01}, n₁₀={n10})"),
        ("Concordant pairs:", f"{n00 + n11}  (both right={n00}, both wrong={n11})"),
        ("", ""),
        ("Accuracy A:", f"{(n00 + n01) / max(n_total, 1):.1%}"),
        ("Accuracy B:", f"{(n00 + n10) / max(n_total, 1):.1%}"),
        ("", ""),
        ("Verdict:", verdict),
    ]
    y = 0.92
    for key, val in lines:
        if key == "Verdict:":
            ax2.text(0.05, y, key, fontsize=10, fontweight="bold",
                     transform=ax2.transAxes, color="#212121")
            ax2.text(0.40, y, val, fontsize=10, fontweight="bold",
                     transform=ax2.transAxes, color=verdict_color)
        elif key:
            ax2.text(0.05, y, key, fontsize=9, color="#607D8B",
                     transform=ax2.transAxes, fontstyle="italic")
            ax2.text(0.40, y, val, fontsize=9, color="#212121",
                     transform=ax2.transAxes)
        y -= 0.08

    fig.suptitle("McNemar's Paired Classifier Test (McNemar 1947)\n"
                 "H₀: both models have the same error rate  ·  "
                 "Uses only discordant pairs (n₀₁ vs n₁₀)",
                 fontsize=10, fontweight="bold", y=0.98)

    fig.savefig(os.path.join(output_dir, f"{prefix}_mcnemar_test.png"), dpi=150)
    plt.close(fig)


def _plot_disagreement_decomposition(df, output_dir, prefix):
    """Stacked bar decomposing disagreement into systematic / borderline / random."""
    # Use percentile-based thresholds so the split is meaningful on any data distribution
    all_margins = pd.concat([df["margin_a"], df["margin_b"]])
    margin_threshold_high = float(np.percentile(all_margins, 75))  # top quartile = confident
    margin_threshold_low  = float(np.percentile(all_margins, 25))  # bottom quartile = uncertain

    disagree = df[df["agrees"] == 0].copy()

    if len(disagree) == 0:
        # All agree — show a single bar labeled "0% disagreement"
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Disagreement"], [0], color="#4CAF50")
        ax.set_ylim(0, 1)
        ax.set_title("Disagreement Decomposition\n(0% disagreement — models fully agree)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_disagreement_decomposition.png"), dpi=150)
        plt.close(fig)
        return

    systematic = ((disagree["margin_a"] >= margin_threshold_high) &
                  (disagree["margin_b"] >= margin_threshold_high))
    borderline = ((disagree["margin_a"] < margin_threshold_low) |
                  (disagree["margin_b"] < margin_threshold_low))
    random_dis = ~systematic & ~borderline

    n_total = len(df)
    counts = {
        "Systematic\n(both confident,\ndifferent answer)": systematic.sum(),
        "Borderline\n(at least one\nlow-margin)":          borderline.sum(),
        "Random\n(mid-confidence\ndisagreement)":          random_dis.sum(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(top=0.87, bottom=0.14, wspace=0.38)

    # Left: absolute counts
    bar_colors = ["#C62828", "#FF9800", "#607D8B"]
    bars = axes[0].bar(list(counts.keys()), list(counts.values()),
                       color=bar_colors, edgecolor="white", linewidth=1.2)
    for bar, count in zip(bars, counts.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + n_total * 0.005,
                     f"{count}\n({count/n_total:.1%})",
                     ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Disagreement Decomposition\n(absolute counts)", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, max(counts.values()) * 1.25)

    # Right: stacked bar (agree + 3 disagree types)
    agree_count = int(df["agrees"].sum())
    all_counts = [agree_count] + list(counts.values())
    all_labels = ["Agree"] + list(counts.keys())
    all_colors = ["#4CAF50"] + bar_colors
    bottom = 0
    for label, count, color in zip(all_labels, all_counts, all_colors):
        axes[1].bar(["All images"], [count], bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.8,
                    label=f"{label.split(chr(10))[0]} ({count/n_total:.0%})")
        bottom += count
    axes[1].set_ylim(0, n_total * 1.05)
    axes[1].set_ylabel("Cumulative count")
    axes[1].set_title("Composition of All Predictions\n(agree + disagree breakdown)", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(1.02, 1))

    fig.suptitle(
        "Disagreement Decomposition\n"
        f"Systematic = both top-quartile margin (>{margin_threshold_high:.2f}, p75) but disagree  "
        f"·  Borderline = bottom-quartile (<{margin_threshold_low:.2f}, p25)",
        fontsize=9, y=1.01,
    )

    fig.savefig(os.path.join(output_dir, f"{prefix}_disagreement_decomposition.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration metrics  (Guo et al. 2017, Ovadia et al. 2019)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al. 2017).

    ECE = Σ_{m=1}^{M} (|B_m| / n) · |acc(B_m) - conf(B_m)|

    Args:
        probs:  Predicted probabilities, shape (N, C) or (N,).
        labels: True class indices, shape (N,).
        n_bins: Equal-width confidence bins (default 15).

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(predictions[mask] == labels[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (Guo et al. 2017).

    MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        probs:  Predicted probabilities, shape (N, C) or (N,).
        labels: True class indices, shape (N,).
        n_bins: Equal-width confidence bins (default 15).

    Returns:
        MCE in [0, 1]. Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(predictions[mask] == labels[mask]))
        conf = float(np.mean(confidences[mask]))
        mce = max(mce, abs(acc - conf))
    return float(mce)


def calibration_under_shift(
    df: pd.DataFrame,
    n_quartiles: int = 4,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Calibration by entropy quartile — reveals degradation under shift (Ovadia 2019).

    Ovadia et al. 2019 showed that calibration degrades for high-entropy
    (uncertain) predictions under covariate shift.  This function groups
    predictions by their entropy quartile and reports ECE / accuracy per group.

    Requires columns:
        'probs_a'   — numpy array or list of softmax probs for Model A
        'labels'    — true class indices
        'entropy_a' — prediction entropy of Model A

    Returns:
        DataFrame with columns: quartile, entropy_range, ece, accuracy,
        mean_confidence, n_samples.
    """
    if "entropy_a" not in df.columns or "label" not in df.columns:
        return pd.DataFrame()

    entropy = df["entropy_a"].values
    q_edges = np.quantile(entropy, np.linspace(0, 1, n_quartiles + 1))
    rows = []

    for q in range(n_quartiles):
        lo, hi = q_edges[q], q_edges[q + 1]
        if q == n_quartiles - 1:
            mask = (entropy >= lo) & (entropy <= hi)
        else:
            mask = (entropy >= lo) & (entropy < hi)

        if mask.sum() < 2:
            continue

        labels_q = df.loc[mask, "label"].values.astype(np.int64)
        conf_q = df.loc[mask, "confidence_a"].values.astype(np.float64)
        pred_q = df.loc[mask, "pred_a"].values.astype(np.int64) if "pred_a" in df.columns else None

        if pred_q is not None:
            acc = float(np.mean(pred_q == labels_q))
        else:
            acc = float("nan")

        # ECE on this group using confidence column
        probs_1d = conf_q
        if pred_q is not None:
            ece_q = float(
                np.sum(
                    np.abs(
                        (pred_q == labels_q).astype(float) - conf_q
                    )
                ) / len(labels_q)
            )
        else:
            ece_q = float("nan")

        rows.append({
            "quartile": q + 1,
            "entropy_range": f"[{lo:.3f}, {hi:.3f}]",
            "ece": ece_q,
            "accuracy": acc,
            "mean_confidence": float(conf_q.mean()),
            "n_samples": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def aleatoric_epistemic_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class aleatoric vs. epistemic uncertainty report (Kendall & Gal 2017).

    Requires BALD scores computed with compute_uncertainty_bald_full().
    Expects columns: 'pred_a', 'bald_epistemic', 'bald_aleatoric'.

    High aleatoric per class  → inherently ambiguous (label noise, overlapping appearance).
    High epistemic per class  → model needs more labeled data for that class.

    Returns:
        DataFrame with columns: class_id, mean_aleatoric, mean_epistemic,
        epistemic_fraction, n_samples.
    """
    required = {"pred_a", "bald_epistemic", "bald_aleatoric"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for class_id, grp in df.groupby("pred_a"):
        mean_al = float(grp["bald_aleatoric"].mean())
        mean_ep = float(grp["bald_epistemic"].mean())
        total = mean_al + mean_ep + 1e-12
        rows.append({
            "class_id": int(class_id),
            "mean_aleatoric": mean_al,
            "mean_epistemic": mean_ep,
            "epistemic_fraction": mean_ep / total,
            "n_samples": len(grp),
        })

    return pd.DataFrame(rows).sort_values("epistemic_fraction", ascending=False)


# ---------------------------------------------------------------------------
# Multiple testing correction (Benjamini-Hochberg)  — C1
# ---------------------------------------------------------------------------

def per_class_significance_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
    correction: str = "benjamini-hochberg",
) -> pd.DataFrame:
    """Per-class accuracy comparison with multiple testing correction.

    For each class c:
        1. Compute acc_A(c) and acc_B(c)
        2. McNemar's test on the 2x2 contingency (A-correct/B-correct)
        3. Collect all raw p-values
        4. Apply correction to control family-wise or FDR error rate

    Corrections:
        bonferroni:          p_adj = p * n_tests.  Controls FWER.  Conservative.
        benjamini-hochberg:  Rank p-values, p_adj(i) = p * m / rank.  Controls FDR.
        holm:                Step-down Bonferroni.  Less conservative than Bonferroni.

    With 11 defect classes at alpha=0.05, uncorrected FWER = 1-(0.95)^11 = 43%.
    BH controls FDR at 5%.

    Args:
        df:         A/B DataFrame with pred_a, pred_b, label columns.
        alpha:      Significance level.
        correction: Multiple testing correction method.

    Returns:
        DataFrame with: class_name, acc_a, acc_b, delta, n_samples,
        p_value_raw, p_value_adjusted, significant.
    """
    from scipy import stats as sp_stats

    required = {"pred_a", "pred_b", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    classes = sorted(df["label"].unique())
    rows = []

    for c in classes:
        mask = df["label"] == c
        sub = df.loc[mask]
        n = int(mask.sum())
        if n == 0:
            continue

        a_correct = (sub["pred_a"] == c)
        b_correct = (sub["pred_b"] == c)
        acc_a = float(a_correct.mean())
        acc_b = float(b_correct.mean())

        # McNemar contingency on discordant pairs
        n01 = int((a_correct & ~b_correct).sum())   # A right, B wrong
        n10 = int((~a_correct & b_correct).sum())   # B right, A wrong
        discordant = n01 + n10

        if discordant == 0:
            p_raw = 1.0
        elif discordant >= 25:
            chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
            p_raw = float(1 - sp_stats.chi2.cdf(chi2, df=1))
        else:
            p_raw = float(sp_stats.binomtest(n01, discordant, 0.5).pvalue)

        rows.append({
            "class_name": _class_label(c),
            "acc_a": acc_a,
            "acc_b": acc_b,
            "delta": acc_b - acc_a,
            "n_samples": n,
            "p_value_raw": p_raw,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    m = len(result)
    raw_p = result["p_value_raw"].values.copy()

    if correction == "bonferroni":
        adj_p = np.minimum(raw_p * m, 1.0)

    elif correction == "holm":
        order = np.argsort(raw_p)
        adj_p = np.zeros(m)
        for rank_idx, orig_idx in enumerate(order):
            adj_p[orig_idx] = raw_p[orig_idx] * (m - rank_idx)
        # Enforce monotonicity (step-down)
        running_max = 0.0
        for orig_idx in order:
            running_max = max(running_max, adj_p[orig_idx])
            adj_p[orig_idx] = min(running_max, 1.0)

    else:  # benjamini-hochberg (default)
        order = np.argsort(raw_p)
        adj_p = np.zeros(m)
        for rank_idx, orig_idx in enumerate(order):
            adj_p[orig_idx] = raw_p[orig_idx] * m / (rank_idx + 1)
        # Enforce monotonicity (step-up): walk backwards
        for i in range(m - 2, -1, -1):
            orig_idx = order[i]
            next_idx = order[i + 1]
            adj_p[orig_idx] = min(adj_p[orig_idx], adj_p[next_idx])
        adj_p = np.minimum(adj_p, 1.0)

    result["p_value_adjusted"] = adj_p
    result["significant"] = adj_p < alpha
    return result.sort_values("delta", ascending=True)


# ---------------------------------------------------------------------------
# Bootstrap CIs for per-class accuracy delta  — C2
# ---------------------------------------------------------------------------

def bootstrap_per_class_delta(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for per-class accuracy delta (B − A).

    For each class c:
        1. Select rows where label == c
        2. Resample with replacement n_bootstrap times
        3. Compute delta = acc_B − acc_A on each resample
        4. Return percentile CI

    Args:
        df:           A/B DataFrame with pred_a, pred_b, label.
        n_bootstrap:  Number of resamples (default 1000).
        ci:           Confidence level (default 0.95).
        random_state: Seed for reproducibility.

    Returns:
        DataFrame: class_name, delta_mean, ci_low, ci_high, n_samples.
    """
    required = {"pred_a", "pred_b", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    alpha = (1 - ci) / 2
    classes = sorted(df["label"].unique())
    rows = []

    for c in classes:
        mask = df["label"] == c
        sub = df.loc[mask]
        n = int(mask.sum())
        if n < 2:
            rows.append({
                "class_name": _class_label(c),
                "delta_mean": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "n_samples": n,
            })
            continue

        a_correct = (sub["pred_a"] == c).values
        b_correct = (sub["pred_b"] == c).values

        boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            acc_a = a_correct[idx].mean()
            acc_b = b_correct[idx].mean()
            boot_deltas[b] = acc_b - acc_a

        rows.append({
            "class_name": _class_label(c),
            "delta_mean": float(boot_deltas.mean()),
            "ci_low": float(np.percentile(boot_deltas, alpha * 100)),
            "ci_high": float(np.percentile(boot_deltas, (1 - alpha) * 100)),
            "n_samples": n,
        })

    return pd.DataFrame(rows)

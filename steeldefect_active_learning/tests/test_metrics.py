"""Unit tests for the A/B testing cascade in `ab_testing/metrics.py`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steeldefect_active_learning.ab_testing.metrics import (
    apply_cuped,
    bootstrap_ci,
    compute_all_metrics,
    correct_pvalues,
    hedges_correction,
    mcnemar_test,
    per_class_mcnemar,
    srm_check,
    statistical_significance_test,
)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

class TestCorrectPvalues:
    def test_bonferroni_matches_known(self):
        # All p = 0.04, k = 5 → corrected = 0.20.
        out = correct_pvalues([0.04] * 5, method="bonferroni", alpha=0.05)
        np.testing.assert_allclose(out["p_corrected"], [0.20] * 5)
        assert not out["reject"].any()

    def test_holm_monotonic(self):
        ps = [0.001, 0.01, 0.03, 0.06, 0.5]
        out = correct_pvalues(ps, method="holm", alpha=0.05)
        # Holm-corrected p-values must be monotone non-decreasing in the original order.
        ranks = np.argsort(ps)
        adjusted_in_rank_order = out["p_corrected"][ranks]
        assert np.all(np.diff(adjusted_in_rank_order) >= -1e-12)

    def test_bh_smaller_than_bonferroni(self):
        ps = [0.001, 0.01, 0.04, 0.06, 0.5]
        bh   = correct_pvalues(ps, method="benjamini-hochberg")
        bonf = correct_pvalues(ps, method="bonferroni")
        assert (bh["p_corrected"] <= bonf["p_corrected"] + 1e-12).all()

    def test_none_passthrough(self):
        out = correct_pvalues([0.001, 0.5], method="none", alpha=0.05)
        np.testing.assert_allclose(out["p_corrected"], [0.001, 0.5])
        assert out["reject"].tolist() == [True, False]

    def test_empty_input(self):
        out = correct_pvalues([], method="benjamini-hochberg")
        assert len(out["p_corrected"]) == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            correct_pvalues([0.05], method="xyz")


# ---------------------------------------------------------------------------
# SRM
# ---------------------------------------------------------------------------

class TestSRM:
    def test_balanced_passes(self):
        out = srm_check(500, 500, expected_ratio=0.5)
        assert out["srm_detected"] is False
        assert out["p_value"] > 0.99

    def test_imbalanced_flags(self):
        # 600/400 split is roughly 7σ off 50/50 — SRM should fire.
        out = srm_check(600, 400, expected_ratio=0.5)
        assert out["srm_detected"] is True
        assert out["p_value"] < 1e-3

    def test_zero_n(self):
        out = srm_check(0, 0)
        assert out["srm_detected"] is False


# ---------------------------------------------------------------------------
# Hedges correction
# ---------------------------------------------------------------------------

class TestHedges:
    def test_small_n_correction_strict(self):
        d = 0.5
        d_corr = hedges_correction(d, n_a=5, n_b=5)
        assert d_corr < d  # should shrink for small n

    def test_large_n_correction_negligible(self):
        d = 0.5
        d_corr = hedges_correction(d, n_a=500, n_b=500)
        assert abs(d_corr - d) < 0.01  # within 2% for large n


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

class TestMcNemar:
    def test_no_ground_truth_returns_not_applicable(self):
        df = pd.DataFrame({"pred_a": [0, 1], "pred_b": [0, 1]})  # no label column
        out = mcnemar_test(df)
        assert out["applicable"] is False

    def test_paired_correctness_yates(self, rng):
        # Construct discordant pairs > 25 to trigger Yates branch.
        n = 200
        correct_a = rng.random(n) < 0.7
        correct_b = correct_a.copy()
        flip_idx = rng.integers(0, n, size=60)
        correct_b[flip_idx] = ~correct_b[flip_idx]
        df = pd.DataFrame({
            "correct_a": correct_a.astype(int),
            "correct_b": correct_b.astype(int),
        })
        out = mcnemar_test(df)
        assert out["applicable"] is True
        assert out["test_used"] == "mcnemar_yates"
        assert 0.0 <= out["p_value"] <= 1.0

    def test_exact_branch_triggers_for_small_discordant(self):
        df = pd.DataFrame({
            "correct_a": [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            "correct_b": [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        })
        out = mcnemar_test(df)
        # 3 discordant pairs → exact branch
        assert out["test_used"] == "mcnemar_exact"

    def test_per_class_mcnemar_returns_corrected_pvalues(self, ab_df_classification):
        out = per_class_mcnemar(
            ab_df_classification, correction="benjamini-hochberg", alpha=0.05,
        )
        if out.empty:
            pytest.skip("not enough labels per class to run per-class McNemar")
        assert "p_corrected" in out.columns
        assert "reject" in out.columns
        # BH-corrected p-values should be ≥ raw p-values.
        assert (out["p_corrected"].values >= out["p_value"].values - 1e-12).all()


# ---------------------------------------------------------------------------
# CUPED
# ---------------------------------------------------------------------------

class TestCUPED:
    def test_reduces_variance_when_correlated(self, rng):
        n = 1000
        x = rng.normal(0.0, 1.0, size=n)
        y = 2.0 * x + rng.normal(0.0, 1.0, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        adj = apply_cuped(df, "y", "x")
        var_y = float(np.var(y, ddof=1))
        var_y_cuped = float(np.var(adj["y_cuped"].values, ddof=1))
        # Theoretical reduction: 1 - rho^2 ≈ 4/5 → var should drop ~80%.
        assert var_y_cuped < var_y * 0.5

    def test_constant_covariate_is_noop(self, rng):
        # When the covariate has zero variance, theta=0 and y_cuped == y.
        n = 100
        df = pd.DataFrame({"y": rng.normal(size=n), "x": np.zeros(n)})
        adj = apply_cuped(df, "y", "x")
        np.testing.assert_allclose(adj["y_cuped"].values, adj["y"].values, atol=1e-9)
        assert adj.attrs["y_cuped_theta"] == 0.0

    def test_missing_columns_raise(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(ValueError):
            apply_cuped(df, "y", "missing")
        with pytest.raises(ValueError):
            apply_cuped(df, "missing", "y")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_percentile_ci_brackets_estimate(self, rng):
        a = rng.normal(0.7, 0.1, size=200)
        b = rng.normal(0.6, 0.1, size=200)
        out = bootstrap_ci(a, b, "cohens_d", n_resamples=500,
                           method="percentile", alpha=0.05, seed=0)
        assert out["ci_low"] <= out["estimate"] <= out["ci_high"]

    def test_bca_ci_returns_finite(self, rng):
        a = rng.normal(0.7, 0.1, size=200)
        b = rng.normal(0.6, 0.1, size=200)
        out = bootstrap_ci(a, b, "mean_diff", n_resamples=500,
                           method="bca", alpha=0.05, seed=0)
        assert np.isfinite(out["ci_low"])
        assert np.isfinite(out["ci_high"])
        assert out["ci_low"] <= out["ci_high"]

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.zeros(5), np.zeros(5), "xyz")


# ---------------------------------------------------------------------------
# Statistical significance test
# ---------------------------------------------------------------------------

class TestStatisticalSignificance:
    def test_includes_hedges(self, ab_df_classification):
        stat = statistical_significance_test(ab_df_classification)
        assert "cohens_d_hedges" in stat
        # Hedges < |Cohen's d| in magnitude.
        assert abs(stat["cohens_d_hedges"]) <= abs(stat["cohens_d"])

    def test_null_case_pvalue_high(self, ab_df_balanced):
        stat = statistical_significance_test(ab_df_balanced)
        assert stat["p_value"] > 0.01  # not significant under matched arms


# ---------------------------------------------------------------------------
# compute_all_metrics integration
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_srm_present(self, ab_df_classification):
        out = compute_all_metrics(ab_df_classification)
        assert "srm" in out
        assert out["srm"]["srm_detected"] is False

    def test_per_class_mcnemar_with_correction(self, ab_df_classification):
        out = compute_all_metrics(
            ab_df_classification,
            correction="benjamini-hochberg",
        )
        if "per_class_mcnemar" in out:
            assert out["per_class_mcnemar"]["method"] == "benjamini-hochberg"

    def test_cuped_only_when_covariate(self, ab_df_classification):
        out = compute_all_metrics(ab_df_classification, cuped_covariate=None)
        assert "cuped" not in out
        out2 = compute_all_metrics(
            ab_df_classification, cuped_covariate="entropy_a",
        )
        assert "cuped" in out2
        assert "variance_ratio" in out2["cuped"]

    def test_bootstrap_only_when_requested(self, ab_df_classification):
        out = compute_all_metrics(ab_df_classification, bootstrap_n=0)
        assert "bootstrap" not in out
        out2 = compute_all_metrics(ab_df_classification, bootstrap_n=200)
        assert "bootstrap" in out2

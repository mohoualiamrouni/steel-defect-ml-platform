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
    power_analysis,
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

    def test_bh_relaxes_at_the_boundary(self):
        # Raw p = 0.03 rejects alone; BH multiplies it by 3 / 1 and rejects nothing.
        out = correct_pvalues([0.03, 0.5, 0.6], method="benjamini-hochberg", alpha=0.05)
        np.testing.assert_allclose(out["p_corrected"], [0.09, 0.6, 0.6])
        assert not out["reject"].any()


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
        assert len(out) == 11
        assert "p_corrected" in out.columns
        assert "reject" in out.columns
        # BH-corrected p-values should be ≥ raw p-values.
        assert (out["p_corrected"].values >= out["p_value"].values - 1e-12).all()

    @pytest.mark.parametrize("b, c, test_used, p_value", [
        (5, 14, "mcnemar_exact", 0.063568115234375),   # 2 P(X <= 5), X ~ Bin(19, 1/2)
        (12, 24, "mcnemar_yates", 0.0667535),          # chi2 = 121 / 36
    ])
    def test_reference_two_sided(self, b, c, test_used, p_value):
        df = pd.DataFrame({"correct_a": [1] * b + [0] * c + [1] * 10,
                           "correct_b": [0] * b + [1] * c + [1] * 10})
        out = mcnemar_test(df)
        assert (out["b"], out["c"], out["test_used"]) == (b, c, test_used)
        assert out["p_value"] == pytest.approx(p_value, rel=1e-5)
        assert out["n_concordant"] == 10

    def test_yates_does_not_underflow(self):
        df = pd.DataFrame({"correct_a": [0] * 80, "correct_b": [1] * 80})
        # abs=0: pytest.approx would otherwise accept 0.0 within its default 1e-12.
        assert mcnemar_test(df)["p_value"] == pytest.approx(1.0239e-18, rel=1e-4, abs=0)

    def test_invalid_correctness_is_not_applicable(self):
        # A missing outcome used to count as correct (NaN is truthy).
        df = pd.DataFrame({"correct_a": [1.0, 0.0, 1.0], "correct_b": [1.0, np.nan, 0.0]})
        out = mcnemar_test(df)
        assert out["applicable"] is False
        assert "missing value" in out["reason"]

    def test_per_class_mcnemar_reads_rows_by_position(self, ab_df_classification):
        df = ab_df_classification
        ref = per_class_mcnemar(df)
        frames = [
            df.sort_values("confidence_a"),
            df.sample(frac=1.0, random_state=7),
            pd.concat([df.iloc[:250], df.iloc[250:]]),
            df.set_index("image_path"),
        ]
        for frame in frames:
            pd.testing.assert_frame_equal(per_class_mcnemar(frame), ref)
        subset = df[df["label"] != 3]
        pd.testing.assert_frame_equal(per_class_mcnemar(subset),
                                      per_class_mcnemar(subset.reset_index(drop=True)))

    def test_per_class_mcnemar_without_labels_is_empty_with_a_reason(self):
        df = pd.DataFrame({"correct_a": [1, 0], "correct_b": [1, 1]})
        out = per_class_mcnemar(df)
        assert out.empty and "class column 'label' is missing" in out.attrs["reason"]


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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_power_that_cannot_be_computed_is_not_adequate():
    # No usable confidence: the power is NaN, and min(1.0, nan) used to give 1.0.
    out = power_analysis(pd.DataFrame({"confidence_a": [np.nan, np.nan],
                                       "confidence_b": [np.nan, np.nan]}))
    assert np.isnan(out["achieved_power"])
    assert out["adequately_powered"] is False


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
        assert out["per_class_mcnemar"]["method"] == "benjamini-hochberg"
        assert len(out["per_class_mcnemar"]["rows"]) == 11

    def test_paired_counts_block(self):
        df = pd.DataFrame({
            "image_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "label":    [0, 0, 0, 1, 1, 1],
            "pred_a":   [0, 0, 1, 1, 0, 1],
            "pred_b":   [0, 1, 0, 0, 0, 2],
        })
        pc = compute_all_metrics(df)["paired_counts"]
        assert pc["valid"] is True and pc["id_col"] == "image_id"
        # i1 both, i2 only A, i3 only B, i4 only A, i5 neither, i6 only A
        assert [pc[c] for c in ("both_correct", "only_a_correct", "only_b_correct",
                                "both_wrong")] == [1, 3, 1, 1]
        assert [(r["class"], r["only_a_correct"], r["only_b_correct"])
                for r in pc["per_class"]] == [(0, 1, 1), (1, 2, 0)]

    def test_mcnemar_blocks_match_counts(self, ab_df_classification):
        out = compute_all_metrics(ab_df_classification)
        pc = out["paired_counts"]
        assert (out["mcnemar"]["b"], out["mcnemar"]["c"]) == (pc["only_a_correct"], pc["only_b_correct"])
        assert out["mcnemar"]["alternative"] == "two-sided"
        rows = out["per_class_mcnemar"]["rows"]
        assert [(r["class_id"], r["b"], r["c"]) for r in rows] == [
            (r["class"], r["only_a_correct"], r["only_b_correct"]) for r in pc["per_class"]]

    def test_invalid_slice_omits_mcnemar_blocks(self, ab_df_classification):
        doubled = pd.concat([ab_df_classification] * 2)
        out = compute_all_metrics(doubled)
        assert out["paired_counts"]["valid"] is False
        assert any("duplicated row(s)" in e for e in out["paired_counts"]["errors"])
        assert "mcnemar" not in out and "per_class_mcnemar" not in out
        assert "statistical" in out and "disagreement" in out   # diagnostics still run

    def test_documented_job_schema_runs(self, ab_df_classification):
        # The decision job's slice has no 'agrees' / 'confidence_delta': both are derived.
        df = ab_df_classification.drop(columns=["agrees", "confidence_delta"])
        out = compute_all_metrics(df)
        assert out["diagnostic_errors"] == []
        expected = float((df["pred_a"] != df["pred_b"]).mean())
        assert out["disagreement"]["rate"] == pytest.approx(expected)
        assert out["delta"]["mean"] == pytest.approx(
            float((df["confidence_a"] - df["confidence_b"]).abs().mean()))

    def test_stale_agrees_is_replaced(self, ab_df_classification):
        df = ab_df_classification.copy()
        df.loc[df.index[:7], "agrees"] = 1 - df.loc[df.index[:7], "agrees"]
        out = compute_all_metrics(df)
        assert out["diagnostic_errors"] == [{
            "block": "input",
            "error": "agrees disagrees with pred_a == pred_b on 7 row(s); derived values used"}]
        assert out["disagreement"]["rate"] == pytest.approx(float((df["pred_a"] != df["pred_b"]).mean()))

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_failed_diagnostic_is_recorded_not_raised(self, ab_df_classification):
        df = ab_df_classification.assign(confidence_a="high")
        out = compute_all_metrics(df)
        failed = {e["block"] for e in out["diagnostic_errors"]}
        assert {"statistical", "power"} <= failed
        assert "statistical" not in out and "power" not in out
        assert out["paired_counts"]["valid"] is True and "mcnemar" in out

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

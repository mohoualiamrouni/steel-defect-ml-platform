"""Unit tests for Bayesian A/B layer."""
from __future__ import annotations

import numpy as np
import pytest

from steeldefect_active_learning.ab_testing.bayesian import (
    bayesian_summary,
    beta_binomial_decision,
    normal_decision,
)


class TestBetaBinomial:
    def test_b_obviously_wins(self):
        out = beta_binomial_decision(
            successes_a=80, trials_a=100,
            successes_b=95, trials_b=100,
            n_samples=20_000, seed=0,
        )
        assert out["p_b_gt_a"] > 0.99
        assert out["decision"] == "ship_b"
        assert out["expected_loss_ship_b"] < out["expected_loss_keep_a"]

    def test_no_winner_under_h0(self):
        out = beta_binomial_decision(
            successes_a=80, trials_a=100,
            successes_b=80, trials_b=100,
            n_samples=20_000, seed=0,
        )
        # P(B > A) should hover around 0.5 (uniform prior, identical data).
        assert 0.4 < out["p_b_gt_a"] < 0.6
        assert out["decision"] == "keep_a"

    def test_credible_interval_brackets_zero_under_h0(self):
        out = beta_binomial_decision(
            successes_a=80, trials_a=100,
            successes_b=80, trials_b=100,
            n_samples=20_000, seed=0,
        )
        lo, hi = out["cred_interval_diff"]
        assert lo < 0 < hi


class TestNormalDecision:
    def test_b_obviously_wins(self, rng):
        a = rng.normal(0.5, 0.1, size=200)
        b = rng.normal(0.7, 0.1, size=200)
        out = normal_decision(a, b, n_samples=20_000, seed=0)
        assert out["p_b_gt_a"] > 0.99
        assert out["decision"] == "ship_b"

    def test_n_too_small_returns_not_applicable(self):
        out = normal_decision(np.array([0.5]), np.array([0.6]))
        assert out.get("applicable") is False


class TestBayesianSummary:
    def test_runs_on_full_df(self, ab_df_classification):
        out = bayesian_summary(ab_df_classification, n_samples=10_000)
        # All three sub-models should fire because the fixture has agrees /
        # confidence_a/b / pred_a/b/label.
        assert "agreement" in out
        assert "accuracy" in out
        assert "confidence" in out

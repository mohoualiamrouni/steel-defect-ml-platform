"""Unit tests for Bayesian A/B layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
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


def _paired_frame(n11: int, n10: int, n01: int, n00: int) -> pd.DataFrame:
    """One class, correctness given as columns, one row per image."""
    a = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
    b = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
    return pd.DataFrame({"image_id": [f"i{j}" for j in range(len(a))], "label": 0,
                         "correct_a": a, "correct_b": b})


class TestBayesianSummary:
    def test_runs_on_full_df(self, ab_df_classification):
        out = bayesian_summary(ab_df_classification, n_samples=10_000)
        # All three sub-models should fire because the fixture has agrees /
        # confidence_a/b / pred_a/b/label.
        assert "agreement" in out
        assert "accuracy" in out
        assert "confidence" in out

    def test_accuracy_block_is_paired_and_exact(self):
        acc = bayesian_summary(_paired_frame(32, 293, 325, 0), n_samples=10_000)["accuracy"]
        assert acc["model"] == "dirichlet_paired"
        assert acc["p_b_gt_a"] == pytest.approx(0.9008310969459695, abs=1e-12)
        assert acc["n_discordant"] == 618
        lo, hi = acc["cred_interval_diff"]
        assert lo < 32 / 650 < hi        # observed acc_B - acc_A = (325 - 293) / 650

    def test_accuracy_posterior_uses_the_pairing(self):
        # B fixes 20 of A's errors and breaks none (McNemar p = 2^-20): independent
        # Beta posteriors gave P(B > A) = 0.94; the paired posterior gives 1 - 2^-21.
        acc = bayesian_summary(_paired_frame(900, 0, 20, 80), n_samples=10_000)["accuracy"]
        assert acc["p_b_gt_a"] == pytest.approx(1 - 2 ** -21, abs=1e-12)

    def test_accuracy_block_absent_on_invalid_input(self):
        frame = _paired_frame(32, 293, 325, 0).drop(columns=["image_id"])
        assert "accuracy" not in bayesian_summary(frame, n_samples=1_000)

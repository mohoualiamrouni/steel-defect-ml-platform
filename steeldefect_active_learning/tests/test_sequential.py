"""Unit tests for sequential / always-valid testing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from steeldefect_active_learning.ab_testing.sequential import (
    MSPRTState,
    confidence_sequence,
    obrien_fleming_boundary,
    obrien_fleming_decide,
)


class TestMSPRT:
    def test_under_h0_pvalue_above_alpha_most_of_time(self, rng):
        # Type-I control: under H0 (μ_A = μ_B), the always-valid p stays above
        # alpha for most of a long run.
        state = MSPRTState(tau_sq=1.0, sigma_sq=0.025)
        n_total = 200
        peeks_below = 0
        for _ in range(n_total):
            x = rng.normal(0.7, 0.15)
            y = rng.normal(0.7, 0.15)
            state.update(x_a=x, x_b=y)
            if state.always_valid_p() < 0.05:
                peeks_below += 1
        # Under H0, < 30% of peeks should fire (more lenient than the
        # asymptotic guarantee since the prior is not perfectly tuned).
        assert peeks_below / n_total < 0.30

    def test_under_h1_eventually_rejects(self, rng):
        # When the effect is large, the always-valid p must drop below 0.05.
        state = MSPRTState(tau_sq=1.0, sigma_sq=0.025)
        for _ in range(500):
            state.update(x_a=rng.normal(0.5, 0.1), x_b=rng.normal(0.7, 0.1))
        assert state.always_valid_p() < 0.05

    def test_decide_records_history(self, rng):
        state = MSPRTState()
        state.update_batch(rng.normal(0.5, 0.1, 100), rng.normal(0.7, 0.1, 100))
        rec = state.decide(alpha=0.05)
        assert "decision" in rec
        assert len(state.history) == 1
        assert rec["n_a"] == 100
        assert rec["n_b"] == 100

    def test_confidence_sequence_widens_with_alpha(self, rng):
        state = MSPRTState()
        state.update_batch(rng.normal(0.5, 0.1, 200), rng.normal(0.6, 0.1, 200))
        lo_1, hi_1 = confidence_sequence(state, alpha=0.05)
        lo_2, hi_2 = confidence_sequence(state, alpha=0.01)
        # 99% sequence should be wider than 95%.
        assert (hi_2 - lo_2) >= (hi_1 - lo_1) - 1e-9


class TestOBrienFleming:
    def test_boundary_decreases_with_look(self):
        # Earlier looks should have stricter (larger) critical z-values.
        z1 = obrien_fleming_boundary(1, 4, alpha=0.05)
        z2 = obrien_fleming_boundary(2, 4, alpha=0.05)
        z4 = obrien_fleming_boundary(4, 4, alpha=0.05)
        assert z1 > z2 > z4

    def test_invalid_look_raises(self):
        with pytest.raises(ValueError):
            obrien_fleming_boundary(0, 4)
        with pytest.raises(ValueError):
            obrien_fleming_boundary(5, 4)

    def test_decide_continue_or_stop(self):
        small_z = obrien_fleming_decide(z_observed=1.5, look=2, n_looks=4)
        big_z   = obrien_fleming_decide(z_observed=10.0, look=4, n_looks=4)
        assert small_z["decision"] == "continue"
        assert big_z["decision"] == "stop_reject_h0"

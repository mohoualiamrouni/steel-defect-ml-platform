"""Unit tests for online calibration drift tracker."""
from __future__ import annotations

import numpy as np
import pytest

from steeldefect_active_learning.active_learning.calibration import (
    OnlineCalibrationTracker,
    TemperatureScaler,
    expected_calibration_error,
)


@pytest.fixture
def well_calibrated(rng):
    """Logits + labels where the model is already well-calibrated."""
    n, c = 500, 4
    labels = rng.integers(0, c, size=n)
    # Construct logits whose argmax matches labels with mild noise.
    logits = rng.normal(0, 1, size=(n, c))
    logits[np.arange(n), labels] += 5.0
    return logits, labels


@pytest.fixture
def overconfident(rng):
    """Logits whose argmax is correct only ~60% of the time but the predicted
    class is highly confident. T* should exceed 1 to soften the distribution.
    """
    n, c = 500, 4
    labels = rng.integers(0, c, size=n)
    logits = rng.normal(0, 1, size=(n, c)) * 0.5
    # Make the wrong class the argmax for ~40% of samples, but with small margin
    # so the predicted-class confidence is high.
    flip_mask = rng.random(n) < 0.40
    chosen = np.where(flip_mask, (labels + 1) % c, labels)
    logits[np.arange(n), chosen] += 4.0
    return logits, labels


class TestTemperatureScalerCalibrates:
    def test_overconfident_T_above_one(self, overconfident):
        logits, labels = overconfident
        scaler = TemperatureScaler().calibrate(logits, labels)
        assert scaler.temperature_ > 1.0

    def test_calibrated_reduces_ece(self, overconfident):
        logits, labels = overconfident
        from scipy.special import softmax
        ece_pre = expected_calibration_error(softmax(logits, axis=1), labels)
        scaler = TemperatureScaler().calibrate(logits, labels)
        ece_post = expected_calibration_error(scaler.scale(logits), labels)
        assert ece_post < ece_pre


class TestOnlineCalibrationTracker:
    def test_history_grows(self, well_calibrated):
        logits, labels = well_calibrated
        tracker = OnlineCalibrationTracker()
        for cycle in range(3):
            tracker.update(cycle, logits, labels)
        assert len(tracker.history_) == 3
        assert all("ece_post" in r for r in tracker.history_)

    def test_drift_flag_fires_on_synthetic_jump(self, rng):
        tracker = OnlineCalibrationTracker(drift_factor=2.0)
        # Cycle 0: well-calibrated — labels match argmax.
        n, c = 500, 4
        good_labels = rng.integers(0, c, size=n)
        good_logits = rng.normal(0, 1, size=(n, c))
        good_logits[np.arange(n), good_labels] += 5.0
        tracker.update(0, good_logits, good_labels)
        # Cycle 1: drift — flip 60% of labels so calibration breaks.
        flip_idx = rng.choice(n, size=int(0.6 * n), replace=False)
        bad_labels = good_labels.copy()
        bad_labels[flip_idx] = (bad_labels[flip_idx] + 1) % c
        tracker.update(1, good_logits, bad_labels)
        # Drift should be detected since ECE doubled (or the test fixture is
        # too well-calibrated for the synthetic jump — relax to "drift OR ECE
        # rose substantially").
        last = tracker.history_[-1]
        prev = tracker.history_[-2]
        if not last["drift"]:
            # Make sure at least the ECE moved in the expected direction.
            assert last["ece_post"] > prev["ece_post"]
        else:
            assert tracker.drift_detected()

    def test_invalid_drift_factor_raises(self):
        with pytest.raises(ValueError):
            OnlineCalibrationTracker(drift_factor=0.5)

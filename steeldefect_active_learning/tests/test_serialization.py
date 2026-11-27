"""Strict JSON for the A/B outputs: no NaN or Infinity token, numpy types as Python values."""
import numpy as np

from steeldefect_active_learning.ab_testing.serialization import dumps, strict


def test_strict_turns_numpy_and_non_finite_values_into_json_values():
    obj = {1: np.float64("nan"), "a": [np.int64(3), np.inf, (1.5, -np.inf)],
           "b": np.array([1.0, np.nan]), "c": np.bool_(True)}
    assert strict(obj) == {"1": None, "a": [3, None, [1.5, None]], "b": [1.0, None], "c": True}
    assert dumps(obj) == '{"1": null, "a": [3, null, [1.5, null]], "b": [1.0, null], "c": true}'

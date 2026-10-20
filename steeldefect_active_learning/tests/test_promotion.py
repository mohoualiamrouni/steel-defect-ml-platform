"""Regression tests for the per-class guard of the A/B promotion gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from steeldefect_active_learning.ab_testing.metrics import per_class_mcnemar
from steeldefect_active_learning.ab_testing.promotion import decide_promotion


def _two_class_ab(n_per_class: int = 200, n_flip: int = 60) -> pd.DataFrame:
    """Paired correctness for two classes.

    Class 0: B is wrong on ``n_flip`` images that A gets right (B regresses).
    Class 1: B is right on ``n_flip`` images that A gets wrong (B improves).
    """
    a0 = np.ones(n_per_class, dtype=int)
    b0 = a0.copy()
    b0[:n_flip] = 0
    a1 = np.ones(n_per_class, dtype=int)
    a1[:n_flip] = 0
    b1 = np.ones(n_per_class, dtype=int)
    return pd.DataFrame({
        "label": [0] * n_per_class + [1] * n_per_class,
        "correct_a": np.concatenate([a0, a1]),
        "correct_b": np.concatenate([b0, b1]),
    })


def _cascade(df: pd.DataFrame) -> dict:
    """Pack per-class rows exactly as compute_all_metrics does."""
    pcm = per_class_mcnemar(df, correction="benjamini-hochberg", alpha=0.05)
    return {"per_class_mcnemar": {"method": "benjamini-hochberg", "alpha": 0.05,
                                  "rows": pcm.to_dict(orient="records")}}


def _criteria(verdict: dict) -> dict:
    return {r["criterion"]: r for r in verdict["reasons"]}


def test_significant_class_regression_blocks_promotion():
    verdict = decide_promotion(_cascade(_two_class_ab()))
    assert verdict["per_class_regress"] == ["0"]
    assert _criteria(verdict)["no_class_regression"]["passed"] is False


def test_significant_class_improvement_is_reported():
    verdict = decide_promotion(_cascade(_two_class_ab()))
    assert verdict["per_class_promote"] == ["1"]


def test_non_significant_differences_do_not_trigger_the_guard():
    verdict = decide_promotion(_cascade(_two_class_ab(n_flip=2)))
    assert verdict["per_class_regress"] == []
    assert verdict["per_class_promote"] == []
    assert _criteria(verdict)["no_class_regression"]["passed"] is True

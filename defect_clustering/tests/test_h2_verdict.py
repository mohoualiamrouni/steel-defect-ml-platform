"""The H2 decision rule, applied to run_ablation tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defect_clustering.analysis.group_analysis import H2_MIN_GAIN, h2_verdict, run_ablation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _table(cnn_nmi, fusion_nmi, fusion_ci_low, fusion_ci_high, delta=None):
    """Minimal run_ablation table: the two rows the rule reads.

    ``delta`` defaults to the difference of the two NMIs. Pass the stored value
    when there is one: run_ablation rounds the unrounded difference, which can
    differ by 1e-4 from the difference of the rounded NMIs.
    """
    if delta is None:
        delta = round(fusion_nmi - cnn_nmi, 4)
    return pd.DataFrame([
        {"feature_set": "cnn_only", "nmi": cnn_nmi, "nmi_ci_low": cnn_nmi - 0.01,
         "nmi_ci_high": cnn_nmi + 0.01, "delta_nmi_vs_cnn": 0.0},
        {"feature_set": "fusion_block_pca", "nmi": fusion_nmi, "nmi_ci_low": fusion_ci_low,
         "nmi_ci_high": fusion_ci_high, "delta_nmi_vs_cnn": delta},
    ])


def test_passes_when_the_ci_clears_cnn_and_the_gain_is_large():
    # Numbers of the calibrated-synthetic ablation run.
    v = h2_verdict(_table(0.6947, 0.7985, 0.7959, 0.8034))
    assert v["ci_low_above_cnn"] and v["gain_ok"] and v["passed"]


def test_fails_on_the_neu_det_numbers():
    v = h2_verdict(_table(0.8765, 0.8836, 0.8714, 0.8994))
    assert not v["ci_low_above_cnn"] and not v["gain_ok"] and not v["passed"]


def test_needs_the_minimum_gain_even_when_the_ci_clears_cnn():
    v = h2_verdict(_table(0.70, 0.72, 0.715, 0.725))
    assert v["ci_low_above_cnn"] and not v["gain_ok"] and not v["passed"]


def test_gain_threshold_is_inclusive():
    v = h2_verdict(_table(0.70, 0.73, 0.72, 0.74))
    assert v["delta_nmi"] == pytest.approx(H2_MIN_GAIN)
    assert v["passed"]


def test_the_ci_alone_can_fail_the_rule():
    v = h2_verdict(_table(0.70, 0.75, 0.69, 0.80))
    assert not v["ci_low_above_cnn"] and v["gain_ok"] and not v["passed"]


def test_a_ci_bound_equal_to_cnn_fails():
    v = h2_verdict(_table(0.70, 0.75, 0.70, 0.80))
    assert not v["ci_low_above_cnn"] and not v["passed"]


def test_a_missing_ci_fails_the_rule():
    v = h2_verdict(_table(0.70, 0.80, float("nan"), float("nan")))
    assert not v["passed"]


def test_missing_rows_raise():
    with pytest.raises(ValueError, match="fusion_block_pca"):
        h2_verdict(_table(0.70, 0.80, 0.79, 0.81).iloc[:1])


def test_tracked_neu_summary_matches_the_rule():
    """The verdict stored in the tracked NEU summary follows from its own numbers."""
    p2 = json.loads((REPO_ROOT / "outputs_neu_p1p2_summary.json").read_text(encoding="utf-8"))["P2_H2"]
    lo, hi = p2["fusion_nmi_ci"]
    # The stored gain is rounded from the unrounded difference: within 1e-4 of the rounded NMIs.
    assert abs(p2["delta_nmi"] - (p2["fusion_nmi"] - p2["cnn_only_nmi"])) <= 1.5e-4
    v = h2_verdict(_table(p2["cnn_only_nmi"], p2["fusion_nmi"], lo, hi, delta=p2["delta_nmi"]))
    assert v["passed"] == p2["H2_pass"]


def test_applies_to_a_real_run_ablation_table():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 40)
    cnn = rng.normal(size=(120, 12)) + labels[:, None] * 2.0
    seg = rng.normal(size=(120, 6)) + labels[:, None] * 1.5
    table = run_ablation(cnn_features=cnn, seg_features=seg, labels_expert=labels,
                         k=3, random_state=0, n_bootstrap=20)
    v = h2_verdict(table)
    assert {"cnn_nmi", "fusion_nmi", "fusion_nmi_ci", "delta_nmi", "passed"} <= set(v)
    assert isinstance(v["passed"], bool)
    # The percentile bootstrap need not contain the point estimate (resampling
    # pairs with replacement biases NMI), so only check the interval itself.
    assert v["fusion_nmi_ci"][0] <= v["fusion_nmi_ci"][1]

"""The A/B comparison plots, the A/B PDF report and the reliability diagram render."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from steeldefect_active_learning.ab_testing import metrics
from steeldefect_active_learning.ab_testing.report import generate_ab_report
from steeldefect_active_learning.active_learning.calibration import reliability_diagram

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"


@pytest.fixture(scope="module")
def demo():
    """The tracked synthetic A/B predictions (1,100 rows, 11 classes)."""
    return pd.read_csv(DEMO)


@pytest.mark.parametrize("plot, expected", [
    (metrics.plot_comparison,
     {"disagreement_rate", "confidence_delta_dist", "per_class_agreement"}),
    (metrics.plot_advanced_comparison,
     {"calibration_curve", "entropy_distribution", "confidence_bands", "per_class_accuracy_delta"}),
    (metrics.plot_advanced_comparison_v2,
     {"kde_heatmap", "violin_per_class", "mcnemar_test", "disagreement_decomposition"}),
], ids=["plot_comparison", "plot_advanced_comparison", "plot_advanced_comparison_v2"])
def test_ab_plots_render(plot, expected, demo, tmp_path):
    plot(demo, output_dir=str(tmp_path))
    pngs = {p.stem.removeprefix("ab_"): p for p in tmp_path.glob("*.png")}
    assert set(pngs) == expected
    assert all(p.stat().st_size > 0 for p in pngs.values())


def test_ab_report_renders(demo, tmp_path):
    out = tmp_path / "ab_report.pdf"
    generate_ab_report(demo, output_path=str(out))
    # 11 pages on the demo: classification (no IoU page) and no BALD columns.
    assert len(re.findall(rb"/Type /Page\b", out.read_bytes())) == 11


@pytest.mark.parametrize("n_classes", [2, 5])
def test_reliability_diagram_renders(n_classes, tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, n_classes))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, n_classes, size=200)
    out = tmp_path / "reliability.png"
    reliability_diagram(probs, labels, out)
    assert out.stat().st_size > 0

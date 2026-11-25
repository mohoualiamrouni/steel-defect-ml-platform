"""The A/B comparison plots, the A/B PDF report and the reliability diagram render."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import matplotlib.text

from steeldefect_active_learning.ab_testing import metrics, report
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


def test_cover_has_no_agreement_verdict(demo, tmp_path, monkeypatch):
    # B repeats A except on every 30th image A gets right, where B is wrong: strictly
    # worse, with 97.3 % agreement. The old cover printed "GO" from agreement alone.
    worse = demo.assign(pred_b=demo["pred_a"])
    right = np.flatnonzero((demo["pred_a"] == demo["label"]).to_numpy())[::30]
    worse.loc[worse.index[right], "pred_b"] = (worse["label"].iloc[right] + 1) % 11
    worse["agrees"] = (worse["pred_a"] == worse["pred_b"]).astype(int)   # the report reads it
    pages = []

    class RecordingPdf(report.PdfPages):
        def savefig(self, figure=None, **kwargs):
            pages.append([t.get_text() for t in figure.findobj(matplotlib.text.Text)])
            super().savefig(figure, **kwargs)

    monkeypatch.setattr(report, "PdfPages", RecordingPdf)
    generate_ab_report(worse, output_path=str(tmp_path / "ab_report.pdf"))
    assert len(pages) == 11
    cover = pages[0]
    assert "Promotion verdict" in cover
    assert not [t for t in cover if t.startswith(("GO", "LIKELY GO", "STOP", "REVIEW"))]


@pytest.mark.parametrize("n_classes", [2, 5])
def test_reliability_diagram_renders(n_classes, tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, n_classes))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, n_classes, size=200)
    out = tmp_path / "reliability.png"
    reliability_diagram(probs, labels, out)
    assert out.stat().st_size > 0


def test_ab_plots_leave_the_global_style_alone(demo, tmp_path, monkeypatch):
    # seaborn's global setters also remap the single-letter colour codes, which
    # rcParams do not hold, so the plots must not call them at all.
    setters = []
    for name in ("set_theme", "set", "set_style", "set_context", "set_palette", "set_color_codes"):
        if hasattr(sns, name):
            monkeypatch.setattr(sns, name, lambda *args, _name=name, **kwargs: setters.append(_name))
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        before = matplotlib.rcParams.copy()
        metrics.plot_comparison(demo, output_dir=str(tmp_path))
        metrics.plot_advanced_comparison(demo, output_dir=str(tmp_path))
        changed = sorted(k for k, v in before.items() if matplotlib.rcParams[k] != v)
    assert changed == []
    assert setters == []

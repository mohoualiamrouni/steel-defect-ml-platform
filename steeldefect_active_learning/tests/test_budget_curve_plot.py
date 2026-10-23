"""Confidence band and rendering of the active-learning budget curve."""

import numpy as np
import pandas as pd
import pytest

from steeldefect_active_learning.active_learning.budget_curve import _ci95_halfwidth, plot_budget_curve


def test_halfwidth_uses_the_student_quantile():
    # t(0.975, 4) = 2.7764 for 5 seeds, not the normal 1.96.
    assert _ci95_halfwidth([1.0], [5])[0] == pytest.approx(2.7764 / np.sqrt(5), rel=1e-4)


def test_halfwidth_is_zero_without_replicates():
    assert _ci95_halfwidth([np.nan, 0.3], [1, 1]).tolist() == [0.0, 0.0]


def test_plotted_band_is_the_student_t_interval(tmp_path, monkeypatch):
    import matplotlib.axes
    from scipy import stats

    bands = []
    original = matplotlib.axes.Axes.fill_between

    def spy(self, x, y1, y2=0, **kwargs):
        bands.append((np.asarray(y1, float), np.asarray(y2, float)))
        return original(self, x, y1, y2, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", spy)
    f1 = {20: (0.50, 0.55, 0.60, 0.52, 0.58), 50: (0.70, 0.72, 0.69, 0.75, 0.71)}
    rows = [{"strategy": "random", "budget": b, "seed": k, "macro_f1": v}
            for b, vals in f1.items() for k, v in enumerate(vals)]
    plot_budget_curve(pd.DataFrame(rows), str(tmp_path / "curve.png"))
    (lo, hi), = bands
    for i, vals in enumerate(f1.values()):
        v = np.asarray(vals)
        hw = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        assert lo[i] == pytest.approx(v.mean() - hw)
        assert hi[i] == pytest.approx(v.mean() + hw)


def test_plot_writes_a_png(tmp_path):
    rows = [{"strategy": s, "budget": b, "seed": k, "macro_f1": 0.5 + 0.001 * b + 0.01 * k}
            for s in ("random", "typiclust") for b in (20, 50) for k in range(3)]
    savings = {"reference_strategy": "typiclust", "ref_budget": 50, "baseline": "random",
               "baseline_budget_to_match": 50, "savings_ratio": 1.0}
    out = tmp_path / "curve.png"
    plot_budget_curve(pd.DataFrame(rows), str(out), savings=savings)
    assert out.exists() and out.stat().st_size > 0

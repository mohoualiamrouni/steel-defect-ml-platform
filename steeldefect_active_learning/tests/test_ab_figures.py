"""The A/B comparison figures make no promotion call; the McNemar one shows the cascade's test."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.text
import pandas as pd
import pytest

from steeldefect_active_learning.ab_testing import metrics

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"

# What the figures used to print: verdicts, tags and cut-offs that decide nothing.
CALLS = re.compile(r"significant\]|Verdict|differ significantly|No significant difference|"
                   r"DEGRADED|IMPROVED|B improved|B degraded|% threshold")
PLOTS = [metrics.plot_comparison, metrics.plot_advanced_comparison,
         metrics.plot_advanced_comparison_v2]


@pytest.fixture(scope="module")
def demo():
    """The tracked synthetic A/B predictions (1,100 rows, 11 classes)."""
    return pd.read_csv(DEMO)


def _figure_texts(monkeypatch, plot, df, tmp_path):
    """The texts of every figure a plot function saves, by file stem."""
    texts = {}
    save = matplotlib.figure.Figure.savefig

    def recording(self, fname, *args, **kwargs):
        texts[Path(str(fname)).stem] = [t.get_text() for t in self.findobj(matplotlib.text.Text)]
        return save(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording)
    plot(df, output_dir=str(tmp_path))
    return texts


@pytest.mark.parametrize("plot", PLOTS, ids=[p.__name__ for p in PLOTS])
def test_figures_make_no_promotion_call(plot, demo, tmp_path, monkeypatch):
    texts = _figure_texts(monkeypatch, plot, demo, tmp_path)
    assert texts
    calls = {stem: [t for t in found if CALLS.search(t)] for stem, found in texts.items()}
    assert {stem: found for stem, found in calls.items() if found} == {}


def test_the_mcnemar_figure_shows_the_cascade_test(demo, tmp_path, monkeypatch):
    texts = _figure_texts(monkeypatch, metrics.plot_advanced_comparison_v2, demo, tmp_path)
    mc = metrics.compute_all_metrics(demo)["mcnemar"]       # b = n10 (A only), c = n01 (B only)
    figure = texts["ab_mcnemar_test"]
    assert f"{mc['p_value']:.6f}  (two-sided)" in figure
    assert f"{mc['b'] + mc['c']}  (n₁₀={mc['b']} only A, n₀₁={mc['c']} only B)" in figure


def test_the_mcnemar_figure_with_no_discordant_pairs(demo, tmp_path, monkeypatch):
    texts = _figure_texts(monkeypatch, metrics.plot_advanced_comparison_v2,
                          demo.assign(pred_b=demo["pred_a"]), tmp_path)
    figure = texts["ab_mcnemar_test"]
    assert "no test: no discordant pairs" in figure and "N/A (exact test)" not in figure


def test_the_mcnemar_figure_shows_no_test_on_an_invalid_slice(demo, tmp_path, monkeypatch):
    # Every image three times: pseudo-replication, which the cascade refuses.
    tripled = pd.concat([demo] * 3, ignore_index=True)
    texts = _figure_texts(monkeypatch, metrics.plot_advanced_comparison_v2, tripled, tmp_path)
    figure = " ".join(texts["ab_mcnemar_test"])
    assert "No McNemar test" in figure and "p-value" not in figure

"""The A/B comparison plots, the A/B PDF report and the reliability diagram render."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import matplotlib.colors
import matplotlib.text

from steeldefect_active_learning.ab_testing import metrics, paired, report
from steeldefect_active_learning.ab_testing.promotion import decide_promotion
from steeldefect_active_learning.ab_testing.report import generate_ab_report
from steeldefect_active_learning.active_learning.calibration import reliability_diagram
from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config

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


def _record_pages(monkeypatch):
    """The texts of every page the report saves, in order."""
    pages = []

    class RecordingPdf(report.PdfPages):
        def savefig(self, figure=None, **kwargs):
            pages.append([t.get_text() for t in figure.findobj(matplotlib.text.Text)])
            super().savefig(figure, **kwargs)

    monkeypatch.setattr(report, "PdfPages", RecordingPdf)
    return pages


def test_cover_has_no_agreement_verdict(demo, tmp_path, monkeypatch):
    # B repeats A except on every 30th image A gets right, where B is wrong: strictly
    # worse, with 97.3 % agreement. The old cover printed "GO" from agreement alone.
    worse = demo.assign(pred_b=demo["pred_a"])
    right = np.flatnonzero((demo["pred_a"] == demo["label"]).to_numpy())[::30]
    worse.loc[worse.index[right], "pred_b"] = (worse["label"].iloc[right] + 1) % 11
    pages = _record_pages(monkeypatch)
    generate_ab_report(worse, output_path=str(tmp_path / "ab_report.pdf"))
    assert len(pages) == 11
    cover = pages[0]
    assert "Promotion verdict" in cover
    assert not [t for t in cover if t.startswith(("GO", "LIKELY GO", "STOP", "REVIEW"))]


def _verdict(df):
    """The gate's verdict on a frame, with the shipped thresholds."""
    return decide_promotion({"paired_counts": paired.paired_counts(df)},
                            **load_config(SHIPPED_CONFIG_PATH).ab_testing.gate_kwargs())


def test_report_shows_the_gate_verdict(demo, tmp_path, monkeypatch):
    verdict = _verdict(demo)
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"), verdict=verdict)
    assert len(pages) == 12                       # the gate page follows the cover
    cover, gate, summary = pages[0], pages[1], pages[-1]
    assert "keep_A   (inconclusive)" in cover
    assert "failing: sample_size, superiority, no_class_regression" in cover
    # The cover shows the difference the gate judged: B is ahead on the slice (+2.1
    # points) and behind at the deployment prior, which the gate reads.
    est = verdict["estimates"]
    assert est["estimand"] == "deployment_prior" and round(est["delta"], 3) == 0.021
    assert est["estimand_delta"] < 0
    assert "Δ (B − A), deployment prior" in cover
    assert f"{est['estimand_delta']:+.1%}  (lower {est['estimand_lower']:+.1%})" in cover
    assert "Promotion Gate" in gate
    assert [t for t in gate if t.startswith(("PASS  ", "FAIL  "))] == [
        f"{'PASS' if r['passed'] else 'FAIL'}  {r['criterion']}" for r in verdict["reasons"]]
    assert "keep_A" in summary and "inconclusive" in summary


def test_cover_without_a_verdict_says_so(demo, tmp_path, monkeypatch):
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"))
    assert "Not computed: run ab_testing.promotion.decide_promotion and pass verdict=" in pages[0]
    assert "Δ (B − A), slice" in pages[0] and "+2.1%" in pages[0]
    assert "not computed" in pages[-1]


def test_report_reads_derived_columns_and_the_cascade_mcnemar(demo, tmp_path, monkeypatch):
    # A supplied 'agrees' that says A and B always agree is replaced, and reported.
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo.assign(agrees=1), output_path=str(tmp_path / "ab_report.pdf"))
    cover, statistics, mcnemar = pages[0], pages[1], pages[2]
    per_class_delta, summary = pages[5], pages[-1]
    disagree = demo["pred_a"] != demo["pred_b"]
    assert f"{disagree.mean():.1%}" in cover
    assert any(f"agrees disagrees with pred_a == pred_b on {int(disagree.sum())} row(s)" in t
               for t in cover)
    # The McNemar page shows the cascade's own block (b = n10, c = n01).
    mc = metrics.compute_all_metrics(demo)["mcnemar"]
    assert f"p-value:  {report._fmt_pvalue(mc['p_value'])}" in mcnemar
    assert f"Discordant:  {mc['b'] + mc['c']}  (n10 = {mc['b']}, n01 = {mc['c']})" in mcnemar
    assert "Two-sided p >= 0.05: no difference detected" in mcnemar           # p = 0.2137
    assert "A diagnostic: the promotion gate decides." in mcnemar
    # No verdict on the diagnostic pages.
    assert not any("SIGNIFICANT" in t for t in statistics + mcnemar)
    assert not any("REVIEW BEFORE DEPLOYMENT" in t or "ALL CLASSES STABLE" in t
                   for t in per_class_delta)
    assert not any("improved" in t.lower() or "degraded" in t.lower() for t in per_class_delta)
    assert not any("[+]" in t or "[-]" in t for t in summary)
    # Mann-Whitney ran on the demo's confidences: no Welch interval, on any page.
    assert "n/a (Welch only)" in summary and "(nan, nan)" not in summary


def test_an_invalid_slice_gets_no_mcnemar_test_and_no_accuracies(demo, tmp_path, monkeypatch):
    # Every image three times: pseudo-replication, which the cascade and the gate refuse.
    tripled = pd.concat([demo] * 3, ignore_index=True)
    assert "mcnemar" not in metrics.compute_all_metrics(tripled)
    pages = _record_pages(monkeypatch)
    generate_ab_report(tripled, output_path=str(tmp_path / "ab_report.pdf"),
                       verdict=_verdict(tripled))
    cover, mcnemar = pages[0], pages[3]
    assert "keep_A   (invalid_input)" in cover
    assert "failing: valid_input   (the other 4 not evaluated)" in cover
    assert cover.count("n/a") == 3                     # accuracy A, accuracy B, the difference
    assert any("Paired counts invalid:" in t for t in cover)
    assert any(t.startswith("No McNemar test: the slice is not a valid paired evaluation")
               for t in mcnemar)
    assert not any(t.startswith("p-value") for t in mcnemar)


def test_the_report_reads_the_cascade_counts(demo, tmp_path, monkeypatch):
    # Two images per coil: the cascade, told the cluster column, refuses the slice. Given
    # its counts, the report agrees; with its default columns it would show numbers.
    by_coil = demo.assign(coil=np.arange(len(demo)) // 2)
    cascade = metrics.compute_all_metrics(by_coil, cluster_col="coil")
    assert not cascade["paired_counts"]["valid"] and "mcnemar" not in cascade
    pages = _record_pages(monkeypatch)
    generate_ab_report(by_coil, output_path=str(tmp_path / "ab_report.pdf"),
                       verdict=decide_promotion(cascade, **load_config(
                           SHIPPED_CONFIG_PATH).ab_testing.gate_kwargs()),
                       counts=cascade["paired_counts"])
    assert pages[0].count("n/a") == 3 and "keep_A   (invalid_input)" in pages[0]
    assert any(t.startswith("No McNemar test") for t in pages[3])


def test_no_discordant_pairs_is_no_test(demo, tmp_path, monkeypatch):
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo.assign(pred_b=demo["pred_a"]),
                       output_path=str(tmp_path / "ab_report.pdf"))
    mcnemar = pages[2]
    assert "No test: no discordant pairs" in mcnemar
    assert "No discordant pairs: no test" in mcnemar                 # the banner
    assert "Exact binomial used" not in mcnemar


def test_an_inconsistent_counts_block_is_refused(demo, tmp_path, monkeypatch):
    # A block built by hand or read back is checked as the gate checks it: here B gets
    # 1,000 more right answers than the 1,100 images allow.
    bad = dict(paired.paired_counts(demo))
    bad["only_b_correct"] += 1000
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"), counts=bad)
    assert pages[0].count("n/a") == 3
    assert any("cells sum to 2100, n = 1100" in t for t in pages[2])


def test_per_class_labels_take_one_colour(demo, tmp_path, monkeypatch):
    # The labels beside the bars were red below -5 points: a call on a point estimate.
    colours = []

    class RecordingPdf(report.PdfPages):
        def savefig(self, figure=None, **kwargs):
            texts = figure.findobj(matplotlib.text.Text)
            if any(t.get_text() == "Per-class Accuracy Delta" for t in texts):
                colours.extend(matplotlib.colors.to_hex(t.get_color()) for t in texts
                               if re.match(r"A:\d+%", t.get_text()))
            super().savefig(figure, **kwargs)

    monkeypatch.setattr(report, "PdfPages", RecordingPdf)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"))
    assert len(colours) == 11 and set(colours) <= {"#ffffff", report._DARK.lower()}


def test_an_estimand_the_gate_cannot_estimate_is_not_replaced(demo, tmp_path, monkeypatch):
    # Without class 5, which has weight in the deployment prior, the gate's estimand is
    # undefined: the cover says so instead of showing the slice's difference in its place.
    without = demo[demo["label"] != 5]
    verdict = _verdict(without)
    assert verdict["estimates"]["estimand_delta"] is None
    pages = _record_pages(monkeypatch)
    generate_ab_report(without, output_path=str(tmp_path / "ab_report.pdf"), verdict=verdict)
    assert "Δ (B − A), deployment prior" in pages[0] and "n/a (not estimable)" in pages[0]


def test_the_gate_page_marks_a_cut_detail(demo, tmp_path, monkeypatch):
    verdict = _verdict(demo)
    verdict["reasons"][3] = {**verdict["reasons"][3],
                             "detail": " ".join(f"w{i}" for i in range(400))}
    pages = _record_pages(monkeypatch)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"), verdict=verdict)
    shown = [t for t in pages[1] if re.match(r"w\d+ ", t)]
    assert len(shown) == 4 and shown[-1].endswith(" [...]")


def _texts_outside_their_box(figure):
    """Texts drawn on a coloured box (an axes without ticks) that spill out of it."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    spilled = []
    for ax in figure.axes:
        if not ax.axison or len(ax.get_xticks()) or len(ax.get_yticks()):
            continue                                  # a plot, or a panel drawn without a box
        box = ax.get_window_extent(renderer)
        for text in ax.texts:
            extent = text.get_window_extent(renderer)
            if (extent.x0 < box.x0 or extent.x1 > box.x1
                    or extent.y0 < box.y0 or extent.y1 > box.y1):
                spilled.append(text.get_text())
    return spilled


@pytest.mark.parametrize("with_verdict", [True, False], ids=["verdict", "no verdict"])
def test_texts_fit_their_boxes(demo, tmp_path, monkeypatch, with_verdict):
    # The cover's tiles and verdict box, and the McNemar banner, whose sentence once ran
    # past both ends of its box.
    verdict = _verdict(demo) if with_verdict else None
    spilled = []

    class CheckingPdf(report.PdfPages):
        def savefig(self, figure=None, **kwargs):
            spilled.extend(_texts_outside_their_box(figure))
            super().savefig(figure, **kwargs)

    monkeypatch.setattr(report, "PdfPages", CheckingPdf)
    generate_ab_report(demo, output_path=str(tmp_path / "ab_report.pdf"), verdict=verdict)
    assert spilled == []


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

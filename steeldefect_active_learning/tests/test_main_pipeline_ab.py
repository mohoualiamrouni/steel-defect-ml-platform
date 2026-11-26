"""The A/B block of the CLI pipeline runs the promotion gate and shows one verdict.

It runs on the tracked demo predictions, as the CLI does when no checkpoint is configured,
with the shipped thresholds and every output written to a temporary directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.text
import pandas as pd
import pytest

from steeldefect_active_learning.ab_testing import paired, report
from steeldefect_active_learning.ab_testing.promotion import decide_promotion
from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config
from steeldefect_active_learning.pipeline import main_pipeline

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"


def _strict_json(path: str):
    """Parse a file as strict JSON: NaN and Infinity tokens are an error."""
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse)


def _config(out: Path, predictions: Path):
    """The shipped config, writing under ``out`` and reading ``predictions`` as the demo."""
    cfg = load_config(SHIPPED_CONFIG_PATH)
    cfg.output.dir = str(out)
    cfg.output.plots_dir = str(out / "plots")
    cfg.ab_testing.bootstrap_n = 0          # the BCa jackknife is quadratic in n
    cfg.ab_testing.demo_predictions_path = str(predictions)
    return cfg


def _record_pages(monkeypatch):
    """The texts of every page the report saves, in order."""
    pages = []

    class RecordingPdf(report.PdfPages):
        def savefig(self, figure=None, **kwargs):
            pages.append([t.get_text() for t in figure.findobj(matplotlib.text.Text)])
            super().savefig(figure, **kwargs)

    monkeypatch.setattr(report, "PdfPages", RecordingPdf)
    return pages


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return _config(tmp_path_factory.mktemp("ab_cli"), DEMO)


def test_the_cli_runs_the_gate_and_writes_its_verdict(run, capsys, monkeypatch):
    cfg = run
    pages = _record_pages(monkeypatch)
    result = main_pipeline._run_ab_testing(cfg, model_a=None, loader=None, device="cpu")
    printed = capsys.readouterr().out

    expected = decide_promotion({"paired_counts": paired.paired_counts(pd.read_csv(DEMO))},
                                **cfg.ab_testing.gate_kwargs())
    decision = _strict_json(result["paths"]["decision"])
    assert decision == json.loads(json.dumps(expected))
    assert (decision["decision"], decision["conclusion"]) == ("keep_A", "inconclusive")

    # One verdict, every criterion with its detail, then the diagnostics, labelled as such.
    assert "Promotion decision           keep_A  (inconclusive)" in printed
    for reason in expected["reasons"]:
        assert f"{'PASS' if reason['passed'] else 'FAIL'} {reason['criterion']}" in printed
        assert reason["detail"] in printed
    assert printed.index("Promotion decision") < printed.index("Diagnostics (they do not decide promotion)")
    assert "SRM check                    not applicable (paired evaluation: no 'arm' column)" in printed
    assert "Mann-Whitney U = 691179" in printed and "Welch t" not in printed
    assert not re.search(r"SIGNIFICANT|\[OK\] adequate|\[!\] HIGH|winner", printed)

    # ab_metrics.json is strict JSON, and holds the block the gate read.
    metrics = _strict_json(result["paths"]["metrics"])
    assert metrics["paired_counts"] == json.loads(json.dumps(paired.paired_counts(pd.read_csv(DEMO))))
    assert metrics["srm"]["applicable"] is False and metrics["srm"]["p_value"] is None

    # The report shows the same verdict, on its cover and then on the gate's page.
    assert len(pages) == 12
    assert "keep_A   (inconclusive)" in pages[0] and "Promotion Gate" in pages[1]
    pdf = Path(result["paths"]["report"]).read_bytes()
    assert len(re.findall(rb"/Type /Page\b", pdf)) == 12
    assert Path(result["paths"]["csv"]).stat().st_size > 0


def test_the_cli_report_lists_the_columns_it_replaced(tmp_path, capsys, monkeypatch):
    # A supplied 'agrees' that says A and B always agree: the cascade replaces it, and both
    # the terminal and the cover of the CLI's report say so.
    predictions = tmp_path / "predictions.csv"
    pd.read_csv(DEMO).assign(agrees=1).to_csv(predictions, index=False)
    pages = _record_pages(monkeypatch)
    main_pipeline._run_ab_testing(_config(tmp_path / "out", predictions),
                                  model_a=None, loader=None, device="cpu")
    issue = "agrees disagrees with pred_a == pred_b on 325 row(s)"
    assert f"diagnostic 'input': {issue}" in capsys.readouterr().out
    assert any(issue in text for text in pages[0])

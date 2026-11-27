"""The decision job runs the gate with the shipped thresholds and stores the gate's values."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from steeldefect_active_learning.ab_testing.metrics import (
    compute_all_metrics, statistical_significance_test,
)
from steeldefect_active_learning.ab_testing.promotion import decide_promotion
from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config
from steeldefect_platform.obt_schema import COMPANION_DDL

JOB = Path(__file__).resolve().parents[1] / "databricks" / "02_ab_decision.py"
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"


def _ddl_columns() -> list:
    body = COMPANION_DDL["steeldefect.ab_decisions"].split("(", 1)[1].rsplit(")", 1)[0]
    return [column.strip().split()[0] for column in body.split(",")]


def _promote_frame() -> pd.DataFrame:
    """B fixes 36 of A's 57 errors per defect class and breaks 3; clean gains 90 net."""
    rows = []
    for k in range(11):
        n11, n10, n01, n00 = (1450, 10, 100, 40) if k == 10 else (240, 3, 36, 21)
        wrong_a, wrong_b = (k + 1) % 11, (k + 2) % 11
        rows += [(k, k, k)] * n11 + [(k, k, wrong_b)] * n10
        rows += [(k, wrong_a, k)] * n01 + [(k, wrong_a, wrong_a)] * n00
    df = pd.DataFrame(rows, columns=["label", "pred_a", "pred_b"])
    df.insert(0, "image_path", [f"img_{j:06d}.png" for j in range(len(df))])
    rng = np.random.default_rng(0)
    return df.assign(confidence_a=rng.uniform(0.5, 1.0, len(df)),
                     confidence_b=rng.uniform(0.5, 1.0, len(df)))


class _Slice:
    def __init__(self, df):
        self.df = df

    def where(self, predicate):
        return self

    def toPandas(self):
        return self.df.copy()


class _Writer:
    def __init__(self, rows, saved):
        self.rows, self.saved = rows, saved

    def format(self, _):
        return self

    def mode(self, _):
        return self

    def saveAsTable(self, name):
        self.saved.append((name, self.rows))


class FakeSpark:
    """Serves one evaluation slice and records the appended audit rows."""

    def __init__(self, df):
        self.df, self.saved = df, []

    def table(self, name):
        assert name == "steeldefect.ab_eval_slices"
        return _Slice(self.df)

    def createDataFrame(self, rows):
        return type("Frame", (), {"write": _Writer(rows, self.saved)})()


@pytest.fixture
def job(monkeypatch):
    # main() prepends a workspace path to sys.path: restore it after the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("ab_decision_job", JOB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry_calls(monkeypatch):
    """Records the compare-and-set calls on a registry whose champion is v6."""
    import steeldefect_platform.mlflow_registry as registry
    calls = []

    def promote_if_champion(model_name, version, expected_champion, client=None):
        calls.append((model_name, version, expected_champion))
        return {"applied": expected_champion == 6, "champion_before": 6,
                "champion_after": version if expected_champion == 6 else 6, "detail": ""}

    monkeypatch.setattr(registry, "promote_if_champion", promote_if_champion)
    return calls


@pytest.fixture
def config_path(tmp_path):
    """The shipped YAML without the BCa bootstrap, whose jackknife is quadratic in n."""
    raw = yaml.safe_load(Path(SHIPPED_CONFIG_PATH).read_text(encoding="utf-8"))
    raw["ab_testing"]["bootstrap_n"] = 0
    path = tmp_path / "pipeline_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def _expected_verdict(df: pd.DataFrame, config_path: str) -> dict:
    cfg = load_config(config_path).ab_testing
    metrics = compute_all_metrics(
        df, task="classification", alpha=cfg.alpha,
        correction=cfg.multiple_testing_correction, srm_threshold=cfg.srm_threshold,
        expected_ratio=cfg.expected_arm_ratio, bootstrap_n=cfg.bootstrap_n, bayesian=True,
        bayesian_decision_threshold=cfg.bayesian_decision_threshold, sequential=cfg.sequential)
    return decide_promotion(metrics, **cfg.gate_kwargs())


@pytest.mark.parametrize("make, decision", [(_promote_frame, "promote_B"),
                                            (lambda: pd.read_csv(DEMO), "keep_A")],
                         ids=["clearly better B", "demo slice"])
def test_job_stores_the_configured_gate_verdict(job, registry_calls, config_path, make, decision):
    df = make()
    spark = FakeSpark(df)
    promoted = job.main(spark, "eval-1", "models:/steeldefect_segmenter/6",
                        "models:/steeldefect_segmenter/7", config_path=config_path)
    expected = _expected_verdict(df, config_path)
    assert expected["decision"] == decision
    [(table, [row])] = spark.saved
    assert table == "steeldefect.ab_decisions"
    assert list(row) == _ddl_columns()
    assert row["decision"] == decision
    assert row["mcnemar_p"] == pytest.approx(expected["estimates"]["mcnemar_p_one_sided"], rel=1e-12, abs=0)
    assert row["prob_b_better"] == pytest.approx(expected["estimates"]["p_b_better_posterior"], rel=1e-12)
    # No model version: the job never touches the registry.
    assert promoted is None and row["promoted_model_uri"] is None and registry_calls == []


def _normal_confidences(df: pd.DataFrame) -> pd.DataFrame:
    """The same images with normal confidences: Shapiro-Wilk keeps Welch's t (seed 1)."""
    rng = np.random.default_rng(1)
    return df.assign(confidence_a=rng.normal(0.70, 0.05, len(df)),
                     confidence_b=rng.normal(0.68, 0.05, len(df)))


@pytest.mark.parametrize("make, test_used", [
    (_promote_frame, "mann_whitney_u"),
    (lambda: _normal_confidences(_promote_frame()), "welch_t"),
], ids=["uniform confidences", "normal confidences"])
def test_job_stores_each_confidence_test_in_its_own_columns(job, registry_calls, config_path,
                                                            make, test_used):
    # A Mann-Whitney run used to store its U as welch_t and its p-value as welch_p.
    df = make()
    spark = FakeSpark(df)
    job.main(spark, "eval-1", "a", "b", config_path=config_path)
    row = spark.saved[0][1][0]
    stat = statistical_significance_test(df)
    assert row["metric"] == stat["test_used"] == test_used
    welch = test_used == "welch_t"
    expected = {"welch_t": stat["t_stat"] if welch else None,
                "welch_p": stat["p_value"] if welch else None,
                "mannwhitney_p": None if welch else stat["p_value"]}
    for column, value in expected.items():
        if value is None:
            assert np.isnan(row[column]), column
        else:
            assert row[column] == pytest.approx(value, rel=1e-12), column


def test_job_promotes_only_with_a_model_version(job, registry_calls, config_path):
    spark = FakeSpark(_promote_frame())
    promoted = job.main(spark, "eval-1", "models:/steeldefect_segmenter/6",
                        "models:/steeldefect_segmenter/7", model_b_version=7,
                        config_path=config_path, model_a_version=6)
    assert promoted == "models:/steeldefect_segmenter/7"
    assert registry_calls == [("steeldefect_segmenter", 7, 6)]
    assert spark.saved[0][1][0]["promoted_model_uri"] == "models:/steeldefect_segmenter/7"


def _config_with(config_path: str, tmp_path: Path, name: str, ab_testing=None, promotion=None) -> str:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw["ab_testing"].update(ab_testing or {})
    raw["ab_testing"]["promotion"].update(promotion or {})
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_job_reads_the_gate_thresholds_from_its_config(job, registry_calls, config_path, tmp_path):
    # The shipped thresholds promote this B; a min_gain it cannot clear must keep A.
    strict = _config_with(config_path, tmp_path, "strict.yaml", promotion={
        "min_gain": 0.10, "min_detectable_gain": 0.12, "planning_discordance": 0.20})
    spark = FakeSpark(_promote_frame())
    assert job.main(spark, "eval-1", "a", "models:/steeldefect_segmenter/7", model_b_version=7,
                    config_path=strict) is None
    assert spark.saved[0][1][0]["decision"] == "keep_A" and registry_calls == []


def _class_0_differs() -> pd.DataFrame:
    """Only class 0 has discordant images (A 5, B 21): its BH-adjusted McNemar p is 0.036."""
    rows = []
    for k in range(11):
        n11, n10, n01, n00 = (200, 5, 21, 20) if k == 0 else (200, 0, 0, 20)
        rows += [(k, k, k)] * n11 + [(k, k, (k + 2) % 11)] * n10
        rows += [(k, (k + 1) % 11, k)] * n01 + [(k, (k + 1) % 11, (k + 1) % 11)] * n00
    df = pd.DataFrame(rows, columns=["label", "pred_a", "pred_b"])
    df.insert(0, "image_path", [f"img_{j:06d}.png" for j in range(len(df))])
    return df


@pytest.mark.parametrize("alpha, reject", [(0.05, True), (0.025, False)])
def test_job_passes_its_alpha_to_the_diagnostics(job, registry_calls, config_path, tmp_path,
                                                 alpha, reject):
    path = _config_with(config_path, tmp_path, f"alpha_{alpha}.yaml", ab_testing={"alpha": alpha})
    spark = FakeSpark(_class_0_differs())
    job.main(spark, "eval-1", "a", "b", config_path=path)
    assert spark.saved[0][1][0]["bh_reject"] is reject


def test_job_keeps_a_on_an_invalid_slice(job, registry_calls, config_path):
    spark = FakeSpark(pd.concat([_promote_frame()] * 2))     # every image twice
    assert job.main(spark, "eval-1", "a", "b", model_b_version=7, config_path=config_path) is None
    row = spark.saved[0][1][0]
    assert row["decision"] == "keep_A" and np.isnan(row["mcnemar_p"]) and registry_calls == []

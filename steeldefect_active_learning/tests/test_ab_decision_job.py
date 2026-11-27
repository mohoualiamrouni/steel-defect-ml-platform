"""The decision job reads a paired slice, checks it, runs the gate with the shipped
thresholds, and leaves an audit trail that explains its decision.

The Spark session is a fake on an SQLite database: it serves the slice, creates the tables
from the job's DDL columns, and runs the job's provenance and prior-decision queries as
written, with their named parameters: that SQL is executed, not imitated.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from steeldefect_active_learning.ab_testing.metrics import (
    compute_all_metrics, statistical_significance_test,
)
from steeldefect_active_learning.ab_testing.promotion import VERDICT_SCHEMA, decide_promotion
from steeldefect_active_learning.ab_testing.serialization import strict
from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config
from steeldefect_platform.obt_schema import ddl_columns, ddl_schema

JOB = Path(__file__).resolve().parents[1] / "databricks" / "02_ab_decision.py"
DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"
DECISIONS = "steeldefect.ab_decisions"
IMAGES = "steeldefect.ab_decision_images"
MANIFESTS = "steeldefect.model_training_images"
SLICE_QUERY = ("SELECT * FROM steeldefect.ab_eval_slices "
               "WHERE eval_dataset_id = :eval_dataset_id")
A, B = 6, 7                      # the champion that scored the slice, and the challenger
#: The columns of a table created before this schema (its rows have no gate_version).
V1_COLUMNS = ["ab_run_id", "model_a_uri", "model_b_uri", "eval_dataset_id", "metric",
              "welch_t", "welch_p", "mannwhitney_p", "mcnemar_p", "bh_reject", "cohen_d",
              "ci_low", "ci_high", "srm_p", "prob_b_better", "decision",
              "promoted_model_uri", "decided_ts"]
SQLITE_TYPE = {"STRING": "TEXT", "INT": "INTEGER", "DOUBLE": "REAL", "TIMESTAMP": "TEXT",
               "BOOLEAN": "INTEGER"}


def as_slice(df: pd.DataFrame) -> pd.DataFrame:
    """A prediction frame in the slice schema: one image per coil, scored by A v6 and B v7."""
    out = df.drop(columns=[c for c in ("image_path",) if c in df.columns]).reset_index(drop=True)
    n = len(out)
    out.insert(0, "image_id", [f"img_{j:06d}" for j in range(n)])
    out.insert(1, "coil_id", [f"coil_{j:06d}" for j in range(n)])
    out.insert(2, "sampling_frame", "one image per coil, lines 1-3, 2026-05")
    return out.assign(model_a_version=A, model_b_version=B)


def _promote_frame() -> pd.DataFrame:
    """B fixes 36 of A's 57 errors per defect class and breaks 3; clean gains 90 net."""
    rows = []
    for k in range(11):
        n11, n10, n01, n00 = (1450, 10, 100, 40) if k == 10 else (240, 3, 36, 21)
        wrong_a, wrong_b = (k + 1) % 11, (k + 2) % 11
        rows += [(k, k, k)] * n11 + [(k, k, wrong_b)] * n10
        rows += [(k, wrong_a, k)] * n01 + [(k, wrong_a, wrong_a)] * n00
    df = pd.DataFrame(rows, columns=["label", "pred_a", "pred_b"])
    rng = np.random.default_rng(0)
    return as_slice(df.assign(confidence_a=rng.uniform(0.5, 1.0, len(df)),
                              confidence_b=rng.uniform(0.5, 1.0, len(df))))


def _demo_frame() -> pd.DataFrame:
    return as_slice(pd.read_csv(DEMO))


def _manifests(**versions) -> pd.DataFrame:
    """Training manifests: version -> [(image_id, coil_id)] (default: A and B on 100 images
    of their own coils each, none of the slice's)."""
    versions = versions or {"6": [(f"train_{j:05d}", f"tcoil_{j:05d}") for j in range(100)],
                            "7": [(f"train_{j:05d}", f"tcoil_{j:05d}") for j in range(50, 150)]}
    return pd.DataFrame([("steeldefect_segmenter", int(v), i, c)
                         for v, pairs in versions.items() for i, c in pairs],
                        columns=["model_name", "model_version", "image_id", "coil_id"])


class _Frame:
    def __init__(self, df):
        self.df = df

    def toPandas(self):
        return self.df.copy()


class _Writer:
    def __init__(self, spark, rows):
        self.spark, self.rows = spark, rows

    def format(self, _):
        return self

    def mode(self, _):
        return self

    def saveAsTable(self, name):
        self.spark.append(name, self.rows)


class _View:
    def __init__(self, spark, pairs):
        self.spark, self.pairs = spark, pairs

    def createOrReplaceTempView(self, name):
        db = self.spark.db
        db.execute(f"DROP TABLE IF EXISTS main.{name}")
        db.execute(f"CREATE TABLE main.{name} (image_id TEXT, coil_id TEXT)")
        db.executemany(f"INSERT INTO main.{name} VALUES (?, ?)", self.pairs)


class FakeSpark:
    """Serves one slice, and runs the job's DDL, appends and other queries on SQLite."""

    def __init__(self, df, decisions_columns=None, manifests=None):
        self.df = df
        self.db = sqlite3.connect(":memory:")
        self.db.execute("ATTACH DATABASE ':memory:' AS steeldefect")
        self.queries, self.statements, self.schemas, self.saved = [], [], [], []
        self.fail_append = None               # (table, event) -> an error to raise once
        if decisions_columns is not None:     # a table created before this schema
            self.db.execute(f"CREATE TABLE {DECISIONS} ({', '.join(decisions_columns)})")
        self._create(MANIFESTS)
        m = _manifests() if manifests is None else manifests
        self.db.executemany(f"INSERT INTO {MANIFESTS} VALUES (?, ?, ?, ?)",
                            m.itertuples(index=False, name=None))

    def _create(self, table):
        cols = ", ".join(f"{name} {SQLITE_TYPE[kind]}" for name, kind in ddl_columns(table))
        self.db.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")

    def sql(self, query, args=None):
        args = dict(args or {})
        if query.startswith("CREATE TABLE IF NOT EXISTS "):
            self.statements.append(query)
            self._create(query.split()[5])
            return None
        if query.startswith("ALTER TABLE "):
            self.statements.append(query)
            table, added = query.split()[2], query.split("ADD COLUMNS (", 1)[1][:-1]
            for column in added.split(", "):
                name, kind = column.split()
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {SQLITE_TYPE[kind]}")
            return None
        self.queries.append((query, args))
        if query == SLICE_QUERY:
            return _Frame(self.df)
        return _Frame(pd.read_sql_query(query, self.db, params=args))

    def table(self, name):
        schema, table = name.split(".")
        columns = [r[1] for r in self.db.execute(f"PRAGMA {schema}.table_info({table})")]
        return type("Table", (), {"columns": columns})()

    def createDataFrame(self, data, schema):
        if schema == "image_id STRING, coil_id STRING":
            return _View(self, data)
        self.schemas.append(schema)
        names = [column.split()[0] for column in schema.split(", ")]
        rows = [dict(zip(names, values)) for values in data]
        return type("Frame", (), {"write": _Writer(self, rows)})()

    def append(self, table, rows):
        if self.fail_append is not None and (table, rows[0].get("event")) == self.fail_append[0]:
            error, self.fail_append = self.fail_append[1], None
            raise error
        for row in rows:
            values = [v.isoformat() if isinstance(v, datetime) else v for v in row.values()]
            self.db.execute(f"INSERT INTO {table} ({', '.join(row)}) "
                            f"VALUES ({', '.join('?' * len(row))})", values)
        self.saved.append((table, rows))

    def rows(self, table=DECISIONS):
        return [row for t, rows in self.saved for row in rows if t == table]


@pytest.fixture
def job(monkeypatch):
    # main() prepends the checkout to sys.path: restore it after the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("ab_decision_job", JOB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry(monkeypatch):
    """A registry whose champion is v6. A test may make reads fail from the n-th one on,
    move the champion after the first read, or make the compare-and-set fail, before or
    after its write."""
    import steeldefect_platform.mlflow_registry as mlflow_registry
    state = {"champion": 6, "reads": 0, "read_error": None, "read_errors_from": 1,
             "after_first_read": None, "error": None, "error_after_write": False,
             "calls": [], "on_call": None, "registry_uri": None}

    def alias_version(model_name, alias="champion", client=None):
        state["reads"] += 1
        if state["read_error"] is not None and state["reads"] >= state["read_errors_from"]:
            raise state["read_error"]
        champion = state["champion"]
        if state["reads"] == 1 and state["after_first_read"] is not None:
            state["champion"] = state["after_first_read"]
        return champion

    def promote_if_champion(model_name, version, expected_champion, client=None):
        state["calls"].append((model_name, version, expected_champion))
        if state["on_call"] is not None:
            state["on_call"]()
        if state["error"] is not None:
            error, state["error"] = state["error"], None
            if state["error_after_write"]:
                state["champion"] = version
            raise error
        before = state["champion"]
        if before == version:
            return {"applied": True, "champion_before": before, "champion_after": before,
                    "detail": f"already v{version}"}
        if before != expected_champion:
            return {"applied": False, "champion_before": before, "champion_after": before,
                    "detail": f"@champion is v{before}, but B was compared with v{expected_champion}"}
        state["champion"] = version
        return {"applied": True, "champion_before": before, "champion_after": version,
                "detail": f"v{before} -> v{version}"}

    monkeypatch.setattr(mlflow_registry, "alias_version", alias_version)
    monkeypatch.setattr(mlflow_registry, "promote_if_champion", promote_if_champion)
    monkeypatch.setattr(mlflow_registry, "use_registry",
                        lambda uri: state.update(registry_uri=uri))
    return state


@pytest.fixture
def config_path(tmp_path):
    """The shipped YAML without the BCa bootstrap, whose jackknife is quadratic in n."""
    raw = yaml.safe_load(Path(SHIPPED_CONFIG_PATH).read_text(encoding="utf-8"))
    raw["ab_testing"]["bootstrap_n"] = 0
    path = tmp_path / "pipeline_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def _cascade(df: pd.DataFrame, config_path: str):
    cfg = load_config(config_path).ab_testing
    metrics = compute_all_metrics(
        df, task="classification", alpha=cfg.alpha,
        correction=cfg.multiple_testing_correction, srm_threshold=cfg.srm_threshold,
        expected_ratio=cfg.expected_arm_ratio, bootstrap_n=cfg.bootstrap_n, bayesian=True,
        bayesian_decision_threshold=cfg.bayesian_decision_threshold, sequential=cfg.sequential,
        id_col="image_id", cluster_col="coil_id")
    return metrics, decide_promotion(metrics, **cfg.gate_kwargs())


def _valid_input(row) -> str:
    [reason] = [r for r in json.loads(row["verdict_json"])["reasons"]
                if r["criterion"] == "valid_input"]
    return reason["detail"]


@pytest.mark.parametrize("make, decision", [(_promote_frame, "promote_B"),
                                            (_demo_frame, "keep_A")],
                         ids=["clearly better B", "demo slice"])
def test_the_decision_row_explains_the_verdict(job, registry, config_path, make, decision):
    df = make()
    spark = FakeSpark(df)
    promoted = job.main(spark, "eval-1", A, B, promote=False, config_path=config_path)
    metrics, expected = _cascade(df, config_path)
    assert expected["decision"] == decision
    [row] = spark.rows()
    assert (row["event"], row["gate_version"]) == ("decision", VERDICT_SCHEMA)
    assert len(row["decision_id"]) == 32
    assert (row["model_a_version"], row["model_b_version"]) == (A, B)
    assert (row["model_a_uri"], row["model_b_uri"]) == ("models:/steeldefect_segmenter/6",
                                                        "models:/steeldefect_segmenter/7")
    assert (row["eval_dataset_id"], row["sampling_frame"]) == ("eval-1", df["sampling_frame"][0])
    assert (row["decision"], row["conclusion"]) == (decision, expected["conclusion"])
    assert row["n_images"] == expected["n_images"] == len(df)
    cells = ("both_correct", "only_a_correct", "only_b_correct", "both_wrong")
    assert {c: row[c] for c in cells} == {c: expected["counts"][c] for c in cells}
    est = expected["estimates"]
    assert row["estimand"] == est["estimand"]
    assert row["estimand_lower"] == pytest.approx(est["estimand_lower"], rel=1e-12)
    assert row["mcnemar_p"] == pytest.approx(est["mcnemar_p_one_sided"], rel=1e-12, abs=0)
    assert row["prob_b_better"] == pytest.approx(est["p_b_better_posterior"], rel=1e-12)
    failing = [r["criterion"] for r in expected["reasons"] if not r["passed"]]
    assert row["failing_criteria"] == (",".join(failing) or None)
    # The verdict, the diagnostics and the job's parameters, whole and as strict JSON.
    assert json.loads(row["verdict_json"]) == strict(expected)
    diagnostics = json.loads(row["diagnostics_json"])
    assert "paired_counts" not in diagnostics and diagnostics["mcnemar"] == strict(metrics["mcnemar"])
    params = json.loads(row["job_json"])
    assert (params["id_col"], params["cluster_col"], params["promote"]) == ("image_id", "coil_id", False)
    assert params["cascade"]["bootstrap_n"] == 0 and params["registry_uri"] is None
    # promote=False: the decision is recorded, and the registry is left alone.
    assert (row["promotion_status"], row["promotion_detail"]) == (
        ("not_requested", "promote_B with promote=False: decision only")
        if decision == "promote_B" else ("not_applicable", "decision keep_A: nothing to apply"))
    assert promoted is None and row["promoted_model_uri"] is None
    assert registry["calls"] == [] and registry["reads"] == 0
    # The images of an evaluation that counts.
    assert [(r["image_id"], r["coil_id"]) for r in spark.rows(IMAGES)] == list(
        zip(df["image_id"], df["coil_id"]))


def test_the_audit_rows_have_the_table_types(job, registry, config_path):
    # A decided_ts string in a TIMESTAMP column, and a column None on every row, which
    # Spark cannot type: the rows now carry the table's own schema.
    spark = FakeSpark(_promote_frame())
    job.main(spark, "eval-1", A, B, config_path=config_path)
    assert spark.schemas == [ddl_schema(IMAGES), ddl_schema(DECISIONS), ddl_schema(DECISIONS)]
    python_type = {"STRING": str, "INT": int, "DOUBLE": float, "TIMESTAMP": datetime}
    for row in spark.rows():
        assert list(row) == [name for name, _ in ddl_columns(DECISIONS)]
        for name, kind in ddl_columns(DECISIONS):
            value = row[name]
            assert value is None or type(value) is python_type[kind], (name, value)
        assert row["event_ts"].tzinfo is timezone.utc


def _normal_confidences(df: pd.DataFrame) -> pd.DataFrame:
    """The same images with normal confidences: Shapiro-Wilk keeps Welch's t (seed 1)."""
    rng = np.random.default_rng(1)
    return df.assign(confidence_a=rng.normal(0.70, 0.05, len(df)),
                     confidence_b=rng.normal(0.68, 0.05, len(df)))


@pytest.mark.parametrize("make, test_used", [
    (_promote_frame, "mann_whitney_u"),
    (lambda: _normal_confidences(_promote_frame()), "welch_t"),
], ids=["uniform confidences", "normal confidences"])
def test_the_diagnostics_keep_each_confidence_test_under_its_name(job, registry, config_path,
                                                                 make, test_used):
    # The audit row had no column for the Mann-Whitney U.
    df = make()
    spark = FakeSpark(df)
    job.main(spark, "eval-1", A, B, promote=False, config_path=config_path)
    stored = json.loads(spark.rows()[0]["diagnostics_json"])["statistical"]
    stat = strict(statistical_significance_test(df))
    assert stored["test_used"] == stat["test_used"] == test_used
    assert (stored["t_stat"], stored["u_stat"], stored["p_value"]) == (
        stat["t_stat"], stat["u_stat"], stat["p_value"])
    assert (stored["t_stat"] is None) == (test_used == "mann_whitney_u")


def test_the_decision_is_recorded_before_the_registry_moves(job, registry, config_path):
    # The job used to promote first: a failed append left @champion moved and no row.
    spark = FakeSpark(_promote_frame())
    seen = []
    registry["on_call"] = lambda: seen.append([t for t, _ in spark.saved])
    promoted = job.main(spark, "eval-1", A, B, config_path=config_path)
    assert seen == [[IMAGES, DECISIONS]] and registry["calls"] == [("steeldefect_segmenter", 7, 6)]
    decision, promotion = spark.rows()
    assert (decision["promotion_status"], decision["promotion_detail"]) == (
        "pending", "promote steeldefect_segmenter v7 over v6")
    assert promotion["decision_id"] == decision["decision_id"]
    assert (promotion["event"], promotion["promotion_status"]) == ("promotion", "applied")
    assert (promotion["champion_before"], promotion["champion_after"]) == (6, 7)
    assert promoted == promotion["promoted_model_uri"] == "models:/steeldefect_segmenter/7"
    assert promotion["decision"] is None and promotion["verdict_json"] is None


@pytest.mark.parametrize("setup, message, status, detail", [
    (lambda r: r.update(after_first_read=8), "was not applied \\(refused\\)", "refused",
     "@champion is v8, but B was compared with v6"),
    (lambda r: r.update(error=ConnectionError("registry unreachable")),
     "was not applied \\(failed\\)", "failed",
     "registry error: ConnectionError: registry unreachable; @champion reads v6"),
    (lambda r: r.update(error=ConnectionError("registry unreachable"), error_after_write=True,
                        read_error=TimeoutError("again"), read_errors_from=2),
     "may have been applied \\(unknown\\)", "unknown",
     "registry error: ConnectionError: registry unreachable; @champion could not be read "
     "back (TimeoutError): check it by hand"),
], ids=["champion moved during the run", "registry error", "outcome unknown"])
def test_a_promotion_that_is_not_applied_fails_the_job(job, registry, config_path,
                                                       setup, message, status, detail):
    # A registry error used to be swallowed without a log, and reported as no promotion.
    setup(registry)
    spark = FakeSpark(_promote_frame())
    with pytest.raises(RuntimeError, match=f"promote_B {message}"):
        job.main(spark, "eval-1", A, B, config_path=config_path)
    decision, promotion = spark.rows()
    assert (decision["decision"], decision["promotion_status"]) == ("promote_B", "pending")
    assert (promotion["promotion_status"], promotion["promotion_detail"]) == (status, detail)
    assert promotion["promoted_model_uri"] is None


@pytest.mark.parametrize("setup, error", [
    (lambda r: r.update(champion=8),
     "@champion of steeldefect_segmenter is v8, but the slice was scored by v6"),
    (lambda r: r.update(read_error=LookupError("RESOURCE_DOES_NOT_EXIST")),
     "RESOURCE_DOES_NOT_EXIST"),
], ids=["champion moved before the run", "wrong model name"])
def test_a_run_that_cannot_promote_stops_before_recording_anything(job, registry, config_path,
                                                                   setup, error):
    # Recorded, the decision of a run that cannot promote would use the slice up (one
    # decision per challenger per slice): it stops before it records anything.
    setup(registry)
    spark = FakeSpark(_promote_frame())
    with pytest.raises((RuntimeError, LookupError), match=error):
        job.main(spark, "eval-1", A, B, config_path=config_path)
    assert spark.saved == [] and registry["calls"] == []


def _config_with(config_path: str, tmp_path: Path, name: str, ab_testing=None, promotion=None) -> str:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw["ab_testing"].update(ab_testing or {})
    raw["ab_testing"]["promotion"].update(promotion or {})
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_job_reads_the_gate_thresholds_from_its_config(job, registry, config_path, tmp_path):
    # The shipped thresholds promote this B; a min_gain it cannot clear must keep A.
    strict_config = _config_with(config_path, tmp_path, "strict.yaml", promotion={
        "min_gain": 0.10, "min_detectable_gain": 0.12, "planning_discordance": 0.20})
    spark = FakeSpark(_promote_frame())
    assert job.main(spark, "eval-1", A, B, config_path=strict_config) is None
    [row] = spark.rows()
    assert (row["decision"], row["promotion_status"]) == ("keep_A", "not_applicable")
    assert json.loads(row["verdict_json"])["thresholds"]["min_gain"] == 0.10
    assert registry["calls"] == []


def _class_0_differs() -> pd.DataFrame:
    """Only class 0 has discordant images (A 5, B 21): its BH-adjusted McNemar p is 0.036."""
    rows = []
    for k in range(11):
        n11, n10, n01, n00 = (200, 5, 21, 20) if k == 0 else (200, 0, 0, 20)
        rows += [(k, k, k)] * n11 + [(k, k, (k + 2) % 11)] * n10
        rows += [(k, (k + 1) % 11, k)] * n01 + [(k, (k + 1) % 11, (k + 1) % 11)] * n00
    return as_slice(pd.DataFrame(rows, columns=["label", "pred_a", "pred_b"]))


@pytest.mark.parametrize("alpha, reject", [(0.05, True), (0.025, False)])
def test_job_passes_its_alpha_to_the_diagnostics(job, registry, config_path, tmp_path,
                                                 alpha, reject):
    path = _config_with(config_path, tmp_path, f"alpha_{alpha}.yaml", ab_testing={"alpha": alpha})
    spark = FakeSpark(_class_0_differs())
    job.main(spark, "eval-1", A, B, config_path=path)
    per_class = json.loads(spark.rows()[0]["diagnostics_json"])["per_class_mcnemar"]
    assert per_class["alpha"] == alpha and any(r["reject"] for r in per_class["rows"]) is reject


def test_an_invalid_slice_is_recorded_without_diagnostics_and_does_not_count(
        job, registry, config_path):
    # The row keeps the reasons and no diagnostic, so that it shows nothing of the outcome
    # before the slice is fixed; and it is not the decision, so the fixed slice gets one.
    frame = _promote_frame()
    spark = FakeSpark(pd.concat([frame, frame]))          # every image twice
    assert job.main(spark, "eval-1", A, B, config_path=config_path) is None
    [row] = spark.rows()
    assert (row["decision"], row["conclusion"]) == ("keep_A", "invalid_input")
    assert row["failing_criteria"] == "valid_input" and row["diagnostics_json"] is None
    assert row["mcnemar_p"] is None and row["n_images"] is None and row["both_correct"] is None
    assert spark.rows(IMAGES) == [] and registry["calls"] == []
    # The fixed slice, under the same id, gets its decision.
    spark.df = frame
    assert job.main(spark, "eval-1", A, B, config_path=config_path) == "models:/steeldefect_segmenter/7"


def test_an_older_decisions_table_gets_the_new_columns(job, registry, config_path):
    spark = FakeSpark(_promote_frame(), decisions_columns=list(V1_COLUMNS))
    job.main(spark, "eval-1", A, B, promote=False, config_path=config_path)
    new = [f"{name} {kind}" for name, kind in ddl_columns(DECISIONS) if name not in V1_COLUMNS]
    assert f"ALTER TABLE {DECISIONS} ADD COLUMNS ({', '.join(new)})" in spark.statements
    assert set(spark.table(DECISIONS).columns) == set(V1_COLUMNS) | {
        name for name, _ in ddl_columns(DECISIONS)}
    # A table already in the new schema is left alone.
    fresh = FakeSpark(_promote_frame())
    job.main(fresh, "eval-2", A, B, promote=False, config_path=config_path)
    assert not any(s.startswith("ALTER") for s in fresh.statements)


def test_the_slice_id_is_a_query_parameter(job, registry, config_path):
    # Interpolated into the SQL text, this id used to select every slice.
    hostile = "eval-1' OR '1'='1"
    spark = FakeSpark(_promote_frame())
    job.main(spark, hostile, A, B, promote=False, config_path=config_path)
    assert (SLICE_QUERY, {"eval_dataset_id": hostile}) in spark.queries
    assert not any("'1'='1" in query for query, _ in spark.queries)


def test_the_decision_row_identifies_the_slice_by_its_images(job, registry, config_path):
    df = _promote_frame()
    spark = FakeSpark(df)
    job.main(spark, "eval-1", A, B, promote=False, config_path=config_path)
    expected = hashlib.sha256("\n".join(sorted(df["image_id"])).encode("utf-8")).hexdigest()
    assert spark.rows()[0]["slice_sha256"] == expected


def test_one_decision_per_challenger_per_slice(job, registry, config_path):
    # Deciding again on the same images or coils, under another id or other thresholds,
    # until B passes is a multiple test that the gate's level does not cover.
    df = _promote_frame()
    spark = FakeSpark(df)
    # The decision of another model on these images does not count for this one.
    spark.db.executemany(f"INSERT INTO {MANIFESTS} VALUES (?, ?, ?, ?)",
                         [("other_segmenter", 6, "train_0", "tcoil_0"),
                          ("other_segmenter", 7, "train_1", "tcoil_1")])
    job.main(spark, "eval-1", A, B, model_name="other_segmenter", promote=False,
             config_path=config_path)
    job.main(spark, "eval-1", A, B, promote=False, config_path=config_path)
    first = spark.rows()[1]["decision_id"]
    fresh = df.assign(image_id=df["image_id"] + "_new", coil_id=df["coil_id"] + "_new")
    renamed = df.iloc[1:].assign(coil_id=lambda d: d["coil_id"] + "_renamed")
    for eval_id, frame in (("eval-1", df), ("eval-1-copy", df),
                           ("eval-1-minus-one", df.iloc[1:]),
                           ("eval-other-images", df.assign(image_id=df["image_id"] + "_b")),
                           ("eval-renamed-coils", renamed),   # its images, under new coil ids
                           ("eval-1", fresh)):     # the same id, on images and coils never used
        spark.df = frame
        with pytest.raises(RuntimeError, match=f"v7 already has a decision on this slice \\({first}\\)"):
            job.main(spark, eval_id, A, B, config_path=config_path)
    # The same images under another id, once the decision's image rows are gone: the
    # slice's digest still names them.
    spark.db.execute(f"DELETE FROM {IMAGES}")
    spark.df = df
    with pytest.raises(RuntimeError, match=f"already has a decision on this slice \\({first}\\)"):
        job.main(spark, "eval-1-copy", A, B, config_path=config_path)
    assert len(spark.rows()) == 2 and registry["calls"] == []
    # Another challenger, scored on the same images, gets its own decision.
    spark.df = df.assign(model_b_version=8)
    spark.db.execute(f"INSERT INTO {MANIFESTS} VALUES ('steeldefect_segmenter', 8, 'train_1', 'tcoil_1')")
    job.main(spark, "eval-1", A, 8, promote=False, config_path=config_path)
    assert [(r["model_name"], r["model_b_version"]) for r in spark.rows()] == [
        ("other_segmenter", 7), ("steeldefect_segmenter", 7), ("steeldefect_segmenter", 8)]


@pytest.mark.parametrize("manifests, reasons", [
    (_manifests(**{"6": [("train_0", "tcoil_0")],
                   "7": [("img_000003", "coil_000003"), ("img_000010", "coil_000010"),
                         ("img_004599", "coil_004599"), ("train_1", "tcoil_1")]}),
     ["provenance: 3 image(s) of the slice are in the training manifest of B (v7)",
      "provenance: 3 coil(s) of the slice have images in the training manifest of B (v7)"]),
    (_manifests(**{"6": [("train_0", "coil_000042")], "7": [("train_1", "tcoil_1")]}),
     ["provenance: 1 coil(s) of the slice have images in the training manifest of A (v6)"]),
    (_manifests(**{"6": [("train_0", "tcoil_0")],
                   "7": [("img_000003", "coil_000003")] * 3}),
     ["provenance: 1 image(s) of the slice are in the training manifest of B (v7)"]),
    (_manifests(**{"7": [("train_1", "tcoil_1")]}),
     ["provenance: no training manifest for A (steeldefect_segmenter v6) in "
      "steeldefect.model_training_images: the slice cannot be checked against its training "
      "images"]),
], ids=["B trained on 3 slice images", "a slice coil in A's training set",
        "a manifest row repeated", "no manifest for A"])
def test_a_slice_the_models_may_have_trained_on_keeps_a(job, registry, config_path,
                                                       manifests, reasons):
    spark = FakeSpark(_promote_frame(), manifests=manifests)
    assert job.main(spark, "eval-1", A, B, config_path=config_path) is None
    [row] = spark.rows()
    assert (row["decision"], row["conclusion"]) == ("keep_A", "invalid_input")
    assert all(reason in _valid_input(row) for reason in reasons) and registry["calls"] == []


def test_another_models_training_images_do_not_count(job, registry, config_path):
    other = pd.DataFrame([("other_segmenter", 7, "img_000003", "coil_000003")],
                         columns=_manifests().columns)
    spark = FakeSpark(_promote_frame(), manifests=pd.concat([_manifests(), other]))
    assert job.main(spark, "eval-1", A, B, config_path=config_path) == \
        "models:/steeldefect_segmenter/7"


@pytest.mark.parametrize("change, reason", [
    (lambda df: df.assign(model_a_version=np.where(df.index < 30, 5, A)),
     "slice: 30 row(s) were scored by another version of A than v6 (model_a_version: ['5'])"),
    (lambda df: df.assign(model_b_version=8),
     "slice: 4600 row(s) were scored by another version of B than v7 (model_b_version: ['8'])"),
    (lambda df: df.drop(columns="sampling_frame"),
     "slice: missing column(s) ['sampling_frame']"),
    (lambda df: df.assign(sampling_frame=np.where(df.index % 2 == 0, "lines 1-3", "line 4")),
     "slice: sampling_frame must hold one value on every row (2 distinct, 0 missing)"),
    (lambda df: df.assign(sampling_frame=np.where(df.index < 10, None, df["sampling_frame"])),
     "slice: sampling_frame must hold one value on every row (1 distinct, 10 missing)"),
    (lambda df: df.assign(coil_id=[f"coil_{j // 2:06d}" for j in range(len(df))]),
     "coil_id"),
    (lambda df: df.assign(image_id=np.arange(len(df), dtype=float)),
     "slice: 4600 value(s) of image_id are not strings (['float'])"),
    (lambda df: df.iloc[0:0], "no rows"),
], ids=["A scored by v5", "B scored by v8", "no sampling frame", "two sampling frames",
        "sampling frame missing on 10 rows", "two images per coil", "float image ids",
        "empty slice"])
def test_a_slice_that_cannot_stand_for_the_comparison_keeps_a(job, registry, config_path,
                                                             change, reason):
    spark = FakeSpark(change(_promote_frame()))
    assert job.main(spark, "eval-1", A, B, config_path=config_path) is None
    [row] = spark.rows()
    assert (row["decision"], row["conclusion"]) == ("keep_A", "invalid_input")
    detail = _valid_input(row)
    assert detail.startswith("invalid input: ") and reason in detail
    assert row["diagnostics_json"] is None and registry["calls"] == []


@pytest.mark.parametrize("setup, crash", [
    (lambda spark, r: setattr(spark, "fail_append", ((DECISIONS, "promotion"),
                                                     RuntimeError("cluster lost"))),
     "cluster lost"),
    (lambda spark, r: r.update(error=ConnectionError("registry unreachable")),
     "was not applied \\(failed\\)"),
], ids=["crash after the alias moved", "registry error"])
def test_the_next_run_settles_an_unsettled_promotion(job, registry, config_path, setup, crash):
    # Without the resume, the one-decision rule would refuse the retry, and the slice
    # would be lost.
    spark = FakeSpark(_promote_frame())
    setup(spark, registry)
    with pytest.raises(RuntimeError, match=crash):
        job.main(spark, "eval-1", A, B, config_path=config_path)
    decision_id = spark.rows()[0]["decision_id"]
    # The job is upgraded before the retry: the promotion row keeps the gate version of
    # the decision it settles.
    spark.db.execute(f"UPDATE {DECISIONS} SET gate_version = 'promotion_verdict/1'"
                     f" WHERE event = 'decision'")
    assert job.main(spark, "eval-1", A, B, config_path=config_path) == \
        "models:/steeldefect_segmenter/7"
    assert [r["event"] for r in spark.rows()][0] == "decision"
    last = spark.rows()[-1]
    assert (last["decision_id"], last["event"], last["promotion_status"]) == (
        decision_id, "promotion", "applied")
    assert (last["eval_dataset_id"], last["gate_version"]) == ("eval-1", "promotion_verdict/1")
    assert registry["champion"] == 7 and len(spark.rows(IMAGES)) == 4600


def _first_run(job, spark, registry, config_path, outcome: str) -> str:
    """A first decision on eval-1, promote_B, whose promotion ended as ``outcome``."""
    if outcome == "failed":
        registry["error"] = ConnectionError("registry unreachable")
    elif outcome == "refused":
        registry["after_first_read"] = 8          # @champion moves during the run
    if outcome == "applied":
        job.main(spark, "eval-1", A, B, config_path=config_path)
    else:
        with pytest.raises(RuntimeError, match=f"was not applied \\({outcome}\\)"):
            job.main(spark, "eval-1", A, B, config_path=config_path)
    return spark.rows()[0]["decision_id"]


def _another_a(spark, kwargs):
    kwargs["model_a_version"] = 5


def _decision_only(spark, kwargs):
    kwargs["promote"] = False


def _one_image_fewer(spark, kwargs):
    spark.df = spark.df.iloc[1:]


def _a_copy_under_another_id(spark, kwargs):
    kwargs["eval_dataset_id"] = "eval-1-copy"


def _another_slice_sharing_a_coil(spark, kwargs):
    df = spark.df
    other = df.assign(image_id=df["image_id"] + "_2", coil_id=df["coil_id"] + "_2")
    other.loc[0, "coil_id"] = df.loc[0, "coil_id"]
    spark.df, kwargs["eval_dataset_id"] = other, "eval-2"


def _a_second_decision(spark, kwargs):
    """The same decision recorded twice, as two runs from two targets could."""
    cols = [name for name, _ in ddl_columns(DECISIONS)]
    copied = ", ".join("'copy' || decision_id" if c == "decision_id" else c for c in cols)
    spark.db.execute(f"INSERT INTO {DECISIONS} ({', '.join(cols)}) SELECT {copied} "
                     f"FROM {DECISIONS} WHERE event = 'decision'")


def _the_same_run(spark, kwargs):
    pass


@pytest.mark.parametrize("outcome, change", [
    ("failed", _another_a), ("failed", _decision_only), ("failed", _one_image_fewer),
    ("failed", _a_copy_under_another_id), ("failed", _another_slice_sharing_a_coil),
    ("failed", _a_second_decision), ("applied", _the_same_run), ("refused", _the_same_run),
], ids=["another A", "promote=false", "the same id, one image fewer",
        "the same images, another id", "another slice sharing a coil",
        "two decisions on the slice", "after applied", "after refused"])
def test_only_the_same_unsettled_promotion_is_resumed(job, registry, config_path,
                                                     outcome, change):
    # Any other run on the slice is a second decision. A resume on another slice would
    # also promote B on evidence that run never looked at.
    spark = FakeSpark(_promote_frame())
    first = _first_run(job, spark, registry, config_path, outcome)
    registry["champion"] = 6                  # A is the champion again (a rollback)
    calls, rows = list(registry["calls"]), len(spark.rows())
    kwargs = {"eval_dataset_id": "eval-1", "model_a_version": A, "promote": True}
    change(spark, kwargs)
    with pytest.raises(RuntimeError, match="already has a decision on this slice") as info:
        job.main(spark, kwargs.pop("eval_dataset_id"), kwargs.pop("model_a_version"), B,
                 config_path=config_path, **kwargs)
    assert first in str(info.value)
    assert registry["calls"] == calls and len(spark.rows()) == rows


def test_the_dispatcher_runs_the_decision_and_the_promotion(job, registry, config_path,
                                                           monkeypatch, capsys):
    # The orchestrated path could never promote: jobs.job_ab_decision took no version.
    # run_task -> jobs.job_ab_decision -> 02_ab_decision.main -> the registry, with the
    # versions and promote as strings, as Databricks job parameters pass them.
    from steeldefect_platform import jobs, run_task
    spark = FakeSpark(_promote_frame())
    monkeypatch.setattr(jobs, "_spark", lambda: spark)
    params = {"eval_dataset_id": "eval-1", "model_a_version": "6", "model_b_version": "7",
              "promote": "true", "config_path": config_path}
    assert run_task.main(["ab_decision", "--params", json.dumps(params)]) == 0
    assert registry["calls"] == [("steeldefect_segmenter", 7, 6)]
    assert [row["event"] for row in spark.rows()] == ["decision", "promotion"]
    assert capsys.readouterr().out.strip().splitlines()[-1] == '"models:/steeldefect_segmenter/7"'


def test_job_parameters_are_checked_before_any_table_is_touched(job, registry, config_path):
    # Databricks job parameters are strings: versions as digits, promote as true or false.
    assert job.as_version("7", "v") == 7 and job.as_version(np.int64(7), "v") == 7
    for bad in (0, -1, "7.0", "", "v7", "７", True, 7.0, None):     # ７: a fullwidth 7
        with pytest.raises(ValueError):
            job.as_version(bad, "model_b_version")
    assert job.as_flag("TRUE", "p") is True and job.as_flag("false", "p") is False
    for bad in ("yes", "0", "", None, 1):
        with pytest.raises(ValueError):
            job.as_flag(bad, "promote")
    for kwargs in ({"model_b_version": "6"}, {"promote": "nope"}, {"eval_dataset_id": ""},
                   {"model_name": " "}, {"registry_uri": ""}):
        spark = FakeSpark(_promote_frame())
        args = {"eval_dataset_id": "eval-1", "model_a_version": 6, "model_b_version": 7,
                "promote": True, "model_name": "steeldefect_segmenter", **kwargs}
        with pytest.raises(ValueError):
            job.main(spark, args.pop("eval_dataset_id"), args.pop("model_a_version"),
                     args.pop("model_b_version"), config_path=config_path, **args)
        assert spark.statements == [] and spark.queries == []
    assert registry["registry_uri"] is None


def test_the_dispatcher_passes_an_empty_name_on_to_be_refused(job, registry, config_path,
                                                             monkeypatch):
    # An empty model name is refused, not replaced by the short default name, which Unity
    # Catalog would complete with the session's catalog and schema.
    from steeldefect_platform import jobs
    spark = FakeSpark(_promote_frame())
    monkeypatch.setattr(jobs, "_spark", lambda: spark)
    for kwargs in ({"model_name": ""}, {"registry_uri": ""}):
        with pytest.raises(ValueError):
            jobs.job_ab_decision("eval-1", "6", "7", config_path=config_path, **kwargs)
    assert spark.statements == [] and spark.queries == [] and registry["registry_uri"] is None


def test_the_job_selects_the_registry_it_is_given(job, registry, config_path):
    spark = FakeSpark(_promote_frame())
    job.main(spark, "eval-1", A, B, config_path=config_path, registry_uri="databricks-uc")
    assert registry["registry_uri"] == "databricks-uc"
    assert json.loads(spark.rows()[0]["job_json"])["registry_uri"] == "databricks-uc"


def test_the_shipped_configuration_runs(job, registry):
    # The other tests skip the BCa bootstrap; the shipped config runs it, with mSPRT.
    spark = FakeSpark(_promote_frame())
    assert job.main(spark, "eval-1", A, B) == "models:/steeldefect_segmenter/7"
    row = spark.rows()[0]
    assert json.loads(row["job_json"])["cascade"]["bootstrap_n"] == 1000
    diagnostics = json.loads(row["diagnostics_json"])
    assert "bootstrap" in diagnostics and diagnostics["sequential"]["method"] == "msprt"


def test_the_job_imports_from_its_own_checkout(job, monkeypatch, tmp_path):
    # It used to put a hard-coded /Workspace/Repos path ahead of the deployed checkout.
    monkeypatch.delenv("STEELDEFECT_WORKSPACE_ROOT", raising=False)
    assert Path(job._repo_root()) == JOB.parents[2]
    monkeypatch.setenv("STEELDEFECT_WORKSPACE_ROOT", str(tmp_path))
    assert job._repo_root() == str(tmp_path)

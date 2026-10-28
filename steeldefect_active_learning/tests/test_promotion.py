"""Decision-level tests of the A/B promotion gate (``ab_testing/promotion.py``).

The expected values come from an independent reference implementation of the same
design; the thresholds are written here rather than read from the YAML, so that the gate
is tested independently of the config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from steeldefect_active_learning.ab_testing import paired
from steeldefect_active_learning.ab_testing.metrics import compute_all_metrics
from steeldefect_active_learning.ab_testing.promotion import decide_promotion, gate_and_register

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"
PRIOR = {0: 0.118, 1: 0.031, 2: 0.018, 3: 0.0061, 4: 0.024, 5: 0.0025, 6: 0.043, 7: 0.071,
         8: 0.0039, 9: 0.0125, 10: 0.610}
SHIPPED = dict(alpha=0.05, min_gain=0.0, noninferiority_margin=0.05,
               class_margins={3: 0.03, 5: 0.03, 8: 0.03}, expected_classes=list(range(11)),
               class_prior=PRIOR, false_alarm_margin=0.005,
               class_false_alarm_margins={3: 0.002, 5: 0.002, 8: 0.002},
               min_detectable_gain=0.02, planning_discordance=0.10, power=0.80,
               require_no_class_regression=True)
SLICE = {**SHIPPED, "class_prior": "slice"}
ALL_CRITERIA = ["valid_input", "sample_size", "superiority", "no_class_regression",
                "no_false_alarm_increase"]


def frame_from_tables(tables: Dict[int, Tuple[int, int, int, int]],
                      wrong_a: Optional[Callable[[int], int]] = None,
                      wrong_b: Optional[Callable[[int], int]] = None,
                      n_classes: int = 11) -> pd.DataFrame:
    """A prediction frame realising per-class cells (n11, n10, n01, n00).

    On a class-k image, A's wrong prediction is ``wrong_a(k)`` (default (k + 1) % 11) and
    B's own wrong prediction ``wrong_b(k)`` (default (k + 2) % 11); on images both get
    wrong, B repeats A's prediction.
    """
    wa = wrong_a or (lambda k: (k + 1) % n_classes)
    wb = wrong_b or (lambda k: (k + 2) % n_classes)
    rows = []
    for k, (n11, n10, n01, n00) in tables.items():
        rows += [(k, k, k)] * n11 + [(k, k, wb(k))] * n10
        rows += [(k, wa(k), k)] * n01 + [(k, wa(k), wa(k))] * n00
    df = pd.DataFrame(rows, columns=["label", "pred_a", "pred_b"])
    df.insert(0, "image_path", [f"img_{j:06d}.png" for j in range(len(df))])
    return df


def verdict_of(df: pd.DataFrame, **thresholds) -> dict:
    return decide_promotion({"paired_counts": paired.paired_counts(df)}, **thresholds)


def passed(verdict: dict) -> Dict[str, bool]:
    return {r["criterion"]: r["passed"] for r in verdict["reasons"]}


def promote_frame() -> pd.DataFrame:
    """B fixes 36 of A's 57 errors per defect class and breaks 3; clean gains 90 net."""
    tables = {k: (240, 3, 36, 21) for k in range(10)}
    tables[10] = (1450, 10, 100, 40)
    return frame_from_tables(tables)


def _skip_coil_break(k: int) -> int:
    j = (k + 1) % 11
    return 6 if j == 5 else j


def _g3_frame(concentrated: bool) -> pd.DataFrame:
    """Every clause but the false alarms passes; B's 40 new errors predict coil_break or not."""
    tables = {5: (100, 0, 0, 0)}
    for k in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
        tables[k] = (266, 4, 20, 10)

    def spread(k: int) -> int:
        j = (k + 2) % 11
        while j in (5, k):
            j = (j + 1) % 11
        return j

    return frame_from_tables(tables, wrong_a=_skip_coil_break,
                             wrong_b=(lambda k: 5) if concentrated else spread)


def _g2_tables(clean: Tuple[int, int, int, int]) -> Dict[int, Tuple[int, int, int, int]]:
    """B is better on the rare classes but worse on clean and on the frequent defects."""
    tables = {}
    for k in range(11):
        if k == 10:
            tables[k] = clean
        elif k in (0, 6, 7):
            tables[k] = (520, 20, 10, 50)
        else:
            tables[k] = (462, 18, 78, 42)
    return tables


def _f10_tables(fixes_in_class_0: int) -> Dict[int, Tuple[int, int, int, int]]:
    """B fixes 44 of each class's A errors (fewer in class 0) and breaks none."""
    ns, errors_a = [334, 334, 333, 333, 333, 333], [98, 98, 98, 97, 97, 97]
    tables = {k: (ns[k] - errors_a[k], 0, 44, errors_a[k] - 44) for k in range(6)}
    tables[0] = (ns[0] - errors_a[0], 0, fixes_in_class_0, errors_a[0] - fixes_in_class_0)
    return tables


def _fixture_frame() -> pd.DataFrame:
    """The ab_df_classification fixture of conftest.py (rng 42)."""
    rng = np.random.default_rng(42)
    n, k = 500, 11
    labels = rng.integers(0, k, size=n)
    pred_a = labels.copy()
    flip_a = rng.random(n) < 0.20
    pred_a[flip_a] = (pred_a[flip_a] + rng.integers(1, k, size=int(flip_a.sum()))) % k
    pred_b = labels.copy()
    flip_b = rng.random(n) < 0.15
    pred_b[flip_b] = (pred_b[flip_b] + rng.integers(1, k, size=int(flip_b.sum()))) % k
    return pd.DataFrame({"image_path": [f"img_{i:04d}.png" for i in range(n)],
                         "label": labels, "pred_a": pred_a, "pred_b": pred_b})


F2_BASE = {10: (1319, 0, 60, 121), **{k: (80, 0, 0, 20) for k in [0, 1, 2, 3, 4, 6, 7, 8, 9]}}
SIX = {**SHIPPED, "expected_classes": list(range(6)), "class_prior": "slice",
       "class_margins": {}, "class_false_alarm_margins": {}}

# (scenario, frame, thresholds, decision, conclusion, failing criteria, key values)
SCENARIOS = [
    ("the demo at its own class mix", lambda: pd.read_csv(DEMO), {**SLICE, "class_margins": {}},
     "keep_A", "inconclusive", ["sample_size", "superiority", "no_class_regression"],
     {"delta_lower": -0.006473660240595499, "n_effective": 1100.0}),
    ("F10: B fixes 264 errors, breaks none", lambda: frame_from_tables(_f10_tables(44), n_classes=6),
     SIX, "promote_B", "promote_B", [],
     {"mcnemar_p_one_sided": 3.3735033418337674e-80, "delta_lower": 0.11956656692836817,
      "n_effective": 2000.0}),
    ("F10: the same B with 259 fixes", lambda: frame_from_tables(_f10_tables(39), n_classes=6),
     SIX, "promote_B", "promote_B", [], {"delta_lower": 0.11717067143983767}),
    ("F2: coil_break absent", lambda: frame_from_tables(F2_BASE),
     {**SLICE, "class_margins": {}}, "keep_A", "inconclusive", ["no_class_regression"],
     {"delta_lower": 0.019894889394964883}),
    ("F2: coil_break loses its 5 images", lambda: frame_from_tables({**F2_BASE, 5: (0, 5, 0, 0)}),
     SLICE, "keep_A", "evidence_against_B", ["no_class_regression"], {}),
    ("G3: 40 new false coil_break alarms", lambda: _g3_frame(concentrated=True),
     {**SLICE, "class_margins": {}}, "keep_A", "evidence_against_B", ["no_false_alarm_increase"],
     {}),
    ("G3: the same errors spread over classes", lambda: _g3_frame(concentrated=False),
     {**SLICE, "class_margins": {}}, "promote_B", "promote_B", [], {}),
    ("G2: worse at the prior, 600 per class", lambda: frame_from_tables(_g2_tables((573, 15, 3, 9))),
     SHIPPED, "keep_A", "inconclusive", ["sample_size", "superiority"],
     {"mcnemar_p_one_sided": 1.5664053436817189e-43, "estimand_lower": -0.01706059052548682,
      "estimand_upper": 0.0020155235340032305, "n_effective": 1342.2114670673116}),
    ("G2: same slice, certified at its own mix", lambda: frame_from_tables(_g2_tables((573, 15, 3, 9))),
     SLICE, "promote_B", "promote_B", [], {"estimand_lower": 0.05031793413399098}),
    ("G2: worse at the prior, 1,200 clean", lambda: frame_from_tables(_g2_tables((1146, 30, 6, 18))),
     SHIPPED, "keep_A", "evidence_against_B", ["superiority"],
     {"estimand_delta": -0.006666666666666666, "estimand_lower": -0.01380884252193531,
      "estimand_upper": -0.00041405537969812275, "n_effective": 2537.3801234345265}),
    ("G2: 1,200 clean, slice mix", lambda: frame_from_tables(_g2_tables((1146, 30, 6, 18))),
     SLICE, "promote_B", "promote_B", [], {"estimand_lower": 0.0443694498743829}),
    ("conftest fixture: 500 images", _fixture_frame, SHIPPED, "keep_A", "inconclusive",
     ["sample_size", "superiority", "no_class_regression"],
     {"mcnemar_p_one_sided": 0.0021854690363093568, "estimand_lower": -0.09525705667925985}),
    ("B clearly better everywhere", promote_frame, SHIPPED, "promote_B", "promote_B", [],
     {"mcnemar_p_one_sided": 7.505686185602336e-92, "estimand_delta": 0.07511968085106382,
      "estimand_lower": 0.0663642659874471, "n_effective": 2860.8224141356663}),
    ("the same with A and B swapped",
     lambda: promote_frame().rename(columns={"pred_a": "pred_b", "pred_b": "pred_a"}),
     SHIPPED, "keep_A", "evidence_against_B",
     ["superiority", "no_class_regression", "no_false_alarm_increase"], {}),
    ("B identical to A", lambda: promote_frame().assign(pred_b=lambda d: d["pred_a"]),
     SHIPPED, "keep_A", "inconclusive", ["superiority"],
     {"mcnemar_p_one_sided": 1.0, "estimand_lower": -0.0029947420458672092,
      "p_b_better_posterior": 0.5}),
]


@pytest.mark.parametrize("name, make, thresholds, decision, conclusion, failing, values",
                         SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_decision_scenarios(name, make, thresholds, decision, conclusion, failing, values):
    v = verdict_of(make(), **thresholds)
    assert (v["decision"], v["conclusion"]) == (decision, conclusion)
    assert sorted(c for c, ok in passed(v).items() if not ok) == sorted(failing)
    numbers = {**v["estimates"], **v["plan"]}
    for key, expected in values.items():
        assert numbers[key] == pytest.approx(expected, rel=1e-6, abs=0), key


def test_demo_slice_cannot_promote_by_design():
    v = verdict_of(pd.read_csv(DEMO), **SHIPPED)
    assert (v["decision"], v["conclusion"]) == ("keep_A", "inconclusive")
    status = {r["class"]: r["status"] for r in v["per_class"]}
    assert [k for k, s in status.items() if s == "inconclusive"] == [0, 1, 4, 6, 7, 10]
    assert status[5] == "insufficient_n" and [k for k, s in status.items() if s == "pass"] == [2, 3, 8, 9]
    assert v["plan"]["n_effective"] == pytest.approx(223.70191117788525)
    assert v["estimates"]["mcnemar_p_one_sided"] == pytest.approx(0.10680621141867858)
    assert v["vetoed_classes"] == []
    details = {r["criterion"]: r["detail"] for r in v["reasons"]}
    assert "coil_break: n = 100 < 147" in details["no_class_regression"]


def test_a_regressed_rare_class_is_named():
    v = verdict_of(frame_from_tables({**F2_BASE, 5: (0, 5, 0, 0)}), **SLICE)
    row = next(r for r in v["per_class"] if r["class"] == 5)
    assert (row["status"], row["lower"]) == ("regressed", -1.0)
    assert row["upper"] == pytest.approx(-0.066003, abs=5e-7)


def test_concentrated_false_alarms_are_vetoed():
    v = verdict_of(_g3_frame(concentrated=True), **{**SLICE, "class_margins": {}})
    assert v["vetoed_classes"] == [5]
    row = next(r for r in v["per_predicted_class"] if r["class"] == 5)
    assert (row["only_a_alarm"], row["only_b_alarm"], row["n_other"]) == (0, 40, 3000)
    assert row["fa_lower"] == pytest.approx(0.010010432526210266, abs=1e-12)


def test_swap_never_promotes_both_ways():
    rng = np.random.default_rng(20261028)
    promotions = 0
    for _ in range(200):
        tables = {}
        for k in range(11):
            n = int(rng.integers(150, 600))
            n10, n01 = (int(x) for x in rng.multinomial(int(0.12 * n), [0.35, 0.65]))
            n00 = int(rng.integers(0, n - n10 - n01))
            tables[k] = (n - n10 - n01 - n00, n10, n01, n00)
        df = frame_from_tables(tables)
        forward = verdict_of(df, **SLICE)["decision"]
        backward = verdict_of(df.rename(columns={"pred_a": "pred_b", "pred_b": "pred_a"}), **SLICE)["decision"]
        assert not (forward == backward == "promote_B")
        promotions += forward == "promote_B"
    assert promotions > 0   # the property is not vacuous


def test_row_order_index_and_json_do_not_change_the_verdict():
    df = promote_frame()
    ref = verdict_of(df, **SHIPPED)
    shuffled = df.sample(frac=1.0, random_state=11).set_index("image_path", drop=False)
    assert verdict_of(shuffled, **SHIPPED) == ref
    cascade = json.loads(json.dumps({"paired_counts": paired.paired_counts(df)}))
    assert decide_promotion(cascade, **SHIPPED) == ref


def test_verdict_is_strict_json():
    for make, thresholds in [(promote_frame, SHIPPED), (_fixture_frame, SHIPPED),
                             (lambda: frame_from_tables(F2_BASE), SLICE)]:
        v = verdict_of(make(), **thresholds)
        assert json.loads(json.dumps(v, allow_nan=False)) == v


def _assert_invalid(v: dict, criteria=ALL_CRITERIA) -> None:
    assert (v["decision"], v["conclusion"], v["gate_passed"]) == ("keep_A", "invalid_input", False)
    assert [r["criterion"] for r in v["reasons"]] == criteria
    assert not any(r["passed"] for r in v["reasons"])
    assert all(r["detail"] == "not evaluated: invalid input" for r in v["reasons"][1:])


def test_missing_or_invalid_evidence_keeps_a():
    good = paired.paired_counts(promote_frame())
    bad_cells = {**good, "only_b_correct": good["only_b_correct"] + 1}
    for cascade in [{}, {"paired_counts": None}, {"paired_counts": bad_cells},
                    {"statistical": {"significant": True, "p_value": 1e-9, "cohens_d": 2.0},
                     "mcnemar": {"p_value": 1e-9, "b": 0, "c": 100}},
                    {"paired_counts": paired.paired_counts(pd.concat([promote_frame()] * 2))}]:
        _assert_invalid(decide_promotion(cascade, **SHIPPED))


def test_contract_violations_keep_a():
    df = promote_frame()
    frames = [
        df.assign(arm="a"),
        df.drop(columns=["image_path"]),
        df.assign(label=df["label"].astype(float).where(df["label"] != 5)),
        df.assign(pred_b=df["pred_b"].astype(float).where(df.index % 50 != 0)),
    ]
    for frame in frames:
        _assert_invalid(verdict_of(frame, **SHIPPED))


def test_unconfigured_gate_keeps_a():
    df = promote_frame()
    v = verdict_of(df, **{**SHIPPED, "expected_classes": None, "class_prior": "slice"})
    _assert_invalid(v)
    assert "expected_classes is not configured" in v["reasons"][0]["detail"]
    v = verdict_of(df, **{**SHIPPED, "class_prior": None})
    _assert_invalid(v)
    assert "class_prior is not configured" in v["reasons"][0]["detail"]


def test_classes_outside_the_taxonomy_keep_a():
    v = verdict_of(promote_frame(), **{**SHIPPED, "expected_classes": list(range(10)),
                                       "class_prior": "slice"})
    _assert_invalid(v)
    assert "classes outside expected_classes in the slice: [10]" in v["reasons"][0]["detail"]


def test_gate_ignores_the_agreement_block():
    # The v1 gate fell back to bayesian['agreement'] when accuracy was absent.
    cascade = {"bayesian": {"agreement": {"p_b_gt_a": 1.0}}}
    _assert_invalid(decide_promotion(cascade, **SHIPPED))


def _correctness_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(correct_a=(df["pred_a"] == df["label"]).astype(int),
                     correct_b=(df["pred_b"] == df["label"]).astype(int)).drop(columns=["pred_a", "pred_b"])


def test_false_alarm_clause_needs_predictions():
    v = verdict_of(_correctness_only(promote_frame()), **SHIPPED)
    _assert_invalid(v)
    assert "the false-alarm clause needs pred_a and pred_b" in v["reasons"][0]["detail"]


def test_false_alarm_clause_can_be_disabled():
    v = verdict_of(_correctness_only(promote_frame()), **{**SHIPPED, "false_alarm_margin": None})
    assert v["decision"] == "promote_B"
    assert v["disabled_criteria"] == ["no_false_alarm_increase"]
    assert [r["criterion"] for r in v["reasons"]] == ALL_CRITERIA[:4]


def test_class_clause_can_be_disabled():
    v = verdict_of(frame_from_tables(F2_BASE), **{**SLICE, "class_margins": {},
                                                  "require_no_class_regression": False})
    assert v["decision"] == "promote_B"
    assert v["disabled_criteria"] == ["no_class_regression"]
    assert v["failing_classes"] == [5]      # still reported


def test_a_threshold_changes_the_verdict():
    v = verdict_of(promote_frame(), **{**SHIPPED, "min_gain": 0.10, "min_detectable_gain": 0.12,
                                       "planning_discordance": 0.20})
    assert v["decision"] == "keep_A"
    detail = {r["criterion"]: r["detail"] for r in v["reasons"]}["superiority"]
    assert "deployment_prior lower bound +0.0664 <= 0.1" in detail


def test_detail_strings_state_the_comparison_and_its_outcome():
    cases = [(promote_frame(), SHIPPED), (_fixture_frame(), SHIPPED),
             (_g3_frame(concentrated=True), {**SLICE, "class_margins": {}}),
             (promote_frame().assign(arm="a"), SHIPPED)]
    for df, thresholds in cases:
        for r in verdict_of(df, **thresholds)["reasons"]:
            if r["detail"] != "not evaluated: invalid input":
                assert r["detail"].endswith(": pass" if r["passed"] else ": fail"), r
    v = verdict_of(promote_frame(), **SHIPPED)
    detail = {r["criterion"]: r["detail"] for r in v["reasons"]}
    assert detail["superiority"].startswith("one-sided exact McNemar p = 7.506e-92 < 0.05; "
                                            "deployment_prior lower bound +0.0664 > 0.0")
    assert detail["sample_size"].startswith("n_effective = 2860.8 >= n_planned = 1544")
    v = verdict_of(_fixture_frame(), **SHIPPED)
    detail = {r["criterion"]: r["detail"] for r in v["reasons"]}
    assert "lower bound -0.0953 <= 0.0" in detail["superiority"]
    # Nine classes fail: five are listed, then the list is marked as cut.
    assert detail["no_class_regression"] == (
        "9 of 11 classes fail: 0: n = 40 < 87; 1: n = 58 < 87; 3: n = 48 < 147; "
        "4: n = 49 < 87; 5: n = 48 < 147; ...: fail")


@pytest.mark.parametrize("override", [
    {"alpha": 0.6}, {"noninferiority_margin": 0.0}, {"power": 0.3},
    {"min_gain": 0.02, "min_detectable_gain": 0.02}, {"planning_discordance": 0.01},
    {"class_prior": {0: 1.0}}, {"class_margins": {11: 0.03}}, {"expected_classes": [0, 0, 1]},
    {"require_no_class_regression": "yes"}, {"false_alarm_margin": 1.5},
])
def test_out_of_range_thresholds_raise(override):
    with pytest.raises(ValueError):
        verdict_of(promote_frame(), **{**SHIPPED, **override})


@pytest.mark.parametrize("retired", ["min_cohens_d", "bayesian_threshold"])
def test_retired_keywords_raise_typeerror(retired):
    with pytest.raises(TypeError):
        decide_promotion({}, **{retired: 0.5})


def test_end_to_end_from_compute_all_metrics():
    df = promote_frame()
    rng = np.random.default_rng(0)
    df = df.assign(confidence_a=rng.uniform(0.5, 1.0, len(df)), confidence_b=rng.uniform(0.5, 1.0, len(df)))
    v = decide_promotion(compute_all_metrics(df), **SHIPPED)
    assert v["decision"] == "promote_B" and v["n_images"] == len(df)


def test_gate_and_register_forwards_thresholds(monkeypatch):
    import steeldefect_platform.mlflow_registry as registry
    calls = []
    monkeypatch.setattr(registry, "promote", lambda *args, **kwargs: calls.append((args, kwargs)))
    cascade = {"paired_counts": paired.paired_counts(promote_frame())}

    keep = gate_and_register({"paired_counts": paired.paired_counts(_fixture_frame())},
                             "models:/steeldefect_segmenter/8", model_b_version=8, **SHIPPED)
    assert keep["decision"] == "keep_A" and keep["promoted_model_uri"] is None and calls == []

    v = gate_and_register(cascade, "models:/steeldefect_segmenter/7", model_b_version=7, **SHIPPED)
    assert v["decision"] == "promote_B"
    assert calls == [(("steeldefect_segmenter", 7), {"to_stage": "Production"})]
    assert v["promoted_model_uri"] == "models:/steeldefect_segmenter/7"
    assert v["thresholds"]["alpha"] == 0.05 and v["thresholds"]["class_prior"][0] == {"class": 0, "weight": 0.118}

    with pytest.raises(TypeError):
        gate_and_register(cascade, "models:/x/1", model_b_version=1, min_cohen_d=0.2)


def test_gate_and_register_without_thresholds_keeps_a(monkeypatch):
    # Unconfigured (no taxonomy, no prior): fail-closed, and the registry is not called.
    import steeldefect_platform.mlflow_registry as registry
    calls = []
    monkeypatch.setattr(registry, "promote", lambda *args, **kwargs: calls.append(args))
    v = gate_and_register({"paired_counts": paired.paired_counts(promote_frame())},
                          "models:/steeldefect_segmenter/7", model_b_version=7)
    assert v["decision"] == "keep_A" and calls == []

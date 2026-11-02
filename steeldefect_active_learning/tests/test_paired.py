"""Reference values for the paired-counts contract and the statistics of the promotion gate.

Every expected value was obtained by two independent computations (exact rational
arithmetic, a 50-digit mpmath implementation, or a published value) before being copied
here. Tables are written (n11, n10, n01, n00): both correct, only A correct, only B
correct, both wrong.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from steeldefect_active_learning.ab_testing import paired

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo_ab_predictions.csv"
Z95 = float(stats.norm.isf(0.05))


@pytest.fixture(scope="module")
def demo():
    """The tracked synthetic A/B predictions (1,100 rows, 11 classes, 100 each)."""
    return pd.read_csv(DEMO)


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n10, n01, expected", [
    (5, 14, 0.0317840576171875),       # 2083 / 2^16
    (12, 24, 0.032622667611576624),
    (2, 8, 0.0546875),                 # 7 / 128
    (0, 5, 0.03125),                   # 1 / 32
    (0, 4, 0.0625),                    # 1 / 16
    (5, 0, 1.0),
    (0, 0, 1.0),
    (3, 20, 0.000244140625),           # 1 / 4096
    (20, 65, 5.149437156259905e-07),
    (145, 168, 0.10680621141867858),   # the demo slice
    (0, 80, 8.271806125530277e-25),    # 2^-80: 1 - cdf would give 0
    (40, 100, 2.0716982159402628e-07),
    (1, 9, 0.0107421875),              # 11 / 1024
    (0, 1000, 9.332636185032189e-302), # 2^-1000
])
def test_mcnemar_one_sided_reference(n10, n01, expected):
    # abs=0: pytest.approx would otherwise accept any value within 1e-12, 0.0 included.
    p = paired.mcnemar_one_sided_p(n10, n01)
    assert p == pytest.approx(expected, rel=1e-12, abs=0)
    if n10 + n01 > 0:
        assert p == pytest.approx(
            stats.binomtest(n01, n10 + n01, 0.5, alternative="greater").pvalue, rel=1e-12, abs=0)


def test_mcnemar_keeps_the_atom_at_n01():
    # P(X >= 8) for X ~ Bin(10, 1/2) is 7/128; P(X > 8) would be 11/1024 and promote.
    assert paired.mcnemar_one_sided_p(2, 8) == 7 / 128


@pytest.mark.parametrize("n10, n01, test_used, p_value", [
    (0, 0, "mcnemar_degenerate", 1.0),
    (5, 14, "mcnemar_exact", 0.063568115234375),
    (12, 24, "mcnemar_yates", float(stats.chi2.sf(121 / 36, 1))),
])
def test_mcnemar_two_sided(n10, n01, test_used, p_value):
    out = paired.mcnemar_two_sided(n10, n01)
    assert out["test_used"] == test_used
    assert out["p_value"] == pytest.approx(p_value, rel=1e-12)
    assert (out["b"], out["c"], out["n_discordant"]) == (n10, n01, n10 + n01)
    assert out["alternative"] == "two-sided"


def test_mcnemar_two_sided_does_not_underflow():
    # Yates chi-square (80 - 1)^2 / 80 = 78.0125: 1 - cdf rounds the p-value to 0.
    assert paired.mcnemar_two_sided(0, 80)["p_value"] == pytest.approx(1.0239e-18, rel=1e-4, abs=0)


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n10, n01, expected", [
    (293, 325, 0.9008310969459695),
    (0, 20, 0.9999995231628418),
    (8, 30, 0.9998529615622829),
    (145, 168, 0.9028907029671362),
    (3, 10, 0.9713134765625),
    (5, 14, 0.9793052673339844),
    (0, 0, 0.5),
    (20, 0, 4.76837158203125e-07),     # 2^-21
    (60, 0, 4.336808689942018e-19),    # 2^-61: 1 - cdf would give 0
])
def test_posterior_reference(n10, n01, expected):
    assert paired.posterior_b_better(n10, n01) == pytest.approx(expected, rel=1e-12, abs=0)


def test_posterior_is_one_minus_mid_p():
    # 1 - mid-p written without the cancellation of 1 - (...), so that the tail is checked.
    for n10 in range(0, 61, 3):
        for n01 in range(0, 61, 3):
            m = n10 + n01
            one_minus_mid_p = 0.5 if m == 0 else (stats.binom.cdf(n01 - 1, m, 0.5)
                                                  + 0.5 * stats.binom.pmf(n01, m, 0.5))
            assert paired.posterior_b_better(n10, n01) == pytest.approx(one_minus_mid_p,
                                                                        rel=1e-10, abs=0)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

# Two-sided limits at level 1 - 2a are the one-sided bounds at a. The published values
# are Tango (1998) Table I and sections 6.1-6.2, and the examples of the R packages
# contingencytables and ratesci.
@pytest.mark.parametrize("table, level, lower, upper, published", [
    ((27, 1, 0, 2), 0.90, -0.13640414215181745, 0.05214845172926899, (-0.136, 0.052)),
    ((78, 2, 0, 0), 0.90, -0.07276951853259757, 0.008530787957242304, (-0.073, 0.009)),
    ((43, 1, 0, 0), 0.90, -0.0956624263991715, 0.03651693841806421, (-0.096, 0.037)),
    ((4, 3, 9, 16), 0.95, -0.027090464503980602, 0.3896974871387835, (-0.027, 0.390)),
    ((59, 16, 6, 80), 0.95, -0.12402343246406822, -0.005433395731676817, (-0.1240, -0.0054)),
    ((53, 8, 16, 9), 0.95, -0.019658922719051238, 0.20691077062200097, (-0.020, 0.207)),
    ((1, 7, 1, 12), 0.95, -0.5172322764213775, -0.02600303248086845, (-0.517, -0.026)),
    ((8, 1, 3, 2), 0.95, -0.16697451229843152, 0.4326638516980377, (-0.1670, 0.4327)),
    ((7, 2, 25, 68), 0.95, 0.13918254632705251, 0.3213297505918714, (0.139, 0.321)),
])
def test_plain_bounds_reproduce_published_values(table, level, lower, upper, published):
    n11, n10, n01, n00 = table
    n, a = sum(table), (1 - level) / 2
    got = (paired.score_lower_bound(n10, n01, n, a, cc=0),
           paired.score_upper_bound(n10, n01, n, a, cc=0))
    assert got == pytest.approx((lower, upper), abs=1e-9)
    for value, pub in zip(got, published):
        digits = len(str(pub).split(".")[1])
        assert abs(value - pub) <= 0.5 * 10 ** -digits + 1e-12


def test_plain_statistic_reproduces_published_values():
    # Tango (1998) section 6.1 prints 1.709 (one-sided p 0.044), and Table I 1.89.
    z = paired.score_statistic(1, 0, 44, -0.10, cc=0)
    assert z == pytest.approx(1.7085642859406607, abs=1e-12)
    assert stats.norm.sf(z) == pytest.approx(0.043766, abs=5e-7)
    assert paired.score_statistic(1, 0, 50, -0.10, cc=0) == pytest.approx(1.8856180831641265, abs=1e-12)


@pytest.mark.parametrize("n", [30, 51, 52, 100])
def test_zero_discordance_closed_form(n):
    # Tango's eq. 34: without discordant pairs the plain bound is -z^2 / (n + z^2).
    assert paired.score_lower_bound(0, 0, n, 0.05, cc=0) == pytest.approx(
        -Z95 ** 2 / (n + Z95 ** 2), abs=1e-12)


@pytest.mark.parametrize("table, lower", [
    ((40, 0, 0, 0), -0.1045885669617105),
    ((27, 1, 0, 2), -0.1827155628750877),
    ((90, 3, 7, 0), -0.025118751107425397),
    ((85, 10, 3, 2), -0.1466586049272644),
    ((45, 0, 5, 0), 0.009793466545160199),
    ((59, 16, 6, 80), -0.11992532090221864),
    ((2000, 60, 100, 240), 0.0076487785879968545),
    ((4, 3, 9, 16), -0.023575718887112095),
    ((96, 4, 0, 0), -0.09916108527905315),
    ((95, 5, 0, 0), -0.11184096522645981),
    ((94, 6, 0, 0), -0.12426011306679302),
    ((753, 145, 168, 34), -0.006473660240595499),   # the demo slice
    ((77, 23, 10, 0), -0.213019759832848),
    ((0, 0, 3, 0), -0.25227413784175057),
    ((4, 0, 4, 0), -0.04445899988705318),
])
def test_adjusted_lower_bound_reference(table, lower):
    n11, n10, n01, n00 = table
    assert paired.score_lower_bound(n10, n01, sum(table), 0.05) == pytest.approx(lower, abs=1e-9)


@pytest.mark.parametrize("table, upper", [
    ((272, 8, 0, 0), -0.0124362647017758),
    ((753, 145, 168, 34), 0.04831811056364971),
    ((0, 3, 0, 0), 0.25227413784175057),
])
def test_adjusted_upper_bound_reference(table, upper):
    n11, n10, n01, n00 = table
    assert paired.score_upper_bound(n10, n01, sum(table), 0.05) == pytest.approx(upper, abs=1e-9)


@pytest.mark.parametrize("f10, f01, n_other, cc, lower", [
    (0, 40, 2140, 0, 0.014448591380416088),
    (11, 24, 1000, 0, 0.0034000772893586885),
    (11, 24, 1000, 1, 0.002368816908858995),    # demo, streak: below the 0.005 margin
    (0, 40, 3000, 1, 0.010010432526210266),     # 40 new false coil_break alarms
    (0, 40, 2140, 1, 0.014040748828198666),
])
def test_false_alarm_bound_reference(f10, f01, n_other, cc, lower):
    assert paired.score_lower_bound(f10, f01, n_other, 0.05, cc=cc) == pytest.approx(lower, abs=1e-9)


def test_upper_bound_is_the_lower_bound_with_a_and_b_swapped():
    for n10, n01, n in [(3, 7, 100), (0, 0, 40), (16, 6, 161), (145, 168, 1100)]:
        assert paired.score_upper_bound(n10, n01, n, 0.05) == -paired.score_lower_bound(n01, n10, n, 0.05)


def test_swapping_a_and_b_changes_the_bound():
    # Calling with the arms swapped on (27, 1, 0, 2) returns minus the upper limit.
    assert paired.score_lower_bound(0, 1, 30, 0.05, cc=0) == pytest.approx(-0.05214845172926899, abs=1e-9)


@pytest.mark.parametrize("n10, n01, n, statistic, passes", [
    (0, 0, 87, 1.647929, True),     # lower bound -0.049887
    (0, 0, 86, 1.632743, False),    # -0.050448
    (1, 0, 87, 1.156010, False),    # -0.067114
    (1, 0, 86, 1.137973, False),    # -0.067867
])
def test_margin_decisions(n10, n01, n, statistic, passes):
    assert paired.score_statistic(n10, n01, n, -0.05) == pytest.approx(statistic, abs=5e-7)
    assert (paired.score_lower_bound(n10, n01, n, 0.05) > -0.05) is passes


def test_bound_and_test_decide_alike():
    # L > d0 exactly when the score test rejects H0: Delta <= d0, on every table up to n = 30.
    for n in range(1, 31):
        for n10 in range(n + 1):
            for n01 in range(n + 1 - n10):
                for margin in (0.02, 0.05, 0.10):
                    for cc in (0, 1):
                        rejects = paired.score_statistic(n10, n01, n, -margin, cc=cc) > Z95
                        assert rejects == (paired.score_lower_bound(n10, n01, n, 0.05, cc=cc) > -margin)


def test_inconsistent_tables_raise():
    with pytest.raises(ValueError):
        paired.score_lower_bound(0, 0, 0, 0.05)
    with pytest.raises(ValueError):
        paired.score_lower_bound(6, 5, 10, 0.05)
    with pytest.raises(ValueError):
        paired.score_statistic(-1, 0, 10, 0.0)


@pytest.mark.parametrize("margin, cc, floor", [
    (0.10, 1, 42), (0.05, 1, 87), (0.03, 1, 147), (0.02, 1, 222), (0.01, 1, 446),
    (0.10, 0, 25), (0.05, 0, 52), (0.02, 0, 133),
])
def test_noninferiority_floor(margin, cc, floor):
    assert paired.noninferiority_floor(margin, 0.05, cc=cc) == floor
    assert paired.score_lower_bound(0, 0, floor - 1, 0.05, cc=cc) <= -margin
    assert paired.score_lower_bound(0, 0, floor, 0.05, cc=cc) > -margin


@pytest.mark.parametrize("margin, alpha, floor", [
    (0.05, 0.025, 110), (0.03, 0.025, 185), (0.05, 0.01, 140), (0.03, 0.01, 237),
])
def test_noninferiority_floor_at_other_levels(margin, alpha, floor):
    assert paired.noninferiority_floor(margin, alpha) == floor
    assert paired.score_lower_bound(0, 0, floor - 1, alpha) <= -margin
    assert paired.score_lower_bound(0, 0, floor, alpha) > -margin


@pytest.mark.parametrize("alpha", [0.0, -0.1, 0.6, 1.0])
def test_bounds_need_a_one_sided_level(alpha):
    # The bound search assumes z_(1-alpha) >= 0; above 0.5 it returned its bracket's end.
    for call in (lambda: paired.score_lower_bound(3, 9, 32, alpha),
                 lambda: paired.score_upper_bound(3, 9, 32, alpha),
                 lambda: paired.noninferiority_floor(0.05, alpha),
                 lambda: paired.planning_n(0.10, 0.02, 0.0, alpha, 0.80)):
        with pytest.raises(ValueError, match="alpha must be in"):
            call()
    assert paired.score_lower_bound(3, 9, 32, 0.5) == pytest.approx(5 / 32, abs=1e-12)


def _first_promoting_count(n: int, gain: float, alpha: float) -> int:
    """Smallest n01 at n10 = 0 that passes both halves of the superiority clause."""
    def passes(n01: int) -> bool:
        return (paired.score_lower_bound(0, n01, n, alpha) > gain
                and paired.mcnemar_one_sided_p(0, n01) < alpha)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _superiority_size(n: int, gain: float, alpha: float) -> float:
    # Worst null found: B never loses and gains exactly min_gain (p10 = 0, p01 = gain).
    return float(stats.binom.sf(_first_promoting_count(n, gain, alpha) - 1, n, gain))


def test_superiority_size_at_the_smallest_alpha():
    sizes = [_superiority_size(n, g, 0.025)
             for g in (0.005, 0.01, 0.02, 0.05) for n in range(100, 6001, 211)]
    assert max(sizes) <= 0.025
    assert max(sizes) > 0.9 * 0.025      # the grid reaches the worst cases


def test_superiority_size_exceeds_alpha_below_the_range():
    # Why alpha >= 0.025: at 0.01 the exact size reaches 1.116 alpha (n = 832, gain 0.01:
    # n01 = 16 passes, Z = 2.32754 > z = 2.32635; P(Bin(832, 0.01) >= 16) in exact rationals).
    assert _first_promoting_count(832, 0.01, 0.01) == 16
    assert _superiority_size(832, 0.01, 0.01) == pytest.approx(0.011157954115420666, rel=1e-9)


# ---------------------------------------------------------------------------
# Planning, weighted difference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args, n", [
    ((0.10, 0.02, 0.0, 0.05, 0.80), 1544),    # Connor (1987): 1543.55
    ((0.05, 0.02, 0.0, 0.05, 0.80), 771),
    ((0.10, 0.02, 0.0, 0.05, 0.90), 2138),
    ((0.05, 0.0, -0.05, 0.05, 0.80), 167),
    ((0.05, 0.0, -0.10, 0.05, 0.80), 56),
    ((0.10, 0.0, -0.05, 0.05, 0.80), 274),
    ((0.05, 0.0, -0.05, 0.05, 0.8 ** (1 / 11)), 336),
    ((0.05, 0.0, -0.10, 0.05, 0.8 ** (1 / 11)), 104),
    ((0.05, 0.0, -0.02, 0.05, 0.80), 837),
    ((0.10, 0.02, -0.05, 0.05, 0.8 ** (1 / 11)), 306),
    ((0.10, 0.02, -0.03, 0.05, 0.8 ** (1 / 11)), 573),
    ((0.02, 0.02, 0.0, 0.05, 0.80), 308),     # psi = delta1: B never loses, still feasible
    ((0.10, 0.02, 0.0, 0.025, 0.80), 1960),
    ((0.10, 0.02, 0.0, 0.01, 0.80), 2507),
])
def test_planning_n(args, n):
    assert paired.planning_n(*args) == n


@pytest.mark.parametrize("alpha", [0.025, 0.05, 0.10])
def test_planning_n_at_zero_is_connor(alpha):
    psi, delta = 0.10, 0.02
    connor = (stats.norm.isf(alpha) * math.sqrt(psi)
              + stats.norm.ppf(0.80) * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2
    assert paired.planning_n(psi, delta, 0.0, alpha, 0.80) == math.ceil(connor)


@pytest.mark.parametrize("args", [
    (0.01, 0.02, 0.0, 0.05, 0.8),    # discordance below the gain
    (0.10, 0.0, 0.0, 0.05, 0.8),     # alternative at the null
    (0.0, 0.0, -0.05, 0.05, 0.8),    # no discordance
])
def test_planning_n_infeasible(args):
    assert paired.planning_n(*args) is None


def test_mover_with_one_class_is_its_own_bound():
    lower = paired.score_lower_bound(3, 9, 32, 0.05)
    upper = paired.score_upper_bound(3, 9, 32, 0.05)
    assert paired.mover_bounds([6 / 32], [lower], [upper], [0.4]) == pytest.approx((6 / 32, lower, upper))


def test_n_effective_with_slice_weights_is_n():
    ns = [100, 250, 50]
    assert paired.n_effective(ns, ns) == pytest.approx(400)
    assert paired.n_effective([0.2, 0.8, 0.0], [10, 20, 0]) > 0
    assert paired.n_effective([0.2, 0.8], [10, 0]) == 0.0


def test_mover_on_the_demo_at_the_shipped_prior(demo):
    prior = {0: 0.118, 1: 0.031, 2: 0.018, 3: 0.0061, 4: 0.024, 5: 0.0025,
             6: 0.043, 7: 0.071, 8: 0.0039, 9: 0.0125, 10: 0.610}
    rows = paired.paired_counts(demo)["per_class"]
    ns = [r["n"] for r in rows]
    deltas = [(r["only_b_correct"] - r["only_a_correct"]) / r["n"] for r in rows]
    lowers = [paired.score_lower_bound(r["only_a_correct"], r["only_b_correct"], r["n"], 0.05) for r in rows]
    uppers = [paired.score_upper_bound(r["only_a_correct"], r["only_b_correct"], r["n"], 0.05) for r in rows]
    weights = [prior[r["class"]] for r in rows]
    delta_w, lower_w, upper_w = paired.mover_bounds(deltas, lowers, uppers, weights)
    assert delta_w == pytest.approx(-0.019769148936, abs=1e-12)
    assert lower_w == pytest.approx(-0.084356, abs=5e-7)
    assert upper_w == pytest.approx(0.044819, abs=5e-7)
    assert paired.n_effective(weights, ns) == pytest.approx(223.70, abs=5e-3)


# ---------------------------------------------------------------------------
# paired_counts: the input contract
# ---------------------------------------------------------------------------

DEMO_PER_CLASS = [(0, 23, 10), (1, 16, 8), (2, 8, 16), (3, 7, 23), (4, 14, 12), (5, 14, 15),
                  (6, 17, 13), (7, 18, 13), (8, 7, 21), (9, 8, 24), (10, 13, 13)]
DEMO_ALARMS = [(0, 1000, 37, 22), (1, 1000, 14, 15), (2, 1000, 18, 13), (3, 1000, 0, 0),
               (4, 1000, 13, 17), (5, 1000, 21, 14), (6, 1000, 22, 16), (7, 1000, 11, 24),
               (8, 1000, 3, 7), (9, 1000, 25, 16), (10, 1000, 16, 13)]


def test_paired_counts_on_the_demo(demo):
    pc = paired.paired_counts(demo)
    assert pc["valid"] is True and pc["errors"] == []
    assert [pc[c] for c in paired.CELLS] == [753, 145, 168, 34] and pc["n"] == 1100
    assert pc["id_col"] == "image_path" and pc["correctness_source"] == "pred_vs_label"
    assert [(r["class"], r["only_a_correct"], r["only_b_correct"]) for r in pc["per_class"]] == DEMO_PER_CLASS
    assert pc["per_class"][0]["name"] == "scratch" and pc["per_class"][10]["name"] == "clean"
    assert [(r["class"], r["n_other"], r["only_a_alarm"], r["only_b_alarm"])
            for r in pc["per_predicted_class"]] == DEMO_ALARMS
    assert paired.paired_counts_errors(pc) == []


def test_paired_counts_do_not_depend_on_row_order_or_index(demo):
    ref = paired.paired_counts(demo)
    frames = [
        demo.sort_values("confidence_a"),
        demo.sample(frac=1.0, random_state=3),
        pd.concat([demo.iloc[:550], demo.iloc[550:]]),          # index kept, not ignored
        demo.set_index("image_path", drop=False),
    ]
    for frame in frames:
        assert paired.paired_counts(frame) == ref
    subset = demo[demo["label"] != 5]
    assert paired.paired_counts(subset) == paired.paired_counts(subset.reset_index(drop=True))


def test_paired_counts_block_is_strict_json(demo):
    pc = paired.paired_counts(demo)
    assert json.loads(json.dumps(pc, allow_nan=False)) == pc


def _with_correctness(demo: pd.DataFrame) -> pd.DataFrame:
    out = demo.drop(columns=["pred_a", "pred_b"])
    out["correct_a"] = (demo["pred_a"] == demo["label"]).astype(int)
    out["correct_b"] = (demo["pred_b"] == demo["label"]).astype(int)
    return out


def _contract_violations(demo: pd.DataFrame):
    """(case, frame, expected error prefix) for every rule of the contract."""
    corr = _with_correctness(demo)
    cases = [("correctness and names but no label", corr.drop(columns=["label"]),
              "class column 'label' is missing")]
    nan_label = corr.assign(label=np.nan)
    cases.append(("all-NaN label", nan_label, "label: 1100 missing value(s)"))
    coil_nan = demo.assign(label=demo["label"].astype(float).where(demo["label"] != 5))
    cases.append(("label missing on coil_break", coil_nan, "label: 100 missing value(s)"))
    cases.append(("arm column", demo.assign(arm="a"), "an 'arm' column is present"))
    pred_nan = demo.assign(pred_a=demo["pred_a"].astype(float))
    pred_nan.loc[pred_nan.index[:150], "pred_a"] = np.nan
    cases.append(("missing predictions", pred_nan, "pred_a: 150 missing value(s)"))
    float_nan = corr.assign(correct_b=corr["correct_b"].astype(float))
    float_nan.loc[float_nan.index[:10], "correct_b"] = np.nan
    cases.append(("correct_b float with NaN", float_nan, "correct_b: 10 missing value(s)"))
    cases.append(("correct_b as text", corr.assign(correct_b=corr["correct_b"].astype(str)),
                  "correct_b: dtype"))
    nullable = corr.assign(correct_b=corr["correct_b"].astype(bool).astype("boolean"))
    nullable.loc[nullable.index[0], "correct_b"] = pd.NA
    cases.append(("nullable boolean with NA", nullable, "correct_b: 1 missing value(s)"))
    cases.append(("slice appended four times", pd.concat([demo] * 4),
                  "image_path: 3300 duplicated row(s) (1100 unique of 4400)"))
    both = demo.assign(correct_a=(demo["pred_a"] == demo["label"]).astype(int),
                       correct_b=(demo["pred_b"] == demo["label"]).astype(int))
    both.loc[both.index[:20], "correct_b"] = 1 - both.loc[both.index[:20], "correct_b"]
    cases.append(("correctness disagrees with predictions", both,
                  "correct_a/correct_b disagree with pred_*==label on 20 row(s)"))
    cases.append(("text predictions, integer labels", demo.assign(pred_a=demo["pred_name_a"]),
                  "pred_a and label use different encodings"))
    names = demo.copy()
    names.loc[names.index[0], "label_name"] = "pit"
    cases.append(("two names for one class", names, "label_name is not one-to-one with label"))
    shared = demo.copy()
    shared.loc[shared["label"] == 1, "label_name"] = "scratch"
    cases.append(("one name for two classes", shared, "label_name is not one-to-one with label"))
    cases.append(("empty frame", demo.iloc[:0], "no rows"))
    cases.append(("no identifier", demo.drop(columns=["image_path"]),
                  "no identifier column (looked for ['segmentation_id', 'image_id', 'image_path'])"))
    one_id = demo.astype({"image_path": object})
    one_id.loc[one_id.index[7], "image_path"] = None
    cases.append(("one missing identifier", one_id, "image_path: 1 missing identifier(s)"))
    cases.append(("one row appended twice", pd.concat([demo, demo.iloc[[3]]]),
                  "image_path: 1 duplicated row(s) (1100 unique of 1101)"))
    half = demo.drop(columns=["label_name"]).astype({"label": float})
    half.loc[half.index[0], "label"] = 2.5
    cases.append(("non-integral label", half, "label: non-integral float values"))
    two = corr.copy()
    two.loc[two.index[0], "correct_b"] = 2
    cases.append(("correct_b holds a 2", two, "correct_b: integer values other than 0/1: [2]"))
    one = demo.assign(correct_a=(demo["pred_a"] == demo["label"]).astype(int),
                      correct_b=(demo["pred_b"] == demo["label"]).astype(int))
    one.loc[one.index[0], "correct_b"] = 1 - one.loc[one.index[0], "correct_b"]
    cases.append(("correctness disagrees on one row", one,
                  "correct_a/correct_b disagree with pred_*==label on 1 row(s)"))
    cases.append(("repeated label column", pd.concat([demo, demo[["label"]]], axis=1),
                  "repeated column names: ['label']"))
    cases.append(("repeated identifier column", pd.concat([demo, demo[["image_path"]]], axis=1),
                  "repeated column names: ['image_path']"))
    listed = demo.astype({"image_path": object})
    listed.at[listed.index[2], "image_path"] = ["img", 2]
    cases.append(("an unhashable identifier", listed, "image_path: 1 unhashable value(s)"))
    listed_names = demo.astype({"label_name": object})
    listed_names.at[listed_names.index[2], "label_name"] = ["scratch"]
    cases.append(("an unhashable name", listed_names, "label_name: 1 unhashable value(s)"))
    return cases


def test_contract_violations_make_the_block_invalid(demo):
    for case, frame, prefix in _contract_violations(demo):
        pc = paired.paired_counts(frame)
        assert pc["valid"] is False, case
        assert pc["n"] is None and all(pc[c] is None for c in paired.CELLS), case
        assert pc["per_class"] == [] and pc["per_predicted_class"] is None, case
        assert any(e.startswith(prefix) for e in pc["errors"]), (case, pc["errors"])
        assert paired.paired_counts_errors(pc), case


def test_label_and_prediction_errors_are_both_listed(demo):
    df = demo.assign(label=demo["label"].astype(float), pred_a=demo["pred_a"].astype(float))
    df.loc[df.index[:3], "label"] = np.nan
    df.loc[df.index[10:15], "pred_a"] = np.nan
    assert paired.paired_counts(df)["errors"] == ["label: 3 missing value(s)",
                                                  "pred_a: 5 missing value(s)"]


def test_unnamed_classes_keep_the_slice_valid(demo):
    # label_name is display only: two classes without a name are not "one name for two".
    df = demo.astype({"label_name": object})
    df.loc[df["label"].isin([3, 5]), "label_name"] = None
    pc = paired.paired_counts(df)
    assert pc["valid"] is True
    names = {r["class"]: r["name"] for r in pc["per_class"]}
    assert names[3] is None and names[5] is None and names[0] == "scratch"


def test_correctness_columns_alone_are_valid_without_alarm_counts(demo):
    pc = paired.paired_counts(_with_correctness(demo))
    assert pc["valid"] is True and pc["correctness_source"] == "correct_cols"
    assert [pc[c] for c in paired.CELLS] == [753, 145, 168, 34]
    assert pc["predictions_available"] is False and pc["per_predicted_class"] is None


def test_agreeing_correctness_and_predictions(demo):
    both = demo.assign(correct_a=demo["pred_a"] == demo["label"],
                       correct_b=demo["pred_b"] == demo["label"])
    pc = paired.paired_counts(both)
    assert pc["valid"] is True and pc["correctness_source"] == "pred_vs_label+correct_cols"


def test_cluster_column_needs_one_image_per_cluster(demo):
    unique = demo.assign(coil=[f"coil_{i}" for i in range(len(demo))])
    assert paired.paired_counts(unique, cluster_col="coil")["valid"] is True
    shared = unique.copy()
    shared.loc[shared.index[1], "coil"] = "coil_0"
    errors = paired.paired_counts(shared, cluster_col="coil")["errors"]
    assert errors == ["coil: 1099 clusters hold 1100 rows; the paired tests need one image per cluster"]
    assert paired.paired_counts(demo, cluster_col="coil")["errors"] == ["cluster column 'coil' is missing"]


def test_resolve_correctness_needs_labels_only_for_predictions():
    preds = pd.DataFrame({"pred_a": [0, 1], "pred_b": [0, 1]})
    ca, cb, errors, source = paired.resolve_correctness(preds)
    assert ca is None and source is None
    assert errors == ["no correctness: need (pred_a, pred_b, label) or (correct_a, correct_b)"]
    corr = pd.DataFrame({"correct_a": [True, False], "correct_b": [True, True]})
    ca, cb, errors, source = paired.resolve_correctness(corr)
    assert errors == [] and source == "correct_cols"
    assert ca.tolist() == [True, False] and cb.tolist() == [True, True]


def test_class_cells_need_no_identifier(demo):
    rows, errors = paired.class_cells(demo.drop(columns=["image_path"]))
    assert errors == []
    cells = [{k: r[k] for k in ("class", "n", *paired.CELLS)} for r in rows]
    assert cells == [{k: r[k] for k in ("class", "n", *paired.CELLS)}
                     for r in paired.paired_counts(demo)["per_class"]]
    rows, errors = paired.class_cells(pd.DataFrame({"correct_a": [1], "correct_b": [0]}))
    assert rows == [] and errors == ["class column 'label' is missing"]


def test_paired_counts_errors_on_hand_built_blocks(demo):
    good = paired.paired_counts(demo)
    assert paired.paired_counts_errors(None) == ["paired_counts block missing"]
    assert paired.paired_counts_errors({**good, "schema": None})
    assert paired.paired_counts_errors({**good, "n": good["n"] + 1})
    assert paired.paired_counts_errors({**good, "both_correct": True}) == [
        "paired_counts cells must be non-negative integers"]
    assert paired.paired_counts_errors({**good, "both_correct": 753.0}) == [
        "paired_counts cells must be non-negative integers"]
    assert paired.paired_counts_errors({**good, "errors": ["late"]}) == [
        "paired_counts marked valid but lists errors"]
    rows = [dict(r) for r in good["per_class"]]
    rows[0]["only_b_correct"] += 1
    rows[0]["both_correct"] -= 1
    assert paired.paired_counts_errors({**good, "per_class": rows})
    alarms = [dict(r) for r in good["per_predicted_class"]]
    alarms[0]["n_other"] += 1
    alarms[0]["neither_alarm"] += 1
    assert paired.paired_counts_errors({**good, "per_predicted_class": alarms})


def _alarm_rows(block, drop=()):
    return [dict(r) for r in block["per_predicted_class"] if r["class"] not in drop]


def test_every_labelled_class_needs_its_alarm_row(demo):
    # A block without its false-alarm rows would let any B through the veto.
    good = paired.paired_counts(demo)
    missing = "paired_counts.per_predicted_class has no row for labelled classes "
    errors = paired.paired_counts_errors({**good, "per_predicted_class": _alarm_rows(good, drop={5})})
    assert missing + "[5]" in errors
    errors = paired.paired_counts_errors({**good, "per_predicted_class": []})
    assert missing + "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and 1 more" in errors
    dropped = json.loads(json.dumps({**good, "per_predicted_class": _alarm_rows(good, drop={5})}))
    assert missing + "[5]" in paired.paired_counts_errors(dropped)


def test_alarms_add_up_to_the_wrong_predictions(demo):
    # Each wrong prediction is one false alarm into the class it predicts.
    good = paired.paired_counts(demo)
    alarms = _alarm_rows(good)
    alarms[5]["only_b_alarm"] -= 1
    alarms[5]["neither_alarm"] += 1
    assert paired.paired_counts_errors({**good, "per_predicted_class": alarms}) == [
        "paired_counts.per_predicted_class: A and B have 202 and 178 false alarms, "
        "but 202 and 179 wrong predictions"]
    # A class predicted but never labelled: its row cannot be dropped either.
    df = demo.copy()
    df.loc[df.index[:3], "pred_b"] = 11
    block = paired.paired_counts(df)
    assert [r["class"] for r in block["per_predicted_class"]][-1] == 11
    assert paired.paired_counts_errors({**block, "per_predicted_class": _alarm_rows(block, drop={11})})


def test_predictions_flag_matches_the_alarm_rows(demo):
    good = paired.paired_counts(demo)
    flag = "paired_counts.per_predicted_class must be a list when predictions_available is true, null otherwise"
    assert paired.paired_counts_errors({**good, "per_predicted_class": None}) == [flag]
    assert paired.paired_counts_errors({**good, "predictions_available": False}) == [flag]
    assert paired.paired_counts_errors({**good, "per_predicted_class": {}}) == [flag]
    assert paired.paired_counts_errors({**good, "predictions_available": "yes"}) == [
        "paired_counts.predictions_available must be true or false"]
    corr = paired.paired_counts(_with_correctness(demo))
    assert paired.paired_counts_errors(corr) == []
    assert paired.paired_counts_errors({**corr, "per_predicted_class": _alarm_rows(good)}) == [flag]


def test_repeated_or_bad_classes_and_empty_rows(demo):
    good = paired.paired_counts(demo)
    rows = [dict(r) for r in good["per_class"]]
    rows[1]["class"] = 0
    errors = paired.paired_counts_errors({**good, "per_class": rows})
    assert "paired_counts.per_class[1]: bad or repeated class 0" in errors
    for key in (None, True, 2.0, [0], "3"):
        rows = [dict(r) for r in good["per_class"]]
        rows[3]["class"] = key
        errors = paired.paired_counts_errors({**good, "per_class": rows})
        if key == "3":   # a str class is a valid key, but then class 3 has no alarm row
            assert any("no row for labelled classes" in e for e in errors), errors
        else:
            assert f"paired_counts.per_class[3]: bad or repeated class {key!r}" in errors
    alarms = _alarm_rows(good) + [dict(good["per_predicted_class"][5])]
    assert paired.paired_counts_errors({**good, "per_predicted_class": alarms}) == [
        "paired_counts.per_predicted_class[11]: bad or repeated class 5"]
    # paired_counts never emits a class without images; an absent class has no row.
    zero = {**good, **{c: 0 for c in paired.CELLS}, "n": 0, "predictions_available": False,
            "per_predicted_class": None,
            "per_class": [{"class": 0, "name": None, "n": 0, **{c: 0 for c in paired.CELLS}}]}
    assert paired.paired_counts_errors(zero) == [
        "paired_counts: cells sum to 0, n = 0",
        "paired_counts.per_class[0] (class 0): cells must be counts summing to n > 0"]


@pytest.mark.parametrize("change", [
    {"per_class": [1, 2]},
    {"per_class": [None]},
    {"per_class": "rows"},
    {"per_predicted_class": [None]},
    {"per_predicted_class": ["row"]},
    {"errors": [1, None], "valid": False},
    {"id_col": float("nan")},
    {"id_col": ["image_path"]},
    {"n": None},
    {"n": float("nan")},
])
def test_paired_counts_errors_never_raise(demo, change):
    errors = paired.paired_counts_errors({**paired.paired_counts(demo), **change})
    assert errors and all(isinstance(e, str) for e in errors)


def test_names_must_be_strings_or_null(demo):
    good = paired.paired_counts(demo)
    rows = [dict(r) for r in good["per_class"]]
    rows[3]["name"] = float("nan")
    assert paired.paired_counts_errors({**good, "per_class": rows}) == [
        "paired_counts.per_class[3] (class 3): name must be a string or null"]
    alarms = _alarm_rows(good)
    alarms[2]["name"] = 7
    assert paired.paired_counts_errors({**good, "per_predicted_class": alarms}) == [
        "paired_counts.per_predicted_class[2] (class 2): name must be a string or null"]
    unnamed = [dict(r, name=None) for r in good["per_class"]]
    assert paired.paired_counts_errors({**good, "per_class": unnamed}) == []

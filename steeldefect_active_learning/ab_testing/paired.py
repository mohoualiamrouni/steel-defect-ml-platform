"""Paired evaluation of two classifiers: the input contract and the statistics of the gate.

Both models score the same labelled images. For every image, a = 1[A correct] and
b = 1[B correct], and the promotion gate needs only four counts, overall and per class:

    n11  A correct, B correct   both_correct
    n10  A correct, B wrong     only_a_correct
    n01  A wrong, B correct     only_b_correct
    n00  A wrong, B wrong       both_wrong

so that Delta = acc_B - acc_A = (n01 - n10) / n. ``paired_counts`` validates a prediction
frame and returns these counts overall, per true class, and per predicted class (false
alarms). The functions below test them:

    mcnemar_one_sided_p   exact conditional McNemar test of H0: B is not more accurate
    mcnemar_two_sided     the two-sided test of the diagnostic McNemar blocks
    score_lower_bound     one-sided lower bound of Delta: Tango's score interval, with a
    score_upper_bound     one-count continuity adjustment for every decision
    noninferiority_floor  smallest class size at which a class where A and B agree on every
                          image can pass a margin
    mover_bounds          bounds of a weighted sum of per-class differences
    n_effective           size of the evidence on that weighted sum
    planning_n            prospective sample size of the score test
    posterior_b_better    exact P(acc_B > acc_A) under a uniform Dirichlet prior

Tango writes the table with the new treatment first: his b is our n01 and his c our n10.

References:
    McNemar (1947) "Note on the sampling error of the difference between correlated
        proportions or percentages", Psychometrika 12(2):153-157.
    Tango (1998) "Equivalence test and confidence interval for the difference in
        proportions for the paired-sample design", Statistics in Medicine 17:891-908,
        eq. 24-26 (score statistic) and 33-34 (limits).
    Laud, R package ratesci, ``pairbinci``: the continuity adjustment (cc = 0.5 there),
        which its documentation attributes to Chang et al. (2024), J. Applied Statistics
        51(1):139-152.
    Connor (1987) "Sample size for testing differences in proportions for the
        paired-sample design", Biometrics 43:207-211.
    Zou & Donner (2008) "Construction of confidence limits about effect measures: a
        general approach", Statistics in Medicine 27:1693-1702 (MOVER).
    Fagerland, Lydersen & Laake (2014) "Recommended tests and confidence intervals for
        paired binomial proportions", Statistics in Medicine 33:2850-2875.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SCHEMA = "paired_counts/1"
CELLS = ("both_correct", "only_a_correct", "only_b_correct", "both_wrong")   # n11, n10, n01, n00
FA_CELLS = ("both_alarm", "only_a_alarm", "only_b_alarm", "neither_alarm")  # f11, f10, f01, f00
ID_CANDIDATES = ("segmentation_id", "image_id", "image_path")
# One count subtracted from the score numerator of every decision bound: it keeps the
# exact size of each clause at or below alpha (Tango's plain interval reaches 0.07).
CONTINUITY_ADJUSTMENT = 1


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

def _py(value: Any) -> Any:
    """A plain Python scalar, so that the blocks are JSON-serialisable."""
    return value.item() if isinstance(value, np.generic) else value


def _class_values(s: pd.Series, name: str, errors: List[str]) -> Optional[np.ndarray]:
    """Class ids: integers (or integral floats) or strings, never missing, never bool."""
    missing = int(s.isna().sum())
    if missing:
        errors.append(f"{name}: {missing} missing value(s)")
        return None
    if pd.api.types.is_bool_dtype(s):
        errors.append(f"{name}: boolean values are not class ids")
        return None
    if pd.api.types.is_integer_dtype(s):
        return s.to_numpy(dtype=np.int64)
    if pd.api.types.is_float_dtype(s):
        values = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
            errors.append(f"{name}: non-integral float values")
            return None
        return values.astype(np.int64)
    values = s.to_numpy(dtype=object)
    if {type(v) for v in values} <= {str}:
        return values
    errors.append(f"{name}: mixed or unsupported value types")
    return None


def _correct_values(s: pd.Series, name: str, errors: List[str]) -> Optional[np.ndarray]:
    """A correctness column: bool or integer 0/1, never missing."""
    missing = int(s.isna().sum())
    if missing:
        errors.append(f"{name}: {missing} missing value(s); a missing outcome is never 'wrong'")
        return None
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    if pd.api.types.is_integer_dtype(s):
        values = s.to_numpy(dtype=np.int64)
        bad = sorted(set(np.unique(values).tolist()) - {0, 1})
        if bad:
            errors.append(f"{name}: integer values other than 0/1: {bad[:5]}")
            return None
        return values == 1
    errors.append(f"{name}: dtype {s.dtype} is not bool or integer 0/1")
    return None


def _validated(df: pd.DataFrame, class_col: str, *, require_labels: bool) -> Dict[str, Any]:
    """Validate labels, predictions and correctness; arrays are positional."""
    errors: List[str] = []
    cols = set(df.columns)
    labels = pred_a = pred_b = correct_a = correct_b = None
    source = None
    if class_col in cols:
        labels = _class_values(df[class_col], class_col, errors)
    elif require_labels:
        errors.append(f"class column '{class_col}' is missing")
    has_pred = {"pred_a", "pred_b"} <= cols
    has_corr = {"correct_a", "correct_b"} <= cols
    if has_pred and labels is not None:
        values = {}
        for name in ("pred_a", "pred_b"):
            v = _class_values(df[name], name, errors)
            if v is not None and (labels.dtype == object) != (v.dtype == object):
                errors.append(f"{name} and {class_col} use different encodings")
                v = None
            values[name] = v
        pred_a, pred_b = values["pred_a"], values["pred_b"]
        if pred_a is not None and pred_b is not None:
            correct_a, correct_b = pred_a == labels, pred_b == labels
            source = "pred_vs_label"
    if has_corr:
        xa = _correct_values(df["correct_a"], "correct_a", errors)
        xb = _correct_values(df["correct_b"], "correct_b", errors)
        if xa is not None and xb is not None:
            if correct_a is not None:
                k = int(np.sum((xa != correct_a) | (xb != correct_b)))
                if k:
                    errors.append(
                        f"correct_a/correct_b disagree with pred_*=={class_col} on {k} row(s)")
                else:
                    source = "pred_vs_label+correct_cols"
            elif not (has_pred and class_col in cols):
                correct_a, correct_b, source = xa, xb, "correct_cols"
    if correct_a is None and not errors:
        errors.append("no correctness: need (pred_a, pred_b, label) or (correct_a, correct_b)")
    if errors:
        correct_a = correct_b = None
    return {"labels": labels, "pred_a": pred_a, "pred_b": pred_b, "correct_a": correct_a,
            "correct_b": correct_b, "errors": errors, "source": source}


def resolve_correctness(
    df: pd.DataFrame, *, class_col: str = "label",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[str]]:
    """Validated per-row correctness of A and B, the errors found, and its source.

    Correctness comes from (pred_a, pred_b, label) or from (correct_a, correct_b); when
    both are present they must agree on every row. Labels are needed only to derive it
    from the predictions. A missing prediction or outcome is invalid, never "wrong".

    Returns:
        (correct_a, correct_b, errors, source): positional bool arrays (None on any
        error), the list of errors, and "pred_vs_label", "correct_cols" or
        "pred_vs_label+correct_cols".
    """
    v = _validated(df, class_col, require_labels=False)
    return v["correct_a"], v["correct_b"], v["errors"], v["source"]


def paired_counts(
    df: pd.DataFrame,
    *,
    class_col: str = "label",
    id_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    name_col: Optional[str] = "label_name",
) -> Dict[str, Any]:
    """Validate a paired prediction frame and count its 2x2 cells.

    Never raises on bad data: every violation is listed in ``errors`` and makes the block
    invalid, with every count None. Rows are used by position, so the index plays no
    role. The contract:

    - at least one row, and no ``arm`` column (an online split is not a paired design);
    - a unique, non-missing image identifier: ``id_col``, or the first present of
      ``segmentation_id``, ``image_id``, ``image_path``;
    - with ``cluster_col`` (for example the coil), one image per cluster;
    - ``class_col`` with integer or string class ids and no missing value;
    - correctness as in :func:`resolve_correctness`;
    - ``name_col``, if present, one-to-one with ``class_col`` (display only).

    Returns:
        A ``paired_counts/1`` block: ``valid``, ``errors``, ``n`` and the four cells
        overall, ``per_class`` (the cells per true class) and, when predictions exist,
        ``per_predicted_class`` (for each class k, the alarm cells of A and B on the
        ``n_other`` images whose label is not k). Values are plain Python types.
    """
    errors: List[str] = []
    n_rows = int(len(df))
    out: Dict[str, Any] = {
        "schema": SCHEMA, "valid": False, "errors": errors, "n_rows": n_rows,
        "id_col": None, "cluster_col": cluster_col, "class_col": class_col,
        "correctness_source": None, "predictions_available": False,
        "n": None, **{c: None for c in CELLS},
        "per_class": [], "per_predicted_class": None,
    }
    cols = set(df.columns)
    if n_rows == 0:
        errors.append("no rows")
    if "arm" in cols:
        errors.append("an 'arm' column is present: an online traffic split is not a paired "
                      "evaluation")
    if id_col is None:
        id_col = next((c for c in ID_CANDIDATES if c in cols), None)
    if id_col is None or id_col not in cols:
        errors.append(f"no identifier column (looked for {id_col or list(ID_CANDIDATES)})")
    else:
        out["id_col"] = id_col
        ids = df[id_col]
        missing = int(ids.isna().sum())
        if missing:
            errors.append(f"{id_col}: {missing} missing identifier(s)")
        duplicated = int(ids.duplicated().sum())
        if duplicated:
            errors.append(f"{id_col}: {duplicated} duplicated row(s) "
                          f"({int(ids.nunique())} unique of {n_rows})")
    if cluster_col is not None:
        if cluster_col not in cols:
            errors.append(f"cluster column '{cluster_col}' is missing")
        else:
            clusters = df[cluster_col]
            missing = int(clusters.isna().sum())
            if missing:
                errors.append(f"{cluster_col}: {missing} missing value(s)")
            if int(clusters.duplicated().sum()):
                errors.append(f"{cluster_col}: {int(clusters.nunique())} clusters hold "
                              f"{n_rows} rows; the paired tests need one image per cluster")
    v = _validated(df, class_col, require_labels=True)
    errors.extend(v["errors"])
    labels = v["labels"]
    names: Dict[Any, Optional[str]] = {}
    if labels is not None and name_col and name_col in cols:
        pairs = pd.DataFrame({"k": labels, "v": df[name_col].to_numpy()}).drop_duplicates()
        if pairs["k"].duplicated().any() or pairs["v"].duplicated().any():
            errors.append(f"{name_col} is not one-to-one with {class_col}")
        else:
            names = {_py(k): (None if pd.isna(name) else str(name))
                     for k, name in zip(pairs["k"], pairs["v"])}
    if errors:
        return out

    ca, cb = v["correct_a"], v["correct_b"]
    both, only_a, only_b, neither = ca & cb, ca & ~cb, ~ca & cb, ~ca & ~cb
    out.update({
        "valid": True, "id_col": id_col, "correctness_source": v["source"], "n": n_rows,
        "both_correct": int(both.sum()), "only_a_correct": int(only_a.sum()),
        "only_b_correct": int(only_b.sum()), "both_wrong": int(neither.sum()),
    })
    keys, inverse = np.unique(labels, return_inverse=True)
    for j, k in enumerate(keys):
        sel = inverse == j
        out["per_class"].append({
            "class": _py(k), "name": names.get(_py(k)), "n": int(sel.sum()),
            "both_correct": int(both[sel].sum()), "only_a_correct": int(only_a[sel].sum()),
            "only_b_correct": int(only_b[sel].sum()), "both_wrong": int(neither[sel].sum()),
        })
    pa, pb = v["pred_a"], v["pred_b"]
    if pa is not None and pb is not None:
        out["predictions_available"] = True
        rows = []
        for k in np.unique(np.concatenate([labels, pa, pb])):
            other = labels != k
            alarm_a, alarm_b = (pa == k) & other, (pb == k) & other
            rows.append({
                "class": _py(k), "name": names.get(_py(k)), "n_other": int(other.sum()),
                "both_alarm": int((alarm_a & alarm_b).sum()),
                "only_a_alarm": int((alarm_a & ~alarm_b).sum()),
                "only_b_alarm": int((~alarm_a & alarm_b).sum()),
                "neither_alarm": int((other & ~alarm_a & ~alarm_b).sum()),
            })
        out["per_predicted_class"] = rows
    return out


def _is_count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def paired_counts_errors(block: Any) -> List[str]:
    """Why a ``paired_counts`` block (built here, by hand or read from JSON) is unusable.

    Returns an empty list for a valid, internally consistent block: the cells are
    non-negative integers summing to ``n > 0``, the per-class cells sum to the overall
    ones, and every false-alarm row covers the images whose label is not its class.
    """
    if not isinstance(block, Mapping):
        return ["paired_counts block missing"]
    errors: List[str] = []
    if block.get("schema") != SCHEMA:
        errors.append(f"paired_counts schema {block.get('schema')!r} != {SCHEMA!r}")
    if block.get("valid") is not True:
        return errors + list(block.get("errors") or ["paired_counts marked invalid"])
    if block.get("errors"):
        errors.append("paired_counts marked valid but lists errors")
    cells = [block.get(c) for c in CELLS]
    if not all(_is_count(x) for x in cells) or not _is_count(block.get("n")):
        return errors + ["paired_counts cells must be non-negative integers"]
    if sum(cells) != block["n"] or block["n"] == 0:
        errors.append(f"paired_counts: cells sum to {sum(cells)}, n = {block['n']}")
    rows = block.get("per_class")
    if not isinstance(rows, list) or not rows:
        return errors + ["paired_counts.per_class missing or empty"]
    seen, totals, n_per_class = set(), [0, 0, 0, 0], {}
    for row in rows:
        k = row.get("class")
        row_cells = [row.get(c) for c in CELLS]
        if (not all(_is_count(x) for x in row_cells) or not _is_count(row.get("n"))
                or sum(row_cells) != row["n"]):
            errors.append(f"paired_counts.per_class[{k!r}]: cells inconsistent")
            continue
        if k in seen or k is None or isinstance(k, (bool, float)):
            errors.append(f"paired_counts.per_class: bad or repeated class {k!r}")
        seen.add(k)
        n_per_class[k] = row["n"]
        totals = [t + x for t, x in zip(totals, row_cells)]
    if totals != cells:
        errors.append(f"paired_counts: per-class cells sum to {totals}, overall {cells}")
    alarms = block.get("per_predicted_class")
    if alarms is not None:
        if not isinstance(alarms, list):
            errors.append("paired_counts.per_predicted_class must be a list or null")
        else:
            seen_alarm = set()
            for row in alarms:
                k = row.get("class")
                row_cells = [row.get(c) for c in FA_CELLS]
                if (not all(_is_count(x) for x in row_cells) or not _is_count(row.get("n_other"))
                        or sum(row_cells) != row["n_other"]
                        or row["n_other"] != block["n"] - n_per_class.get(k, 0)
                        or k in seen_alarm or k is None or isinstance(k, (bool, float))):
                    errors.append(f"paired_counts.per_predicted_class[{k!r}]: inconsistent")
                seen_alarm.add(k)
    return errors


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def mcnemar_one_sided_p(n10: int, n01: int) -> float:
    """Exact conditional McNemar p-value of H0: p01 <= p10 (B not more accurate than A).

    p = P(X >= n01) with X ~ Binomial(n10 + n01, 1/2), and p = 1 without discordant
    pairs. ``binom.sf(n01 - 1, ...)`` keeps the atom at n01, and does not underflow to 0
    as ``1 - cdf`` would.
    """
    m = int(n10) + int(n01)
    if m == 0:
        return 1.0
    return float(stats.binom.sf(int(n01) - 1, m, 0.5))


def mcnemar_two_sided(n10: int, n01: int, exact_threshold: int = 25) -> Dict[str, Any]:
    """Two-sided McNemar test, as reported by the diagnostic blocks.

    Exact binomial below ``exact_threshold`` discordant pairs, Yates-corrected chi-square
    from there. ``b`` is the A-only cell (n10) and ``c`` the B-only cell (n01).
    """
    b, c = int(n10), int(n01)
    m = b + c
    if m == 0:
        statistic, p_value, test_used = 0.0, 1.0, "mcnemar_degenerate"
    elif m < exact_threshold:
        statistic = float(min(b, c))
        p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), m, 0.5)))
        test_used = "mcnemar_exact"
    else:
        statistic = (abs(b - c) - 1.0) ** 2 / float(m)
        p_value = float(stats.chi2.sf(statistic, 1))
        test_used = "mcnemar_yates"
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value,
            "test_used": test_used, "n_discordant": m, "alternative": "two-sided"}


# ---------------------------------------------------------------------------
# Score bounds (Tango 1998, with a continuity adjustment)
# ---------------------------------------------------------------------------

def _check_table(n10: int, n01: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n10 < 0 or n01 < 0 or n10 + n01 > n:
        raise ValueError(f"inconsistent cells: n10={n10}, n01={n01}, n={n}")


def _p10_restricted(n10: float, n01: float, n: float, d0: float) -> float:
    """Restricted MLE of p10 under p01 = p10 + d0: the larger root of A q^2 + B q + C."""
    a = 2.0 * n
    b = -n10 - n01 + d0 * (2.0 * n + n10 - n01)
    c = -n10 * d0 * (1.0 - d0)
    root = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    if b <= 0.0:
        return (root - b) / (2.0 * a)
    denominator = b + root
    return (-2.0 * c) / denominator if denominator > 0.0 else 0.0   # same root, no cancellation


def score_statistic(n10: int, n01: int, n: int, d0: float, *,
                    cc: int = CONTINUITY_ADJUSTMENT) -> float:
    """Score statistic of the one-sided test of H0: Delta <= d0 (large values reject).

    Z = (n01 - n10 - n d0 - cc) / sqrt(n (2 p10~ + d0 (1 - d0))), with the numerator set
    to 0 when 0 < n01 - n10 - n d0 < cc. ``cc = 0`` is Tango's eq. 24; at d0 = 0 and
    ``cc = 1`` it is the one-sided Yates McNemar statistic.
    """
    _check_table(n10, n01, n)
    s = n01 - n10 - n * d0
    numerator = 0.0 if 0 < s < cc else s - cc
    q = _p10_restricted(n10, n01, n, d0)
    variance = max(n * (2.0 * q + d0 * (1.0 - d0)), 0.0)
    if variance == 0.0:
        return math.inf if numerator > 0 else (-math.inf if numerator < 0 else 0.0)
    return numerator / math.sqrt(variance)


def score_lower_bound(n10: int, n01: int, n: int, alpha: float, *,
                      cc: int = CONTINUITY_ADJUSTMENT) -> float:
    """One-sided (1 - alpha) lower bound of Delta = acc_B - acc_A.

    The smallest d0 the score test does not reject, found by bisection (the closed-form
    quartic has a double-root trap, and brentq can return the end point of the bracket).
    ``L > d0`` exactly when ``score_statistic(..., d0) > z_(1-alpha)``.
    """
    _check_table(n10, n01, n)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n10 == n:
        return -1.0
    z = float(stats.norm.isf(alpha))
    lo, hi = -1.0, (n01 - n10 - cc) / n
    if hi <= -1.0:
        return -1.0

    def excess(d: float) -> float:
        q = _p10_restricted(n10, n01, n, d)
        return (n01 - n10 - n * d - cc) - z * math.sqrt(max(n * (2.0 * q + d * (1.0 - d)), 0.0))

    if excess(lo) <= 0.0:
        return -1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15:
            break
    return 0.5 * (lo + hi)


def score_upper_bound(n10: int, n01: int, n: int, alpha: float, *,
                      cc: int = CONTINUITY_ADJUSTMENT) -> float:
    """One-sided (1 - alpha) upper bound of Delta: the lower bound with A and B swapped."""
    return -score_lower_bound(n01, n10, n, alpha, cc=cc)


def noninferiority_floor(margin: float, alpha: float, *, cc: int = CONTINUITY_ADJUSTMENT) -> int:
    """Smallest class size at which a class with no discordant image passes ``-margin``.

    At n10 = n01 = 0 and d0 = -m, Z = (n m - cc) / sqrt(n m (1 - m)); with
    k = z^2 (1 - m), the class passes once n m exceeds the larger root of
    r^2 - (2 cc + k) r + cc^2 (r = k without adjustment, Tango's eq. 30).
    """
    if not 0.0 < margin < 1.0:
        raise ValueError(f"margin must be in (0, 1), got {margin}")
    z = float(stats.norm.isf(alpha))
    k = z * z * (1.0 - margin)
    b = 2.0 * cc + k
    r = (b + math.sqrt(b * b - 4.0 * cc * cc)) / 2.0
    return int(math.floor(r / margin)) + 1


# ---------------------------------------------------------------------------
# Weighted difference, planning, posterior
# ---------------------------------------------------------------------------

def mover_bounds(
    deltas: Sequence[float],
    lowers: Sequence[float],
    uppers: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float, float]:
    """Weighted difference sum_k w_k Delta_k and its MOVER bounds.

    The classes are disjoint sets of images, hence independent, and each one-sided bound
    is recovered from its own distance to the point estimate (Zou & Donner 2008). The
    weights are renormalised to sum to 1.

    Returns:
        (Delta_w, L_w, U_w).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    d = np.asarray(deltas, dtype=float)
    lo = np.asarray(lowers, dtype=float)
    hi = np.asarray(uppers, dtype=float)
    delta_w = float(np.sum(w * d))
    return (delta_w,
            delta_w - math.sqrt(float(np.sum((w * (d - lo)) ** 2))),
            delta_w + math.sqrt(float(np.sum((w * (hi - d)) ** 2))))


def n_effective(weights: Sequence[float], ns: Sequence[int]) -> float:
    """Effective number of images behind a weighted difference: 1 / sum_k w_k^2 / n_k.

    It equals n when the weights are the slice's own class shares, and 0 when a class
    with a positive weight has no image. The weights are renormalised to sum to 1.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n = np.asarray(ns, dtype=float)
    positive = w > 0
    if np.any(n[positive] <= 0):
        return 0.0
    return float(1.0 / np.sum(w[positive] ** 2 / n[positive]))


def planning_n(psi: float, delta1: float, d0: float, alpha: float, power: float) -> Optional[int]:
    """Images the score test of H0: Delta <= d0 needs to reach ``power`` at Delta = delta1.

    The planning alternative has discordance psi, with p10 = (psi - delta1) / 2 and
    p01 = (psi + delta1) / 2. At d0 = 0 this is Connor (1987):
    [z_(1-alpha) sqrt(psi) + z_power sqrt(psi - delta1^2)]^2 / delta1^2. A normal
    approximation without the continuity adjustment: a planning number, not a guarantee.

    Returns:
        The number of images, or None when the alternative is impossible
        (psi < |delta1|, psi outside (0, 1], or delta1 <= d0).
    """
    if not (0.0 < psi <= 1.0 and psi >= abs(delta1) and delta1 > d0):
        return None
    p10 = (psi - delta1) / 2.0
    p01 = (psi + delta1) / 2.0
    q = _p10_restricted(p10, p01, 1.0, d0)
    s0 = math.sqrt(max(2.0 * q + d0 * (1.0 - d0), 0.0))
    s1 = math.sqrt(max(psi - delta1 * delta1, 0.0))
    z_alpha, z_power = float(stats.norm.isf(alpha)), float(stats.norm.ppf(power))
    return int(math.ceil((z_alpha * s0 + z_power * s1) ** 2 / (delta1 - d0) ** 2))


def posterior_b_better(n10: int, n01: int) -> float:
    """P(p01 > p10) under a uniform Dirichlet(1, 1, 1, 1) prior on the four cells.

    The posterior of p01 / (p10 + p01) is Beta(1 + n01, 1 + n10), so the probability is
    exact; it equals 1 minus the one-sided mid-p McNemar value. A diagnostic: the gate
    thresholds no posterior.
    """
    return float(stats.beta.sf(0.5, 1 + int(n01), 1 + int(n10)))

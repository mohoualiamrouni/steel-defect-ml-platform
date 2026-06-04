"""2D guillotine cutting optimiser using dynamic programming.

Implements a two-stage approach for steel coil cutting:
  1. **Transversal cuts** along the coil length (DP over length bins).
  2. **Longitudinal cuts** per resulting sheet (DP over width bins).

Grading is **defect-driven**: each candidate piece is evaluated by
querying the defects that fall within its physical bounds and taking
the maximum predicted severity.  This eliminates the information loss
inherent in heatmap-based grading (COAML principle: align prediction
with decision, cf. Schiffer et al. 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constraints import (
    CuttingConstraints,
    compute_piece_grade_from_defects,
    compute_piece_value,
)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    """A single rectangular piece produced by cutting."""

    x_start: float  # metres (along length)
    x_end: float
    y_start: float  # metres (along width)
    y_end: float
    grade: int
    grade_name: str
    value: float
    max_severity: float
    n_defects: int = 0

    @property
    def length(self) -> float:
        return self.x_end - self.x_start

    @property
    def width(self) -> float:
        return self.y_end - self.y_start

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass
class CuttingPlan:
    """Complete cutting plan for one coil."""

    pieces: List[Piece] = field(default_factory=list)
    total_value: float = 0.0
    max_possible_value: float = 0.0
    yield_pct: float = 0.0
    n_pieces: int = 0
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    # AREA (m²) per grade — H3 thresholds are about material, not piece counts,
    # so the grade fractions reported for H3 are area-weighted from this.
    area_by_grade: Dict[str, float] = field(default_factory=dict)
    transversal_cuts: List[float] = field(default_factory=list)
    longitudinal_cuts: Dict[int, List[float]] = field(default_factory=dict)

    def grade_area_fractions(self) -> Dict[str, float]:
        """Return ``{grade_name: area_fraction}`` (area-weighted, sums to ~1)."""
        total = sum(self.area_by_grade.values()) or 1.0
        return {g: a / total for g, a in self.area_by_grade.items()}


# ---------------------------------------------------------------------------
# Helper: grade + value for a candidate piece
# ---------------------------------------------------------------------------


def _piece_grade_value(
    defects_df: pd.DataFrame,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    constraints: CuttingConstraints,
    severity_col: str = "severity",
) -> Tuple[int, float]:
    """Return (grade, value) for a candidate rectangular piece."""
    grade = compute_piece_grade_from_defects(
        defects_df, x_start, x_end, y_start, y_end,
        constraints.quality_thresholds, severity_col,
    )
    area = (x_end - x_start) * (y_end - y_start)
    value = compute_piece_value(grade, area, constraints.grade_values)
    return grade, value


# ---------------------------------------------------------------------------
# 1D DP: optimal longitudinal cuts for a single sheet
# ---------------------------------------------------------------------------


def _dp_longitudinal(
    defects_df: pd.DataFrame,
    x_start: float,
    x_end: float,
    coil_width: float,
    constraints: CuttingConstraints,
    n_width_bins: int = 40,
    severity_col: str = "severity",
) -> Tuple[float, List[float]]:
    """Find optimal longitudinal cut positions for one sheet.

    Parameters
    ----------
    defects_df : DataFrame
        Defects for this coil (already filtered or full).
    x_start, x_end : float
        Length bounds of this sheet segment (metres).
    coil_width : float
        Physical width of the coil in metres.
    constraints : CuttingConstraints
    n_width_bins : int
        Number of candidate cut positions along the width.
    severity_col : str
        Column with predicted severity.

    Returns
    -------
    best_value : float
        Maximum total value achievable.
    cut_positions : list of float
        Optimal cut positions in metres along the width.
    """
    bin_width = coil_width / n_width_bins
    min_bins = max(1, int(np.ceil(constraints.min_piece_width / bin_width)))

    # Filter defects to this sheet segment
    sheet_defects = defects_df[
        (defects_df["x"] >= x_start) & (defects_df["x"] < x_end)
    ]

    # dp[j] = max value from width bins 0..j-1
    dp = np.full(n_width_bins + 1, -np.inf)
    dp[0] = 0.0
    parent = np.full(n_width_bins + 1, -1, dtype=int)

    for j in range(min_bins, n_width_bins + 1):
        y_end_j = j * bin_width
        for k in range(0, j - min_bins + 1):
            if dp[k] == -np.inf:
                continue
            y_start_k = k * bin_width

            grade, value = _piece_grade_value(
                sheet_defects, x_start, x_end,
                y_start_k, y_end_j, constraints, severity_col,
            )

            val = dp[k] + value
            if val > dp[j]:
                dp[j] = val
                parent[j] = k

    # Backtrack to find cut positions
    cuts_bins: List[int] = []
    pos = n_width_bins
    while pos > 0 and parent[pos] >= 0:
        cuts_bins.append(parent[pos])
        pos = parent[pos]
    cuts_bins = sorted(set(cuts_bins) - {0})
    cuts_metres = [c * bin_width for c in cuts_bins]

    # Invariant: strictly increasing cut sequence, all strictly inside (0, coil_width)
    if cuts_metres:
        assert cuts_metres == sorted(set(cuts_metres)), \
            f"Longitudinal backtracking produced unsorted/duplicate cuts: {cuts_metres}"
        assert all(0.0 < c < coil_width for c in cuts_metres), \
            f"Longitudinal backtracking produced out-of-range cuts: {cuts_metres}"

    best = float(dp[n_width_bins]) if dp[n_width_bins] > -np.inf else 0.0
    return best, cuts_metres


# ---------------------------------------------------------------------------
# 1D DP: optimal transversal cuts along coil length
# ---------------------------------------------------------------------------


def _dp_transversal(
    defects_df: pd.DataFrame,
    coil_length: float,
    coil_width: float,
    constraints: CuttingConstraints,
    n_length_bins: int = 200,
    n_width_bins: int = 40,
    change_point_hints: Optional[List[int]] = None,
    severity_col: str = "severity",
) -> Tuple[float, List[float]]:
    """Find optimal transversal cut positions along the coil length.

    Parameters
    ----------
    defects_df : DataFrame
        All defects for this coil.
    coil_length, coil_width : float
        Physical dimensions (metres).
    constraints : CuttingConstraints
    n_length_bins : int
        Number of candidate cut positions along the length.
    n_width_bins : int
        Number of candidate cut positions along the width.
    change_point_hints : list of int, optional
        Preferred cut positions (bin indices).
    severity_col : str
        Column with predicted severity.

    Returns
    -------
    best_value : float
    cut_positions : list of float (metres)
    """
    bin_length = coil_length / n_length_bins
    min_bins = max(1, int(np.ceil(constraints.min_piece_length / bin_length)))

    hint_set = set(change_point_hints or [])
    hint_bonus = 0.01

    dp = np.full(n_length_bins + 1, -np.inf)
    dp[0] = 0.0
    parent = np.full(n_length_bins + 1, -1, dtype=int)

    for i in range(min_bins, n_length_bins + 1):
        x_end_i = i * bin_length
        for j in range(0, i - min_bins + 1):
            if dp[j] == -np.inf:
                continue
            x_start_j = j * bin_length

            # Evaluate this sheet: run longitudinal DP
            sheet_value, _ = _dp_longitudinal(
                defects_df, x_start_j, x_end_i,
                coil_width, constraints, n_width_bins, severity_col,
            )

            bonus = hint_bonus if i in hint_set else 0.0
            val = dp[j] + sheet_value + bonus

            if val > dp[i]:
                dp[i] = val
                parent[i] = j

    # Backtrack
    cuts_bins: List[int] = []
    pos = n_length_bins
    while pos > 0 and parent[pos] >= 0:
        cuts_bins.append(parent[pos])
        pos = parent[pos]
    cuts_bins = sorted(set(cuts_bins) - {0})
    cuts_metres = [c * bin_length for c in cuts_bins]

    if cuts_metres:
        assert cuts_metres == sorted(set(cuts_metres)), \
            f"Transversal backtracking produced unsorted/duplicate cuts: {cuts_metres}"
        assert all(0.0 < c < coil_length for c in cuts_metres), \
            f"Transversal backtracking produced out-of-range cuts: {cuts_metres}"

    best = float(dp[n_length_bins]) if dp[n_length_bins] > -np.inf else 0.0
    return best, cuts_metres


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cutting_plan(
    defects_df: pd.DataFrame,
    constraints: CuttingConstraints,
    coil_length: float,
    coil_width: float,
    quality_heatmap: Optional[np.ndarray] = None,
    change_point_hints: Optional[List[int]] = None,
    n_length_bins: int = 200,
    n_width_bins: int = 40,
    severity_col: str = "severity",
) -> CuttingPlan:
    """Compute an optimal 2D guillotine cutting plan.

    Stage 1: transversal cuts along the coil length.
    Stage 2: longitudinal cuts per sheet along the width.

    Grading is **defect-driven**: each candidate piece is graded by the
    maximum severity of defects within its bounds (not the heatmap).

    Parameters
    ----------
    defects_df : DataFrame
        Defects with columns ``x``, ``y``, and *severity_col*.
    constraints : CuttingConstraints
        Customer quality requirements and economic parameters.
    coil_length, coil_width : float
        Physical coil dimensions (metres).
    quality_heatmap : np.ndarray, optional
        Kept for visualisation; not used for grading.
    change_point_hints : list of int, optional
        Change-point indices from change-point detection.
    n_length_bins, n_width_bins : int
        Resolution of the candidate cut grid.
    severity_col : str
        Column with predicted severity in [0, 1].

    Returns
    -------
    CuttingPlan
        Optimised plan with pieces, cuts, value, and yield metrics.
    """
    bin_length = coil_length / n_length_bins
    bin_width = coil_width / n_width_bins

    # Stage 1: transversal cuts
    total_value, trans_cuts = _dp_transversal(
        defects_df, coil_length, coil_width, constraints,
        n_length_bins, n_width_bins, change_point_hints, severity_col,
    )

    # Build sheet boundaries from transversal cuts (metres)
    sheet_bounds = sorted(set([0.0] + trans_cuts + [coil_length]))

    # Stage 2: longitudinal cuts per sheet + collect pieces
    pieces: List[Piece] = []
    long_cuts_map: Dict[int, List[float]] = {}

    for s_idx in range(len(sheet_bounds) - 1):
        x_s = sheet_bounds[s_idx]
        x_e = sheet_bounds[s_idx + 1]

        _, l_cuts = _dp_longitudinal(
            defects_df, x_s, x_e, coil_width,
            constraints, n_width_bins, severity_col,
        )
        long_cuts_map[s_idx] = l_cuts

        strip_bounds = sorted(set([0.0] + l_cuts + [coil_width]))

        for t_idx in range(len(strip_bounds) - 1):
            y_s = strip_bounds[t_idx]
            y_e = strip_bounds[t_idx + 1]

            grade = compute_piece_grade_from_defects(
                defects_df, x_s, x_e, y_s, y_e,
                constraints.quality_thresholds, severity_col,
            )
            area = (x_e - x_s) * (y_e - y_s)
            value = compute_piece_value(grade, area, constraints.grade_values)

            # Compute piece stats
            mask = (
                (defects_df["x"] >= x_s) & (defects_df["x"] < x_e)
                & (defects_df["y"] >= y_s) & (defects_df["y"] < y_e)
            )
            piece_defects = defects_df.loc[mask, severity_col]
            max_sev = float(piece_defects.max()) if not piece_defects.empty else 0.0
            n_def = int(mask.sum())

            pieces.append(Piece(
                x_start=x_s, x_end=x_e,
                y_start=y_s, y_end=y_e,
                grade=grade,
                grade_name=constraints.grade_names[grade],
                value=value,
                max_severity=max_sev,
                n_defects=n_def,
            ))

    # Compute summary
    total_val = sum(p.value for p in pieces)
    total_area = coil_length * coil_width
    max_val = total_area * constraints.grade_values[0]
    yield_pct = (total_val / max_val * 100.0) if max_val > 0 else 0.0

    grade_dist: Dict[str, int] = {}
    area_by_grade: Dict[str, float] = {}
    for name in constraints.grade_names:
        grade_dist[name] = sum(1 for p in pieces if p.grade_name == name)
        area_by_grade[name] = float(sum(p.area for p in pieces if p.grade_name == name))

    return CuttingPlan(
        pieces=pieces,
        total_value=total_val,
        max_possible_value=max_val,
        yield_pct=yield_pct,
        n_pieces=len(pieces),
        grade_distribution=grade_dist,
        area_by_grade=area_by_grade,
        transversal_cuts=trans_cuts,
        longitudinal_cuts=long_cuts_map,
    )

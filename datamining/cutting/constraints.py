"""Quality constraints and grading for the cutting optimiser.

Grading is based on the **worst defect** (maximum predicted severity)
that falls within a piece's physical bounds.  This is the standard
industrial approach: a single critical defect downgrades the entire
piece.  The heatmap is kept for visualisation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CuttingConstraints:
    """Customer quality requirements for cutting optimisation.

    Attributes
    ----------
    min_piece_length : float
        Minimum acceptable piece length in metres.
    min_piece_width : float
        Minimum acceptable piece width in metres.
    quality_thresholds : tuple of float
        Ascending severity boundaries between grades.
        Applied to the **maximum defect severity** within a piece.
    grade_values : tuple of float
        Economic value (EUR/m²) for each grade.
    grade_names : tuple of str
        Human-readable grade labels.
    """

    min_piece_length: float = 5.0
    min_piece_width: float = 0.3
    quality_thresholds: Tuple[float, ...] = (0.20, 0.40, 0.60)
    grade_values: Tuple[float, ...] = (1.0, 0.7, 0.4, 0.05)
    grade_names: Tuple[str, ...] = ("A", "B", "C", "scrap")


def compute_piece_grade_from_defects(
    defects_df: pd.DataFrame,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    thresholds: Tuple[float, ...],
    severity_col: str = "severity",
) -> int:
    """Determine quality grade from defects inside a rectangular piece.

    The grade is driven by the **maximum severity** of any defect that
    falls within the piece bounds.  This is physically correct: a single
    critical defect (e.g. edge crack) downgrades the entire sheet.

    Parameters
    ----------
    defects_df : DataFrame
        Must have columns ``x``, ``y``, and *severity_col*.
    x_start, x_end, y_start, y_end : float
        Physical bounds of the piece (metres).
    thresholds : tuple of float
        Ascending severity boundaries.
    severity_col : str
        Column with predicted severity values in [0, 1].

    Returns
    -------
    int
        Grade index (0 = best / A, len(thresholds) = worst / scrap).
    """
    mask = (
        (defects_df["x"] >= x_start)
        & (defects_df["x"] < x_end)
        & (defects_df["y"] >= y_start)
        & (defects_df["y"] < y_end)
    )
    piece_defects = defects_df.loc[mask, severity_col]

    if piece_defects.empty:
        return 0  # no defects → grade A

    max_sev = float(piece_defects.max())
    grade = 0
    for t in thresholds:
        if max_sev >= t:
            grade += 1
    return grade


def compute_piece_grade(
    heatmap_slice: np.ndarray,
    thresholds: Tuple[float, ...],
) -> int:
    """Legacy heatmap-based grading (kept for backward compatibility).

    Uses the **maximum** cell value in the slice.
    """
    if heatmap_slice.size == 0:
        return 0
    sev = float(np.max(heatmap_slice))
    grade = 0
    for t in thresholds:
        if sev >= t:
            grade += 1
    return grade


def compute_piece_value(
    grade: int,
    area: float,
    grade_values: Tuple[float, ...],
) -> float:
    """Economic value of a piece: area × grade value."""
    if grade < 0 or grade >= len(grade_values):
        return 0.0
    return area * grade_values[grade]

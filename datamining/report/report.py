"""
Datamining & Cutting Plan -- PDF Report Generator
==================================================

Generates a multi-page PDF report summarising the cutting plan
optimisation for a single coil.

Pages:
    1.  Cover              — coil info, yield, grade breakdown
    2.  Defect map         — scatter of defects on coil surface
    3.  Quality heatmap    — 2D severity with change-point lines
    4.  Cutting plan       — heatmap + cut lines + grade labels
    5.  Yield breakdown    — per-piece value, grade pie chart
    6.  Severity distribution — histogram with grade thresholds
    7.  SHAP importance    — bar chart of mean |SHAP|
    8.  SHAP dependence    — top-3 feature dependence plots
    9.  SHAP interactions  — interaction heatmap
    10. Boruta selection   — confirmed / tentative / rejected features
    11. Optuna history     — convergence, parallel coords, importances

Uses only matplotlib (PdfPages).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

GRADE_COLOURS = {
    "A": "#2ecc71",
    "B": "#f39c12",
    "C": "#e74c3c",
    "scrap": "#7f8c8d",
}

_HEADER_COLOUR = "#2c3e50"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _header(ax: plt.Axes, title: str) -> None:
    """Draw a page header on the given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color=_HEADER_COLOUR)
    ax.axis("off")


def _footer(ax: plt.Axes) -> None:
    """Draw a page footer with timestamp."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, f"Generated {datetime.now():%Y-%m-%d %H:%M}",
            ha="center", va="center", fontsize=7, color="grey")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Page functions
# ---------------------------------------------------------------------------


def _page_cover(pdf: PdfPages, coil_id: str, coil_length: float,
                coil_width: float, cutting_plan, n_defects: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))

    ax_h = fig.add_axes([0.05, 0.88, 0.9, 0.10])
    _header(ax_h, "Datamining & Cutting Plan Report")

    ax = fig.add_axes([0.1, 0.3, 0.8, 0.50])
    ax.axis("off")
    info = (
        f"Coil ID: {coil_id}\n"
        f"Dimensions: {coil_length:.0f} m × {coil_width:.2f} m\n"
        f"Total defects: {n_defects}\n"
        f"Number of pieces: {cutting_plan.n_pieces}\n"
        f"Total value: {cutting_plan.total_value:.1f} EUR\n"
        f"Yield: {cutting_plan.yield_pct:.1f}%\n\n"
        f"Grade distribution:\n"
    )
    for name, count in cutting_plan.grade_distribution.items():
        info += f"  {name}: {count} pieces\n"
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace")

    ax_f = fig.add_axes([0.05, 0.02, 0.9, 0.05])
    _footer(ax_f)
    pdf.savefig(fig)
    plt.close(fig)


def _page_defect_map(pdf: PdfPages, defects_df: pd.DataFrame,
                     coil_length: float, coil_width: float) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax_h = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    _header(ax_h, "Defect Map")

    if "severity" in defects_df.columns:
        color = defects_df["severity"].to_numpy()
    elif "severity_gt" in defects_df.columns:
        color = defects_df["severity_gt"].to_numpy()
    else:
        color = np.zeros(len(defects_df))

    sc = ax.scatter(
        defects_df["x"], defects_df["y"],
        c=color,
        cmap="YlOrRd", s=15, alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    ax.set_xlim(0, coil_length)
    ax.set_ylim(0, coil_width)
    ax.set_xlabel("Position along length (m)")
    ax.set_ylabel("Position along width (m)")
    ax.set_aspect("auto")
    plt.colorbar(sc, ax=ax, label="Severity")

    pdf.savefig(fig)
    plt.close(fig)


def _page_quality_heatmap(pdf: PdfPages, heatmap: np.ndarray,
                          coil_length: float, coil_width: float) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax_h = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    _header(ax_h, "Quality Heatmap (Smoothed Severity)")

    im = ax.imshow(
        heatmap.T, aspect="auto", origin="lower", cmap="YlOrRd",
        extent=[0, coil_length, 0, coil_width],
    )
    ax.set_xlabel("Position along length (m)")
    ax.set_ylabel("Position along width (m)")
    plt.colorbar(im, ax=ax, label="Severity")

    pdf.savefig(fig)
    plt.close(fig)


def _page_cutting_plan(pdf: PdfPages, heatmap: np.ndarray,
                       cutting_plan, coil_length: float,
                       coil_width: float, constraints) -> None:
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax_h = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    _header(ax_h, "Cutting Plan Overlay")

    # Draw grade-coloured fills for each piece
    for piece in cutting_plan.pieces:
        colour = GRADE_COLOURS.get(piece.grade_name, "#95a5a6")
        rect = Rectangle(
            (piece.x_start, piece.y_start),
            piece.x_end - piece.x_start,
            piece.y_end - piece.y_start,
            facecolor=colour, alpha=0.35, edgecolor="black", linewidth=0.8,
        )
        ax.add_patch(rect)

    # Draw transversal cuts
    for x_cut in cutting_plan.transversal_cuts:
        ax.axvline(x_cut, color="blue", linewidth=1.5, linestyle="-",
                   alpha=0.8)

    # Draw longitudinal cuts
    for sheet_idx, l_cuts in cutting_plan.longitudinal_cuts.items():
        for y_cut in l_cuts:
            ax.axhline(y_cut, color="green", linewidth=1.0,
                       linestyle="--", alpha=0.7)

    # Label pieces with grade + defect count
    for piece in cutting_plan.pieces:
        cx = (piece.x_start + piece.x_end) / 2
        cy = (piece.y_start + piece.y_end) / 2
        colour = GRADE_COLOURS.get(piece.grade_name, "black")
        n_def = getattr(piece, "n_defects", 0)
        label = f"{piece.grade_name}\n({n_def})"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=6, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85))

    ax.set_xlim(0, coil_length)
    ax.set_ylim(0, coil_width)
    ax.set_xlabel("Position along length (m)")
    ax.set_ylabel("Position along width (m)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GRADE_COLOURS[g], alpha=0.5, edgecolor="black",
              label=f"Grade {g}")
        for g in constraints.grade_names
        if cutting_plan.grade_distribution.get(g, 0) > 0
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


def _page_yield_breakdown(pdf: PdfPages, cutting_plan, constraints) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax_h = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    _header(ax_h, "Yield Breakdown")

    # Pie chart of grade distribution
    ax_pie = fig.add_axes([0.05, 0.3, 0.4, 0.55])
    labels = []
    sizes = []
    colours = []
    for name in constraints.grade_names:
        count = cutting_plan.grade_distribution.get(name, 0)
        if count > 0:
            labels.append(f"{name} ({count})")
            sizes.append(count)
            colours.append(GRADE_COLOURS.get(name, "#95a5a6"))
    if sizes:
        ax_pie.pie(sizes, labels=labels, colors=colours, autopct="%1.0f%%",
                   startangle=90)
    ax_pie.set_title("Grade distribution (piece count)")

    # Bar chart: value per piece
    ax_bar = fig.add_axes([0.55, 0.3, 0.40, 0.55])
    if cutting_plan.pieces:
        pieces_sorted = sorted(cutting_plan.pieces,
                               key=lambda p: p.x_start)
        bar_colours = [GRADE_COLOURS.get(p.grade_name, "#95a5a6")
                       for p in pieces_sorted]
        ax_bar.bar(range(len(pieces_sorted)),
                   [p.value for p in pieces_sorted],
                   color=bar_colours, edgecolor="k", linewidth=0.3)
        ax_bar.set_xlabel("Piece index")
        ax_bar.set_ylabel("Value (EUR)")
        ax_bar.set_title("Value per piece")

    ax_f = fig.add_axes([0.05, 0.02, 0.9, 0.05])
    _footer(ax_f)
    pdf.savefig(fig)
    plt.close(fig)


def _page_severity_distribution(pdf: PdfPages, defects_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax_h = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    _header(ax_h, "Severity Distribution")

    sev_col = "severity" if "severity" in defects_df.columns else "severity_gt"
    ax.hist(defects_df[sev_col], bins=50, color="#3498db",
            edgecolor="white", alpha=0.8)

    # Show per-defect severity percentile markers instead of the heatmap
    # thresholds (which apply to the normalised 2D grid, not raw scores).
    sev_vals = defects_df[sev_col].dropna().values
    for q, label, colour in [
        (0.50, "Median", "#2ecc71"),
        (0.75, "P75", "#f39c12"),
        (0.90, "P90", "#e74c3c"),
        (0.95, "P95", "#8e44ad"),
    ]:
        v = float(np.percentile(sev_vals, q * 100))
        ax.axvline(v, color=colour, linewidth=1.5, linestyle="--")
        ax.text(v + 0.01, ax.get_ylim()[1] * 0.95,
                f"{label} ({v:.2f})", fontsize=9, color=colour)

    ax.set_xlabel("Severity score")
    ax.set_ylabel("Count")

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------


def _save_existing_fig(pdf: PdfPages, fig) -> None:
    """Save a pre-built matplotlib Figure into the PDF, then close it."""
    if fig is None:
        return
    pdf.savefig(fig)
    plt.close(fig)


def generate_datamining_report(
    coil_id: str,
    defects_df: pd.DataFrame,
    quality_heatmap: np.ndarray,
    cutting_plan,
    severity_model=None,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    coil_length: float = 1000.0,
    coil_width: float = 1.5,
    constraints=None,
    output_path: str = "datamining/outputs/reports/cutting_report.pdf",
    # --- New figures from Boruta / Optuna / SHAP extensions ----
    boruta_fig=None,
    optuna_fig=None,
    shap_dependence_fig=None,
    shap_interaction_fig=None,
) -> str:
    """Generate a multi-page PDF report for the cutting plan.

    Returns the path to the generated PDF.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with PdfPages(output_path) as pdf:
        # Page 1: Cover
        _page_cover(pdf, coil_id, coil_length, coil_width,
                    cutting_plan, len(defects_df))

        # Page 2: Defect map
        _page_defect_map(pdf, defects_df, coil_length, coil_width)

        # Page 3: Quality heatmap
        _page_quality_heatmap(pdf, quality_heatmap, coil_length, coil_width)

        # Page 4: Cutting plan overlay
        if constraints is not None:
            _page_cutting_plan(pdf, quality_heatmap, cutting_plan,
                               coil_length, coil_width, constraints)

        # Page 5: Yield breakdown
        if constraints is not None:
            _page_yield_breakdown(pdf, cutting_plan, constraints)

        # Page 6: Severity distribution
        _page_severity_distribution(pdf, defects_df)

        # Page 7: SHAP bar chart (if provided)
        if shap_values is not None and feature_names is not None:
            try:
                fig_bar, ax_bar = plt.subplots(figsize=(11.69, 8.27))
                mean_abs = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-15:]
                ax_bar.barh(
                    [feature_names[i] for i in top_idx],
                    mean_abs[top_idx],
                    color="#3498db",
                )
                ax_bar.set_xlabel("Mean |SHAP value|")
                ax_bar.set_title("Top 15 Feature Importances (SHAP)")
                fig_bar.tight_layout()
                pdf.savefig(fig_bar)
                plt.close(fig_bar)
            except Exception:
                pass

        # Page 8: SHAP dependence plots (top-3 features)
        _save_existing_fig(pdf, shap_dependence_fig)

        # Page 9: SHAP interaction heatmap
        _save_existing_fig(pdf, shap_interaction_fig)

        # Page 10: Boruta feature selection summary
        _save_existing_fig(pdf, boruta_fig)

        # Page 11: Optuna optimisation history
        _save_existing_fig(pdf, optuna_fig)

    return output_path

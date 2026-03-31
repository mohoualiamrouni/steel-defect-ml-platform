"""
A/B Testing PDF Report Generator
==================================

Generates a multi-page PDF report from an A/B comparison DataFrame.

Pages:
    1. Cover page  — title, key scalar metrics, run metadata
    2. Statistical significance — Welch t-test, Cohen's d, 95% CI, power analysis
    3. Disagreement rate bar chart  (annotated with p-value and effect size)
    4. Confidence delta distribution  (histogram + statistics box)
    5. Per-class agreement rates  (classification only)
    6. IoU / segmentation comparison  (segmentation only)
    7. Confidence scatter  — Model A vs Model B per image
    8. Summary table  — all metrics in one printable page
    9. Calibration analysis — reliability diagram, ECE/MCE, pre/post temperature scaling
   10. Uncertainty decomposition — aleatoric vs epistemic per class (BALD only)

Uses only matplotlib (PdfPages) — no external PDF library required.

Usage
-----
    from steeldefect_active_learning.ab_testing.report import generate_ab_report

    generate_ab_report(
        df,
        output_path="outputs/reports/ab_report.pdf",
        model_a_name="ResNet-50 v1",
        model_b_name="ResNet-50 v2",
        task="classification",
    )
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from steeldefect_active_learning.ab_testing.metrics import (
    aleatoric_epistemic_report,
    calibration_under_shift,
    confidence_delta_summary,
    disagreement_rate,
    expected_calibration_error,
    maximum_calibration_error,
    per_class_agreement,
    power_analysis,
    required_sample_size,
    statistical_significance_test,
    _STEELDEFECT_CLASS_NAMES,
)

_CLASS_NAMES = _STEELDEFECT_CLASS_NAMES  # local alias


def _class_label(class_id: int) -> str:
    try:
        return _CLASS_NAMES[int(class_id)]
    except (IndexError, ValueError, TypeError):
        return str(class_id)


def _fmt_pvalue(p: float) -> str:
    """Format p-value for display: scientific notation for very small values."""
    if p >= 0.001:
        return f"{p:.4f}"
    if p >= 1e-300:
        return f"{p:.2e}"
    return "< 1e-300"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ab_report(
    df: pd.DataFrame,
    output_path: str = "outputs/reports/ab_report.pdf",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    task: str = "classification",
    alpha: float = 0.05,
) -> str:
    """Generate a multi-page PDF report for an A/B comparison.

    Args:
        df:            A/B comparison DataFrame (output of run_ab_comparison).
        output_path:   Where to save the PDF.
        model_a_name:  Display name for Model A.
        model_b_name:  Display name for Model B.
        task:          'classification' or 'segmentation'.
        alpha:         Significance level for all statistical tests.

    Returns:
        Absolute path to the saved PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    stat  = statistical_significance_test(df, alpha=alpha)
    pwr   = power_analysis(df, alpha=alpha)
    delta = confidence_delta_summary(df)
    dr    = disagreement_rate(df)

    with PdfPages(output_path) as pdf:
        _page_cover(pdf, df, stat, pwr, dr, model_a_name, model_b_name, task, alpha)
        _page_statistics(pdf, stat, pwr, delta, model_a_name, model_b_name, alpha)
        _page_mcnemar(pdf, df, model_a_name, model_b_name)
        _page_confidence_delta(pdf, df, stat, pwr)
        if task == "classification" and "pred_a" in df.columns:
            _page_per_class_agreement(pdf, df)
        _page_per_class_delta(pdf, df, model_a_name, model_b_name)
        if task == "segmentation" and "iou" in df.columns:
            _page_iou(pdf, df)
        _page_confidence_scatter(pdf, df, stat, model_a_name, model_b_name)
        _page_kde_violin(pdf, df, model_a_name, model_b_name)
        _page_entropy_disagreement(pdf, df, model_a_name, model_b_name)
        _page_calibration_analysis(pdf, df, model_a_name)
        _page_summary_table(pdf, df, stat, pwr, delta, dr, model_a_name, model_b_name, alpha)
        # Only emit uncertainty decomposition page if BALD columns exist
        if {"bald_epistemic", "bald_aleatoric"}.issubset(df.columns):
            _page_uncertainty_decomposition(pdf, df)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"A/B Testing Report — {model_a_name} vs {model_b_name}"
        d["Subject"] = "Steel Surface Inspection A/B Test"
        d["Creator"] = "steeldefect_active_learning.ab_testing.report"

    print(f"[report] A/B PDF saved: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

_BLUE   = "#2196F3"
_GREEN  = "#4CAF50"
_RED    = "#F44336"
_ORANGE = "#FF9800"
_GREY   = "#607D8B"
_DARK   = "#212121"


def _header(fig, title: str, subtitle: str = "") -> None:
    """Add a consistent page header band."""
    fig.text(0.5, 0.97, title,
             ha="center", va="top", fontsize=14, fontweight="bold", color=_DARK)
    if subtitle:
        fig.text(0.5, 0.935, subtitle,
                 ha="center", va="top", fontsize=9, color=_GREY)


def _footer(fig) -> None:
    fig.text(0.5, 0.01,
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
             "SteelDefect Active Learning — A/B Testing Report",
             ha="center", va="bottom", fontsize=7, color=_GREY)


# ── Page 1: Cover ──────────────────────────────────────────────────────────

def _page_cover(pdf, df, stat, pwr, dr, name_a, name_b, task, alpha):
    fig = plt.figure(figsize=(8.27, 11.69))   # A4
    _footer(fig)

    # Title block
    fig.text(0.5, 0.88, "A/B Testing Report",
             ha="center", fontsize=22, fontweight="bold", color=_DARK)
    fig.text(0.5, 0.83, f"{name_a}  vs  {name_b}",
             ha="center", fontsize=14, color=_BLUE)
    fig.text(0.5, 0.80, datetime.now().strftime("%d %B %Y  •  %H:%M"),
             ha="center", fontsize=10, color=_GREY)
    fig.text(0.5, 0.77, f"Task: {task.capitalize()}  •  α = {alpha}  •  n = {len(df)} images",
             ha="center", fontsize=10, color=_GREY)

    # Horizontal rule
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.745, 0.745],
                               transform=fig.transFigure, color=_GREY, lw=0.8))

    # Compute accuracy if label + predictions available
    acc_a = acc_b = None
    if {"label", "pred_a", "pred_b"}.issubset(df.columns):
        acc_a = float((df["pred_a"] == df["label"]).mean())
        acc_b = float((df["pred_b"] == df["label"]).mean())

    # Key metrics grid (3 × 3)
    metrics = [
        ("Accuracy A",          f"{acc_a:.1%}" if acc_a is not None else "n/a",
         _GREEN if acc_a and acc_a >= 0.80 else _ORANGE if acc_a else _GREY),
        ("Accuracy B",          f"{acc_b:.1%}" if acc_b is not None else "n/a",
         _GREEN if acc_b and acc_b >= 0.80 else _ORANGE if acc_b else _GREY),
        ("Accuracy Δ (A-B)",    f"{acc_a - acc_b:+.1%}" if acc_a is not None else "n/a",
         _GREEN if acc_a and acc_a > acc_b else _RED if acc_a else _GREY),
        ("Disagreement Rate",   f"{dr:.1%}",
         _RED if dr > 0.15 else _ORANGE if dr > 0.05 else _GREEN),
        ("p-value (Welch t)",   _fmt_pvalue(stat['p_value']),
         _RED if stat["significant"] else _GREEN),
        ("Cohen's d",           f"{stat['cohens_d']:.3f}  ({stat['effect_magnitude']})",
         _ORANGE),
        ("Statistical Result",  "SIGNIFICANT" if stat["significant"] else "NOT SIGNIFICANT",
         _RED if stat["significant"] else _GREEN),
        ("Achieved Power",      f"{pwr['achieved_power']:.1%}",
         _GREEN if pwr["adequately_powered"] else _ORANGE),
        ("Mean Conf. Δ",        f"{df['confidence_delta'].mean():.4f}", _BLUE),
    ]

    cols, rows = 3, 3
    box_w, box_h = 0.24, 0.085
    x_starts = [0.08, 0.37, 0.66]
    y_starts  = [0.61, 0.50, 0.39]

    for idx, (label, value, color) in enumerate(metrics):
        r, c = divmod(idx, cols)
        x, y = x_starts[c], y_starts[r]
        ax = fig.add_axes([x, y, box_w, box_h])
        ax.set_facecolor(color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.72, label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.28, value, ha="center", va="center",
                fontsize=11, color="white", transform=ax.transAxes)

    # Data quality warnings (P12 + P13)
    warnings = []
    if stat["std_a"] < 0.01 or stat["std_b"] < 0.01:
        warnings.append(
            f"Low variance detected (std_A={stat['std_a']:.4f}, "
            f"std_B={stat['std_b']:.4f}). "
            "t-stat and Cohen's d may be inflated.")
    if pwr["required_n_for_80pct"] <= 5:
        warnings.append(
            f"Required n for 80% power = {pwr['required_n_for_80pct']} "
            "(extremely low — likely synthetic/demo data).")
    # Demo mode detection: check if image paths contain 'sample_images' or 'demo'
    demo_mode = False
    if "image_path" in df.columns:
        paths = df["image_path"].astype(str)
        demo_mode = paths.str.contains("sample_images|demo|synthetic", case=False).any()
    if demo_mode:
        warnings.append("DEMO MODE: using synthetic/sample data, not real model inference.")

    warn_y = 0.32
    if warnings:
        warn_text = "Data Quality Warnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        fig.text(0.5, warn_y, warn_text,
                 ha="center", fontsize=8, color="#C62828", linespacing=1.6,
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE",
                           edgecolor="#C62828", alpha=0.9))
        warn_y -= 0.06

    # Reference note
    fig.text(0.5, warn_y - 0.02,
             "Statistical framework: Quin & Weyns (2023) AutoPABS\n"
             "Welch's t-test (unequal variance) + Cohen's d effect size\n"
             "Power analysis: n_required = 2*(z_a/2 + z_b)^2 * sigma^2 / delta^2",
             ha="center", fontsize=8, color=_GREY, linespacing=1.8,
             style="italic")

    # Model descriptions
    fig.text(0.5, 0.22, "Models compared:",
             ha="center", fontsize=9, color=_DARK, fontweight="bold")
    fig.text(0.5, 0.19, f"A (baseline):   {name_a}",
             ha="center", fontsize=9, color=_DARK)
    fig.text(0.5, 0.16, f"B (challenger): {name_b}",
             ha="center", fontsize=9, color=_DARK)

    # Go/no-go deployment recommendation
    try:
        per_class = df.groupby("label").apply(
            lambda g: (g["pred_a"] == g["pred_b"]).mean()
        )
        min_agreement = float(per_class.min())
    except Exception:
        min_agreement = 1.0

    overall_agreement = 1.0 - dr
    if min_agreement < 0.80:
        verdict      = f"STOP  —  per-class agreement < 80 % on ≥ 1 class"
        verdict_color = _RED
    elif overall_agreement >= 0.95:
        verdict      = "GO  —  overall agreement ≥ 95 %"
        verdict_color = _GREEN
    elif dr < 0.10:
        verdict      = "LIKELY GO  —  low disagreement rate"
        verdict_color = _ORANGE
    else:
        verdict      = "REVIEW  —  spot-check disagreement cases before deploying"
        verdict_color = _ORANGE

    ax_v = fig.add_axes([0.08, 0.05, 0.84, 0.07])
    ax_v.set_facecolor(verdict_color)
    ax_v.set_xlim(0, 1); ax_v.set_ylim(0, 1)
    ax_v.set_xticks([]); ax_v.set_yticks([])
    for spine in ax_v.spines.values():
        spine.set_visible(False)
    ax_v.text(0.5, 0.55, "Deployment Recommendation", ha="center", va="center",
              fontsize=8, color="white", transform=ax_v.transAxes)
    ax_v.text(0.5, 0.20, verdict, ha="center", va="center",
              fontsize=11, fontweight="bold", color="white", transform=ax_v.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 2: Statistical significance ───────────────────────────────────────

def _page_statistics(pdf, stat, pwr, delta, name_a, name_b, alpha):
    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Statistical Significance Analysis",
            "Quin & Weyns (2023) — Welch's t-test + Cohen's d + Power Analysis")
    _footer(fig)

    ax = fig.add_axes([0.08, 0.15, 0.84, 0.72])
    ax.axis("off")

    sections = [
        ("Hypothesis Test (Welch's t-test, unequal variance)", [
            ("H0", "mu_A = mu_B  (no difference between models)"),
            ("H1", "mu_A != mu_B  (two-tailed)"),
            ("t-statistic",      f"{stat['t_stat']:.4f}"),
            ("p-value",          f"{_fmt_pvalue(stat['p_value'])}  ->  {'REJECT H0' if stat['significant'] else 'FAIL TO REJECT H0'}"),
            ("Significance",     f"alpha = {alpha}  ->  {'SIGNIFICANT' if stat['significant'] else 'NOT SIGNIFICANT'}"),
            ("95% CI (A - B)",   f"({stat['confidence_interval_95'][0]:.4f},  {stat['confidence_interval_95'][1]:.4f})"),
        ]),
        ("Effect Size - Cohen's d", [
            ("Formula",          "d = (mu_A - mu_B) / sigma_pooled"),
            ("Cohen's d",        f"{stat['cohens_d']:.4f}"),
            ("Magnitude",        stat["effect_magnitude"].upper()),
            ("Mean A",           f"{stat['mean_a']:.4f}  (std = {stat['std_a']:.4f})"),
            ("Mean B",           f"{stat['mean_b']:.4f}  (std = {stat['std_b']:.4f})"),
            ("Sample size",      f"n = {stat['n']} images"),
        ]),
        ("Power Analysis", [
            ("Observed effect",  f"{pwr['observed_effect']:.4f}"),
            ("Pooled sigma",     f"{pwr['sigma']:.4f}"),
            ("Achieved power",   f"{pwr['achieved_power']:.1%}  ({'adequate [OK]' if pwr['adequately_powered'] else 'insufficient [!!]'})"),
            ("Required n (80% pwr)",f"{pwr['required_n_for_80pct']} per group"
             + ("  (very large effect — statistical minimum, not operational)" if pwr['required_n_for_80pct'] < 30 else "")),
            ("Current n",           f"{pwr['n']} per group"),
        ]),
        ("Confidence Delta Summary", [
            ("Mean |conf_A − conf_B|",   f"{delta['mean']:.4f}"),
            ("Median",                    f"{delta['median']:.4f}"),
            ("Std",                       f"{delta['std']:.4f}"),
            ("Max",                       f"{delta['max']:.4f}"),
            ("90th percentile",           f"{delta['p90']:.4f}"),
        ]),
    ]

    y = 0.97
    for section_title, rows in sections:
        ax.text(0.0, y, section_title, fontsize=10, fontweight="bold",
                color=_BLUE, transform=ax.transAxes)
        y -= 0.04
        for key, val in rows:
            ax.text(0.02, y, f"{key}:", fontsize=9, color=_GREY,
                    transform=ax.transAxes, fontstyle="italic")
            ax.text(0.35, y, val, fontsize=9, color=_DARK, transform=ax.transAxes)
            y -= 0.035
        y -= 0.02
        # separator line
        ax.plot([0, 1], [y + 0.01, y + 0.01], color=_GREY, lw=0.4, alpha=0.4, transform=ax.transAxes)
        y -= 0.01

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3: Disagreement bar ────────────────────────────────────────────────

def _page_disagreement_bar(pdf, df, stat, name_a, name_b):
    fig, ax = plt.subplots(figsize=(8.27, 5))
    fig.subplots_adjust(top=0.82, bottom=0.18)
    _footer(fig)

    agree    = int(df["agrees"].sum())
    disagree = len(df) - agree
    bars = ax.bar(["Agree", "Disagree"], [agree, disagree],
                  color=[_GREEN, _RED], width=0.5, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, [agree, disagree]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{count}\n({count/len(df):.1%})", ha="center", va="bottom", fontsize=11)

    p_str  = f"p = {_fmt_pvalue(stat['p_value'])}"
    d_str  = f"d = {stat['cohens_d']:.3f}  ({stat['effect_magnitude']})"
    sig_str = "[OK] Significant" if stat["significant"] else "[--] Not significant"
    ax.set_title(f"Agreement vs Disagreement - {name_a} vs {name_b}\n"
                 f"{p_str}   {d_str}   {sig_str}", fontsize=10)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(agree, disagree) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3b: McNemar's Paired Classifier Test (McNemar 1947) ──────────────

def _page_mcnemar(pdf, df, name_a, name_b):
    """Full-page McNemar's test: 2×2 contingency + statistical results."""
    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "McNemar's Paired Classifier Test (McNemar 1947)",
            r"H$_0$: both models have the same error rate  ·  "
            r"Uses only discordant pairs (n$_{01}$ vs n$_{10}$)")
    _footer(fig)

    required = {"pred_a", "pred_b", "label"}
    if not required.issubset(df.columns):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "McNemar's test requires columns:\npred_a, pred_b, label\n"
                "(ground-truth labels needed)",
                ha="center", va="center", fontsize=13, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    a_correct = (df["pred_a"] == df["label"]).values
    b_correct = (df["pred_b"] == df["label"]).values
    n00 = int((a_correct & b_correct).sum())
    n01 = int((a_correct & ~b_correct).sum())
    n10 = int((~a_correct & b_correct).sum())
    n11 = int((~a_correct & ~b_correct).sum())
    n_total = len(df)
    discordant = n01 + n10

    if discordant >= 25:
        chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = 1.0 - scipy_stats.chi2.cdf(chi2, df=1)
        test_type = "χ² with continuity correction"
    elif discordant > 0:
        p_value = float(scipy_stats.binomtest(n01, discordant, 0.5).pvalue)
        chi2 = float("nan")
        test_type = "Exact binomial test (discordant < 25)"
    else:
        chi2, p_value = 0.0, 1.0
        test_type = "No discordant pairs"

    # 2×2 heatmap
    ax_hm = fig.add_axes([0.15, 0.48, 0.35, 0.30])
    table = np.array([[n00, n01], [n10, n11]])
    im = ax_hm.imshow(table, cmap="RdYlGn", vmin=0, vmax=max(n_total // 2, 1))
    ax_hm.set_xticks([0, 1])
    ax_hm.set_xticklabels([f"{name_b}\ncorrect", f"{name_b}\nwrong"], fontsize=9)
    ax_hm.set_yticks([0, 1])
    ax_hm.set_yticklabels([f"{name_a}\ncorrect", f"{name_a}\nwrong"], fontsize=9)
    for i in range(2):
        for j in range(2):
            pct = table[i, j] / max(n_total, 1) * 100
            ax_hm.text(j, i, f"{table[i, j]}\n({pct:.1f}%)",
                       ha="center", va="center", fontsize=12, fontweight="bold",
                       color="white" if table[i, j] > n_total * 0.3 else "black")
    ax_hm.set_title("Contingency Table", fontsize=10)

    # Statistical results text
    ax_txt = fig.add_axes([0.55, 0.48, 0.40, 0.30])
    ax_txt.axis("off")
    verdict = "SIGNIFICANT — models differ" if p_value < 0.05 else "NOT SIGNIFICANT"
    vc = _RED if p_value < 0.05 else _GREEN

    lines = [
        f"Test type:  {test_type}",
        f"χ² statistic:  {chi2:.4f}" if np.isfinite(chi2) else "Exact binomial used",
        f"p-value:  {_fmt_pvalue(p_value)}",
        "",
        f"Discordant:  {discordant}  (n_01={n01}, n_10={n10})",
        f"Concordant:  {n00 + n11}  (both right={n00}, both wrong={n11})",
        "",
        f"Accuracy A:  {(n00 + n01) / max(n_total, 1):.1%}",
        f"Accuracy B:  {(n00 + n10) / max(n_total, 1):.1%}",
    ]
    y = 0.95
    for line in lines:
        ax_txt.text(0.0, y, line, fontsize=9, color=_DARK,
                    transform=ax_txt.transAxes)
        y -= 0.10

    # Verdict banner
    ax_v = fig.add_axes([0.10, 0.38, 0.80, 0.06])
    ax_v.set_facecolor(vc)
    ax_v.set_xlim(0, 1); ax_v.set_ylim(0, 1)
    ax_v.set_xticks([]); ax_v.set_yticks([])
    for s in ax_v.spines.values():
        s.set_visible(False)
    ax_v.text(0.5, 0.5, verdict, ha="center", va="center",
              fontsize=12, fontweight="bold", color="white",
              transform=ax_v.transAxes)

    # Theory note
    fig.text(0.5, 0.28,
             "McNemar (1947): Tests whether the marginal proportions of errors differ\n"
             "between two classifiers evaluated on the same test set.\n\n"
             r"$\chi^2$ = (|n$_{01}$ - n$_{10}$| - 1)$^2$ / (n$_{01}$ + n$_{10}$)   with 1 df" + "\n"
             r"If n$_{01}$ + n$_{10}$ < 25, exact binomial test is used instead." + "\n\n"
             "Key insight: Only DISCORDANT pairs matter — concordant pairs\n"
             "(both right or both wrong) are ignored.",
             ha="center", fontsize=9, color=_GREY, linespacing=1.6, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page: Per-class Accuracy Delta ────────────────────────────────────────

def _page_per_class_delta(pdf, df, name_a, name_b):
    """Horizontal bar chart: acc(B) − acc(A) per class."""
    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Per-class Accuracy Delta",
            f"acc({name_b}) − acc({name_a}) per class  ·  "
            "Green = B improved  ·  Red = B degraded")
    _footer(fig)

    required = {"pred_a", "pred_b", "label"}
    if not required.issubset(df.columns):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Accuracy delta requires columns:\npred_a, pred_b, label",
                ha="center", va="center", fontsize=13, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    from steeldefect_active_learning.ab_testing.metrics import _STEELDEFECT_CLASS_NAMES

    classes = sorted(df["label"].unique())
    deltas, acc_a_list, acc_b_list, counts = [], [], [], []
    for c in classes:
        mask = df["label"] == c
        n_c = int(mask.sum())
        acc_a = float((df.loc[mask, "pred_a"] == c).mean())
        acc_b = float((df.loc[mask, "pred_b"] == c).mean())
        deltas.append(acc_b - acc_a)
        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        counts.append(n_c)

    ax = fig.add_axes([0.20, 0.12, 0.70, 0.72])
    bar_colors = [_GREEN if d >= 0 else _RED for d in deltas]
    ax.barh(range(len(classes)), deltas, color=bar_colors,
            edgecolor="white", height=0.6)
    ax.axvline(0, color=_DARK, lw=1.2)

    x_min = min(deltas) if deltas else -0.1
    x_max = max(deltas) if deltas else 0.1
    x_range = max(abs(x_min), abs(x_max), 0.05)

    for i, (d, a_a, a_b, n_c) in enumerate(zip(deltas, acc_a_list, acc_b_list, counts)):
        tag = "DEGRADED " if d < -0.05 else ("IMPROVED " if d > 0.05 else "")
        label = f"{tag}A:{a_a:.0%}\u2192B:{a_b:.0%} (n={n_c})"
        # If bar is long (>60% of axis range), place label inside the bar
        bar_fraction = abs(d) / x_range if x_range > 0 else 0
        if bar_fraction > 0.6 and abs(d) > 0.08:
            x_pos = d * 0.5  # middle of bar
            ha = "center"
            txt_color = "white"
        else:
            x_pos = d + (0.005 if d >= 0 else -0.005)
            ha = "left" if d >= 0 else "right"
            txt_color = "#C62828" if d < -0.05 else _DARK
        ax.text(x_pos, i, label, ha=ha, va="center", fontsize=7,
                color=txt_color, fontweight="bold" if txt_color == "white" else "normal")

    try:
        y_labels = [_STEELDEFECT_CLASS_NAMES[int(c)] for c in classes]
    except (IndexError, ValueError):
        y_labels = [str(c) for c in classes]

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Accuracy delta  (acc_B − acc_A)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Summary
    n_degraded = sum(1 for d in deltas if d < -0.05)
    summary_color = _RED if n_degraded > 0 else _GREEN
    fig.text(0.5, 0.06,
             f"Classes with Δ < −5%: {n_degraded} / {len(classes)}  "
             f"{'— REVIEW BEFORE DEPLOYMENT' if n_degraded > 0 else '— ALL CLASSES STABLE'}",
             ha="center", fontsize=10, fontweight="bold", color=summary_color)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 4: Confidence delta distribution ──────────────────────────────────

def _page_confidence_delta(pdf, df, stat, pwr):
    fig, ax = plt.subplots(figsize=(8.27, 5))
    fig.subplots_adjust(top=0.85, bottom=0.15)
    _footer(fig)

    ax.hist(df["confidence_delta"], bins=40, color=_BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(df["confidence_delta"].mean(), color=_RED, linestyle="--", lw=1.5,
               label=f"mean = {df['confidence_delta'].mean():.4f}")
    ax.axvline(df["confidence_delta"].quantile(0.9), color=_ORANGE, linestyle=":",
               lw=1.5, label=f"p90 = {df['confidence_delta'].quantile(0.9):.4f}")

    p_str = f"p = {stat['p_value']:.4f}" if stat["p_value"] >= 0.0001 else "p < 0.0001"
    artifact_note = "  ⚠ synth. artifact" if stat["cohens_d"] > 3.0 else ""
    box_text = (
        f"Welch  t = {stat['t_stat']:.2f}\n"
        f"{p_str}\n"
        f"Cohen's d = {stat['cohens_d']:.3f}  ({stat['effect_magnitude']}){artifact_note}\n"
        f"Power = {pwr['achieved_power']:.1%}   n = {pwr['n']}"
    )
    ax.text(0.98, 0.97, box_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor=_GREY))

    ax.set_xlabel("|confidence_A − confidence_B|")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Delta Distribution — Quin & Weyns (2023) AutoPABS", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 5: Per-class agreement ────────────────────────────────────────────

def _page_per_class_agreement(pdf, df):
    stats = per_class_agreement(df)
    fig, ax = plt.subplots(figsize=(8.27, 5))
    fig.subplots_adjust(top=0.85, bottom=0.15)
    _footer(fig)

    colors = [_GREEN if r >= 0.8 else _ORANGE if r >= 0.5 else _RED
              for r in stats["agreement_rate"]]
    ax.bar(stats["class_id"].astype(str), stats["agreement_rate"],
           color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(0.8, color=_GREEN, linestyle="--", alpha=0.6, label="80% threshold")

    for i, (r, n) in enumerate(zip(stats["agreement_rate"], stats["count"])):
        ax.text(i, r + 0.02, f"n={n}", ha="center", fontsize=8, color=_DARK)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Class ID (predicted by Model A)")
    ax.set_ylabel("Agreement rate")
    ax.set_title("Per-class Agreement Rate (A vs B)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 6: IoU (segmentation) ─────────────────────────────────────────────

def _page_iou(pdf, df):
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 5))
    fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.35)
    _footer(fig)

    axes[0].hist(df["iou"], bins=40, color="#9C27B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["iou"].mean(), color=_RED, linestyle="--",
                    label=f"mean = {df['iou'].mean():.3f}")
    axes[0].set_xlabel("IoU (A vs B)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mask IoU Distribution")
    axes[0].legend(fontsize=9)

    if "mask_defect_fraction_a" in df.columns and "mask_defect_fraction_b" in df.columns:
        axes[1].scatter(df["mask_defect_fraction_a"], df["mask_defect_fraction_b"],
                        alpha=0.4, s=10, color="#FF5722")
        lim = max(df["mask_defect_fraction_a"].max(), df["mask_defect_fraction_b"].max()) * 1.05
        axes[1].plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect agreement")
        axes[1].set_xlabel("Defect fraction — Model A")
        axes[1].set_ylabel("Defect fraction — Model B")
        axes[1].set_title("Defect Coverage: A vs B")
        axes[1].legend(fontsize=9)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "mask_defect_fraction\nnot available",
                     ha="center", va="center", fontsize=10, color=_GREY,
                     transform=axes[1].transAxes)

    fig.suptitle("Segmentation Comparison — IoU Metrics", fontsize=10, y=0.97)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 7: Confidence scatter ─────────────────────────────────────────────

def _page_confidence_scatter(pdf, df, stat, name_a, name_b):
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 5))
    fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.35)
    _footer(fig)

    # Scatter: conf_A vs conf_B
    agree_mask = df["agrees"] == 1
    axes[0].scatter(df.loc[agree_mask, "confidence_a"],
                    df.loc[agree_mask, "confidence_b"],
                    alpha=0.3, s=8, color=_GREEN, label="Agree")
    axes[0].scatter(df.loc[~agree_mask, "confidence_a"],
                    df.loc[~agree_mask, "confidence_b"],
                    alpha=0.5, s=8, color=_RED, label="Disagree")
    # Auto-zoom to data range with 5% padding so points aren't all in one corner
    ca_all = df["confidence_a"].values
    cb_all = df["confidence_b"].values
    pad = 0.05
    xlim = (max(0, ca_all.min() - pad), min(1, ca_all.max() + pad))
    ylim = (max(0, cb_all.min() - pad), min(1, cb_all.max() + pad))
    lim_min = min(xlim[0], ylim[0])
    lim_max = max(xlim[1], ylim[1])
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, lw=1)
    axes[0].set_xlabel(f"Confidence - {name_a}")
    axes[0].set_ylabel(f"Confidence - {name_b}")
    axes[0].set_title("Confidence: A vs B\n(zoomed to data range)")
    axes[0].legend(fontsize=8, markerscale=2)
    axes[0].set_xlim(xlim); axes[0].set_ylim(ylim)

    # Confidence distribution comparison — zoom to actual data range
    hist_lo = max(0.0, min(ca_all.min(), cb_all.min()) - 0.05)
    hist_hi = min(1.0, max(ca_all.max(), cb_all.max()) + 0.05)
    bins = np.linspace(hist_lo, hist_hi, 31)
    axes[1].hist(df["confidence_a"], bins=bins, alpha=0.6,
                 color=_BLUE, label=name_a, density=True)
    axes[1].hist(df["confidence_b"], bins=bins, alpha=0.6,
                 color=_ORANGE, label=name_b, density=True)
    axes[1].axvline(stat["mean_a"], color=_BLUE, lw=1.5, linestyle="--",
                    label=f"mean_A = {stat['mean_a']:.3f}")
    axes[1].axvline(stat["mean_b"], color=_ORANGE, lw=1.5, linestyle="--",
                    label=f"mean_B = {stat['mean_b']:.3f}")
    axes[1].set_xlabel("Confidence score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence Distribution Comparison\n(zoomed to data range)")
    axes[1].set_xlim(hist_lo, hist_hi)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Confidence Comparison — {name_a} vs {name_b}", fontsize=10, y=0.97)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 8: Summary table ──────────────────────────────────────────────────

def _page_summary_table(pdf, df, stat, pwr, delta, dr, name_a, name_b, alpha):
    fig = plt.figure(figsize=(8.27, 11.69))
    _header(fig, "Complete Metrics Summary",
            f"{name_a}  vs  {name_b}   •   n = {len(df)} images   •   α = {alpha}")
    _footer(fig)

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.78])
    ax.axis("off")

    rows = [
        # Section, Key, Value, Interpretation
        ["Overview",       "Total images",           str(len(df)),             ""],
        ["",               "Task",                   df.get("task", ["classification"])[0]
                                                      if hasattr(df, "get") else "classification", ""],
        ["",               "Model A",                name_a,                    ""],
        ["",               "Model B",                name_b,                    ""],
        ["Agreement",      "Disagreement rate",      f"{dr:.2%}",
         "[+] good" if dr < 0.05 else "[!] moderate" if dr < 0.15 else "[-] high"],
        ["",               "Agreement rate",         f"{1-dr:.2%}",            ""],
        ["Welch t-test",   "t-statistic",            f"{stat['t_stat']:.4f}",  ""],
        ["",               "p-value",                _fmt_pvalue(stat['p_value']),
         "[-] significant" if stat["significant"] else "[+] not significant"],
        ["",               "Result",
         "REJECT H0" if stat["significant"] else "FAIL TO REJECT H0",
         f"alpha = {alpha}"],
        ["",               "95% CI (A-B)",
         f"({stat['confidence_interval_95'][0]:.4f}, {stat['confidence_interval_95'][1]:.4f})", ""],
        ["Effect Size",    "Cohen's d",              f"{stat['cohens_d']:.4f}",
         stat["effect_magnitude"]],
        ["",               "Mean A",                 f"{stat['mean_a']:.4f}",  ""],
        ["",               "Std A",                  f"{stat['std_a']:.4f}",   ""],
        ["",               "Mean B",                 f"{stat['mean_b']:.4f}",  ""],
        ["",               "Std B",                  f"{stat['std_b']:.4f}",   ""],
        ["Power",          "Achieved power",         f"{pwr['achieved_power']:.1%}",
         "[+] adequate" if pwr["adequately_powered"] else "[!] insufficient"],
        ["",               "Required n (80% power)", str(pwr["required_n_for_80pct"]), "per group"],
        ["",               "Observed effect Δ",      f"{pwr['observed_effect']:.4f}", ""],
        ["Conf. Delta",    "Mean |conf_A − conf_B|", f"{delta['mean']:.4f}",   ""],
        ["",               "Median",                 f"{delta['median']:.4f}", ""],
        ["",               "Std",                    f"{delta['std']:.4f}",    ""],
        ["",               "90th percentile",        f"{delta['p90']:.4f}",    ""],
        ["",               "Max",                    f"{delta['max']:.4f}",    ""],
    ]

    col_widths = [0.18, 0.38, 0.24, 0.20]
    col_x      = [0.0,  0.18, 0.56, 0.80]
    headers    = ["Section", "Metric", "Value", "Interpretation"]

    y = 0.97
    row_h = 0.036

    # Header row
    for col_label, x in zip(headers, col_x):
        ax.text(x, y, col_label, fontsize=9, fontweight="bold",
                color="white", transform=ax.transAxes,
                bbox=dict(facecolor=_BLUE, edgecolor="none",
                          boxstyle="square,pad=0.15"))
    y -= row_h * 1.3

    last_section = ""
    for row in rows:
        section, key, val, interp = row
        bg = "#F5F5F5" if section != last_section and section != "" else "white"
        if section != "":
            last_section = section

        ax.text(col_x[0], y, section, fontsize=8, color=_BLUE,
                fontweight="bold" if section else "normal",
                transform=ax.transAxes)
        ax.text(col_x[1], y, key,    fontsize=8, color=_DARK,  transform=ax.transAxes)
        ax.text(col_x[2], y, val,    fontsize=8, color=_DARK,  transform=ax.transAxes,
                fontweight="bold")
        color = _GREEN if "[+]" in interp else _RED if "[-]" in interp else _ORANGE if "[!]" in interp else _GREY
        ax.text(col_x[3], y, interp, fontsize=8, color=color,  transform=ax.transAxes)

        ax.plot([0, 1], [y - row_h * 0.1, y - row_h * 0.1], color="#E0E0E0", lw=0.4, transform=ax.transAxes)
        y -= row_h

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 9: KDE heatmap + violin per class ─────────────────────────────────

def _page_kde_violin(pdf, df, name_a, name_b):
    """2D KDE confidence density + per-class violin side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))  # A4 landscape
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.35)
    _header(fig, "Confidence Density & Per-class Distribution",
            f"{name_a} vs {name_b}")
    _footer(fig)

    # ── Left: 2D KDE heatmap ─────────────────────────────────────────────
    ax = axes[0]
    conf_a = df["confidence_a"].values.astype(float)
    conf_b = df["confidence_b"].values.astype(float)
    agree_mask = df["agrees"].astype(bool).values

    xy = np.vstack([conf_a, conf_b])
    try:
        kde = scipy_stats.gaussian_kde(xy, bw_method="silverman")
        # Zoom grid to data range + 5% padding to avoid KDE bleeding far outside data
        pad = 0.05
        xa_lo = max(0.0, float(conf_a.min()) - pad)
        xa_hi = min(1.0, float(conf_a.max()) + pad)
        xb_lo = max(0.0, float(conf_b.min()) - pad)
        xb_hi = min(1.0, float(conf_b.max()) + pad)
        grid_pts = 60
        xa = np.linspace(xa_lo, xa_hi, grid_pts)
        xb = np.linspace(xb_lo, xb_hi, grid_pts)
        XX, YY = np.meshgrid(xa, xb)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(grid_pts, grid_pts)
        ax.contourf(XX, YY, ZZ, levels=14, cmap="Blues", alpha=0.85)
        ax.contour(XX, YY, ZZ, levels=7, colors="steelblue", linewidths=0.5, alpha=0.4)
    except Exception:
        pass  # if KDE fails (e.g. singular), skip contour

    ax.scatter(conf_a[agree_mask], conf_b[agree_mask],
               s=6, alpha=0.3, color=_GREEN, label="Agree")
    ax.scatter(conf_a[~agree_mask], conf_b[~agree_mask],
               s=6, alpha=0.5, color=_RED, label="Disagree")
    # Diagonal only within visible range
    _pad = 0.05
    _xlo = max(0, conf_a.min() - _pad); _xhi = min(1, conf_a.max() + _pad)
    _ylo = max(0, conf_b.min() - _pad); _yhi = min(1, conf_b.max() + _pad)
    _lo, _hi = min(_xlo, _ylo), max(_xhi, _yhi)
    ax.plot([_lo, _hi], [_lo, _hi], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel(f"Confidence - {name_a}", fontsize=9)
    ax.set_ylabel(f"Confidence - {name_b}", fontsize=9)
    ax.set_title("2D KDE: Confidence Space\n(Silverman bandwidth, zoomed)", fontsize=9)
    ax.set_xlim(_xlo, _xhi); ax.set_ylim(_ylo, _yhi)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    # ── Right: violin per predicted class ───────────────────────────────
    ax2 = axes[1]
    if "pred_a" in df.columns:
        classes = sorted(df["pred_a"].unique())
        positions_a = [i - 0.18 for i in range(len(classes))]
        positions_b = [i + 0.18 for i in range(len(classes))]
        data_a = [df.loc[df["pred_a"] == c, "confidence_a"].values for c in classes]
        data_b = [df.loc[df["pred_a"] == c, "confidence_b"].values for c in classes]

        parts_a = ax2.violinplot(data_a, positions=positions_a, widths=0.32,
                                  showmedians=True, showextrema=False, bw_method=0.3)
        parts_b = ax2.violinplot(data_b, positions=positions_b, widths=0.32,
                                  showmedians=True, showextrema=False, bw_method=0.3)
        for pc in parts_a["bodies"]:
            pc.set_facecolor(_BLUE); pc.set_alpha(0.6)
        parts_a["cmedians"].set_color(_BLUE)
        for pc in parts_b["bodies"]:
            pc.set_facecolor(_ORANGE); pc.set_alpha(0.6)
        parts_b["cmedians"].set_color(_ORANGE)

        for i, c in enumerate(classes):
            mask_c = df["pred_a"] == c
            n_c = mask_c.sum()
            agr_c = df.loc[mask_c, "agrees"].mean()
            ax2.text(i, -0.05, f"n={n_c}\n{agr_c:.0%}", ha="center",
                     fontsize=7, color=_GREY, transform=ax2.get_xaxis_transform())

        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels([_class_label(c) for c in classes],
                            fontsize=9, rotation=30, ha="right")
        ax2.set_xlabel("Predicted class (Model A)", fontsize=9)
        patch_a = mpatches.Patch(color=_BLUE, alpha=0.6, label=name_a)
        patch_b = mpatches.Patch(color=_ORANGE, alpha=0.6, label=name_b)
        ax2.legend(handles=[patch_a, patch_b], fontsize=8)
    else:
        ax2.text(0.5, 0.5, "pred_a column\nnot available",
                 ha="center", va="center", fontsize=11, color=_GREY,
                 transform=ax2.transAxes)
        ax2.axis("off")

    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Confidence score", fontsize=9)
    ax2.set_title("Per-class Confidence Violin\n(A: blue, B: orange)", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 10: Entropy CDF + disagreement decomposition ─────────────────────

def _page_entropy_disagreement(pdf, df, name_a, name_b):
    """Empirical entropy CDF and disagreement decomposition stacked bar."""
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))  # A4 landscape
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.38)
    _header(fig, "Prediction Entropy & Disagreement Decomposition",
            f"{name_a} vs {name_b}")
    _footer(fig)

    # ── Left: Entropy distribution with KL divergence ────────────────────
    ax = axes[0]
    if "entropy_a" in df.columns and "entropy_b" in df.columns:
        bins = 35
        # Check if distributions overlap; if not, use twin axes for visibility
        range_a = (df["entropy_a"].min(), df["entropy_a"].max())
        range_b = (df["entropy_b"].min(), df["entropy_b"].max())
        overlaps = range_a[1] > range_b[0] and range_b[1] > range_a[0]
        if overlaps:
            ax.hist(df["entropy_a"], bins=bins, alpha=0.65, color=_BLUE,
                    label=f"{name_a}  \u03bc={df['entropy_a'].mean():.3f}", density=True)
            ax.hist(df["entropy_b"], bins=bins, alpha=0.65, color=_ORANGE,
                    label=f"{name_b}  \u03bc={df['entropy_b'].mean():.3f}", density=True)
        else:
            ax.hist(df["entropy_a"], bins=bins, alpha=0.65, color=_BLUE,
                    label=f"{name_a}  \u03bc={df['entropy_a'].mean():.3f}", density=True)
            ax.set_ylabel("Density (Model A)", fontsize=9, color=_BLUE)
            ax2 = ax.twinx()
            ax2.hist(df["entropy_b"], bins=bins, alpha=0.55, color=_ORANGE,
                     label=f"{name_b}  \u03bc={df['entropy_b'].mean():.3f}", density=True)
            ax2.set_ylabel("Density (Model B)", fontsize=9, color=_ORANGE)
            # Merge legends
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8)
        ax.set_xlabel("Prediction entropy (nats)", fontsize=9)
        if overlaps:
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
        ax.set_title("Entropy Distribution + KL Divergence\n"
                     "(Kullback & Leibler 1951)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # KL divergence annotation
        try:
            all_ent = pd.concat([df["entropy_a"].dropna(), df["entropy_b"].dropna()])
            kl_bin_edges = np.linspace(all_ent.min(), all_ent.max(), 51)
            hist_a_, _ = np.histogram(df["entropy_a"].dropna(), bins=kl_bin_edges, density=True)
            hist_b_, _ = np.histogram(df["entropy_b"].dropna(), bins=kl_bin_edges, density=True)
            eps = 1e-10
            ha_ = hist_a_ + eps; hb_ = hist_b_ + eps
            ha_ = ha_ / ha_.sum(); hb_ = hb_ / hb_.sum()
            kl_ab = float(np.sum(ha_ * np.log(ha_ / hb_)))
            if not np.isfinite(kl_ab) or kl_ab > 50:
                kl_text = r"D_KL(A||B) $\gg$ 0"
                kl_interp = "Non-overlapping distributions"
            elif kl_ab < 0.05:
                kl_text = f"D_KL(A||B) = {kl_ab:.4f}"
                kl_interp = "Similar"
            else:
                kl_text = f"D_KL(A||B) = {kl_ab:.4f}"
                kl_interp = "Different profiles"
            ax.text(0.97, 0.97, f"{kl_text}\n{kl_interp}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.85, edgecolor=_GREY))
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, "entropy_a / entropy_b\nnot in DataFrame",
                ha="center", va="center", fontsize=11, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")

    # ── Right: disagreement decomposition stacked bar ────────────────────
    ax2 = axes[1]
    if "margin_a" in df.columns and "margin_b" in df.columns:
        disagree = df[df["agrees"] == 0] if "agrees" in df.columns else df.iloc[0:0]
        n_total = len(df)
        n_agree = int(df["agrees"].sum()) if "agrees" in df.columns else 0

        if len(disagree) > 0:
            # Use data-relative percentile thresholds so that the decomposition
            # is meaningful regardless of absolute confidence scale.
            # "Systematic" = both models are in the TOP-25% margin of the full dataset
            # "Borderline" = at least one model is in the BOTTOM-25% margin
            all_margins = pd.concat([df["margin_a"], df["margin_b"]])
            thr_high = float(np.percentile(all_margins, 75))  # top-quartile threshold
            thr_low  = float(np.percentile(all_margins, 25))  # bottom-quartile threshold
            sys_mask  = (disagree["margin_a"] >= thr_high) & (disagree["margin_b"] >= thr_high)
            bord_mask = (disagree["margin_a"] <= thr_low)  | (disagree["margin_b"] <= thr_low)
            counts = {
                "Agree":      n_agree,
                "Borderline": int(bord_mask.sum()),
                "Random":     int((~sys_mask & ~bord_mask).sum()),
                "Systematic": int(sys_mask.sum()),
            }
            thr_note = f"thresholds: high={thr_high:.3f} (p75)  low={thr_low:.3f} (p25)"
        else:
            counts = {"Agree": n_agree, "Borderline": 0, "Random": 0, "Systematic": 0}
            thr_note = ""

        colors_bar = [_GREEN, _ORANGE, _BLUE, _RED]
        vals_bar   = list(counts.values())
        bars = ax2.bar(list(counts.keys()), vals_bar, color=colors_bar,
                       edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, vals_bar):
            pct = v / max(n_total, 1)
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + n_total * 0.01,
                     f"{v}\n({pct:.1%})", ha="center", va="bottom", fontsize=9)

        ax2.set_ylabel("Number of images", fontsize=9)
        ax2.set_title(f"Disagreement Decomposition\n"
                      f"Systematic: both top-quartile margin, disagree  |  Borderline: bottom-quartile\n"
                      f"{thr_note}", fontsize=7.5)
        ax2.set_ylim(0, max(vals_bar) * 1.3)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "margin_a / margin_b\nnot in DataFrame",
                 ha="center", va="center", fontsize=11, color=_GREY,
                 transform=ax2.transAxes)
        ax2.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 9: Calibration Analysis  (Guo et al. 2017)
# ---------------------------------------------------------------------------

def _page_calibration_analysis(
    pdf: "PdfPages",
    df: "pd.DataFrame",
    model_name: str = "Model A",
) -> None:
    """Reliability diagram + ECE/MCE + calibration-under-shift table."""
    fig = plt.figure(figsize=(11, 8))
    _header(fig, "Page 9 \u2014 Calibration Analysis",
            f"Reliability diagram and calibration metrics for {model_name}  "
            "(Guo et al. 2017)")

    required = {"confidence_a", "pred_a", "label"}
    if not required.issubset(df.columns):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Calibration analysis requires columns:\n"
                "confidence_a, pred_a, label",
                ha="center", va="center", fontsize=13, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")
        _footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    labels = df["label"].values.astype(np.int64)
    pred_a = df["pred_a"].values.astype(np.int64)
    conf_a = df["confidence_a"].values.astype(np.float64)

    n_bins = 15
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins)
    bin_acc  = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins, dtype=int)

    for m, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (conf_a > lo) & (conf_a <= hi)
        cnt = int(mask.sum())
        bin_cnt[m] = cnt
        if cnt > 0:
            bin_conf[m] = float(conf_a[mask].mean())
            bin_acc[m]  = float(np.mean(pred_a[mask] == labels[mask]))

    # Adaptive binning fallback: if < 3 bins populated, switch to quantile bins
    n_populated = int((bin_cnt > 0).sum())
    adaptive_mode = n_populated < 3
    if adaptive_mode:
        n_bins = 10
        bin_edges = np.unique(np.percentile(conf_a, np.linspace(0, 100, n_bins + 1)))
        n_bins = len(bin_edges) - 1
        bin_conf = np.zeros(n_bins)
        bin_acc  = np.zeros(n_bins)
        bin_cnt  = np.zeros(n_bins, dtype=int)
        for m, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (conf_a > lo) & (conf_a <= hi) if m > 0 else (conf_a >= lo) & (conf_a <= hi)
            cnt = int(mask.sum())
            bin_cnt[m] = cnt
            if cnt > 0:
                bin_conf[m] = float(conf_a[mask].mean())
                bin_acc[m]  = float(np.mean(pred_a[mask] == labels[mask]))

    ece = float(sum(
        (bin_cnt[m] / len(labels)) * abs(bin_acc[m] - bin_conf[m])
        for m in range(n_bins) if bin_cnt[m] > 0
    ))
    mce = float(max(
        (abs(bin_acc[m] - bin_conf[m]) for m in range(n_bins) if bin_cnt[m] > 0),
        default=0.0,
    ))

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.10,
                           hspace=0.45, wspace=0.4)
    ax_rel = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for m in range(n_bins):
        if bin_cnt[m] == 0:
            continue
        w = bin_widths[m] * 0.85
        ax_rel.bar(bin_centers[m], bin_acc[m], width=w,
                   color=_BLUE, alpha=0.75, edgecolor="white")
        gap = bin_acc[m] - bin_conf[m]
        color_gap = _RED if gap < 0 else _GREEN
        ax_rel.bar(bin_centers[m], gap, width=w,
                   bottom=bin_conf[m], color=color_gap, alpha=0.30, edgecolor="white")
    # Auto-zoom reliability diagram to where data actually is
    active = bin_centers[bin_cnt > 0]
    x_lo = max(0.0, float(active.min()) - 0.05) if len(active) > 0 else 0.0
    ax_rel.plot([x_lo, 1], [x_lo, 1], "k--", lw=1.5, label="Perfect calibration")
    ax_rel.set_xlabel("Confidence")
    ax_rel.set_ylabel("Accuracy")
    ece_flag = "well-calibrated" if ece < 0.02 else "temperature scaling recommended (Guo 2017)"
    bin_mode = "adaptive quantile bins" if adaptive_mode else "equal-width bins"
    n_active = int((bin_cnt > 0).sum())
    ax_rel.set_title(
        f"Reliability Diagram  (ECE={ece:.4f}  MCE={mce:.4f})\n"
        f"{ece_flag}  ·  {n_active}/{n_bins} bins ({bin_mode})",
        fontsize=9,
    )
    ax_rel.set_xlim(x_lo, 1.0)
    ax_rel.set_ylim(0, 1.0)
    ax_rel.legend(fontsize=8)

    gaps = np.array([
        bin_acc[m] - bin_conf[m] if bin_cnt[m] > 0 else 0.0
        for m in range(n_bins)
    ])
    colors_gap = [_RED if g < 0 else _GREEN for g in gaps]
    ax_gap.bar(bin_centers, gaps, width=bin_widths * 0.85,
               color=colors_gap, alpha=0.8, edgecolor="white")
    ax_gap.axhline(0, color="black", lw=0.8)
    ax_gap.set_xlabel("Confidence bin")
    ax_gap.set_ylabel("Accuracy minus Confidence")
    ax_gap.set_title("Calibration Gap per Bin\n"
                     "Green = underconfident  Red = overconfident", fontsize=9)
    ax_gap.set_xlim(0, 1)

    shift_df = calibration_under_shift(df, n_quartiles=4)
    ax_txt.axis("off")
    if not shift_df.empty:
        col_labels = ["Quartile", "Entropy range", "ECE", "Accuracy", "Mean conf.", "N"]
        table_data = [
            [
                f"Q{row.quartile}",
                row.entropy_range,
                f"{row.ece:.4f}" if row.ece == row.ece else "n/a",
                f"{row.accuracy:.3f}" if row.accuracy == row.accuracy else "n/a",
                f"{row.mean_confidence:.3f}",
                str(row.n_samples),
            ]
            for row in shift_df.itertuples()
        ]
        tbl = ax_txt.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax_txt.set_title(
            "Calibration by Entropy Quartile (Ovadia et al. 2019)\n"
            "ECE should stay stable; degradation signals dataset shift",
            fontsize=8, pad=6,
        )
    else:
        ax_txt.text(0.5, 0.5,
                    "Calibration-under-shift requires entropy_a + labels columns.",
                    ha="center", va="center", fontsize=10, color=_GREY,
                    transform=ax_txt.transAxes)

    _footer(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 10: Uncertainty Decomposition  (Kendall & Gal 2017)
# ---------------------------------------------------------------------------

def _page_uncertainty_decomposition(
    pdf: "PdfPages",
    df: "pd.DataFrame",
) -> None:
    """Stacked bar of aleatoric vs epistemic uncertainty per predicted class."""
    fig = plt.figure(figsize=(11, 8))
    _header(fig, "Page 10 \u2014 Uncertainty Decomposition",
            "Aleatoric vs Epistemic uncertainty per class  (Kendall & Gal 2017 / BALD)")

    required = {"pred_a", "bald_epistemic", "bald_aleatoric"}
    if not required.issubset(df.columns):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Requires BALD scores.\nRun with method='bald' and "
                "compute_uncertainty_bald_full().\n"
                "Expected columns: pred_a, bald_epistemic, bald_aleatoric",
                ha="center", va="center", fontsize=12, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")
        _footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    ae_df = aleatoric_epistemic_report(df)
    if ae_df.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data for uncertainty decomposition.",
                ha="center", va="center", fontsize=12, color=_GREY,
                transform=ax.transAxes)
        ax.axis("off")
        _footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    gs = gridspec.GridSpec(1, 2, figure=fig, top=0.88, bottom=0.14,
                           wspace=0.45, left=0.08, right=0.95)
    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_frac = fig.add_subplot(gs[0, 1])

    class_ids = ae_df["class_id"].values
    x = np.arange(len(class_ids))
    al = ae_df["mean_aleatoric"].values
    ep = ae_df["mean_epistemic"].values
    bar_w = 0.6

    ax_bar.bar(x, al, width=bar_w, color=_ORANGE,
               label="Aleatoric (irreducible noise)", edgecolor="white")
    ax_bar.bar(x, ep, width=bar_w, bottom=al, color=_BLUE,
               label="Epistemic (model uncertainty)", edgecolor="white")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"C{c}" for c in class_ids],
                           rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("Mean BALD score (nats)")
    ax_bar.set_title("Aleatoric vs Epistemic per Class\n"
                     "Epistemic high => add more labels for that class", fontsize=9)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)

    sorted_df = ae_df.sort_values("epistemic_fraction", ascending=True)
    fracs = sorted_df["epistemic_fraction"].values
    cids  = sorted_df["class_id"].values
    x2 = np.arange(len(cids))
    colors_frac = [_RED if f > 0.6 else _ORANGE if f > 0.4 else _GREEN for f in fracs]
    ax_frac.barh(x2, fracs, color=colors_frac, edgecolor="white", height=0.6)
    ax_frac.set_yticks(x2)
    ax_frac.set_yticklabels([f"C{c}" for c in cids], fontsize=9)
    ax_frac.set_xlabel("Epistemic fraction (epistemic / total)")
    ax_frac.set_title("Epistemic Fraction per Class\n"
                      "Red > 0.6 => prioritize for active labeling", fontsize=9)
    ax_frac.axvline(0.5, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax_frac.set_xlim(0, 1)
    ax_frac.grid(axis="x", alpha=0.3)

    patch_al = mpatches.Patch(color=_ORANGE, label="Aleatoric - ambiguous defect")
    patch_ep = mpatches.Patch(color=_BLUE,   label="Epistemic - model needs data")
    fig.legend(handles=[patch_al, patch_ep], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.02))

    _footer(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

"""Generate the headline figures (3 per pipeline) from the regenerated artifacts.

Run from the repo root:  python make_headline_figures.py

Reads the CSV/JSON artifacts each pipeline writes and produces clean figures
(English labels) into each pipeline's outputs/plots/. Every verdict shown on a
figure is computed from the artifact it plots, never written by hand.
Re-runnable: it only reads artifacts, never re-trains.

Headline set
------------
Active Learning + A/B  : al_budget_curve · ab_promotion_gate · ab_per_class_delta
Defect clustering      : h2_ablation · group_confusion_heatmap · regrouping_families
Datamining + cutting   : h3_grades · yield_distribution · shap_bar (pre-existing)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defect_clustering.analysis.group_analysis import h2_verdict

ROOT = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, BLUE, GREY = "#2e8b57", "#c0392b", "#2c6fbb", "#7f8c8d"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _verdict_word(flag) -> str:
    """'met', 'not met', or 'n/a' when the artifact carries no verdict."""
    if flag is None:
        return "n/a"
    return "met" if flag else "not met"


# ===========================================================================
# 1) Active Learning + A/B
# ===========================================================================
def al_ab_figures():
    base = os.path.join(ROOT, "steeldefect_active_learning", "outputs")
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    print("[AL + A/B]")

    # --- al_budget_curve : mean macro-F1 with a Student-t 95% band ----------
    curve_p = os.path.join(base, "al_budget_curve.csv")
    if os.path.exists(curve_p):
        from steeldefect_active_learning.active_learning.budget_curve import plot_budget_curve
        sav_p = os.path.join(base, "al_budget_savings.json")
        sav = json.load(open(sav_p)) if os.path.exists(sav_p) else None
        out = os.path.join(plots, "al_budget_curve.png")
        plot_budget_curve(pd.read_csv(curve_p), out, savings=sav,
                          data_label="calibrated-synthetic data")
        print(f"  wrote {out}")

    # --- ab_promotion_gate : the decision card -----------------------------
    dec_p = os.path.join(base, "ab_decision.json")
    met_p = os.path.join(base, "ab_metrics.json")
    if os.path.exists(dec_p) and os.path.exists(met_p):
        dec = json.load(open(dec_p))
        met = json.load(open(met_p))
        st = met.get("statistical", {})
        mc = met.get("mcnemar", {})
        reasons = dec.get("reasons", [])
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.axis("off")
        verdict = dec.get("decision", "?")
        vcol = GREEN if verdict == "promote_B" else RED
        ax.text(0.5, 0.96, f"A/B promotion gate (synthetic demo predictions): decision {verdict.upper()}",
                ha="center", va="top", fontsize=13, fontweight="bold", color=vcol,
                transform=ax.transAxes)
        y = 0.84
        for r in reasons:
            ok = r.get("passed")
            ax.text(0.04, y, "PASS" if ok else "FAIL", color=GREEN if ok else RED,
                    fontweight="bold", family="monospace", transform=ax.transAxes)
            ax.text(0.17, y, f"{r.get('criterion')}", fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.17, y - 0.035, f"{r.get('detail')}", fontsize=7.5, color=GREY,
                    transform=ax.transAxes)
            y -= 0.105
        foot = (f"test={st.get('test_used')}  p={st.get('p_value'):.2g}  "
                f"Cohen d={st.get('cohens_d'):.2f} ({st.get('effect_magnitude')})  "
                f"McNemar p={mc.get('p_value', float('nan')):.2g}")
        ax.text(0.5, 0.02, foot, ha="center", fontsize=8, color="black",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="#f3f3f3"))
        _save(fig, os.path.join(plots, "ab_promotion_gate.png"))

    # --- ab_per_class_delta : per-class accuracy B - A ---------------------
    demo_p = os.path.join(os.path.dirname(base), "data", "demo_ab_predictions.csv")
    if os.path.exists(demo_p):
        df = pd.read_csv(demo_p)
        rows = []
        for c, g in df.groupby("label_name"):
            acc_a = (g["pred_a"] == g["label"]).mean()
            acc_b = (g["pred_b"] == g["label"]).mean()
            rows.append((c, acc_b - acc_a))
        rows.sort(key=lambda r: r[1])
        names = [r[0] for r in rows]
        delta = [r[1] for r in rows]
        retrained = {"edge_crack", "sliver", "orange_peel"}
        cols = [BLUE if n in retrained else GREY for n in names]
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        ax.barh(names, delta, color=cols)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Per-class accuracy change (model B - model A)")
        ax.set_title("A/B: per-class accuracy change on synthetic predictions\n"
                     "(blue = the 3 classes where the simulated B is set to beat A)")
        ax.grid(axis="x", alpha=0.3)
        _save(fig, os.path.join(plots, "ab_per_class_delta.png"))


# ===========================================================================
# 2) Defect clustering
# ===========================================================================
def defect_figures(base: str | None = None, data_label: str = "calibrated-synthetic data"):
    """Defect-clustering figures from the artifacts in ``base``.

    ``base`` defaults to ``defect_clustering/outputs`` (the calibrated-synthetic
    run); pass ``defect_clustering/outputs_neu`` with ``data_label="NEU-DET"``
    for the public-data run.
    """
    base = base or os.path.join(ROOT, "defect_clustering", "outputs")
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    print("[Defect clustering]")

    # --- h2_ablation : NMI per feature set with bootstrap CI ---------------
    abl_p = os.path.join(base, "ablation_results.csv")
    raw = pd.read_csv(abl_p) if os.path.exists(abl_p) else None
    verdict = None
    if raw is not None:
        try:
            verdict = h2_verdict(raw)
        except (KeyError, ValueError) as exc:  # e.g. a CNN-only run has no fusion row
            print(f"  skip h2_ablation: no H2 verdict in {abl_p} ({exc})")
    if verdict is not None:
        a = raw.set_index("feature_set")
        order = [r for r in ["cnn_only", "seg_only", "concat_naive",
                             "fusion_block_pca", "weighted_08_02"] if r in a.index]
        a = a.loc[order]
        nmi = a["nmi"].values
        lo = a["nmi_ci_low"].values
        hi = a["nmi_ci_high"].values
        # Without a CI (n_bootstrap=0) the rule fails closed, but no test was run.
        flag = verdict["passed"] if np.isfinite(verdict["fusion_nmi_ci"][0]) else None
        vcol = GREY if flag is None else (GREEN if flag else RED)
        bar_color = {"cnn_only": GREY, "seg_only": GREY, "concat_naive": "#d4a017",
                     "fusion_block_pca": vcol, "weighted_08_02": BLUE}
        cols = [bar_color.get(o, GREY) for o in order]
        n_pca = int(a.loc["cnn_only", "n_features"]) if "n_features" in a.columns else None
        labels = {"cnn_only": f"CNN only\n(PCA-{n_pca})" if n_pca else "CNN only",
                  "seg_only": "Morphology only",
                  "concat_naive": "Naive concat.\n(no PCA)",
                  "fusion_block_pca": "Fusion z-score\n+ PCA (H2 test)",
                  "weighted_08_02": "Weighted 0.8/0.2"}
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        x = np.arange(len(order))
        ax.bar(x, nmi, color=cols)
        # Draw each CI as computed: a percentile CI need not contain the point estimate.
        ax.errorbar(x, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(o, o) for o in order], fontsize=8.5)
        ax.set_ylabel("NMI against the reference labels")
        base_nmi = verdict["cnn_nmi"]
        ax.set_ylim(0, 1.0)
        fi = order.index("fusion_block_pca")
        rule = (f"fusion CI low {verdict['fusion_nmi_ci'][0]:.4f} vs CNN-only {base_nmi:.4f}; "
                f"gain {verdict['delta_nmi']:+.4f} vs {verdict['min_gain']:.2f}")
        ax.annotate(f"{verdict['delta_nmi']:+.4f} NMI\nH2 {_verdict_word(flag)}",
                    xy=(fi, verdict["fusion_nmi"]), xytext=(fi - 1.85, 0.53),
                    fontsize=9.5, color=vcol, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    bbox=dict(boxstyle="round", fc="white", ec=vcol, alpha=0.9))
        ax.axhline(base_nmi, ls="--", color=GREY, lw=0.9)
        ax.set_title(f"H2: do morphology descriptors add NMI? ({data_label})\n"
                     "95% percentile bootstrap CI of each NMI, clustering fixed")
        ax.text(0.5, -0.27, "H2 rule: fusion CI lower bound > CNN-only NMI and gain >= "
                f"{verdict['min_gain']:.2f}. Here: {rule}.\n"
                "One clustering seed: variation across seeds is not part of the CI.",
                transform=ax.transAxes, ha="center", fontsize=7, color="black")
        ax.grid(axis="y", alpha=0.3)
        _save(fig, os.path.join(plots, "h2_ablation.png"))

    # --- group_confusion_heatmap : group x group, ordered by proposed family -
    conf_p = os.path.join(base, "group_confusion.csv")
    remap_p = os.path.join(base, "group_remap.json")
    if os.path.exists(conf_p) and os.path.exists(remap_p):
        M = pd.read_csv(conf_p, index_col=0)
        remap = json.load(open(remap_p)).get("group_map", {})
        # order groups by their proposed family so merges form blocks
        order = sorted(M.index, key=lambda g: (str(remap.get(g, g)), g))
        M = M.loc[order, order]
        fig, ax = plt.subplots(figsize=(8.2, 7.0))
        im = ax.imshow(M.values, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        ax.set_title(f"Co-clustering matrix of the {len(order)} expert groups ({data_label})\n"
                     "(ordered by proposed family)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="share of the row group's samples in clusters led by the column group")
        _save(fig, os.path.join(plots, "group_confusion_heatmap.png"))

    # --- regrouping_families : expert groups -> K families (composition) ----
    # Replaces the flat affinity bar (whose uniform ~0.5 was an artifact of
    # symmetric averaging over asymmetric absorptions). This shows the actual
    # deliverable: which expert groups consolidate into which family, sized by
    # sample count, so both the consolidation AND the class imbalance are
    # visible, with naturally varied bar lengths.
    meta_p = os.path.join(os.path.dirname(base), "data", "synthetic", "metadata.csv")
    if os.path.exists(remap_p) and os.path.exists(meta_p):
        gmap = json.load(open(remap_p)).get("group_map", {})
        kp = json.load(open(remap_p)).get("k_proposed", len(set(gmap.values())))
        meta = pd.read_csv(meta_p)
        grp_counts = meta["expert_group"].value_counts().to_dict()
        # family -> [(group, n_samples), ...]
        fam: dict = {}
        for g, f in gmap.items():
            fam.setdefault(f, []).append((g, grp_counts.get(g, 0)))
        fam_items = sorted(fam.items(), key=lambda kv: sum(n for _, n in kv[1]))
        fams = [f for f, _ in fam_items]
        sizes = [sum(n for _, n in members) for _, members in fam_items]
        ngroups = [len(members) for _, members in fam_items]
        cols = [GREEN if ng > 1 else GREY for ng in ngroups]
        fig, ax = plt.subplots(figsize=(9.2, 0.46 * len(fams) + 1.8))
        bars = ax.barh(fams, sizes, color=cols)
        for bar, members, ng in zip(bars, [m for _, m in fam_items], ngroups):
            names = ", ".join(g for g, _ in sorted(members, key=lambda x: -x[1]))
            tag = f"  {ng} group{'s' if ng > 1 else ''}: {names}"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    tag[:78], va="center", fontsize=7, color="black")
        ax.set_xlabel("number of samples (family size)")
        ax.set_title(f"Regrouping: {len(gmap)} expert groups -> {kp} families ({data_label})\n"
                     "(green = merged family, grey = group kept alone)")
        ax.margins(x=0.28)
        ax.grid(axis="x", alpha=0.3)
        _save(fig, os.path.join(plots, "regrouping_families.png"))
        old = os.path.join(plots, "regrouping_merges.png")
        if os.path.exists(old):
            os.remove(old)


# ===========================================================================
# 3) Datamining + cutting plan
# ===========================================================================
def datamining_figures():
    base = os.path.join(ROOT, "datamining", "outputs")
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    print("[Datamining]")

    res_p = os.path.join(base, "cutting_results.csv")
    met_p = os.path.join(base, "model_metrics.json")
    if not os.path.exists(res_p):
        return
    df = pd.read_csv(res_p)
    met = json.load(open(met_p)) if os.path.exists(met_p) else {}
    h3 = met.get("h3", {})

    # --- h3_grades : mean area fractions + thresholds + per-coil scrap ------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    fr_cols = [c for c in ["area_frac_A", "area_frac_B", "area_frac_C", "area_frac_scrap"]
               if c in df.columns]
    means = [df[c].mean() for c in fr_cols]
    names = [c.replace("area_frac_", "") for c in fr_cols]
    cols = {"A": GREEN, "B": "#7fb069", "C": "#d4a017", "scrap": RED}
    bottom = 0.0
    for n, m in zip(names, means):
        ax1.bar(0, m, bottom=bottom, color=cols.get(n, GREY), label=f"grade {n}")
        if m > 0.02:
            ax1.text(0, bottom + m / 2, f"{n}\n{m:.0%}", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        bottom += m
    ax1.axhline(0.25, ls="--", color="black", lw=1)
    ax1.text(0.35, 0.25, "grade A threshold >= 25%", fontsize=8, va="bottom")
    ax1.set_xlim(-0.6, 1.2); ax1.set_xticks([])
    ax1.set_ylabel("surface fraction (area-weighted)")
    h3a = h3.get("H3a_pass_grade_A_ge_0.25"); h3b = h3.get("H3b_pass_scrap_le_0.05")
    ax1.set_title(f"H3: mean grade split (calibrated-synthetic coils)\n"
                  f"H3a (A >= 25%): {_verdict_word(h3a)} · "
                  f"H3b (scrap <= 5%): {_verdict_word(h3b)}")
    ax1.legend(fontsize=8, loc="upper right")

    sc = df["area_frac_scrap"] * 100.0
    ax2.scatter(range(len(sc)), sorted(sc), s=18, color=RED, alpha=0.7)
    ax2.axhline(5.0, ls="--", color="black", lw=1, label="scrap ceiling 5%")
    ax2.set_xlabel("coil (sorted)"); ax2.set_ylabel("scrap (% of surface)")
    ax2.set_title(f"Scrap per coil: {h3.get('n_coils_scrap_le_5pct','?')}/"
                  f"{h3.get('n_coils','?')} coils <= 5%")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    _save(fig, os.path.join(plots, "h3_grades.png"))

    # --- yield_distribution : spread across coils ---------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(df["yield_pct"], bins=12, color=BLUE, edgecolor="white", alpha=0.85)
    mu = df["yield_pct"].mean()
    ax.axvline(mu, color=RED, lw=1.5, label=f"mean {mu:.1f}%")
    ax.set_xlabel("material yield per coil (%)")
    ax.set_ylabel("number of coils")
    r2 = met.get("severity_model_r2")
    ax.set_title(f"Material yield over {len(df)} calibrated-synthetic coils"
                 + (f"  (severity model R² = {r2:.3f})" if r2 else ""))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(plots, "yield_distribution.png"))


if __name__ == "__main__":
    al_ab_figures()
    defect_figures()
    neu = os.path.join(ROOT, "defect_clustering", "outputs_neu")
    if os.path.isdir(neu):  # the public-data run, if it was reproduced locally
        defect_figures(base=neu, data_label="NEU-DET")
    datamining_figures()
    print("\nDone. Headline figures written to each <pipeline>/outputs/plots/.")

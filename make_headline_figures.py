"""Generate the headline figures (3 per pipeline) from the regenerated artifacts.

Run from the repo root:  python make_headline_figures.py

Reads the CSV/JSON artifacts each pipeline writes and produces clean,
report-ready figures (French labels) into each pipeline's outputs/plots/.
Re-runnable: it only reads artifacts, never re-trains.

Headline set
------------
Active Learning + A/B  : al_budget_curve (pre-existing) · ab_promotion_gate · ab_per_class_delta
Defect clustering      : h2_ablation · group_confusion_heatmap · regrouping_merges
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

ROOT = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, BLUE, GREY = "#2e8b57", "#c0392b", "#2c6fbb", "#7f8c8d"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# 1) Active Learning + A/B
# ===========================================================================
def al_ab_figures():
    base = os.path.join(ROOT, "steeldefect_active_learning", "outputs")
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    print("[AL + A/B]")

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
        ax.text(0.5, 0.96, f"Porte de promotion A/B — décision : {verdict.upper()}",
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
    demo_p = os.path.join(base.replace("outputs", "data"), "demo_ab_predictions.csv")
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
        ax.set_xlabel("Δ exactitude par classe  (modèle B − modèle A)")
        ax.set_title("A/B — gain/perte d'exactitude par classe\n"
                     "(bleu = 3 classes ré-entraînées de B)")
        ax.grid(axis="x", alpha=0.3)
        _save(fig, os.path.join(plots, "ab_per_class_delta.png"))


# ===========================================================================
# 2) Defect clustering
# ===========================================================================
def defect_figures():
    base = os.path.join(ROOT, "defect_clustering", "outputs")
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    print("[Defect clustering]")

    # --- h2_ablation : NMI per feature set with bootstrap CI ---------------
    abl_p = os.path.join(base, "ablation_results.csv")
    if os.path.exists(abl_p):
        a = pd.read_csv(abl_p).set_index("feature_set")
        order = [r for r in ["cnn_only", "seg_only", "concat_naive",
                             "fusion_block_pca", "weighted_08_02"] if r in a.index]
        a = a.loc[order]
        nmi = a["nmi"].values
        lo = a.get("nmi_ci_low", a["nmi"]).values
        hi = a.get("nmi_ci_high", a["nmi"]).values
        err = np.vstack([np.maximum(nmi - lo, 0), np.maximum(hi - nmi, 0)])
        labels = {"cnn_only": "CNN seul\n(960→ACP-50)", "seg_only": "DefChars seul\n(48-D)",
                  "concat_naive": "Concat. naïve\n(sans ACP)",
                  "fusion_block_pca": "Fusion z-score\n+ ACP (prod.)",
                  "weighted_08_02": "Pondérée 0.8/0.2"}
        cols = [GREY, GREY, "#d4a017", GREEN, BLUE][:len(order)]
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        x = np.arange(len(order))
        ax.bar(x, nmi, yerr=err, color=cols, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(o, o) for o in order], fontsize=8.5)
        ax.set_ylabel("NMI vs 28 groupes experts")
        base_nmi = a.loc["cnn_only", "nmi"] if "cnn_only" in a.index else nmi[0]
        ax.set_ylim(0, 0.95)
        if "fusion_block_pca" in a.index:
            fu = a.loc["fusion_block_pca", "nmi"]
            fi = order.index("fusion_block_pca")
            ax.annotate(f"+{fu - base_nmi:.3f} NMI\n(H2 validée, IC exclut 0)",
                        xy=(fi, fu), xytext=(fi - 1.7, 0.52),
                        fontsize=9.5, color=GREEN, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                        bbox=dict(boxstyle="round", fc="white", ec=GREEN, alpha=0.9))
        ax.axhline(base_nmi, ls="--", color=GREY, lw=0.9)
        ax.set_title("H2 — apport des descripteurs morphologiques\n(ablation, IC bootstrap B=100)")
        ax.grid(axis="y", alpha=0.3)
        _save(fig, os.path.join(plots, "h2_ablation.png"))

    # --- group_confusion_heatmap : 28×28 reordered by proposed family ------
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
        ax.set_title("Matrice de co-clustering des 28 groupes experts\n"
                     "(réordonnée par famille métallurgique proposée)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="affinité")
        _save(fig, os.path.join(plots, "group_confusion_heatmap.png"))

    # --- regrouping_families : the 28 -> K consolidation (composition) -----
    # Replaces the flat affinity bar (whose uniform ~0.5 was an artifact of
    # symmetric averaging over asymmetric absorptions). This shows the actual
    # deliverable: which expert groups consolidate into which metallurgical
    # family, sized by sample count — so both the consolidation AND the class
    # imbalance are visible, with naturally varied bar lengths.
    meta_p = os.path.join(base.replace("outputs", "data"), "synthetic", "metadata.csv")
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
            tag = f"  {ng} groupe{'s' if ng > 1 else ''} : {names}"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    tag[:78], va="center", fontsize=7, color="black")
        ax.set_xlabel("nombre d'échantillons (taille de la famille)")
        ax.set_title(f"Re-groupement métallurgique : 28 groupes experts → {kp} familles\n"
                     f"(vert = famille fusionnée, gris = groupe conservé seul)")
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
    ax1.text(0.35, 0.25, "seuil grade A ≥ 25 %", fontsize=8, va="bottom")
    ax1.set_xlim(-0.6, 1.2); ax1.set_xticks([])
    ax1.set_ylabel("fraction de surface (pondérée par l'aire)")
    h3a = h3.get("H3a_pass_grade_A_ge_0.25"); h3b = h3.get("H3b_pass_scrap_le_0.05")
    ax1.set_title(f"H3 — répartition moyenne des grades\n"
                  f"H3a (A≥25%): {'OK' if h3a else 'NON'} · "
                  f"H3b (rebut≤5%): {'OK' if h3b else 'NON'}")
    ax1.legend(fontsize=8, loc="upper right")

    sc = df["area_frac_scrap"] * 100.0
    ax2.scatter(range(len(sc)), sorted(sc), s=18, color=RED, alpha=0.7)
    ax2.axhline(5.0, ls="--", color="black", lw=1, label="plafond rebut 5 %")
    ax2.set_xlabel("bobine (triée)"); ax2.set_ylabel("rebut (% surface)")
    ax2.set_title(f"Rebut par bobine — {h3.get('n_coils_scrap_le_5pct','?')}/"
                  f"{h3.get('n_coils','?')} bobines ≤ 5 %")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    _save(fig, os.path.join(plots, "h3_grades.png"))

    # --- yield_distribution : realistic spread ------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(df["yield_pct"], bins=12, color=BLUE, edgecolor="white", alpha=0.85)
    mu = df["yield_pct"].mean()
    ax.axvline(mu, color=RED, lw=1.5, label=f"moyenne {mu:.1f}%")
    ax.set_xlabel("rendement matière par bobine (%)")
    ax.set_ylabel("nombre de bobines")
    r2 = met.get("severity_model_r2")
    ax.set_title(f"Distribution du rendement matière sur {len(df)} bobines calibrées"
                 + (f"  (modèle sévérité R²={r2:.3f})" if r2 else ""))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(plots, "yield_distribution.png"))


if __name__ == "__main__":
    al_ab_figures()
    defect_figures()
    datamining_figures()
    print("\nDone. Headline figures written to each <pipeline>/outputs/plots/.")

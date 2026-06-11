"""Render the 9 figures from saved artifacts, in a FRESH process.

The main pipeline loads torch (whose bundled OpenMP runtime, under the
Windows ``KMP_DUPLICATE_LIB_OK`` workaround, leaves the heap unstable for
late-stage matplotlib/numpy allocations). Rendering the figures from the
saved ``steeldefect_ssl_metrics.json`` + ``_plot_data.npz`` in a separate
process — which never imports torch — sidesteps that entirely.

    python -m steeldefect_ssl.eval.render_plots <out_dir> <plots_dir>
"""

from __future__ import annotations

import os

# Single-thread the BLAS/OpenMP pools in this (torch-free) renderer so numpy /
# sklearn / matplotlib never enter the parallel regions that destabilise the
# heap on this Windows stack. Must precede numpy import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json   # noqa: E402
import sys    # noqa: E402

import numpy as np   # noqa: E402

from steeldefect_ssl.eval import plots as P   # noqa: E402


def render(out_dir: str, plots_dir: str) -> int:
    os.makedirs(plots_dir, exist_ok=True)
    with open(os.path.join(out_dir, "steeldefect_ssl_metrics.json"), "r", encoding="utf-8") as fh:
        M = json.load(fh)
    d = np.load(os.path.join(out_dir, "_plot_data.npz"), allow_pickle=True)

    def pjoin(name):
        return os.path.join(plots_dir, name)

    # open-set ROC/PR curves rebuilt from saved scores + metrics AUROCs
    osr = {r["scorer"]: r for r in M.get("openset_results", [])}
    cfg_scorer = M["config"]["openset"]["scorer"]
    roc = {}
    if "primary_scores" in d:
        roc[f"ssl·{cfg_scorer}"] = {
            "scores": d["primary_scores"], "is_novel": d["is_novel"],
            "auroc": osr.get(cfg_scorer, {}).get("auroc", float("nan"))}
    if "pc_scores" in d and d["pc_scores"].size:
        roc["patchcore"] = {"scores": d["pc_scores"], "is_novel": d["is_novel"],
                            "auroc": osr.get("patchcore_baseline", {}).get("auroc", float("nan"))}
    if "r_scores" in d and d["r_scores"].size:
        roc["random_init"] = {"scores": d["r_scores"], "is_novel": d["is_novel"],
                              "auroc": osr.get("random_init_features", {}).get("auroc", float("nan"))}

    specs = [
        ("ssl_loss_curve.png", lambda p: P.plot_ssl_loss_curve(M.get("histories", []), p)),
        ("embedding_umap.png", lambda p: P.plot_embedding_umap(d["X_eval"], d["y_eval"], M.get("class_names"), p)),
        ("probe_bar.png", lambda p: P.plot_probe_bar(M.get("probe_results", {}), p)),
        ("clustering_uplift_bar.png", lambda p: P.plot_clustering_uplift(M.get("clustering_uplift", []), p)),
        ("openset_score_hist.png", lambda p: P.plot_openset_hist(d["primary_scores"], d["is_novel"], p)),
        ("openset_roc_pr.png", lambda p: P.plot_openset_roc_pr(roc, p)),
        ("gcd_umap.png", lambda p: P.plot_gcd_umap(d["X_gcd"], d["gcd_labels"], d["novel_global"], p)),
        ("gcd_outliers.png", lambda p: P.plot_gcd_outliers(M.get("gcd_outlier_info", {}), p)),
        ("continual_forgetting.png", lambda p: P.plot_continual_forgetting(M.get("continual_results", {}), p)),
    ]
    n_ok = 0
    for fname, fn in specs:
        try:
            fn(pjoin(fname)); n_ok += 1
        except Exception as exc:
            print(f"  [warn] plot {fname} failed: {exc}")
    print(f"  [ok] {n_ok}/{len(specs)} figures -> {plots_dir}")
    return 0 if n_ok == len(specs) else 2


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print("usage: python -m steeldefect_ssl.eval.render_plots <out_dir> <plots_dir>")
        return 1
    return render(argv[0], argv[1])


if __name__ == "__main__":
    raise SystemExit(main())

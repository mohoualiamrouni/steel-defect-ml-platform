"""Single CPU entrypoint for the steeldefect_ssl pipeline (P4).

Orchestrates the four stages — SSL pretraining, embedding eval/refresh (H4-A),
open-set novelty detection (H4-B), and generalized category discovery +
continual learning — then writes the metrics JSON, per-stage CSVs, the 9-figure
set, and prints the H4 acceptance verdict.

    python -m steeldefect_ssl.pipeline.run_pipeline --config steeldefect_ssl/config/pipeline_config.yaml [--full]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List

import numpy as np

# repo root on path so sibling packages import regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steeldefect_ssl import SOURCE_NEU, SOURCE_SYNTHETIC, __version__
from steeldefect_ssl.config import SSLPipelineConfig, apply_fast_mode, load_config, to_dict


# ─────────────────────────────────────────────────────────────── reused (defensive)
try:
    from defect_clustering.analysis.group_analysis import (
        _bootstrap_nmi_ci, compute_group_metrics,
    )
except Exception:  # pragma: no cover
    from sklearn.metrics import (adjusted_rand_score as _ari,
                                 normalized_mutual_info_score as _nmi)

    def compute_group_metrics(lp, le):
        return {"nmi": float(_nmi(le, lp)), "ari": float(_ari(le, lp))}

    def _bootstrap_nmi_ci(le, lp, n_bootstrap=100, random_state=42):
        from sklearn.metrics import normalized_mutual_info_score as nmi
        rng = np.random.default_rng(random_state)
        le, lp = np.asarray(le), np.asarray(lp)
        vals = [nmi(le[idx], lp[idx]) for idx in
                (rng.integers(0, len(le), len(le)) for _ in range(n_bootstrap))]
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _detect_outliers(X, labels, contamination, random_state):
    try:
        from defect_clustering.clustering.outlier_detection import detect_outliers_per_cluster
        mask, info = detect_outliers_per_cluster(
            X, labels, contamination=contamination, strategy="union",
            min_cluster_size=10, random_state=random_state)
        return int(mask.sum()), info
    except Exception as exc:  # pragma: no cover
        print(f"[gcd] outlier detection skipped: {exc}")
        return 0, {}


# ─────────────────────────────────────────────────────────────── small helpers
def _banner(msg): print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")
def _ok(msg): print(f"  [ok] {msg}")
def _warn(msg): print(f"  [warn] {msg}")


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _embed(encoder, df, image_size, device):
    """Return (X, y) for a pool df (order preserved -> aligns with df rows)."""
    from steeldefect_ssl.data.ssl_dataset import LabeledEvalDataset
    from steeldefect_ssl.models.encoder import embed_dataset
    ds = LabeledEvalDataset(df, transform=None, image_size=image_size)
    X, y, _ = embed_dataset(encoder, ds, device=device)
    return X, y


# ─────────────────────────────────────────────────────────────── the pipeline
def run_pipeline(cfg: SSLPipelineConfig) -> Dict:
    t_start = time.time()
    if cfg.runtime.fast_mode:
        apply_fast_mode(cfg)
    _set_seed(cfg.runtime.seed)
    device = cfg.runtime.device

    out, plots = cfg.output.dir, cfg.output.plots_dir
    if cfg.output.clear_previous and os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    for d in (out, plots, cfg.output.reports_dir):
        os.makedirs(d, exist_ok=True)

    import pandas as pd
    from steeldefect_ssl.data.pool_builder import (
        build_pools, known_classes, load_real_pools, real_known_classes,
    )
    from steeldefect_ssl.models.encoder import build_encoder
    from steeldefect_ssl.ssl.trainer import EncoderBundle, train_encoder

    _banner(f"steeldefect_ssl (P4) v{__version__}  ·  mode={'FAST/smoke' if cfg.runtime.fast_mode else 'FULL'}  ·  method={cfg.ssl.method}")
    _real = cfg.data.source == "real"
    metrics: Dict = {"version": __version__,
                     "source_version": SOURCE_NEU if _real else SOURCE_SYNTHETIC,
                     "fast_mode": cfg.runtime.fast_mode, "config": to_dict(cfg)}

    # ---- pools ----------------------------------------------------------------
    if _real:
        _banner("[1/6] Load REAL public dataset pools (NEU-DET; novel held out)")
        pools = load_real_pools(cfg.data)
        known = real_known_classes(cfg.data)
    else:
        _banner("[1/6] Build calibrated-synthetic pools (novel held out)")
        pools = build_pools(cfg.data)
        known = known_classes(cfg.data.novel_classes)
    K = len(known)
    novel_list = ([cfg.data.novel_classes] if isinstance(cfg.data.novel_classes, str)
                  else list(cfg.data.novel_classes))
    novel_id = {c: K + i for i, c in enumerate(novel_list)}     # distinct id per novel family
    class_names = list(known) + [f"{c} (novel)" for c in novel_list]
    for name, df in pools.items():
        _ok(f"{name}: {len(df)} crops")
    metrics["pools"] = {k: int(len(v)) for k, v in pools.items()}
    metrics["known_classes"] = known
    metrics["novel_classes"] = novel_list

    # ---- SSL pretrain ---------------------------------------------------------
    _banner("[2/6] Self-supervised pretraining")
    unlabeled_paths = pools["unlabeled"]["path"].tolist()
    bundle = train_encoder(cfg, unlabeled_paths, device=device, log=print)
    bundle.save(cfg.ssl.checkpoint)
    encoder = bundle.build(device)
    _ok(f"primary encoder [{cfg.ssl.method}] saved -> {cfg.ssl.checkpoint}")
    histories = [bundle.history]
    pd.DataFrame({"epoch": range(1, len(bundle.history["loss_curve"]) + 1),
                  "loss": bundle.history["loss_curve"],
                  "collapse_std": bundle.history["collapse_curve"]}
                 ).to_csv(os.path.join(out, "pretrain_curve.csv"), index=False)

    compare_bundle = None
    if cfg.ssl.compare:
        other = "simclr" if cfg.ssl.method == "simsiam" else "simsiam"
        cmp_epochs = max(5, cfg.ssl.epochs // 2)   # shorter comparison run (budget)
        _ok(f"comparison run: {other} ({cmp_epochs} epochs)")
        compare_bundle = train_encoder(cfg, unlabeled_paths, method=other,
                                       epochs=cmp_epochs, device=device, log=print)
        histories.append(compare_bundle.history)

    # ---- embeddings -----------------------------------------------------------
    X_lab, y_lab = _embed(encoder, pools["labeled"], cfg.data.image_size, device)
    X_eval, y_eval = _embed(encoder, pools["eval"], cfg.data.image_size, device)
    X_stream, _ = _embed(encoder, pools["stream"], cfg.data.image_size, device)
    is_novel = pools["stream"]["is_novel"].to_numpy().astype(bool)
    stream_label = pools["stream"]["label"].to_numpy().astype(int)
    stream_cls = pools["stream"]["class_name"].to_numpy()       # for per-novel-class open-set

    # ---- Stage 2: probes + clustering uplift (H4-A) ---------------------------
    _banner("[3/6] Embedding eval — probes + clustering uplift (H4-A)")
    from steeldefect_ssl.eval.probes import flatten_pixels, pca_reduce, probe_representation

    probe_results: Dict[str, Dict[str, float]] = {}
    probe_results["ssl"] = probe_representation(X_lab, y_lab, X_eval, y_eval,
                                                cfg.eval.knn_k, cfg.runtime.seed)
    # random-init baseline
    rand_enc = build_encoder(cfg.encoder.arch, cfg.encoder.in_channels,
                             cfg.encoder.embed_dim, cfg.encoder.width).to(device).eval()
    Xr_lab, _ = _embed(rand_enc, pools["labeled"], cfg.data.image_size, device)
    Xr_eval, _ = _embed(rand_enc, pools["eval"], cfg.data.image_size, device)
    Xr_stream, _ = _embed(rand_enc, pools["stream"], cfg.data.image_size, device)
    probe_results["random_init"] = probe_representation(Xr_lab, y_lab, Xr_eval, y_eval,
                                                         cfg.eval.knn_k, cfg.runtime.seed)
    # raw-pixel PCA baseline
    px_lab = flatten_pixels(pools["labeled"]["path"], cfg.data.image_size)
    px_eval = flatten_pixels(pools["eval"]["path"], cfg.data.image_size)
    _, pca_tf = pca_reduce(px_lab, cfg.eval.raw_pca_dim, cfg.runtime.seed)
    Xp_lab, Xp_eval = pca_tf(px_lab), pca_tf(px_eval)
    probe_results["raw_pixel_pca"] = probe_representation(Xp_lab, y_lab, Xp_eval, y_eval,
                                                          cfg.eval.knn_k, cfg.runtime.seed)
    if compare_bundle is not None:
        cmp_enc = compare_bundle.build(device)
        Xc_lab, _ = _embed(cmp_enc, pools["labeled"], cfg.data.image_size, device)
        Xc_eval, _ = _embed(cmp_enc, pools["eval"], cfg.data.image_size, device)
        probe_results[f"ssl_{compare_bundle.meta['method']}"] = probe_representation(
            Xc_lab, y_lab, Xc_eval, y_eval, cfg.eval.knn_k, cfg.runtime.seed)
    for r, v in probe_results.items():
        _ok(f"probe {r:18s} linear={v['linear_probe_acc']:.3f}  knn={v['knn_acc']:.3f}")
    pd.DataFrame([{"repr": r, **v} for r, v in probe_results.items()]).to_csv(
        os.path.join(out, "probe_results.csv"), index=False)
    metrics["probe_results"] = probe_results

    # clustering uplift (KMeans on eval features -> NMI/ARI vs true classes)
    from sklearn.cluster import KMeans
    uplift_rows: List[Dict] = []
    for rep, Xe in (("ssl", X_eval), ("raw_pixel_pca", Xp_eval), ("random_init", Xr_eval)):
        lbl = KMeans(n_clusters=K, n_init=10, random_state=cfg.runtime.seed).fit_predict(Xe)
        gm = compute_group_metrics(lbl, y_eval)
        lo, hi = _bootstrap_nmi_ci(y_eval, lbl, n_bootstrap=cfg.eval.bootstrap_n,
                                   random_state=cfg.runtime.seed)
        uplift_rows.append({"repr": rep, "nmi": gm["nmi"], "nmi_ci_low": lo,
                            "nmi_ci_high": hi, "ari": gm["ari"]})
        _ok(f"cluster {rep:18s} NMI={gm['nmi']:.3f} [{lo:.3f},{hi:.3f}]  ARI={gm['ari']:.3f}")
    pd.DataFrame(uplift_rows).to_csv(os.path.join(out, "clustering_uplift.csv"), index=False)
    metrics["clustering_uplift"] = uplift_rows

    # ---- Stage 3: open-set (H4-B) --------------------------------------------
    _banner("[4/6] Open-set novelty detection (H4-B)")
    from steeldefect_ssl.openset.evaluate import auroc_aupr_fpr95, baseline_patchcore_score
    from steeldefect_ssl.openset.scorers import build_scorer

    L2 = cfg.openset.l2_normalize
    openset_rows: List[Dict] = []
    roc_curves: Dict[str, Dict] = {}
    primary_scores = None
    for sname in ("mahalanobis", "knn", "cos_proto", "energy"):
        scorer = build_scorer(sname, cfg.openset.shrinkage, cfg.openset.knn_k, cfg.runtime.seed, L2)
        scorer.fit(X_lab, y_lab)
        s = scorer.score(X_stream)
        m = auroc_aupr_fpr95(s, is_novel)
        openset_rows.append({"scorer": sname, **m})
        _ok(f"open-set {sname:12s} AUROC={m['auroc']:.3f}  AUPR={m['aupr']:.3f}  FPR95={m['fpr95']:.3f}")
        if sname == cfg.openset.scorer:
            primary_scores = s
            roc_curves[f"ssl·{sname}"] = {"scores": s.tolist(), "is_novel": is_novel.tolist(),
                                          "auroc": m["auroc"]}
            # per-novel-class AUROC (known stream vs each held-out family individually)
            per_class = {}
            for c in novel_list:
                sel = (~is_novel) | (stream_cls == c)
                mc = auroc_aupr_fpr95(s[sel], (stream_cls[sel] == c))
                per_class[c] = mc["auroc"]
            metrics["openset_per_novel_auroc"] = per_class
            vals = [v for v in per_class.values() if v == v]  # drop nan
            metrics["openset_mean_novel_auroc"] = float(np.mean(vals)) if vals else float("nan")
            _ok("open-set per-novel AUROC: " + ", ".join(f"{c}={a:.3f}" for c, a in per_class.items()))
    # baseline: same scorer on RANDOM-INIT features (shows SSL features detect novelty better)
    rscorer = build_scorer(cfg.openset.scorer, cfg.openset.shrinkage, cfg.openset.knn_k, cfg.runtime.seed, L2)
    rscorer.fit(Xr_lab, y_lab)
    rand_scores = rscorer.score(Xr_stream)
    mr = auroc_aupr_fpr95(rand_scores, is_novel)
    openset_rows.append({"scorer": "random_init_features", **mr})
    roc_curves["random_init"] = {"scores": rand_scores.tolist(), "is_novel": is_novel.tolist(),
                                 "auroc": mr["auroc"]}
    _ok(f"open-set {'random_init':12s} AUROC={mr['auroc']:.3f}  FPR95={mr['fpr95']:.3f}")
    if cfg.openset.run_patchcore_baseline:
        clean_paths = pools["labeled"].loc[pools["labeled"]["class_name"] == "clean", "path"].tolist()
        if not clean_paths:
            clean_paths = pools["labeled"]["path"].tolist()[:100]
        bscore = baseline_patchcore_score(clean_paths, pools["stream"]["path"].tolist(), device)
        if bscore is not None:
            mb = auroc_aupr_fpr95(bscore, is_novel)
            openset_rows.append({"scorer": "patchcore_baseline", **mb})
            roc_curves["patchcore"] = {"scores": bscore.tolist(), "is_novel": is_novel.tolist(),
                                       "auroc": mb["auroc"]}
            _ok(f"open-set {'patchcore':12s} AUROC={mb['auroc']:.3f}  FPR95={mb['fpr95']:.3f}")
    pd.DataFrame(openset_rows).to_csv(os.path.join(out, "openset_results.csv"), index=False)
    metrics["openset_results"] = openset_rows

    # ---- Stage 4: GCD + continual --------------------------------------------
    _banner("[5/6] Generalized category discovery + continual learning")
    from steeldefect_ssl.continual.ncm import NCMClassifier
    from steeldefect_ssl.discovery.cluster_accuracy import hungarian_cluster_accuracy
    from steeldefect_ssl.discovery.gcd import select_k, semi_supervised_kmeans

    X_gcd = np.concatenate([X_lab, X_stream], axis=0)
    mask_known = np.concatenate([np.ones(len(X_lab), bool), np.zeros(len(X_stream), bool)])
    y_seed = np.concatenate([y_lab, np.full(len(X_stream), -1, int)])
    stream_true = np.array([novel_id[stream_cls[i]] if is_novel[i] else int(stream_label[i])
                            for i in range(len(is_novel))], dtype=int)
    y_true_gcd = np.concatenate([y_lab, stream_true])

    k = select_k(X_gcd, cfg.discovery, cfg.runtime.seed)
    gcd_labels = semi_supervised_kmeans(X_gcd, y_seed, mask_known, k, cfg.runtime.seed)
    gm = compute_group_metrics(gcd_labels, y_true_gcd)
    cacc = hungarian_cluster_accuracy(y_true_gcd, gcd_labels)
    n_out, out_info = (_detect_outliers(X_gcd, gcd_labels, cfg.discovery.outlier_contamination,
                                        cfg.runtime.seed)
                       if cfg.discovery.outlier_detection else (0, {}))
    novel_global = np.concatenate([np.zeros(len(X_lab), bool), is_novel])
    # per-novel-family recovery: does a cluster capture each held-out family?
    per_novel = {}
    n_recovered = 0
    for c in novel_list:
        cmask = np.concatenate([np.zeros(len(X_lab), bool), (stream_cls == c) & is_novel])
        if not cmask.any():
            continue
        cl = gcd_labels[cmask]
        best = np.bincount(cl).argmax()
        in_best = int((cl == best).sum())
        purity = in_best / max(1, int((gcd_labels == best).sum()))
        recall = in_best / max(1, int(cmask.sum()))
        rec = bool(purity >= 0.5 and recall >= 0.5)
        per_novel[c] = {"purity": round(purity, 3), "recall": round(recall, 3), "recovered": rec}
        n_recovered += int(rec)
    frac_rec = n_recovered / max(1, len(novel_list))
    gcd_res = {"k_selected": int(k), "nmi": gm["nmi"], "ari": gm["ari"],
               "cluster_accuracy": float(cacc), "n_outliers": int(n_out),
               "per_novel": per_novel, "n_novel_recovered": n_recovered,
               "frac_novel_recovered": frac_rec,
               "novel_family_recovered": bool(n_recovered >= 1)}
    _ok(f"GCD k={k}  NMI={gm['nmi']:.3f}  ARI={gm['ari']:.3f}  cluster_acc={cacc:.3f}")
    _ok(f"novel recovered {n_recovered}/{len(novel_list)}: " +
        ", ".join(f"{c}({'Y' if v['recovered'] else 'n'} p{v['purity']:.2f}/r{v['recall']:.2f})"
                  for c, v in per_novel.items()))
    pd.DataFrame([{**{k_: gcd_res[k_] for k_ in gcd_res if k_ != "per_novel"}}]).to_csv(
        os.path.join(out, "gcd_results.csv"), index=False)
    metrics["gcd_results"] = gcd_res

    # continual: NCM prototypes; add EACH novel family, then measure forgetting + new-class acc
    ncm = NCMClassifier().fit(X_lab, y_lab)
    acc_before = ncm.accuracy(X_eval, y_eval)
    test_sets = []
    for c in novel_list:
        Xc = X_stream[(stream_cls == c) & is_novel]
        if len(Xc) >= 4:
            half = len(Xc) // 2
            ncm.add_class(Xc[:half], novel_id[c])
            test_sets.append((Xc[half:], novel_id[c]))
    f = ncm.forgetting(X_eval, y_eval, acc_before)
    new_accs = [ncm.accuracy(Xt, np.full(len(Xt), idc)) for Xt, idc in test_sets]
    cont = {**f, "acc_new": float(np.mean(new_accs)) if new_accs else float("nan"),
            "n_novel_added": len(test_sets)}
    _ok(f"continual: old {cont['acc_old_before']:.3f}->{cont['acc_old_after']:.3f} "
        f"(forgetting delta={cont['forgetting_delta']:+.3f})  new={cont['acc_new']:.3f} "
        f"({cont['n_novel_added']} families)")
    pd.DataFrame([cont]).to_csv(os.path.join(out, "continual_results.csv"), index=False)
    metrics["continual_results"] = cont

    # ---- plots ----------------------------------------------------------------
    _banner("[6/6] Metrics + figures")
    metrics["histories"] = histories
    metrics["class_names"] = class_names
    metrics["gcd_outlier_info"] = out_info
    verdict = _h4_verdict(metrics)
    metrics["h4_verdict"] = verdict
    metrics["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(out, "steeldefect_ssl_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    pc = np.asarray(roc_curves["patchcore"]["scores"]) if "patchcore" in roc_curves else np.array([])
    np.savez(os.path.join(out, "_plot_data.npz"),
             X_eval=X_eval, y_eval=y_eval, X_gcd=X_gcd, gcd_labels=gcd_labels,
             novel_global=novel_global, primary_scores=np.asarray(primary_scores),
             is_novel=is_novel, pc_scores=pc, r_scores=np.asarray(rand_scores))
    _render_figures(out, plots)

    _banner("H4 verdict")
    for k_, v_ in verdict.items():
        print(f"  {('PASS' if v_ is True else 'FAIL' if v_ is False else '-'):>4}  {k_}")
    print(f"\nDone in {metrics['runtime_seconds']}s  -  outputs -> {out}")
    return metrics


def _render_figures(out: str, plots: str) -> None:
    """Render the 9 figures in a fresh torch-free subprocess (heap-stable)."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-m", "steeldefect_ssl.eval.render_plots", out, plots],
                           cwd=_REPO_ROOT, timeout=300)
        if r.returncode == 0:
            return
        _warn(f"plot subprocess rc={r.returncode}; trying in-process")
    except Exception as exc:
        _warn(f"plot subprocess failed ({exc}); trying in-process")
    try:
        from steeldefect_ssl.eval.render_plots import render
        render(out, plots)
    except Exception as exc:
        _warn(f"in-process plotting failed: {exc}")


def _h4_verdict(metrics: Dict) -> Dict:
    up = {r["repr"]: r for r in metrics.get("clustering_uplift", [])}
    pr = metrics.get("probe_results", {})
    osr = {r["scorer"]: r for r in metrics.get("openset_results", [])}
    cfg_scorer = metrics["config"]["openset"]["scorer"]
    v: Dict = {}
    # H4-A
    if "ssl" in up and "raw_pixel_pca" in up:
        v["H4A_clustering_uplift"] = bool(
            up["ssl"]["nmi_ci_low"] > up["raw_pixel_pca"]["nmi"]
            and (up["ssl"]["nmi"] - up["raw_pixel_pca"]["nmi"]) >= 0.05)
    if "ssl" in pr and "random_init" in pr and "raw_pixel_pca" in pr:
        v["H4A_probe_beats_baselines"] = bool(
            pr["ssl"]["linear_probe_acc"] > pr["random_init"]["linear_probe_acc"]
            and pr["ssl"]["linear_probe_acc"] > pr["raw_pixel_pca"]["linear_probe_acc"])
    # H4-B
    if cfg_scorer in osr:
        v["H4B_openset_auroc>=0.85"] = bool(osr[cfg_scorer]["auroc"] >= 0.85)
        v["H4B_openset_fpr95<=0.30"] = bool(osr[cfg_scorer]["fpr95"] <= 0.30)
        if "random_init_features" in osr:
            v["H4B_beats_random_baseline"] = bool(osr[cfg_scorer]["auroc"] > osr["random_init_features"]["auroc"])
        if "patchcore_baseline" in osr:
            v["H4B_beats_patchcore"] = bool(osr[cfg_scorer]["auroc"] >= osr["patchcore_baseline"]["auroc"])
    # corollary
    g = metrics.get("gcd_results", {})
    if g:
        v["H4_novel_recovered"] = bool(g.get("novel_family_recovered"))
    c = metrics.get("continual_results", {})
    if c:
        v["H4_no_forgetting"] = bool(abs(c.get("forgetting_delta", 1.0)) <= 0.05)
    return v


# ─────────────────────────────────────────────────────────────── CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="steeldefect_ssl (P4) — CPU self-supervised pipeline")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "pipeline_config.yaml"))
    ap.add_argument("--full", action="store_true", help="best-results run (disable fast_mode)")
    ap.add_argument("--fast", action="store_true", help="force smoke run (fast_mode)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.full:
        cfg.runtime.fast_mode = False
    if args.fast:
        cfg.runtime.fast_mode = True
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

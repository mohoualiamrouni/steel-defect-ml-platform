"""
End-to-end SteelDefect Active Learning + A/B Testing pipeline.

Orchestration steps:
    1. Load configuration
    2. Discover and load images (SteelDefectDataset)
    3. Load model(s)
    4. Active Learning
        a. Compute uncertainty scores  (entropy / BALD / mc_entropy / margin)
        b. Extract CNN embeddings
        c. Cluster embeddings for diversity  (kmeans / hdbscan)
        d. Select stratified / coreset / sensitivity candidates
        e. Export candidates CSV (for Vision Toolbox)
    5. Pipeline Active Learning (optional, Roth & Small 2008)
        a. Multi-stage β-weight adaptation over N cycles
        b. Export beta history CSV + PDF report
    6. A/B Testing (optional)
        a. Run both models — classification or segmentation
        b. Compute full metric suite (Welch t-test, Cohen's d, power, entropy, margin)
        c. Save report CSV + standard PNGs + advanced PNGs + PDF report
        d. Dump all metrics to JSON

CLI usage:
    python -m steeldefect_active_learning.pipeline.main_pipeline \\
        --config config/pipeline_config.yaml

Python usage:
    from steeldefect_active_learning.pipeline.main_pipeline import run_pipeline
    from steeldefect_active_learning.config import load_config

    cfg = load_config("steeldefect_active_learning/config/pipeline_config.yaml")
    run_pipeline(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import pandas as pd

logger = logging.getLogger("steeldefect_active_learning")

from steeldefect_active_learning.ab_testing.ab_pipeline import run_ab_comparison
from steeldefect_active_learning.ab_testing.metrics import (
    compute_all_metrics,
    confidence_delta_summary,
    disagreement_rate,
    plot_advanced_comparison,
    plot_advanced_comparison_v2,
    plot_comparison,
    statistical_significance_test,
)
from steeldefect_active_learning.ab_testing.report import generate_ab_report
from steeldefect_active_learning.active_learning.candidate_selector import (
    _auto_budget_strategy,
    candidates_to_dataframe,
    select_candidates,
)
from steeldefect_active_learning.active_learning.clustering import cluster_embeddings
from steeldefect_active_learning.active_learning.feature_extractor import extract_features
from steeldefect_active_learning.active_learning.pipeline_al import PipelineActiveLearner
from steeldefect_active_learning.active_learning.uncertainty import compute_uncertainty
from steeldefect_active_learning.config import PipelineConfig, load_config
from steeldefect_active_learning.data.dataset import SteelDefectDataset
from steeldefect_active_learning.models.model_loader import load_model


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 60

def _banner(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)

def _row(key: str, value, width: int = 28) -> None:
    print(f"  {key:<{width}} {value}")

def _ok(msg: str)  -> None:
    print(f"  [OK]  {msg}")
    logger.info("OK: %s", msg)

def _info(msg: str) -> None:
    print(f"  [--]  {msg}")
    logger.info(msg)

def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}")
    logger.warning(msg)


# ---------------------------------------------------------------------------
# Reproducibility & observability
# ---------------------------------------------------------------------------

def _set_global_seed(seed: Optional[int]) -> None:
    """Propagate one seed to numpy / torch / random / cudnn for reproducibility.

    None disables seeding (non-deterministic). Sets PYTHONHASHSEED so subprocesses
    inherit the same hash order, and toggles cudnn.deterministic when a CUDA
    device is available. Sklearn / UMAP / KMeans seeds remain explicit at call
    sites — this only sets the *global* state.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Trade some throughput for determinism on the GPU paths.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the 'steeldefect_active_learning' logger.

    Adds a stderr console handler at the chosen level. If log_file is given,
    also adds a rotating-style file handler (append mode) so each run extends
    the audit trail. The console banner output (`print` in `_banner` / `_row`)
    is left untouched — this only adds a parallel structured log.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)
    # Avoid duplicate handlers when run_pipeline is called more than once
    # in the same process (e.g., from a notebook).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    # Only emit to console at WARNING+ by default — banner prints carry the rest.
    sh.setLevel(logging.WARNING if lvl > logging.DEBUG else lvl)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False


# ---------------------------------------------------------------------------
# Pipeline Active Learning sub-routine
# ---------------------------------------------------------------------------

def _run_pipeline_al(
    cfg: PipelineConfig,
    loader: DataLoader,
    device: str,
) -> None:
    """Execute the Roth & Small 2008 multi-stage pipeline active learning path.

    Outputs written to cfg.output.dir:
        pipeline_al_candidates.csv
        pipeline_al_beta_history.csv
        pipeline_al_summary.json
        plots/pipeline_al_beta_evolution.png
        reports/pipeline_al_report.pdf
    """
    n_stages = len(cfg.pipeline_al.stages)
    _banner("PIPELINE ACTIVE LEARNING  (Roth & Small 2008 Algorithm 1)")
    _row("Stages",          n_stages)
    _row("Cycles",          cfg.pipeline_al.n_cycles)
    _row("Budget/cycle",    cfg.pipeline_al.budget_per_cycle)
    _row("Update rate λ",   cfg.pipeline_al.update_rate)

    stage_models = []
    stage_labels = []
    for i, stage_cfg in enumerate(cfg.pipeline_al.stages):
        _info(f"Loading stage {i}: {stage_cfg.arch} "
              f"({stage_cfg.num_classes} classes, task={stage_cfg.task})")
        m = load_model(
            arch=stage_cfg.arch,
            checkpoint=stage_cfg.checkpoint,
            num_classes=stage_cfg.num_classes,
            device=device,
            task=stage_cfg.task,
        )
        stage_models.append(m)
        stage_labels.append(stage_cfg.label)

    learner = PipelineActiveLearner(
        models=stage_models,
        tasks=[s.task for s in cfg.pipeline_al.stages],
        uncertainty_methods=[s.uncertainty_method for s in cfg.pipeline_al.stages],
        update_rate=cfg.pipeline_al.update_rate,
    )

    for cycle in range(cfg.pipeline_al.n_cycles):
        beta_str = "  ".join(f"{stage_labels[d]}={learner.beta[d]:.3f}"
                             for d in range(n_stages))
        _info(f"Cycle {cycle + 1}/{cfg.pipeline_al.n_cycles}  beta=[{beta_str}]")

        candidates = learner.select_candidates(
            dataloader=loader,
            budget=cfg.pipeline_al.budget_per_cycle,
            device=device,
        )
        _info(f"  Selected {len(candidates)} candidates.")

        # Retrain models on newly labeled data here in production before step()
        learner.step(dataloader=loader, device=device)

        if learner.has_converged():
            _ok(f"Beta converged at cycle {cycle + 1} - stopping early.")
            break

    # Final selection
    final_candidates = learner.select_candidates(
        dataloader=loader,
        budget=cfg.pipeline_al.budget_per_cycle,
        device=device,
    )

    print()
    print(learner.summary())

    # ── Export ────────────────────────────────────────────────────────
    reports_dir = os.path.join(cfg.output.dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Candidates
    cand_path = os.path.join(cfg.output.dir, "pipeline_al_candidates.csv")
    pd.DataFrame({"path": final_candidates}).to_csv(cand_path, index=False)
    _ok(f"Candidates saved: {cand_path}")

    # Full history CSV + summary JSON
    hist_path = os.path.join(cfg.output.dir, "pipeline_al_beta_history.csv")
    learner.save_history(hist_path, stage_labels=stage_labels)
    _ok(f"History CSV saved: {hist_path}")
    _ok(f"Summary JSON saved: {hist_path.replace('.csv', '_summary.json')}")

    # Beta evolution PNG
    plot_path = os.path.join(cfg.output.plots_dir, "pipeline_al_beta_evolution.png")
    learner.plot_beta_history(stage_labels=stage_labels, output_path=plot_path)
    _ok(f"Beta plot saved: {plot_path}")

    # PDF report
    pdf_path = os.path.join(reports_dir, "pipeline_al_report.pdf")
    learner.generate_report(pdf_path, stage_labels=stage_labels)
    _ok(f"PDF report saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Active Learning Visualisations
# ---------------------------------------------------------------------------

def _plot_candidate_uncertainty(
    uncertainty_scores: dict,
    selected_candidates: list,
    output_path: str,
) -> None:
    """Histogram of uncertainty scores: selected (red) vs pool (blue).

    Validates that the acquisition function is selecting high-uncertainty
    samples, not random ones (Settles 2010).

    Args:
        uncertainty_scores: {path: float} mapping.
        selected_candidates: list of selected paths.
        output_path: Where to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ks_2samp

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_paths = list(uncertainty_scores.keys())
    all_scores = np.array([uncertainty_scores[p] for p in all_paths])
    selected_set = set(selected_candidates)
    selected_mask = np.array([p in selected_set for p in all_paths])

    fig, ax = plt.subplots(figsize=(8, 5))
    n_pool = int((~selected_mask).sum())
    n_sel = int(selected_mask.sum())

    ax.hist(all_scores[~selected_mask], bins=40, alpha=0.6,
            color="steelblue", label=f"Pool (n={n_pool})", density=True)
    ax.hist(all_scores[selected_mask], bins=40, alpha=0.8,
            color="crimson", label=f"Selected (n={n_sel})", density=True)

    # KS-test: are distributions different?
    if n_sel >= 2 and n_pool >= 2:
        ks_stat, ks_p = ks_2samp(all_scores[selected_mask], all_scores[~selected_mask])
        ax.text(0.97, 0.97,
                f"KS stat = {ks_stat:.3f}\np = {ks_p:.4f}\n"
                + ("Distributions differ" if ks_p < 0.05 else "No significant difference"),
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#607D8B"))

    ax.set_xlabel("Uncertainty score")
    ax.set_ylabel("Density")
    ax.set_title("Candidate Selection: Uncertainty Distribution (Settles 2010)\n"
                 "Selected candidates should skew toward higher uncertainty",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_diversity(
    embeddings: dict,
    selected_candidates: list,
    output_path: str,
) -> None:
    """UMAP scatter of all samples coloured by selection status.

    Validates that selected candidates are diverse in feature space,
    not clustered in one region (Sener & Savarese 2018 — CoreSet).

    Args:
        embeddings: {path: np.ndarray} mapping (CNN features).
        selected_candidates: list of selected paths.
        output_path: Where to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import umap
    except ImportError:
        return  # skip silently if umap-learn not installed

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_paths = list(embeddings.keys())
    all_emb = np.array([embeddings[p] for p in all_paths])
    selected_set = set(selected_candidates)
    selected_mask = np.array([p in selected_set for p in all_paths])

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb_2d = reducer.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Pool points (grey, small)
    ax.scatter(emb_2d[~selected_mask, 0], emb_2d[~selected_mask, 1],
               s=8, alpha=0.3, color="#B0BEC5", label=f"Pool (n={int((~selected_mask).sum())})")
    # Selected points (red, larger)
    ax.scatter(emb_2d[selected_mask, 0], emb_2d[selected_mask, 1],
               s=30, alpha=0.8, color="crimson", edgecolors="white", linewidths=0.5,
               label=f"Selected (n={int(selected_mask.sum())})", zorder=5)

    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_title("Selection Diversity in Feature Space (Sener & Savarese 2018)\n"
                 "Red = selected candidates  ·  Diverse spread = good selection",
                 fontsize=10)
    ax.legend(fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute the full pipeline according to config.

    Args:
        cfg: Populated PipelineConfig dataclass (from load_config).
    """
    run_start = datetime.now()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility + structured logging — must run before any RNG is used.
    _setup_logging(level=cfg.runtime.log_level, log_file=cfg.runtime.log_file)
    _set_global_seed(cfg.runtime.seed)

    _banner(f"STEELDEFECT ACTIVE LEARNING PIPELINE  [{run_start.strftime('%Y-%m-%d %H:%M')}]")
    _row("Device",     device)
    _row("Output dir", cfg.output.dir)
    _row("Seed",       cfg.runtime.seed if cfg.runtime.seed is not None else "—")
    if cfg.runtime.log_file:
        _row("Log file",  cfg.runtime.log_file)
    logger.info(
        "pipeline.start device=%s seed=%s output=%s",
        device, cfg.runtime.seed, cfg.output.dir,
    )

    # Clear all outputs from previous runs so no stale files survive
    _info("Clearing previous outputs...")
    out      = Path(cfg.output.dir)
    plots    = Path(cfg.output.plots_dir)
    reports  = out / "reports"
    for subdir in (plots, reports):
        if subdir.exists():
            shutil.rmtree(subdir)
    for pattern in ("*.csv", "*.json"):
        for f in out.glob(pattern):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(cfg.data.image_dir):
        raise FileNotFoundError(
            f"Image directory not found: '{cfg.data.image_dir}'\n"
            f"  Run: python data/generate_sample_data.py  to create sample data."
        )

    # ── 1. Data ──────────────────────────────────────────────────────
    _banner("DATA")
    dataset = SteelDefectDataset(
        source=cfg.data.image_dir,
        image_size=tuple(cfg.data.image_size),
        extensions=cfg.data.extensions,
    )
    _row("Images found", len(dataset))
    _row("Image dir",    cfg.data.image_dir)

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        pin_memory=device == "cuda",
    )

    # ── 2. Models ─────────────────────────────────────────────────────
    _banner("MODELS")
    _row("Model A arch",  cfg.model_a.arch)
    _row("Model A task",  cfg.model_a.task)
    _row("Model A classes", cfg.model_a.num_classes)

    model_a = load_model(
        arch=cfg.model_a.arch,
        checkpoint=cfg.model_a.checkpoint,
        num_classes=cfg.model_a.num_classes,
        device=device,
        task=cfg.model_a.task,
    )

    # ── 2b. Calibration (optional, Guo et al. 2017) ──────────────────
    _temperature_scaler = None
    if cfg.calibration.enabled:
        _banner("CALIBRATION  (Temperature Scaling — Guo et al. 2017)")
        if cfg.calibration.val_labels_csv and os.path.exists(cfg.calibration.val_labels_csv):
            from steeldefect_active_learning.active_learning.calibration import TemperatureScaler
            import pandas as _pd_cal
            val_df = _pd_cal.read_csv(cfg.calibration.val_labels_csv)
            _row("Val set size",    len(val_df))
            _row("Temperature path", cfg.calibration.temperature_path)

            # Run model on val set to get logits
            _info("Running model on validation set to get logits ...")
            val_dataset = SteelDefectDataset(
                source=val_df["path"].tolist(),
                image_size=tuple(cfg.data.image_size),
                extensions=cfg.data.extensions,
            )
            val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size,
                                    num_workers=cfg.data.num_workers, shuffle=False)
            import torch as _torch
            all_logits, all_labels_list = [], []
            model_a.eval()
            with _torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    logits = model_a(imgs)
                    all_logits.append(logits.cpu().numpy())
            import numpy as _np_cal
            all_logits = _np_cal.concatenate(all_logits, axis=0)
            val_labels_arr = val_df["label"].values.astype(_np_cal.int64)

            _temperature_scaler = TemperatureScaler()
            _temperature_scaler.calibrate(all_logits, val_labels_arr)
            _temperature_scaler.save(cfg.calibration.temperature_path)

            cal_probs = _temperature_scaler.scale(all_logits)
            accuracy_after_cal = float((
                cal_probs.argmax(axis=1) == val_labels_arr
            ).astype(float).mean())  # accuracy on calibrated probs
            _ok(f"Temperature T* = {_temperature_scaler.temperature_:.4f}, accuracy = {accuracy_after_cal:.4f}")
            _ok(f"Saved to: {cfg.calibration.temperature_path}")

            # Save reliability diagram
            rel_diag_path = os.path.join(cfg.output.plots_dir, "calibration_reliability.png")
            _temperature_scaler.reliability_diagram(
                cal_probs, val_labels_arr,
                output_path=rel_diag_path,
                title=f"Model A Reliability Diagram (T={_temperature_scaler.temperature_:.3f})",
            )
            _ok(f"Reliability diagram: {rel_diag_path}")
        else:
            _warn("Calibration enabled but val_labels_csv not found — skipping.")
            _warn(f"  Set calibration.val_labels_csv in config to a CSV with 'path','label' columns.")

    # ── 3. Active Learning ───────────────────────────────────────────
    _banner("ACTIVE LEARNING")
    _row("Uncertainty method", cfg.active_learning.uncertainty_method)
    _row("Feature layer",      cfg.active_learning.feature_layer)
    _row("Clustering method",  cfg.active_learning.clustering_method)
    _row("n_clusters",         cfg.active_learning.n_clusters)
    _row("Labeling budget",    cfg.active_learning.labeling_budget)

    # 3a. Uncertainty
    # Apply spectral normalization for DDU (Mukhoti et al. 2022) — CRITICAL
    if cfg.active_learning.uncertainty_method == "ddu":
        from steeldefect_active_learning.active_learning.uncertainty import apply_spectral_normalization
        _info("Applying spectral normalization for DDU (Mukhoti 2022) ...")
        apply_spectral_normalization(model_a)

    # Temperature scaling for calibrated uncertainty (Guo 2017, Minderer 2021)
    _temperature = 1.0
    if _temperature_scaler is not None and hasattr(_temperature_scaler, "temperature_"):
        _temperature = float(_temperature_scaler.temperature_)
        _info(f"Using calibrated temperature T={_temperature:.4f} for uncertainty scoring")

    _info(f"Computing uncertainty [{cfg.active_learning.uncertainty_method}] ...")
    uncertainty_scores = compute_uncertainty(
        model=model_a,
        dataloader=loader,
        method=cfg.active_learning.uncertainty_method,
        device=device,
        task=cfg.model_a.task,
        normalize=True,
        temperature=_temperature,
    )
    scores_arr = list(uncertainty_scores.values())
    _ok(f"Uncertainty: n={len(scores_arr)}  "
        f"mean={sum(scores_arr)/len(scores_arr):.4f}  "
        f"max={max(scores_arr):.4f}")

    # Optional: compare a second uncertainty method for diagnostics
    alt_method = {"entropy": "bald", "bald": "entropy", "margin": "entropy"}.get(
        cfg.active_learning.uncertainty_method, None
    )
    if alt_method:
        _info(f"Computing comparison uncertainty [{alt_method}] ...")
        alt_scores = compute_uncertainty(
            model=model_a, dataloader=loader,
            method=alt_method, device=device,
            task=cfg.model_a.task, normalize=True,
        )
        alt_arr = list(alt_scores.values())
        _info(f"  [{alt_method}] mean={sum(alt_arr)/len(alt_arr):.4f}  "
              f"max={max(alt_arr):.4f}")
        # Rank correlation between methods
        import numpy as np
        from scipy.stats import spearmanr
        paths = list(uncertainty_scores.keys())
        r1 = [uncertainty_scores[p] for p in paths]
        r2 = [alt_scores[p]         for p in paths]
        rho, _ = spearmanr(r1, r2)
        _info(f"  Spearman r ({cfg.active_learning.uncertainty_method} vs {alt_method}): {rho:.4f}")

    # 3b. Feature embeddings
    _info(f"Extracting features (layer='{cfg.active_learning.feature_layer}') ...")
    embeddings = extract_features(
        model=model_a,
        dataloader=loader,
        layer_name=cfg.active_learning.feature_layer,
        device=device,
    )
    _ok(f"Embeddings: {len(embeddings)} images")

    # 3c. Clustering
    _info(f"Clustering [{cfg.active_learning.clustering_method}, "
          f"k={cfg.active_learning.n_clusters}] ...")
    cluster_ids = cluster_embeddings(
        embeddings=embeddings,
        n_clusters=cfg.active_learning.n_clusters,
        method=cfg.active_learning.clustering_method,
    )
    from collections import Counter
    cluster_sizes = Counter(cluster_ids.values())
    _ok(f"Clusters: {len(cluster_sizes)} groups  "
        f"sizes={sorted(cluster_sizes.values(), reverse=True)}")

    # 3d. Candidate selection — with budget-aware strategy recommendation
    _info(f"Selecting {cfg.active_learning.labeling_budget} candidates ...")
    selection_method = getattr(cfg.active_learning, "selection_method", "stratified")
    recommended = _auto_budget_strategy(cfg.active_learning.labeling_budget, n_labeled=0)
    if recommended != selection_method:
        _info(
            f"  [Auto-strategy] Budget={cfg.active_learning.labeling_budget}  "
            f"Configured='{selection_method}'  Recommended='{recommended}'  "
            f"({'Hacohen 2022: TypiClust optimal for budget<100' if recommended=='typiclust' else 'SIMILAR for imbalanced'})"
        )
    candidates = select_candidates(
        uncertainty_scores=uncertainty_scores,
        cluster_assignments=cluster_ids,
        labeling_budget=cfg.active_learning.labeling_budget,
        method=selection_method,
    )
    _ok(f"Selected {len(candidates)} candidates  (method={selection_method})")

    # 3e. Candidate uncertainty histogram (Settles 2010 validation)
    try:
        _plot_candidate_uncertainty(
            uncertainty_scores=uncertainty_scores,
            selected_candidates=candidates,
            output_path=os.path.join(cfg.output.plots_dir, "candidate_uncertainty.png"),
        )
        _ok(f"Candidate uncertainty plot: {cfg.output.plots_dir}/candidate_uncertainty.png")
    except Exception as _exc:
        import traceback
        _warn(f"Candidate uncertainty plot skipped: {_exc}")
        traceback.print_exc()

    # 3f. Selection diversity scatter (Sener & Savarese 2018 CoreSet validation)
    try:
        _plot_selection_diversity(
            embeddings=embeddings,
            selected_candidates=candidates,
            output_path=os.path.join(cfg.output.plots_dir, "selection_diversity.png"),
        )
        _ok(f"Selection diversity plot: {cfg.output.plots_dir}/selection_diversity.png")
    except Exception as _exc:
        import traceback
        _warn(f"Selection diversity plot skipped: {_exc}")
        traceback.print_exc()

    # 3g. Export candidates
    candidates_df = candidates_to_dataframe(candidates, uncertainty_scores, cluster_ids)
    out_path = os.path.join(cfg.output.dir, cfg.output.candidates_file)
    candidates_df.to_csv(out_path, index=False)
    _ok(f"Candidates CSV: {out_path}")

    # ── 3f. Anomaly detection (optional, PatchCore — Roth et al. 2022) ──
    if cfg.anomaly.enabled:
        _banner("ANOMALY DETECTION  (PatchCore — Roth et al. 2022)")
        _row("Normal class",    cfg.anomaly.normal_class)
        _row("Coreset ratio",   cfg.anomaly.coreset_ratio)
        _row("FPR threshold",   cfg.anomaly.fpr)
        _row("Model path",      cfg.anomaly.model_path)
        try:
            from steeldefect_active_learning.active_learning.anomaly import PatchCoreDetector
            from PIL import Image as _PILImage
            import numpy as _np_anom

            # Collect normal-class images from the dataset
            _normal_images = []
            for i in range(len(dataset)):
                item = dataset[i]
                meta = getattr(item, "metadata", {})
                label_str = str(meta.get("defect_type", meta.get("label", "")))
                if label_str == str(cfg.anomaly.normal_class):
                    img_path = item["path"] if isinstance(item, dict) else dataset.paths[i]
                    try:
                        _normal_images.append(_PILImage.open(img_path).convert("RGB"))
                    except Exception:
                        pass

            if len(_normal_images) == 0:
                _warn(f"No images found for normal_class='{cfg.anomaly.normal_class}' — "
                      "trying to load saved PatchCore model.")
                if os.path.exists(cfg.anomaly.model_path):
                    detector = PatchCoreDetector.load(cfg.anomaly.model_path)
                else:
                    _warn("  No saved model either. Skipping anomaly detection.")
                    detector = None
            else:
                _info(f"Fitting PatchCore on {len(_normal_images)} normal images ...")
                detector = PatchCoreDetector(
                    backbone=cfg.anomaly.backbone,
                    device=cfg.anomaly.device,
                    coreset_ratio=cfg.anomaly.coreset_ratio,
                )
                detector.fit(_normal_images)
                detector.save(cfg.anomaly.model_path)
                _ok(f"PatchCore fitted + saved: {cfg.anomaly.model_path}")

            if detector is not None:
                threshold = detector.threshold(fpr=cfg.anomaly.fpr)
                _info(f"Anomaly threshold (FPR={cfg.anomaly.fpr:.2f}): {threshold:.4f}")

                # Score all unlabeled images
                _anom_paths = list(uncertainty_scores.keys())
                _anom_images = []
                for p in _anom_paths:
                    try:
                        _anom_images.append(_PILImage.open(p).convert("RGB"))
                    except Exception:
                        _anom_images.append(None)

                anom_scores_list = []
                for img in _anom_images:
                    if img is not None:
                        try:
                            anom_scores_list.append(detector.score([img])[0])
                        except Exception:
                            anom_scores_list.append(0.0)
                    else:
                        anom_scores_list.append(0.0)

                anom_scores_map = dict(zip(_anom_paths, anom_scores_list))
                n_anomalies = sum(1 for s in anom_scores_list if s > threshold)
                _ok(f"Anomaly scoring complete: {n_anomalies}/{len(_anom_paths)} "
                    f"images above threshold ({cfg.anomaly.fpr:.0%} FPR)")

                # Annotate candidates CSV
                candidates_df["anomaly_score"] = candidates_df["path"].map(
                    lambda p: anom_scores_map.get(p, 0.0)
                )
                candidates_df["is_anomaly"] = candidates_df["anomaly_score"] > threshold
                candidates_df.to_csv(out_path, index=False)
                _ok(f"Candidates CSV updated with anomaly scores: {out_path}")

                # Save full anomaly scores CSV
                anom_csv = os.path.join(cfg.output.dir, "anomaly_scores.csv")
                pd.DataFrame({
                    "path": _anom_paths,
                    "anomaly_score": anom_scores_list,
                    "is_anomaly": [s > threshold for s in anom_scores_list],
                }).to_csv(anom_csv, index=False)
                _ok(f"Full anomaly scores: {anom_csv}")

        except ImportError as e:
            _warn(f"Anomaly detection requires torch + torchvision: {e}")

    # ── 4. Pipeline Active Learning (Roth & Small 2008) ──────────────
    if cfg.pipeline_al.enabled and len(cfg.pipeline_al.stages) > 0:
        _run_pipeline_al(cfg, loader, device)

    # ── 5. A/B Testing ───────────────────────────────────────────────
    if cfg.ab_testing.enabled:
        _banner("A/B TESTING  (Quin & Weyns 2023)")
        _row("Model B arch",  cfg.model_b.arch)
        _row("Model B task",  cfg.model_b.task)
        _row("iou_threshold", cfg.ab_testing.iou_threshold)
        _row("conf_threshold",cfg.ab_testing.confidence_threshold)

        # ── Demo-data shortcut ────────────────────────────────────────
        # When both checkpoints are null (no trained weights), random-weight
        # inference produces degenerate results: each model develops a systematic
        # bias toward a single class -> 100% disagreement, all margins ~0.026,
        # all disagreements classified as Borderline. Instead, load pre-generated
        # realistic predictions from demo_predictions_path when available.
        _both_null = (cfg.model_a.checkpoint is None and cfg.model_b.checkpoint is None)
        _demo_path = cfg.ab_testing.demo_predictions_path

        if _both_null and _demo_path and os.path.exists(_demo_path):
            _info(f"No checkpoints configured - loading demo predictions: {_demo_path}")
            ab_df = pd.read_csv(_demo_path)
            _ok(f"Demo data: {len(ab_df)} rows  "
                f"disagreement={1 - ab_df['agrees'].mean():.1%}  "
                f"classes={ab_df['pred_a'].nunique()}")
        else:
            model_b = load_model(
                arch=cfg.model_b.arch,
                checkpoint=cfg.model_b.checkpoint,
                num_classes=cfg.model_b.num_classes,
                device=device,
                task=cfg.model_b.task,
            )

            _info("Running A/B comparison ...")
            ab_df = run_ab_comparison(
                model_a=model_a,
                model_b=model_b,
                dataloader=loader,
                task=cfg.model_a.task,
                device=device,
                iou_threshold=cfg.ab_testing.iou_threshold,
                confidence_threshold=cfg.ab_testing.confidence_threshold,
                store_probs=True,
            )

        # ── Full metric suite ─────────────────────────────────────────
        all_metrics = compute_all_metrics(
            ab_df,
            task=cfg.model_a.task,
            alpha=cfg.ab_testing.alpha,
            correction=cfg.ab_testing.multiple_testing_correction,
            srm_threshold=cfg.ab_testing.srm_threshold,
            expected_ratio=cfg.ab_testing.expected_arm_ratio,
            cuped_covariate=cfg.ab_testing.cuped_covariate,
            bootstrap_n=cfg.ab_testing.bootstrap_n,
            bayesian=True,  # always run; cheap given existing data
            bayesian_decision_threshold=cfg.ab_testing.bayesian_decision_threshold,
            sequential=cfg.ab_testing.sequential,
        )
        stat  = all_metrics["statistical"]
        pwr   = all_metrics["power"]
        delta = all_metrics["delta"]
        dr    = all_metrics["disagreement"]["rate"]
        srm   = all_metrics.get("srm", {})

        print()
        if srm.get("srm_detected"):
            _warn(
                f"SRM DETECTED  observed_ratio_a={srm.get('observed_ratio_a'):.3f} "
                f"(expected {srm.get('expected_ratio'):.3f})  "
                f"chi2={srm.get('chi2'):.2f}  p={srm.get('p_value'):.2e}  "
                "— results may be invalid; investigate the randomiser."
            )
        else:
            _row("SRM check",
                 f"observed_ratio_a={srm.get('observed_ratio_a', float('nan')):.3f}  "
                 f"p={srm.get('p_value', float('nan')):.3f}  [OK]")

        _row("Disagreement rate",  f"{dr:.1%}  "
             + ("[!] HIGH" if dr > 0.15 else "[OK]"))
        _row("Welch t / p-value",  f"t={stat['t_stat']:.3f}  "
             f"p={stat['p_value']:.4f}  "
             + ("-> SIGNIFICANT" if stat["significant"] else "-> not significant"))
        _row("Cohen's d (Hedges)",
             f"{stat['cohens_d']:.4f} / {stat.get('cohens_d_hedges', float('nan')):.4f}  "
             f"({stat['effect_magnitude']})")
        _row("95% CI (A-B)",       f"({stat['confidence_interval_95'][0]:.4f}, "
             f"{stat['confidence_interval_95'][1]:.4f})")
        _row("Statistical power",  f"{pwr['achieved_power']:.1%}  "
             + ("[OK] adequate" if pwr["adequately_powered"] else
                f"[!] need n>={pwr['required_n_for_80pct']} per group"))
        _row("Conf delta mean/p90",f"{delta['mean']:.4f} / {delta['p90']:.4f}")
        if "mcnemar" in all_metrics:
            mc = all_metrics["mcnemar"]
            _row("McNemar (paired)",
                 f"b={mc['b']} c={mc['c']}  stat={mc['statistic']:.3f}  "
                 f"p={mc['p_value']:.4f}  [{mc['test_used']}]")
        if "per_class_mcnemar" in all_metrics:
            pcm = all_metrics["per_class_mcnemar"]
            n_sig = sum(1 for r in pcm["rows"] if r["reject"])
            _row(f"Per-class McNemar ({pcm['method']})",
                 f"{n_sig}/{len(pcm['rows'])} classes reject H0  alpha={pcm['alpha']}")
        if "entropy" in all_metrics:
            ent = all_metrics["entropy"]
            _row("Entropy A vs B",
                 f"H_A={ent.get('mean_entropy_a', 0):.4f}  "
                 f"H_B={ent.get('mean_entropy_b', 0):.4f}  "
                 f"(winner={ent.get('entropy_winner', '?')})")
        if "segmentation" in all_metrics:
            seg = all_metrics["segmentation"]
            _row("Mean IoU",    f"{seg.get('mean_iou', 0):.4f}")
            _row("Mean Dice",   f"{seg.get('mean_dice', 0):.4f}")
            _row("Boundary IoU",f"{seg.get('mean_boundary_iou', 0):.4f}")

        # ── Export ───────────────────────────────────────────────────
        reports_dir = os.path.join(cfg.output.dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Raw comparison CSV
        ab_out = os.path.join(cfg.output.dir, cfg.output.ab_report_file)
        ab_df.to_csv(ab_out, index=False)
        _ok(f"A/B report CSV: {ab_out}")

        # All metrics JSON
        metrics_json = os.path.join(cfg.output.dir, "ab_metrics.json")
        _dump_metrics_json(all_metrics, metrics_json)
        _ok(f"Metrics JSON:   {metrics_json}")

        # Standard diagnostic PNGs
        plot_comparison(
            df=ab_df,
            output_dir=cfg.output.plots_dir,
            task=cfg.model_a.task,
        )
        _ok(f"Standard plots: {cfg.output.plots_dir}/ab_*.png")

        # Advanced diagnostic PNGs (entropy, calibration, bands, margin)
        plot_advanced_comparison(
            df=ab_df,
            output_dir=cfg.output.plots_dir,
            model_a_name=cfg.model_a.arch,
            model_b_name=cfg.model_b.arch,
        )
        _ok(f"Advanced plots: {cfg.output.plots_dir}/ab_entropy_*.png etc.")

        # Deep-dive advanced plots (KDE, violin, entropy CDF, disagreement decomposition)
        plot_advanced_comparison_v2(
            df=ab_df,
            output_dir=cfg.output.plots_dir,
            prefix="ab",
            model_a_name=cfg.model_a.arch,
            model_b_name=cfg.model_b.arch,
        )
        _ok(f"Deep-dive plots: {cfg.output.plots_dir}/ab_kde_*.png etc.")

        # Multi-page PDF report
        pdf_path = os.path.join(reports_dir, "ab_report.pdf")
        generate_ab_report(
            df=ab_df,
            output_path=pdf_path,
            model_a_name=cfg.model_a.arch,
            model_b_name=cfg.model_b.arch,
            task=cfg.model_a.task,
        )
        _ok(f"PDF report:     {pdf_path}")

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = (datetime.now() - run_start).total_seconds()
    _banner(f"DONE  (elapsed: {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump_metrics_json(metrics: dict, path: str) -> None:
    """Serialise the metrics dict to JSON, converting numpy types."""
    import numpy as np

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="SteelDefect Active Learning + A/B Testing Pipeline")
    p.add_argument("--config", default="steeldefect_active_learning/config/pipeline_config.yaml",
                   help="Path to pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_config(args.config)
    run_pipeline(config)

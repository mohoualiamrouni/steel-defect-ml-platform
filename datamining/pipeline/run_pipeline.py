"""
Datamining & Cutting Plan -- Pipeline
======================================

Orchestrates the full datamining workflow:
    1. Generate synthetic coil data (if not present)
    2. Load defects DataFrame
    3. Train XGBoost severity model + SHAP
    4. For each coil:
        a. Build 2D quality heatmap from predicted severities
        b. Detect change points along length and width
        c. Run 2D guillotine cutting optimisation
        d. Collect results
    5. Generate PDF report for a sample coil
    6. Save all artefacts

Usage::

    python -m datamining.pipeline.run_pipeline
    python -m datamining.pipeline.run_pipeline --n-coils 100 --penalty 2.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


def summarize_h3(results_df: pd.DataFrame, n_boot: int = 100,
                 random_state: int = 42) -> dict:
    """Test hypothesis H3 from the per-coil cutting results (area-weighted).

    H3a: mean grade-A area fraction ≥ 0.25.  H3b: mean scrap area fraction ≤ 0.05.
    Returns the means, 95% bootstrap CIs (resampling coils), per-coil
    satisfaction counts, and pass/fail — so the report tests H3 instead of
    asserting it.
    """
    a = results_df.get("area_frac_A")
    s = results_df.get("area_frac_scrap")
    if a is None or s is None:
        return {"available": False}
    a = a.to_numpy(dtype=float)
    s = s.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(a)

    def _ci(x):
        if n < 2:
            return [float(x.mean()), float(x.mean())]
        boot = [float(x[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
        return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    mean_a, mean_s = float(a.mean()), float(s.mean())
    return {
        "available": True,
        "grade_A_frac_mean": round(mean_a, 4),
        "grade_A_frac_ci95": [round(v, 4) for v in _ci(a)],
        "scrap_frac_mean": round(mean_s, 4),
        "scrap_frac_ci95": [round(v, 4) for v in _ci(s)],
        "n_coils_grade_A_ge_25pct": int((a >= 0.25).sum()),
        "n_coils_scrap_le_5pct": int((s <= 0.05).sum()),
        "n_coils": n,
        "H3a_pass_grade_A_ge_0.25": bool(mean_a >= 0.25),
        "H3b_pass_scrap_le_0.05": bool(mean_s <= 0.05),
    }


def run_pipeline(
    config: Optional["DataminingConfig"] = None,
    **overrides,
) -> dict:
    """Run the complete datamining + cutting plan pipeline.

    Parameters
    ----------
    config : DataminingConfig, optional
        Configuration object.  If ``None``, default values are used.
    **overrides
        Any config field can be overridden as a keyword argument.

    Returns
    -------
    dict
        Summary metrics: mean yield, model R², per-grade counts, etc.
    """
    from datamining.config import DataminingConfig

    if config is None:
        config = DataminingConfig()
    for key, val in overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)

    # Seed top-level RNGs so sklearn internals not reached by per-estimator
    # random_state are still reproducible across runs.
    np.random.seed(config.random_state)
    import random as _random
    _random.seed(config.random_state)

    out = Path(config.output_dir)
    plots_dir = out / "plots"
    reports_dir = out / "reports"
    for d in [out, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data if needed
    # ------------------------------------------------------------------
    defects_csv = os.path.join(config.synthetic_output_dir, "defects.csv")
    coils_csv = os.path.join(config.synthetic_output_dir, "coils_metadata.csv")

    if not os.path.isfile(defects_csv):
        print("[pipeline] Generating synthetic coil data ...")
        from datamining.data.generate_synthetic_coils import generate_synthetic_coils

        generate_synthetic_coils(
            output_dir=config.synthetic_output_dir,
            n_coils=config.n_coils,
            coil_length_range=config.coil_length_range,
            coil_width_range=config.coil_width_range,
            defects_per_coil_range=config.defects_per_coil_range,
            mask_size=config.mask_size,
            random_state=config.random_state,
        )
    else:
        print("[pipeline] Synthetic data already exists, skipping generation.")

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    print("[pipeline] Loading data ...")
    defects_df = pd.read_csv(defects_csv)
    coils_df = pd.read_csv(coils_csv)
    print(f"  {len(coils_df)} coils, {len(defects_df)} defects total")

    # ------------------------------------------------------------------
    # Step 3: Feature selection, tuning, and severity model training
    # ------------------------------------------------------------------
    from datamining.features.severity_model import (
        boruta_select,
        compute_shap_explanations,
        optuna_tune,
        plot_optuna_history,
        predict_severity,
        save_severity_model,
        train_severity_model,
    )

    # Identify feature columns (24 segmentation features)
    feat_cols = [c for c in defects_df.columns if c.startswith("feat_")]
    if not feat_cols:
        meta_cols = {"coil_id", "defect_id", "x", "y", "defect_type",
                     "expert_group", "severity_gt"}
        feat_cols = sorted([c for c in defects_df.columns if c not in meta_cols
                            and defects_df[c].dtype in [np.float64, np.float32]])

    # Expose expert_group as a feature.  base_sev varies 0.2..0.8 across the
    # 28 expert groups in the simulator (see generate_synthetic_coils.py),
    # which makes it the dominant determinant of severity_gt.  Without this
    # one-hot encoding, the regressor only sees morphology and R^2 is
    # structurally capped around 0.05 by the simulator's variance budget.
    from sklearn.preprocessing import OneHotEncoder

    group_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    group_oh = group_enc.fit_transform(defects_df[["expert_group"]])
    group_feat_names = [f"group_{c}" for c in group_enc.categories_[0]]

    n_morpho  = len(feat_cols)   # the morphology columns precede the group one-hots
    features  = np.hstack([defects_df[feat_cols].values, group_oh])
    feat_cols = list(feat_cols) + group_feat_names
    targets   = defects_df["severity_gt"].values

    # Persist the encoder next to the trained model so inference on new
    # coils uses the same group → one-hot mapping.
    joblib.dump(group_enc, str(out / "group_encoder.joblib"))

    # --- Boruta feature selection (Kursa & Rudnicki 2010) ---
    boruta_result = None
    if config.use_boruta:
        print("[pipeline] Running Boruta feature selection (morphology block only) ...")
        # Boruta is applied ONLY to the 24 morphological descriptors. The
        # expert-group one-hot is the known causal driver of severity (base_sev
        # is group-determined) and is available at inference (the inspection system emits the
        # group), so it is ALWAYS retained — feature-selecting away the dominant
        # predictor would artificially cap R². (This is the fix for the earlier
        # Boruta tell where the group encoding was being dropped.)
        boruta_result = boruta_select(
            features[:, :n_morpho], targets, feat_cols[:n_morpho],
            random_state=config.random_state,
        )
        selected = boruta_result["selected_mask"]
        n_sel = int(selected.sum())
        print(f"  {n_sel}/{n_morpho} morphology features selected "
              f"({len(boruta_result['confirmed'])} confirmed, "
              f"{len(boruta_result['tentative'])} tentative, "
              f"{len(boruta_result['rejected'])} rejected); "
              f"{len(feat_cols) - n_morpho} group one-hots always kept")
        keep_morpho = [i for i in range(n_morpho) if selected[i]]
        group_idx = list(range(n_morpho, features.shape[1]))
        keep_idx = keep_morpho + group_idx
        features = features[:, keep_idx]
        feat_cols = [feat_cols[i] for i in keep_idx]
        if boruta_result["boruta_fig"] is not None:
            boruta_result["boruta_fig"].savefig(
                str(plots_dir / "boruta_selection.png"), dpi=150, bbox_inches="tight",
            )

    # --- Optuna hyperparameter tuning (Akiba et al. 2019) ---
    optuna_study = None
    optuna_fig = None
    xgb_params = dict(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
    )
    if config.use_optuna:
        print("[pipeline] Running Optuna hyperparameter tuning ...")
        best_params, optuna_study = optuna_tune(
            features, targets,
            n_trials=config.optuna_n_trials,
            random_state=config.random_state,
        )
        if optuna_study is not None:
            xgb_params = best_params
            optuna_fig = plot_optuna_history(optuna_study)
            optuna_fig.savefig(
                str(plots_dir / "optuna_history.png"), dpi=150, bbox_inches="tight",
            )

    # --- Train severity model ---
    print("[pipeline] Training XGBoost severity model ...")
    model, metrics = train_severity_model(
        features, targets,
        **xgb_params,
        random_state=config.random_state,
    )
    print(f"  R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")

    save_severity_model(model, str(out / "severity_model.joblib"))

    # Predict severity for all defects using K-fold out-of-fold predictions.
    # Using in-sample predictions (predict_severity on the training matrix) would
    # leak the model's fit into the downstream cutting optimisation, biasing
    # the estimated yield upwards.  OOF removes that leakage at the cost of
    # training K extra models.
    from sklearn.model_selection import KFold

    _KFOLD = max(3, min(5, len(features) // 200))
    kf = KFold(n_splits=_KFOLD, shuffle=True, random_state=config.random_state)
    oof = np.empty(len(features), dtype=np.float32)
    for tr_idx, val_idx in kf.split(features):
        fold_model, _ = train_severity_model(
            features[tr_idx], targets[tr_idx],
            **xgb_params,
            random_state=config.random_state,
        )
        oof[val_idx] = predict_severity(fold_model, features[val_idx])
    defects_df["severity"] = oof

    # ------------------------------------------------------------------
    # Optional: Conformalised Quantile Regression (Romano et al. 2019)
    # Attaches severity_lo / severity_hi prediction intervals with
    # finite-sample valid coverage at level (1 - cqr_alpha).  Off by
    # default; enabling does not change the existing severity column.
    # ------------------------------------------------------------------
    if getattr(config, "use_cqr", False):
        from datamining.features.severity_intervals import ConformalSeverity
        from sklearn.model_selection import GroupShuffleSplit

        print(f"[pipeline] Fitting CQR (alpha={config.cqr_alpha}, "
              f"target coverage={1 - config.cqr_alpha:.2f}) ...")
        # Split into train / calibration (60% / 40%) for the conformal step,
        # DISJOINT BY coil_id.  A plain random split would put defects from the
        # same coil in both train and calibration, leaking spatial correlation
        # and inflating the reported coverage.  GroupShuffleSplit on coil_id
        # keeps calibration coils strictly separate from training coils, so the
        # empirical coverage is an honest estimate (consistent with the
        # report's "séparation stricte par coil_id" protocol).
        groups = defects_df["coil_id"].to_numpy()
        _gss = GroupShuffleSplit(
            n_splits=1, test_size=0.4, random_state=config.random_state
        )
        _tr_idx, _cal_idx = next(_gss.split(features, targets, groups))
        X_tr, X_cal = features[_tr_idx], features[_cal_idx]
        y_tr, y_cal = targets[_tr_idx], targets[_cal_idx]
        cqr = ConformalSeverity(
            alpha=float(config.cqr_alpha),
            random_state=config.random_state,
        ).fit(X_tr, y_tr, X_cal, y_cal)
        cov = cqr.empirical_coverage(X_cal, y_cal)
        width = cqr.empirical_width(X_cal)
        print(f"  CQR empirical coverage on calibration split: {cov:.3f} "
              f"(target {1 - config.cqr_alpha:.2f})")
        print(f"  CQR mean interval width: {width:.4f}")
        lo, hi = cqr.predict_interval(features)
        defects_df["severity_lo"] = lo.astype(np.float32)
        defects_df["severity_hi"] = hi.astype(np.float32)
        joblib.dump(cqr, str(out / "cqr_severity.joblib"))

        # Optional: grade by the upper bound rather than the point estimate.
        # This is risk-averse: pieces are graded as if every defect were
        # at its 95th-percentile predicted severity.
        if getattr(config, "grade_by", "mean") == "upper":
            print("[pipeline] grade_by='upper' — using severity_hi for grading")
            defects_df["severity"] = defects_df["severity_hi"]

    # SHAP explanations + interactions + dependence
    print("[pipeline] Computing SHAP explanations ...")
    shap_values, fig_beeswarm, fig_bar, fig_dep, fig_interact = \
        compute_shap_explanations(
            model, features, feat_cols,
            n_background=min(config.shap_n_samples, len(features)),
            output_dir=str(plots_dir),
        )

    # ------------------------------------------------------------------
    # Step 4: Per-coil cutting optimisation
    # ------------------------------------------------------------------
    print("[pipeline] Running cutting optimisation per coil ...")
    from datamining.quality.quality_profile import (
        build_quality_heatmap,
        assign_quality_grades,
    )
    from datamining.quality.change_points import detect_length_change_points
    from datamining.cutting.constraints import CuttingConstraints
    from datamining.cutting.guillotine_optimizer import compute_cutting_plan

    constraints = CuttingConstraints(
        min_piece_length=config.min_piece_length,
        min_piece_width=config.min_piece_width,
        quality_thresholds=config.quality_thresholds,
        grade_values=config.grade_values,
        grade_names=config.grade_names,
    )

    coil_results = []
    sample_plan = None
    sample_heatmap = None
    sample_coil = None

    for idx, coil_row in coils_df.iterrows():
        coil_id = coil_row["coil_id"]
        length = coil_row["length"]
        width = coil_row["width"]

        coil_defects = defects_df[defects_df["coil_id"] == coil_id]

        # Build quality heatmap
        heatmap = build_quality_heatmap(
            coil_defects, length, width,
            grid_shape=config.quality_grid_shape,
            kernel_bandwidth=config.kernel_bandwidth,
        )

        # Detect change points along length.
        # Narrow except: only swallow the specific "cannot segment" conditions
        # from ruptures (too few samples, bad penalty). Everything else — shape
        # mismatches, NaN arrays, library bugs — should surface loudly.
        length_profile = heatmap.mean(axis=1)
        try:
            cp_hints = detect_length_change_points(
                length_profile,
                penalty=config.changepoint_penalty,
                model=config.changepoint_model,
            )
            # Remove the trailing endpoint that ruptures always adds
            cp_hints = [c for c in cp_hints if c < len(length_profile)]
        except (ValueError, IndexError) as cp_err:
            # ruptures raises ValueError on too-short signal / bad penalty;
            # log so the failure is visible rather than silently "no change points".
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Change-point detection failed for coil %s: %s",
                coil_id, cp_err,
            )
            cp_hints = []

        # Optimise cutting plan (defect-driven grading)
        plan = compute_cutting_plan(
            defects_df=coil_defects,
            constraints=constraints,
            coil_length=length,
            coil_width=width,
            quality_heatmap=heatmap,
            change_point_hints=cp_hints,
            n_length_bins=config.quality_grid_shape[0],
            n_width_bins=config.quality_grid_shape[1],
        )

        _fracs = plan.grade_area_fractions()
        coil_results.append({
            "coil_id": coil_id,
            "length": length,
            "width": width,
            "n_defects": len(coil_defects),
            "n_pieces": plan.n_pieces,
            "total_value": plan.total_value,
            "max_value": plan.max_possible_value,
            "yield_pct": plan.yield_pct,
            **{f"n_{g}": plan.grade_distribution.get(g, 0)
               for g in constraints.grade_names},
            # area-weighted grade fractions (H3 evidence)
            **{f"area_frac_{g}": float(_fracs.get(g, 0.0))
               for g in constraints.grade_names},
        })

        # Keep first coil for the sample report
        if sample_plan is None:
            sample_plan = plan
            sample_heatmap = heatmap
            sample_coil = coil_row
            sample_defects = coil_defects

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(coils_df)} coils")

    results_df = pd.DataFrame(coil_results)
    results_df.to_csv(str(out / "cutting_results.csv"), index=False)
    print(f"  Mean yield: {results_df['yield_pct'].mean():.1f}%")

    # ------------------------------------------------------------------
    # Step 5: Generate PDF report for sample coil
    # ------------------------------------------------------------------
    print("[pipeline] Generating PDF report ...")
    from datamining.report.report import generate_datamining_report

    report_path = generate_datamining_report(
        coil_id=sample_coil["coil_id"],
        defects_df=sample_defects,
        quality_heatmap=sample_heatmap,
        cutting_plan=sample_plan,
        severity_model=model,
        shap_values=shap_values,
        feature_names=feat_cols,
        coil_length=sample_coil["length"],
        coil_width=sample_coil["width"],
        constraints=constraints,
        output_path=str(reports_dir / "cutting_report.pdf"),
        boruta_fig=boruta_result["boruta_fig"] if boruta_result else None,
        optuna_fig=optuna_fig,
        shap_dependence_fig=fig_dep,
        shap_interaction_fig=fig_interact,
    )
    print(f"  Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Step 6: Save summary
    # ------------------------------------------------------------------
    summary = {
        "n_coils": len(coils_df),
        "n_defects_total": len(defects_df),
        "severity_model_r2": float(metrics["r2"]),
        "severity_model_rmse": float(metrics["rmse"]),
        "severity_model_mae": float(metrics.get("mae", 0.0)),
        "mean_yield_pct": float(results_df["yield_pct"].mean()),
        "std_yield_pct": float(results_df["yield_pct"].std()),
        "min_yield_pct": float(results_df["yield_pct"].min()),
        "max_yield_pct": float(results_df["yield_pct"].max()),
        "mean_n_pieces": float(results_df["n_pieces"].mean()),
        "grade_totals": {
            g: int(results_df[f"n_{g}"].sum())
            for g in constraints.grade_names
        },
        "h3": summarize_h3(results_df, random_state=config.random_state),
    }

    import json
    metrics_path = str(out / "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    print("[pipeline] Done.")
    print(f"  Model R² = {summary['severity_model_r2']:.4f}")
    print(f"  Mean yield = {summary['mean_yield_pct']:.1f}% "
          f"± {summary['std_yield_pct']:.1f}%")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Datamining & cutting plan pipeline"
    )
    parser.add_argument("--n-coils", type=int, default=50)
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="Change-point penalty (higher = fewer cuts)")
    parser.add_argument("--min-piece-length", type=float, default=5.0)
    parser.add_argument("--min-piece-width", type=float, default=0.3)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="datamining/outputs")

    args = parser.parse_args()
    run_pipeline(
        n_coils=args.n_coils,
        changepoint_penalty=args.penalty,
        min_piece_length=args.min_piece_length,
        min_piece_width=args.min_piece_width,
        xgb_n_estimators=args.xgb_estimators,
        random_state=args.seed,
        output_dir=args.output_dir,
    )

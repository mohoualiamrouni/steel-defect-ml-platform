"""Train an XGBoost regressor to predict defect severity from segmentation
features, with SMOTE oversampling (Chawla et al. 2002) for the minority
high-severity class, Boruta feature selection (Kursa & Rudnicki 2010),
Optuna hyperparameter tuning (Akiba et al. 2019), and SHAP explainability
with interaction values (Lundberg et al. 2020).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score


def _apply_smote(
    features: np.ndarray,
    targets: np.ndarray,
    bins: tuple[float, ...] = (0.3, 0.6),
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance severity classes before regression.

    Severity is binned into discrete classes (low / medium / high) for
    SMOTE's k-NN sampling, then the original continuous targets are
    propagated to the synthetic samples via nearest-neighbour
    interpolation.

    Implements the algorithm from Chawla et al. (JAIR 2002, Section 4.2):
    for each minority sample, generate a synthetic point along the line
    segment to one of its k nearest neighbours.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        # Graceful fallback: skip SMOTE if imbalanced-learn is missing
        return features, targets

    severity_classes = np.digitize(targets, bins)

    # SMOTE needs at least k_neighbors+1 samples per class
    unique, counts = np.unique(severity_classes, return_counts=True)
    min_count = counts.min()
    if min_count <= k_neighbors:
        k_neighbors = max(1, min_count - 1)
    if k_neighbors < 1 or len(unique) < 2:
        return features, targets

    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    # SMOTE works on features + class labels
    combined = np.column_stack([features, targets])
    combined_resampled, _ = smote.fit_resample(combined, severity_classes)

    features_resampled = combined_resampled[:, :-1]
    targets_resampled = combined_resampled[:, -1]

    return features_resampled, targets_resampled


def boruta_select(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    random_state: int = 42,
) -> dict:
    """Run Boruta feature selection (Kursa & Rudnicki, JOSS 2010).

    Boruta compares each feature's importance Z-score against the MZSA
    (Maximum Z-Score Among Shadow Attributes — shuffled copies of all
    features).  Features consistently beating the MZSA are *confirmed*;
    those consistently below are *rejected*; the rest are *tentative*.

    Returns a dict with keys: selected_mask (bool array), confirmed,
    rejected, tentative (lists of feature names), ranking (int array),
    and boruta_fig (matplotlib Figure or None).
    """
    result: dict = {
        "selected_mask": np.ones(features.shape[1], dtype=bool),
        "confirmed": list(feature_names),
        "rejected": [],
        "tentative": [],
        "ranking": np.ones(features.shape[1], dtype=int),
        "boruta_fig": None,
    }

    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        warnings.warn("boruta not installed — skipping feature selection.")
        return result

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=7, n_jobs=-1, random_state=random_state,
    )
    selector = BorutaPy(
        rf, n_estimators="auto", random_state=random_state,
        max_iter=100, verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(features, targets)

    confirmed_mask = selector.support_
    tentative_mask = selector.support_weak_
    selected_mask = confirmed_mask | tentative_mask

    # If Boruta rejects everything, keep all features
    if not selected_mask.any():
        warnings.warn("Boruta rejected all features — keeping all.")
        return result

    confirmed = [feature_names[i] for i in range(len(feature_names)) if confirmed_mask[i]]
    rejected = [feature_names[i] for i in range(len(feature_names))
                if not confirmed_mask[i] and not tentative_mask[i]]
    tentative = [feature_names[i] for i in range(len(feature_names)) if tentative_mask[i]]

    # Build visualization
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ranking = selector.ranking_
    colors = []
    for i in range(len(feature_names)):
        if confirmed_mask[i]:
            colors.append("#27ae60")  # green
        elif tentative_mask[i]:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    sorted_idx = np.argsort(ranking)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        [ranking[i] for i in sorted_idx],
        color=[colors[i] for i in sorted_idx],
    )
    ax.set_xlabel("Boruta Ranking (1 = confirmed)")
    ax.set_title(
        f"Boruta Feature Selection — "
        f"{len(confirmed)} confirmed, {len(tentative)} tentative, "
        f"{len(rejected)} rejected"
    )
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Confirmed"),
        Patch(facecolor="#f39c12", label="Tentative"),
        Patch(facecolor="#e74c3c", label="Rejected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()

    result.update({
        "selected_mask": selected_mask,
        "confirmed": confirmed,
        "rejected": rejected,
        "tentative": tentative,
        "ranking": ranking,
        "boruta_fig": fig,
    })
    return result


def optuna_tune(
    features: np.ndarray,
    targets: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple[dict, "optuna.Study | None"]:
    """Tune XGBoost hyperparameters with Optuna (Akiba et al., KDD 2019).

    Uses TPE (Tree-structured Parzen Estimator) sampler with 3-fold CV
    to minimize RMSE.  Returns (best_params dict, study object).
    Falls back to default params if optuna is not installed.
    """
    defaults = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=1.0,
        reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1,
    )

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        warnings.warn("optuna not installed — using default hyperparameters.")
        return defaults, None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = xgb.XGBRegressor(
            **params, random_state=random_state, objective="reg:squarederror",
        )
        scores = cross_val_score(
            model, features, targets, cv=3,
            scoring="neg_root_mean_squared_error",
        )
        return -scores.mean()  # minimize RMSE

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Optuna best RMSE: {study.best_value:.4f} (trial #{study.best_trial.number})")

    return best, study


def plot_optuna_history(study: "optuna.Study") -> plt.Figure:
    """Create a 2×2 figure summarizing the Optuna optimization.

    Panels: (1) optimization history, (2) parallel coordinate of top trials,
    (3) hyperparameter importances, (4) slice plot of top-2 params.
    """
    import optuna

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))

    # Panel 1: Optimization history (convergence)
    ax = axes[0, 0]
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = np.minimum.accumulate(values)
    ax.plot(range(len(values)), values, "o", ms=3, alpha=0.4, label="Trial RMSE")
    ax.plot(range(len(best_so_far)), best_so_far, "r-", lw=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("RMSE (3-fold CV)")
    ax.set_title("Optimization History")
    ax.legend(fontsize=8)

    # Panel 2: Top-10 trials parallel coordinate
    ax = axes[0, 1]
    top_trials = sorted(
        [t for t in trials if t.value is not None],
        key=lambda t: t.value,
    )[:10]
    param_names = list(study.best_params.keys())
    # Normalize params to [0, 1] for parallel coordinate
    param_ranges = {}
    for p in param_names:
        vals = [t.params.get(p, 0) for t in top_trials]
        lo, hi = min(vals), max(vals)
        param_ranges[p] = (lo, hi if hi > lo else lo + 1)
    for i, t in enumerate(top_trials):
        normed = []
        for p in param_names:
            lo, hi = param_ranges[p]
            normed.append((t.params.get(p, lo) - lo) / (hi - lo))
        alpha = 1.0 - 0.06 * i
        ax.plot(range(len(param_names)), normed, "o-", ms=3,
                alpha=max(alpha, 0.3), color=plt.cm.viridis(i / 10))
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels([p.replace("_", "\n") for p in param_names],
                       fontsize=6, rotation=0)
    ax.set_ylabel("Normalized value")
    ax.set_title("Top-10 Trials (parallel coordinates)")

    # Panel 3: Hyperparameter importances
    ax = axes[1, 0]
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        vals = list(importances.values())
        ax.barh(names[::-1], vals[::-1], color="#3498db")
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importances")
    except Exception:
        ax.text(0.5, 0.5, "Not enough trials\nfor importance",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Hyperparameter Importances")

    # Panel 4: Best params summary table
    ax = axes[1, 1]
    ax.axis("off")
    best = study.best_params
    rows = [[k, f"{v:.4g}" if isinstance(v, float) else str(v)]
            for k, v in best.items()]
    table = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Best Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title(f"Best Trial #{study.best_trial.number} — RMSE={study.best_value:.4f}")

    fig.suptitle("Optuna Hyperparameter Optimization (TPE Sampler)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def train_severity_model(
    features: np.ndarray,
    severity_targets: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    random_state: int = 42,
    test_size: float = 0.2,
    use_smote: bool = True,
    **xgb_kwargs,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost regressor on segmentation features -> severity.

    When *use_smote* is ``True``, high-severity defects (>0.6) are
    oversampled using SMOTE (Chawla et al. 2002) so the model learns
    the critical tail of the severity distribution.

    Returns (fitted model, metrics dict).
    """
    # XGBoost silently coerces object-dtype arrays; enforce numeric input so
    # any future categorical feature must be encoded explicitly upstream.
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(
            f"train_severity_model expects a numeric feature matrix, "
            f"got dtype={features.dtype!r}. Add an OrdinalEncoder / "
            f"OneHotEncoder in the feature pipeline before calling."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        severity_targets,
        test_size=test_size,
        random_state=random_state,
    )

    # Apply SMOTE on training set only (never on test set)
    if use_smote:
        X_train, y_train = _apply_smote(
            X_train, y_train, random_state=random_state,
        )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
        objective="reg:squarederror",
        **xgb_kwargs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    importance_map: dict[str, float] = {}
    importances = model.feature_importances_
    for idx, imp in enumerate(importances):
        importance_map[f"f{idx}"] = float(imp)

    metrics: dict = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "feature_importances": importance_map,
        "smote_applied": use_smote,
        "train_size_after_smote": len(y_train),
    }

    return model, metrics


def predict_severity(
    model: xgb.XGBRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict severity scores, clipped to [0, 1]."""
    predictions = model.predict(features)
    return np.clip(predictions, 0.0, 1.0)


def compute_shap_explanations(
    model: xgb.XGBRegressor,
    features: np.ndarray,
    feature_names: list[str],
    n_background: int = 500,
    output_dir: str | None = None,
) -> tuple[np.ndarray, plt.Figure, plt.Figure, plt.Figure, plt.Figure | None]:
    """Compute SHAP values, dependence plots, and interaction values.

    Uses TreeExplainer (Lundberg et al. 2020) for exact Shapley values
    in O(TLD²) time, plus interaction values Φᵢⱼ for feature pairs.

    Returns:
        shap_values: array of shape (n_samples, n_features)
        fig_beeswarm: SHAP beeswarm summary plot
        fig_bar: SHAP mean absolute bar plot
        fig_dep: SHAP dependence plots for top-3 features
        fig_interact: SHAP interaction heatmap (or None if failed)
    If output_dir is provided, saves figures there.
    """
    # Use a background subset if the dataset is larger than n_background
    if features.shape[0] > n_background:
        indices = np.random.default_rng(42).choice(
            features.shape[0], size=n_background, replace=False
        )
        background = features[indices]
    else:
        background = features

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(features)

    # --- Beeswarm summary plot ---
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
    plt.sca(ax_beeswarm)
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        show=False,
    )
    fig_beeswarm = plt.gcf()
    fig_beeswarm.tight_layout()

    # --- Bar plot (mean |SHAP|) ---
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    plt.sca(ax_bar)
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    fig_bar = plt.gcf()
    fig_bar.tight_layout()

    # --- Dependence plots for top features (up to 3) ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    n_dep = min(3, len(feature_names))
    top_dep_idx = np.argsort(mean_abs)[-n_dep:][::-1]

    fig_dep, dep_axes = plt.subplots(1, n_dep, figsize=(11.69, 4.5))
    if n_dep == 1:
        dep_axes = [dep_axes]
    for ax_i, feat_idx in enumerate(top_dep_idx):
        ax = dep_axes[ax_i]
        # Find strongest interaction partner for coloring
        if len(feature_names) > 1:
            corr_row = np.abs(
                np.corrcoef(shap_values[:, feat_idx], features.T)[0, 1:]
            )
            # Exclude self from candidates
            corr_row[feat_idx] = -1
            other_idx = int(np.argmax(corr_row))
        else:
            other_idx = feat_idx
        scatter = ax.scatter(
            features[:, feat_idx], shap_values[:, feat_idx],
            c=features[:, other_idx], cmap="coolwarm",
            s=12, alpha=0.7, edgecolors="none",
        )
        ax.set_xlabel(feature_names[feat_idx])
        ax.set_ylabel("SHAP value")
        ax.set_title(f"{feature_names[feat_idx]}")
        cb = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cb.set_label(feature_names[other_idx], fontsize=7)
    n_label = min(n_dep, 3)
    fig_dep.suptitle(f"SHAP Dependence Plots (top-{n_label} features)", fontsize=12)
    fig_dep.tight_layout(rect=[0, 0, 1, 0.94])

    # --- Interaction heatmap ---
    fig_interact = None
    try:
        explainer_interact = shap.TreeExplainer(model)
        interaction_values = explainer_interact.shap_interaction_values(background)
        # interaction_values shape: (n_samples, n_features, n_features)
        mean_interact = np.abs(interaction_values).mean(axis=0)
        n_show = min(10, mean_interact.shape[0])
        # Rank by total interaction strength (off-diagonal row sums), not by
        # marginal SHAP — the heatmap is about pairwise interactions.
        interact_strength = mean_interact.sum(axis=1) - np.diag(mean_interact)
        top_interact_idx = np.argsort(interact_strength)[-n_show:][::-1]
        sub_matrix = mean_interact[np.ix_(top_interact_idx, top_interact_idx)]
        sub_names = [feature_names[i] for i in top_interact_idx]

        fig_interact, ax_int = plt.subplots(figsize=(11.69, 8.27))
        im = ax_int.imshow(sub_matrix, cmap="YlOrRd", aspect="auto")
        ax_int.set_xticks(range(n_show))
        ax_int.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
        ax_int.set_yticks(range(n_show))
        ax_int.set_yticklabels(sub_names, fontsize=8)
        # Annotate cells
        for i in range(n_show):
            for j in range(n_show):
                val = sub_matrix[i, j]
                color = "white" if val > sub_matrix.max() * 0.6 else "black"
                ax_int.text(j, i, f"{val:.3f}", ha="center", va="center",
                           fontsize=6, color=color)
        plt.colorbar(im, ax=ax_int, shrink=0.8, label="Mean |SHAP interaction|")
        ax_int.set_title("SHAP Interaction Values (Lundberg et al. 2020)")
        fig_interact.tight_layout()
    except Exception as e:
        warnings.warn(f"SHAP interaction computation failed: {e}")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_beeswarm.savefig(out / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        fig_bar.savefig(out / "shap_bar.png", dpi=150, bbox_inches="tight")
        fig_dep.savefig(out / "shap_dependence.png", dpi=150, bbox_inches="tight")
        if fig_interact is not None:
            fig_interact.savefig(out / "shap_interactions.png", dpi=150, bbox_inches="tight")

    return shap_values, fig_beeswarm, fig_bar, fig_dep, fig_interact


def save_severity_model(model: xgb.XGBRegressor, path: str) -> None:
    """Persist the trained XGBoost model to disk via joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_severity_model(path: str) -> xgb.XGBRegressor:
    """Load a previously saved XGBoost model from disk via joblib."""
    return joblib.load(path)

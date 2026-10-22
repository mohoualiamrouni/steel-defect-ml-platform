"""Configuration dataclass for the datamining & cutting plan pipeline.

Centralises every tuneable parameter: coil generation, XGBoost severity
model, quality profile, change-point detection, and 2D guillotine cutting
optimisation.  Supports YAML persistence with a JSON fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class DataminingConfig:
    """Pipeline-wide configuration for the datamining module."""

    # -- Data paths -----------------------------------------------------------
    synthetic_output_dir: str = "datamining/data/synthetic"
    output_dir: str = "datamining/outputs"

    # -- Coil generation ------------------------------------------------------
    n_coils: int = 50
    coil_length_range: Tuple[float, float] = (500.0, 1500.0)
    coil_width_range: Tuple[float, float] = (1.0, 2.0)
    defects_per_coil_range: Tuple[int, int] = (20, 200)
    mask_size: int = 64

    # -- Severity model (XGBoost) ---------------------------------------------
    # NOTE: these four xgb_* values are FALLBACK defaults, used only when
    # ``use_optuna=False``.  In the nominal pipeline (use_optuna=True) the TPE
    # study in ``optuna_tune`` (50 trials, 3-fold RMSE) OVERWRITES them with the
    # tuned hyper-parameters reported in the thesis (≈131 trees, depth 4,
    # learning_rate≈0.011).  Do not mistake these defaults for the tuned
    # model's hyper-parameters — see report tab:xgb_hparams.
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    shap_n_samples: int = 500

    # -- Quality profile ------------------------------------------------------
    quality_grid_shape: Tuple[int, int] = (20, 5)
    kernel_bandwidth: float = 2.0

    # -- Change-point detection (ruptures) ------------------------------------
    changepoint_penalty: float = 1.0
    changepoint_model: str = "rbf"

    # -- Cutting optimisation -------------------------------------------------
    min_piece_length: float = 5.0
    min_piece_width: float = 0.3
    n_quality_grades: int = 4
    quality_thresholds: Tuple[float, ...] = (0.20, 0.40, 0.60)
    grade_values: Tuple[float, ...] = (1.0, 0.7, 0.4, 0.05)
    grade_names: Tuple[str, ...] = ("A", "B", "C", "scrap")

    # -- Feature selection (Boruta, Kursa & Rudnicki 2010) --------------------
    use_boruta: bool = True

    # -- Hyperparameter tuning (Optuna, Akiba et al. 2019) ------------------
    use_optuna: bool = True
    optuna_n_trials: int = 50

    # -- Conformalised Quantile Regression (Romano, Patterson, Candès 2019) ---
    # Adds severity_lo / severity_hi columns with finite-sample-valid coverage
    # at level (1 - cqr_alpha).  Enabled by default: the calibration split is
    # coil-disjoint (GroupShuffleSplit on coil_id) so the reported coverage is
    # honest.  ``grade_by`` selects how intervals feed the cutting decision:
    #   "mean"     — grade on the point severity (baseline);
    #   "upper"    — grade on severity_hi (risk-averse, downgrade-safe);
    #   "interval" — grade on severity_hi AND flag pieces whose [lo,hi] straddles
    #                a grade threshold for human re-inspection (Phase-2 mode).
    use_cqr: bool = True
    cqr_alpha: float = 0.1
    grade_by: str = "mean"   # "mean" | "upper" | "interval"

    # -- General --------------------------------------------------------------
    random_state: int = 42

    # -- Serialisation --------------------------------------------------------

    _TUPLE_FIELDS = (
        "coil_length_range",
        "coil_width_range",
        "defects_per_coil_range",
        "quality_grid_shape",
        "quality_thresholds",
        "grade_values",
        "grade_names",
    )

    def _as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    @classmethod
    def from_yaml(cls, path: str) -> "DataminingConfig":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML required for .yaml configs.")
            data: Dict[str, Any] = yaml.safe_load(raw) or {}
        elif ext == ".json":
            data = json.loads(raw)
        else:
            data = (yaml.safe_load(raw) or {}) if _HAS_YAML else json.loads(raw)

        for field_name in cls._TUPLE_FIELDS:
            if field_name in data and isinstance(data[field_name], list):
                data[field_name] = tuple(data[field_name])

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self._as_dict()
        ext = os.path.splitext(path)[1].lower()

        if _HAS_YAML and ext in (".yaml", ".yml", ""):
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh, default_flow_style=False,
                          sort_keys=False, allow_unicode=True)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for key, value in self._as_dict().items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)

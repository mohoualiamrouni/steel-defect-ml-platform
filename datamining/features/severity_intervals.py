r"""Conformalized Quantile Regression (CQR) wrapper for the severity model.

Reference: Romano, Patterson, Candès (2019),
    "Conformalized Quantile Regression", arXiv:1905.03222.

This module wraps the existing XGBoost severity regressor with two
auxiliary quantile heads at $\alpha/2$ and $1-\alpha/2$, then applies a
conformal calibration step on a held-out calibration split so that the
output prediction intervals have **finite-sample valid coverage** at
level $1-\alpha$ -- i.e. for any new $(x,y)$ from the same distribution::

    P(\hat{q}_\text{lo}(x) \le y \le \hat{q}_\text{hi}(x)) \ge 1 - \alpha.

Validity holds for any base regressor and any data distribution, with
no assumption beyond exchangeability of calibration and test points.

Public API
----------
    ConformalSeverity(alpha=0.1)
        .fit(X_train, y_train, X_cal, y_cal)
        .predict_interval(X)         -> (lo, hi)
        .predict_mean(X)             -> point estimate from the median head
        .empirical_coverage(X, y)    -> measured coverage on a held-out fold

The base point estimator is unchanged -- existing
:func:`severity_model.train_severity_model` callers see no behavioural
difference; :mod:`pipeline.run_pipeline` adds the intervals when the ``use_cqr``
config flag is on, which it is by default.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xgboost as xgb


class ConformalSeverity:
    """CQR wrapper for the severity regressor.

    Parameters
    ----------
    alpha : float, default 0.1
        Miscoverage rate.  ``alpha=0.1`` -> nominal $1-\\alpha=0.9$ coverage.
    base_params : dict, optional
        XGBoost hyperparameters for the two quantile regressors.  If
        ``None``, sensible defaults are used (depth 4, lr 0.05,
        n_estimators 200, ``min_child_weight=5`` to avoid degenerate
        leaves at the extreme quantiles).
    random_state : int
        Seed for the underlying XGBoost calls.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        base_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.base_params = base_params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
        self.q_lo: Optional[xgb.XGBRegressor] = None
        self.q_hi: Optional[xgb.XGBRegressor] = None
        self.q_med: Optional[xgb.XGBRegressor] = None
        self._conformity_quantile: Optional[float] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "ConformalSeverity":
        """Fit the two quantile heads on the train split, then conformalise
        on the calibration split.

        Calibration step computes the conformity score
            E_i = max( q_lo(x_i) - y_i, y_i - q_hi(x_i) )
        and sets the empirical $(1-\\alpha)$-quantile (with finite-sample
        correction $\\lceil (n_{cal}+1)(1-\\alpha) \\rceil / n_{cal}$) as
        the inflation $\\hat{Q}$ added to both bounds at predict time.
        """
        a_lo = self.alpha / 2.0
        a_hi = 1.0 - self.alpha / 2.0

        common = dict(
            objective="reg:quantileerror",
            random_state=self.random_state,
            **self.base_params,
        )
        self.q_lo = xgb.XGBRegressor(quantile_alpha=a_lo, **common).fit(X_train, y_train)
        self.q_hi = xgb.XGBRegressor(quantile_alpha=a_hi, **common).fit(X_train, y_train)
        self.q_med = xgb.XGBRegressor(quantile_alpha=0.5, **common).fit(X_train, y_train)

        # Conformity scores on the calibration split.
        lo_cal = self.q_lo.predict(X_cal)
        hi_cal = self.q_hi.predict(X_cal)
        E = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        n_cal = len(E)
        # Finite-sample-corrected quantile level (Romano 2019 §3.1).
        level = np.minimum(1.0, np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal)
        self._conformity_quantile = float(np.quantile(E, level))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) prediction interval of nominal coverage 1 - alpha.

        Both bounds are clipped to [0, 1] because severity is bounded.
        """
        if self.q_lo is None or self._conformity_quantile is None:
            raise RuntimeError("Call .fit(...) before .predict_interval(...).")
        lo = self.q_lo.predict(X) - self._conformity_quantile
        hi = self.q_hi.predict(X) + self._conformity_quantile
        # Severity is bounded in [0, 1] in the simulator and by physical
        # interpretation -- clip to avoid uninformative negative bounds.
        return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Median (50%-quantile) point estimate from the median head."""
        if self.q_med is None:
            raise RuntimeError("Call .fit(...) before .predict_mean(...).")
        return np.clip(self.q_med.predict(X), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def empirical_coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of held-out points that fall inside the predicted interval."""
        lo, hi = self.predict_interval(X)
        return float(np.mean((y >= lo) & (y <= hi)))

    def empirical_width(self, X: np.ndarray) -> float:
        """Mean width of the prediction interval over a sample."""
        lo, hi = self.predict_interval(X)
        return float(np.mean(hi - lo))

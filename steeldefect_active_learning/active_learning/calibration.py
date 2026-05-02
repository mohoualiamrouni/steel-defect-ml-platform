"""
Post-hoc model calibration.

Implements Temperature Scaling (Guo et al. 2017, "On Calibration of Modern
Neural Networks") together with standard calibration metrics.

Key insight from Guo et al.: modern neural networks are overconfident —
their predicted probabilities do not match empirical accuracy. Temperature
scaling fits a single scalar T* on a held-out validation set that brings
ECE close to zero without affecting accuracy.

Temperature Scaling
-------------------
    p̂(y|x) = softmax(f(x) / T*)
    T* = argmin_T NLL(softmax(logits / T), labels_val)

    Solved via scipy L-BFGS-B (convex in T, fast).

ECE — Expected Calibration Error (15 equal-width confidence bins)
-------------------------------------------------------------------
    ECE = Σ_{m=1}^{M} (|B_m| / n) · |acc(B_m) - conf(B_m)|

MCE — Maximum Calibration Error
--------------------------------
    MCE = max_m |acc(B_m) - conf(B_m)|

Reliability Diagram
--------------------
    Bar chart of mean confidence vs. fraction of corrects per bin.
    Perfect calibration = all bars on the diagonal.

References
----------
- Guo et al. 2017: "On Calibration of Modern Neural Networks"
  https://arxiv.org/abs/1706.04599
- Ovadia et al. 2019: "Can you trust your model's uncertainty under dataset
  shift?" — calibration degrades with covariate shift; monitor ECE on OOD data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECE / MCE (standalone, no model required)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al. 2017).

    Args:
        probs:  Predicted probabilities, shape (N, C) or (N,) for binary.
                For multi-class, confidence = max(probs[i]).
        labels: True class indices, shape (N,).
        n_bins: Number of equal-width confidence bins (default 15).

    Returns:
        ECE in [0, 1]. Lower is better. Perfect calibration → 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(predictions[mask] == labels[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (Guo et al. 2017).

    Args:
        probs:  Predicted probabilities, shape (N, C) or (N,).
        labels: True class indices, shape (N,).
        n_bins: Number of equal-width confidence bins.

    Returns:
        MCE in [0, 1]. Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(predictions[mask] == labels[mask]))
        conf = float(np.mean(confidences[mask]))
        mce = max(mce, abs(acc - conf))

    return float(mce)


def _calibration_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_confidences, bin_accuracies, bin_counts) for plotting."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(np.int64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins, dtype=int)

    for m, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences > lo) & (confidences <= hi)
        cnt = int(mask.sum())
        bin_cnt[m] = cnt
        if cnt > 0:
            bin_conf[m] = float(np.mean(confidences[mask]))
            bin_acc[m] = float(np.mean(predictions[mask] == labels[mask]))

    return bin_conf, bin_acc, bin_cnt


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
    n_bins: int = 15,
    title: str = "Reliability Diagram",
) -> None:
    """Save a reliability diagram PNG.

    Shows confidence vs. accuracy per bin. Perfect calibration = diagonal.

    Args:
        probs:       Predicted probabilities, shape (N, C) or (N,).
        labels:      True class indices, shape (N,).
        output_path: Path to save the PNG file.
        n_bins:      Number of confidence bins.
        title:       Plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("reliability_diagram requires matplotlib") from e

    bin_conf, bin_acc, bin_cnt = _calibration_bins(probs, labels, n_bins)
    ece = expected_calibration_error(probs, labels, n_bins)
    mce = maximum_calibration_error(probs, labels, n_bins)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))

    # Gap bars (over-/under-confidence)
    for m in range(n_bins):
        if bin_cnt[m] == 0:
            continue
        gap = bin_acc[m] - bin_conf[m]
        color = "tomato" if gap < 0 else "steelblue"
        # Draw actual accuracy bar
        ax.bar(bin_centers[m], bin_acc[m], width=width * 0.9,
               color="steelblue", alpha=0.7, edgecolor="white")
        # Draw gap bar
        ax.bar(bin_centers[m], gap, width=width * 0.9,
               bottom=bin_conf[m], color=color, alpha=0.3, edgecolor="white")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE={ece:.4f}  MCE={mce:.4f}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved reliability diagram → %s", output_path)


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc calibration via Temperature Scaling (Guo et al. 2017).

    Usage
    -----
    scaler = TemperatureScaler()
    scaler.calibrate(val_logits, val_labels)      # fit T* once
    cal_probs = scaler.scale(test_logits)          # apply at test time
    ece = scaler.ece(cal_probs, test_labels)

    The temperature T* is saved to `temperature_` after fitting.
    A value T > 1 softens the distribution (model was overconfident).
    A value T < 1 sharpens it (model was underconfident — rare).

    Attributes
    ----------
    temperature_ : float
        Fitted temperature. 1.0 before calibration.
    history_ : list[dict]
        NLL history from the optimiser (for diagnostics).
    """

    def __init__(self) -> None:
        self.temperature_: float = 1.0
        self.history_: List[Dict] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        T_init: float = 1.0,
        bounds: Tuple[float, float] = (0.05, 20.0),
    ) -> "TemperatureScaler":
        """Fit T* on a validation set by minimising cross-entropy.

        Args:
            logits: Raw model logits, shape (N, C).  Do NOT pass softmax output.
            labels: True class indices, shape (N,).
            T_init: Initial temperature guess (default 1.0).
            bounds: (T_min, T_max) to constrain search.

        Returns:
            self (for chaining).
        """
        try:
            from scipy.optimize import minimize_scalar
        except ImportError as e:
            raise ImportError("calibrate() requires scipy: pip install scipy") from e

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        from scipy.special import logsumexp as _logsumexp

        def nll(T: float) -> float:
            scaled = logits / T
            # Numerically stable logsumexp — no eps needed; sum of exp's is >= 1
            # once the max is factored out, so log is well-defined.
            log_Z = _logsumexp(scaled, axis=1)
            log_preds = scaled[np.arange(len(labels)), labels] - log_Z
            loss = -float(log_preds.mean())
            self.history_.append({"T": T, "nll": loss})
            return loss

        result = minimize_scalar(nll, bounds=bounds, method="bounded")
        self.temperature_ = float(result.x)
        logger.info(
            "Temperature scaling: T*=%.4f (NLL %.4f → %.4f)",
            self.temperature_, nll(1.0), result.fun,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw model logits, shape (N, C) or (C,) for single sample.

        Returns:
            Calibrated softmax probabilities, same shape as input.
        """
        logits = np.asarray(logits, dtype=np.float64)
        single = logits.ndim == 1
        if single:
            logits = logits[np.newaxis, :]

        scaled = logits / self.temperature_
        # Stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)

        return probs[0] if single else probs

    # ------------------------------------------------------------------
    # Metrics (convenience wrappers)
    # ------------------------------------------------------------------

    def ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """Compute ECE for given probabilities and labels."""
        return expected_calibration_error(probs, labels, n_bins)

    def mce(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """Compute MCE for given probabilities and labels."""
        return maximum_calibration_error(probs, labels, n_bins)

    def reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        output_path: Union[str, Path],
        n_bins: int = 15,
        title: str = "Reliability Diagram",
    ) -> None:
        """Save reliability diagram. Wrapper around module-level function."""
        reliability_diagram(probs, labels, output_path, n_bins, title)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save temperature to JSON."""
        import json
        data = {"temperature": self.temperature_}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved temperature scaler → %s  (T=%.4f)", path, self.temperature_)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemperatureScaler":
        """Load temperature from JSON saved by save()."""
        import json
        with open(str(path)) as f:
            data = json.load(f)
        scaler = cls()
        scaler.temperature_ = float(data["temperature"])
        logger.info("Loaded temperature scaler from %s  (T=%.4f)", path, scaler.temperature_)
        return scaler

    def __repr__(self) -> str:
        return f"TemperatureScaler(T={self.temperature_:.4f})"


# ---------------------------------------------------------------------------
# Online calibration tracker (cycle-over-cycle drift detection)
# ---------------------------------------------------------------------------

class OnlineCalibrationTracker:
    """Per-cycle calibration tracker for the AL loop.

    After each AL cycle, the labeled set grows. This tracker re-fits the
    Temperature Scaler on the *cumulative* labeled set, records ECE/MCE,
    and flags drift when the latest ECE more than doubles the previous
    cycle's ECE — Ovadia et al. (2019) showed that exactly this pattern is
    the canonical signature of dataset shift.

    Closes report limitation L3: distribution shift was previously untested.

    Usage
    -----
    tracker = OnlineCalibrationTracker(drift_factor=2.0)
    for cycle in range(n_cycles):
        # ... grow labeled set ...
        tracker.update(cycle, val_logits, val_labels)
        if tracker.drift_detected():
            logger.warning("ECE doubled — possible distribution shift.")
    """

    def __init__(
        self,
        drift_factor: float = 2.0,
        n_bins: int = 15,
    ) -> None:
        """Args:
            drift_factor: Trigger drift flag when latest ECE / previous ECE
                          exceeds this multiplier. Default 2.0 (doubled).
            n_bins:       Calibration bin count for ECE/MCE.
        """
        if drift_factor <= 1.0:
            raise ValueError("drift_factor must be > 1.0")
        self.drift_factor = float(drift_factor)
        self.n_bins = int(n_bins)
        self.history_: List[Dict] = []
        self._scalers: List[TemperatureScaler] = []

    def update(
        self,
        cycle: int,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Re-fit T* on the supplied (cumulative) validation set; record metrics.

        Args:
            cycle:  Cycle index (used for sorting / plotting).
            logits: Raw logits, shape (N, C). Should be from the cumulative
                    labeled set, *not* a fresh held-out batch.
            labels: True class indices, shape (N,).

        Returns:
            The history record just appended (also accessible via .history_).
        """
        scaler = TemperatureScaler().calibrate(logits, labels)
        self._scalers.append(scaler)
        cal_probs = scaler.scale(logits)
        ece = scaler.ece(cal_probs, labels, n_bins=self.n_bins)
        mce = scaler.mce(cal_probs, labels, n_bins=self.n_bins)
        # Pre-calibration ECE so we can also report the *gain* per cycle.
        from scipy.special import softmax as _softmax  # local import; lightweight
        raw_probs = _softmax(np.asarray(logits, dtype=np.float64), axis=1)
        ece_pre = expected_calibration_error(raw_probs, labels, self.n_bins)

        rec = {
            "cycle":      int(cycle),
            "n_labeled":  int(len(labels)),
            "T_star":     float(scaler.temperature_),
            "ece_pre":    float(ece_pre),
            "ece_post":   float(ece),
            "mce_post":   float(mce),
            "drift":      False,
        }
        if self.history_:
            prev = self.history_[-1]["ece_post"]
            if prev > 0 and ece / max(prev, 1e-12) > self.drift_factor:
                rec["drift"] = True
        self.history_.append(rec)
        logger.info(
            "online-calibration cycle=%d T*=%.3f ECE %.4f→%.4f drift=%s",
            cycle, scaler.temperature_, ece_pre, ece, rec["drift"],
        )
        return rec

    def drift_detected(self) -> bool:
        """True iff the most recent cycle flagged drift."""
        return bool(self.history_ and self.history_[-1]["drift"])

    def latest_scaler(self) -> Optional[TemperatureScaler]:
        return self._scalers[-1] if self._scalers else None

    def to_dataframe(self) -> "pd.DataFrame":  # noqa: F821
        """Return the cycle history as a pandas DataFrame for plotting."""
        import pandas as pd
        return pd.DataFrame(self.history_)


# ---------------------------------------------------------------------------
# Calibration under shift (Ovadia et al. 2019)
# ---------------------------------------------------------------------------

def calibration_under_shift_report(
    probs: np.ndarray,
    labels: np.ndarray,
    n_quartiles: int = 4,
    n_bins: int = 15,
) -> List[Dict]:
    """Analyse calibration across uncertainty quartiles (Ovadia et al. 2019).

    Ovadia et al. found that calibration degrades under dataset shift;
    high-uncertainty samples show the worst ECE. This function groups
    predictions by their entropy quartile and computes ECE per group,
    revealing where the model loses calibration.

    Args:
        probs:       Predicted probabilities, shape (N, C).
        labels:      True class indices, shape (N,).
        n_quartiles: Number of entropy groups (default 4).
        n_bins:      Calibration bins per group.

    Returns:
        List of dicts with keys: quartile, entropy_range, ece, mce,
        accuracy, mean_confidence, n_samples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    # Predictive entropy
    p_clipped = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)  # (N,)

    q_edges = np.quantile(entropy, np.linspace(0, 1, n_quartiles + 1))
    results = []

    for q in range(n_quartiles):
        lo, hi = q_edges[q], q_edges[q + 1]
        if q == n_quartiles - 1:
            mask = (entropy >= lo) & (entropy <= hi)
        else:
            mask = (entropy >= lo) & (entropy < hi)

        if mask.sum() < 2:
            continue

        g_probs = probs[mask]
        g_labels = labels[mask]
        ece = expected_calibration_error(g_probs, g_labels, n_bins)
        mce = maximum_calibration_error(g_probs, g_labels, n_bins)
        acc = float(np.mean(g_probs.argmax(axis=1) == g_labels))
        conf = float(np.mean(g_probs.max(axis=1)))

        results.append({
            "quartile": q + 1,
            "entropy_range": (float(lo), float(hi)),
            "ece": ece,
            "mce": mce,
            "accuracy": acc,
            "mean_confidence": conf,
            "n_samples": int(mask.sum()),
        })

    return results

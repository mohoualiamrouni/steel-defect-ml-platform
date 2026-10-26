"""
Pipeline-aware Active Learning — Roth & Small (2008) Algorithm 1
=================================================================

Reference: "Active Learning for Pipeline Models", D. Roth & K. Small,
AAAI 2008.  Algorithm 1 adapted for steel surface inspection.

Core idea
---------
When the inspection pipeline has D sequential model stages (e.g.
  Stage 1: defect *detection*  — is there a defect? (binary)
  Stage 2: defect *classification* — which defect type?
  Stage 3: severity / pass-fail decision
), errors in early stages cascade into later ones.  Naively splitting
the annotation budget equally wastes it: if the detector is still bad,
giving labels to the classifier is pointless.

Algorithm 1 (exact implementation)
------------------------------------
    Initialise: β = [1, 0, ..., 0]
    Each cycle t:
        U(d)_t  = mean_{x∈S_u} certainty(d)(x)   # certainty = 1 - H_norm
        ΔU_t    = U_t − U_{t-1}                   # per stage
        Δ_t     = ΔU_t / ‖ΔU_t‖                  # normalize (line 16)
        β_t     = β_{t-1} + λ · Δ_t               # update   (line 18)
        β_t     = β_t / ‖β_t‖₁                    # L1-norm  (line 19)

Sign convention fix:
    The paper uses query scoring q(d) where higher = MORE CERTAIN.
    entropy-based uncertainty is HIGHER for UNCERTAIN images.
    We convert: certainty(d)(x) = 1 − H_norm(d)(x)
    so that certainty INCREASES as the model improves → ΔU > 0.
    When a stage plateaus (ΔU → 0), β for that stage stops growing and
    weight naturally shifts toward later stages.

Stopping criterion:
    Beta is considered converged when max |β_t − β_{t-1}| < tol for
    `patience` consecutive cycles. Call has_converged() to check.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd

from steeldefect_active_learning.active_learning.uncertainty import compute_uncertainty


class PipelineActiveLearner:
    """Adaptive multi-stage active learner implementing Roth & Small 2008.

    Args:
        models:              List of D PyTorch models, one per pipeline stage.
        tasks:               List of D task strings ('classification'/'segmentation').
        uncertainty_methods: List of D method strings ('entropy','bald', etc.).
        update_rate:         λ — controls how fast β shifts between stages (0.05–0.3).
        normalize_scores:    If True (default), entropy divided by log(C) → [0,1].
        convergence_tol:     β is converged when max change < this value.
        convergence_patience: Number of consecutive converged cycles before stopping.
    """

    def __init__(
        self,
        models: List[nn.Module],
        tasks: Optional[List[str]] = None,
        uncertainty_methods: Optional[List[str]] = None,
        update_rate: float = 0.1,
        normalize_scores: bool = True,
        convergence_tol: float = 0.01,
        convergence_patience: int = 3,
    ):
        if len(models) == 0:
            raise ValueError("At least one model required.")

        self.D = len(models)
        self.models = models
        self.tasks = tasks or ["classification"] * self.D
        self.uncertainty_methods = uncertainty_methods or ["entropy"] * self.D
        self.lam = update_rate
        self.normalize_scores = normalize_scores
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience

        # β — mixing weights for each stage (Algorithm 1 lines 5-6)
        self.beta = np.zeros(self.D, dtype=np.float64)
        self.beta[0] = 1.0   # start: all weight on stage 1

        # U^(d)_{t-1} — previous average CERTAINTY per stage (not uncertainty!)
        self._prev_U: Optional[np.ndarray] = None

        # History for diagnostics / plotting / PDF report
        self.beta_history:  List[np.ndarray] = [self.beta.copy()]
        self.U_history:     List[np.ndarray] = []    # certainty per stage per cycle
        self.delta_history: List[np.ndarray] = []    # ΔU per cycle (normalized)
        self.score_history: List[Dict[str, float]] = []  # combined scores per cycle

        # Convergence tracking
        self._converged_streak: int = 0

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    @torch.no_grad()
    def compute_stage_scores(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        """Compute per-image uncertainty scores for every pipeline stage.

        Returns:
            Dict {path: ndarray(D,)} — uncertainty score per stage.
        """
        stage_uncertainty: List[Dict[str, float]] = []
        for model, task, method in zip(self.models, self.tasks, self.uncertainty_methods):
            scores_d = compute_uncertainty(
                model=model,
                dataloader=dataloader,
                method=method,
                device=device,
                task=task,
                normalize=self.normalize_scores,
            )
            stage_uncertainty.append(scores_d)

        all_paths = list(stage_uncertainty[0].keys())
        combined: Dict[str, np.ndarray] = {}
        for path in all_paths:
            combined[path] = np.array(
                [stage_uncertainty[d].get(path, 0.0) for d in range(self.D)],
                dtype=np.float64,
            )
        return combined

    def select_candidates(
        self,
        dataloader: DataLoader,
        budget: int,
        device: str = "cpu",
    ) -> List[str]:
        """Select top-`budget` images for labeling using the current β.

        Combined score (Algorithm 1 line 9):
            combined(x) = Σ_d β^(d) * uncertainty^(d)(x)
        Images with the highest weighted uncertainty are selected.

        Returns:
            Ordered list of image paths (most informative first).
        """
        stage_scores = self.compute_stage_scores(dataloader, device)
        combined = {path: float(self.beta @ scores)
                    for path, scores in stage_scores.items()}

        # Log combined score distribution for this selection
        scores_arr = np.array(list(combined.values()))
        self.score_history.append({
            "mean":   float(scores_arr.mean()),
            "std":    float(scores_arr.std()),
            "max":    float(scores_arr.max()),
            "min":    float(scores_arr.min()),
            "p75":    float(np.percentile(scores_arr, 75)),
            "p90":    float(np.percentile(scores_arr, 90)),
        })

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        return [path for path, _ in ranked[:budget]]

    def step(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> np.ndarray:
        """Update β after one labeling cycle (Algorithm 1 lines 12-20).

        Call AFTER retraining models on newly labeled data.

        Returns:
            Updated β array of shape (D,).
        """
        stage_scores = self.compute_stage_scores(dataloader, device)

        # Average CERTAINTY per stage: U^(d)_t = mean(1 - uncertainty)
        U_t = np.zeros(self.D, dtype=np.float64)
        for scores in stage_scores.values():
            U_t += 1.0 - scores   # certainty = 1 − H_norm
        n = len(stage_scores)
        if n > 0:
            U_t /= n
        self.U_history.append(U_t.copy())

        if self._prev_U is None:
            self._prev_U = U_t.copy()
            return self.beta.copy()

        # ΔU^(d) = U^(d)_t − U^(d)_{t-1}  (Algorithm 1 line 15)
        delta_U = U_t - self._prev_U

        # Normalize Δ_t = ΔU / ‖ΔU‖  (line 16)
        norm = np.linalg.norm(delta_U)
        delta_norm = (delta_U / norm) if norm > 1e-10 else np.zeros(self.D)
        self.delta_history.append(delta_norm.copy())

        # β^(d)_t = β^(d)_{t-1} + λ · Δ^(d)_t  (line 18)
        new_beta = self.beta + self.lam * delta_norm
        new_beta = np.clip(new_beta, 0.0, None)

        # L1-normalize β so weights sum to 1  (line 19)
        beta_sum = new_beta.sum()
        if beta_sum > 1e-10:
            new_beta = new_beta / beta_sum
        else:
            new_beta = np.zeros(self.D)
            new_beta[0] = 1.0

        # Track convergence
        max_change = float(np.abs(new_beta - self.beta).max())
        if max_change < self.convergence_tol:
            self._converged_streak += 1
        else:
            self._converged_streak = 0

        self.beta = new_beta
        self._prev_U = U_t.copy()
        self.beta_history.append(self.beta.copy())
        return self.beta.copy()

    # ----------------------------------------------------------------
    # Stopping criterion
    # ----------------------------------------------------------------

    def has_converged(self) -> bool:
        """Return True if β has been stable for `convergence_patience` cycles.

        Convergence = max |β_t − β_{t-1}| < convergence_tol for
        `convergence_patience` consecutive cycles.
        """
        return self._converged_streak >= self.convergence_patience

    def cycles_run(self) -> int:
        """Number of step() calls completed so far, one per labeling cycle.

        The first call only records the baseline certainty U, so β has been
        updated one time fewer.
        """
        return len(self.U_history)

    # ----------------------------------------------------------------
    # Export / serialization
    # ----------------------------------------------------------------

    def to_dataframe(self, stage_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Export full history as a tidy DataFrame.

        Returns:
            DataFrame with columns:
                cycle, stage, stage_label, beta, certainty_U,
                delta_normalized, combined_score_mean, combined_score_p90
        """
        labels = stage_labels or _stage_labels(self.D)
        beta_arr = np.array(self.beta_history)   # (T+1, D)

        rows = []
        n_cycles = len(self.beta_history)
        for t in range(n_cycles):
            for d in range(self.D):
                row: Dict = {
                    "cycle":       t,
                    "stage":       d,
                    "stage_label": labels[d],
                    "beta":        float(beta_arr[t, d]),
                }
                if t > 0 and t - 1 < len(self.U_history):
                    row["certainty_U"] = float(self.U_history[t - 1][d])
                if t > 0 and t - 1 < len(self.delta_history):
                    row["delta_normalized"] = float(self.delta_history[t - 1][d])
                if t - 1 < len(self.score_history) and t > 0:
                    row["combined_score_mean"] = self.score_history[t - 1]["mean"]
                    row["combined_score_p90"]  = self.score_history[t - 1]["p90"]
                rows.append(row)

        return pd.DataFrame(rows)

    def save_history(self, output_path: str, stage_labels: Optional[List[str]] = None) -> None:
        """Save full history to CSV and a JSON summary."""
        df = self.to_dataframe(stage_labels)
        df.to_csv(output_path, index=False)

        # Also save a compact JSON summary
        json_path = output_path.replace(".csv", "_summary.json")
        summary = {
            "cycles_run":        self.cycles_run(),
            "has_converged":     self.has_converged(),
            "final_beta":        self.beta.tolist(),
            "dominant_stage":    int(np.argmax(self.beta)),
            "dominant_label":    (stage_labels or _stage_labels(self.D))[int(np.argmax(self.beta))],
            "convergence_tol":   self.convergence_tol,
            "convergence_streak": self._converged_streak,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def summary(self) -> str:
        labels = _stage_labels(self.D)
        lines = [
            f"PipelineActiveLearner  (D={self.D} stages, λ={self.lam})",
            f"  Beta (current)   : {np.round(self.beta, 4)}",
            f"  Dominant stage   : {int(np.argmax(self.beta))} ({labels[int(np.argmax(self.beta))]})",
            f"  Cycles run       : {self.cycles_run()}",
            f"  Converged        : {self.has_converged()} "
            f"(streak={self._converged_streak}/{self.convergence_patience})",
        ]
        if self.U_history:
            lines.append(f"  Last avg U       : {np.round(self.U_history[-1], 4)}")
        if self.score_history:
            last = self.score_history[-1]
            lines.append(f"  Last score dist  : mean={last['mean']:.4f}  "
                         f"p90={last['p90']:.4f}  max={last['max']:.4f}")
        return "\n".join(lines)

    def plot_beta_history(
        self,
        stage_labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Plot β evolution, certainty U(d), combined score distribution.

        Three subplots:
          1. β weight per stage over cycles
          2. Average certainty U(d) per stage (tracks model improvement)
          3. Combined score distribution (mean ± std band + p90 line)
        """
        import matplotlib
        matplotlib.use("Agg" if output_path else "TkAgg")
        import matplotlib.pyplot as plt

        beta_arr = np.array(self.beta_history)   # (T+1, D)
        labels = stage_labels or _stage_labels(self.D)
        colors = plt.cm.tab10(np.linspace(0, 1, self.D))

        has_U      = len(self.U_history) > 0
        has_scores = len(self.score_history) > 0
        n_rows = 1 + int(has_U) + int(has_scores)

        fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        # ── Subplot 1: β weights ──────────────────────────────────────
        ax0 = axes[0]
        cycles = np.arange(len(beta_arr))
        for d in range(self.D):
            ax0.plot(cycles, beta_arr[:, d], marker="o", markersize=4,
                     label=labels[d], color=colors[d], lw=1.8)
        ax0.set_ylabel("β weight  (L1-normalised, sums to 1)")
        ax0.set_title(
            "Roth & Small (2008) — Stage Weights β\n"
            "Weight shifts from early to later stages as each plateaus",
            fontsize=9,
        )
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_ylim(-0.05, 1.1)
        ax0.grid(alpha=0.3)

        # Annotate convergence if reached
        if self.has_converged() and len(self.delta_history) > 0:
            first_conv = max(0, len(beta_arr) - self._converged_streak - 1)
            ax0.axvline(first_conv, color="#F44336", linestyle=":",
                        alpha=0.7, label="convergence")

        row = 1

        # ── Subplot 2: Certainty U(d) ─────────────────────────────────
        if has_U:
            U_arr = np.array(self.U_history)   # (T, D)
            ax1 = axes[row]; row += 1
            for d in range(self.D):
                ax1.plot(np.arange(1, len(U_arr) + 1), U_arr[:, d],
                         marker="s", markersize=4, linestyle="--",
                         label=labels[d], color=colors[d], lw=1.5)
            ax1.set_ylabel("Avg certainty U(d) = mean(1 − H_norm)")
            ax1.set_title(
                "Per-Stage Certainty  U(d)\n"
                "Increases as model improves; drives β update direction",
                fontsize=9,
            )
            ax1.legend(loc="lower right", fontsize=8)
            ax1.set_ylim(0, 1.05)
            ax1.grid(alpha=0.3)

        # ── Subplot 3: Combined score distribution ────────────────────
        if has_scores:
            ax2 = axes[row]
            cycles_s = np.arange(1, len(self.score_history) + 1)
            means = np.array([s["mean"] for s in self.score_history])
            stds  = np.array([s["std"]  for s in self.score_history])
            p90s  = np.array([s["p90"]  for s in self.score_history])

            ax2.plot(cycles_s, means, color="#2196F3", lw=1.8, label="Mean score")
            ax2.fill_between(cycles_s, means - stds, means + stds,
                             alpha=0.2, color="#2196F3", label="±1 std")
            ax2.plot(cycles_s, p90s, color="#FF9800", linestyle="--",
                     lw=1.5, label="p90 score")
            ax2.set_xlabel("Labeling cycle")
            ax2.set_ylabel("Combined uncertainty score")
            ax2.set_title(
                "Combined Score Distribution  Σ_d β^(d)·uncertainty^(d)\n"
                "Rising mean = more uncertain images remain; falling = pool depleting",
                fontsize=9,
            )
            ax2.legend(fontsize=8)
            ax2.set_ylim(0, 1.05)
            ax2.grid(alpha=0.3)

        fig.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def generate_report(
        self,
        output_path: str,
        stage_labels: Optional[List[str]] = None,
    ) -> str:
        """Generate a multi-page PDF report of the active learning run.

        Pages:
            1. Cover + summary table
            2. β evolution + U(d) certainty curves
            3. Combined score distribution over cycles
            4. ΔU (normalized) per cycle — shows which stages are still improving

        Args:
            output_path:  Path to save the PDF.
            stage_labels: Display names for pipeline stages.

        Returns:
            Absolute path to the saved PDF.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        labels = stage_labels or _stage_labels(self.D)
        colors = plt.cm.tab10(np.linspace(0, 1, self.D))

        beta_arr = np.array(self.beta_history)   # (T+1, D)
        T = self.cycles_run()

        DARK  = "#212121"
        BLUE  = "#1565C0"
        GREY  = "#546E7A"
        GREEN = "#2E7D32"
        RED   = "#C62828"

        def _footer(fig):
            fig.text(0.5, 0.008,
                     f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                     "SteelDefect Pipeline Active Learning — Roth & Small (2008)",
                     ha="center", va="bottom", fontsize=7, color=GREY)

        with PdfPages(output_path) as pdf:

            # ── Page 1: Cover ──────────────────────────────────────────
            fig = plt.figure(figsize=(8.27, 11.69))
            _footer(fig)
            fig.text(0.5, 0.90, "Pipeline Active Learning Report",
                     ha="center", fontsize=20, fontweight="bold", color=DARK)
            fig.text(0.5, 0.855, "Roth & Small (2008) — Algorithm 1",
                     ha="center", fontsize=13, color=BLUE)
            fig.text(0.5, 0.825,
                     f"D={self.D} stages  ·  λ={self.lam}  ·  "
                     f"{T} cycles  ·  Converged={self.has_converged()}",
                     ha="center", fontsize=10, color=GREY)
            fig.add_artist(plt.Line2D([0.08, 0.92], [0.80, 0.80],
                                       transform=fig.transFigure, color=GREY, lw=0.8))

            # Metric cards
            dominant = int(np.argmax(self.beta))
            cards = [
                ("Cycles Run",      str(T),                          BLUE),
                ("Dominant Stage",  f"{dominant}: {labels[dominant]}", GREEN),
                ("Converged",       "YES" if self.has_converged() else f"NO ({self._converged_streak}/{self.convergence_patience})", GREEN if self.has_converged() else RED),
            ]
            for i, (title, val, color) in enumerate(cards):
                x = 0.08 + i * 0.295
                ax = fig.add_axes([x, 0.70, 0.25, 0.08])
                ax.set_facecolor(color)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)
                ax.text(0.5, 0.70, title, ha="center", fontsize=8,
                        color="white", fontweight="bold", transform=ax.transAxes)
                ax.text(0.5, 0.25, val, ha="center", fontsize=10,
                        color="white", transform=ax.transAxes)

            # Final β table
            ax_t = fig.add_axes([0.10, 0.38, 0.80, 0.28])
            ax_t.axis("off")
            ax_t.text(0.0, 1.0, "Final β weights per stage:", fontsize=9,
                      fontweight="bold", color=DARK, transform=ax_t.transAxes)
            col_x = [0.0, 0.25, 0.55, 0.80]
            for hi, hdr in zip(col_x, ["Stage", "Label", "β (final)", "β (initial)"]):
                ax_t.text(hi, 0.88, hdr, fontsize=8.5, fontweight="bold",
                          color=DARK, transform=ax_t.transAxes)
            # axhline rejects a transform (it builds its own), so draw the rules as
            # plain lines in axes coordinates.
            ax_t.plot([0, 1], [0.84, 0.84], color=GREY, lw=0.5, transform=ax_t.transAxes)
            for d in range(self.D):
                y = 0.78 - d * 0.14
                row_vals = [str(d), labels[d],
                            f"{beta_arr[-1, d]:.4f}", f"{beta_arr[0, d]:.4f}"]
                for xi, val in zip(col_x, row_vals):
                    fw = "bold" if d == dominant else "normal"
                    ax_t.text(xi, y, val, fontsize=8.5, color=BLUE if d == dominant else DARK,
                              fontweight=fw, transform=ax_t.transAxes)
                ax_t.plot([0, 1], [y - 0.05, y - 0.05], color="#E0E0E0", lw=0.3,
                          transform=ax_t.transAxes)

            # Algorithm reference
            fig.text(0.5, 0.20,
                     "Algorithm 1: β(d)_t = β(d)_{t-1} + λ · ΔU(d)_t / ‖ΔU_t‖\n"
                     "             β_t = β_t / ‖β_t‖₁\n"
                     "U(d) = mean certainty = 1 − H_norm(d)  (increases as model improves)",
                     ha="center", fontsize=8, color=GREY, linespacing=2.0, style="italic")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 2: β + U(d) evolution ────────────────────────────
            has_U = len(self.U_history) > 0
            fig, axes = plt.subplots(2 if has_U else 1, 1,
                                      figsize=(8.27, 9 if has_U else 5),
                                      sharex=True)
            _footer(fig)
            if not has_U:
                axes = [axes]
            fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.35)
            fig.suptitle("Stage Weights β and Certainty U(d) Evolution", fontsize=11)

            cycles = np.arange(len(beta_arr))
            for d in range(self.D):
                axes[0].plot(cycles, beta_arr[:, d], marker="o", markersize=4,
                             label=labels[d], color=colors[d], lw=1.8)
            axes[0].set_ylabel("β weight")
            axes[0].set_title("β per stage — weight shifts as stages plateau", fontsize=9)
            axes[0].legend(fontsize=8, loc="upper right")
            axes[0].set_ylim(-0.05, 1.1); axes[0].grid(alpha=0.3)

            if has_U:
                U_arr = np.array(self.U_history)
                for d in range(self.D):
                    axes[1].plot(np.arange(1, len(U_arr) + 1), U_arr[:, d],
                                 marker="s", markersize=4, linestyle="--",
                                 label=labels[d], color=colors[d], lw=1.5)
                axes[1].set_xlabel("Labeling cycle")
                axes[1].set_ylabel("Avg certainty U(d)")
                axes[1].set_title("U(d) = mean(1 − H_norm) — increases as model learns", fontsize=9)
                axes[1].legend(fontsize=8, loc="lower right")
                axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 3: Score distribution + ΔU ──────────────────────
            has_scores = len(self.score_history) > 0
            has_delta  = len(self.delta_history) > 0
            if has_scores or has_delta:
                ncols = 1 + int(has_delta)
                fig, axes = plt.subplots(1, ncols, figsize=(8.27, 4.5))
                _footer(fig)
                if ncols == 1:
                    axes = [axes]
                fig.subplots_adjust(top=0.85, bottom=0.14, wspace=0.38)
                fig.suptitle("Combined Score Distribution and ΔU per Cycle", fontsize=10)

                if has_scores:
                    cycles_s = np.arange(1, len(self.score_history) + 1)
                    means = np.array([s["mean"] for s in self.score_history])
                    stds  = np.array([s["std"]  for s in self.score_history])
                    p90s  = np.array([s["p90"]  for s in self.score_history])
                    axes[0].plot(cycles_s, means, color=BLUE, lw=1.8, label="Mean")
                    axes[0].fill_between(cycles_s, means - stds, means + stds,
                                         alpha=0.2, color=BLUE, label="±1 std")
                    axes[0].plot(cycles_s, p90s, color="#FF9800", linestyle="--",
                                 lw=1.5, label="p90")
                    axes[0].set_xlabel("Cycle")
                    axes[0].set_ylabel("Combined uncertainty score")
                    axes[0].set_title("Score Distribution  Σ_d β^(d)·u^(d)", fontsize=9)
                    axes[0].legend(fontsize=8)
                    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)

                if has_delta:
                    delta_arr = np.array(self.delta_history)   # (T, D)
                    for d in range(self.D):
                        axes[-1].plot(np.arange(1, len(delta_arr) + 1), delta_arr[:, d],
                                      marker="^", markersize=4, linestyle=":",
                                      label=labels[d], color=colors[d], lw=1.4)
                    axes[-1].axhline(0, color=GREY, lw=0.8, alpha=0.5)
                    axes[-1].set_xlabel("Cycle")
                    axes[-1].set_ylabel("ΔU (normalized)")
                    axes[-1].set_title("Normalized ΔU per Stage\n"
                                       ">0: improving  <0: regressing", fontsize=9)
                    axes[-1].legend(fontsize=8)
                    axes[-1].grid(alpha=0.3)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            d = pdf.infodict()
            d["Title"]   = "Pipeline Active Learning Report"
            d["Subject"] = "Steel Surface Inspection — Roth & Small (2008)"
            d["Creator"] = "steeldefect_active_learning.active_learning.pipeline_al"

        print(f"[pipeline_al] Report saved: {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stage_labels(D: int) -> List[str]:
    defaults = ["detection", "classification", "severity", "quality"]
    return [f"stage_{d}" if d >= len(defaults) else defaults[d] for d in range(D)]

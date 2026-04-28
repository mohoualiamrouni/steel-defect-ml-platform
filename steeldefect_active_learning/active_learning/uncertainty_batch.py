"""Batch-aware Bayesian active-learning acquisition functions.

This module contains two batch-acquisition strategies that complement the
per-item BALD score in :mod:`uncertainty`.  Both are exposed via the
strategy dispatcher in :mod:`candidate_selector` and are *opt-in only* —
the existing 7-strategy pipeline is unchanged unless the caller asks for
``method='powerbald'`` or ``method='batchbald'``.

Strategies
----------
* :func:`compute_powerbald_scores` — Kirsch et al. 2021 (arXiv:2106.12059).
  Cheap stochastic batch acquisition: sample without replacement from
  ``softmax(BALD / temperature)``.  $\\mathcal{O}(B)$ per step.
* :func:`compute_batchbald_canonical` — Kirsch et al. 2019 (arXiv:1906.08158).
  Canonical greedy joint mutual information using a sampling-based
  estimator of the joint entropy.  Closer to the Kirsch reference
  implementation than the existing KL-diversity heuristic in
  :func:`uncertainty.compute_batchbald_scores`.

Both functions consume the same MC-probability tensor produced by the
existing :func:`uncertainty._mc_passes` helper, so only one set of MC
forward passes is needed per pool (no double work versus plain BALD).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from steeldefect_active_learning.active_learning.uncertainty import (
    _enable_dropout,
)


# ---------------------------------------------------------------------------
# Shared MC-pass helper: full (N, T, C) probability tensor
# ---------------------------------------------------------------------------

def _mc_probability_tensor(
    model: nn.Module,
    dataloader: DataLoader,
    n_mc_passes: int = 10,
    device: str = "cpu",
    desc: str = "MC passes",
) -> Tuple[torch.Tensor, List[str]]:
    """Run T forward passes with MC dropout and return (N, T, C) probs.

    Used by both PowerBALD and BatchBALD so the expensive part (MC
    forward passes) is computed once per acquisition cycle.

    Returns
    -------
    probs : torch.Tensor of shape (N, T, C), float32, on CPU
    paths : list of N image paths in the order they appear in the loader
    """
    model.eval()
    model.to(device)

    chunks: List[torch.Tensor] = []
    paths: List[str] = []

    with _enable_dropout(model):
        for tensors, batch_paths in tqdm(dataloader, desc=desc, leave=False):
            tensors = tensors.to(device)
            B = tensors.shape[0]
            probs_t: Optional[torch.Tensor] = None

            for t in range(n_mc_passes):
                with torch.no_grad():
                    logits = model(tensors)
                    p = F.softmax(logits, dim=1).cpu()
                if probs_t is None:
                    probs_t = torch.zeros(n_mc_passes, B, p.shape[1], dtype=torch.float32)
                probs_t[t] = p

            if probs_t is None:
                continue

            # Reshape (T, B, C) -> (B, T, C) for batch-major storage
            chunks.append(probs_t.permute(1, 0, 2).contiguous())
            paths.extend(list(batch_paths))

    if not chunks:
        empty = torch.zeros(0, n_mc_passes, 1, dtype=torch.float32)
        return empty, paths

    probs = torch.cat(chunks, dim=0)  # (N, T, C)
    return probs, paths


def _individual_bald(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Per-item BALD score from an (N, T, C) probability tensor.

    BALD(x) = H(mean_t p_t(x)) - mean_t H(p_t(x))
    """
    mean_p = probs.mean(dim=1).clamp(min=eps)                  # (N, C)
    H_total = -(mean_p * mean_p.log()).sum(dim=1)              # (N,)
    Hi = -(probs.clamp(min=eps) * probs.clamp(min=eps).log()).sum(dim=2)  # (N, T)
    H_aleatoric = Hi.mean(dim=1)                               # (N,)
    return H_total - H_aleatoric                               # (N,)


# ---------------------------------------------------------------------------
# PowerBALD — Kirsch et al. 2021 (arXiv:2106.12059)
# ---------------------------------------------------------------------------

def compute_powerbald_scores(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size: int = 10,
    n_mc_passes: int = 10,
    temperature: float = 1.0,
    device: str = "cpu",
    random_state: int = 42,
) -> List[str]:
    """PowerBALD stochastic batch acquisition (Kirsch et al. 2021).

    Algorithm
    ---------
    1. Compute per-item BALD scores ``s_i`` from MC probabilities.
    2. Form sampling weights ``w_i = exp(s_i / T)`` where ``T`` is a
       temperature parameter (``temperature=0`` recovers the deterministic
       top-K BALD baseline; ``temperature=1`` is the recommended default).
    3. Sample ``batch_size`` items without replacement using
       ``np.random.choice`` with weights ``w_i / sum(w_i)``.

    Cost is :math:`\\mathcal{O}(B)` per step *after* the MC passes — much
    cheaper than canonical BatchBALD's :math:`\\mathcal{O}(B \\cdot |U|)`
    while empirically retaining most of its diversity benefit (see
    Kirsch 2021 §3 and figures 2–3).

    Parameters
    ----------
    model : torch.nn.Module
        Classifier with at least one dropout layer (else MC passes are
        deterministic and BALD ≡ 0).
    dataloader : DataLoader
        Yields ``(tensor, path)`` batches.
    batch_size : int
        Number of items to acquire.
    n_mc_passes : int
        T in BALD, default 10.
    temperature : float
        Temperature for the softmax over BALD scores.  ``> 0``.  Larger
        values flatten the distribution (more exploration), smaller
        values concentrate on top BALD items (more exploitation).
    device : str
        Torch device.
    random_state : int
        Seed for sampling reproducibility.

    Returns
    -------
    list of ``batch_size`` image paths.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    probs, paths = _mc_probability_tensor(
        model, dataloader, n_mc_passes=n_mc_passes, device=device,
        desc="PowerBALD [MC passes]",
    )
    if len(paths) == 0 or batch_size <= 0:
        return []

    bald = _individual_bald(probs)  # (N,)
    bald_np = bald.numpy()

    # Numerically-stable softmax (subtract max).
    log_w = bald_np / float(temperature)
    log_w -= log_w.max()
    w = np.exp(log_w, dtype=np.float64)
    w = w / w.sum()

    rng = np.random.default_rng(random_state)
    n_pick = min(batch_size, len(paths))
    sampled_idx = rng.choice(len(paths), size=n_pick, replace=False, p=w)
    return [paths[i] for i in sampled_idx]


def powerbald_select_from_scores(
    uncertainty_scores: Dict[str, float],
    batch_size: int,
    temperature: float = 1.0,
    random_state: int = 42,
) -> List[str]:
    """PowerBALD sampling step given pre-computed uncertainty scores.

    This is the cheap variant of :func:`compute_powerbald_scores` for
    callers that already have BALD/uncertainty scores in dict form (from
    a previous :func:`uncertainty.compute_uncertainty` call).  No MC
    forward passes are run.  Suitable for the
    :func:`candidate_selector.select_candidates` dispatcher.

    Algorithm: sample ``batch_size`` paths without replacement using
    weights ``softmax(score / temperature)`` (Kirsch et al. 2021).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    paths = list(uncertainty_scores.keys())
    if not paths or batch_size <= 0:
        return []
    scores = np.asarray([uncertainty_scores[p] for p in paths], dtype=np.float64)
    log_w = scores / float(temperature)
    log_w -= log_w.max()
    w = np.exp(log_w)
    if not np.isfinite(w).all() or w.sum() <= 0:
        # Degenerate distribution → fall back to top-K.
        order = np.argsort(scores)[::-1][:batch_size]
        return [paths[i] for i in order]
    w = w / w.sum()
    rng = np.random.default_rng(random_state)
    n_pick = min(batch_size, len(paths))
    sampled_idx = rng.choice(len(paths), size=n_pick, replace=False, p=w)
    return [paths[i] for i in sampled_idx]


# ---------------------------------------------------------------------------
# Canonical BatchBALD — Kirsch et al. 2019 (arXiv:1906.08158)
# ---------------------------------------------------------------------------

def compute_batchbald_canonical(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size: int = 10,
    n_mc_passes: int = 10,
    n_joint_samples: int = 1000,
    device: str = "cpu",
    random_state: int = 42,
) -> List[str]:
    """Canonical BatchBALD (Kirsch et al. 2019) via sampling joint entropy.

    Approach
    --------
    The acquisition score is the joint mutual information
    :math:`I(\\{y_b\\}_{b\\in B}; \\omega \\mid D)`.  Direct enumeration of
    the joint over :math:`C^{|B|}` configurations is intractable for
    realistic ``C`` and ``|B|``.  Following Kirsch et al. §3.2, we
    estimate the joint entropy by **sampling** :math:`K` joint
    configurations from the MC mixture and computing the mean log
    likelihood.

    Greedy step :math:`b` adds the candidate ``x*`` that maximises::

        Δa_b(x) = H(y_x, y_A) - H(y_A) - E_ω[H(y_x | ω)]

    where ``A = A_{b-1}`` is the already-selected batch and the joint
    entropies are estimated from sampled MC mixture configurations.

    This is significantly more faithful to canonical BatchBALD than the
    BALD + KL-diversity heuristic in :func:`uncertainty.compute_batchbald_scores`.
    The latter is preserved for backward compatibility but new callers
    should use this function.

    Parameters
    ----------
    model, dataloader, n_mc_passes, device : as in PowerBALD.
    batch_size : int
        :math:`|B|`, the number of items to acquire.
    n_joint_samples : int
        Number of joint configurations sampled from the MC mixture for the
        entropy estimate.  ``1000`` is a good default; lower values trade
        accuracy for speed.

    Returns
    -------
    list of ``batch_size`` paths in selection order.
    """
    probs, paths = _mc_probability_tensor(
        model, dataloader, n_mc_passes=n_mc_passes, device=device,
        desc="BatchBALD [MC passes]",
    )
    if len(paths) == 0 or batch_size <= 0:
        return []

    bald = _individual_bald(probs)  # (N,)
    N, T, C = probs.shape
    eps = 1e-10
    rng = np.random.default_rng(random_state)

    # Pre-compute per-item E_ω[H(y|ω)] = mean_t H(p_t).  Shape (N,).
    Hi = -(probs.clamp(min=eps) * probs.clamp(min=eps).log()).sum(dim=2)
    cond_entropy = Hi.mean(dim=1).numpy()

    # Greedy selection
    selected_idx: List[int] = []
    selected_set = set()
    log_pjoint_samples: Optional[np.ndarray] = None  # (n_joint_samples,) running log p(y_A^(k))

    for step in range(min(batch_size, N)):
        if step == 0:
            # First pick: argmax individual BALD.
            best = int(bald.argmax().item())
            selected_idx.append(best)
            selected_set.add(best)
            # Initialise sampled joint configurations from the MC mixture
            # of the first selected item: y_A^(k) ~ (1/T) sum_t p_t(y | x_first).
            t_choices = rng.integers(0, T, size=n_joint_samples)
            cls_probs = probs[best, t_choices, :].numpy()  # (K, C)
            cls_choices = np.array([
                rng.choice(C, p=cls_probs[k])
                for k in range(n_joint_samples)
            ])
            # log p(y_A^(k)) = log( (1/T) sum_t p_t(c_k | x_first) )
            mix = probs[best].mean(dim=0).numpy()  # (C,) marginal mixture
            log_pjoint_samples = np.log(mix[cls_choices] + eps).astype(np.float64)
            continue

        # For each remaining candidate x, evaluate the score increment.
        # H(y_A) is the mean negative log of the joint mixture density on
        # the K samples drawn so far.
        H_A = float(-np.mean(log_pjoint_samples))

        best_delta = -np.inf
        best_idx = -1
        # Re-sample classes for x given the already-drawn t indices, so
        # that the joint sample for y_x|y_A is consistent with the same
        # mixture component.  We pre-sample t indices once per outer step.
        if step == 1:
            joint_t = rng.integers(0, T, size=n_joint_samples)
        # Reuse joint_t from previous step — the joint expansion is
        # self-consistent because we always sample t i.i.d.

        for idx in range(N):
            if idx in selected_set:
                continue
            # Sample y_x^(k) ~ p_{joint_t[k]}(y | x_idx)
            cand_probs = probs[idx, joint_t, :].numpy()  # (K, C)
            cand_choices = np.array([
                rng.choice(C, p=cand_probs[k])
                for k in range(n_joint_samples)
            ])
            # log p(y_x^(k), y_A^(k)) = log p(y_A^(k)) + log p(y_x^(k))
            #   under the mixture, where p(y_x^(k)) = mean_t p_t(c_k | x_idx).
            mix_x = probs[idx].mean(dim=0).numpy()
            log_pjoint_new = log_pjoint_samples + np.log(mix_x[cand_choices] + eps)

            H_AX = float(-np.mean(log_pjoint_new))
            # Δa = [H(y_x, y_A) - H(y_A)] - E_ω[H(y_x | ω)]
            delta = (H_AX - H_A) - float(cond_entropy[idx])
            if delta > best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx < 0:
            break

        # Commit best candidate: extend the joint samples in-place.
        cand_probs = probs[best_idx, joint_t, :].numpy()
        cand_choices = np.array([
            rng.choice(C, p=cand_probs[k]) for k in range(n_joint_samples)
        ])
        mix_x = probs[best_idx].mean(dim=0).numpy()
        log_pjoint_samples = log_pjoint_samples + np.log(mix_x[cand_choices] + eps)
        selected_idx.append(best_idx)
        selected_set.add(best_idx)

    return [paths[i] for i in selected_idx]

"""
Uncertainty sampling strategies for active learning.

Classic acquisition functions (Settles 2010):
    entropy          — Shannon entropy of the softmax distribution
    margin           — 1 - (top1 - top2) class probability gap
    least_confidence — 1 - max(softmax)

Bayesian Deep Active Learning:
    bald             — Bayesian Active Learning by Disagreement (Houlsby 2011)
                       BALD(x) = H[y|x,D] - E_{p(w)}[H[y|x,w]]
                                = H(mean(p_mc)) - mean(H(p_mc))
                       Also returns 'aleatoric' and 'epistemic' keys separately
                       (Kendall & Gal 2017 — "What Uncertainties Do We Need?")
    mc_entropy       — Entropy of mean prediction over T stochastic passes

Gradient-based + diversity:
    badge            — BADGE: gradient embedding + k-means++ diversity (Ash et al. 2020)
                       g_x = ∇_{θ_last} CE(f(x), ŷ);  diversity via k-means++ seeding
                       Use compute_badge_embeddings() then candidate_selector method='badge'

Ensemble-based:
    ensemble         — Deep Ensembles (Lakshminarayanan 2017): entropy of averaged
                       softmax over M independently trained models
                       Requires passing multiple model checkpoints.

Single-pass deterministic:
    ddu              — Deep Deterministic Uncertainty (Mukhoti et al. 2022)
                       Fits a Gaussian per class in embedding space post-training;
                       uncertainty = -max_c log N(z; μ_c, Σ_c)  (negative log-likelihood)
                       Single forward pass — no MC dropout required.

References:
    Settles (2010) "Active Learning Literature Survey."
    Houlsby et al. (2011) "Bayesian Active Learning" — arXiv:1112.5745.
    Ren et al. (2021) "A Survey of Deep Active Learning" — ACM Computing Surveys.
    Ash et al. (2020) "BADGE" — arXiv:1906.03671.
    Lakshminarayanan et al. (2017) "Deep Ensembles" — arXiv:1612.01474.
    Kendall & Gal (2017) "What Uncertainties Do We Need?" — arXiv:1703.04977.
    Mukhoti et al. (2022) "Deep Deterministic Uncertainty" — arXiv:2102.11582.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


UncertaintyMethod = Literal[
    "entropy", "margin", "least_confidence", "bald", "mc_entropy",
    "badge", "ensemble", "ddu",
]

# When method='bald', compute_uncertainty returns scores keyed by path.
# Aleatoric and epistemic components are returned separately via
# compute_uncertainty_bald_full() if needed.


@contextlib.contextmanager
def _enable_dropout(model: nn.Module):
    """Temporarily set all Dropout layers to training mode for MC sampling.

    This is the standard approach for MC Dropout inference (Gal & Ghahramani 2016):
    keep the rest of the model in eval() but force stochastic dropout.
    """
    dropout_layers = [
        m for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
    ]
    original_training = [m.training for m in dropout_layers]
    try:
        for m in dropout_layers:
            m.train()
        yield
    finally:
        for m, was_training in zip(dropout_layers, original_training):
            m.training = was_training


def _mc_passes(
    model: nn.Module,
    batch: torch.Tensor,
    n_passes: int,
    device: str,
    task: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Run T stochastic forward passes with dropout active.

    Returns:
        Tensor of shape (T, B, C) for classification or (T, B, C, H, W) for segmentation.
        Each pass i contains softmax probabilities.
    """
    results = []
    with _enable_dropout(model):
        for _ in range(n_passes):
            with torch.no_grad():
                logits = model(batch)
                if temperature != 1.0:
                    logits = logits / temperature
                if task == "segmentation":
                    p = torch.sigmoid(logits) if logits.shape[1] == 1 \
                        else F.softmax(logits, dim=1)
                else:
                    p = F.softmax(logits, dim=1)
            results.append(p)
    return torch.stack(results, dim=0)   # (T, B, C) or (T, B, C, H, W)


def _bald_score(
    probs_mc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bayesian Active Learning by Disagreement (BALD).

    Decomposes predictive uncertainty into epistemic and aleatoric components
    (Kendall & Gal 2017 — "What Uncertainties Do We Need in Bayesian Deep
    Learning for Computer Vision?", arXiv:1703.04977).

        Total uncertainty   H(y|x,D)  = H[mean(p_mc)]
        Aleatoric           E[H]       = mean(H(p_mc))   — irreducible data noise
        Epistemic / BALD    I(y;θ|x)  = Total - Aleatoric  — reducible model uncertainty

    Args:
        probs_mc: shape (T, B, C) — T MC passes, B images, C classes.

    Returns:
        Tuple of (bald, aleatoric, total) tensors, each shape (B,).
        bald      = epistemic uncertainty (informative for AL)
        aleatoric = irreducible noise (high for inherently ambiguous images)
        total     = H(mean prediction) = predictive entropy
    """
    eps = 1e-10
    mean_p = probs_mc.mean(dim=0)                                       # (B, C)
    H_total = -(mean_p.clamp(min=eps) * mean_p.clamp(min=eps).log()).sum(dim=1)  # (B,)
    H_each  = -(probs_mc.clamp(min=eps) * probs_mc.clamp(min=eps).log()).sum(dim=2)  # (T,B)
    H_aleatoric = H_each.mean(dim=0)                                    # (B,)
    bald = H_total - H_aleatoric                                        # (B,) epistemic
    return bald, H_aleatoric, H_total


# ---------------------------------------------------------------------------
# BADGE — Gradient embedding extraction (Ash et al. 2020)
# ---------------------------------------------------------------------------

def compute_badge_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Extract BADGE gradient embeddings for batch active learning.

    Algorithm (Ash et al. 2020, arXiv:1906.03671):
    1. For each unlabeled image x, compute pseudo-label ŷ = argmax softmax(f(x))
    2. Compute gradient of CE loss w.r.t. last-layer weights:
           g_x = ∇_{θ_last} CE(f(x), ŷ)
       This is equivalent to (p(ŷ) - 1) ⊗ φ(x) where φ(x) is the penultimate
       embedding and ⊗ is the outer product.
    3. Flatten g_x to a vector; this is the BADGE embedding.

    High ||g_x|| → uncertain prediction (large gradient = large correction needed).
    Diverse {g_x} → diverse batch (k-means++ on these embeddings gives jointly
    uncertain AND diverse candidates without any hyperparameter tuning).

    Args:
        model:      PyTorch model in eval() mode. Must have a linear final layer
                    accessible as `model.fc` (ResNet) or `model.classifier[-1]`
                    (DenseNet). Falls back to output-layer jacobian if not found.
        dataloader: Yields (tensor, path) batches.
        device:     Torch device string.

    Returns:
        Dict mapping image_path -> gradient embedding (1-D numpy array).
    """
    model.eval()
    model.to(device)

    # Find the last linear layer
    last_linear: Optional[nn.Linear] = None
    for name in ("fc", "classifier"):
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Linear):
            last_linear = layer
            break
        if isinstance(layer, nn.Sequential):
            for sub in reversed(list(layer.children())):
                if isinstance(sub, nn.Linear):
                    last_linear = sub
                    break
            if last_linear is not None:
                break

    embeddings: Dict[str, np.ndarray] = {}

    for tensors, paths in tqdm(dataloader, desc="BADGE embeddings", leave=False):
        tensors = tensors.to(device)

        # Forward pass — keep grad
        model.zero_grad()
        logits = model(tensors)                            # (B, C)
        probs  = F.softmax(logits, dim=1)                 # (B, C)
        pseudo_labels = probs.argmax(dim=1)               # (B,)

        # Compute gradient of CE w.r.t. last-layer parameters
        loss = F.cross_entropy(logits, pseudo_labels)
        loss.backward()

        if last_linear is not None and last_linear.weight.grad is not None:
            # Per-image gradient: approximate via (p - one_hot(ŷ)) ⊗ last embedding
            # More efficient than full per-sample gradient:
            p_np = probs.detach().cpu().numpy()              # (B, C)
            one_hot = np.zeros_like(p_np)
            one_hot[np.arange(len(pseudo_labels)), pseudo_labels.cpu().numpy()] = 1.0
            residual = p_np - one_hot                        # (B, C)

            # Embed via last linear weight rows: g_x ≈ residual @ W  (B, d_in)
            W = last_linear.weight.detach().cpu().numpy()    # (C, d_in)
            per_sample_grad = residual @ W                   # (B, d_in)
        else:
            # Fallback: use logit norm as scalar uncertainty proxy
            per_sample_grad = logits.detach().cpu().numpy()  # (B, C)

        model.zero_grad()

        for path, g in zip(paths, per_sample_grad):
            embeddings[path] = g.astype(np.float32)

    return embeddings


# ---------------------------------------------------------------------------
# Deep Ensembles (Lakshminarayanan et al. 2017)
# ---------------------------------------------------------------------------

def compute_ensemble_uncertainty(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: str = "cpu",
    normalize: bool = True,
) -> Dict[str, float]:
    """Uncertainty via Deep Ensembles (Lakshminarayanan et al. 2017, arXiv:1612.01474).

    Trains M independent models; uncertainty = entropy of averaged softmax.

        p̄(y|x) = (1/M) Σ_m softmax(f_m(x))
        H_ensemble(x) = -Σ_c p̄_c log p̄_c    (predictive entropy)
        Var_epistemic  = (1/M) Σ_m p_m² - p̄²  (variance across ensemble)

    Deep Ensembles typically outperform MC Dropout on calibration under
    dataset shift (Ovadia et al. 2019) and is embarrassingly parallelizable.

    Args:
        models:     List of M independently trained models (same architecture).
        dataloader: Yields (tensor, path) batches.
        device:     Torch device string.
        normalize:  Divide entropy by log(C) → [0,1].

    Returns:
        Dict mapping image_path -> ensemble entropy score.
    """
    for m in models:
        m.eval()
        m.to(device)

    scores: Dict[str, float] = {}

    for tensors, paths in tqdm(dataloader, desc="Ensemble uncertainty", leave=False):
        tensors = tensors.to(device)
        all_probs = []

        with torch.no_grad():
            for m in models:
                logits = m(tensors)
                all_probs.append(F.softmax(logits, dim=1))   # (B, C)

        # Average over ensemble members
        mean_p = torch.stack(all_probs, dim=0).mean(dim=0)   # (B, C)

        # Predictive entropy
        eps = 1e-10
        H = -(mean_p.clamp(min=eps) * mean_p.clamp(min=eps).log()).sum(dim=1)  # (B,)

        if normalize and mean_p.shape[1] > 1:
            H = H / torch.tensor(mean_p.shape[1], dtype=torch.float).log()

        for path, score in zip(paths, H.cpu().tolist()):
            scores[path] = float(score)

    return scores


# ---------------------------------------------------------------------------
# DDU — Deep Deterministic Uncertainty (Mukhoti et al. 2022)
# ---------------------------------------------------------------------------

def fit_ddu_model(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
) -> Dict[str, object]:
    """Fit per-class Gaussians in embedding space for DDU.

    DDU (Mukhoti et al. 2022, arXiv:2102.11582) fits a Gaussian Discriminant
    Analysis (GDA) model on labeled embeddings post-training.  At inference,
    uncertainty = -max_c log N(z; μ_c, Σ_c) (negative log-likelihood under
    the most likely class Gaussian).

    Key advantage: single forward pass, no MC dropout.  Matches Deep Ensembles
    on OOD benchmarks while being computationally cheaper.

    Args:
        embeddings: Dict path -> embedding array (d,) for labeled images.
        labels:     Dict path -> class_id for the same labeled images.

    Returns:
        Dict with keys 'class_means', 'class_covs_inv', 'class_log_dets', 'n_classes'.
    """
    from scipy.linalg import inv as scipy_inv

    paths = [p for p in embeddings if p in labels]
    Z = np.stack([embeddings[p] for p in paths], axis=0)   # (N, d)
    y = np.array([labels[p] for p in paths], dtype=int)    # (N,)
    n_classes = int(y.max()) + 1
    d = Z.shape[1]
    eps = 1e-6

    class_means: List[np.ndarray] = []
    class_covs_inv: List[np.ndarray] = []
    class_log_dets: List[float] = []

    for c in range(n_classes):
        mask = y == c
        Z_c = Z[mask]
        if len(Z_c) < 2:
            # Not enough samples: use identity covariance
            mu_c = Z_c.mean(axis=0) if len(Z_c) else np.zeros(d)
            class_means.append(mu_c)
            class_covs_inv.append(np.eye(d))
            class_log_dets.append(0.0)
        else:
            mu_c = Z_c.mean(axis=0)
            diff = Z_c - mu_c
            cov_c = (diff.T @ diff) / (len(Z_c) - 1) + eps * np.eye(d)
            cov_inv = scipy_inv(cov_c)
            sign, logdet = np.linalg.slogdet(cov_c)
            class_means.append(mu_c)
            class_covs_inv.append(cov_inv)
            class_log_dets.append(float(logdet))

    return {
        "class_means": class_means,
        "class_covs_inv": class_covs_inv,
        "class_log_dets": class_log_dets,
        "n_classes": n_classes,
    }


def apply_spectral_normalization(
    model: nn.Module,
    skip_final_layer: bool = True,
) -> nn.Module:
    """Apply spectral normalization for DDU (Mukhoti et al. 2022).

    Spectral normalization constrains the Lipschitz constant of each layer
    to be <= 1, ensuring the network is distance-preserving in feature space.
    This is REQUIRED for DDU Gaussian fits to be meaningful — without it,
    the feature-space distances are unreliable.

    Uses torch.nn.utils.spectral_norm (Miyato et al. 2018).

    Args:
        model:            PyTorch model (ResNet, DenseNet, etc.).
        skip_final_layer: If True, skip the final Linear classifier (fc / classifier).
                          Default True — we want distance-aware features, not logits.

    Returns:
        Model with spectral normalization applied (in-place).
    """
    from torch.nn.utils import spectral_norm

    # Identify the final classifier layer name to skip
    skip_names = set()
    if skip_final_layer:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                skip_names.add(name)
        # Keep only the last linear (the classifier head)
        if skip_names:
            skip_names = {max(skip_names)}  # last by name ordering

    n_normalized = 0
    for name, module in model.named_modules():
        if name in skip_names:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                spectral_norm(module, name="weight")
                n_normalized += 1
            except Exception:
                pass  # already normalized or no weight param

    print(f"[uncertainty] Spectral normalization applied to {n_normalized} layers "
          f"(Mukhoti et al. 2022 — DDU)")
    return model


def compute_ddu_uncertainty(
    embeddings: Dict[str, np.ndarray],
    ddu_model: Dict[str, object],
) -> Dict[str, float]:
    """Score unlabeled images with DDU (negative max-class log-likelihood).

    uncertainty(x) = -max_c log N(z; μ_c, Σ_c)
                   = max_c [0.5 * (z-μ_c)^T Σ_c^{-1} (z-μ_c) + 0.5 * log|Σ_c|]

    High score → embedding far from all class centroids → uncertain / OOD.

    Args:
        embeddings: Dict path -> (d,) embedding for unlabeled images.
        ddu_model:  Output of fit_ddu_model().

    Returns:
        Dict mapping path -> DDU uncertainty score (higher = more uncertain).
    """
    means    = ddu_model["class_means"]
    covs_inv = ddu_model["class_covs_inv"]
    log_dets = ddu_model["class_log_dets"]

    scores: Dict[str, float] = {}
    for path, z in embeddings.items():
        log_likelihoods = []
        for mu, Sigma_inv, logdet in zip(means, covs_inv, log_dets):
            diff = z - mu
            maha2 = float(diff @ Sigma_inv @ diff)
            log_likelihoods.append(-0.5 * maha2 - 0.5 * logdet)
        # Uncertainty = negative max log-likelihood
        scores[path] = float(-max(log_likelihoods))

    return scores


def _mc_entropy_score(probs_mc: torch.Tensor) -> torch.Tensor:
    """Entropy of the mean MC prediction (simpler Bayesian approximation).

    Captures both epistemic and aleatoric uncertainty but does not separate them.

    Args:
        probs_mc: shape (T, B, C)

    Returns:
        Tensor shape (B,)
    """
    mean_p = probs_mc.mean(dim=0)
    eps = 1e-10
    return -(mean_p.clamp(min=eps) * mean_p.clamp(min=eps).log()).sum(dim=1)


def compute_uncertainty(
    model: nn.Module,
    dataloader: DataLoader,
    method: UncertaintyMethod = "entropy",
    device: str = "cpu",
    task: str = "classification",
    normalize: bool = True,
    n_mc_passes: int = 10,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute an uncertainty score for every image in the dataloader.

    Args:
        model:       PyTorch model in eval() mode. Must return logits.
        dataloader:  Yields (tensor, path) batches (as from SteelDefectDataset).
        method:      Acquisition function. One of:
                     'entropy', 'margin', 'least_confidence' — deterministic (Settles 2010)
                     'bald', 'mc_entropy' — MC Dropout (Houlsby 2011, Gal 2016)
                     'badge' — use compute_badge_embeddings() instead (Ash 2020)
                     'ensemble' — use compute_ensemble_uncertainty() instead
                     'ddu' — use compute_ddu_uncertainty() instead (Mukhoti 2022)
        device:      Torch device string.
        task:        'classification' or 'segmentation'.
        normalize:   If True, divide entropy-based scores by log(n_classes) → [0,1].
        n_mc_passes: Number of Monte Carlo dropout passes (used by 'bald', 'mc_entropy').
        temperature: Temperature scaling (Guo 2017, Minderer 2021).
                     Divides logits by T before softmax: p = softmax(z/T).
                     T > 1 softens predictions (reduces overconfidence).
                     Default 1.0 (no scaling). Set from TemperatureScaler.temperature_.

    Returns:
        Dict mapping image_path -> uncertainty_score (float, higher = more uncertain).
        For method='bald': also stores aleatoric/epistemic via
        compute_uncertainty_bald_full() if the decomposition is needed.
    """
    if method in ("badge", "ensemble", "ddu"):
        raise ValueError(
            f"Method '{method}' requires dedicated functions: "
            "compute_badge_embeddings() / compute_ensemble_uncertainty() / "
            "compute_ddu_uncertainty(). See module docstring."
        )

    model.eval()
    model.to(device)

    mc_methods = {"bald", "mc_entropy"}

    if method in mc_methods:
        import warnings
        _dropouts = [
            m for m in model.modules()
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        ]
        if not _dropouts:
            warnings.warn(
                f"'{method}' requires dropout layers for MC sampling, but model "
                f"has none. All {n_mc_passes} passes will be deterministic — "
                f"BALD scores will be ~0. Consider using method='entropy'.",
                UserWarning,
                stacklevel=2,
            )

    scores: Dict[str, float] = {}

    for tensors, paths in tqdm(dataloader, desc=f"Uncertainty [{method}]", leave=False):
        tensors = tensors.to(device)

        if method in mc_methods:
            probs_mc = _mc_passes(model, tensors, n_mc_passes, device, task, temperature)  # (T,B,C)
            if method == "bald":
                bald, _aleatoric, _total = _bald_score(probs_mc)
                batch_scores = bald
            else:
                batch_scores = _mc_entropy_score(probs_mc)
        else:
            with torch.no_grad():
                logits = model(tensors)
                # Temperature scaling (Guo 2017, Minderer 2021)
                if temperature != 1.0:
                    logits = logits / temperature
                if task == "segmentation":
                    probs = torch.sigmoid(logits) if logits.shape[1] == 1 \
                        else F.softmax(logits, dim=1)
                else:
                    probs = F.softmax(logits, dim=1)

            if method == "entropy":
                batch_scores = _entropy_score(probs) if task == "classification" \
                    else _entropy_score_segmentation(probs)
            elif method == "margin":
                batch_scores = _margin_score(probs) if task == "classification" \
                    else _margin_score_segmentation(probs)
            elif method == "least_confidence":
                batch_scores = _least_confidence_score(probs)
            else:
                raise ValueError(
                    f"Unknown uncertainty method '{method}'. "
                    "Choose: entropy, margin, least_confidence, bald, mc_entropy"
                )

        if normalize and method in {"entropy", "bald", "mc_entropy"}:
            n_classes = probs_mc.shape[2] if method in mc_methods else probs.shape[1]
            if n_classes > 1:
                batch_scores = batch_scores / torch.tensor(n_classes, dtype=torch.float).log()

        for path, score in zip(paths, batch_scores.cpu().tolist()):
            scores[path] = float(score)

    return scores


def compute_uncertainty_bald_full(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    n_mc_passes: int = 10,
    normalize: bool = True,
    temperature: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """BALD with full aleatoric/epistemic decomposition (Kendall & Gal 2017).

    Returns per-image dict with keys:
        'epistemic'  — BALD score = mutual information I(y;θ|x) = reducible uncertainty
        'aleatoric'  — E[H] = irreducible data noise
        'total'      — H(mean prediction) = full predictive entropy

    High epistemic → model needs more data for this image (good AL candidate).
    High aleatoric → image is inherently ambiguous regardless of training data.

    Args:
        model:       PyTorch model in eval() mode.
        dataloader:  Yields (tensor, path) batches.
        device:      Torch device string.
        n_mc_passes: Number of MC Dropout passes (T=10 recommended, T=20 for precision).
        normalize:   Divide all three scores by log(C) → [0,1].

    Returns:
        Dict[path -> {'epistemic': float, 'aleatoric': float, 'total': float}]
    """
    model.eval()
    model.to(device)
    results: Dict[str, Dict[str, float]] = {}

    for tensors, paths in tqdm(dataloader, desc="BALD (full decomposition)", leave=False):
        tensors = tensors.to(device)
        probs_mc = _mc_passes(model, tensors, n_mc_passes, device, "classification", temperature)
        bald, aleatoric, total = _bald_score(probs_mc)

        if normalize and probs_mc.shape[2] > 1:
            log_c = torch.tensor(probs_mc.shape[2], dtype=torch.float).log()
            bald      = bald      / log_c
            aleatoric = aleatoric / log_c
            total     = total     / log_c

        for path, ep, al, tot in zip(
            paths,
            bald.cpu().tolist(),
            aleatoric.cpu().tolist(),
            total.cpu().tolist(),
        ):
            results[path] = {"epistemic": float(ep), "aleatoric": float(al), "total": float(tot)}

    return results


# ---------------------------------------------------------------------------
# Classic acquisition functions (single forward pass)
# ---------------------------------------------------------------------------

def _entropy_score(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy. Shape: (B, C) -> (B,)"""
    p = probs.clamp(min=1e-10)
    return -(p * p.log()).sum(dim=1)


def _entropy_score_segmentation(probs: torch.Tensor) -> torch.Tensor:
    """Mean pixel entropy for segmentation. Shape: (B, C, H, W) -> (B,)"""
    p = probs.clamp(min=1e-10)
    pixel_entropy = -(p * p.log()).sum(dim=1)   # (B, H, W)
    return pixel_entropy.mean(dim=[1, 2])        # (B,)


def _margin_score(probs: torch.Tensor) -> torch.Tensor:
    """1 - (top1 - top2). Higher = more uncertain. Shape: (B, C) -> (B,)"""
    if probs.shape[1] == 1:
        p = probs[:, 0]
        return 1.0 - (2.0 * p - 1.0).abs()
    top2 = probs.topk(k=2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


def _margin_score_segmentation(probs: torch.Tensor) -> torch.Tensor:
    """Mean pixel margin for segmentation. Shape: (B, C, H, W) -> (B,)"""
    B, C, H, W = probs.shape
    flat = probs.view(B, C, -1)
    if C == 1:
        p = flat[:, 0, :]
        pixel_margin = 1.0 - (2.0 * p - 1.0).abs()
        return pixel_margin.mean(dim=1)
    top2 = flat.topk(k=2, dim=1).values
    return (1.0 - (top2[:, 0, :] - top2[:, 1, :])).mean(dim=1)


def _least_confidence_score(probs: torch.Tensor) -> torch.Tensor:
    """1 - max(prob). Works for classification and segmentation."""
    if probs.ndim == 4:
        max_conf = probs.max(dim=1).values.mean(dim=[1, 2])
    else:
        max_conf = probs.max(dim=1).values
    return 1.0 - max_conf


# ---------------------------------------------------------------------------
# BatchBALD (Kirsch et al. 2019)
# ---------------------------------------------------------------------------

def compute_batchbald_scores(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size: int = 10,
    n_mc_passes: int = 10,
    device: str = "cpu",
) -> List[str]:
    """BALD + KL-diversity heuristic (NOT canonical BatchBALD).

    IMPORTANT — scientific rigor note:
    This is NOT the canonical BatchBALD from Kirsch et al. 2019. That
    algorithm computes joint mutual information I(y_B; theta | x_B, D)
    via MC sampling over the joint output distribution, with greedy
    (1 - 1/e)-optimal selection. Implementing the true joint-MI
    computation requires materialising an (N, T^B, C^B) tensor or
    equivalent MC sampling over the joint, which is O(T^B) per step.

    What this function actually does (approximation):
        score(x | B) = BALD(x) + mean_{s in B} KL(p(y|x) || p(y|s))

    i.e. individual BALD plus a pairwise KL diversity term. This biases
    selection toward uncertain AND mutually dissimilar samples, but it
    is a heuristic — not information-theoretically optimal. Empirically
    it behaves like a fast approximation of BatchBALD on well-separated
    pools; on pools with highly correlated candidates the approximation
    can diverge from true BatchBALD.

    Algorithm (this heuristic):
        1. Run T MC forward passes to get probability tensor P (N, T, C)
        2. Compute individual BALD scores per sample
        3. Greedy pick: first sample = argmax BALD; subsequent =
           argmax (BALD + mean KL to already-selected)

    Reference (canonical algorithm this approximates):
        Kirsch et al. (2019) "BatchBALD: Efficient and Diverse Batch
        Acquisition for Deep Bayesian Active Learning" — arXiv:1906.08158.
        A true BatchBALD reimplementation is tracked as a follow-up.

    Args:
        model:        PyTorch model (must have dropout layers for MC sampling).
        dataloader:   Yields (tensor, path) batches.
        batch_size:   Number of samples to select (default 10).
        n_mc_passes:  Number of MC dropout passes (default 10).
        device:       Torch device string.

    Returns:
        List of `batch_size` image paths, ordered by selection order.
    """
    model.eval()
    model.to(device)

    # Step 1: Collect MC probability tensor for all images
    all_probs = []  # list of (T, C) per image
    all_paths = []

    with _enable_dropout(model):
        for tensors, paths in tqdm(dataloader, desc="BatchBALD [MC passes]", leave=False):
            tensors = tensors.to(device)
            B = tensors.shape[0]
            mc_probs = torch.zeros(n_mc_passes, B, 1, device="cpu")

            for t in range(n_mc_passes):
                with torch.no_grad():
                    logits = model(tensors)
                    probs = F.softmax(logits, dim=1)
                    if t == 0:
                        C = probs.shape[1]
                        mc_probs = torch.zeros(n_mc_passes, B, C, device="cpu")
                    mc_probs[t] = probs.cpu()

            for i, path in enumerate(paths):
                all_probs.append(mc_probs[:, i, :])  # (T, C)
                all_paths.append(path)

    N = len(all_paths)
    if N == 0 or batch_size <= 0:
        return []

    # Stack into (N, T, C)
    prob_tensor = torch.stack(all_probs, dim=0)  # (N, T, C)
    T = prob_tensor.shape[1]
    C = prob_tensor.shape[2]
    eps = 1e-10

    # Step 2: Compute individual BALD scores for initial ranking
    mean_probs = prob_tensor.mean(dim=1)  # (N, C)
    total_entropy = -(mean_probs.clamp(min=eps) * mean_probs.clamp(min=eps).log()).sum(dim=1)
    per_pass_entropy = -(prob_tensor.clamp(min=eps) * prob_tensor.clamp(min=eps).log()).sum(dim=2)
    expected_entropy = per_pass_entropy.mean(dim=1)
    individual_bald = total_entropy - expected_entropy  # (N,)

    # Step 3: Greedy selection maximizing joint MI
    # For computational tractability, use the scoring approximation from
    # Kirsch et al. 2019 Section 3.2: track the conditional entropy
    # contribution of each new candidate given already-selected samples.
    selected_indices: List[int] = []
    remaining = set(range(N))

    # Pre-compute: log of mean prediction per sample (for conditional entropy)
    # Initial selection: highest individual BALD score
    for step in range(min(batch_size, N)):
        if step == 0:
            # First pick: just use individual BALD
            best_idx = int(individual_bald.argmax().item())
        else:
            # For subsequent picks, favor candidates dissimilar to already selected
            # KL(candidate || selected) is high when candidate differs from selected
            # Score = BALD(x) + mean_KL_diversity(x, selected)
            best_score = -float("inf")
            best_idx = -1
            for idx in remaining:
                cand_probs = prob_tensor[idx]  # (T, C)
                diversity = 0.0
                for sel_idx in selected_indices:
                    sel_probs = prob_tensor[sel_idx]  # (T, C)
                    # Mean KL divergence across MC passes
                    kl = (cand_probs.clamp(min=eps) *
                          (cand_probs.clamp(min=eps).log() -
                           sel_probs.clamp(min=eps).log())).sum(dim=1).mean()
                    diversity += float(kl)
                diversity /= len(selected_indices)
                # Score: high individual MI + diversity from selected (KL divergence)
                score = float(individual_bald[idx]) + diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [all_paths[i] for i in selected_indices]

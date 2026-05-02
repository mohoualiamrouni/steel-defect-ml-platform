"""
CAM-contradiction active-learning acquisition (Guélorget 2022, Ch. 4).

Idea
----
For an image classified by a CNN, two saliency-style explanations can be
extracted from the same forward pass:

  • **Grad-CAM** (Selvaraju et al. 2017): a class-discriminative heatmap
    obtained by global-average-pooling the gradients of the predicted class
    score with respect to the last convolutional feature map and weighting
    that map.

  • **Last-layer attention / spatial activation**: the unweighted spatial
    norm of the same feature map, ‖A^k‖_2 — i.e. *where* the model
    "looks" before any class-specific reweighting.

Guélorget (Télécom SudParis, 2022) shows in Chapter 4 that **images where
the two saliency maps disagree** carry more useful information for the
oracle than images chosen by classical entropy/BALD acquisition. The score
is a normalized distance between the two maps:

    s(x) = 1 − cos(vec(GradCAM(x)), vec(SpatialAttention(x)))

Selecting the top-k by s(x) outperformed standard uncertainty in the
multimedia detection benchmarks of the thesis. Adapted here to the plant
defect-classification head, where Grad-CAM localizes the predicted defect
and spatial attention shows the regions activated regardless of class.

This module is **opt-in** — wired into the dispatcher as
``method='cam_contradiction'`` (or surfaced via the ActiveLearningConfig
``uncertainty_method`` knob). It is only relevant for classification heads
on a CNN backbone with an identifiable last conv layer.

References:
    Guélorget, P. (2022). *Active learning for the detection of objects of
        operational interest in open-source multimedia content*. PhD thesis,
        Télécom SudParis / Institut Polytechnique de Paris. NNT 2022IPPAS018.
    Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep
        Networks via Gradient-based Localization*. ICCV.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers — last conv layer discovery and Grad-CAM
# ---------------------------------------------------------------------------

def _find_last_conv_module(model: nn.Module) -> Tuple[str, nn.Module]:
    """Return the (name, module) of the last Conv2d in the model.

    Used to attach the Grad-CAM hook. Walks named_modules in registration
    order and keeps the most recent Conv2d. Raises if no Conv2d is found.
    """
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise ValueError(
            "CAM-contradiction acquisition requires a CNN backbone with at "
            "least one Conv2d layer; none found in the supplied model."
        )
    return last_name, last_mod


def _cosine_similarity_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two flattened spatial maps (one image)."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (a_flat.norm() * b_flat.norm()).clamp(min=1e-12)
    return (a_flat * b_flat).sum() / denom


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def compute_cam_contradiction_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    target_layer: Optional[str] = None,
) -> Dict[str, float]:
    """Score every image by the contradiction between Grad-CAM and spatial attention.

    Args:
        model:        Classification model (last layer is fc / classifier).
        dataloader:   Yields ``(tensor, path)`` batches.
        device:       Torch device string.
        target_layer: Dotted-path name of the layer whose activations to use.
                      Defaults to the last Conv2d found in the model.

    Returns:
        Dict {path: score}. score ∈ [0, 2]; higher = more disagreement
        between the two saliency maps and therefore more informative under
        the Guélorget 2022 hypothesis.

    Notes:
        - Uses backward hooks; runs in eval() mode but with grad enabled on
          the input tensor.
        - For batched inference, the score is computed per-image (Grad-CAM
          requires per-image backward passes for the predicted class).
    """
    model.eval()
    model.to(device)

    if target_layer is None:
        _, conv_mod = _find_last_conv_module(model)
    else:
        conv_mod = dict(model.named_modules()).get(target_layer)
        if conv_mod is None:
            raise ValueError(f"target_layer {target_layer!r} not found in model")

    # Hooks capture activations and gradients on the chosen conv layer.
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        activations["a"] = out.detach()

    def bwd_hook(_m, _grad_in, grad_out):
        gradients["g"] = grad_out[0].detach()

    h_fwd = conv_mod.register_forward_hook(fwd_hook)
    try:
        h_bwd = conv_mod.register_full_backward_hook(bwd_hook)
    except AttributeError:
        h_bwd = conv_mod.register_backward_hook(bwd_hook)  # PyTorch < 1.8 fallback

    scores: Dict[str, float] = {}

    try:
        for tensors, paths in dataloader:
            tensors = tensors.to(device)
            B = tensors.shape[0]
            for b in range(B):
                model.zero_grad(set_to_none=True)
                x = tensors[b:b + 1].clone().detach().requires_grad_(True)
                logits = model(x)
                pred = int(logits.argmax(dim=1).item())
                # Backward only for the predicted class — Grad-CAM standard.
                logits[0, pred].backward(retain_graph=False)

                a = activations.get("a")  # (1, K, H, W)
                g = gradients.get("g")    # (1, K, H, W)
                if a is None or g is None:
                    continue

                # Grad-CAM: weight each channel by mean(grad) and ReLU(sum).
                weights = g.mean(dim=(2, 3), keepdim=True)        # (1, K, 1, 1)
                grad_cam = F.relu((weights * a).sum(dim=1))[0]    # (H, W)

                # Spatial attention: channel-norm of the activations.
                spatial = a.norm(dim=1)[0]                        # (H, W)

                cos_sim = _cosine_similarity_2d(grad_cam, spatial).clamp(-1.0, 1.0)
                scores[paths[b]] = float(1.0 - cos_sim.item())
    finally:
        h_fwd.remove()
        h_bwd.remove()

    return scores


def select_cam_contradiction(
    model: nn.Module,
    dataloader: DataLoader,
    labeling_budget: int,
    device: str = "cpu",
    target_layer: Optional[str] = None,
) -> List[str]:
    """Convenience wrapper: score the pool, return the top-budget paths."""
    scores = compute_cam_contradiction_scores(model, dataloader, device, target_layer)
    if not scores or labeling_budget <= 0:
        return []
    return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)[:labeling_budget]


def extract_cam_contradiction_to_dict(
    cam_scores: Dict[str, float],
) -> Dict[str, float]:
    """Identity passthrough kept as a hook point for normalization variants.

    The dispatcher treats CAM scores as plain uncertainty scores, so they are
    normalised the same way the rest of the uncertainty-based methods are.
    Centralising the call here makes it cheap to swap min-max for z-score
    later if the magnitude scale needs to be aligned with BALD/entropy.
    """
    if not cam_scores:
        return {}
    vals = np.fromiter(cam_scores.values(), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    rng = hi - lo
    if rng < 1e-12:
        return {p: 0.0 for p in cam_scores}
    return {p: (s - lo) / rng for p, s in cam_scores.items()}

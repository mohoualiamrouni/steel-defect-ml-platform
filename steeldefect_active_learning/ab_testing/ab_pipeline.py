"""
A/B Testing pipeline for the inspection models: run two models on the same images and compare.

Supports:
    - Classification: predicted class, confidence, entropy, margin score,
                      calibration error (ECE), per-class breakdown
    - Segmentation:   predicted masks — IoU, Dice coefficient, boundary IoU,
                      defect fraction comparison

Output: a pandas DataFrame with one row per image.

Classification columns:
    image_path, pred_a, confidence_a, pred_b, confidence_b, agrees,
    confidence_delta, entropy_a, entropy_b, margin_a, margin_b

Segmentation columns:
    image_path, mask_defect_fraction_a, mask_defect_fraction_b,
    iou, dice, boundary_iou, agrees, confidence_delta, confidence_a, confidence_b

Usage:
    from steeldefect_active_learning.ab_testing.ab_pipeline import run_ab_comparison

    df = run_ab_comparison(
        model_a, model_b, dataloader,
        task="classification",
        device="cuda",
        store_probs=True,       # attach full softmax vectors for ECE
    )
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ab_comparison(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: DataLoader,
    task: str = "classification",
    device: str = "cpu",
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    store_probs: bool = False,
    temperature_a: float = 1.0,
    temperature_b: float = 1.0,
) -> pd.DataFrame:
    """Run both models on every batch and collect rich comparison results.

    Args:
        model_a:              First model (current production model).
        model_b:              Second model (challenger / new version).
        dataloader:           Yields (tensor, path) batches (SteelDefectDataset).
        task:                 'classification' or 'segmentation'.
        device:               Torch device ('cpu' or 'cuda').
        iou_threshold:        For segmentation: IoU >= threshold → 'agrees'.
        confidence_threshold: Min softmax confidence to count as a prediction.
        store_probs:          If True, attach full softmax vectors as
                              'probs_a' and 'probs_b' list columns (for ECE).
        temperature_a:        Temperature for Model A logits (Minderer 2021).
                              Pre-comparison calibration: divide logits by T
                              before softmax to correct overconfidence.
        temperature_b:        Temperature for Model B logits.

    Returns:
        pd.DataFrame with one row per image.
    """
    model_a.eval().to(device)
    model_b.eval().to(device)

    rows: List[dict] = []

    for tensors, paths in tqdm(dataloader, desc="A/B inference", leave=False):
        tensors = tensors.to(device)

        logits_a = model_a(tensors)
        logits_b = model_b(tensors)

        # Pre-comparison calibration (Minderer 2021)
        if temperature_a != 1.0:
            logits_a = logits_a / temperature_a
        if temperature_b != 1.0:
            logits_b = logits_b / temperature_b

        if task == "classification":
            batch_rows = _compare_classification(
                paths, logits_a, logits_b,
                confidence_threshold, store_probs,
            )
        else:
            batch_rows = _compare_segmentation(
                paths, logits_a, logits_b, iou_threshold,
            )

        rows.extend(batch_rows)

    df = pd.DataFrame(rows)

    # Attach ECE to the DataFrame metadata (not a per-row column)
    if task == "classification" and store_probs and "probs_a" in df.columns:
        ece_a = expected_calibration_error(df, model="a", n_bins=10)
        ece_b = expected_calibration_error(df, model="b", n_bins=10)
        df.attrs["ece_a"] = ece_a
        df.attrs["ece_b"] = ece_b

    return df


# ---------------------------------------------------------------------------
# Classification comparison
# ---------------------------------------------------------------------------

def _compare_classification(paths, logits_a, logits_b, conf_threshold, store_probs):
    probs_a = F.softmax(logits_a, dim=1).cpu()   # (B, C)
    probs_b = F.softmax(logits_b, dim=1).cpu()

    conf_a, pred_a = probs_a.max(dim=1)           # (B,)
    conf_b, pred_b = probs_b.max(dim=1)

    rows = []
    for i, path in enumerate(paths):
        pa = int(pred_a[i])
        pb = int(pred_b[i])
        ca = float(conf_a[i])
        cb = float(conf_b[i])

        # Shannon entropy: H = -Σ p log p  (nats)
        p_a = probs_a[i]
        p_b = probs_b[i]
        p_a_safe = p_a.clamp(min=1e-10)
        p_b_safe = p_b.clamp(min=1e-10)
        ent_a = float(-torch.sum(p_a_safe * p_a_safe.log()))
        ent_b = float(-torch.sum(p_b_safe * p_b_safe.log()))

        # Margin score: diff between top-1 and top-2 softmax probs
        sorted_a = p_a.sort(descending=True).values
        sorted_b = p_b.sort(descending=True).values
        margin_a = float(sorted_a[0] - sorted_a[1]) if len(sorted_a) > 1 else float(sorted_a[0])
        margin_b = float(sorted_b[0] - sorted_b[1]) if len(sorted_b) > 1 else float(sorted_b[0])

        row = {
            "image_path":       path,
            "pred_a":           pa,
            "confidence_a":     ca,
            "pred_b":           pb,
            "confidence_b":     cb,
            "agrees":           int(pa == pb),
            "confidence_delta": abs(ca - cb),
            "entropy_a":        ent_a,
            "entropy_b":        ent_b,
            "margin_a":         margin_a,
            "margin_b":         margin_b,
        }
        if store_probs:
            row["probs_a"] = p_a.numpy().tolist()
            row["probs_b"] = p_b.numpy().tolist()

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Segmentation comparison
# ---------------------------------------------------------------------------

def _compare_segmentation(paths, logits_a, logits_b, iou_threshold):
    if logits_a.shape[1] == 1:
        probs_a = torch.sigmoid(logits_a).squeeze(1).cpu()    # (B, H, W) ∈ [0,1]
        probs_b = torch.sigmoid(logits_b).squeeze(1).cpu()
        masks_a = (probs_a > 0.5).bool()
        masks_b = (probs_b > 0.5).bool()
    else:
        masks_a = logits_a.argmax(dim=1).cpu().bool()
        masks_b = logits_b.argmax(dim=1).cpu().bool()
        probs_a = F.softmax(logits_a, dim=1)[:, 1, :, :].cpu()
        probs_b = F.softmax(logits_b, dim=1)[:, 1, :, :].cpu()

    rows = []
    for i, path in enumerate(paths):
        ma = masks_a[i].numpy()
        mb = masks_b[i].numpy()

        iou        = _mask_iou(ma, mb)
        dice       = _mask_dice(ma, mb)
        b_iou      = _boundary_iou(ma, mb)

        # Use defect fraction as a proxy confidence for segmentation
        conf_a = float(probs_a[i].mean())
        conf_b = float(probs_b[i].mean())

        rows.append({
            "image_path":              path,
            "confidence_a":            conf_a,
            "confidence_b":            conf_b,
            "confidence_delta":        abs(conf_a - conf_b),
            "mask_defect_fraction_a":  float(ma.mean()),
            "mask_defect_fraction_b":  float(mb.mean()),
            "iou":                     iou,
            "dice":                    dice,
            "boundary_iou":            b_iou,
            "agrees":                  int(iou >= iou_threshold),
        })
    return rows


# ---------------------------------------------------------------------------
# Mask metrics
# ---------------------------------------------------------------------------

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two boolean masks."""
    intersection = int((a & b).sum())
    union        = int((a | b).sum())
    return 1.0 if union == 0 else float(intersection / union)


def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice / F1 coefficient: 2|A∩B| / (|A| + |B|)."""
    intersection = int((a & b).sum())
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else float(2 * intersection / total)


def _boundary_iou(a: np.ndarray, b: np.ndarray, dilation: int = 1) -> float:
    """Boundary IoU: IoU computed only on the mask boundary pixels.

    Boundary pixels are those that differ from at least one 4-connected neighbor.
    This metric emphasises boundary agreement over area agreement.
    """
    def _get_boundary(mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2 or mask.sum() == 0:
            return mask.astype(bool)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=dilation)
        return mask & ~eroded

    ba = _get_boundary(a)
    bb = _get_boundary(b)
    return _mask_iou(ba, bb)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    df: pd.DataFrame,
    model: str = "a",
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — requires ground-truth labels.

    If the DataFrame has a 'label' column, computes ECE by binning predictions
    by confidence and comparing mean confidence to mean accuracy within each bin.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) − conf(B_b)|

    If no 'label' column is present, returns 0.0 (not computable).

    Args:
        df:     A/B comparison DataFrame (with 'probs_a'/'probs_b' if store_probs=True).
        model:  'a' or 'b'.
        n_bins: Number of equal-width confidence bins (default 10).

    Returns:
        ECE value in [0, 1].
    """
    pred_col = f"pred_{model}"
    conf_col = f"confidence_{model}"

    if pred_col not in df.columns or conf_col not in df.columns:
        return 0.0
    if "label" not in df.columns:
        return 0.0   # ground truth not available

    preds = df[pred_col].values
    confs = df[conf_col].values
    labels = df["label"].values
    n = len(preds)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        in_bin = (confs > b_low) & (confs <= b_high)
        n_bin = int(in_bin.sum())
        if n_bin == 0:
            continue
        acc  = float((preds[in_bin] == labels[in_bin]).mean())
        conf = float(confs[in_bin].mean())
        ece += (n_bin / n) * abs(acc - conf)

    return float(ece)


def confidence_by_band(
    df: pd.DataFrame,
    n_bands: int = 5,
) -> pd.DataFrame:
    """Compute disagreement rate and entropy by confidence band.

    Groups images into equal-width confidence bands for Model A and reports
    the disagreement rate and mean entropy in each band.

    Args:
        df:      A/B comparison DataFrame.
        n_bands: Number of confidence bands (default 5 → 0–20%, 20–40%, …).

    Returns:
        DataFrame with columns: band, conf_min, conf_max, n, disagreement_rate,
        mean_entropy_a, mean_entropy_b.
    """
    df = df.copy()
    edges = np.linspace(0.0, 1.0, n_bands + 1)

    # Check if equal-width bands are degenerate (< 2 populated)
    test_cut = pd.cut(df["confidence_a"], bins=edges, include_lowest=True)
    n_populated = int(test_cut.value_counts().gt(0).sum())
    if n_populated < 2:
        # Adaptive mode: quantile-based bands
        edges = np.unique(np.percentile(df["confidence_a"].values, np.linspace(0, 100, n_bands + 1)))
        n_bands = len(edges) - 1

    labels = [f"{lo:.0%}-{hi:.0%}" for lo, hi in zip(edges[:-1], edges[1:])]
    df["band"] = pd.cut(
        df["confidence_a"], bins=edges,
        labels=labels,
        include_lowest=True,
    )

    records = []
    for band in df["band"].cat.categories:
        g = df[df["band"] == band]
        idx = list(df["band"].cat.categories).index(band)
        rec = {
            "band":             str(band),
            "conf_min":         float(edges[idx]),
            "conf_max":         float(edges[idx + 1]),
            "n":                int(len(g)),
            "disagreement_rate": float(1 - g["agrees"].mean()) if len(g) > 0 else 0.0,
        }
        if "entropy_a" in df.columns:
            rec["mean_entropy_a"] = float(g["entropy_a"].mean()) if len(g) > 0 else 0.0
        if "entropy_b" in df.columns:
            rec["mean_entropy_b"] = float(g["entropy_b"].mean()) if len(g) > 0 else 0.0
        records.append(rec)

    return pd.DataFrame(records)

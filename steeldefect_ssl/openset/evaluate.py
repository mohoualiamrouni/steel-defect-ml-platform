"""Open-set evaluation metrics + the PatchCore baseline bridge (P1).

``auroc_aupr_fpr95`` is the standard OOD-detection scoreboard.
``baseline_patchcore_score`` reuses P1's ``PatchCoreDetector`` (fit on the
``clean`` family, score the stream) so the SSL-feature scorer can be compared
against the pixel-patch anomaly detector at *family-level* novelty.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc_aupr_fpr95(scores: np.ndarray, is_novel: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(is_novel).astype(int)
    if y.min() == y.max():            # degenerate (all/none novel)
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    auroc = float(roc_auc_score(y, scores))
    aupr = float(average_precision_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    idx = min(idx, len(fpr) - 1)
    return {"auroc": auroc, "aupr": aupr, "fpr95": float(fpr[idx])}


def baseline_patchcore_score(clean_paths: List[str], stream_paths: List[str],
                             device: str = "cpu",
                             image_size: int = 224) -> Optional[np.ndarray]:
    """PatchCore novelty scores for the stream (fit on clean). Returns None if
    the P1 detector / torchvision weights are unavailable."""
    try:
        from PIL import Image
        from steeldefect_active_learning.active_learning.anomaly import PatchCoreDetector

        def _load_rgb(paths):
            return [Image.open(p).convert("RGB").resize((image_size, image_size))
                    for p in paths]

        det = PatchCoreDetector(backbone="resnet18", device=device, image_size=image_size)
        det.fit(_load_rgb(clean_paths))
        return np.asarray(det.score(_load_rgb(stream_paths)), dtype=float)
    except Exception as exc:  # pragma: no cover - optional baseline
        print(f"[openset] PatchCore baseline skipped: {exc}")
        return None

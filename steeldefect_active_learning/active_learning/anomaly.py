"""
Anomaly detection for zero-shot / unsupervised defect discovery.

Two production-quality detectors based on recent CVPR/ICCV work:

PatchCore (Roth et al. CVPR 2022)
----------------------------------
Builds a memory bank M of patch-level feature vectors from normal images.
Reduces it to a coreset M_C (≈1% of M) via greedy farthest-point sampling
with random projection for speed. At inference computes the softmax-weighted
nearest-neighbour distance as anomaly score.

    # Training (fit on normal images only):
    For each normal image x:
      patches = extract from {layer2, layer3} of ResNet
      M = M ∪ patches

    # Greedy coreset reduction:
    M_C = greedy_farthest_point(M, ratio=0.01, proj_dim=128)

    # Inference for test image:
    For each patch p_test:
      p* = NN(p_test, M_C)
      N_b = b-NN(p*, M_C)
      w = exp(||p_test - p*||) / Σ_{n∈N_b} exp(||p_test - n||)
      s_patch = w · ||p_test - p*||
    s_image = max over patches

PaDiM (Defard et al. 2021)
----------------------------
Models patch-level distributions with per-position multivariate Gaussians.
Uses features from multiple CNN layers concatenated at each spatial position.

    # Training:
    For each patch position (i,j):
      X_ij = concat of features from 3 layers at (i,j) over N normal images
      μ_ij = mean(X_ij)
      Σ_ij = cov(X_ij) + ε·I  (regularised)

    # Inference:
    For each test patch position (i,j):
      score_ij = sqrt((x_ij - μ_ij)^T Σ_ij^{-1} (x_ij - μ_ij))
    s_image = max over positions

References
----------
- Roth et al. 2022: "Towards Total Recall in Industrial Anomaly Detection"
  https://arxiv.org/abs/2106.08265
- Defard et al. 2021: "PaDiM: a Patch Distribution Modeling Framework for
  Anomaly Detection and Localization"
  https://arxiv.org/abs/2011.10678
- Bergmann et al. 2019: MVTec AD dataset (benchmark used by both papers)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extractor (shared by both detectors)
# ---------------------------------------------------------------------------

class _CNNFeatureExtractor:
    """Lightweight wrapper to extract intermediate CNN feature maps.

    Uses torchvision ResNet18 by default (fast, GPU-optional).
    Hooks into layer2 and layer3 (stride 8 and stride 16).

    Args:
        backbone:    'resnet18' | 'resnet50' (default: 'resnet18')
        device:      'cpu' | 'cuda' (default: 'cpu')
        image_size:  Resize shorter side to this before feature extraction.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        self.device = device
        self.image_size = image_size
        self._setup(backbone)

    def _setup(self, backbone: str) -> None:
        try:
            import torch
            import torchvision.models as tvm
            import torchvision.transforms as T
        except ImportError as e:
            raise ImportError(
                "anomaly.py requires torch + torchvision: pip install torch torchvision"
            ) from e

        self._torch = torch
        model_fn = getattr(tvm, backbone, None)
        if model_fn is None:
            raise ValueError(f"Unknown backbone: {backbone}")

        model = model_fn(pretrained=True)
        model.eval()
        model = model.to(self.device)

        self._features: Dict[str, "torch.Tensor"] = {}

        def _hook(name):
            def _fn(module, inp, out):
                self._features[name] = out
            return _fn

        model.layer2.register_forward_hook(_hook("layer2"))
        model.layer3.register_forward_hook(_hook("layer3"))
        self._model = model

        self._transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image) -> Dict[str, np.ndarray]:
        """Extract feature maps from a PIL Image or ndarray.

        Returns:
            Dict with keys 'layer2' and 'layer3', each a np.ndarray
            of shape (C, H, W).
        """
        import torch
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image.astype(np.uint8))

        x = self._transform(image).unsqueeze(0).to(self.device)

        self._features.clear()
        with torch.no_grad():
            self._model(x)

        return {
            k: v.squeeze(0).cpu().numpy()
            for k, v in self._features.items()
        }

    def extract_patches(
        self,
        image,
        stride: int = 1,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract and concatenate patch embeddings from layer2 + layer3.

        Resizes layer3 to match layer2 spatially, then concatenates along
        channel dimension, and flattens spatial positions to a patch matrix.

        Args:
            image:   PIL Image or np.ndarray (H, W, 3).
            stride:  Spatial stride for downsampling patches (default 1).

        Returns:
            patches: np.ndarray shape (H'*W', C2+C3)
            spatial: (H', W') spatial grid dimensions
        """
        feats = self.extract(image)
        f2 = feats["layer2"]  # (C2, H2, W2)
        f3 = feats["layer3"]  # (C3, H3, W3)  — typically half size of f2

        C2, H2, W2 = f2.shape
        C3, H3, W3 = f3.shape

        # Upsample layer3 to match layer2 spatial size using nearest interpolation
        if H3 != H2 or W3 != W2:
            try:
                import torch
                import torch.nn.functional as F
                f3_t = torch.tensor(f3).unsqueeze(0)  # (1, C3, H3, W3)
                f3_t = F.interpolate(f3_t, size=(H2, W2), mode="nearest")
                f3 = f3_t.squeeze(0).numpy()
            except ImportError:
                # Fallback: simple repeat upsampling
                f3 = np.repeat(np.repeat(f3, H2 // H3, axis=1), W2 // W3, axis=2)
                f3 = f3[:, :H2, :W2]

        # Concatenate channels: (C2+C3, H2, W2)
        combined = np.concatenate([f2, f3], axis=0)

        # Apply stride
        combined = combined[:, ::stride, ::stride]
        _, H_out, W_out = combined.shape

        # Flatten to patch matrix: (H_out*W_out, C2+C3)
        patches = combined.reshape(C2 + C3, -1).T  # (N_patches, D)

        return patches, (H_out, W_out)


# ---------------------------------------------------------------------------
# Helper: greedy farthest-point coreset
# ---------------------------------------------------------------------------

def _greedy_coreset(
    M: np.ndarray,
    coreset_ratio: float = 0.01,
    proj_dim: int = 128,
    random_state: int = 42,
) -> np.ndarray:
    """Greedy farthest-point coreset with random projection (Roth et al. 2022).

    Args:
        M:            Feature matrix, shape (N, D).
        coreset_ratio: Fraction of M to keep (default 0.01 = 1%).
        proj_dim:     Random projection dimension for distance speed-up.
        random_state: Seed for random projection.

    Returns:
        Coreset M_C: np.ndarray shape (k, D) where k = ceil(N * ratio).
    """
    N, D = M.shape
    k = max(1, int(np.ceil(N * coreset_ratio)))

    if k >= N:
        return M

    # Random projection for faster distances
    rng = np.random.default_rng(random_state)
    P = rng.standard_normal((D, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
    M_proj = (M @ P).astype(np.float32)  # (N, proj_dim)

    selected = [int(rng.integers(N))]
    # D_min[i] = min distance from M_proj[i] to any selected point
    D_min = np.full(N, np.inf, dtype=np.float32)

    for _ in range(k - 1):
        last = selected[-1]
        diff = M_proj - M_proj[last]
        dists = np.sum(diff ** 2, axis=1)
        D_min = np.minimum(D_min, dists)
        D_min[selected] = -np.inf  # already selected
        next_idx = int(np.argmax(D_min))
        selected.append(next_idx)

    return M[selected]


# ---------------------------------------------------------------------------
# PatchCore
# ---------------------------------------------------------------------------

class PatchCoreDetector:
    """PatchCore anomaly detector (Roth et al. CVPR 2022).

    Train on normal images only; at inference returns a per-image anomaly
    score (higher = more anomalous).

    Example
    -------
    detector = PatchCoreDetector()
    detector.fit(normal_images)          # list of PIL Images or np.ndarrays
    scores = detector.score(test_images)  # List[float]
    threshold = detector.threshold(fpr=0.05)  # from fit distribution
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        coreset_ratio: float = 0.01,
        proj_dim: int = 128,
        n_neighbors: int = 9,
        image_size: int = 224,
        random_state: int = 42,
    ) -> None:
        self.backbone = backbone
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.proj_dim = proj_dim
        self.n_neighbors = n_neighbors
        self.image_size = image_size
        self.random_state = random_state

        self._extractor: Optional[_CNNFeatureExtractor] = None
        self._memory_bank: Optional[np.ndarray] = None
        self._fit_scores: Optional[np.ndarray] = None

    def fit(self, images: Sequence) -> "PatchCoreDetector":
        """Build memory bank from normal images.

        Args:
            images: Sequence of PIL Images or np.ndarray (H,W,3).

        Returns:
            self
        """
        logger.info("PatchCore: extracting patches from %d normal images…", len(images))
        self._extractor = _CNNFeatureExtractor(
            self.backbone, self.device, self.image_size
        )

        all_patches = []
        for img in images:
            patches, _ = self._extractor.extract_patches(img)
            all_patches.append(patches)

        M = np.concatenate(all_patches, axis=0).astype(np.float32)
        logger.info("PatchCore: memory bank size before coreset: %d patches", len(M))

        self._memory_bank = _greedy_coreset(
            M, self.coreset_ratio, self.proj_dim, self.random_state
        )
        logger.info(
            "PatchCore: coreset size: %d patches (%.2f%%)",
            len(self._memory_bank),
            100.0 * len(self._memory_bank) / len(M),
        )

        # Compute fit-distribution scores for thresholding
        self._fit_scores = np.array([self._score_image(img) for img in images])
        return self

    def score(self, images: Sequence) -> List[float]:
        """Compute anomaly scores for a batch of images.

        Args:
            images: Sequence of PIL Images or np.ndarray (H,W,3).

        Returns:
            List of float anomaly scores (higher = more anomalous).
        """
        if self._memory_bank is None:
            raise RuntimeError("Call fit() before score()")
        return [self._score_image(img) for img in images]

    def threshold(self, fpr: float = 0.05) -> float:
        """Return score threshold corresponding to given false positive rate.

        Uses the distribution of scores on training (normal) images.
        threshold = percentile(fit_scores, (1 - fpr) * 100)

        Args:
            fpr: Desired false positive rate (default 0.05 = 5%).

        Returns:
            Score threshold.
        """
        if self._fit_scores is None:
            raise RuntimeError("Call fit() first to compute threshold")
        return float(np.percentile(self._fit_scores, (1 - fpr) * 100))

    # ------------------------------------------------------------------
    # Per-class FPR calibration (fixes report limitation L5 / mention 4/6)
    # ------------------------------------------------------------------
    # A single global percentile threshold is calibrated on a pool dominated by
    # the majority classes (clean ≫ rare structural defects). The effective FPR
    # on rare classes then drifts away from the nominal target. Calibrating one
    # threshold per class on a labelled validation set flattens the achieved
    # per-class FPR; sparse classes fall back to the global threshold.

    def fit_per_class(self, images: Sequence, class_ids: Sequence[int],
                      min_n: int = 30) -> "PatchCoreDetector":
        """Store per-class score distributions from a labelled validation set.

        Must be called after :meth:`fit` (the memory bank must exist). *images*
        and *class_ids* are aligned; scores are grouped by class so that
        :meth:`threshold_per_class` can target the nominal FPR within each class.
        """
        if self._memory_bank is None:
            raise RuntimeError("Call fit() before fit_per_class()")
        scores = np.asarray([self._score_image(img) for img in images], dtype=float)
        class_ids = np.asarray(list(class_ids))
        self._fit_scores_by_class = {}
        for c in np.unique(class_ids):
            self._fit_scores_by_class[int(c)] = scores[class_ids == c]
        self._per_class_min_n = int(min_n)
        return self

    def threshold_per_class(self, fpr: float = 0.05) -> Dict[int, float]:
        """Return ``{class_id: threshold}`` at the nominal *fpr* per class.

        Classes with fewer than ``min_n`` calibration samples fall back to the
        global threshold (insufficient data for a stable per-class percentile).
        """
        by_class = getattr(self, "_fit_scores_by_class", None)
        if not by_class:
            raise RuntimeError("Call fit_per_class() first")
        global_thr = self.threshold(fpr)
        min_n = getattr(self, "_per_class_min_n", 30)
        out: Dict[int, float] = {}
        for c, sc in by_class.items():
            if len(sc) >= min_n:
                out[int(c)] = float(np.percentile(sc, (1 - fpr) * 100))
            else:
                out[int(c)] = global_thr
        return out

    def is_anomaly(self, score: float, class_id: Optional[int] = None,
                   fpr: float = 0.05) -> bool:
        """Flag *score* as anomalous against the class-specific threshold.

        Falls back to the global threshold when no per-class calibration exists
        or the class is unknown.
        """
        by_class = getattr(self, "_fit_scores_by_class", None)
        if class_id is not None and by_class:
            thr = self.threshold_per_class(fpr).get(int(class_id), self.threshold(fpr))
        else:
            thr = self.threshold(fpr)
        return bool(score > thr)

    def _score_image(self, image) -> float:
        """Score a single image."""
        patches, (H, W) = self._extractor.extract_patches(image)
        M_C = self._memory_bank

        # For each test patch: find nearest neighbour in M_C
        # Use batched distance computation to avoid OOM
        BATCH = 1024
        scores = np.zeros(len(patches), dtype=np.float32)

        for start in range(0, len(patches), BATCH):
            p_batch = patches[start:start + BATCH]  # (B, D)
            # Squared distances: (B, |M_C|)
            diff = p_batch[:, np.newaxis, :] - M_C[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2)  # (B, |M_C|)

            # Nearest neighbour index
            nn_idx = sq_dists.argmin(axis=1)  # (B,)
            nn_dists = np.sqrt(sq_dists[np.arange(len(p_batch)), nn_idx])  # (B,)

            # Softmax re-weighting: w ∝ exp(dist to NN's neighbors in M_C)
            b_count = self.n_neighbors
            for bi in range(len(p_batch)):
                n_star = nn_idx[bi]
                diff_nb = M_C - M_C[n_star]
                nb_dists = np.sqrt(np.sum(diff_nb ** 2, axis=1))
                top_b = np.argsort(nb_dists)[:b_count]
                nb_exp = np.exp(nb_dists[top_b])
                w = np.exp(nn_dists[bi]) / (nb_exp.sum() + 1e-12)
                scores[start + bi] = w * nn_dists[bi]

        return float(scores.max())

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the fitted detector to disk."""
        state = {
            "backbone": self.backbone,
            "device": self.device,
            "coreset_ratio": self.coreset_ratio,
            "proj_dim": self.proj_dim,
            "n_neighbors": self.n_neighbors,
            "image_size": self.image_size,
            "memory_bank": self._memory_bank,
            "fit_scores": self._fit_scores,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved PatchCoreDetector → %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PatchCoreDetector":
        """Load a detector saved with save()."""
        with open(str(path), "rb") as f:
            state = pickle.load(f)
        det = cls(
            backbone=state["backbone"],
            device=state["device"],
            coreset_ratio=state["coreset_ratio"],
            proj_dim=state["proj_dim"],
            n_neighbors=state["n_neighbors"],
            image_size=state["image_size"],
        )
        det._memory_bank = state["memory_bank"]
        det._fit_scores = state["fit_scores"]
        det._extractor = _CNNFeatureExtractor(det.backbone, det.device, det.image_size)
        logger.info("Loaded PatchCoreDetector from %s", path)
        return det


# ---------------------------------------------------------------------------
# PaDiM
# ---------------------------------------------------------------------------

class PaDiMDetector:
    """PaDiM anomaly detector (Defard et al. 2021).

    Models per-patch-position feature distributions as multivariate Gaussians
    and uses Mahalanobis distance at inference.

    Example
    -------
    detector = PaDiMDetector()
    detector.fit(normal_images)
    scores = detector.score(test_images)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        image_size: int = 224,
        eps: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.backbone = backbone
        self.device = device
        self.image_size = image_size
        self.eps = eps
        self.random_state = random_state

        self._extractor: Optional[_CNNFeatureExtractor] = None
        self._means: Optional[np.ndarray] = None      # (H*W, D)
        self._cov_invs: Optional[np.ndarray] = None   # (H*W, D, D)
        self._spatial: Optional[Tuple[int, int]] = None

    def fit(self, images: Sequence) -> "PaDiMDetector":
        """Fit per-patch-position Gaussians from normal images.

        Args:
            images: Sequence of PIL Images or np.ndarray (H,W,3).

        Returns:
            self
        """
        logger.info("PaDiM: fitting on %d normal images…", len(images))
        self._extractor = _CNNFeatureExtractor(
            self.backbone, self.device, self.image_size
        )

        patch_sets: List[np.ndarray] = []
        spatial = None

        for img in images:
            patches, sp = self._extractor.extract_patches(img)
            patch_sets.append(patches)  # (N_patches, D)
            spatial = sp

        self._spatial = spatial
        N_patches = patch_sets[0].shape[0]
        D = patch_sets[0].shape[1]

        # Stack all images: (N_images, N_patches, D)
        X = np.stack(patch_sets, axis=0).astype(np.float64)  # (N, P, D)
        N = X.shape[0]

        logger.info(
            "PaDiM: fitting %d patch positions, dim=%d, n_images=%d",
            N_patches, D, N,
        )

        # Per-position mean and regularised covariance
        means = X.mean(axis=0)  # (P, D)
        X_centered = X - means[np.newaxis, :, :]  # (N, P, D)

        cov_invs = np.zeros((N_patches, D, D), dtype=np.float32)
        I_eps = self.eps * np.eye(D, dtype=np.float64)

        for pos in range(N_patches):
            Z = X_centered[:, pos, :]  # (N, D)
            cov = (Z.T @ Z) / max(N - 1, 1) + I_eps  # (D, D)
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            cov_invs[pos] = cov_inv.astype(np.float32)

        self._means = means.astype(np.float32)
        self._cov_invs = cov_invs
        logger.info("PaDiM: fitting complete.")
        return self

    def score(self, images: Sequence) -> List[float]:
        """Compute per-image Mahalanobis anomaly scores.

        Args:
            images: Sequence of PIL Images or np.ndarray (H,W,3).

        Returns:
            List of float scores (higher = more anomalous).
        """
        if self._means is None:
            raise RuntimeError("Call fit() before score()")
        return [self._score_image(img) for img in images]

    def _score_image(self, image) -> float:
        """Score a single image via max Mahalanobis distance over patches."""
        patches, _ = self._extractor.extract_patches(image)
        patches = patches.astype(np.float32)

        N_patches = patches.shape[0]
        scores = np.zeros(N_patches, dtype=np.float32)

        for pos in range(min(N_patches, len(self._means))):
            delta = patches[pos] - self._means[pos]  # (D,)
            cov_inv = self._cov_invs[pos]  # (D, D)
            # Mahalanobis: sqrt(delta^T Σ^{-1} delta)
            maha_sq = float(delta @ cov_inv @ delta)
            scores[pos] = np.sqrt(max(0.0, maha_sq))

        return float(scores.max())

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the fitted detector."""
        state = {
            "backbone": self.backbone,
            "device": self.device,
            "image_size": self.image_size,
            "eps": self.eps,
            "means": self._means,
            "cov_invs": self._cov_invs,
            "spatial": self._spatial,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved PaDiMDetector → %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PaDiMDetector":
        """Load a detector saved with save()."""
        with open(str(path), "rb") as f:
            state = pickle.load(f)
        det = cls(
            backbone=state["backbone"],
            device=state["device"],
            image_size=state["image_size"],
            eps=state["eps"],
        )
        det._means = state["means"]
        det._cov_invs = state["cov_invs"]
        det._spatial = state["spatial"]
        det._extractor = _CNNFeatureExtractor(det.backbone, det.device, det.image_size)
        logger.info("Loaded PaDiMDetector from %s", path)
        return det

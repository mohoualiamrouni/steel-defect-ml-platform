"""Coordinate loading of CNN features, segmentation masks, and metadata.

``DefectDataset`` is the single entry-point for obtaining a consistent,
aligned view of every data source the pipeline consumes.  It gracefully
degrades when masks or metadata are missing and can auto-generate
synthetic data for quick experimentation.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from defect_clustering.config import DefectClusteringConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segmentation feature helpers
# ---------------------------------------------------------------------------

_SEG_FEATURE_NAMES: List[str] = [
    # Per-mask spatial / shape statistics
    "area_pixels",
    "area_fraction",
    "centroid_y",
    "centroid_x",
    "bbox_height",
    "bbox_width",
    "aspect_ratio",
    "perimeter",
    "circularity",
    "solidity",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "extent",
    "equivalent_diameter",
    # Intensity-weighted stats (if raw intensities available)
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "min_intensity",
    # Hu moments (7)
    "hu_0",
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
    # Texture / gradient statistics (simple)
    "edge_pixel_count",
    "edge_density",
    # Zonal stats
    "top_half_area_frac",
    "bottom_half_area_frac",
    "left_half_area_frac",
    "right_half_area_frac",
    # Component counts
    "n_components",
    "largest_component_area_frac",
    # Additional shape descriptors
    "convex_area",
    "convex_perimeter",
    "rectangularity",
    "compactness",
    "elongation",
    "roughness",
    # Skeleton features
    "skeleton_length",
    "skeleton_branch_points",
    # Boundary curvature stats
    "mean_curvature",
    "std_curvature",
    "max_curvature",
    "min_curvature",
    # Filled area ratio (area / filled_area — convex-hull fill)
    "filled_area_ratio",
]
"""48 segmentation-derived features extracted from each binary mask."""


def _extract_seg_features_from_mask(
    mask: np.ndarray,
    aggregation: str = "mean_max",
) -> np.ndarray:
    """Extract a fixed-length feature vector from a single binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary (or uint8 0/255) mask.
    aggregation : str
        ``"mean"`` returns one vector per mask; ``"mean_max"`` concatenates
        mean and max pooling across connected components.

    Returns
    -------
    np.ndarray
        Feature vector of shape ``(48,)`` for ``"mean"`` or ``(48,)`` for
        ``"mean_max"`` (mean and max are blended per feature, not doubled).
    """
    try:
        from scipy import ndimage
        from skimage import measure, morphology  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "scikit-image / scipy not installed — returning zero vector for "
            "segmentation features."
        )
        return np.zeros(len(_SEG_FEATURE_NAMES), dtype=np.float32)

    # Normalise mask to binary {0, 1}.
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    h, w = mask.shape
    total_pixels = h * w

    # Label connected components.
    labelled, n_components = ndimage.label(mask)
    regions = measure.regionprops(labelled)

    if n_components == 0 or len(regions) == 0:
        return np.zeros(len(_SEG_FEATURE_NAMES), dtype=np.float32)

    component_features: List[np.ndarray] = []

    for region in regions:
        feats: List[float] = []

        # Spatial / shape --------------------------------------------------
        feats.append(float(region.area))                       # area_pixels
        feats.append(float(region.area) / total_pixels)        # area_fraction
        feats.append(float(region.centroid[0]) / h)            # centroid_y (normalised)
        feats.append(float(region.centroid[1]) / w)            # centroid_x
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]
        feats.append(float(bbox_h) / h)                        # bbox_height
        feats.append(float(bbox_w) / w)                        # bbox_width
        feats.append(float(bbox_h) / max(bbox_w, 1))           # aspect_ratio
        feats.append(float(region.perimeter))                   # perimeter
        perimeter = max(region.perimeter, 1e-8)
        feats.append(4.0 * np.pi * region.area / (perimeter ** 2))  # circularity
        feats.append(float(region.solidity))                    # solidity
        feats.append(float(region.eccentricity))                # eccentricity
        feats.append(float(region.major_axis_length) / max(h, w))  # major_axis_length
        feats.append(float(region.minor_axis_length) / max(h, w))  # minor_axis_length
        feats.append(float(region.orientation))                 # orientation
        feats.append(float(region.extent))                      # extent
        feats.append(float(region.equivalent_diameter) / max(h, w))  # equivalent_diameter

        # Intensity stats (mask-only — treat mask as intensity) ------------
        component_mask = (labelled == region.label).astype(np.float32)
        feats.append(float(component_mask.mean()))              # mean_intensity
        feats.append(float(component_mask.std()))               # std_intensity
        feats.append(float(component_mask.max()))               # max_intensity
        feats.append(float(component_mask.min()))               # min_intensity

        # Hu moments -------------------------------------------------------
        hu = measure.moments_hu(measure.moments_central(component_mask))
        # Log-transform for numerical stability.
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
        for i in range(7):
            feats.append(float(hu_log[i]))

        # Edge / boundary ---------------------------------------------------
        eroded = ndimage.binary_erosion(component_mask.astype(bool))
        edge = component_mask.astype(bool) & ~eroded
        edge_count = int(edge.sum())
        feats.append(float(edge_count))                         # edge_pixel_count
        feats.append(float(edge_count) / max(region.area, 1))  # edge_density

        # Zonal stats -------------------------------------------------------
        mid_h, mid_w = h // 2, w // 2
        feats.append(float(component_mask[:mid_h, :].sum()) / max(region.area, 1))  # top_half
        feats.append(float(component_mask[mid_h:, :].sum()) / max(region.area, 1))  # bottom_half
        feats.append(float(component_mask[:, :mid_w].sum()) / max(region.area, 1))  # left_half
        feats.append(float(component_mask[:, mid_w:].sum()) / max(region.area, 1))  # right_half

        # Component counts (same for all — filled once) ---------------------
        feats.append(float(n_components))                       # n_components
        largest_area = max(r.area for r in regions)
        feats.append(float(largest_area) / total_pixels)        # largest_component_area_frac

        # Additional shape descriptors --------------------------------------
        feats.append(float(region.area) / max(float(region.convex_area), 1))  # convex_area (ratio)
        convex_perim = perimeter  # approximate
        feats.append(float(convex_perim))                       # convex_perimeter
        feats.append(float(region.area) / max(bbox_h * bbox_w, 1))  # rectangularity
        feats.append(float(region.area) / (perimeter ** 2 + 1e-8))  # compactness
        feats.append(float(region.major_axis_length) / max(region.minor_axis_length, 1e-8))  # elongation
        feats.append(perimeter / max(float(region.equivalent_diameter) * np.pi, 1e-8))  # roughness

        # Skeleton features -------------------------------------------------
        try:
            skel = morphology.skeletonize(component_mask.astype(bool))
            skel_length = float(skel.sum())
            # Branch points: pixels with > 2 neighbours in the skeleton.
            kernel = np.ones((3, 3), dtype=int)
            neighbour_count = ndimage.convolve(skel.astype(int), kernel, mode="constant")
            branch_pts = float(((neighbour_count > 3) & skel).sum())
        except Exception:
            skel_length = 0.0
            branch_pts = 0.0
        feats.append(skel_length)                               # skeleton_length
        feats.append(branch_pts)                                # skeleton_branch_points

        # Boundary curvature stats ------------------------------------------
        contours = measure.find_contours(component_mask, 0.5)
        if contours:
            contour = contours[0]
            dx = np.gradient(contour[:, 1])
            dy = np.gradient(contour[:, 0])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (
                (dx ** 2 + dy ** 2) ** 1.5 + 1e-8
            )
            feats.append(float(np.mean(curvature)))             # mean_curvature
            feats.append(float(np.std(curvature)))              # std_curvature
            feats.append(float(np.max(curvature)))              # max_curvature
            feats.append(float(np.min(curvature)))              # min_curvature
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        # Filled area ratio: area / filled_area (convex-hull fill)
        feats.append(
            float(region.area) / max(float(region.filled_area), 1.0)
        )

        component_features.append(np.array(feats, dtype=np.float32))

    # Stack across components and aggregate.
    stacked = np.stack(component_features, axis=0)  # (n_components, 48)

    if aggregation == "mean":
        return stacked.mean(axis=0)
    elif aggregation == "mean_max":
        # Blend mean and max into a single 48-d vector (element-wise average
        # of the two pooling strategies).
        return ((stacked.mean(axis=0) + stacked.max(axis=0)) / 2.0).astype(
            np.float32
        )
    else:
        raise ValueError(f"Unknown seg_aggregation: {aggregation!r}")


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class DefectDataset:
    """Load and align CNN features, segmentation masks, and metadata.

    Can work with:

    * **Full data** — pickle + masks + metadata CSV.
    * **CNN features only** — pickle + metadata CSV, no masks.
    * **Synthetic data** — auto-generated when the specified paths do not
      exist (handy for testing and demos).

    Parameters
    ----------
    config : DefectClusteringConfig
        Pipeline configuration object.
    """

    def __init__(self, config: DefectClusteringConfig) -> None:
        self.config = config

        # Populated after ``load()``
        self._cnn_features: Optional[np.ndarray] = None
        self._seg_features: Optional[np.ndarray] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._image_ids: Optional[List[str]] = None
        self._masks: Optional[List[np.ndarray]] = None
        self._loaded: bool = False

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Load all data sources, aligned by ``image_id``.

        Returns
        -------
        dict
            ``'cnn_features'``
                ``(N, D_cnn)`` float32 array — typically ``(N, 960)``.
            ``'seg_features'``
                ``(N, 48)`` float32 array, or ``None`` if masks are
                unavailable.
            ``'metadata'``
                :class:`~pandas.DataFrame` with at least ``image_id``,
                ``defect_type``, and ``expert_group`` columns.
            ``'image_ids'``
                List of *N* string identifiers in the same order as the
                feature rows.
            ``'masks'``
                List of *N* ndarrays, or ``None`` if masks are unavailable.
        """
        # 1. CNN features ---------------------------------------------------
        cnn_path = self.config.cnn_features_pickle
        if os.path.isfile(cnn_path):
            logger.info("Loading CNN features from %s", cnn_path)
            with open(cnn_path, "rb") as fh:
                cnn_data = pickle.load(fh)
        else:
            logger.warning(
                "CNN features pickle not found at %s — generating synthetic "
                "features.",
                cnn_path,
            )
            cnn_data = self._generate_synthetic_cnn_features()

        # cnn_data may be:
        #   - dict  {image_id: np.ndarray}
        #   - tuple (ids, features_array)
        #   - np.ndarray (N, D) — image_ids inferred from metadata or index
        cnn_features, cnn_ids = self._normalise_cnn_data(cnn_data)

        # 2. Metadata -------------------------------------------------------
        meta_path = self.config.metadata_csv
        if meta_path and os.path.isfile(meta_path):
            logger.info("Loading metadata from %s", meta_path)
            metadata = pd.read_csv(meta_path)
            if "image_id" not in metadata.columns:
                raise ValueError(
                    "Metadata CSV must contain an 'image_id' column."
                )
            metadata["image_id"] = metadata["image_id"].astype(str)
        else:
            logger.warning(
                "Metadata CSV not found — creating synthetic metadata."
            )
            metadata = self._generate_synthetic_metadata(cnn_ids)

        # 3. Align by image_id ---------------------------------------------
        common_ids = self._align_ids(cnn_ids, metadata)
        id_to_cnn_idx = {img_id: idx for idx, img_id in enumerate(cnn_ids)}

        ordered_indices = [id_to_cnn_idx[img_id] for img_id in common_ids]
        cnn_features = cnn_features[ordered_indices]
        metadata = (
            metadata.set_index("image_id")
            .loc[common_ids]
            .reset_index()
        )

        # 4. Masks & segmentation features ----------------------------------
        masks: Optional[List[np.ndarray]] = None
        seg_features: Optional[np.ndarray] = None

        if (
            self.config.use_segmentation_features
            and self.config.masks_dir
            and os.path.isdir(self.config.masks_dir)
        ):
            logger.info("Loading masks from %s", self.config.masks_dir)
            masks = self._load_masks(common_ids)
            if masks is not None:
                seg_features = self._compute_seg_features(masks)
        else:
            if self.config.use_segmentation_features:
                logger.info(
                    "Masks directory not found — segmentation features will "
                    "be None."
                )

        # 5. Store ----------------------------------------------------------
        self._cnn_features = cnn_features.astype(np.float32)
        self._seg_features = (
            seg_features.astype(np.float32) if seg_features is not None else None
        )
        self._metadata = metadata
        self._image_ids = common_ids
        self._masks = masks
        self._loaded = True

        return {
            "cnn_features": self._cnn_features,
            "seg_features": self._seg_features,
            "metadata": self._metadata,
            "image_ids": self._image_ids,
            "masks": self._masks,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the loaded dataset.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called yet.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before requesting a summary.")

        lines: List[str] = ["=== DefectDataset Summary ==="]
        n = len(self._image_ids)  # type: ignore[arg-type]
        lines.append(f"  Samples (aligned): {n}")
        lines.append(
            f"  CNN feature dim  : {self._cnn_features.shape[1]}"  # type: ignore[union-attr]
        )
        if self._seg_features is not None:
            lines.append(f"  Seg feature dim  : {self._seg_features.shape[1]}")
        else:
            lines.append("  Seg features     : None (masks unavailable)")

        if self._metadata is not None:
            dt_col = self.config.defect_type_col
            eg_col = self.config.expert_group_col
            if dt_col in self._metadata.columns:
                n_types = self._metadata[dt_col].nunique()
                lines.append(f"  Defect types     : {n_types}")
            if eg_col in self._metadata.columns:
                n_groups = self._metadata[eg_col].nunique()
                lines.append(f"  Expert groups    : {n_groups}")

        lines.append(
            f"  Masks loaded     : {'Yes' if self._masks is not None else 'No'}"
        )
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _normalise_cnn_data(
        cnn_data: Any,
    ) -> tuple[np.ndarray, List[str]]:
        """Convert heterogeneous pickle formats to (features, ids)."""
        if isinstance(cnn_data, dict):
            ids = [str(k) for k in cnn_data.keys()]
            features = np.stack([cnn_data[k] for k in cnn_data], axis=0)
            return features, ids
        if isinstance(cnn_data, (list, tuple)) and len(cnn_data) == 2:
            ids_raw, features = cnn_data
            ids = [str(i) for i in ids_raw]
            return np.asarray(features), ids
        if isinstance(cnn_data, np.ndarray):
            n = cnn_data.shape[0]
            ids = [str(i) for i in range(n)]
            return cnn_data, ids
        if isinstance(cnn_data, pd.DataFrame):
            if "image_id" in cnn_data.columns:
                ids = cnn_data["image_id"].astype(str).tolist()
                features = cnn_data.drop(columns=["image_id"]).values
            else:
                ids = [str(i) for i in range(len(cnn_data))]
                features = cnn_data.values
            return features, ids
        raise TypeError(
            f"Unsupported CNN features format: {type(cnn_data).__name__}"
        )

    def _align_ids(
        self,
        cnn_ids: List[str],
        metadata: pd.DataFrame,
    ) -> List[str]:
        """Return the intersection of CNN ids and metadata ids, ordered."""
        meta_ids = set(metadata["image_id"].astype(str))
        cnn_set = set(cnn_ids)
        common = sorted(cnn_set & meta_ids)

        if not common:
            raise ValueError(
                "No overlapping image_ids between CNN features and metadata. "
                f"CNN sample: {cnn_ids[:5]}, meta sample: "
                f"{metadata['image_id'].head().tolist()}"
            )

        n_dropped_cnn = len(cnn_ids) - len(common)
        n_dropped_meta = len(meta_ids) - len(common)
        if n_dropped_cnn or n_dropped_meta:
            logger.warning(
                "Alignment dropped %d CNN-only and %d metadata-only entries.",
                n_dropped_cnn,
                n_dropped_meta,
            )
        return common

    def _load_masks(
        self,
        image_ids: List[str],
    ) -> Optional[List[np.ndarray]]:
        """Load masks from disk, one per image_id.

        Supported file formats: ``.npy``, ``.png``, ``.tif``.
        """
        masks_dir = self.config.masks_dir
        assert masks_dir is not None

        masks: List[np.ndarray] = []
        missing: List[str] = []

        for img_id in image_ids:
            loaded = False
            for ext in (".npy", ".png", ".tif", ".tiff", ".bmp"):
                fpath = os.path.join(masks_dir, f"{img_id}{ext}")
                if os.path.isfile(fpath):
                    if ext == ".npy":
                        mask = np.load(fpath)
                    else:
                        try:
                            from PIL import Image  # type: ignore[import-untyped]

                            mask = np.array(Image.open(fpath).convert("L"))
                        except ImportError:
                            import cv2  # type: ignore[import-untyped]

                            mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    masks.append(mask)
                    loaded = True
                    break
            if not loaded:
                missing.append(img_id)

        if missing:
            logger.warning(
                "%d / %d masks not found (e.g. %s). Returning None for masks.",
                len(missing),
                len(image_ids),
                missing[:3],
            )
            return None

        return masks

    def _compute_seg_features(
        self,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Extract segmentation features from a list of masks.

        Uses a cache file to avoid re-extracting on subsequent runs.
        """
        # Check for cached features (saved alongside masks, not in outputs)
        cache_path = Path(self.config.masks_dir) / ".seg_features_cache.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            # Validate BOTH N and D — D changes when _SEG_FEATURE_NAMES changes.
            if (cached["features"].shape[0] == len(masks)
                    and cached["features"].shape[1] == len(_SEG_FEATURE_NAMES)):
                logger.info("Loaded cached segmentation features from %s", cache_path)
                return cached["features"]
            logger.info(
                "Seg-feature cache stale (shape=%s, expected (%d, %d)) — recomputing",
                cached["features"].shape, len(masks), len(_SEG_FEATURE_NAMES),
            )

        from joblib import Parallel, delayed

        aggregation = self.config.seg_aggregation
        n = len(masks)
        # threading backend: avoids pickling large mask arrays to loky workers
        # (scikit-image regionprops/GLCM release the GIL during the heavy C loops)
        print(f"  [seg_features] Extracting {n} masks (threads, n_jobs={self.config.n_jobs})...")
        features = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
            delayed(_extract_seg_features_from_mask)(m, aggregation=aggregation)
            for m in masks
        )
        result = np.stack(features, axis=0)

        # Cache for future runs
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=result)
        logger.info("Cached segmentation features to %s", cache_path)
        return result

    # -----------------------------------------------------------------
    # Synthetic data generators (for testing / demos)
    # -----------------------------------------------------------------

    def _generate_synthetic_cnn_features(
        self,
        n_samples: int = 500,
        n_features: int = 960,
    ) -> Dict[str, np.ndarray]:
        """Create random CNN-like feature vectors."""
        rng = np.random.default_rng(self.config.random_state)
        ids = [f"img_{i:05d}" for i in range(n_samples)]
        features = rng.standard_normal((n_samples, n_features)).astype(
            np.float32
        )
        return dict(zip(ids, features))

    def _generate_synthetic_metadata(
        self,
        image_ids: List[str],
    ) -> pd.DataFrame:
        """Create synthetic metadata with defect types and expert groups."""
        rng = np.random.default_rng(self.config.random_state)
        n = len(image_ids)

        defect_types = [f"type_{i}" for i in range(20)]
        expert_groups = [f"group_{i}" for i in range(8)]

        # Assign each defect type to an expert group (deterministically).
        type_to_group = {
            dt: expert_groups[i % len(expert_groups)]
            for i, dt in enumerate(defect_types)
        }

        chosen_types = rng.choice(defect_types, size=n)
        chosen_groups = [type_to_group[dt] for dt in chosen_types]

        return pd.DataFrame(
            {
                "image_id": image_ids,
                self.config.defect_type_col: chosen_types,
                self.config.expert_group_col: chosen_groups,
            }
        )

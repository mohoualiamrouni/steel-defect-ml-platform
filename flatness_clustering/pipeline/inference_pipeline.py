"""
Flatness Clustering — Inference Pipeline
=========================================

Assigns a new coil flatness profile to the nearest cluster using a
pre-trained CoilClusterer.

Confidence is computed via the DEC Student-t soft assignment kernel
(Asadi & Regan 2021):
    q_j = (1 + d_j² / α)^{-(α+1)/2} / Σ_{j'} (1 + d_{j'}² / α)^{-(α+1)/2}
so `soft_assignments` sums to 1.0 and `confidence = max(soft_assignments)`.

Usage
-----
    from flatness_clustering.pipeline.inference_pipeline import assign_new_coil

    # From a numpy array
    result = assign_new_coil(profile)
    print(result)
    # {
    #     "cluster_id":        2,
    #     "cluster_distances": array([3.1, 1.2, 0.4, 4.5, 2.0, 3.7]),
    #     "soft_assignments":  array([0.05, 0.08, 0.72, 0.04, 0.07, 0.04]),  # DEC
    #     "confidence":        0.72,   # = max(soft_assignments)
    #     "is_outlier":        False,
    # }

    # From an NPZ file
    result = assign_new_coil_from_file("path/to/coil.npz")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


def assign_new_coil(
    profile: np.ndarray,
    clusterer_path: str = "flatness_clustering/outputs/clusterer.joblib",
    rocket_variant: Optional[str] = None,
) -> dict:
    """Assign a new coil flatness profile to the nearest cluster.

    Features are extracted on-the-fly from the raw profile array, then
    the pre-trained CoilClusterer assigns the coil.

    The feature-extraction variant (``rocket`` vs ``minirocket``) MUST match
    the one used at training, otherwise the feature dimension differs and the
    assignment is silently wrong.  We therefore default to the variant
    persisted on the clusterer at train time; an explicit ``rocket_variant``
    overrides it, and a dimension check guards against any residual mismatch.

    Args:
        profile:        float32 array of shape (W, L) — the flatness profile
                        in I-units.  W = width points, L = length points.
        clusterer_path: Path to the serialised CoilClusterer (.joblib).
        rocket_variant: Override the feature variant ('rocket' | 'minirocket').
                        ``None`` (default) uses the variant stored on the
                        clusterer, falling back to 'rocket' for legacy models.

    Returns:
        dict with keys:
            cluster_id       : int — nearest cluster (argmax of soft_assignments)
            cluster_distances: np.ndarray (K,) — distance to each centroid
            soft_assignments : np.ndarray (K,) — DEC soft membership probabilities,
                               sums to 1.0 (Asadi & Regan 2021)
            confidence       : float in [0, 1] — max(soft_assignments)
            is_outlier       : bool

    Raises:
        FileNotFoundError: If the clusterer file does not exist.
    """
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.features.statistical_features import (
        extract_statistical_features,
    )

    if not Path(clusterer_path).exists():
        raise FileNotFoundError(
            f"Clusterer not found: '{clusterer_path}'\n"
            "  Run: python flatness_clustering/pipeline/train_pipeline.py"
        )

    clusterer = CoilClusterer.load(clusterer_path)
    # Use the training-time variant unless explicitly overridden.
    variant = rocket_variant or getattr(clusterer, "rocket_variant", None) or "rocket"
    feature_vec = extract_statistical_features(profile, rocket_variant=variant)
    _check_feature_dim(feature_vec, clusterer, variant)
    return clusterer.predict_with_confidence(feature_vec)


def _check_feature_dim(feature_vec: np.ndarray, clusterer, variant: str) -> None:
    """Fail loudly if the extracted feature dim != the training dim."""
    expected = getattr(clusterer, "expected_n_features", None)
    got = int(np.asarray(feature_vec).reshape(-1).shape[0])
    if expected is not None and got != expected:
        raise ValueError(
            f"Feature dimension mismatch: extracted {got} features with "
            f"rocket_variant='{variant}', but the clusterer was trained on "
            f"{expected}. Pass the matching rocket_variant (the model was "
            f"trained with '{getattr(clusterer, 'rocket_variant', 'unknown')}')."
        )


def assign_new_coil_from_file(
    npz_path: str,
    clusterer_path: str = "flatness_clustering/outputs/clusterer.joblib",
    rocket_variant: Optional[str] = None,
) -> dict:
    """Load a coil from an NPZ file and assign it to a cluster.

    Args:
        npz_path:       Path to a .npz file containing a 'profile' key.
        clusterer_path: Path to the serialised CoilClusterer (.joblib).
        rocket_variant: Feature variant override; ``None`` uses the model's.

    Returns:
        Same dict as assign_new_coil(), plus 'coil_id' and 'pattern' if available.
    """
    data = np.load(npz_path, allow_pickle=True)
    profile = data["profile"].astype(np.float32)
    result = assign_new_coil(
        profile, clusterer_path=clusterer_path, rocket_variant=rocket_variant
    )

    # Attach metadata if present in the file
    for key in ("coil_id", "pattern", "grade", "customer"):
        if key in data:
            result[key] = str(data[key])

    return result


def batch_assign(
    profiles: np.ndarray,
    clusterer_path: str = "flatness_clustering/outputs/clusterer.joblib",
    rocket_variant: Optional[str] = None,
) -> list:
    """Assign a batch of coil profiles to clusters.

    Loads the clusterer once and extracts all features in one pass, which is
    significantly faster than calling assign_new_coil() N times for large batches.

    Args:
        profiles:       float32 array of shape (N, W, L).
        clusterer_path: Path to the serialised CoilClusterer.
        rocket_variant: Feature variant override; ``None`` uses the model's.

    Returns:
        List of N result dicts (same structure as assign_new_coil).
    """
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    from flatness_clustering.clustering.coil_clusterer import CoilClusterer
    from flatness_clustering.features.statistical_features import (
        extract_batch,
    )

    if not Path(clusterer_path).exists():
        raise FileNotFoundError(
            f"Clusterer not found: '{clusterer_path}'\n"
            "  Run: python flatness_clustering/pipeline/train_pipeline.py"
        )

    clusterer = CoilClusterer.load(clusterer_path)
    variant = rocket_variant or getattr(clusterer, "rocket_variant", None) or "rocket"

    # Extract all features in one vectorised call, using the model's variant.
    features = extract_batch(profiles, rocket_variant=variant)   # (N, F)
    if len(features):
        _check_feature_dim(features[0], clusterer, variant)

    results = []
    for i in range(len(features)):
        results.append(clusterer.predict_with_confidence(features[i]))
    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Assign a flatness profile to a cluster")
    p.add_argument("--profile",   default=None,
                   help="Path to NPZ file.  If not given, uses a random synthetic profile.")
    p.add_argument("--clusterer", default="flatness_clustering/outputs/clusterer.joblib")
    p.add_argument("--pattern",   default="center_wave",
                   help="Pattern class to generate (if --profile is not given)")
    args = p.parse_args()

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    if args.profile:
        result = assign_new_coil_from_file(args.profile, clusterer_path=args.clusterer)
    else:
        # Generate a quick synthetic test profile
        from flatness_clustering.data.generate_flatness_data import GENERATORS
        rng = np.random.default_rng(42)
        gen = GENERATORS.get(args.pattern, GENERATORS["center_wave"])
        profile = gen(rng, width=150, length=2000)
        result = assign_new_coil(profile, clusterer_path=args.clusterer)

    print("\nAssignment result:")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {np.round(v, 4)}")
        else:
            print(f"  {k}: {v}")

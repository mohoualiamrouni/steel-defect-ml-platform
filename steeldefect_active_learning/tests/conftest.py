"""Shared pytest fixtures for the steeldefect_active_learning test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def ab_df_classification(rng):
    """Synthetic A/B classification DataFrame with ground truth.

    Constructs:
      - 500 images over 11 classes (taxonomy cardinality), no IoU column.
      - A is right on about 80 % of the images and B on about 85 % (realised 0.784 and
        0.856), errors drawn independently.
      - Confidence distributions are Beta-shaped with arm-specific means.
    """
    n = 500
    n_classes = 11
    labels = rng.integers(0, n_classes, size=n)

    # Model A: 80% accuracy
    pred_a = labels.copy()
    flip_a = rng.random(n) < 0.20
    pred_a[flip_a] = (pred_a[flip_a] + rng.integers(1, n_classes, size=int(flip_a.sum()))) % n_classes

    # Model B: 85% accuracy (better, by design)
    pred_b = labels.copy()
    flip_b = rng.random(n) < 0.15
    pred_b[flip_b] = (pred_b[flip_b] + rng.integers(1, n_classes, size=int(flip_b.sum()))) % n_classes

    conf_a = rng.beta(8, 3, size=n)        # mean ≈ 0.73
    conf_b = rng.beta(9, 3, size=n)        # mean ≈ 0.75
    entropy_a = rng.uniform(0.1, 0.9, size=n)
    entropy_b = rng.uniform(0.1, 0.9, size=n) * 0.9   # B is slightly more confident

    df = pd.DataFrame({
        "image_path":       [f"img_{i:04d}.png" for i in range(n)],
        "label":            labels,
        "pred_a":           pred_a,
        "pred_b":           pred_b,
        "confidence_a":     conf_a,
        "confidence_b":     conf_b,
        "entropy_a":        entropy_a,
        "entropy_b":        entropy_b,
        "agrees":           (pred_a == pred_b).astype(int),
        "confidence_delta": np.abs(conf_a - conf_b),
    })
    return df


@pytest.fixture
def ab_df_paired_null(rng):
    """A paired null for correctness: A and B equally accurate on the same images.

    2,200 images, 200 per class over 11 classes. A and B are each right on about 85 % of
    the images, with independent errors, so about a quarter of the images are discordant
    and neither model is better.
    """
    n_per_class, n_classes = 200, 11
    labels = np.repeat(np.arange(n_classes), n_per_class)
    n = len(labels)
    preds = []
    for _ in range(2):
        pred = labels.copy()
        wrong = rng.random(n) < 0.15
        pred[wrong] = (pred[wrong] + rng.integers(1, n_classes, size=int(wrong.sum()))) % n_classes
        preds.append(pred)
    return pd.DataFrame({
        "image_path": [f"img_{i:05d}.png" for i in range(n)],
        "label":      labels,
        "pred_a":     preds[0],
        "pred_b":     preds[1],
    })


@pytest.fixture
def ab_df_balanced(rng):
    """A/B df that is a null case only for the confidence columns.

    The confidences of A and B are drawn from the same distribution. The labels copy
    pred_a, so A is the oracle (accuracy 1.0 against about 0.95 for B): not a null for
    correctness.
    """
    n = 400
    a = rng.normal(0.7, 0.15, size=n).clip(0, 1)
    b = rng.normal(0.7, 0.15, size=n).clip(0, 1)
    pred_a = rng.integers(0, 11, size=n)
    pred_b = pred_a.copy()
    # Inject 5% disagreement so 'agrees' isn't degenerate.
    disagree = rng.random(n) < 0.05
    pred_b[disagree] = (pred_b[disagree] + 1) % 11
    return pd.DataFrame({
        "label":            pred_a,           # treat A as oracle so accuracy=100%
        "pred_a":           pred_a,
        "pred_b":           pred_b,
        "confidence_a":     a,
        "confidence_b":     b,
        "agrees":           (pred_a == pred_b).astype(int),
        "confidence_delta": np.abs(a - b),
    })

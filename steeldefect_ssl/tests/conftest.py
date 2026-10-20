"""Shared fixtures: deterministic toy embeddings (no image-gen / SSL training)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch():
    """Seed torch's global RNG before every test: several tests draw random
    tensors or apply random augmentations, and must see the same inputs on
    every run."""
    torch.manual_seed(0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def toy_known(rng):
    """Well-separated Gaussian classes. Returns (X, y), K=4, d=8, 60/class."""
    K, d, n = 4, 8, 60
    centers = rng.normal(0, 6, (K, d))
    X = np.concatenate([centers[c] + rng.normal(0, 0.6, (n, d)) for c in range(K)])
    y = np.concatenate([np.full(n, c) for c in range(K)])
    return X.astype(np.float32), y.astype(int)


@pytest.fixture
def toy_with_novel(rng, toy_known):
    """Known classes + a held-out NOVEL cluster far away. Returns (X, y, is_novel)."""
    Xk, yk = toy_known
    d = Xk.shape[1]
    n_novel = 40
    Xn = np.full(d, 20.0) + rng.normal(0, 0.6, (n_novel, d))
    X = np.concatenate([Xk, Xn]).astype(np.float32)
    y = np.concatenate([yk, np.full(n_novel, yk.max() + 1)]).astype(int)
    is_novel = np.concatenate([np.zeros(len(Xk), bool), np.ones(n_novel, bool)])
    return X, y, is_novel

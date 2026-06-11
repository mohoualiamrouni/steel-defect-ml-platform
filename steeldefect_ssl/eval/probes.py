"""Frozen-feature evaluation probes + baseline representations.

Linear probe (logistic regression on frozen features) and kNN probe are the
standard SSL evaluation protocol. The SSL representation is compared honestly
against two baselines: a random-init encoder and raw-pixel PCA.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def linear_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, y_te: np.ndarray, random_state: int = 42) -> float:
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, random_state=random_state)
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te))


def knn_probe(X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray, k: int = 10) -> float:
    k = int(min(k, max(1, len(X_tr) - 1)))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def probe_representation(X_tr, y_tr, X_te, y_te, knn_k: int = 10,
                         random_state: int = 42) -> Dict[str, float]:
    return {
        "linear_probe_acc": linear_probe(X_tr, y_tr, X_te, y_te, random_state),
        "knn_acc": knn_probe(X_tr, y_tr, X_te, y_te, knn_k),
    }


def pca_reduce(X_fit: np.ndarray, dim: int, random_state: int = 42):
    """Fit PCA on ``X_fit``; return (pca, transform-callable)."""
    dim = int(min(dim, X_fit.shape[1], max(2, X_fit.shape[0] - 1)))
    pca = PCA(n_components=dim, random_state=random_state).fit(X_fit)
    return pca, (lambda X: pca.transform(X).astype(np.float32))


def flatten_pixels(paths, image_size: int) -> np.ndarray:
    """Load crops as flattened grayscale pixel vectors (raw-pixel baseline)."""
    from steeldefect_ssl.data.ssl_dataset import load_gray
    rows = [load_gray(p, image_size).numpy().ravel() for p in paths]
    return np.asarray(rows, dtype=np.float32)

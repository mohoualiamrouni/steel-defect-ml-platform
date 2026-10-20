"""Smoke tests for flatness_clustering: generate profiles, extract features,
fit the clusterer and assign coils, on tiny deterministic synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from flatness_clustering.clustering.coil_clusterer import CoilClusterer
from flatness_clustering.data.generate_flatness_data import GENERATORS
from flatness_clustering.features.statistical_features import extract_batch, feature_names

N_PER_CLASS = 3
K = 4


@pytest.fixture(scope="module")
def features() -> np.ndarray:
    rng = np.random.default_rng(0)
    profiles = np.stack([
        gen(rng, width=40, length=300)
        for gen in GENERATORS.values()
        for _ in range(N_PER_CLASS)
    ])
    return extract_batch(profiles)


def test_feature_matrix_matches_feature_names(features):
    assert features.shape == (N_PER_CLASS * len(GENERATORS), len(feature_names()))
    assert np.isfinite(features).all()


@pytest.mark.parametrize("method", ["kmeans", "minibatch_kmeans", "agglomerative"])
def test_clusterer_fits_and_assigns_every_coil(features, method):
    clusterer = CoilClusterer(method=method, n_clusters=K, random_state=0).fit(features)
    labels = clusterer.predict(features)
    assert labels.shape == (len(features),)
    assert set(np.unique(labels)) <= set(range(K))
    single = clusterer.predict(features[0])
    assert isinstance(single, int) and 0 <= single < K


def test_kmeans_assignment_is_reproducible(features):
    first = CoilClusterer(method="kmeans", n_clusters=K, random_state=0).fit(features)
    second = CoilClusterer(method="kmeans", n_clusters=K, random_state=0).fit(features)
    np.testing.assert_array_equal(first.predict(features), second.predict(features))

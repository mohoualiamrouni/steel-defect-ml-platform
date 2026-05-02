"""Unit tests for AL acquisition strategies and dispatcher routing."""
from __future__ import annotations

import numpy as np
import pytest

from steeldefect_active_learning.active_learning.candidate_selector import (
    _auto_budget_strategy,
    select_candidates,
)


@pytest.fixture
def synthetic_pool(rng):
    n = 200
    paths = [f"img_{i:04d}.png" for i in range(n)]
    embeddings = {p: rng.normal(0.0, 1.0, size=16).astype(np.float32) for p in paths}
    uncertainty = {p: float(rng.uniform()) for p in paths}
    clusters = {p: int(rng.integers(0, 5)) for p in paths}
    label_probs = {p: rng.dirichlet(np.ones(11)).astype(np.float32) for p in paths}
    return paths, embeddings, uncertainty, clusters, label_probs


class TestStratifiedSelect:
    def test_respects_budget(self, synthetic_pool):
        _, _, unc, clusters, _ = synthetic_pool
        out = select_candidates(unc, clusters, labeling_budget=20, method="stratified")
        assert len(out) <= 20

    def test_no_duplicates(self, synthetic_pool):
        _, _, unc, clusters, _ = synthetic_pool
        out = select_candidates(unc, clusters, labeling_budget=20, method="stratified")
        assert len(set(out)) == len(out)


class TestSubmodularSimilar:
    def test_returns_unique_paths(self, synthetic_pool):
        _, embeddings, unc, clusters, _ = synthetic_pool
        out = select_candidates(
            unc, clusters,
            labeling_budget=15,
            method="similar_submodular",
            embeddings_dict=embeddings,
        )
        assert len(out) == 15
        assert len(set(out)) == 15

    def test_requires_embeddings(self, synthetic_pool):
        _, _, unc, clusters, _ = synthetic_pool
        with pytest.raises(ValueError):
            select_candidates(
                unc, clusters,
                labeling_budget=10,
                method="similar_submodular",
                embeddings_dict=None,
            )


class TestAutoBudgetStrategy:
    def test_phase_1_cold_start(self):
        method = _auto_budget_strategy(budget=10, n_labeled=5, n_classes=11)
        assert method == "typiclust"

    def test_phase_2_exploration(self):
        method = _auto_budget_strategy(budget=10, n_labeled=50, n_classes=11)
        assert method == "badge"

    def test_phase_3_imbalanced(self):
        # 10×11 = 110 → phase 3
        counts = {0: 100, 1: 5, 2: 5}
        method = _auto_budget_strategy(
            budget=10, n_labeled=200, class_counts=counts, n_classes=11,
        )
        assert method == "similar"

    def test_phase_3_balanced(self):
        counts = {c: 20 for c in range(11)}
        method = _auto_budget_strategy(
            budget=10, n_labeled=220, class_counts=counts, n_classes=11,
        )
        assert method == "stratified"

    def test_family_adaptive_routes_typiclust_when_a_class_is_starved(self):
        # Class 5 ("rare") has only seen 1 sample; target is 4. Phase 1 must
        # not exit until every class has ≥ 2 × target.
        counts = {c: 100 for c in range(11)}
        counts[5] = 1
        targets = [10] * 11
        targets[5] = 4   # rare class needs less to satisfy 2x but is still starved
        method = _auto_budget_strategy(
            budget=10,
            n_labeled=sum(counts.values()),
            class_counts=counts,
            n_classes=11,
            target_per_class=targets,
        )
        assert method == "typiclust"

    def test_family_adaptive_target_length_validated(self):
        with pytest.raises(ValueError):
            _auto_budget_strategy(
                budget=10,
                n_labeled=100,
                class_counts={c: 10 for c in range(11)},
                n_classes=11,
                target_per_class=[10, 10, 10],  # wrong length
            )

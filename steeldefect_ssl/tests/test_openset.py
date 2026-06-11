import numpy as np

from steeldefect_ssl.openset.evaluate import auroc_aupr_fpr95
from steeldefect_ssl.openset.scorers import (EnergyScorer, KNNScorer,
                                       MahalanobisScorer, build_scorer)


def _split_known(X, y, is_novel):
    return X[~is_novel], y[~is_novel]


def test_mahalanobis_separates_novel(toy_with_novel):
    X, y, is_novel = toy_with_novel
    Xk, yk = _split_known(X, y, is_novel)
    s = MahalanobisScorer(shrinkage=1e-3).fit(Xk, yk).score(X)
    m = auroc_aupr_fpr95(s, is_novel)
    assert m["auroc"] > 0.9
    assert m["fpr95"] <= 0.3


def test_knn_scorer_separates_novel(toy_with_novel):
    X, y, is_novel = toy_with_novel
    Xk, yk = _split_known(X, y, is_novel)
    s = KNNScorer(k=5).fit(Xk).score(X)
    assert auroc_aupr_fpr95(s, is_novel)["auroc"] > 0.9


def test_energy_scorer_runs(toy_with_novel):
    X, y, is_novel = toy_with_novel
    Xk, yk = _split_known(X, y, is_novel)
    s = EnergyScorer().fit(Xk, yk).score(X)
    assert s.shape == (len(X),)


def test_metrics_degenerate_returns_nan():
    m = auroc_aupr_fpr95(np.arange(10.0), np.zeros(10))  # all known
    assert np.isnan(m["auroc"])


def test_build_scorer():
    assert isinstance(build_scorer("mahalanobis"), MahalanobisScorer)
    assert isinstance(build_scorer("knn"), KNNScorer)

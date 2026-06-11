import numpy as np

from steeldefect_ssl.config import DiscoveryConfig
from steeldefect_ssl.continual.ncm import NCMClassifier, herding_buffer
from steeldefect_ssl.discovery.cluster_accuracy import hungarian_cluster_accuracy
from steeldefect_ssl.discovery.gcd import select_k, semi_supervised_kmeans


def test_hungarian_accuracy_perfect_permutation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])   # a permutation of the truth
    assert hungarian_cluster_accuracy(y_true, y_pred) == 1.0


def test_select_k_fixed():
    cfg = DiscoveryConfig(k_select="fixed", fixed_k=4)
    assert select_k(np.random.RandomState(0).randn(50, 5), cfg) == 4


def test_semi_supervised_kmeans_recovers_clusters(toy_known):
    X, y = toy_known
    rng = np.random.default_rng(0)
    mask = rng.random(len(X)) < 0.3          # 30% labeled
    y_seed = np.where(mask, y, -1)
    labels = semi_supervised_kmeans(X, y_seed, mask, k=4, random_state=42)
    assert hungarian_cluster_accuracy(y, labels) > 0.8


def test_ncm_predicts_and_no_forgetting(toy_with_novel):
    X, y, is_novel = toy_with_novel
    Xk, yk = X[~is_novel], y[~is_novel]
    ncm = NCMClassifier().fit(Xk, yk)
    acc_before = ncm.accuracy(Xk, yk)
    assert acc_before > 0.95
    # add the novel family -> additive prototype, old prototypes untouched
    Xn = X[is_novel]
    ncm.add_class(Xn[: len(Xn) // 2], label=yk.max() + 1)
    f = ncm.forgetting(Xk, yk, acc_before)
    assert abs(f["forgetting_delta"]) < 0.05           # near-zero forgetting
    assert ncm.accuracy(Xn[len(Xn) // 2:], np.full(len(Xn) - len(Xn) // 2, yk.max() + 1)) > 0.9


def test_herding_buffer_size():
    X = np.random.RandomState(0).randn(40, 6)
    idx = herding_buffer(X, 5)
    assert len(idx) == 5 and len(set(idx.tolist())) == 5

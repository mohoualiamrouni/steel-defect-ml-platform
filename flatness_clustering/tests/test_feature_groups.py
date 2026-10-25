"""Code that reads flatness features by group follows the extractor's column layout."""

from flatness_clustering.clustering import advanced_viz
from flatness_clustering.clustering.report import _feature_groups
from flatness_clustering.features.statistical_features import (
    N_FEATURES_TOTAL,
    feature_group_slices,
    feature_names,
)

PREFIXES = {"A": "zone_", "B": "snap", "C": "spectral_", "D": "temporal_", "E": "rocket_"}


def test_groups_are_contiguous_and_cover_every_column():
    slices = [cols for _, cols in feature_group_slices()]
    assert [s.start for s in slices[1:]] == [s.stop for s in slices[:-1]]
    assert slices[0].start == 0 and slices[-1].stop == N_FEATURES_TOTAL == len(feature_names())


def test_each_group_holds_only_its_own_features():
    names = feature_names()
    for group, cols in feature_group_slices():
        assert all(n.startswith(PREFIXES[group]) for n in names[cols]), group
    assert feature_group_slices()[-1][1].stop - feature_group_slices()[-1][1].start == 1000


def test_plots_and_report_use_the_extractor_slices():
    expected = [cols for _, cols in feature_group_slices()]
    assert [cols for _, cols in advanced_viz._GROUPS] == expected
    assert [cols for _, cols in _feature_groups()] == expected

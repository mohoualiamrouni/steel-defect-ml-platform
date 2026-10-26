"""make_umap_plots pairs each feature row with its own assignment row."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_umap_plots.py"


@pytest.fixture(scope="module")
def script():
    """scripts/ is not a package, so load the file by path."""
    spec = importlib.util.spec_from_file_location("make_umap_plots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def assignments():
    """Five images in the order training writes them (the full metadata)."""
    return pd.DataFrame({"image_id": ["a", "b", "c", "d", "e"],
                         "cluster_id": [0, 1, 2, 3, 4],
                         "expert_group": ["g0", "g1", "g2", "g3", "g4"]})


def test_rows_pair_by_image_id(script, assignments):
    # A subsampled run keeps three of the five images, in its own order.
    paired = script.pair_rows(3, np.array(["c", "a", "d"]), assignments)
    assert paired["cluster_id"].tolist() == [2, 0, 3]
    assert paired["expert_group"].tolist() == ["g2", "g0", "g3"]


def test_unknown_image_ids_raise(script, assignments):
    with pytest.raises(ValueError, match="image_ids"):
        script.pair_rows(2, np.array(["a", "z"]), assignments)


def test_feature_rows_and_image_ids_must_match_in_number(script, assignments):
    with pytest.raises(ValueError, match="3 feature rows but 2 image_ids"):
        script.pair_rows(3, np.array(["a", "b"]), assignments)


def test_files_without_image_ids_pair_by_position_only_when_lengths_match(script, assignments):
    assert script.pair_rows(5, None, assignments) is assignments
    with pytest.raises(ValueError, match="has 3 rows"):
        script.pair_rows(3, None, assignments)


class _FirstTwoColumns:
    """Stand-in for umap.UMAP: these tests check the pairing and the files."""

    def __init__(self, **kwargs):
        pass

    def fit_transform(self, X):
        return X[:, :2]


@pytest.fixture
def outputs(script, assignments, tmp_path, monkeypatch):
    """Point the script at tmp_path; return a writer for its feature file."""
    assignments.to_csv(tmp_path / "cluster_assignments.csv", index=False)
    monkeypatch.setattr(script, "FEAT", tmp_path / "feature_matrix.npz")
    monkeypatch.setattr(script, "ASSIGN", tmp_path / "cluster_assignments.csv")
    monkeypatch.setattr(script, "OUT", tmp_path / "figures")
    monkeypatch.setattr(script, "umap", SimpleNamespace(UMAP=_FirstTwoColumns))

    def write_features(**arrays):
        np.savez(tmp_path / "feature_matrix.npz", features=np.zeros((3, 4)), **arrays)
    return write_features


def test_main_colours_each_point_with_its_own_rows(script, outputs, tmp_path, monkeypatch):
    colours = []
    scatter = Axes.scatter

    def spy(ax, *args, **kwargs):
        colours.append([int(code) for code in kwargs["c"]])
        return scatter(ax, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", spy)
    outputs(image_ids=np.array(["c", "a", "d"]))
    script.main()
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "dc_umap_expert.png", "dc_umap_proposed.png"]
    # Category codes of the paired rows: clusters [2, 0, 3] and groups [g2, g0, g3]
    # both code to [1, 0, 2]; pairing by position would give [0, 1, 2].
    assert colours == [[1, 0, 2], [1, 0, 2]]


def test_figures_go_to_the_package_outputs(script):
    assert script.OUT == SCRIPT.parents[1] / "outputs" / "figures"


def test_unpairable_files_raise_before_any_output(script, outputs, tmp_path):
    outputs()   # three feature rows, five assignment rows, no image_ids
    with pytest.raises(ValueError, match="has 3 rows"):
        script.main()
    assert not (tmp_path / "figures").exists()

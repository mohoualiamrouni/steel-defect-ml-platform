"""make_umap_plots pairs each feature row with its own assignment row."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

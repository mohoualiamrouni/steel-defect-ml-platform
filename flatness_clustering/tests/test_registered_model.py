"""Regression test: the flatness model registered in MLflow is the full CoilClusterer."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from flatness_clustering.clustering.coil_clusterer import CoilClusterer  # noqa: E402
from flatness_clustering.features.statistical_features import feature_names  # noqa: E402

job = importlib.import_module("flatness_clustering.databricks.02_clustering_training")


class _Writer:
    def format(self, *_args):
        return self

    def mode(self, *_args):
        return self

    def option(self, *_args):
        return self

    def saveAsTable(self, *_args):
        return None


class _SparkFrame:
    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def toPandas(self) -> pd.DataFrame:
        return self._pdf.copy()

    @property
    def write(self) -> _Writer:
        return _Writer()


class _FakeSpark:
    """The two Spark calls the training job makes, served from memory."""

    def __init__(self, table: pd.DataFrame):
        self._table = table

    def table(self, _name: str) -> _SparkFrame:
        return _SparkFrame(self._table)

    def createDataFrame(self, pdf: pd.DataFrame) -> _SparkFrame:
        return _SparkFrame(pdf)


def _feature_table(n_per_cluster: int = 20, seed: int = 0):
    names = feature_names()
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 5.0, size=(3, len(names)))
    blobs = [c + rng.normal(0.0, 1.0, size=(n_per_cluster, len(names))) for c in centers]
    # Raw features sit far from the standardized space the estimator is fitted in.
    raw = (np.concatenate(blobs) * 50.0 + 1000.0).astype(np.float32)
    table = pd.DataFrame(raw, columns=names)
    table["coil_id"] = [f"coil_{i}" for i in range(len(table))]
    table["grade"], table["customer"], table["timestamp"] = "g1", "k1", "2026-01-01"
    return table, raw


def test_registered_model_reproduces_clusterer_predictions(tmp_path, monkeypatch):
    table, raw = _feature_table()
    monkeypatch.setattr(job, "spark", _FakeSpark(table), raising=False)
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        mlflow.create_experiment("flatness-test", artifact_location=(tmp_path / "artifacts").as_uri())
        job.run_clustering_training(
            n_clusters=3, mlflow_experiment="flatness-test", artifact_dir=str(tmp_path / "models"),
        )
        # Loaded exactly as flatness_clustering/databricks/03_online_assignment.py does.
        registered = mlflow.sklearn.load_model("models:/steeldefect_flatness_clusterer/1")
        reference = CoilClusterer.load(str(tmp_path / "models" / "clusterer.joblib"))
        assert isinstance(registered, CoilClusterer)
        np.testing.assert_array_equal(registered.predict(raw), reference.predict(raw))
    finally:
        mlflow.set_tracking_uri(original_uri)

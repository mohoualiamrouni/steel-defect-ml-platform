"""Unit tests for the SteelDefect platform layer (no Spark / no MLflow required).

Covers the parts that are pure-Python and previously untested: the Jobs DAG
(topological validity), the dispatcher↔DAG contract, OBT row validation, the
registry model set + failure contract, and the drift detectors.
"""

import os
import sys

import numpy as np
import pytest

# repo root on path so `steeldefect_platform` imports when pytest runs from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steeldefect_platform import drift, jobs, mlflow_registry, obt_schema, run_task  # noqa: E402


# --- Jobs DAG ---------------------------------------------------------------

def test_topological_order_is_valid():
    order = jobs.topological_order()
    assert set(order) == set(jobs.JOBS_DAG), "topo order must cover every DAG node"
    assert len(order) == len(jobs.JOBS_DAG), "no node repeated/dropped (acyclic)"
    pos = {n: i for i, n in enumerate(order)}
    for node, deps in jobs.JOBS_DAG.items():
        for d in deps:
            assert pos[d] < pos[node], f"{d} must precede {node}"


def test_dispatcher_matches_dag():
    # every schedulable task maps to a real DAG node
    for task in run_task.TASKS:
        assert task in jobs.JOBS_DAG, f"dispatcher task {task} absent from JOBS_DAG"
    # the two conceptual nodes are in the DAG but intentionally NOT schedulable
    for conceptual in ("segmenter_train", "online_regroup"):
        assert conceptual in jobs.JOBS_DAG
        assert conceptual not in run_task.TASKS


# --- OBT schema constants + row validation ----------------------------------

def test_obt_constants():
    assert obt_schema.CNN_DIM == 960
    assert obt_schema.N_MORPHO == 24 == len(obt_schema.MORPHO_NAMES)
    assert obt_schema.N_EXPERT_GROUPS == 28


def test_validate_row_accepts_valid():
    obt_schema.validate_row({"defect_kind": "local", "pos_x_m": 0.1, "pos_y_m": 0.2})
    obt_schema.validate_row({"defect_kind": "group", "density_per_m2": 3.0})
    obt_schema.validate_row({
        "defect_kind": "local", "pos_x_m": 0.0, "pos_y_m": 0.0,
        "cnn_vector": [0.0] * 960, "cnn_dim": 960, "expert_group_id": 0,
    })


@pytest.mark.parametrize("row", [
    {"defect_kind": "spurious"},                                  # bad kind
    {"defect_kind": "local"},                                     # local w/o position
    {"defect_kind": "group"},                                     # group w/o density
    {"defect_kind": "local", "pos_x_m": 0, "pos_y_m": 0, "cnn_vector": [0.0] * 10},
    {"defect_kind": "local", "pos_x_m": 0, "pos_y_m": 0, "expert_group_id": 99},
])
def test_validate_row_rejects(row):
    with pytest.raises(ValueError):
        obt_schema.validate_row(row)


# --- MLflow registry --------------------------------------------------------

def test_registry_model_set():
    assert len(mlflow_registry.ALL_MODELS) == 5
    assert mlflow_registry.SSL_ENCODER in mlflow_registry.ALL_MODELS
    assert len(set(mlflow_registry.ALL_MODELS)) == 5  # unique


def test_load_production_raises_without_mlflow(monkeypatch):
    def _boom():
        raise ImportError("mlflow not installed")
    monkeypatch.setattr(mlflow_registry, "_mlflow", _boom)
    with pytest.raises(RuntimeError):
        mlflow_registry.load_production("steeldefect_segmenter")


# --- Drift detectors --------------------------------------------------------

def test_mmd_separates_identical_from_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=(120, 8))
    b = rng.normal(0, 1, size=(120, 8))      # same distribution
    c = rng.normal(3, 1, size=(120, 8))      # shifted distribution
    _, p_same = drift.mmd_drift(a, b, n_permutations=100)
    _, p_shift = drift.mmd_drift(a, c, n_permutations=100)
    assert p_same > 0.05, "no drift expected between same-distribution cohorts"
    assert p_shift < 0.05, "drift expected against a shifted cohort"


def test_adwin_detects_level_shift():
    stream = [0.5] * 40 + [0.95] * 40
    state = drift.ADWINState()
    assert any(state.update(v) for v in stream), "ADWIN should flag the level shift"


# --- OBT morphology backfill --------------------------------------------------

def _two_blobs():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:8, 2:8] = 255        # 36 px
    mask[20:35, 15:35] = 255    # 300 px
    return mask


def _backfill(monkeypatch, mask, **row):
    """Run the ingestion UDF on one landing row whose mask decodes to *mask*."""
    from types import SimpleNamespace

    import pandas as pd

    from steeldefect_platform import ingestion, io_utils

    cols = ["segmentation_id", "mask_uri", *obt_schema.MORPHO_NAMES, "_dq_valid", "_dq_reason"]
    monkeypatch.setattr(ingestion, "_udf_struct_type",
                        lambda: SimpleNamespace(fields=[SimpleNamespace(name=c) for c in cols]))
    monkeypatch.setattr(io_utils, "decode_mask_uri", lambda uri, loader=None: mask)
    monkeypatch.setattr(sys, "path", list(sys.path))   # the UDF prepends a workspace root
    return ingestion._segmentation_to_obt_udf(
        pd.DataFrame([{"segmentation_id": "s1", "mask_uri": "m.npy", **row}]))


def test_backfill_stores_the_mean_over_the_mask_components(monkeypatch):
    assert _backfill(monkeypatch, _two_blobs()).loc[0, "area"] == 168.0


def test_backfill_leaves_an_empty_mask_null(monkeypatch):
    out = _backfill(monkeypatch, np.zeros((40, 40), dtype=np.uint8))
    assert out[list(obt_schema.MORPHO_NAMES)].isna().to_numpy().all()

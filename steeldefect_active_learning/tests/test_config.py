"""Unit tests for config loader and YAML schema validation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from steeldefect_active_learning.config import load_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loads_minimal_yaml(tmp_path):
    cfg_path = _write(tmp_path, """
        data:
          image_dir: "imgs"
        ab_testing:
          alpha: 0.05
        runtime:
          seed: 7
    """)
    cfg = load_config(str(cfg_path))
    assert cfg.runtime.seed == 7
    assert cfg.ab_testing.alpha == 0.05


def test_unknown_key_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          alpaha: 0.05  # typo
    """)
    with pytest.raises(ValueError, match="Unknown YAML key"):
        load_config(str(cfg_path))


def test_unknown_top_level_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        nonsense:
          x: 1
    """)
    with pytest.raises(ValueError, match="top-level"):
        load_config(str(cfg_path))


def test_alpha_out_of_range_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          alpha: 5.0
    """)
    with pytest.raises(ValueError, match="alpha"):
        load_config(str(cfg_path))


def test_invalid_correction_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          multiple_testing_correction: "xyz"
    """)
    with pytest.raises(ValueError, match="multiple_testing_correction"):
        load_config(str(cfg_path))


def test_invalid_sequential_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          sequential: "bogus"
    """)
    with pytest.raises(ValueError, match="sequential"):
        load_config(str(cfg_path))


def test_log_level_validated(tmp_path):
    cfg_path = _write(tmp_path, """
        runtime:
          log_level: "VERBOSE"
    """)
    with pytest.raises(ValueError, match="log_level"):
        load_config(str(cfg_path))


def test_known_extra_section_does_not_raise(tmp_path):
    """flatness_clustering is consumed elsewhere — shouldn't trip the schema."""
    cfg_path = _write(tmp_path, """
        flatness_clustering:
          rocket_n_kernels: 100
    """)
    cfg = load_config(str(cfg_path))
    assert cfg is not None

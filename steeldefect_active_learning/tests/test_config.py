"""Unit tests for config loader and YAML schema validation."""
from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest
import yaml

from steeldefect_active_learning.ab_testing.promotion import decide_promotion
from steeldefect_active_learning.config import SHIPPED_CONFIG_PATH, load_config


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


_SHIPPED = Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml"


def test_shipped_default_config_loads():
    """The YAML shipped with the package must load through its own schema."""
    cfg = load_config(str(_SHIPPED))
    assert cfg.anomaly.detector == "patchcore"
    assert cfg.anomaly.per_class_fpr is True
    pr = cfg.ab_testing.promotion
    assert pr.expected_classes == list(range(11))
    assert pr.class_margins == {3: 0.03, 5: 0.03, 8: 0.03}
    assert pr.class_false_alarm_margins == {3: 0.002, 5: 0.002, 8: 0.002}
    assert (pr.noninferiority_margin, pr.false_alarm_margin, pr.min_gain) == (0.05, 0.005, 0.0)
    assert (pr.min_detectable_gain, pr.planning_discordance, pr.power) == (0.02, 0.10, 0.80)
    assert pr.require_no_class_regression is True
    assert SHIPPED_CONFIG_PATH == str(_SHIPPED)


def test_shipped_prior_is_the_generator_prior():
    from steeldefect_active_learning.data.generate_sample_data import (
        CLASS_IDS, STEELDEFECT_CLASS_PRIOR,
    )
    prior = load_config(str(_SHIPPED)).ab_testing.promotion.class_prior
    assert prior == {CLASS_IDS[name]: share for name, share in STEELDEFECT_CLASS_PRIOR.items()}


def test_gate_kwargs_are_the_gate_parameters():
    params = inspect.signature(decide_promotion).parameters
    keyword_only = {n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
    assert set(load_config(str(_SHIPPED)).ab_testing.gate_kwargs()) == keyword_only


def test_minimal_yaml_leaves_the_gate_unconfigured(tmp_path):
    cfg = load_config(str(_write(tmp_path, "ab_testing:\n  alpha: 0.05\n")))
    kwargs = cfg.ab_testing.gate_kwargs()
    assert kwargs["expected_classes"] is None and kwargs["class_prior"] is None


def test_retired_keys_raise_with_their_replacement(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          power: 0.8
          mde: 0.05
          promotion:
            min_cohens_d: 0.2
            bayesian_threshold: 0.95
            no_class_regression: true
    """)
    with pytest.raises(ValueError) as err:
        load_config(str(cfg_path))
    message = str(err.value)
    for key, replacement in [
        ("ab_testing.power", "promotion.power"),
        ("ab_testing.mde", "promotion.min_detectable_gain"),
        ("ab_testing.promotion.min_cohens_d", "removed"),
        ("ab_testing.promotion.bayesian_threshold", "removed"),
        ("ab_testing.promotion.no_class_regression", "require_no_class_regression"),
    ]:
        assert f"{key}: " in message and replacement in message


def test_unknown_promotion_key_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        ab_testing:
          promotion:
            min_cohen_d: 0.2  # typo
    """)
    with pytest.raises(ValueError, match="min_cohen_d"):
        load_config(str(cfg_path))


@pytest.mark.parametrize("promotion, key", [
    ({"min_gain": -0.01}, "min_gain"),
    ({"noninferiority_margin": 0}, "noninferiority_margin"),
    ({"class_margins": {3: 1.5}}, "class_margins"),
    ({"expected_classes": []}, "expected_classes"),
    ({"expected_classes": [0, 1], "class_prior": {0: 0.5}}, "class_prior"),
    ({"class_prior": {0: 1.0}}, "class_prior"),
    ({"false_alarm_margin": 1.0}, "false_alarm_margin"),
    ({"min_detectable_gain": 0.0}, "min_detectable_gain"),
    ({"planning_discordance": 0.01}, "planning_discordance"),
    ({"planning_discordance": 1.5}, "planning_discordance"),
    ({"power": 0.3}, "power"),
    ({"power": 1.0}, "power"),
    ({"expected_classes": list(range(11)), "class_prior": {k: 0.0 for k in range(11)}},
     "class_prior"),
    ({"expected_classes": [0, "1"]}, "expected_classes"),
    ({"expected_classes": list(range(11)), "class_prior": {k: 10 ** 308 for k in range(11)}},
     "class_prior"),
    ({"require_no_class_regression": 1}, "require_no_class_regression"),
])
def test_promotion_ranges(tmp_path, promotion, key):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"ab_testing": {"promotion": promotion}}), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"promotion\.{key}"):
        load_config(str(cfg_path))


@pytest.mark.parametrize("alpha", [0.5, 0.02, 0.01])
def test_alpha_must_lie_in_the_checked_range(tmp_path, alpha):
    # The exact size of the superiority clause exceeds alpha at 0.02 and at 0.01 (1.001 and
    # 1.12 alpha); 0.025 keeps a margin above the first exceedance found.
    with pytest.raises(ValueError, match=r"ab_testing\.alpha must be a number in \[0\.025, 0\.5\)"):
        load_config(str(_write(tmp_path, f"ab_testing:\n  alpha: {alpha}\n")))


def test_alpha_at_the_bottom_of_the_range_loads(tmp_path):
    assert load_config(str(_write(tmp_path, "ab_testing:\n  alpha: 0.025\n"))).ab_testing.alpha == 0.025


def test_invalid_anomaly_detector_raises(tmp_path):
    cfg_path = _write(tmp_path, """
        anomaly:
          detector: "isolation_forest"
    """)
    with pytest.raises(ValueError, match=r"anomaly\.detector must be one of"):
        load_config(str(cfg_path))

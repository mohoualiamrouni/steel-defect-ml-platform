import pytest

from steeldefect_ssl.config import (SSLPipelineConfig, apply_fast_mode, from_dict,
                              load_config, to_dict)


def test_defaults_load_and_roundtrip():
    cfg = load_config(None)
    assert isinstance(cfg, SSLPipelineConfig)
    assert cfg.data.image_size == 64
    assert cfg.ssl.method in ("simsiam", "simclr")
    # round-trip through dict
    cfg2 = from_dict(to_dict(cfg))
    assert to_dict(cfg2) == to_dict(cfg)


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        from_dict({"ssl": {"epochs": 10, "bogus_key": 1}})


def test_validation_catches_bad_values():
    from steeldefect_ssl.config import _validate
    cfg = SSLPipelineConfig()
    cfg.ssl.temperature = 5.0          # outside (0,1)
    with pytest.raises(ValueError):
        _validate(cfg)
    cfg = SSLPipelineConfig()
    cfg.discovery.k_range = (10, 4)    # lo > hi
    with pytest.raises(ValueError):
        _validate(cfg)


def test_fast_mode_shrinks_workload():
    cfg = SSLPipelineConfig()
    big_epochs, big_pool = cfg.ssl.epochs, cfg.data.n_total_unlabeled
    apply_fast_mode(cfg)
    assert cfg.ssl.epochs <= 5
    assert cfg.data.n_total_unlabeled <= big_pool
    assert cfg.openset.run_patchcore_baseline is False

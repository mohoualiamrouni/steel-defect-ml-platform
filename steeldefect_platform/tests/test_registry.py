"""promote_if_champion against a real MLflow registry: a local SQLite store, no mock."""
from __future__ import annotations

import itertools

import pytest

pytest.importorskip("mlflow")
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from steeldefect_platform import mlflow_registry as registry  # noqa: E402

_names = itertools.count()


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One registry for the module: creating the SQLite store takes about ten seconds."""
    root = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{(root / 'registry.db').as_posix()}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri), root


@pytest.fixture
def model(store):
    """A fresh registered model with versions 1, 2 and 3, and no alias."""
    client, root = store
    name = f"segmenter_{next(_names)}"
    client.create_registered_model(name)
    for v in (1, 2, 3):
        source = root / name / f"v{v}"
        source.mkdir(parents=True)
        client.create_model_version(name, source=str(source))
    return client, name


def test_promotes_over_the_champion_b_was_compared_with(model):
    client, name = model
    client.set_registered_model_alias(name, "champion", 1)
    out = registry.promote_if_champion(name, 2, expected_champion=1, client=client)
    assert out["applied"] is True
    assert (out["champion_before"], out["champion_after"]) == (1, 2)
    assert registry.alias_version(name, client=client) == 2


def test_refuses_when_the_champion_moved(model):
    # B was compared with v1, but v2 became the champion since: B never met v2.
    client, name = model
    client.set_registered_model_alias(name, "champion", 2)
    out = registry.promote_if_champion(name, 3, expected_champion=1, client=client)
    assert out["applied"] is False
    assert (out["champion_before"], out["champion_after"]) == (2, 2)
    assert out["detail"] == f"@champion of {name} is v2, but B was compared with v1"
    assert registry.alias_version(name, client=client) == 2


def test_first_promotion_when_no_version_is_champion(model):
    client, name = model
    out = registry.promote_if_champion(name, 1, expected_champion=None, client=client)
    assert out["applied"] is True and (out["champion_before"], out["champion_after"]) == (None, 1)


def test_refuses_a_first_promotion_when_a_champion_appeared(model):
    client, name = model
    client.set_registered_model_alias(name, "champion", 1)
    out = registry.promote_if_champion(name, 2, expected_champion=None, client=client)
    assert out["applied"] is False and registry.alias_version(name, client=client) == 1


def test_an_alias_already_on_b_counts_as_applied_with_nothing_written(model):
    # What a promotion resumed after a crash between the write and its record finds.
    client, name = model
    client.set_registered_model_alias(name, "champion", 2)
    out = registry.promote_if_champion(name, 2, expected_champion=1, client=client)
    assert out["applied"] is True and (out["champion_before"], out["champion_after"]) == (2, 2)
    assert out["detail"] == f"@champion of {name} already points at v2: nothing written"


def test_an_unknown_version_raises_and_moves_nothing(model):
    client, name = model
    client.set_registered_model_alias(name, "champion", 1)
    with pytest.raises(MlflowException):
        registry.promote_if_champion(name, 9, expected_champion=1, client=client)
    assert registry.alias_version(name, client=client) == 1


def test_alias_version_tells_an_unset_alias_from_a_missing_model(model):
    client, name = model
    assert registry.alias_version(name, client=client) is None
    with pytest.raises(MlflowException):
        registry.alias_version(f"{name}_missing", client=client)

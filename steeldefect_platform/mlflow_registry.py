"""Thin helpers over the MLflow Model Registry.

The platform versions four models (the report's "registre centralisé"):

* ``steeldefect_segmenter``        — the inspection CNN (A/B-gated promotion);
* ``steeldefect_defect_clusterer`` — subject 2 re-grouping model;
* ``steeldefect_severity_xgb``     — subject 3 severity regressor (+ encoder + CQR);
* ``steeldefect_flatness_clusterer`` — the flatness deployment-pattern pilot.

Mirrors the register/load idiom already used in
``flatness_clustering/databricks/{02_clustering_training,03_online_assignment}.py``.
``mlflow`` is imported lazily so this module loads without it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Registered model names — the single place they are spelled.
SEGMENTER = "steeldefect_segmenter"
DEFECT_CLUSTERER = "steeldefect_defect_clusterer"
SEVERITY_XGB = "steeldefect_severity_xgb"
FLATNESS_CLUSTERER = "steeldefect_flatness_clusterer"
SSL_ENCODER = "steeldefect_ssl_encoder"   # P4 self-supervised foundation encoder

ALL_MODELS = (SEGMENTER, DEFECT_CLUSTERER, SEVERITY_XGB, FLATNESS_CLUSTERER, SSL_ENCODER)


#: Default registry alias (MLflow >= 2.9 replaced the deprecated Stage API with
#: named aliases). "champion" is the served/production version; "challenger" is
#: the A/B candidate.
PROD_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def _mlflow():
    import mlflow  # lazy
    return mlflow


def _log():
    from steeldefect_platform.logging_utils import get_logger
    return get_logger("steeldefect.mlflow")


def load_production(model_name: str, alias: str = PROD_ALIAS,
                    stage: str = "Production"):
    """Load the served model by registry **alias** (e.g. ``@champion``).

    Falls back to the deprecated stage URI for models registered before the
    alias migration, then raises a clear ``RuntimeError`` if neither resolves —
    same failure-mode contract as flatness ``03_online_assignment``.
    """
    try:
        mlflow = _mlflow()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"MLflow is not available: {exc}") from exc

    for uri in (f"models:/{model_name}@{alias}", f"models:/{model_name}/{stage}"):
        try:
            try:
                return mlflow.sklearn.load_model(uri)
            except Exception:
                return mlflow.pyfunc.load_model(uri)
        except Exception as exc:
            _log().warning("could not load '%s' (%s); trying next resolver", uri, exc)
    raise RuntimeError(
        f"Could not load model '{model_name}' by alias @{alias} or stage {stage}. "
        "Ensure the training/promotion job ran and registered the model."
    )


def register(run, model, model_name: str, flavor: str = "sklearn") -> str:
    """Log *model* under the active *run* and register it; return the model URI."""
    mlflow = _mlflow()
    artifact_path = "model"
    if flavor == "sklearn":
        mlflow.sklearn.log_model(model, artifact_path, registered_model_name=model_name)
    else:
        mlflow.pyfunc.log_model(artifact_path, python_model=model,
                                registered_model_name=model_name)
    run_id = getattr(getattr(run, "info", run), "run_id", run)
    _log().info("registered %s (run %s)", model_name, run_id)
    return f"runs:/{run_id}/{artifact_path}"


def promote(model_name: str, version: int, alias: str = PROD_ALIAS,
            to_stage: str = "Production", archive_existing: bool = True) -> None:
    """Point the production **alias** at a model version, whatever it points at now.

    An unconditional write: the A/B gate does not use it (it calls
    :func:`promote_if_champion`, which first checks that the alias still points at the
    model B was compared with). The registry round trip of
    ``demo_industrialization.demo_registry`` uses it, and a rollback does the same write
    by hand (DEPLOY_RUNBOOK.md). Uses the alias API (MLflow >= 2.9); falls back to the
    deprecated stage transition on older MLflow, logging which path was taken rather than
    failing silently.
    """
    mlflow = _mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        client.set_registered_model_alias(model_name, alias, int(version))
        _log().info("promoted %s v%s -> @%s", model_name, version, alias)
    except AttributeError:  # MLflow < 2.9 has no alias API
        client.transition_model_version_stage(
            name=model_name, version=int(version), stage=to_stage,
            archive_existing_versions=archive_existing,
        )
        _log().warning("alias API unavailable; used deprecated stage %s for %s v%s",
                       to_stage, model_name, version)


def use_registry(registry_uri: str) -> None:
    """Point MLflow at a model registry, e.g. ``databricks-uc`` (Unity Catalog), where a
    model name is ``catalog.schema.model``: a short name is completed with the session's
    current catalog and schema, which is not the model the repository registers."""
    _mlflow().set_registry_uri(registry_uri)


def alias_version(model_name: str, alias: str = PROD_ALIAS,
                  client: Optional[Any] = None) -> Optional[int]:
    """The version that ``alias`` points at, or None when the alias is not set.

    Reads the registered model's alias map, so an unset alias is told apart from a
    registry error: a missing model or an unreachable registry raises. ``client`` is an
    ``MlflowClient`` (default: one on the configured registry).
    """
    client = client if client is not None else _mlflow().tracking.MlflowClient()
    version = client.get_registered_model(model_name).aliases.get(alias)
    return None if version is None else int(version)


def promote_if_champion(model_name: str, version: int, expected_champion: Optional[int],
                        alias: str = PROD_ALIAS,
                        client: Optional[Any] = None) -> Dict[str, Any]:
    """Point ``alias`` at ``version`` only if it still points at ``expected_champion``.

    The A/B gate compared B with one model, A, the champion when the slice was scored. If
    the alias has moved since (a newer champion, or a rollback), B was never compared with
    the model it would replace, and it is not promoted. ``expected_champion`` None means
    that no version held the alias: the first promotion of a model. An alias that already
    points at ``version`` counts as applied, with nothing written: a promotion resumed
    after a crash finds it there.

    Returns ``{"applied", "champion_before", "champion_after", "detail"}``; a registry error
    (an unknown version, an unreachable registry) raises, so that the caller records it.
    MLflow has no atomic compare-and-set on aliases: the alias is read, then written, then
    read back, and a change made by someone else between the read and the write is not
    detected. The bundle's A/B job allows one run at a time, which covers that one
    deployed job only: another target, or a manual write, can still interleave.
    """
    version = int(version)
    expected = None if expected_champion is None else int(expected_champion)
    client = client if client is not None else _mlflow().tracking.MlflowClient()
    before = alias_version(model_name, alias, client)
    if before == version:
        return {"applied": True, "champion_before": before, "champion_after": before,
                "detail": f"@{alias} of {model_name} already points at v{version}: nothing written"}
    if before != expected:
        return {"applied": False, "champion_before": before, "champion_after": before,
                "detail": (f"@{alias} of {model_name} is "
                           f"{'unset' if before is None else f'v{before}'}, but B was compared "
                           f"with {'no champion' if expected is None else f'v{expected}'}")}
    client.set_registered_model_alias(model_name, alias, version)
    after = alias_version(model_name, alias, client)
    if after != version:
        raise RuntimeError(f"@{alias} of {model_name} reads v{after} after it was set to "
                           f"v{version}")
    _log().info("promoted %s v%s -> @%s (was %s)", model_name, version, alias,
                "unset" if before is None else f"v{before}")
    return {"applied": True, "champion_before": before, "champion_after": after,
            "detail": (f"@{alias} of {model_name}: "
                       f"{'unset' if before is None else f'v{before}'} -> v{after}")}


def latest_uri(model_name: str, alias: str = PROD_ALIAS,
               stage: str = "Production") -> Optional[str]:
    """Return the ``models:/`` URI of the served version (alias, then stage), or None."""
    try:
        mlflow = _mlflow()
        client = mlflow.tracking.MlflowClient()
        try:
            client.get_model_version_by_alias(model_name, alias)
            return f"models:/{model_name}@{alias}"
        except Exception:
            versions = client.get_latest_versions(model_name, stages=[stage])
            return f"models:/{model_name}/{stage}" if versions else None
    except Exception as exc:
        _log().warning("latest_uri('%s') unavailable: %s", model_name, exc)
        return None

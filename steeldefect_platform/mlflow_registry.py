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

from typing import Optional

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
    """Point the production **alias** at a model version (alias-based promotion).

    Called ONLY by the A/B promotion gate (``02_ab_decision`` /
    ``ab_testing.promotion.gate_and_register``) for ``steeldefect_segmenter`` and by
    the offline training jobs for the other models. Uses the modern alias API
    (MLflow >= 2.9); falls back to the deprecated stage transition on older
    MLflow, logging which path was taken rather than failing silently.
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

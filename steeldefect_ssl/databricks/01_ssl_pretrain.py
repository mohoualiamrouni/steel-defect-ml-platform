"""Databricks task: pretrain the SSL foundation encoder on the unlabeled stream.

In production the unlabeled crops come from the OBT (``crop_uri``); here they are
the calibrated-synthetic pool (``source_version='synthetic-cal-v1'``). The
heavy lifting is the CPU ``steeldefect_ssl`` code; Spark/MLflow are used only for I/O
and registration (both best-effort so the script also runs locally).

    python steeldefect_ssl/databricks/01_ssl_pretrain.py --run_id ssl-2026-06 --epochs 20
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_DEFAULT_CFG = os.path.join(_REPO, "steeldefect_ssl", "config", "pipeline_config.yaml")


def main(spark=None, run_id: str = "ssl-dev", epochs: int = 20,
         config: str = _DEFAULT_CFG, **kw) -> dict:
    from steeldefect_ssl.config import load_config
    from steeldefect_ssl.data.pool_builder import build_pools
    from steeldefect_ssl.ssl.trainer import train_encoder

    cfg = load_config(config)
    cfg.ssl.epochs = int(epochs)
    cfg.ssl.compare = False                       # the comparison run is for the report, not prod
    pools = build_pools(cfg.data)
    bundle = train_encoder(cfg, pools["unlabeled"]["path"].tolist(),
                           device=cfg.runtime.device, log=print)
    bundle.save(cfg.ssl.checkpoint)
    print(f"[ssl_pretrain] encoder -> {cfg.ssl.checkpoint} "
          f"(final_loss={bundle.history.get('final_loss')})")

    # best-effort MLflow registration of the torch encoder
    try:
        import mlflow
        import mlflow.pytorch
        from steeldefect_platform.mlflow_registry import SSL_ENCODER
        with mlflow.start_run(run_name=f"ssl_pretrain_{run_id}"):
            mlflow.log_params({"method": cfg.ssl.method, "epochs": cfg.ssl.epochs,
                               "embed_dim": cfg.encoder.embed_dim, "run_id": run_id})
            mlflow.log_metric("final_loss", float(bundle.history.get("final_loss") or 0.0))
            mlflow.pytorch.log_model(bundle.build("cpu"), "model",
                                     registered_model_name=SSL_ENCODER)
        print(f"[ssl_pretrain] registered MLflow model '{SSL_ENCODER}'")
    except Exception as exc:
        print(f"[ssl_pretrain] MLflow registration skipped: {exc}")

    return {"run_id": run_id, "checkpoint": cfg.ssl.checkpoint,
            "final_loss": bundle.history.get("final_loss")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="ssl-dev")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    main(None, run_id=a.run_id, epochs=a.epochs)

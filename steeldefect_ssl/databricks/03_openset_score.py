"""Databricks task: score the stream for novelty -> ``steeldefect.novelty_candidates``.

Loads the SSL encoder, fits the open-set scorer on the known labeled crops,
scores the stream (known + held-out novel family), and writes the novelty
candidates to the Delta companion table when a Spark session is supplied
(best-effort; the computation itself is CPU-local). Returns the open-set
summary (AUROC / FPR95 / #flagged) — the H4-B evidence.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_DEFAULT_CFG = os.path.join(_REPO, "steeldefect_ssl", "config", "pipeline_config.yaml")


def main(spark=None, since: str = "2026-01-01", run_id: str = "ssl-dev",
         config: str = _DEFAULT_CFG, **kw) -> dict:
    import numpy as np

    from steeldefect_ssl import SOURCE_SYNTHETIC
    from steeldefect_ssl.config import load_config
    from steeldefect_ssl.data.pool_builder import build_pools
    from steeldefect_ssl.data.ssl_dataset import LabeledEvalDataset
    from steeldefect_ssl.models.encoder import embed_dataset
    from steeldefect_ssl.openset.evaluate import auroc_aupr_fpr95
    from steeldefect_ssl.openset.scorers import build_scorer
    from steeldefect_ssl.ssl.trainer import EncoderBundle

    cfg = load_config(config)
    pools = build_pools(cfg.data)
    encoder = EncoderBundle.load(cfg.ssl.checkpoint).build(cfg.runtime.device)

    def emb(df):
        X, y, _ = embed_dataset(
            encoder, LabeledEvalDataset(df, None, cfg.data.image_size), cfg.runtime.device)
        return X, y

    X_lab, y_lab = emb(pools["labeled"])
    X_stream, _ = emb(pools["stream"])
    is_novel = pools["stream"]["is_novel"].to_numpy().astype(bool)

    scorer = build_scorer(cfg.openset.scorer, cfg.openset.shrinkage,
                          cfg.openset.knn_k, cfg.runtime.seed)
    scorer.fit(X_lab, y_lab)
    scores = scorer.score(X_stream)
    metrics = auroc_aupr_fpr95(scores, is_novel)
    # flag the top novelty scores (one-sided threshold at the known 95th pct)
    thr = float(np.percentile(scores[~is_novel], 95)) if (~is_novel).any() else float(np.median(scores))
    flagged = scores >= thr
    n_flag = int(flagged.sum())
    print(f"[openset_score] AUROC={metrics['auroc']:.3f} FPR95={metrics['fpr95']:.3f} "
          f"flagged={n_flag}/{len(scores)}")

    # best-effort write to the Delta companion table
    if spark is not None:
        try:
            now = _dt.datetime.utcnow()
            rows = [
                (str(p), str(p), cfg.ssl.checkpoint, cfg.openset.scorer,
                 float(s), bool(f), -1, run_id, SOURCE_SYNTHETIC, now)
                for p, s, f in zip(pools["stream"]["path"], scores, flagged)
            ]
            cols = ["segmentation_id", "image_id", "encoder_uri", "scorer",
                    "novelty_score", "is_novel_flag", "proposed_family_id",
                    "run_id", "source_version", "created_ts"]
            (spark.createDataFrame(rows, cols)
                  .write.format("delta").mode("append")
                  .saveAsTable("steeldefect.novelty_candidates"))
            print(f"[openset_score] wrote {len(rows)} rows -> steeldefect.novelty_candidates")
        except Exception as exc:
            print(f"[openset_score] Delta write skipped: {exc}")

    return {"run_id": run_id, "n_scored": int(len(scores)), "n_flagged": n_flag, **metrics}


if __name__ == "__main__":
    main(None)

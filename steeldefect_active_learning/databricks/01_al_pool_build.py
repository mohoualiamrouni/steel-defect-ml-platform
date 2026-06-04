"""Databricks job 1 (Subject 1) — Active-Learning pool build.

Reads the unlabeled pool from the Delta OBT, scores it with the existing
acquisition machinery (no CNN re-inference — the 960-d inspection embedding is
already in the OBT), selects a budget of candidates with the phase-aware
strategy, and writes them to ``steeldefect.al_candidates`` for the Vision Toolbox
labeling UI.  After the expert labels them, a MERGE flips ``is_labeled`` on the
OBT rows.

Mirrors the structure of ``flatness_clustering/databricks/`` (sys.path /
STEELDEFECT_WORKSPACE_ROOT bootstrap, applyInPandas-free batch read, MLflow-tracked
model URI).  Spark is injected by Databricks.

    run:  main(spark, cycle_start="2026-05-01", budget=200)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

OBT_TABLE = "steeldefect.obt_segmentations"
CANDIDATES_TABLE = "steeldefect.al_candidates"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def read_unlabeled_pool(spark, cycle_start: str, limit: Optional[int] = None):
    """Return the unlabeled-pool rows (local defects) since *cycle_start*."""
    q = (
        spark.table(OBT_TABLE)
        .where(
            f"is_labeled = false AND defect_kind = 'local' "
            f"AND date_key >= '{cycle_start}' AND cnn_vector IS NOT NULL"
        )
        .select("segmentation_id", "image_id", "crop_uri", "cnn_vector",
                "steeldefect_class_conf", "expert_group_id", "defect_type")
    )
    return q.limit(limit) if limit else q


def score_and_select(pool_pdf, budget: int, strategy: str = "auto",
                     n_classes: int = 28, random_state: int = 42):
    """Score the pool and select *budget* candidates.

    Operates on the precomputed 960-d ``cnn_vector`` (no GPU forward pass):
    builds per-id uncertainty + embedding dicts and reuses the existing
    :func:`candidate_selector.select_candidates` (phase dispatcher when
    ``strategy='auto'``).  Returns a candidates DataFrame (pandas).
    """
    _bootstrap_path()
    import numpy as np
    import pandas as pd
    from steeldefect_active_learning.active_learning.candidate_selector import (
        _auto_budget_strategy, select_candidates,
    )
    from steeldefect_active_learning.active_learning.clustering import cluster_embeddings

    ids = pool_pdf["segmentation_id"].tolist()
    embs = {i: np.asarray(v, dtype=np.float32)
            for i, v in zip(ids, pool_pdf["cnn_vector"])}
    # Uncertainty proxy from the inspection on-line confidence already in the OBT
    # (1 - max softmax); cheap and requires no re-inference.
    conf = pool_pdf["steeldefect_class_conf"].fillna(0.5).to_numpy()
    unc = {i: float(1.0 - c) for i, c in zip(ids, conf)}
    # Cluster the embeddings for stratified / diverse selection.
    clusters = cluster_embeddings(embs, method="kmeans", k=min(n_classes, len(ids)))

    if strategy == "auto":
        strategy = _auto_budget_strategy(budget=budget, n_labeled=0,
                                         n_classes=n_classes)
    selected = select_candidates(
        uncertainty_scores=unc,
        cluster_assignments=clusters,
        labeling_budget=budget,
        method=strategy,
        embeddings_dict=embs,
        random_state=random_state,
    )
    sel_set = set(selected)
    rows = []
    for rank, i in enumerate(selected):
        rows.append({
            "segmentation_id": i,
            "image_id": pool_pdf.set_index("segmentation_id").loc[i, "image_id"]
                        if i in sel_set else None,
            "strategy": strategy,
            "score_uncertainty": unc[i],
            "score_diversity": 0.0,
            "score_anomaly": 0.0,
            "score_total": unc[i],
            "rank": rank,
            "selected": True,
        })
    return pd.DataFrame(rows)


def write_al_candidates(spark, scored_pdf, al_cycle: int, model_uri: str) -> None:
    """Append the scored candidates to ``steeldefect.al_candidates``."""
    import datetime as _dt
    scored_pdf = scored_pdf.copy()
    scored_pdf["al_cycle"] = int(al_cycle)
    scored_pdf["model_uri"] = model_uri
    scored_pdf["created_ts"] = _dt.datetime.utcnow().isoformat()
    sdf = spark.createDataFrame(scored_pdf)
    sdf.write.format("delta").mode("append").saveAsTable(CANDIDATES_TABLE)


def main(spark, cycle_start: str, budget: int = 200, strategy: str = "auto",
         al_cycle: int = 0, model_uri: str = "models:/steeldefect_segmenter/Production"
         ) -> int:
    _bootstrap_path()
    pool = read_unlabeled_pool(spark, cycle_start)
    pool_pdf = pool.toPandas()
    if pool_pdf.empty:
        print(f"[al_pool_build] no unlabeled rows since {cycle_start}")
        return 0
    scored = score_and_select(pool_pdf, budget=budget, strategy=strategy)
    write_al_candidates(spark, scored, al_cycle=al_cycle, model_uri=model_uri)
    print(f"[al_pool_build] wrote {len(scored)} candidates to {CANDIDATES_TABLE} "
          f"(cycle {al_cycle}, strategy {scored['strategy'].iloc[0]})")
    return len(scored)

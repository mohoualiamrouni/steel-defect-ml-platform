"""Databricks task: refresh embeddings with the SSL encoder + report uplift.

Loads the encoder trained by ``01_ssl_pretrain``, re-embeds the labeled/eval
crops and reports the linear-probe accuracy and clustering NMI (the H4-A
evidence that the SSL embedding beats raw features). In production this is where
``steeldefect.obt_segmentations.cnn_vector`` would be refreshed; here it reports the
quality gain. CPU-only; runs without Spark.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_DEFAULT_CFG = os.path.join(_REPO, "steeldefect_ssl", "config", "pipeline_config.yaml")


def main(spark=None, run_id: str = "ssl-dev", config: str = _DEFAULT_CFG, **kw) -> dict:
    import numpy as np
    from sklearn.cluster import KMeans

    from steeldefect_ssl.config import load_config
    from steeldefect_ssl.data.pool_builder import build_pools, known_classes
    from steeldefect_ssl.data.ssl_dataset import LabeledEvalDataset
    from steeldefect_ssl.eval.probes import probe_representation
    from steeldefect_ssl.models.encoder import embed_dataset
    from steeldefect_ssl.ssl.trainer import EncoderBundle

    cfg = load_config(config)
    pools = build_pools(cfg.data)
    encoder = EncoderBundle.load(cfg.ssl.checkpoint).build(cfg.runtime.device)

    def emb(df):
        X, y, _ = embed_dataset(
            encoder, LabeledEvalDataset(df, None, cfg.data.image_size), cfg.runtime.device)
        return X, y

    X_lab, y_lab = emb(pools["labeled"])
    X_eval, y_eval = emb(pools["eval"])
    probe = probe_representation(X_lab, y_lab, X_eval, y_eval, cfg.eval.knn_k, cfg.runtime.seed)
    k = len(known_classes(cfg.data.novel_classes))
    lbl = KMeans(n_clusters=k, n_init=10, random_state=cfg.runtime.seed).fit_predict(X_eval)
    from sklearn.metrics import normalized_mutual_info_score
    nmi = float(normalized_mutual_info_score(y_eval, lbl))

    res = {"run_id": run_id, "linear_probe_acc": probe["linear_probe_acc"],
           "knn_acc": probe["knn_acc"], "cluster_nmi": nmi,
           "encoder": cfg.ssl.checkpoint}
    print(f"[embedding_refresh] linear={res['linear_probe_acc']:.3f} "
          f"knn={res['knn_acc']:.3f} cluster_nmi={nmi:.3f}")
    return res


if __name__ == "__main__":
    main(None)

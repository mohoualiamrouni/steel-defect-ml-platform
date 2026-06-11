# steeldefect_ssl — P4: Cœur perceptif auto-supervisé (CPU)

The **fourth pipeline** of the SteelDefect ecosystem. Where P1/P2/P3 are
*reactive and downstream* of the inspection CNN embedding, P4 turns perception into a
**self-improving loop**:

```
SSL encoder on the unlabeled stream
  → better embeddings for P1 (active learning) & P2 (clustering)
  → open-set detection of NOVEL defect families outside the 28 expert groups
  → generalized category discovery + continual integration (no forgetting)
```

**Hypothesis H4** — an encoder pretrained self-supervised on the unlabeled
stream (a) improves P2 clustering (NMI↑) and (b) detects a held-out novel defect
family (open-set AUROC) that the fixed 28-group taxonomy misses.

All data is **calibrated-synthetic** (`source_version='synthetic-cal-v1'`,
seed 42). **CPU-only**, no new dependencies beyond the repo `requirements.txt`.

## The four stages
1. **Self-supervised pretraining** — SimSiam (primary, CPU-friendly: no
   negatives, collapse-safe) + SimCLR/NT-Xent (comparison; ties to the InfoNCE
   theory in the report). Compact `SSLResNet` (1-ch 64×64 → 256-D, ~1.3M params).
2. **Embedding eval / refresh (H4-A)** — linear & kNN probes vs random-init and
   raw-pixel-PCA baselines; clustering uplift (NMI/ARI + bootstrap CIs, reusing
   P2's `compute_group_metrics`).
3. **Open-set / OOD (H4-B)** — Mahalanobis (DDU) / kNN / energy scorers on the
   SSL features; AUROC / AUPR / FPR95 on known-vs-novel; **PatchCore baseline**
   bridged from P1.
4. **GCD + continual** — seeded semi-supervised k-means (Vaze GCD), K via
   `gap_statistic` / DeepDPM, IF+LOF cleanup; NCM/iCaRL prototypes integrate the
   discovered novel family with near-zero forgetting.

## Run it (CPU)
```bash
# smoke (~4-5 min): proves the architecture end-to-end
python -m steeldefect_ssl.pipeline.run_pipeline --config steeldefect_ssl/config/pipeline_config.yaml

# full best-results run (~20-25 min): the paper numbers
python -m steeldefect_ssl.pipeline.run_pipeline --config steeldefect_ssl/config/pipeline_config.yaml --full

# tests (toy-scale, < 15 s)
python -m pytest steeldefect_ssl/tests -q
```
Use the **`python`** interpreter (3.12 with torch 2.7+cpu), not `python3`.

## Outputs (`steeldefect_ssl/outputs/`)
* `steeldefect_ssl_metrics.json` — single source of truth (all metrics + H4 verdict).
* CSVs: `pretrain_curve`, `probe_results`, `clustering_uplift`, `openset_results`,
  `gcd_results`, `continual_results`.
* `plots/` — 9 figures: SSL loss, embedding UMAP-style map, probe bars, clustering
  uplift, open-set score hist, ROC/PR, GCD clusters, GCD outliers, continual forgetting.

## H4 acceptance bar (numbers flow from the JSON, never hand-edited)
* **H4-A**: `ssl.nmi_ci_low > pca.nmi` and `ssl.nmi − pca.nmi ≥ 0.05`; SSL linear
  probe > random-init & raw-pixel-PCA.
* **H4-B**: SSL-Mahalanobis AUROC ≥ 0.85, FPR95 ≤ 0.30, and SSL beats the
  PatchCore baseline AUROC.
* **Corollary**: GCD recovers the novel family; NCM forgetting ≈ 0.
A target not met is reported `FAIL` — no massaging.

## Platform (Databricks / OBT)
* MLflow model `steeldefect_ssl_encoder`; companion table `steeldefect.novelty_candidates`.
* DAG nodes `ssl_pretrain → embedding_refresh`, `openset_score` (`steeldefect_platform/jobs.py`,
  `run_task.py`), job `steeldefect_ssl_release` in `databricks.yml`.
* Overlays in `steeldefect_ssl/databricks/` (`01_ssl_pretrain`, `02_embedding_refresh`,
  `03_openset_score`) — `main(spark, …)`, CPU-local with best-effort Spark/MLflow I/O.

## Notes / environment
* Windows duplicate-OpenMP (torch vs sklearn) is handled in `steeldefect_ssl/__init__.py`
  (`KMP_DUPLICATE_LIB_OK`, `LOKY_MAX_CPU_COUNT`); the 9 figures are rendered in a
  separate single-thread subprocess (`eval/render_plots.py`) to keep the heap stable.
* Reference papers + code: `Documentation/P4_self_supervised_novelty/`.

# steeldefect_ssl — P4: Cœur perceptif auto-supervisé (CPU)

The **fourth pipeline** of the SteelDefect ecosystem. Where P1/P2/P3 are
*reactive and downstream* of the inspection CNN embedding, P4 is designed as a
**learning loop on the unlabeled stream**, tested as hypothesis H4 below:

```
SSL encoder on the unlabeled stream
  → better embeddings for P1 (active learning) & P2 (clustering)
  → open-set detection of NOVEL defect families held out of the known classes
  → generalized category discovery + continual integration (no forgetting)
```

**Hypothesis H4** — an encoder pretrained self-supervised on the unlabeled
stream (a) improves P2 clustering (NMI↑) and (b) detects held-out defect families
(open-set AUROC). By default the known classes are 8 of the 11 synthetic P1 classes and
the held-out ones are orange_peel, scale and edge_crack; on NEU-DET, 4 classes are known
and crazing and pitted_surface are held out.

The default configuration runs on **calibrated-synthetic** data
(`source_version='synthetic-cal-v1'`, seed 42);
`steeldefect_ssl/config/pipeline_config_neu.yaml` runs the same pipeline on the public
NEU-DET images (`source_version='neu-det-v1'`).
**CPU-only**, no new dependencies beyond the repo `requirements.txt`.

## The four stages
1. **Self-supervised pretraining** — SimCLR/NT-Xent, the configured method
   (`ssl.method: simclr`; ties to the InfoNCE theory in the report). SimSiam (no
   negatives) can run as a shorter comparison (`ssl.compare: true`, off by default).
   Compact `SSLResNet` (1-ch 64×64 → 256-D, ~1.3M params).
2. **Embedding eval / refresh (H4-A)** — linear & kNN probes vs random-init and
   raw-pixel-PCA baselines; clustering uplift (NMI/ARI + bootstrap CIs, reusing
   P2's `compute_group_metrics`).
3. **Open-set / OOD (H4-B)** — Mahalanobis (DDU) / kNN / cosine-prototype / energy
   scorers on the SSL features; AUROC / AUPR / FPR95 on known-vs-novel. The configured
   scorer (energy by default) is also run on random-init features as a baseline. A
   **PatchCore baseline** bridged from P1 exists but is off by default
   (`run_patchcore_baseline: false`) because it crashed on the CPU machine used for
   these runs.
4. **GCD + continual** — seeded semi-supervised k-means (Vaze GCD), K via
   `gap_statistic` / DeepDPM, IF+LOF cleanup. NCM/iCaRL-style prototypes then add each
   held-out novel family from half of its true-labelled stream samples (not from the
   GCD clusters) and measure forgetting on the known classes (target |Δ| ≤ 0.05).

## Run it (CPU)
```bash
# smoke (~4-5 min): proves the architecture end-to-end
python -m steeldefect_ssl.pipeline.run_pipeline --config steeldefect_ssl/config/pipeline_config.yaml

# full run (about 4.6 h on the CPU used): the reported numbers
python -m steeldefect_ssl.pipeline.run_pipeline --config steeldefect_ssl/config/pipeline_config.yaml --full

# tests (toy-scale, < 15 s)
python -m pytest steeldefect_ssl/tests -q
```
Use the **`python`** interpreter (Python 3.12 with a CPU build of torch), not `python3`.

## Outputs (`steeldefect_ssl/outputs/`)
* `steeldefect_ssl_metrics.json` — single source of truth (all metrics + H4 verdict).
* CSVs: `pretrain_curve`, `probe_results`, `clustering_uplift`, `openset_results`,
  `gcd_results`, `continual_results`.
* `plots/` — 9 figures: SSL loss, 2-D PCA map of the embedding, probe bars, clustering
  uplift, open-set score hist, ROC/PR, GCD clusters (PCA map), GCD outliers, continual
  forgetting. The `umap` in `embedding_umap.png` and `gcd_umap.png` predates the switch
  to PCA (`_embed2d` in `eval/plots.py` says why).

## H4 acceptance bar (numbers flow from the JSON, never hand-edited)
* **H4-A**: `ssl.nmi_ci_low > pca.nmi` and `ssl.nmi − pca.nmi ≥ 0.05`; SSL linear
  probe > random-init & raw-pixel-PCA.
* **H4-B**: with the configured scorer (energy by default), AUROC ≥ 0.85 and
  FPR95 ≤ 0.30; it must also beat the same scorer on random-init features, and the
  PatchCore baseline when that baseline is enabled.
* **Corollary**: GCD recovers at least one held-out novel family (its best cluster has
  purity ≥ 0.5 and recall ≥ 0.5); NCM forgetting |Δ| ≤ 0.05.
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

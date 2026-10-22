# SteelDefect: ML and MLOps platform for steel surface defect analytics

Coordinated machine learning pipelines on one shared Azure Databricks / Delta Lake
platform (the *One Big Table*, `steeldefect.obt_segmentations`), designed around an
in-line steel surface inspection system. Building on its output (a 960-d CNN embedding
and 24 morphology features per defect), they provide an active-learning loop with an A/B
promotion gate, a data-driven defect taxonomy and an optimized cutting plan. A
self-supervised core learns from the defect image crops themselves to detect novel defect
families, and a separate pipeline groups coils into flatness families from their flatness
profiles.

| Pipeline | What it does | Code |
|---|---|---|
| **P1: Active learning and A/B** | select the most informative images to label; decide model promotion with a fail-closed statistical gate | `steeldefect_active_learning/` |
| **P2: Defect regrouping** | fuse CNN and morphology features, discover defect families (mini-batch k-means by default; GMM, HDBSCAN, DP-GMM and a lightweight DeepDPM re-implementation as options) | `defect_clustering/` |
| **P3: Datamining and cutting** | severity regression (XGBoost with conformalized quantile regression), then a guillotine cutting-plan optimizer | `datamining/` |
| **P4: Self-supervised core** | self-supervised contrastive encoder (SimCLR / NT-Xent; an optional SimSiam comparison is off by default), open-set detection, generalized category discovery, continual NCM | `steeldefect_ssl/` |
| **Flatness clustering** | coil flatness profiles: statistical and ROCKET / MiniRocket features, k-means clustering (GMM, HDBSCAN and others as options) | `flatness_clustering/` |
| **Platform** | OBT schema, jobs DAG, MLflow registry, ingestion, drift monitoring | `steeldefect_platform/`, `databricks.yml` |

## Data honesty

No plant data is used in this repository. The synthetic data is **calibrated-synthetic**:
generated to mimic the inspection system, not sampled from plant data. The OBT rows follow
its schema and dimensions, but the defect names differ between generators: the OBT seeders
and P2 each use 28 groups, 19 of them shared, and the P1 and P4 images use 10 of the P2
groups plus `clean`. Some generators also follow the class imbalance documented for the
real inspection stream, and the flatness pipeline has its own generator of coil flatness
profiles. Rows written to the OBT and the synthetic P4 pools carry
`source_version = "synthetic-cal-v1"`. The P1 image generator writes this tag only in its
optional long-tail mode, so the P1 command-line run and
`steeldefect_active_learning/data/demo_ab_predictions.csv` carry none, and the standalone
generators of P2, P3 and the flatness pipeline do not write it yet. Runs on this data
validate the *architecture and methods*, not production numbers; a production feed would
carry `steeldefect-prod-*`.

## Results on public real data (NEU-DET)

P1 and P2 were also evaluated on NEU-DET, a public dataset of 1,800 hot-rolled steel
strip images in six defect classes (`run_neu_p1p2.py`, summary in
`outputs_neu_p1p2_summary.json`). Both use a frozen ImageNet EfficientNet-B0 embedding
(global average pooling, 1,280-d, L2-normalized) as a stand-in for the inspection CNN.
P2 also uses morphology features of an Otsu mask: 24 region descriptors (shape, Hu
moments, GLCM texture). The code requests a mean and max aggregation over the mask's
regions (48 columns), but the 0/255 mask is read as a single labelled region, so the two
halves are identical and only 24 values are distinct. This is a bug, to be fixed in a
separate change; the P2 numbers below were computed with it. Neither hypothesis is
confirmed on this dataset, and both results are reported as measured:

- **P2, feature fusion**: fusing morphology features with the CNN embedding gives an
  NMI of 0.8776 (95% bootstrap CI 0.8643 to 0.8963, 100 resamples), against 0.8765 for
  the CNN embedding alone. The +0.0011 difference is not significant.
- **P1, active learning**: with 100 labels, random sampling reaches a mean macro-F1 of
  0.9396 against 0.9034 for TypiClust (5 seeds), so active learning saves no labels at
  that budget (savings ratio 1.0). Over the whole curve, TypiClust's mean is above
  random's at 20 and 50 labels (0.5513 against 0.3851, 0.8627 against 0.8257), below from
  100 to 400 labels, and slightly above at 800 (0.9941 against 0.9896), where all five
  strategies average between 0.9896 and 0.9941. At low budgets these mean gaps are smaller
  than the spread between seeds: at 20 labels, TypiClust's standard deviation over the 5
  seeds is 0.2495, for a lead of 0.1662. Entropy and coreset sampling stay well below
  random up to 400 labels. The full curve, with standard deviations, is under `P1_curve`
  in the summary.

To reproduce: download the official NEU-DET archive, which is flat (`IMAGES/`,
`ANNOTATIONS/`), and sort the 1,800 images into one folder per class under
`steeldefect_ssl/data/neu_det/raw/`, named after the file-name prefix (`crazing`,
`inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`). Install
`opencv-python-headless`, used for the Otsu masks and not yet in `requirements.txt`, then
run `python run_neu_p1p2.py`. P1 and P2 use all 1,800 images; the split manifests in that
folder define the P4 open-set splits. NEU-DET has no published license: cite the three
works its authors list, K. Song and Y. Yan (Applied Surface Science, 2013), Y. He et al.
(IEEE Transactions on Instrumentation and Measurement, 2020) and Y. Bao et al. (IEEE
Transactions on Instrumentation and Measurement, 2021), and do not redistribute the
images.

## Quickstart

On a machine without a GPU, install the CPU build of PyTorch first, as CI does:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Then:

```bash
pip install -r requirements.txt pytest   # core ML deps (pyspark/delta come from the Databricks Runtime)
pytest -q                                # unit tests for P1, P4, the flatness pipeline and the platform (P2 and P3 have none yet)
python -m steeldefect_platform.demo_industrialization
```

The last command replays the platform's operational guarantees on synthetic data:
data-quality gate and dead-letter, idempotent ingest (Delta MERGE semantics emulated
in memory), drift detection (MMD, ADWIN), MLflow promotion and rollback against a local
registry, and the jobs DAG ordering. No Databricks workspace and no GPU are needed.

To deploy on Databricks, see [`DEPLOY_RUNBOOK.md`](DEPLOY_RUNBOOK.md)
(`databricks bundle validate / deploy / run`). CI runs a syntax check and the test
suites on every pull request and on every push to main (`.github/workflows/ci.yml`).

## Status

The repository implements MLOps patterns used in production systems (idempotent MERGE
ingestion with dead-letter and data-quality gate, jobs DAG, MLflow alias-based registry,
drift monitoring, A/B gate, serverless Databricks bundle, CI), validated on synthetic data
and on CPU. In the Databricks bundle, the seed, A/B and daily OBT tasks (7 of 12) run
simplified stand-in notebooks rather than the tested package (the A/B stand-in only
registers two model versions and runs no A/B test); the header of `databricks.yml`
explains why. At-scale Spark/Delta runs, real-data and GPU training,
Unity Catalog governance and live dashboards are the next steps.

## Layout

```
steeldefect_active_learning/  P1   defect_clustering/  P2   datamining/  P3
steeldefect_ssl/              P4   flatness_clustering/     coil flatness clustering
steeldefect_platform/  OBT schema, jobs DAG, MLflow registry, ingestion, drift, tests
databricks.yml         Databricks Asset Bundle (serverless jobs and schedules)
DEPLOY_RUNBOOK.md      deploy and operate
```

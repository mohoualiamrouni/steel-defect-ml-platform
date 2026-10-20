"""
Pipeline configuration loader.

Reads pipeline_config.yaml and exposes typed dataclasses for each section.

Usage:
    from steeldefect_active_learning.config import load_config
    cfg = load_config("steeldefect_active_learning/config/pipeline_config.yaml")
    print(cfg.data.image_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    image_dir: str = "steeldefect_active_learning/data/sample_images"
    image_size: List[int] = field(default_factory=lambda: [256, 256])
    batch_size: int = 32
    num_workers: int = 4
    extensions: List[str] = field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".npy", ".h5", ".hdf5"
    ])


@dataclass
class ModelConfig:
    arch: str = "resnet50"
    checkpoint: Optional[str] = None
    num_classes: int = 5
    task: str = "classification"   # "classification" or "segmentation"


@dataclass
class ActiveLearningConfig:
    uncertainty_method: str = "entropy"    # entropy | margin | least_confidence | bald
                                           # | mc_entropy | badge | ddu | ensemble
                                           # | cam_contradiction (Guélorget 2022)
    feature_layer: str = "avgpool"
    clustering_method: str = "kmeans"      # kmeans | hdbscan
    n_clusters: int = 10
    labeling_budget: int = 200
    # Phase-gated dispatcher (TypiClust → BADGE → BALD/SIMILAR).
    # When None, thresholds default to 2*n_classes and 10*n_classes (uniform).
    # When provided, thresholds are evaluated *per-class* — closes report L7
    # (rare defect classes need more annotations per class than majority classes).
    target_per_class: Optional[List[int]] = None
    # Calibration tracking — re-fit T* every cycle and flag ECE drift.
    online_calibration: bool = False


@dataclass
class PromotionConfig:
    """Thresholds of the A/B promotion gate (``ab_testing.promotion.decide_promotion``).

    Attributes:
        min_cohens_d:         Minimum |Cohen's d| for the difference to count.
        bayesian_threshold:   Minimum Bayesian P(B more accurate than A).
        no_class_regression:  Veto promotion if any class significantly regresses.
    """
    min_cohens_d: float = 0.2
    bayesian_threshold: float = 0.95
    no_class_regression: bool = True


@dataclass
class ABTestingConfig:
    """Statistical A/B-testing configuration (cascade in example-report.tex §3).

    Attributes:
        enabled:                       Run the A/B-testing pipeline.
        iou_threshold:                 IoU threshold for segmentation agreement.
        confidence_threshold:          Min confidence to count a prediction.
        demo_predictions_path:         CSV used when checkpoints are null (avoids
                                       degenerate 100% disagreement on random weights).
        alpha:                         Significance level (default 0.05).
        power:                         Target statistical power (default 0.80).
        mde:                           Minimum detectable effect for sample-size
                                       calculations (default 0.05).
        multiple_testing_correction:   "bonferroni" | "holm" | "benjamini-hochberg"
                                       | "none". Applied across the per-class McNemar
                                       suite (28 defect classes) — closes the gap
                                       where this YAML key was previously ignored.
        srm_threshold:                 χ² p-value below which the SRM gate flips and
                                       results are refused (default 1e-3).
        expected_arm_ratio:            Expected fraction in arm A (default 0.5).
        bootstrap_n:                   Number of bootstrap resamples for percentile/
                                       BCa CIs on Cohen's d and confidence-delta mean.
        cuped_covariate:               Column name to use as the CUPED covariate. None
                                       disables variance reduction. The covariate must
                                       be pre-treatment (e.g., previous-batch confidence).
        sequential:                    "msprt" | "obrien_fleming" | None. Enables
                                       always-valid sequential testing for continuous
                                       monitoring (Johari et al. 2017).
        bayesian_decision_threshold:   P(B > A) cutoff for ship/no-ship in the Bayesian
                                       layer (default 0.95).
        calibrate_before_comparison:   Temperature-scale both models before comparing
                                       (Minderer 2021).
        promotion:                     Thresholds of the promotion gate (nested section).
    """
    enabled: bool = True
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5
    demo_predictions_path: Optional[str] = None
    alpha: float = 0.05
    power: float = 0.80
    mde: float = 0.05
    multiple_testing_correction: str = "benjamini-hochberg"
    srm_threshold: float = 1e-3
    expected_arm_ratio: float = 0.5
    bootstrap_n: int = 1000
    cuped_covariate: Optional[str] = None
    sequential: Optional[str] = None
    bayesian_decision_threshold: float = 0.95
    calibrate_before_comparison: bool = False
    promotion: PromotionConfig = field(default_factory=PromotionConfig)


@dataclass
class RuntimeConfig:
    """Reproducibility and observability knobs.

    Attributes:
        seed:      Global RNG seed propagated to numpy / torch / random / sklearn.
                   None = non-deterministic.
        log_level: Python logging level for the root logger ('DEBUG'..'CRITICAL').
        log_file:  Optional file path for the file-handler. Console handler is always
                   on so existing CLI output is preserved.
    """
    seed: Optional[int] = 42
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class OutputConfig:
    dir: str = "steeldefect_active_learning/outputs"
    candidates_file: str = "candidates.csv"
    ab_report_file: str = "ab_report.csv"
    plots_dir: str = "steeldefect_active_learning/outputs/plots"


# ---------------------------------------------------------------------------
# Pipeline Active Learning (Roth & Small 2008) config
# ---------------------------------------------------------------------------

@dataclass
class PipelineALStageConfig:
    """Configuration for one stage in the multi-stage pipeline AL."""
    arch: str = "resnet18"
    checkpoint: Optional[str] = None
    num_classes: int = 2
    task: str = "classification"
    uncertainty_method: str = "entropy"
    label: str = "stage"


@dataclass
class PipelineALConfig:
    """Configuration for the Roth & Small 2008 pipeline active learning algorithm."""
    enabled: bool = False
    update_rate: float = 0.1       # lambda: how fast beta shifts between stages
    n_cycles: int = 5              # number of simulated labeling cycles
    budget_per_cycle: int = 30     # images selected per cycle
    stages: List[PipelineALStageConfig] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    """Post-hoc temperature scaling (Guo et al. 2017).

    When enabled, a TemperatureScaler is fitted on the validation set after
    the model is loaded. Calibrated probabilities are then used for all
    uncertainty scoring and A/B metric computation.

    Attributes:
        enabled:           Whether to run calibration. Requires val_labels_csv.
        val_labels_csv:    CSV with columns 'path' and 'label' for the val set.
        temperature_path:  Where to save/load the fitted temperature JSON.
        n_bins:            Number of ECE/MCE calibration bins (default 15).
    """
    enabled: bool = False
    val_labels_csv: Optional[str] = None
    temperature_path: str = "steeldefect_active_learning/outputs/temperature_T.json"
    n_bins: int = 15


@dataclass
class AnomalyConfig:
    """Anomaly detection for unsupervised new-defect discovery (Roth et al. 2022).

    When enabled, a PatchCore detector is fitted on images belonging to the
    'clean' (no-defect) class. Unlabeled images are then scored; high scorers
    may represent previously unseen defect types and are flagged in the
    candidates CSV.

    Attributes:
        enabled:           Whether to run anomaly detection.
        normal_class:      Class name (or label int) of the normal / clean class.
        coreset_ratio:     Fraction of memory bank to keep (default 0.01 = 1%).
        fpr:               False positive rate for threshold (default 0.05).
        model_path:        Where to save/load the fitted PatchCore model (pickle).
        backbone:          CNN backbone for feature extraction (default resnet18).
        device:            'cpu' or 'cuda'.
        detector:          'patchcore' or 'padim'. The CLI pipeline fits PatchCore
                           only; PaDiM has no threshold API yet.
        per_class_fpr:     One threshold per class via PatchCoreDetector.fit_per_class
                           (needs a labelled calibration set; not wired in the CLI
                           pipeline yet).
    """
    enabled: bool = False
    normal_class: str = "clean"
    coreset_ratio: float = 0.01
    fpr: float = 0.05
    model_path: str = "steeldefect_active_learning/outputs/patchcore.pkl"
    backbone: str = "resnet18"
    device: str = "cpu"
    detector: str = "patchcore"
    per_class_fpr: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model_a: ModelConfig = field(default_factory=ModelConfig)
    model_b: ModelConfig = field(default_factory=ModelConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    pipeline_al: PipelineALConfig = field(default_factory=PipelineALConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# YAML keys that are intentionally consumed outside the dataclass schema (e.g.,
# experiment-only sections that are not part of the typed pipeline config).
_KNOWN_TOPLEVEL_EXTRA_KEYS = {"flatness_clustering"}


def _dict_to_dataclass(cls, d: dict, *, _path: str = ""):
    """Recursively populate a dataclass from a dict.

    Unknown keys raise ValueError (catches YAML typos at load time).
    """
    if d is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
    unknown = set(d) - field_names
    if unknown:
        raise ValueError(
            f"Unknown YAML key(s) {sorted(unknown)} under '{_path or cls.__name__}'. "
            f"Valid keys: {sorted(field_names)}"
        )
    return cls(**{k: v for k, v in d.items() if k in field_names})


def _validate(cfg: PipelineConfig) -> None:
    """Lightweight invariant check on numeric ranges.

    Catches the worst class of YAML typos: values that load but make no statistical
    sense (e.g., alpha=5 instead of 0.05). Keeps the loader dependency-free
    (no Pydantic).
    """
    ab = cfg.ab_testing
    if not (0.0 < ab.alpha < 1.0):
        raise ValueError(f"ab_testing.alpha must be in (0,1), got {ab.alpha!r}")
    if not (0.0 < ab.power < 1.0):
        raise ValueError(f"ab_testing.power must be in (0,1), got {ab.power!r}")
    if not (0.0 < ab.expected_arm_ratio < 1.0):
        raise ValueError(
            f"ab_testing.expected_arm_ratio must be in (0,1), got {ab.expected_arm_ratio!r}"
        )
    if ab.multiple_testing_correction not in {
        "bonferroni", "holm", "benjamini-hochberg", "none",
    }:
        raise ValueError(
            "ab_testing.multiple_testing_correction must be one of "
            "{'bonferroni','holm','benjamini-hochberg','none'}, "
            f"got {ab.multiple_testing_correction!r}"
        )
    if ab.sequential not in {None, "msprt", "obrien_fleming"}:
        raise ValueError(
            "ab_testing.sequential must be one of {None,'msprt','obrien_fleming'}, "
            f"got {ab.sequential!r}"
        )
    if ab.bootstrap_n < 0:
        raise ValueError(f"ab_testing.bootstrap_n must be >= 0, got {ab.bootstrap_n!r}")
    if not (0.0 < ab.bayesian_decision_threshold < 1.0):
        raise ValueError(
            "ab_testing.bayesian_decision_threshold must be in (0,1), got "
            f"{ab.bayesian_decision_threshold!r}"
        )
    pr = ab.promotion
    if pr.min_cohens_d < 0:
        raise ValueError(
            f"ab_testing.promotion.min_cohens_d must be >= 0, got {pr.min_cohens_d!r}"
        )
    if not (0.0 < pr.bayesian_threshold < 1.0):
        raise ValueError(
            "ab_testing.promotion.bayesian_threshold must be in (0,1), got "
            f"{pr.bayesian_threshold!r}"
        )
    if cfg.anomaly.detector not in {"patchcore", "padim"}:
        raise ValueError(
            "anomaly.detector must be one of {'patchcore','padim'}, "
            f"got {cfg.anomaly.detector!r}"
        )
    pal = cfg.pipeline_al
    if pal.update_rate < 0:
        raise ValueError(f"pipeline_al.update_rate must be >= 0, got {pal.update_rate!r}")
    if pal.n_cycles < 0 or pal.budget_per_cycle < 0:
        raise ValueError("pipeline_al.n_cycles and budget_per_cycle must be >= 0")
    rt = cfg.runtime
    if rt.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "runtime.log_level must be one of "
            "{'DEBUG','INFO','WARNING','ERROR','CRITICAL'}, "
            f"got {rt.log_level!r}"
        )


def load_config(path: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to pipeline_config.yaml

    Returns:
        PipelineConfig dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If the YAML contains unknown keys or out-of-range values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    known_top = {f.name for f in PipelineConfig.__dataclass_fields__.values()} \
                | _KNOWN_TOPLEVEL_EXTRA_KEYS
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ValueError(
            f"Unknown top-level YAML key(s) {sorted(unknown_top)}. "
            f"Valid keys: {sorted(known_top)}"
        )

    # pipeline_al needs special handling: its `stages` field is a list of dataclasses
    pal_raw = dict(raw.get("pipeline_al") or {})
    stages_raw = pal_raw.pop("stages", []) or []
    pal_cfg = _dict_to_dataclass(PipelineALConfig, pal_raw, _path="pipeline_al")
    pal_cfg.stages = [
        _dict_to_dataclass(PipelineALStageConfig, s, _path=f"pipeline_al.stages[{i}]")
        for i, s in enumerate(stages_raw)
    ]

    # ab_testing.promotion is a nested section: build it with the same strict checks
    ab_raw = dict(raw.get("ab_testing") or {})
    promotion_raw = ab_raw.pop("promotion", None)
    ab_cfg = _dict_to_dataclass(ABTestingConfig, ab_raw, _path="ab_testing")
    ab_cfg.promotion = _dict_to_dataclass(
        PromotionConfig, promotion_raw, _path="ab_testing.promotion",
    )

    cfg = PipelineConfig(
        data=_dict_to_dataclass(DataConfig, raw.get("data"), _path="data"),
        model_a=_dict_to_dataclass(ModelConfig, raw.get("model_a"), _path="model_a"),
        model_b=_dict_to_dataclass(ModelConfig, raw.get("model_b"), _path="model_b"),
        active_learning=_dict_to_dataclass(
            ActiveLearningConfig, raw.get("active_learning"), _path="active_learning",
        ),
        pipeline_al=pal_cfg,
        ab_testing=ab_cfg,
        calibration=_dict_to_dataclass(CalibrationConfig, raw.get("calibration"), _path="calibration"),
        anomaly=_dict_to_dataclass(AnomalyConfig, raw.get("anomaly"), _path="anomaly"),
        output=_dict_to_dataclass(OutputConfig, raw.get("output"), _path="output"),
        runtime=_dict_to_dataclass(RuntimeConfig, raw.get("runtime"), _path="runtime"),
    )
    _validate(cfg)
    return cfg

"""
Pipeline configuration loader.

Reads pipeline_config.yaml and exposes typed dataclasses for each section.

Usage:
    from steeldefect_active_learning.config import load_config
    cfg = load_config("steeldefect_active_learning/config/pipeline_config.yaml")
    print(cfg.data.image_dir)
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union
import yaml

# The YAML shipped with the package, next to this module.
SHIPPED_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config", "pipeline_config.yaml")


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

    Field names are the keyword arguments of ``decide_promotion``; the gate's level is
    ``ab_testing.alpha``. None for expected_classes or class_prior means not configured:
    the gate keeps A.

    Attributes:
        min_gain:                    Superiority: the lower bound of acc_B - acc_A at the
                                     class prior must exceed it.
        noninferiority_margin:       Default per-class margin m: a class passes when the
                                     lower bound of recall_B - recall_A exceeds -m.
        class_margins:               Per-class margins, by class id.
        expected_classes:            The class taxonomy; a class absent from the slice fails.
        class_prior:                 Deployment class shares (renormalised), or "slice" to
                                     certify the slice's own class mix.
        false_alarm_margin:          Veto a class when B adds false alarms into it, at level
                                     alpha, beyond this rate; null disables the clause.
        class_false_alarm_margins:   Per-class false-alarm margins.
        min_detectable_gain:         Plan: the gain (accuracy units) the superiority clause
                                     must detect.
        planning_discordance:        Plan: the assumed share of images on which exactly one
                                     model is right.
        power:                       Plan: power of the superiority clause, and of all the
                                     per-class clauses together.
        require_no_class_regression: Enforce the per-class clause.
    """
    min_gain: float = 0.0
    noninferiority_margin: float = 0.05
    class_margins: Dict[Union[int, str], float] = field(default_factory=dict)
    expected_classes: Optional[List[Union[int, str]]] = None
    class_prior: Optional[Union[str, Dict[Union[int, str], float]]] = None
    false_alarm_margin: Optional[float] = 0.005
    class_false_alarm_margins: Dict[Union[int, str], float] = field(default_factory=dict)
    min_detectable_gain: float = 0.02
    planning_discordance: float = 0.10
    power: float = 0.80
    require_no_class_regression: bool = True


@dataclass
class ABTestingConfig:
    """Statistical A/B configuration: the diagnostic cascade (``metrics.compute_all_metrics``)
    and the promotion gate (``promotion``).

    Attributes:
        enabled:                       Run the A/B-testing pipeline.
        iou_threshold:                 IoU threshold for segmentation agreement.
        confidence_threshold:          Not implemented: forwarded to run_ab_comparison,
                                       which ignores it.
        demo_predictions_path:         CSV used when checkpoints are null (avoids
                                       degenerate 100% disagreement on random weights).
        alpha:                         One-sided level of the promotion gate and level of
                                       the diagnostic tests, in (0, 0.5).
        multiple_testing_correction:   "bonferroni" | "holm" | "benjamini-hochberg"
                                       | "none". Correction across the diagnostic
                                       per-class McNemar suite (11 classes in P1); the
                                       gate's per-class clause is an intersection-union
                                       test and needs none.
        srm_threshold:                 χ² p-value below which the diagnostic SRM check
                                       flags a mismatch; computed only for frames with an
                                       'arm' column, which the paired gate refuses.
                                       Nothing refuses results on it.
        expected_arm_ratio:            Expected fraction in arm A (default 0.5).
        bootstrap_n:                   Number of bootstrap resamples for percentile/
                                       BCa CIs on Cohen's d and confidence-delta mean.
        cuped_covariate:               Column name to use as the CUPED covariate. None
                                       disables variance reduction. The covariate must
                                       be pre-treatment (e.g., previous-batch confidence).
        sequential:                    "msprt" | "obrien_fleming" | None. Adds the
                                       sequential diagnostic block (Johari et al. 2017);
                                       the gate does not read it.
        bayesian_decision_threshold:   Threshold of the diagnostic Bayesian blocks'
                                       ``decision`` field; the gate thresholds no posterior.
        calibrate_before_comparison:   Not implemented: read by no code.
        promotion:                     Thresholds of the promotion gate (nested section).
    """
    enabled: bool = True
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5
    demo_predictions_path: Optional[str] = None
    alpha: float = 0.05
    multiple_testing_correction: str = "benjamini-hochberg"
    srm_threshold: float = 1e-3
    expected_arm_ratio: float = 0.5
    bootstrap_n: int = 1000
    cuped_covariate: Optional[str] = None
    sequential: Optional[str] = None
    bayesian_decision_threshold: float = 0.95
    calibrate_before_comparison: bool = False
    promotion: PromotionConfig = field(default_factory=PromotionConfig)

    def gate_kwargs(self) -> Dict[str, Any]:
        """The keyword arguments of ``decide_promotion``: the gate's only config mapping."""
        return {"alpha": self.alpha, **dataclasses.asdict(self.promotion)}


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
# Promotion-gate thresholds
# ---------------------------------------------------------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_class_id(x: Any) -> bool:
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, str)


def check_gate_thresholds(
    *,
    alpha: Any,
    min_gain: Any,
    noninferiority_margin: Any,
    class_margins: Optional[Mapping],
    expected_classes: Any,
    class_prior: Any,
    false_alarm_margin: Any,
    class_false_alarm_margins: Optional[Mapping],
    min_detectable_gain: Any,
    planning_discordance: Any,
    power: Any,
    require_no_class_regression: Any,
) -> None:
    """Raise ValueError if a promotion-gate threshold is out of range or inconsistent.

    Pure Python, so that the gate and the config loader apply the same rules. None for
    expected_classes or class_prior is valid: it means "not configured", and the gate
    then keeps A.
    """
    p = "ab_testing.promotion."
    if not (_is_number(alpha) and 0.0 < alpha < 0.5):
        raise ValueError(f"ab_testing.alpha must be a number in (0, 0.5), got {alpha!r}")
    if not (_is_number(min_gain) and 0.0 <= min_gain < 1.0):
        raise ValueError(f"{p}min_gain must be a number in [0, 1), got {min_gain!r}")
    if not (_is_number(noninferiority_margin) and 0.0 < noninferiority_margin < 1.0):
        raise ValueError(f"{p}noninferiority_margin must be a number in (0, 1), "
                         f"got {noninferiority_margin!r}")
    if expected_classes is not None:
        if isinstance(expected_classes, (str, bytes)) or not hasattr(expected_classes, "__iter__"):
            raise ValueError(f"{p}expected_classes must be a list of class ids, got {expected_classes!r}")
        ec = list(expected_classes)
        all_int = all(isinstance(k, int) and not isinstance(k, bool) for k in ec)
        all_str = all(isinstance(k, str) for k in ec)
        if not ec or not (all_int or all_str) or len(set(ec)) != len(ec):
            raise ValueError(f"{p}expected_classes must be a non-empty list of int or of str "
                             f"class ids, without repeats, got {expected_classes!r}")
    for key, margins in (("class_margins", class_margins),
                         ("class_false_alarm_margins", class_false_alarm_margins)):
        margins = dict(margins or {})
        for k, m in margins.items():
            if not _is_class_id(k):
                raise ValueError(f"{p}{key} keys must be int or str class ids, got {k!r}")
            if not (_is_number(m) and 0.0 < m < 1.0):
                raise ValueError(f"{p}{key}[{k!r}] must be a number in (0, 1), got {m!r}")
        stray = [k for k in margins if expected_classes is not None and k not in set(expected_classes)]
        if stray:
            raise ValueError(f"{p}{key} name classes outside expected_classes: {stray}")
    if class_prior is not None and class_prior != "slice":
        if not isinstance(class_prior, Mapping):
            raise ValueError(f"{p}class_prior must be null, 'slice' or a mapping of class "
                             f"shares, got {class_prior!r}")
        if expected_classes is None or set(class_prior) != set(expected_classes):
            raise ValueError(f"{p}class_prior keys must be exactly expected_classes")
        values = list(class_prior.values())
        if not all(_is_number(v) and v >= 0 for v in values) or sum(values) <= 0:
            raise ValueError(f"{p}class_prior shares must be numbers >= 0 with a positive sum")
    if false_alarm_margin is not None and not (_is_number(false_alarm_margin)
                                               and 0.0 < false_alarm_margin < 1.0):
        raise ValueError(f"{p}false_alarm_margin must be null or a number in (0, 1), "
                         f"got {false_alarm_margin!r}")
    if not (_is_number(min_detectable_gain) and min_gain < min_detectable_gain < 1.0):
        raise ValueError(f"{p}min_detectable_gain must be a number in (min_gain, 1), "
                         f"got {min_detectable_gain!r}")
    if not (_is_number(planning_discordance)
            and min_detectable_gain <= planning_discordance <= 1.0):
        raise ValueError(f"{p}planning_discordance must be a number in "
                         f"[min_detectable_gain, 1], got {planning_discordance!r}")
    if not (_is_number(power) and 0.5 <= power < 1.0):
        raise ValueError(f"{p}power must be a number in [0.5, 1), got {power!r}")
    if not isinstance(require_no_class_regression, bool):
        raise ValueError(f"{p}require_no_class_regression must be true or false, "
                         f"got {require_no_class_regression!r}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# YAML keys that are intentionally consumed outside the dataclass schema (e.g.,
# experiment-only sections that are not part of the typed pipeline config).
_KNOWN_TOPLEVEL_EXTRA_KEYS = {"flatness_clustering"}

# Keys of the first promotion gate, under ab_testing, with what replaces each one.
_RETIRED_KEYS = {
    ("power",): "moved to ab_testing.promotion.power",
    ("mde",): "moved to ab_testing.promotion.min_detectable_gain (accuracy units)",
    ("promotion", "min_cohens_d"): "removed: model confidence is no longer a criterion",
    ("promotion", "bayesian_threshold"): "removed: the gate reports the posterior and "
                                         "thresholds none",
    ("promotion", "no_class_regression"): "renamed ab_testing.promotion."
                                          "require_no_class_regression",
}


def _check_retired_keys(raw: dict) -> None:
    """Fail once, naming every retired key of the YAML and its replacement."""
    found = []
    for path, replacement in _RETIRED_KEYS.items():
        node = raw.get("ab_testing")
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and path[-1] in node:
            found.append(f"ab_testing.{'.'.join(path)}: {replacement}")
    if found:
        raise ValueError("Retired YAML key(s): " + "; ".join(found))


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
    check_gate_thresholds(**ab.gate_kwargs())   # alpha and every promotion threshold
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

    _check_retired_keys(raw)
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

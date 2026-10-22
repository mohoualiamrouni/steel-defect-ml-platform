"""steeldefect_ssl — Pipeline P4 of the SteelDefect ecosystem.

Self-supervised encoder (SSLResNet, ~1.3M parameters) + open-set novelty detection + continual
learning, designed to run on CPU. The intended loop, tested as hypothesis H4:

    SSL encoder on the unlabeled stream
      -> better embeddings for P1 (active learning) and P2 (clustering)
      -> open-set detection of NOVEL defect families held out of the known classes
      -> generalized category discovery + continual integration (no forgetting).

Hypothesis H4: an encoder pretrained self-supervised on the unlabeled stream
improves P2 clustering (NMI) and detects held-out defect families (open-set
AUROC).

The default configuration runs on calibrated-synthetic data
(``source_version='synthetic-cal-v1'``, seed 42);
``steeldefect_ssl/config/pipeline_config_neu.yaml`` runs it on the public NEU-DET
images (``source_version='neu-det-v1'``). CPU-only. No new dependencies beyond the
repo ``requirements.txt``.
"""

__version__ = "0.1.0"

SOURCE_SYNTHETIC = "synthetic-cal-v1"
SOURCE_NEU = "neu-det-v1"   # external PUBLIC real dataset (NEU-DET); not plant data

# --- environment hardening (must run before torch is imported) ---------------
# Windows ships duplicate OpenMP runtimes (scikit-learn / numba vs torch/MKL);
# importing both in one process segfaults unless this is set. Harmless elsewhere.
import os as _os  # noqa: E402
# Windows ships duplicate OpenMP runtimes (torch/MKL vs sklearn/numba); importing
# both in one process segfaults unless this is set. Compute (torch) stays
# multi-threaded for speed; the figures are rendered in a separate single-thread
# subprocess (see eval/render_plots) to dodge the residual heap instability.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# joblib/loky on some Windows shells fails to spawn the physical-core probe
# subprocess; giving it the count directly avoids that crash.
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 4))

# Make console output robust to non-cp1252 chars (Δ, ·, …) on Windows terminals.
import sys as _sys  # noqa: E402
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

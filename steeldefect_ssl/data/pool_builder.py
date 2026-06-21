"""Build the four calibrated-synthetic data pools for steeldefect_ssl.

Reuses the **P1 image generators** (``steeldefect_active_learning/data/generate_sample_data.py``)
— the physical steel-surface background + per-class ``make_*`` morphology — but
keeps full control over the pooling so one defect family is held out as *novel*:

* ``unlabeled`` — long-tail over the KNOWN classes (no novel); SSL pretraining.
* ``labeled``   — ``n_labeled_per_class`` per known class; open-set fit / GCD anchors / probes.
* ``eval``      — held-out known crops for linear/kNN probes.
* ``stream``    — known (by prior) + the injected novel family, with ``is_novel`` flags;
                  open-set evaluation (and, joined with ``labeled``, the GCD pool).

Crops are saved as 1-channel 64x64 PNGs. Deterministic (seed 42). All rows carry
``source_version='synthetic-cal-v1'``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

# Make the sibling P1 package importable regardless of CWD (mirrors run_task.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steeldefect_active_learning.data.generate_sample_data import (  # noqa: E402
    STEELDEFECT_CLASS_PRIOR,
    CLASS_NAMES,
    GENERATORS,
    class_counts_from_prior,
)

from steeldefect_ssl import SOURCE_NEU, SOURCE_SYNTHETIC  # noqa: E402

POOLS = ("unlabeled", "labeled", "eval", "stream")


def known_classes(novel_classes) -> List[str]:
    """All taxonomy classes except the held-out novel families (sorted, stable)."""
    novel = {novel_classes} if isinstance(novel_classes, str) else set(novel_classes)
    return sorted(c for c in CLASS_NAMES if c not in novel)


def label_map(novel_classes) -> Dict[str, int]:
    """Known class name -> integer label in ``[0, K-1]``."""
    return {c: i for i, c in enumerate(known_classes(novel_classes))}


def _save_gray(arr: np.ndarray, path: Path) -> str:
    gray = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)
    return str(path.resolve())


def build_pools(cfg) -> Dict[str, pd.DataFrame]:
    """Generate (or load cached) the four pools. ``cfg`` is a ``DataConfig``."""
    out = Path(cfg.image_dir)
    meta = {p: out / f"{p}_metadata.csv" for p in POOLS}

    if not cfg.regenerate and all(m.exists() for m in meta.values()):
        cached = {p: pd.read_csv(m) for p, m in meta.items()}
        # Guard: only reuse the cache if its novel/known sets match the config,
        # else the held-out families silently mismatch (stale-cache corruption).
        want_novel = {cfg.novel_classes} if isinstance(cfg.novel_classes, str) else set(cfg.novel_classes)
        want_known = set(known_classes(cfg.novel_classes))
        cached_novel = set(cached["stream"].loc[cached["stream"]["is_novel"] == True, "class_name"])  # noqa: E712
        cached_known = set(cached["unlabeled"]["class_name"])
        if cached_novel == want_novel and cached_known == want_known:
            return cached
        # else: fall through to regenerate with the current config

    known = known_classes(cfg.novel_classes)
    lmap = label_map(cfg.novel_classes)
    novel_list = [cfg.novel_classes] if isinstance(cfg.novel_classes, str) else list(cfg.novel_classes)
    rng = np.random.default_rng(cfg.seed)
    size = cfg.image_size
    counter: Dict[str, int] = {}
    records: Dict[str, List[dict]] = {p: [] for p in POOLS}

    def emit(pool: str, class_name: str) -> str:
        key = f"{pool}/{class_name}"
        idx = counter.get(key, 0)
        counter[key] = idx + 1
        d = out / pool / class_name
        d.mkdir(parents=True, exist_ok=True)
        arr = GENERATORS[class_name](rng, size)
        return _save_gray(arr, d / f"{class_name}_{idx:05d}.png")

    def add(pool: str, class_name: str, is_novel: bool) -> None:
        path = emit(pool, class_name)
        records[pool].append({
            "path": path,
            "class_name": class_name,
            "label": -1 if is_novel else lmap[class_name],
            "is_novel": bool(is_novel),
        })

    # --- unlabeled: calibrated long-tail over known classes (novel excluded) ---
    prior_known = {c: STEELDEFECT_CLASS_PRIOR[c] for c in known if c in STEELDEFECT_CLASS_PRIOR}
    counts = class_counts_from_prior(prior_known, cfg.n_total_unlabeled, min_per_class=8)
    for c in known:
        for _ in range(int(counts[c])):
            add("unlabeled", c, is_novel=False)

    # --- labeled known subset + eval known subset ---
    for c in known:
        for _ in range(cfg.n_labeled_per_class):
            add("labeled", c, is_novel=False)
        for _ in range(cfg.n_eval_per_class):
            add("eval", c, is_novel=False)

    # --- stream: known sampled by prior + injected novel family ---
    n_novel = int(round(cfg.stream_n * cfg.novel_frac_in_stream))
    n_known = max(0, cfg.stream_n - n_novel)
    probs = np.array([prior_known[c] for c in known], dtype=float)
    probs = probs / probs.sum()
    for c in rng.choice(known, size=n_known, p=probs):
        add("stream", str(c), is_novel=False)
    for i in range(n_novel):                  # distribute novel budget across the held-out families
        add("stream", novel_list[i % len(novel_list)], is_novel=True)
    rng.shuffle(records["stream"])

    pools: Dict[str, pd.DataFrame] = {}
    for p in POOLS:
        df = pd.DataFrame(records[p])
        df["source_version"] = SOURCE_SYNTHETIC
        df.to_csv(meta[p], index=False)
        pools[p] = df
    return pools


# ─────────────────────────────────────────────────────────────────────────────
# REAL public dataset loader (e.g. NEU-DET) — external validation of the METHOD.
# Independent public data, NOT plant data. Layout: image_dir/<class>/*.{png,bmp,jpg}
# ─────────────────────────────────────────────────────────────────────────────

_REAL_GLOBS = ("*.png", "*.bmp", "*.jpg", "*.jpeg")


def _list_images(d: Path) -> List[str]:
    """Absolute paths of all images in a class folder (case-insensitive ext), sorted."""
    seen = set()
    for pat in _REAL_GLOBS:
        seen.update(str(p.resolve()) for p in d.glob(pat))
        seen.update(str(p.resolve()) for p in d.glob(pat.upper()))
    return sorted(seen)


def real_known_classes(cfg) -> List[str]:
    """Known classes = on-disk class folders under cfg.image_dir minus the novel set (sorted)."""
    novel = {cfg.novel_classes} if isinstance(cfg.novel_classes, str) else set(cfg.novel_classes)
    root = Path(cfg.image_dir)
    return sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in novel)


def load_real_pools(cfg) -> Dict[str, pd.DataFrame]:
    """Build the four pools from a REAL flat class-folder dataset (e.g. NEU-DET).

    Layout: ``cfg.image_dir/<class_name>/*.{png,bmp,jpg,jpeg}``. The NOVEL classes
    are held out of training and injected into the stream; each KNOWN class is
    split disjointly into eval / labeled / unlabeled. The open-set stream reuses
    the held-out ``eval`` crops as its KNOWN portion — unseen by the encoder (SSL
    trains on ``unlabeled``) and by the open-set scorer (fit on ``labeled``), so
    the AUROC is leakage-safe — plus a sample of the novel families sized to
    ``novel_frac_in_stream``. ``stream_n`` is not used here (stream size follows
    the held-out data). Rows carry ``source_version=SOURCE_NEU``; ``path`` points
    at the ORIGINAL files (no copy). Deterministic (``cfg.seed``).
    """
    root = Path(cfg.image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"load_real_pools: data.image_dir not found: {root}")
    meta = {p: root / f"{p}_metadata.csv" for p in POOLS}
    novel_list = ([cfg.novel_classes] if isinstance(cfg.novel_classes, str)
                  else list(cfg.novel_classes))
    novel = set(novel_list)

    if not cfg.regenerate and all(m.exists() for m in meta.values()):
        cached = {p: pd.read_csv(m) for p, m in meta.items()}
        cached_known = set(cached["unlabeled"]["class_name"])
        cached_novel = set(cached["stream"].loc[cached["stream"]["is_novel"] == True, "class_name"])  # noqa: E712
        if cached_known == set(real_known_classes(cfg)) and cached_novel == novel:
            return cached

    known = real_known_classes(cfg)
    if not known:
        raise ValueError(f"load_real_pools: no KNOWN class folders under {root} "
                         f"(novel={sorted(novel)}); check data.image_dir and class names")
    lmap = {c: i for i, c in enumerate(known)}
    rng = np.random.default_rng(cfg.seed)
    records: Dict[str, List[dict]] = {p: [] for p in POOLS}

    def add(pool: str, path: str, class_name: str, is_novel: bool) -> None:
        records[pool].append({
            "path": path, "class_name": class_name,
            "label": -1 if is_novel else lmap[class_name], "is_novel": bool(is_novel),
        })

    # per-known-class disjoint split: eval | labeled | unlabeled
    for c in known:
        imgs = _list_images(root / c)
        if not imgs:
            raise ValueError(f"load_real_pools: known class '{c}' has no images under {root / c}")
        rng.shuffle(imgs)
        n_eval = min(cfg.n_eval_per_class, len(imgs))
        n_lab = min(cfg.n_labeled_per_class, max(0, len(imgs) - n_eval))
        ev, lab, unl = imgs[:n_eval], imgs[n_eval:n_eval + n_lab], imgs[n_eval + n_lab:]
        for p in ev:
            add("eval", p, c, False)
            add("stream", p, c, False)     # held-out eval doubles as the known stream (leakage-safe)
        for p in lab:
            add("labeled", p, c, False)
        for p in unl:
            add("unlabeled", p, c, False)

    # novel stream sized to hit novel_frac_in_stream relative to the known stream
    n_known_stream = len(records["stream"])
    frac = cfg.novel_frac_in_stream
    n_novel = int(round(n_known_stream * frac / max(1e-9, 1.0 - frac)))
    all_novel = [(p, c) for c in novel_list for p in _list_images(root / c)]
    if not all_novel:
        raise ValueError(f"load_real_pools: no images for novel classes {novel_list} under {root}")
    if n_novel:
        sel = rng.choice(len(all_novel), size=n_novel, replace=n_novel > len(all_novel))
        for j in sel:
            p, c = all_novel[int(j)]
            add("stream", p, c, True)
    rng.shuffle(records["stream"])

    pools: Dict[str, pd.DataFrame] = {}
    for p in POOLS:
        df = pd.DataFrame(records[p])
        df["source_version"] = SOURCE_NEU
        df.to_csv(meta[p], index=False)
        pools[p] = df
    return pools

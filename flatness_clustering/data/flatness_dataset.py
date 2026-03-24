"""
FlatnessDataset — loader for synthetic (and real) flatness profiles.

Each coil profile is stored as a compressed NumPy NPZ file with the following keys:
    profile   : float32 array of shape (width, length) — height in I-units
    coil_id   : str
    pattern   : str  (class label)
    class_id  : int
    grade     : str  (steel grade)
    customer  : str
    timestamp : str  (ISO 8601)

Usage
-----
    from flatness_clustering.data.flatness_dataset import FlatnessDataset

    ds = FlatnessDataset("flatness_clustering/data/synthetic_profiles")
    print(len(ds))                  # 300 coils
    item = ds[0]                    # dict with all keys above
    profiles, meta = ds.load_all_profiles()   # (300, 150, 2000), DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FlatnessDataset:
    """Dataset of coil flatness profiles loaded from NPZ files.

    Supports both a metadata-CSV-first loading strategy (fast) and a
    directory-scan fallback when no CSV is available.

    Args:
        root_dir:     Directory containing class subdirectories with NPZ files,
                      or a directory containing NPZ files directly.
        metadata_csv: Path to metadata CSV (optional).  If not provided, the
                      dataset will look for ``root_dir/metadata.csv``.
    """

    def __init__(
        self,
        root_dir: str,
        metadata_csv: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)

        # --- Locate metadata CSV ---
        if metadata_csv is not None:
            meta_path = Path(metadata_csv)
        else:
            meta_path = self.root_dir / "metadata.csv"

        if meta_path.exists():
            self._meta = pd.read_csv(meta_path)
        else:
            # Fall back: scan all NPZ files and build metadata on the fly
            npz_files = sorted(self.root_dir.rglob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(
                    f"No NPZ files found in '{root_dir}'. "
                    "Run generate_flatness_data.py first."
                )
            records = []
            for p in npz_files:
                data = np.load(p, allow_pickle=True)
                records.append({
                    "coil_id":  str(data.get("coil_id", p.stem)),
                    "pattern":  str(data.get("pattern", p.parent.name)),
                    "class_id": int(data.get("class_id", -1)),
                    "grade":    str(data.get("grade", "unknown")),
                    "customer": str(data.get("customer", "unknown")),
                    "timestamp": str(data.get("timestamp", "")),
                    "filepath": str(p.resolve()),
                })
            self._meta = pd.DataFrame(records)

        self._paths: List[str] = self._meta["filepath"].tolist()

    # ------------------------------------------------------------------
    # Standard dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Dict:
        """Load and return one coil profile.

        Returns:
            dict with keys:
                coil_id, pattern, class_id, grade, customer, timestamp,
                profile  (np.ndarray float32, shape (W, L))
        """
        path = self._paths[idx]
        data = np.load(path, allow_pickle=True)
        row  = self._meta.iloc[idx]
        return {
            "coil_id":   row["coil_id"],
            "pattern":   row["pattern"],
            "class_id":  int(row["class_id"]),
            "grade":     row["grade"],
            "customer":  row["customer"],
            "timestamp": row["timestamp"],
            "profile":   data["profile"].astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Bulk loader
    # ------------------------------------------------------------------

    def load_all_profiles(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load every profile into a single NumPy array.

        Returns:
            profiles : np.ndarray  shape (N, W, L) float32
            metadata : pd.DataFrame with all columns from the metadata CSV
        """
        profiles = []
        for idx in range(len(self)):
            item = self[idx]
            profiles.append(item["profile"])
        return np.stack(profiles, axis=0), self._meta.copy()

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_by_pattern(self, pattern: str) -> "FlatnessDataset":
        """Return a new dataset containing only coils of a specific pattern."""
        subset = self._meta[self._meta["pattern"] == pattern].copy()
        new_ds = object.__new__(FlatnessDataset)
        new_ds.root_dir = self.root_dir
        new_ds._meta = subset.reset_index(drop=True)
        new_ds._paths = new_ds._meta["filepath"].tolist()
        return new_ds

    def filter_by_customer(self, customer: str) -> "FlatnessDataset":
        """Return a new dataset containing only coils for a specific customer."""
        subset = self._meta[self._meta["customer"] == customer].copy()
        new_ds = object.__new__(FlatnessDataset)
        new_ds.root_dir = self.root_dir
        new_ds._meta = subset.reset_index(drop=True)
        new_ds._paths = new_ds._meta["filepath"].tolist()
        return new_ds

    @property
    def metadata(self) -> pd.DataFrame:
        """Return a copy of the metadata DataFrame."""
        return self._meta.copy()

    def __repr__(self) -> str:
        pattern_counts = self._meta["pattern"].value_counts().to_dict()
        return (
            f"FlatnessDataset({len(self)} coils, "
            f"root='{self.root_dir}', "
            f"patterns={pattern_counts})"
        )

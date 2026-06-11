"""Datasets for steeldefect_ssl.

* :class:`SSLTwoViewDataset` — returns two augmented views of an unlabeled crop
  (the contrastive / SimSiam input); no labels.
* :class:`LabeledEvalDataset` — returns ``(tensor, label, path)`` for probing,
  open-set scoring and discovery.

Crops are loaded as 1-channel float tensors in ``[0, 1]`` and resized to the
configured size.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_gray(path: str, image_size: int) -> torch.Tensor:
    """Load a PNG as a (1, image_size, image_size) float tensor in [0, 1]."""
    img = Image.open(path).convert("L")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


class SSLTwoViewDataset(Dataset):
    """Two independently-augmented views of each unlabeled crop."""

    def __init__(self, paths: Sequence[str], augment: Callable, image_size: int = 64):
        self.paths = list(paths)
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base = load_gray(self.paths[i], self.image_size)
        return self.augment(base), self.augment(base)


class LabeledEvalDataset(Dataset):
    """``(tensor, label, path)`` for probes / open-set / discovery.

    ``df`` must have columns ``path`` and ``label`` (int; -1 for novel/unknown).
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None,
                 image_size: int = 64):
        self.paths: List[str] = df["path"].tolist()
        self.labels: List[int] = df["label"].astype(int).tolist()
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = load_gray(self.paths[i], self.image_size)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i], self.paths[i]

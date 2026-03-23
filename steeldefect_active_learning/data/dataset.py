"""
PyTorch Dataset for steel surface inspection images.

Returns (tensor, path) tuples so that every prediction can be traced back
to the original file on disk — critical for active learning candidate export.

Usage:
    from steeldefect_active_learning.data.dataset import SteelDefectDataset
    from torch.utils.data import DataLoader

    dataset = SteelDefectDataset("data/sample_images", image_size=(256, 256))
    loader  = DataLoader(dataset, batch_size=32, num_workers=4)

    for tensors, paths in loader:
        # tensors: (B, C, H, W) float32 in [0, 1]
        # paths:   tuple of strings, length B
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from steeldefect_active_learning.data.image_loader import discover_images, load_image


class SteelDefectDataset(Dataset):
    """PyTorch Dataset wrapping a directory (or explicit list) of inspection images.

    Args:
        source:       Path to a directory OR an explicit list of file paths.
        image_size:   (H, W) to resize images. None = keep original size.
        extensions:   File extensions to include when source is a directory.
        transform:    Optional extra torchvision transform applied after
                      the default resize + to-tensor. Pass None to skip.
        grayscale_to_rgb: If True, replicate single-channel images to 3 channels
                          (required by most ImageNet-pretrained backbones).
    """

    def __init__(
        self,
        source: Union[str, List[str]],
        image_size: Optional[Tuple[int, int]] = (256, 256),
        extensions: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        grayscale_to_rgb: bool = True,
    ):
        if isinstance(source, (str, Path)):
            self.paths = discover_images(str(source), extensions=extensions)
        else:
            self.paths = [str(p) for p in source]

        if len(self.paths) == 0:
            raise ValueError(f"No images found in source: {source}")

        self.image_size = image_size
        self.grayscale_to_rgb = grayscale_to_rgb
        self.extra_transform = transform

        # Build base transform pipeline
        t_list = []
        if image_size is not None:
            t_list.append(transforms.Resize(image_size))
        t_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))
        self.base_transform = transforms.Compose(t_list)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        arr = load_image(path)              # (H, W, C) or (H, W), float32 [0,1]

        tensor = self._to_tensor(arr)       # (C, H, W)
        tensor = self.base_transform(tensor)

        if self.extra_transform is not None:
            tensor = self.extra_transform(tensor)

        return tensor, path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert HWC or HW numpy array to CHW float tensor."""
        if arr.ndim == 2:
            # Grayscale (H, W) → (1, H, W)
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]         # drop alpha channel
            # HWC → CHW
            arr = arr.transpose(2, 0, 1)

        tensor = torch.from_numpy(arr.copy())  # (C, H, W)

        if self.grayscale_to_rgb and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor

    def get_paths(self) -> List[str]:
        """Return the full list of image paths in dataset order."""
        return list(self.paths)

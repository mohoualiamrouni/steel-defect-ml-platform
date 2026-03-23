"""
Flexible image loader for steel surface inspection images.

Supports: PNG, JPEG, TIFF, BMP (via Pillow), HDF5 (via h5py), NumPy .npy arrays.
All formats are returned as float32 numpy arrays in [0, 1] range with shape (H, W, C)
or (H, W) for grayscale.

Usage:
    from steeldefect_active_learning.data.image_loader import load_image, discover_images

    img = load_image("path/to/image.tiff")   # -> np.ndarray float32
    paths = discover_images("data/images", extensions=[".tiff", ".png"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

# Lazy imports — only loaded when format is detected
_PIL_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
_H5_FORMATS = {".h5", ".hdf5"}
_NPY_FORMATS = {".npy"}


def load_image(path: str, h5_dataset_key: str = "image") -> np.ndarray:
    """Load an image from disk, auto-detecting the format.

    Args:
        path:            Absolute or relative path to the image file.
        h5_dataset_key:  Key inside HDF5 file that holds the image array.

    Returns:
        np.ndarray of shape (H, W, C) or (H, W), dtype float32, values in [0, 1].

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    ext = Path(path).suffix.lower()

    if ext in _PIL_FORMATS:
        return _load_pil(path)
    elif ext in _H5_FORMATS:
        return _load_h5(path, h5_dataset_key)
    elif ext in _NPY_FORMATS:
        return _load_npy(path)
    else:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {sorted(_PIL_FORMATS | _H5_FORMATS | _NPY_FORMATS)}"
        )


def _load_pil(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path)
    # Convert 16-bit TIFF to 8-bit if needed
    if img.mode == "I;16" or img.mode == "I":
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    # Convert palette/CMYK to RGB
    if img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _load_h5(path: str, key: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if key not in f:
            # Fall back to first available dataset
            key = list(f.keys())[0]
        arr = np.array(f[key], dtype=np.float32)
    # Normalize to [0, 1] if not already
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def discover_images(
    root_dir: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[str]:
    """Walk a directory and return all image file paths.

    Args:
        root_dir:   Root directory to search.
        extensions: List of lowercase extensions to include, e.g. [".png", ".tiff"].
                    Defaults to all supported formats.
        recursive:  Whether to search subdirectories.

    Returns:
        Sorted list of absolute file paths.
    """
    if extensions is None:
        extensions = list(_PIL_FORMATS | _H5_FORMATS | _NPY_FORMATS)
    extensions = {ext.lower() for ext in extensions}

    paths: List[str] = []
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root_dir}")

    glob_pattern = "**/*" if recursive else "*"
    for p in root.glob(glob_pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(str(p.resolve()))

    return sorted(paths)

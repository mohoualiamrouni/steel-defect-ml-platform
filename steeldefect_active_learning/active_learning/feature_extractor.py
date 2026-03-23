"""
CNN feature extractor using forward hooks.

Taps into a named layer (e.g., 'avgpool' for ResNet, 'features' for DenseNet)
and returns L2-normalized embedding vectors for each image — used downstream
for diversity-based clustering.

Usage:
    from steeldefect_active_learning.active_learning.feature_extractor import extract_features

    embeddings = extract_features(
        model, dataloader, layer_name="avgpool", device="cuda"
    )
    # embeddings: dict {image_path: np.ndarray shape (D,)}
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: str = "avgpool",
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Extract L2-normalized feature embeddings from a named layer.

    Works with ResNet (layer_name='avgpool'), DenseNet (layer_name='features'),
    or any nn.Module sub-layer accessible via model.<layer_name>.

    For U-Net models that have a `get_embedding()` method, that is used instead.

    Args:
        model:       PyTorch model in eval() mode.
        dataloader:  Yields (tensor, path) batches.
        layer_name:  Name of the sub-module to hook. Ignored for U-Net.
        device:      Torch device string.

    Returns:
        Dict mapping image_path -> 1-D float32 numpy array (L2-normalized).
    """
    model.eval()
    model.to(device)

    # Check if model has a built-in get_embedding (e.g., _SimpleUNet)
    has_embed_method = callable(getattr(model, "get_embedding", None))

    if has_embed_method:
        return _extract_via_method(model, dataloader, device)
    else:
        return _extract_via_hook(model, dataloader, layer_name, device)


# ---------------------------------------------------------------------------
# Hook-based extraction
# ---------------------------------------------------------------------------

def _extract_via_hook(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: str,
) -> Dict[str, np.ndarray]:
    target_layer = _find_layer(model, layer_name)
    activations: list = []

    def _hook(module, input, output):
        # Flatten spatial dims and keep batch dimension
        out = output.detach()
        if out.ndim == 4:
            out = out.mean(dim=[2, 3])   # Global average pooling
        elif out.ndim == 3:
            out = out.mean(dim=2)
        activations.append(out.cpu())

    handle = target_layer.register_forward_hook(_hook)
    embeddings: Dict[str, np.ndarray] = {}

    try:
        for tensors, paths in tqdm(dataloader, desc="Feature extraction", leave=False):
            activations.clear()
            tensors = tensors.to(device)
            _ = model(tensors)

            batch_emb = activations[0]                    # (B, D)
            batch_emb = F.normalize(batch_emb, dim=1)    # L2 normalize
            for path, emb in zip(paths, batch_emb.numpy()):
                embeddings[path] = emb.astype(np.float32)
    finally:
        handle.remove()

    return embeddings


def _extract_via_method(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}

    for tensors, paths in tqdm(dataloader, desc="Feature extraction (UNet)", leave=False):
        tensors = tensors.to(device)
        batch_emb = model.get_embedding(tensors).detach().cpu()   # (B, D)
        batch_emb = F.normalize(batch_emb, dim=1)
        for path, emb in zip(paths, batch_emb.numpy()):
            embeddings[path] = emb.astype(np.float32)

    return embeddings


# ---------------------------------------------------------------------------
# Layer lookup
# ---------------------------------------------------------------------------

def _find_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Retrieve a sub-module by dot-separated name or direct attribute."""
    # Try direct attribute first (e.g., model.avgpool)
    if hasattr(model, layer_name):
        return getattr(model, layer_name)

    # Try dot-path (e.g., "layer4.1.conv2")
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            raise AttributeError(
                f"Layer '{layer_name}' not found in model. "
                f"Available top-level layers: {[n for n, _ in model.named_children()]}"
            )
    return module

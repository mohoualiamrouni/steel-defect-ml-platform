"""Self-supervised augmentations tuned for steel-surface crops (1-channel).

Two independent applications of :func:`build_ssl_augment` produce the two views
fed to SimSiam / SimCLR. The transforms are deliberately photometric +
geometric but *colour-free* (the crops are grayscale): random resized crop,
H/V flips (a steel crop has no canonical orientation at this scale), small
rotation, Gaussian blur, brightness/contrast jitter (models line-scan specular /
lighting variation — the generator literally jitters background brightness),
and a mild random erasing (occlusion).

All transforms operate on float tensors in ``[0, 1]`` of shape ``(1, H, W)``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torchvision.transforms as T


class _BrightnessContrast(torch.nn.Module):
    """Channel-agnostic brightness/contrast jitter (ColorJitter assumes 3ch)."""

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, p: float = 0.8):
        super().__init__()
        self.b = float(brightness)
        self.c = float(contrast)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        # brightness: multiplicative around 1
        if self.b > 0:
            factor = 1.0 + (torch.rand(()) * 2 - 1) * self.b
            x = x * factor
        # contrast: pull toward / push from the per-image mean
        if self.c > 0:
            factor = 1.0 + (torch.rand(()) * 2 - 1) * self.c
            mean = x.mean()
            x = (x - mean) * factor + mean
        return x.clamp_(0.0, 1.0)


def build_ssl_augment(image_size: int = 64) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a transform producing ONE augmented view of a (1,H,W) tensor."""
    return T.Compose([
        T.RandomResizedCrop(
            image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=10)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
        _BrightnessContrast(brightness=0.3, contrast=0.3, p=0.8),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.5),
    ])


def build_eval_transform(image_size: int = 64, to_rgb: bool = False) -> Callable:
    """Deterministic transform for embedding / probing.

    For the native 1-channel SSL encoder: resize only, values kept in ``[0,1]``
    (matching what the encoder saw in training). For the ``resnet18`` /
    PatchCore baseline path: replicate to 3 channels and ImageNet-normalise.
    """
    ops = [T.Resize((image_size, image_size), antialias=True)]
    if to_rgb:
        ops.append(T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t))
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)

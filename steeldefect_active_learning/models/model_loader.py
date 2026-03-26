"""
Model loader for inspection CNN checkpoints.

Supported architectures:
    Classification : resnet18, resnet34, resnet50, densenet121
    Segmentation   : unet  (simple encoder-decoder, can be replaced by smp UNet)

Usage:
    from steeldefect_active_learning.models.model_loader import load_model

    model = load_model(
        arch="resnet50",
        checkpoint="checkpoints/model_a.pth",
        num_classes=5,
        device="cuda",
        task="classification",
    )
    model.eval()
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    arch: str,
    checkpoint: Optional[str],
    num_classes: int,
    device: str = "cpu",
    task: str = "classification",
) -> nn.Module:
    """Build a model and (optionally) load weights from a checkpoint.

    Args:
        arch:        Architecture name. One of:
                     'resnet18', 'resnet34', 'resnet50', 'densenet121', 'unet'
        checkpoint:  Path to a .pt / .pth file. Pass None to use random weights
                     (useful for testing / synthetic demos).
        num_classes: Number of output classes.
        device:      'cpu' or 'cuda' (or 'cuda:0', etc.)
        task:        'classification' or 'segmentation'

    Returns:
        nn.Module in eval() mode on the requested device.
    """
    arch = arch.lower()

    if task == "segmentation":
        model = _build_unet(num_classes)
    elif arch.startswith("resnet"):
        model = _build_resnet(arch, num_classes)
    elif arch == "densenet121":
        model = _build_densenet121(num_classes)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Choose: resnet18, resnet34, resnet50, densenet121, unet"
        )

    if checkpoint is not None:
        _load_weights(model, checkpoint, device)
    else:
        print(f"[model_loader] No checkpoint provided for '{arch}' - using random weights.")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def _build_resnet(arch: str, num_classes: int) -> nn.Module:
    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown ResNet variant '{arch}'.")

    # weights=None → random init (we load our own checkpoint)
    model = arch_map[arch](weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_densenet121(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def _build_unet(num_classes: int, in_channels: int = 3) -> nn.Module:
    """Lightweight U-Net for binary / multi-class segmentation.

    For production use, replace with segmentation_models_pytorch.Unet
    which supports pretrained encoders.
    """
    return _SimpleUNet(in_channels=in_channels, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_weights(model: nn.Module, checkpoint: str, device: str) -> None:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)

    # Handle common checkpoint formats
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    # Strip 'module.' prefix from DataParallel checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[model_loader] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[model_loader] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")


# ---------------------------------------------------------------------------
# Simple U-Net implementation (drop-in, no external deps)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _SimpleUNet(nn.Module):
    """
    Compact U-Net with 4 down/up levels.
    Input:  (B, in_channels, H, W)
    Output: (B, num_classes, H, W)  — raw logits
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        f = [64, 128, 256, 512]

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f[0])
        self.enc2 = _DoubleConv(f[0], f[1])
        self.enc3 = _DoubleConv(f[1], f[2])
        self.enc4 = _DoubleConv(f[2], f[3])

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(f[3], f[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.dec4 = _DoubleConv(f[3] * 2, f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = _DoubleConv(f[2] * 2, f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = _DoubleConv(f[1] * 2, f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = _DoubleConv(f[0] * 2, f[0])

        self.head = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck feature vector (global average pooled) for AL."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return b.mean(dim=[2, 3])   # (B, 1024)

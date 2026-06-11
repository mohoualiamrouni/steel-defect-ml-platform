"""Compact CPU-friendly encoder + self-supervised heads.

* :class:`SSLResNet` — a ResNet-10-ish backbone (4 stages, 1 BasicBlock each)
  adapted to 1-channel 64x64 input, ending in a linear projection to
  ``embed_dim`` (~1.5-3M params at ``width=32`` — trains from scratch on CPU,
  unlike a 224-tuned torchvision ResNet-18). ``get_embedding(x)`` returns the
  downstream representation (so P1's ``feature_extractor`` reuse hook applies).
* :func:`build_projection_head` / :func:`build_prediction_head` — the SSL MLPs
  (kept separate from the backbone; discarded after pretraining).
* :func:`build_encoder` — backbone factory (``ssl_resnet`` or adapted ``resnet18``).

The representation used by every downstream stage (probes, clustering, open-set,
GCD, continual) is the **backbone output**, before the projection head — the
standard SSL evaluation protocol.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SSLResNet(nn.Module):
    """ResNet-10-ish backbone: 1xHxW -> embed_dim."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 256, width: int = 32):
        super().__init__()
        w = width
        # stride-2 stem (64 -> 32) keeps the expensive early layers off the full
        # 64x64 grid -> ~2x faster on CPU while preserving enough detail at 64px.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        self.layer1 = _BasicBlock(w, w, stride=1)          # 32
        self.layer2 = _BasicBlock(w, w * 2, stride=2)      # 16
        self.layer3 = _BasicBlock(w * 2, w * 4, stride=2)  # 8
        self.layer4 = _BasicBlock(w * 4, w * 8, stride=2)  # 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 8, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)

    # reuse hook: steeldefect_active_learning feature_extractor dispatches to this
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class _ResNet18Adapter(nn.Module):
    """torchvision resnet18 adapted to ``in_channels`` and embedding output."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        import torchvision.models as tvm
        net = tvm.resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        net.fc = nn.Linear(512, embed_dim)
        self.net = net
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def build_encoder(arch: str = "ssl_resnet", in_channels: int = 1,
                  embed_dim: int = 256, width: int = 32) -> nn.Module:
    if arch == "ssl_resnet":
        return SSLResNet(in_channels=in_channels, embed_dim=embed_dim, width=width)
    if arch == "resnet18":
        return _ResNet18Adapter(in_channels=in_channels, embed_dim=embed_dim)
    raise ValueError(f"unknown encoder arch: {arch}")


def build_projection_head(in_dim: int, hidden_dim: int, out_dim: int,
                          n_layers: int = 3, last_bn: bool = True) -> nn.Module:
    """SimSiam: 3-layer w/ output BN. SimCLR: 2-layer, no output BN."""
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden_dim, bias=False),
                   nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


def build_prediction_head(dim: int, hidden_dim: int) -> nn.Module:
    """SimSiam predictor: bottleneck MLP dim -> hidden -> dim."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


@torch.no_grad()
def embed_dataset(encoder: nn.Module, dataset, device: str = "cpu",
                  batch_size: int = 256) -> "tuple":
    """Return (X: np.ndarray [N, embed_dim], labels: np.ndarray, paths: list).

    Works with :class:`steeldefect_ssl.data.ssl_dataset.LabeledEvalDataset`
    (yields ``(tensor, label, path)``). Pure inference, no grad.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    encoder.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels, paths = [], [], []
    for xb, yb, pb in loader:
        z = encoder.get_embedding(xb.to(device)).cpu().numpy()
        feats.append(z)
        labels.append(np.asarray(yb))
        paths.extend(list(pb))
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(int)
    return X, y, paths

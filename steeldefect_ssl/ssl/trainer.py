"""CPU training loop for SimSiam / SimCLR + the picklable :class:`EncoderBundle`.

``train_encoder`` returns the trained backbone wrapped in an EncoderBundle (so
downstream stages and the MLflow overlay can reload it) plus a history dict with
the per-epoch loss curve and the collapse-guard std.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from steeldefect_ssl.data.augmentations import build_ssl_augment
from steeldefect_ssl.data.ssl_dataset import SSLTwoViewDataset
from steeldefect_ssl.models.encoder import (
    build_encoder, build_prediction_head, build_projection_head,
)
from steeldefect_ssl.ssl.simclr import nt_xent_loss
from steeldefect_ssl.ssl.simsiam import simsiam_loss


@dataclass
class EncoderBundle:
    """Trained backbone + the metadata needed to rebuild it."""
    state_dict: Dict
    meta: Dict
    history: Dict = field(default_factory=dict)

    def build(self, device: str = "cpu") -> nn.Module:
        enc = build_encoder(
            arch=self.meta["arch"], in_channels=self.meta["in_channels"],
            embed_dim=self.meta["embed_dim"], width=self.meta["width"],
        )
        enc.load_state_dict(self.state_dict)
        return enc.to(device).eval()

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict, "meta": self.meta,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path: str) -> "EncoderBundle":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(blob["state_dict"], blob["meta"], blob.get("history", {}))


@torch.no_grad()
def _collapse_std(z: torch.Tensor) -> float:
    """SimSiam output std (per-dim std of L2-normalised z, averaged). Healthy
    ~ 1/sqrt(d); near 0 => representational collapse."""
    z = F.normalize(z, dim=1)
    return float(z.std(dim=0).mean().item())


def train_encoder(cfg, unlabeled_paths: Sequence[str], *, method: Optional[str] = None,
                  epochs: Optional[int] = None, device: str = "cpu",
                  log: Optional[Callable[[str], None]] = None) -> EncoderBundle:
    """Train a backbone with ``method`` (default ``cfg.ssl.method``) on the
    unlabeled crops. ``cfg`` is the full :class:`SSLPipelineConfig`."""
    import os
    method = method or cfg.ssl.method
    epochs = int(epochs or cfg.ssl.epochs)
    log = log or (lambda *_: None)
    torch.set_num_threads(max(1, os.cpu_count() or 4))   # OMP_NUM_THREADS=1 -> give torch its own threads
    torch.manual_seed(cfg.runtime.seed)
    np.random.seed(cfg.runtime.seed)

    e = cfg.encoder
    backbone = build_encoder(e.arch, e.in_channels, e.embed_dim, e.width).to(device)
    if method == "simsiam":
        projector = build_projection_head(e.embed_dim, e.embed_dim, e.proj_dim,
                                           n_layers=3, last_bn=True).to(device)
        predictor = build_prediction_head(e.proj_dim, e.pred_dim).to(device)
        params = list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    else:  # simclr
        projector = build_projection_head(e.embed_dim, e.embed_dim, e.proj_dim,
                                           n_layers=2, last_bn=False).to(device)
        predictor = None
        params = list(backbone.parameters()) + list(projector.parameters())

    opt = torch.optim.SGD(params, lr=cfg.ssl.lr, momentum=cfg.ssl.momentum,
                          weight_decay=cfg.ssl.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = SSLTwoViewDataset(unlabeled_paths, build_ssl_augment(cfg.data.image_size),
                           cfg.data.image_size)
    loader = DataLoader(ds, batch_size=cfg.ssl.batch_size, shuffle=True,
                        num_workers=cfg.ssl.num_workers, drop_last=True)

    loss_curve: List[float] = []
    collapse_curve: List[float] = []
    backbone.train(); projector.train()
    if predictor is not None:
        predictor.train()

    log(f"  SSL[{method}] {epochs} epochs · {len(ds)} crops · bs={cfg.ssl.batch_size}")
    for ep in range(epochs):
        t0 = time.time()
        ep_loss, n_batches, last_z = 0.0, 0, None
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad(set_to_none=True)
            if method == "simsiam":
                z1, z2 = projector(backbone(v1)), projector(backbone(v2))
                p1, p2 = predictor(z1), predictor(z2)
                loss = simsiam_loss(p1, z2, p2, z1)
            else:
                z1, z2 = projector(backbone(v1)), projector(backbone(v2))
                loss = nt_xent_loss(z1, z2, cfg.ssl.temperature)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()); n_batches += 1
            last_z = z1.detach()
        sched.step()
        avg = ep_loss / max(1, n_batches)
        cstd = _collapse_std(last_z) if last_z is not None else float("nan")
        loss_curve.append(avg); collapse_curve.append(cstd)
        log(f"    epoch {ep + 1:>2}/{epochs}  loss={avg:+.4f}  collapse_std={cstd:.4f}  ({time.time() - t0:.1f}s)")
        if cstd < 1e-3:
            log(f"    [warn] collapse_std={cstd:.4f} very low — possible representational collapse")

    backbone.eval()
    meta = {"arch": e.arch, "in_channels": e.in_channels,
            "embed_dim": e.embed_dim, "width": e.width, "method": method,
            "epochs": epochs}
    history = {"method": method, "epochs": epochs,
               "loss_curve": loss_curve, "collapse_curve": collapse_curve,
               "final_loss": loss_curve[-1] if loss_curve else None,
               "final_collapse_std": collapse_curve[-1] if collapse_curve else None}
    state = {k: v.cpu() for k, v in backbone.state_dict().items()}
    return EncoderBundle(state_dict=state, meta=meta, history=history)

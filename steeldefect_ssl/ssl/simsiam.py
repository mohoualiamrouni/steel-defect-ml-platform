"""SimSiam objective (Chen & He, 2021): negative cosine + stop-gradient.

No negatives, no momentum encoder, small batches — ideal for CPU. Collapse is
prevented by the predictor + stop-gradient; we additionally monitor the output
std (collapse guard) in the trainer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _neg_cos(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """-cosine(p, stopgrad(z)), averaged over the batch."""
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1: torch.Tensor, z2: torch.Tensor,
                 p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    return 0.5 * _neg_cos(p1, z2) + 0.5 * _neg_cos(p2, z1)

"""SimCLR / NT-Xent objective (Chen et al., 2020).

The NT-Xent loss is exactly the InfoNCE the thesis develops (cosine similarity,
temperature, in-batch negatives) — included as the theory-aligned comparison to
SimSiam. Quality scales with batch size (more negatives); on CPU we use a
moderate batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent over a 2N batch (z1, z2 are the two views, each (N, d))."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)          # (2N, d)
    sim = (z @ z.t()) / temperature                            # (2N, 2N)
    # mask self-similarity
    diag = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))
    # positive index: i <-> i+N  (mod 2N)
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)

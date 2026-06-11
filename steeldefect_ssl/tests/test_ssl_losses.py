import torch
import torch.nn.functional as F

from steeldefect_ssl.models.encoder import (build_encoder, build_prediction_head,
                                      build_projection_head)
from steeldefect_ssl.ssl.simclr import nt_xent_loss
from steeldefect_ssl.ssl.simsiam import simsiam_loss
from steeldefect_ssl.ssl.trainer import _collapse_std


def test_simsiam_loss_bounds():
    z = F.normalize(torch.rand(8, 16), dim=1)
    # identical p,z -> negative cosine = -1 (perfect alignment)
    loss = simsiam_loss(z, z, z, z)
    assert loss.item() < -0.99


def test_nt_xent_positive_and_lower_for_aligned():
    torch.manual_seed(0)
    z1 = torch.randn(16, 8)
    aligned = nt_xent_loss(z1, z1.clone(), 0.2)         # views identical
    random = nt_xent_loss(z1, torch.randn(16, 8), 0.2)  # views unrelated
    assert aligned.item() > 0
    assert aligned.item() < random.item()


def test_collapse_std_constant_vs_diverse():
    constant = torch.ones(32, 16)
    diverse = torch.randn(32, 16)
    assert _collapse_std(constant) < 1e-4
    assert _collapse_std(diverse) > 0.1


def test_simsiam_step_reduces_loss():
    torch.manual_seed(0)
    bb = build_encoder("ssl_resnet", 1, 16, 8)
    proj = build_projection_head(16, 16, 16, 3, True)
    pred = build_prediction_head(16, 8)
    opt = torch.optim.SGD(
        list(bb.parameters()) + list(proj.parameters()) + list(pred.parameters()), lr=0.1)
    bb.train(); proj.train(); pred.train()
    v1, v2 = torch.rand(8, 1, 32, 32), torch.rand(8, 1, 32, 32)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        z1, z2 = proj(bb(v1)), proj(bb(v2))
        loss = simsiam_loss(pred(z1), z2, pred(z2), z1)
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]            # training reduces the SSL loss

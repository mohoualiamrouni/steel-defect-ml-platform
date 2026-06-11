import torch

from steeldefect_ssl.data.augmentations import build_eval_transform, build_ssl_augment
from steeldefect_ssl.models.encoder import (build_encoder, build_prediction_head,
                                      build_projection_head)


def test_encoder_forward_shape_and_embedding():
    enc = build_encoder("ssl_resnet", in_channels=1, embed_dim=32, width=8)
    x = torch.rand(4, 1, 64, 64)
    out = enc(x)
    assert out.shape == (4, 32)
    assert torch.allclose(out, enc.get_embedding(x))


def test_encoder_deterministic_in_eval():
    torch.manual_seed(0)
    enc = build_encoder("ssl_resnet", 1, 16, 8).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))


def test_heads_shapes():
    proj = build_projection_head(32, 32, 16, n_layers=3, last_bn=True)
    pred = build_prediction_head(16, 8)
    z = proj(torch.rand(4, 32))
    assert z.shape == (4, 16)
    assert pred(z).shape == (4, 16)


def test_ssl_augment_two_views_differ():
    aug = build_ssl_augment(32)
    base = torch.rand(1, 32, 32)
    v1, v2 = aug(base), aug(base)
    assert v1.shape == (1, 32, 32)
    assert not torch.allclose(v1, v2)        # two independent views
    assert v1.min() >= -0.05 and v1.max() <= 1.05


def test_eval_transform_to_rgb():
    tf = build_eval_transform(32, to_rgb=True)
    out = tf(torch.rand(1, 32, 32))
    assert out.shape[0] == 3                  # replicated to 3 channels for resnet/PatchCore

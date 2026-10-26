import pytest

torch = pytest.importorskip("torch")

from aether_context.latent.projector import SoftPrefixProjector  # noqa: E402


def test_shape_norm_and_mask():
    torch.manual_seed(0)
    p = SoftPrefixProjector(d_model=16, slots=3, d_attn=8, embed_rms=0.5)
    states = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = p(states, mask)
    assert out.shape == (2, 3, 16)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 0.5), atol=1e-4)
    states2 = states.clone()
    states2[0, 3:] = 100.0  # padding positions cannot influence the output
    assert torch.allclose(p(states2, mask)[0], out[0], atol=1e-5)


def test_gradients_reach_parameters():
    p = SoftPrefixProjector(d_model=16, slots=2, d_attn=8, embed_rms=1.0)
    p(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)).sum().backward()
    assert all(t.grad is not None for t in p.parameters())


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        SoftPrefixProjector(d_model=16, slots=0, d_attn=8, embed_rms=1.0)

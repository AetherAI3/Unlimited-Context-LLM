import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend  # noqa: E402
from aether_context.latent.training import GRAD_CLIP, TrainExample, train_projector  # noqa: E402
from tests.latent_tiny import tiny_pair  # noqa: E402


def test_training_moves_only_the_projector_and_lowers_loss():
    model, tok = tiny_pair()
    b = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=2, d_attn=16, system="Answer with one word")
    ex = [TrainExample("Record for Ada . Ada lives in Paris .", "What city does Ada live ?", "Paris"),
          TrainExample("Record for Ada . Ada lives in Oslo .", "What city does Ada live ?", "Oslo")]
    book = b.codebook_digest()
    stats = train_projector(b, ex, steps=60, batch_size=2, lr=3e-3, seed=0)
    assert stats.base_digest_before == stats.base_digest_after
    assert b.codebook_digest() != book
    assert stats.steps == 60 and stats.examples_seen == 120 and len(stats.losses) == 60
    assert sum(stats.losses[-10:]) / 10 < sum(stats.losses[:10]) / 10
    assert stats.sender_tokens > 0 and stats.receiver_tokens > 0 and stats.head_tokens == 120
    assert stats.trainable_params == sum(p.numel() for p in b.projector.parameters())
    assert stats.wall_s > 0 and not b.projector.training
    assert stats.grad_clip == GRAD_CLIP == 1.0
    assert stats.codebook_digest == b.codebook_digest() != book
    assert stats.adapter_id == b.adapter_id
    assert stats.sender_layer == b.sender_layer and stats.slots == 2


def test_rejects_empty_examples():
    model, tok = tiny_pair()
    b = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=2, d_attn=16)
    with pytest.raises(ValueError):
        train_projector(b, [], steps=1, batch_size=1, lr=1e-3, seed=0)

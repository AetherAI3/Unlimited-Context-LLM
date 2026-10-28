import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

from aether_context.contracts import NamespaceV1, digest  # noqa: E402
from aether_context.contracts.latent_v1 import verify_latent_use  # noqa: E402
from aether_context.latent import NativeMemoryLLM, handoff  # noqa: E402
from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend  # noqa: E402
from tests.latent_tiny import tiny_pair  # noqa: E402

NS = NamespaceV1(owner_id="o", project_id="p", objective_id="obj", context_cycle_id="c")
T0 = 1_700_000_000
DOC = "Record for Ada . Ada lives in Paris ."
Q = "What city does Ada live ?"


@pytest.fixture()
def backend():
    model, tok = tiny_pair()
    return HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=4, d_attn=16, system="Answer with one word")


def capsule_for(b, text=DOC, cid="cap-1"):
    return b.encode_capsule(text, namespace=NS, capsule_id=cid, anchor_digest=digest({"text": text}),
                            base_revision=digest("rev"), created_at=T0)


def grant(cap):
    return handoff.issue_grant(cap, grant_id="g", lane_id="l7", sender_principal_id="s", recipient_principal_id="r",
                               fence=1, policy_digest=digest("pol"), issued_at=T0)


def test_base_model_frozen_projector_trainable(backend):
    assert not any(p.requires_grad for p in backend.model.parameters())
    assert all(p.requires_grad for p in backend.projector.parameters())
    assert isinstance(backend, NativeMemoryLLM)


def test_layout_splits_template_once(backend):
    lay = backend.layout(Q)
    ids = backend.tokenizer(Q, add_special_tokens=False).input_ids
    assert lay.before and lay.after[: len(ids)] == ids


def test_encode_capsule_manifest(backend):
    cap, raw = capsule_for(backend)
    assert cap.slots == 4 and cap.tensor.shape == [4, 32] and cap.tensor.dtype == "float16"
    assert cap.source_model_id == "tiny/llama@0" and cap.source_adapter_id == backend.adapter_id
    assert cap.codebook_digest == backend.codebook_digest() and cap.lossless is False
    assert len(raw) == cap.tensor.byte_count


def test_admit_paired_injects_with_verifiable_receipt(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    adm = backend.admit(cap, g, raw, Q, receipt_id="r1", now=T0 + 1)
    assert adm.prefix is not None and tuple(adm.prefix.shape) == (4, 32)
    start = len(backend.layout(Q).before)
    assert adm.receipt.outcome == "INJECTED" and adm.receipt.positions == [start + i for i in range(4)]
    verify_latent_use(cap, g, adm.receipt)


def test_admit_tampered_refused(backend):
    cap, raw = capsule_for(backend)
    adm = backend.admit(cap, grant(cap), bytes([raw[0] ^ 1]) + raw[1:], Q, receipt_id="r1", now=T0 + 1)
    assert adm.prefix is None and adm.receipt.outcome == "REFUSED"
    assert adm.receipt.reason_code == "TENSOR_DIGEST_MISMATCH"


def test_admit_wrong_principal_or_stale_fence_refused(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    wrong = backend.admit(cap, g, raw, Q, receipt_id="r1", now=T0 + 1, principal_id="other")
    assert wrong.prefix is None and wrong.receipt.outcome == "REFUSED"
    assert wrong.receipt.reason_code == "PRINCIPAL_MISMATCH"
    stale = backend.admit(cap, g, raw, Q, receipt_id="r2", now=T0 + 1, principal_id="r", current_fence=2)
    assert stale.prefix is None and stale.receipt.reason_code == "STALE_FENCE"
    ok = backend.admit(cap, g, raw, Q, receipt_id="r3", now=T0 + 1, principal_id="r", current_fence=1)
    assert ok.prefix is not None and ok.receipt.outcome == "INJECTED"
    text, receipt = backend.receive(cap, g, raw, Q, receipt_id="r4", now=T0 + 1, max_tokens=2, principal_id="other")
    assert text is None and receipt.reason_code == "PRINCIPAL_MISMATCH"


def test_next_token_logprobs_depend_on_prefix(backend):
    a = backend.encode_prefixes([DOC])[0]
    b = backend.encode_prefixes(["Record for Ada . Ada lives in Oslo ."])[0]
    lp = backend.next_token_logprobs([torch.from_numpy(a), torch.from_numpy(b), None], [Q, Q, Q])
    assert lp.shape[0] == 3
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-4)
    assert not torch.allclose(lp[0], lp[1]) and not torch.allclose(lp[0], lp[2])


def test_batched_scoring_matches_single(backend):
    p = torch.from_numpy(backend.encode_prefixes([DOC])[0])
    short = "Answer with one word ?"
    batched = backend.next_token_logprobs([p, None], [short, Q])  # the prefixed row is the shorter one
    assert torch.allclose(backend.next_token_logprobs([p], [short])[0], batched[0], atol=1e-4)
    assert torch.allclose(backend.next_token_logprobs([None], [Q])[0], batched[1], atol=1e-4)
    padded = backend.next_token_logprobs([p, None], [Q, short])  # and with the prefixed row the longer one
    assert torch.allclose(backend.next_token_logprobs([p], [Q])[0], padded[0], atol=1e-4)
    assert torch.allclose(backend.next_token_logprobs([None], [short])[0], padded[1], atol=1e-4)


def test_text_context_arm(backend):
    with_ctx = backend.next_token_logprobs([None], [Q], contexts=[DOC])
    assert with_ctx.shape[0] == 1
    assert not torch.allclose(with_ctx, backend.next_token_logprobs([None], [Q]))


def test_answer_loss_matches_next_token_logprob_for_one_token_answer(backend):
    prefixes = backend.project(backend.sender_states([DOC]))
    loss, counts = backend.answer_loss(prefixes, [Q], ["Paris"])
    gold = backend._ids("Paris")
    assert len(gold) == 1 and counts["head_tokens"] == 1
    lp = backend.next_token_logprobs([prefixes[0].detach()], [Q])
    assert abs(float(loss.detach()) + float(lp[0, gold[0]])) < 1e-5


def test_empty_answer_rejected(backend):
    prefixes = backend.project(backend.sender_states([DOC]))
    with pytest.raises(ValueError):
        backend.answer_loss(prefixes, [Q], [""])


def test_mismatched_lengths_raise(backend):
    p = backend.project(backend.sender_states([DOC, DOC]))
    with pytest.raises(ValueError):
        backend.next_token_logprobs([None, None], [Q])
    with pytest.raises(ValueError):
        backend.next_token_logprobs([None], [Q], contexts=[DOC, DOC])
    with pytest.raises(ValueError):
        backend.answer_loss(p, [Q], ["Paris", "Paris"])
    with pytest.raises(ValueError):
        backend.answer_loss(p, [Q, Q], ["Paris"])


def test_answer_loss_backprops_only_to_projector(backend):
    prefixes = backend.project(backend.sender_states([DOC, DOC]))
    loss, counts = backend.answer_loss(prefixes, [Q, Q], ["Paris", "Paris"])
    loss.backward()
    assert counts["head_tokens"] == 2 and counts["tokens"] > 2
    assert all(p.grad is None for p in backend.model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backend.projector.parameters())


def test_receive_generates_and_refuses_expired(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    text, receipt = backend.receive(cap, g, raw, Q, receipt_id="r1", now=T0 + 1, max_tokens=3)
    assert isinstance(text, str) and receipt.outcome == "INJECTED"
    expired, r2 = backend.receive(cap, g, raw, Q, receipt_id="r2", now=T0 + 10_000, max_tokens=3)
    assert expired is None and r2.reason_code == "GRANT_EXPIRED"


def test_projector_state_round_trip(backend, tmp_path):
    path = tmp_path / "proj.pt"
    backend.save_projector(path)
    model, tok = tiny_pair()
    other = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=4, d_attn=16,
                                system="Answer with one word", seed=99)
    assert other.codebook_digest() != backend.codebook_digest()
    other.load_projector(path)
    assert other.codebook_digest() == backend.codebook_digest() and other.adapter_id == backend.adapter_id


def test_flop_params(backend):
    blocks, head = backend.flop_params()
    assert blocks > 0 and head == 32 * backend.model.config.vocab_size


def test_admit_injected_slice_equals_prefix(backend):
    cap, raw = capsule_for(backend)
    adm = backend.admit(cap, grant(cap), raw, Q, receipt_id="r1", now=T0 + 1)
    seq, lay = backend._sequence(adm.prefix, Q)
    start = len(lay.before)
    assert torch.equal(seq[start:start + 4], adm.prefix.to(seq.dtype))
    assert adm.receipt.mask_digest == digest({"mask": [1] * (len(lay.before) + 4 + len(lay.after))})


def test_capsule_from_before_projector_change_is_refused(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    with torch.no_grad():
        for p in backend.projector.parameters():
            p.add_(0.01)
    adm = backend.admit(cap, g, raw, Q, receipt_id="r1", now=T0 + 1)
    assert adm.prefix is None and adm.receipt.reason_code in {"SOURCE_MISMATCH", "RECIPIENT_MISMATCH"}


def test_cached_adapter_id_is_used_when_given(backend):
    cap, raw = capsule_for(backend)
    adm = backend.admit(cap, grant(cap), raw, Q, receipt_id="r1", now=T0 + 1, adapter_id=backend.adapter_id)
    assert adm.prefix is not None


def test_identity_covers_codec_config():
    def make(**kw):
        model, tok = tiny_pair()
        args = dict(model_id="tiny/llama@0", slots=4, d_attn=16, system="Answer with one word", seed=1)
        args.update(kw)
        return HFSoftPrefixBackend(model, tok, **args)

    base = make()
    cfg = base.codec_config()
    assert cfg["system"] == "Answer with one word" and cfg["slots"] == 4 and cfg["model_id"] == "tiny/llama@0"
    assert {"codec_id", "codec_version", "sender_layer", "d_attn", "storage_dtype"} <= set(cfg)
    assert make().codebook_digest() == base.codebook_digest()  # same config + seed -> same identity
    assert make(system="Be brief").codebook_digest() != base.codebook_digest()  # identical weights, new config
    assert make(sender_layer=1).codebook_digest() != base.codebook_digest()
    assert make(model_id="tiny/llama@1").codebook_digest() != base.codebook_digest()
    assert make(storage_dtype="float32").codebook_digest() != base.codebook_digest()


def test_load_projector_refuses_other_config(backend, tmp_path):
    path = tmp_path / "proj.pt"
    backend.save_projector(path)
    for kw in ({"sender_layer": 1}, {"system": "Be brief"}):
        model, tok = tiny_pair()
        args = dict(model_id="tiny/llama@0", slots=4, d_attn=16, system="Answer with one word")
        args.update(kw)
        other = HFSoftPrefixBackend(model, tok, **args)
        with pytest.raises(ValueError):
            other.load_projector(path)


def test_construction_leaves_global_rng_untouched():
    model, tok = tiny_pair()
    before = torch.get_rng_state()
    HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=2, d_attn=8, seed=5)
    assert torch.equal(before, torch.get_rng_state())


def test_from_pretrained_requires_pinned_revision():
    with pytest.raises(ValueError):
        HFSoftPrefixBackend.from_pretrained("some/model")

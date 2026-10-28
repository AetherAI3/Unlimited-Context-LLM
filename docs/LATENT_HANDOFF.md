# Learned latent handoff (optional `[latent]` extra)

`aether_context.latent` lets one frozen local model pass context to another instance of the same
model as a **learned soft-prefix capsule** instead of text. Every handoff carries the merged
`latent_v1` manifests (see [LATENT_CONTRACTS.md](LATENT_CONTRACTS.md)). This is lane L7 (W7) of
the Predator / Atlas V2 plan. It is experimental: whether the capsule carries usable content is
measured, not assumed. See `bench/latent_transfer/REPORT.md`.

```bash
pip install "aether-context[latent]"   # torch + transformers + pydantic
```

The NumPy-only core never imports this package. Importing `aether_context.latent` itself pulls
in no torch; only `projector`, `hf_soft_prefix` and `training` do.

## Pieces

| Module | What it does | Needs |
|---|---|---|
| `protocol.NativeMemoryLLM` | Protocol with `encode_capsule(text, …) -> (LatentCapsuleV1, bytes)` and `receive(capsule, grant, bytes, prompt, …) -> (text or None, LatentUseReceiptV1)` | stdlib |
| `tensor_codec` | ndarray ⇄ (`TensorRefV1`, bytes). float16/float32 only; digest, byte count and shape are checked before any byte is interpreted | numpy, pydantic |
| `handoff` | `issue_grant` (same-model pair, short TTL), `admission_reason` (ordered reason codes; nothing is injected unless it returns `None`), `injected_receipt` / `refused_receipt` | numpy, pydantic |
| `projector.SoftPrefixProjector` | K learned queries cross-attend over the sender's hidden states; slots are rescaled to the receiver's embedding RMS | torch |
| `hf_soft_prefix.HFSoftPrefixBackend` | Frozen HF causal LM as sender (mid-layer states → projector → capsule) and receiver (capsule spliced into `inputs_embeds` inside the chat template) | torch, transformers |
| `training.train_projector` | Optimises the projector only; the base weights' digest must match before and after | torch, transformers |

## Guarantees and limits

- **Frozen base.** Every base parameter has `requires_grad=False`. The receiver is differentiable
  only through `inputs_embeds`, which is how the loss reaches the projector.
- **Identity.**
  - `adapter_id` hashes the projector weights together with the codec config: model id (a pinned
    revision is required), sender layer, slot count, attention width, system prompt and storage
    dtype.
  - A capsule from any other configuration is refused with `SOURCE_MISMATCH` or
    `RECIPIENT_MISMATCH`.
  - Saved projectors refuse to load under another config.
- **Manifests stay float-free.** Tensors travel as separate bytes referenced by sha256, dtype and
  shape. `canonical` / `digest` apply unchanged.
- **Lossy by contract.** `LatentCapsuleV1.lossless` is always false. The exact source record is
  the anchor, whose digest every capsule carries; anything that must be exact has to come from
  the anchor, never from the capsule.
- **Grants are unsigned manifests.** Who issued a grant must be established by the transport or
  authority layer. `policy_digest` is carried, not evaluated. `principal_id` and `current_fence`
  are checked when the receiver supplies them.
- **Receipt scope.** The receipt describes the unpadded receiver sequence for one question. Score
  or generate with the admitted prefix and that same question.

## Example

```python
from aether_context.contracts import NamespaceV1, digest
from aether_context.latent import handoff
from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend

pair = HFSoftPrefixBackend.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                           revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                           slots=8, system="Answer with a single word.")
pair.load_projector("projector.pt")  # trained with training.train_projector
ns = NamespaceV1(owner_id="me", project_id="p", objective_id="o", context_cycle_id="c")
record = {"text": "Record for Mara. Mara lives in Tokyo."}
capsule, tensor = pair.encode_capsule(record["text"], namespace=ns, capsule_id="cap-1",
                                      anchor_digest=digest(record), base_revision=digest("rev-1"),
                                      created_at=1_700_000_000)
grant = handoff.issue_grant(capsule, grant_id="g-1", lane_id="lane", sender_principal_id="a",
                            recipient_principal_id="b", fence=1, policy_digest=digest("policy"),
                            issued_at=1_700_000_000)
text, receipt = pair.receive(capsule, grant, tensor, "Where does Mara live?", receipt_id="r-1",
                             now=1_700_000_001, principal_id="b", current_fence=1)
```

Whether the answer depends on the capsule's content is exactly what the preregistered bench
measures. Read the report before relying on it.

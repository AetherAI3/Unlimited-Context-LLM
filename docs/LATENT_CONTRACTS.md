# Latent handoff contracts (additive, V1)

`aether_context.contracts.latent_v1` adds three manifests for the learned
handoff seam (master plan W7) without touching the hosted SC-CONTEXT-01 set:

| Model | Purpose |
|---|---|
| `LatentCapsuleV1` | A learned, source-linked capsule: model/adapter/codec identity, slot count, a `TensorRefV1` (sha256 + dtype + shape + byte_count), the exact-IR `anchor_digest` and `base_revision`. `lossless` is always `false`. |
| `ContextHandoffGrantV1` | Short-lived (at most 7 days) authority for one recipient principal/model/adapter to inject one capsule under a fence and policy digest. |
| `LatentUseReceiptV1` | The receiver's injection receipt: injection point, slots and positions, template/mask digests, `INJECTED` / `REFUSED` / `NOT_RUN` with a reason code. |

Manifests contain integers, strings and hashes only, so the existing
`canonical` / `digest` functions apply unchanged. Tensor bytes are separate
hashed objects and never ride inside a manifest. The twenty-one exported hosted
schemas in `schema-v1.json` are unchanged, and the NumPy-only core never imports
this module. Nothing trains, injects or transports a capsule yet; that is lane L7.

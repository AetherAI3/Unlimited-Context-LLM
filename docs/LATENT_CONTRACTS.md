# Latent handoff contracts (additive, V1)

`aether_context.contracts.latent_v1` exports four models for the learned
handoff seam (master plan W7) without touching the hosted SC-CONTEXT-01 set.
`TensorRefV1` is the sub-object of `LatentCapsuleV1`; the other three are
top-level manifests. The module requires the optional hosted extra (pydantic).

| Model | Purpose |
|---|---|
| `TensorRefV1` | Sub-object: a hashed byte object (sha256 + dtype + shape + byte_count). Shape dimensions and `byte_count` are bounded to the JSON-safe integer range (2^53 - 1), and `byte_count` must equal the shape product times the dtype size. |
| `LatentCapsuleV1` | A learned, source-linked capsule: model/adapter/codec identity, slot count, a `TensorRefV1`, the exact-IR `anchor_digest` and `base_revision`. `lossless` is always `false`. |
| `ContextHandoffGrantV1` | Short-lived authority for one recipient principal/model/adapter to inject one capsule under a lane-scoped fence and policy digest. The grant lifetime is capped at 900 seconds (the hosted short-lived authority cap, same as `RetentionCommandV1`). |
| `LatentUseReceiptV1` | The receiver's injection receipt: injection point, slots and positions, template/mask digests, `INJECTED` / `REFUSED` / `NOT_RUN` with a reason code. Template and mask digests are required for `INJECTED` and must be absent otherwise. It carries `used_at` only. |

The grant's `fence` is the lane fence for `(namespace.context_cycle_id, lane_id)`,
matching `CapabilityV1`; the receipt repeats `lane_id` and `fence`. Receipt
`positions[i]` is the injected position of capsule slot `i`, and positions are
strictly increasing.

## Cross-model binding

`verify_latent_use(capsule, grant, receipt)` is a pure function that raises
`ValueError` on the first failed check:

1. the grant's `capsule_digest` is the digest of the capsule;
2. the receipt's `grant_digest` is the digest of the grant;
3. the receipt's `capsule_digest` equals the grant's;
4. the grant's `anchor_digest` equals the capsule's;
5. capsule, grant and receipt share one namespace;
6. grant and receipt share `lane_id` and `fence`;
7. the receipt's recipient model and adapter equal the grant's;
8. `grant.max_slots <= capsule.slots` and `receipt.slots_injected <= grant.max_slots`;
9. `grant.issued_at <= receipt.used_at < grant.expires_at`.

Manifests contain integers, strings and hashes only, so the existing
`canonical` / `digest` functions apply unchanged. Tensor bytes are separate
hashed objects and never ride inside a manifest. The twenty-one exported hosted
schemas in `schema-v1.json` are unchanged, and the NumPy-only core never imports
this module. Nothing trains, injects or transports a capsule yet; that is lane L7.

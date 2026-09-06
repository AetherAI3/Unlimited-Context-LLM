# Hosted context canary (SC-CONTEXT-01)

The optional `hosted` extra adds a separate, model-independent context engine.
The existing local `Session` API remains available. This candidate is a bounded
lexical implementation; it does not establish a billion-token capacity claim.

## Build and verification

Install this exact source revision with `pip install '.[dev]'`. Run `ruff check
aether_context tests`, `python -m mypy aether_context`, and `python -m pytest -q`.
Run `python -m scripts.export_hosted_schema` to regenerate the twenty-one closed wire
schemas. Signed interoperability fixtures are synthetic and never deployment
or promotion credentials.

The package installs both `aether-context` and `aether-contextd`; it owns no
`aether_agent` import namespace or `aether` command. Do not publish a changed
wheel under an already published version. Cross-repository integration must
pin the Git revision and record the built wheel checksum until a new release.

## Trust and persistence

Gateway signs reservation, binding, capability, control and restoration grants.
The daemon signs receipts. An independent verifier signs closure proofs. Each
key has a distinct role; worker/model text cannot mint any of these identities.
Ed25519 signatures cover canonical UTF-8 JSON with ASCII keys and safe integers.
Floats, unknown fields and unpaired surrogates are rejected.

SQLite WAL/FULL transactions commit encrypted segments, HMAC lexical postings,
chain roots, operations and content-free events together. AES-GCM binds each
record's exact namespace, ID and metadata digest. Retrieval applies exact cycle,
plane and lane eligibility before ranking and limiting candidates. P0 remains
authority; P1 is a checksum-pinned encrypted graph snapshot; P2/P3/P4 carry
admitted shared evidence, private scratch and verifier candidates respectively.
No global retrieval fallback exists. Capsules are immutable per task/turn and
their complete injected content is bounded conservatively by UTF-8 bytes.

Startup audits active chains and derived postings. Detected corruption
quarantines the affected cycle durably. Reads/checkpoints that detect integrity
failure also quarantine after rolling back their failed operation. Secret
patterns reject the whole record before persistence. Only visible summaries
and tool results belong in the pool; hidden reasoning must never be supplied.

Checkpoints contain authority, pinned graph, validated chain and replay records,
encrypted as one bounded pack. New cycles use one random DEK exposed only by an
opaque `key_ref`; the signed checkpoint receipt and encrypted pack bind that
reference and provider. A replacement host must resolve it through the same
authenticated shared provider, verify the exact remote-selected checksum and
receipt, and receive a new fence. Its signed hydrate receipt binds the manifest,
checkpoint number, cursor, chain root, key reference, replay digest and fence.
Legacy unkeyed packs remain readable with the original context key. A restored
SEALING cycle cannot reopen worker writes.

The V1 rolling boundary is deliberate. A pre-retention V1 binding that omits
`retention_class` is interpreted as `ephemeral`, while its original signed JSON
and digest are preserved byte-for-canonical-byte rather than rewritten with a
new default. A checkpoint that omits `key_ref` and `key_provider` is accepted
only with the legacy grant shape and original context key. Keyed checkpoints
require both fields in receipt, pack and grant. A profile digest or pinned index
version change fails explicitly; it is never silently reinterpreted as an
in-place upgrade.

Freeze receipts publish a sorted, text-free promotion manifest containing only
record ID, content digest and evidence refs. Its signature also binds the cycle
cursor, chain root and binding digest. An independent exact-head proof chooses
from those entries; seal recomputes the accepted promotion root from the same
canonical entry shape.

## Daemon deployment

`deploy/aether-contextd.service` is an operator-reviewed Linux template. It
uses a dedicated user, a private Unix socket, managed systemd credentials,
read-only system protection, no IP address families and a 512 MiB memory limit.
The local filesystem must meet the approved encryption and headroom policy.
The template does not create users, provision credentials or assert any gate.

Credential names: `context-dek`, `context-cycle-wrapper-key` (32 bytes each),
`context-signing-key` (Ed25519 seed), `context-authority-keys.json`,
`context-proof-keys.json`, `context-checkpoint-keys.json`,
`context-remote-deletion-keys.json`, `context-benchmark-keys.json`,
`context-executor-stability-keys.json`, and
`context-data-plane-isolation-keys.json` (key-ID to base64 public-key maps).
`context-runtime-build-keys.json` verifies
`context-runtime-build-manifest.json`, which binds the source revision, wheel
digest, approved recall-dataset digest and a digest of every installed runtime
source/schema byte. `wheel_digest` is signed supply-chain provenance; the daemon
enforces a digest of the exact imported package-tree bytes because it does not
independently retain or reconstruct the wheel archive. A deployment that needs
wheel-byte enforcement must supply a separately authenticated wheel artifact and
verifier.
The manifest also pins the approved hosted data-plane case-generator digest;
older manifests without that field remain readable but cannot enable reach. An
operator environment string cannot qualify a build.
`context-benchmark-receipt.json` is the signed measured receipt. Build,
benchmark, executor-stability, deletion, Gateway, verifier and service keys
must be cryptographically distinct, including when their key IDs differ. The
profile JSON must pin the deployed SQLite version and benchmark-envelope digest.
Supply all three
`--restore-pack`, `--restore-grant`, and `--restore-receipt` options together
for a recovery start; none of these paths may be chosen by a worker.

Do not delete an active or unsealed pool for disk reclamation. Freeze, checkpoint,
remotely verify, fence and restore according to the cloud runbook. The signed
retention API supports idempotent legal/audit holds, releases and expiry. Default
raw retention is 24 hours after seal and the profile rejects values above seven
days. When a checkpoint exists, secure expiry first verifies an independently
signed, short-lived remote object deletion receipt against the exact namespace,
latest checkpoint checksum and authority-bound object reference. It then destroys
the managed cycle key only through a currently qualified shared provider. A
closed, independently signed KMS destruction receipt must bind the exact
provider, opaque key reference, namespace digest, key version and destroyed
state. The durable service deletion receipt embeds that evidence and both proof
digests. A crash between phases persists the exact operation key and request
digest before key destruction. Only that request can resume from the persisted
remote proof and provider's idempotent destruction result; a competing operation
cannot take over the pending cleanup. Every command carries the cycle
revision it observed, so a stale release cannot clear a newer legal or incident
hold. The legacy `expire` operation fails closed; legacy cycles use this same
signed retention path and remote deletion proof.

`FileCycleKeyProvider` and the systemd template's local cycle-key directory are
canary-only. They exercise wrapping, replacement hydrate and crash-safe cleanup,
but their local tombstone can never claim production cryptographic erasure.
Production needs an injected shared managed KMS/key provider, a short-lived
independently signed provider qualification, independently signed exact-version
destruction evidence, and remote deletion trust roots. Hosted health validates
those receipts on each admission and remains false after they expire. During key
rotation, the provider must resolve the immutable version for every opaque
`key_ref`, and every version used by a non-expired cycle must retain its own valid
signed provider qualification. The `--managed-key-provider-receipt` JSON file may
contain one signed envelope or an array of signed envelopes; old versions remain
configured until their last cycle has been securely expired. A custom production
unit supplies `context-key-provider-keys.json`,
`context-key-destruction-keys.json`, and the signed provider receipt alongside
its managed provider implementation; the checked-in file-provider unit cannot
cross that gate.

Health reports `managed_key_provider_receipt_digests` keyed by qualified key
version. The older singular `managed_key_provider_receipt_digest` remains
populated only when exactly one version is qualified.

## Measured limits and remaining release gates

Defaults: 10,000 records, 64 MB indexed plaintext, 32 KiB per record, 8 MB pinned
snapshot, 4,096 capsule tokens, 256 candidates, four active cycles per owner and
600 append/retrieve calls per minute. The profile is immutable after binding.
Operation and checkpoint quotas are independent; snapshots are full bounded
packs rather than streaming deltas. Larger profiles require new measurements.

`python -m bench.hosted_scale` is the production-shaped measurement harness. It
refuses a dirty or changing Git tree, fills at least 90% of both configured record
and byte capacity, measures concurrent retrieval, checkpoint, replacement-host
hydrate and an actual atomic posting-index rebuild, and runs exactly one million
deterministic owner/project/objective/cycle and peer-lane predicate cases. Recall
comes from a caller-supplied immutable `HostedRecallDatasetV1`; its name and digest
are signed into the result. Unique generated lookup tokens are not recall proof.

```console
python -m bench.hosted_scale \
  --profile /etc/aether/context/profile-to-measure.json \
  --repository /opt/aether-context/source \
  --recall-dataset /opt/aether-context/evidence/project-recall-v1.json \
  --work-directory /var/tmp/aether-context-scale \
  --signing-key /run/credentials/scale-signing-key --key-id scale-v1 \
  --executor-stability-receipt /run/evidence/executor-stability.json \
  --executor-stability-keys /run/credentials/executor-stability-keys.json \
  --output /run/evidence/hosted-scale-receipt.json \
  --receipt-bound-profile-output /run/evidence/hosted-profile.json
```

The harness never manufactures executor stability. Without a separately signed
`ContextExecutorStabilityReceiptV1`, it records `executor_stable=false`, so the
result cannot enable reach. Qualification verifies that independent receipt,
the signed runtime build, approved dataset digest, representative capacity,
memory, latency, recovery and recall thresholds. The local million-case result
is labeled `predicate` and is deliberately nonqualifying. A separate randomized,
independently signed million-case receipt measured through the real hosted data
plane must be embedded and labeled `data_plane` before the isolation gate can
pass. Qualification compares that receipt's case-generator digest to the one
pinned by the independent signed runtime build manifest. The profile must then
point to the exact signed benchmark envelope. Hosted
`reach_claim_enabled` stays false whenever overall hosted readiness is false.

These local measurements and chaos tests do not satisfy Gate 8 or establish a
one-billion-token claim. Real data-plane isolation, provider, worktree, CI,
GitHub, checkpoint, Project Memory evidence and the agreed live observation
window remain external rollout gates.

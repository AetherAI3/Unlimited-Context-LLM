# Hosted context canary (SC-CONTEXT-01)

The optional `hosted` extra adds a separate, model-independent context engine.
The existing local `Session` API remains available. This candidate is a bounded
lexical implementation; it does not establish a billion-token capacity claim.

## Build and verification

Install this exact source revision with `pip install '.[dev]'`. Run `ruff check
aether_context tests`, `python -m mypy aether_context`, and `python -m pytest -q`.
Run `python -m scripts.export_hosted_schema` to regenerate the closed wire
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

A `reserve_v2` reservation receipt carries the current cycle revision. The
original `reserve` operation and `ContextReservationReceiptV1` keep their exact
rolling wire shape. If one-use
authorization fails before objective dispatch starts, Gateway may submit a
signed `ContextReservationReleaseCommandV1` through the
`release_reservation` IPC operation. The closed command binds owner, project,
idempotency key, request/profile digests, reservation revision, a digest of the
authorization-failure receipt and the literal assertion `dispatch_started=false`.
The daemon releases only an exact never-bound reservation. Exact retries return
the persisted signed receipt, including after command expiry; a different proof
conflicts. A new signed `ContextReservationCommandV2` may revive the same cycle
only when it commits the exact released revision. The corresponding `bind_v2`
command embeds that signed revisioned receipt, so a delayed pre-release bind
cannot bind the revived attempt. Once binding exists, an old release cannot
change the cycle.

Two separate signed terminals cover the remaining never-bound cases. After
objective dispatch starts, `abort_reservation` embeds the exact daemon-signed
V2/V3 reservation receipt and the Gateway dispatch receipt digest, then moves
only that clean reservation to `ABORTED`; `expire_aborted_reservation` later
emits its unregistered V2 deletion receipt. If dispatch never starts, Cloud
first atomically seals its generation-bound dispatch fence and signs the exact
`ContextInvocationNoDispatchSnapshotV1`. `expire_reservation` verifies that
snapshot, its digest, the exact V2/V3 reservation receipt and current revision
before moving the empty row to `EXPIRED`. Only this no-dispatch terminal is
revivable: `reserve_v3` embeds the prior expiry receipt and returns a signed V3
receipt carrying the expiry, fence and revival-command digests; `bind_v3`
requires that exact receipt. Delayed commands from any earlier generation fail
their revision or lineage checks.

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

Gateway registers one immutable checkpoint policy before the first pack with
the `checkpoint_durability` IPC operation and a signed
`ContextCheckpointDurabilityCommandV1`. `local_ephemeral` is the Gate 8 steps
1–3 mode: the separate signed registration receipt carries that mode, hydrate is denied, and expiry
rejects remote-deletion evidence because the pack was never registered for
remote persistence. Fresh V2/V3 cycles remain `unregistered` and cannot
checkpoint until this signed registration is stored. The unchanged V1 reserve
path deliberately creates `remote_registered` legacy rows so an old client can
checkpoint, seal and restart while a deployment drains it. Rows that existed
before the durability column, including inserts from an old V1 writer, are
conservatively classified the same way. V1 is deprecated and cannot qualify the
new hosted protocol gates: deploy the new daemon first, drain or upgrade every
V1 client, and only then require V2/V3 capability negotiation. `remote_registered` is the
step 4+ mode: any checkpoint requires an exact,
independently signed remote-deletion receipt before managed-key destruction.
The registration is revision-bound, idempotent and immutable for the cycle. A
new checkpoint pack carries the exact signed registration evidence. Hydrate
verifies that evidence with the checkpoint trust roots, then stores a
replacement-host reattestation. A V2 deletion receipt for any nonlegacy cycle
binds that stored registration-receipt digest; only truly old remote rows emit
the original V1 deletion shape.

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
`context-runtime-build-manifest.json`. Production uses
`ContextRuntimeBuildManifestV3`, which binds the source revision, wheel digest,
approved recall-dataset digest, imported package-tree digest, actual import-root
digest, Python executable digest and ABI, SQLite version, and exact versions and
RECORD-verified artifact digests for the complete hosted import closure:
annotated-types, cffi, cryptography, numpy, pydantic, pydantic-core, pycparser,
typing-inspection, and typing-extensions. Generate the `RuntimeEnvironmentV2`
payload after installing the final wheel in its final
immutable virtual environment with `runtime_environment_identity()`, set
`locked_environment_digest` to `digest(environment)`, and sign the complete V3
manifest with the independent build key. V1/V2 manifests remain parseable for
rollback, but `runtime_environment_unbound` keeps hosted readiness off.

The qualified virtual environment is an intentionally minimal runtime, not an
ordinary development venv. Create it with `python3.12 -m venv --copies` so the
qualified interpreter path is not a symlink. Install the final wheel with `--no-compile`, remove
installer-only distributions such as pip, setuptools, and wheel, remove every
`__pycache__`, and make the interpreter, virtual-environment parents, package
tree, metadata, and manifest credentials root-owned and non-writable by the
service user before signing. Every importable top-level path must belong to the
signed closure; an optional dependency must either be absent or appear in that
closure with its exact version and complete RECORD artifact digest. The system
Python standard-library and dynamic-library roots are part of the host trust
base: they must also be root-owned, non-writable, and supplied by the approved
immutable host image because the guard necessarily imports stdlib code while
bootstrapping.

The systemd unit executes the installed `aether_context/guard.py` directly with
`python -I -S -B`. That stdlib-only guard authenticates the V3 manifest with its
Ed25519 build key before trusting the declared closure, checks the immutable
path and wheel RECORDs, rejects bytecode, package/native collisions, symlinks,
unlocked import roots, and artifact drift, and only then adds site-packages and
imports the daemon. Startup and explicit health or reservation admission
remeasure the complete environment. Other admission paths reuse that full
attestation for at most 60 seconds, remeasure it after expiry, and still check
disk headroom on every call.
The manifest also pins the approved hosted data-plane case-generator digest;
older manifests without that field remain readable but cannot enable reach. An
operator environment string cannot qualify a build.
`context-benchmark-receipt.json` is the signed measured receipt. Build,
benchmark, executor-stability, deletion, Gateway, verifier and service keys
must be cryptographically distinct, including when their key IDs differ. The
profile JSON must pin the deployed SQLite version and benchmark-envelope digest.
Supply all four `--restore-pack`, `--restore-grant`, `--restore-receipt`, and
`--restore-result` options together for a recovery start; none of these paths
may be chosen by a worker. The result is the signed `ContextHydrateReceiptV1`
for a legacy grant or `ContextHydrateReceiptV2` for the closed, source-bound V2
hydrate command, encoded as one canonical JSON line. It is published atomically with mode 0600
before the daemon begins serving. Replaying the same recovery accepts the exact
existing result, while a different pre-existing result fails closed.

An already running replacement daemon exposes the same operation through the
private socket as the exact request `{"operation":"hydrate","grant":...,
"checkpoint_receipt":...,"pack":"<canonical-base64>"}`. Checkpoint signer
keys come only from the daemon's `context-checkpoint-keys.json`; the caller
cannot supply trust roots. `ContextHealthV3` advertises the active emitted-pack
ceiling. The current copy-based transport admits at most 13,000,000 decoded
bytes and separately bounds decrypted checkpoint materialization at 16,000,000
bytes. Both the socket and startup recovery paths enforce those limits before
any database effect; the unchanged V1 profile can remain wider for direct,
non-daemon use. The encoded request uses the daemon's derived large-request limit. Exact
retries return the same signed hydrate receipt. A terminal, advanced or
incomplete restored cycle conflicts instead of reporting recovery success.

Do not delete an active or unsealed pool for disk reclamation. Freeze, checkpoint,
remotely verify, fence and restore according to the cloud runbook. The signed
retention API supports idempotent legal/audit holds, releases and expiry. Default
raw retention is 24 hours after seal and the profile rejects values above seven
days. When a remotely registered checkpoint exists, secure expiry first verifies
an independently signed, short-lived remote object deletion receipt against the
exact namespace, latest checkpoint checksum and authority-bound object reference.
A locally ephemeral checkpoint rejects remote evidence and proceeds directly to
local cleanup and managed-key destruction. It then destroys
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
`ContextHealthV1` preserves its original rolling response. `ContextHealthV2`
adds the observed package-tree digest, locked environment digest, Python
executable/ABI identity, dependency versions and artifact digests, SQLite
version, import-root digest and bytecode policy. Admission can compare those
values with the signed build manifest and data-plane case-generator identity as
one exact build tuple; a live measurement error returns a closed `ready=false`
V2 response. `ContextHealthV3` adds the guarded deployment-lock import closure,
the exact ordered protocol capability set, protocol version 3 and the active
live IPC checkpoint-pack ceiling. New Cloud callers negotiate this V3 evidence
before sending V2/V3 lifecycle or hydrate operations.

## Measured limits and remaining release gates

Defaults: 10,000 records, 64 MB indexed plaintext, 32 KiB per record, 8 MB pinned
snapshot, 4,096 capsule tokens, 256 candidates, four active cycles per owner and
600 append/retrieve calls per minute. The V1 checkpoint default remains 64 MB;
live IPC uses the smaller health-advertised ceiling above. The profile is immutable after binding.
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
  --data-plane-isolation-receipt /run/evidence/data-plane-isolation.json \
  --data-plane-isolation-keys /run/credentials/data-plane-isolation-keys.json \
  --output /run/evidence/hosted-scale-receipt.json \
  --receipt-bound-profile-output /run/evidence/hosted-profile.json
```

The harness never manufactures executor stability or data-plane isolation.
Without both separately signed receipts, it emits provisional evidence and
refuses `--receipt-bound-profile-output`, so the result cannot enable reach. On
Linux, checkpoint and replacement hydrate measurements traverse the real private
Unix socket and its JSON/base64 framing. Qualification verifies those independent receipts,
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

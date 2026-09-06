# Hosted context canary (SC-CONTEXT-01)

The optional `hosted` extra adds a separate, model-independent context engine.
The existing local `Session` API remains available. This candidate is a bounded
lexical implementation; it does not establish a billion-token capacity claim.

## Build and verification

Install this exact source revision with `pip install '.[dev]'`. Run `ruff check
aether_context tests`, `python -m mypy aether_context`, and `python -m pytest -q`.
Run `python -m scripts.export_hosted_schema` to regenerate the nine closed wire
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
encrypted as one bounded pack. Restoration requires the exact remote-selected
checksum, trusted historical receipt keys, the original decryption key and a
new fence. A restored SEALING cycle cannot reopen worker writes. An independent
exact-head proof and accepted P4 promotion root are required for a seal.

## Daemon deployment

`deploy/aether-contextd.service` is an operator-reviewed Linux template. It
uses a dedicated user, a private Unix socket, managed systemd credentials,
read-only system protection, no IP address families and a 512 MiB memory limit.
The local filesystem must meet the approved encryption and headroom policy.
The template does not create users, provision credentials or assert any gate.

Credential names: `context-dek` (32 bytes), `context-signing-key` (Ed25519 seed),
`context-authority-keys.json`, `context-proof-keys.json`, and
`context-checkpoint-keys.json` (key-ID to base64 public-key maps).
The profile JSON must pin the deployed SQLite version. Supply all three
`--restore-pack`, `--restore-grant`, and `--restore-receipt` options together
for a recovery start; none of these paths may be chosen by a worker.

Do not delete an active or unsealed pool for disk reclamation. Freeze, checkpoint,
remotely verify, fence and restore according to the cloud runbook. Expiry here
performs logical deletion while preserving the seal receipt. It is not a claim
of per-cycle cryptographic erasure: independent wrapped cycle keys, retention
holds and object-store lifecycle enforcement remain rollout requirements.

## Measured limits and remaining release gates

Defaults: 10,000 records, 64 MB indexed plaintext, 32 KiB per record, 8 MB pinned
snapshot, 4,096 capsule tokens, 256 candidates, four active cycles per owner and
600 append/retrieve calls per minute. The profile is immutable after binding.
Operation and checkpoint quotas are independent; snapshots are full bounded
packs rather than streaming deltas. Larger profiles require new measurements.

A Linux synthetic canary with 400 records (~262 KB text), 50 retrieval samples
and an 8 MiB SQLite cache observed p50 190 ms, p95 219 ms, peak process RSS
~72 MiB, checkpoint 0.27 s and startup audit 1.72 s. These numbers describe only
that run and include no provider calls. They do not meet the representative
pool-size, million-case isolation, chaos, real-provider or observation-window
requirements of Gates 2, 4, 5 and 8. `reach_claim_enabled` remains false.

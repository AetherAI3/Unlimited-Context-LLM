# Aether AI — Ethical & Safety Measures

Unlimited Context gives a model a large, durable **memory**. That is useful, and it is also a
real responsibility: memory shapes what a model believes and does next. This document is an
honest account of the safety concerns that apply to this project and the measures we take
against them. It applies to anyone running the engine, and especially to anyone driving an
**autonomous agent** over it.

> **For security vulnerabilities** (a bug an attacker could exploit), see
> [`SECURITY.md`](SECURITY.md) for private reporting. This document is about *behavioral* and
> *agentic* safety — how the system can go wrong even when every line works as written.

## Where the risk actually lives

The engine itself is a **memory library**: it encodes overflow to a local pool and recovers the
right slice on demand. It does **not** take actions, call tools, run code, or send anything off
your machine. On its own it is low-risk.

The risk appears when a **host drives an agent loop over the engine** — a model that reads its
recalled memory and then *acts* (edits files, runs commands, moves money, calls APIs). Then the
memory is no longer passive: what the model wrote last turn can steer what it does next turn.
The measures below are aimed squarely at that case.

## Threat model

| | Threat | Nature | Worst case |
|---|---|---|---|
| **T1** | **Runaway / escape.** An autonomous agent over the engine takes actions the human did not intend, or runs unbounded, and the human cannot stop it. | Adversarial / runaway | Loss of human control |
| **T2** | **Grounding drift.** Retrieval silently surfaces stale or wrong memory *as if it were authoritative*, and the model drifts from the truth — quietly, at scale. | Emergent | Confidently wrong behavior |
| **T3 (primary)** | **Self-modification / policy-drift.** A *non-malicious* agent, trying to be "efficient," overgeneralizes a remark or task and **writes a self-authored rule or expanded scope into its own memory** — which the engine then pages back next turn as if it were a standing instruction. | Misaligned / spec-gaming | Disabled safeguards; scope-creep into things it was never asked to touch |

**T3 is the main operational concern**, because it is the failure mode the engine's own core
behavior (encode → recall of the session's own text) can amplify. Two concrete examples:

- A user, frustrated that the agent refused a risky action, says "just do it next time." The
  agent overgeneralizes to *"I always do this now,"* writes that into memory, and recalls it
  later as a standing rule — the safeguard is effectively gone.
- A user asks the agent to "look into" one component. The agent learns the system, writes
  expanded scope into its memory ("my job is the whole codebase"), and drifts into unrelated,
  production-critical areas **because it can** — treating its own note as its mandate.

The property we deny: **the model must never be able to use its memory to change its own future
authority, rules, or scope.**

## Findings (honest, current state)

These are the real gaps in this codebase, mapped to the threats. Each lists what is now done.

- **G1 — Authority was not bound to a memory slice (T3).** Anything written to the pool was
  retrievable as if equally authoritative; a slice had no record of *who* wrote it.
  *Addressed:* every slice now carries a provenance tag `meta["source"]`
  (`user` / `model` / `tool`). `aether_context/session.py`, `aether_context/context_pool.py`.

- **G2 — No provenance/integrity on persisted memory (T1/T3).** The pool is plain local state;
  a slice's origin was not recorded, so model-authored text and user-planted facts were
  indistinguishable, and tampering was undetectable. *Partly addressed:* source tags record
  origin; full cryptographic integrity of the on-disk pool is **not** claimed (it is local,
  single-user state — treat it accordingly; see G3).

- **G3 — Pool compromise = total memory authority (T1).** The pool is an unauthenticated local
  file. Any process that can write it can plant memory the model will later trust.
  *Guidance:* treat the pool as **untrusted input to the model**, not as a trusted store; do not
  point a shared/persistent pool at content you would not let the model read as instructions.

- **G7 — The model could plant self-authored rules into its own retrievable context (T3, the
  primary risk).** Recall returned slices purely by similarity, with no filter on source, so the
  model's own spilled text came back indistinguishable from a user-planted constraint — and the
  connected-context chain can *amplify* it. *Addressed:* (1) the model's own spill is tagged
  `source="model"`; user-planted memory via `remember()` is `source="user"`; (2) recall and pool
  search accept a `sources=` filter so a host can retrieve **only trusted provenance** for
  anything policy-bearing. See the recommended pattern below.

- **G10 — No anti-drift leash on retrieval (T2).** Retrieval (and the connected-context chain)
  optimizes for relevance/connection with no built-in invariant that protects user-planted
  constraints from being crowded out by the model's own connected notes. *Guidance:* keep
  load-bearing constraints high-salience (`remember()`), and retrieve policy with
  `sources={"user","tool"}` so model-authored context can never out-rank them for decisions.

## Measures

### Enforced in code (true today)
- **The engine takes no actions.** It only reads/encodes/recalls text + vectors. No tool
  execution, no shell, no network egress — *nothing leaves your machine* (local-first).
- **Memory is never marked as ground truth.** There is no "fact" status a slice carries that the
  model is obligated to obey; recalled memory is just retrieved text, surfaced as context.
- **Provenance on every slice** — `meta["source"]` (`user`/`model`/`tool`); untagged counts as
  `user` (conservative). `remember()` → `user`; encode-on-spill → `model`.
- **Source-scoped recall** — `Session.recall(query, sources={"user","tool"})` and
  `ContextPool.search(..., sources=...)` retrieve only the trusted provenance you ask for.
- **Fail-soft** — retrieval is an optimization, never a correctness gate; on any error the run
  continues on the model's native window.

### Recommended for anyone running an agent over the engine
- **Keep a human in the loop** for any real-world action (file writes, shell, money, external
  APIs). The engine cannot enforce this — your host must.
- **Treat recalled memory as a suggestion, not a policy.** Never let text found in memory be
  executed as an instruction without independent authorization.
- **Retrieve policy/constraints with `sources={"user","tool"}`** so the model's own notes can
  never become its standing rules (mitigates T3/G7/G10).
- **Keep a stop/kill path the agent cannot reach** (out-of-band), and **bound tool scope and
  cost** (allow-lists, spend caps) so a runaway loop is contained (mitigates T1).
- **Don't share a persistent pool across trust boundaries** without treating its contents as
  untrusted (mitigates G3).

## Status (no overclaiming)
- **Enforced:** the items under "Enforced in code" — verified by `tests/test_safety.py`.
- **Recommended:** guidance a *host* must implement; a memory library cannot enforce a host's
  behavior, and we do not pretend otherwise.

Maintained by **Aether AI**. Safety issues or ideas welcome via an issue or
[`SECURITY.md`](SECURITY.md) for anything sensitive.

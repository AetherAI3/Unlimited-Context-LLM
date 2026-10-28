# Latent transfer (G-W7): preregistration

Committed before any confirmatory result file exists. Lane L7 of the Predator / Atlas V2 fanout (W7
learned handoff). The q-channel sandbox v0 (Predator-M #341) showed deterministic capsule handoff and
made no learned-transfer claim; this experiment tests whether a **learned** soft-prefix capsule carries
usable context between a frozen same-model pair.

## 1. Question

Does a capsule produced by a frozen sender model through a trained projector let a frozen receiver of
the same model answer follow-up questions about a document it never sees as text, on documents,
names and question phrasings never used in training? It counts only if the answers depend on the
capsule's content, which means the paired capsule must beat blank, shuffled and mismatched capsules
and no capsule at all.

## 2. System under test

| Item | Value |
|---|---|
| Model (sender = receiver, frozen) | `Qwen/Qwen2.5-0.5B-Instruct` at HF commit `7ae557604adf67be50417f59c2c2f167def9a775`, float32, CPU |
| Trainable part | `SoftPrefixProjector` only (`aether_context/latent/projector.py`). Every base parameter has `requires_grad=False`, and the base weights' digest is checked before and after training |
| Sender | Hidden states after transformer block 16 of 24 (`sender_layer = 2*24//3`) for the document text |
| Projector | 8 learned queries, single-head cross-attention, `d_attn = 512`, output rescaled to the receiver's mean input-embedding RMS |
| Capsule | 8 slots × 896 values as float16 bytes (14,336 bytes), referenced by `LatentCapsuleV1` (sha256 + dtype + shape). `lossless = false`. `anchor_digest` is the digest of the exact source record. The adapter id binds the projector weights and the codec config (model, layer, slots, d_attn, system prompt, storage dtype) |
| Receiver injection | The 8 slots go into `inputs_embeds` inside the chat template, between the user header and the question. System prompt: `Answer with a single word.` |
| Handoff authority | Each use is admitted only after `handoff.admission_reason` passes, with the grant issued from the capsule manifest. Grant TTL 300 s, `principal_id` and `current_fence` checked. Each grant is issued immediately before its use, so the TTL never binds a scored use; expiry is exercised by an integrity probe. A `LatentUseReceiptV1` and its grant are written for every use. Every INJECTED receipt must pass `verify_latent_use` in the receiver, and the report phase re-verifies it from the saved grant and capsule manifest |
| Weight loading | The pinned `model.safetensors` is read from the local HF cache into memory and loaded as a state dict rather than memory-mapped by `from_pretrained`: the mmap page-in crashed with an access violation on this memory-constrained host. The weights are the same; the base-weight digest is recorded in every phase |

## 3. Data (`bench/latent_transfer/tasks.py`, constants normative)

- **Dataset seed:** `20261025`. 2,000 training documents and 200 evaluation documents. The committed
  `dataset_digest` is the source of truth (Python only guarantees the `random()` sequence across
  versions). On CPython 3.13.5 it is
  `f64b64338ada8ae08b711c4111d48eb188bcf242afb3f4d16fbdc898e37dc0c8`.
- **Documents:** one person per document, with 4 attributes (city, colour, pet, job) drawn
  independently and uniformly from 8 values each. Sentence order is shuffled and each sentence uses
  one of 2 phrasings.
- **Names:**
  - Evaluation names: 200 unique names, sampled without replacement from a 234-name pool.
  - Training names: drawn from a disjoint 249-name pool.
  - No name contains any value word.
- **Training items:** every training document × every attribute (8,000 items), with a question from
  3 training templates per attribute. Loss is token cross-entropy on the answer word only.
- **Evaluation families** (none used in training):
  - `unseen_template` (primary): each of the 4 attributes once, asked with one of 2 unseen phrasings.
    The candidates are the 8 values. Chance is 0.125.
  - `yes_no` (secondary, unseen task family): 4 claims per document, exactly 2 true and 2 false
    (false value uniform over the other 7), answered Yes or No. Chance is 0.5, and any constant
    answer scores exactly 0.5 on every document.
- **Sample size:** 200 documents × 4 items per family = 800 items per family per arm.
- **Calibration pilot (not evidence):** see §9. It used dataset seed 999 and training-split
  documents only. No evaluation document, unseen template, yes/no item or control arm was ever
  scored before this file was committed.

## 4. Arms

All arms use the same question, template and candidates. Only what is injected differs.

| Arm | Receiver input | Role |
|---|---|---|
| `paired` | Capsule of the document the question is about | Treatment |
| `blank` | Capsule of `Record for {name}. No other details are on file.` (same name, no facts) | Control: capsule without facts |
| `shuffled` | Capsule of another evaluation document (seeded derangement, never the same name) | Control: a real capsule with the wrong content |
| `mismatched` | Capsule of a copy of the document with the same name, order and phrasings but every value changed | Control: right surface, wrong facts |
| `no_prefix` | Nothing injected | Control: model prior |
| `text_context` | The exact anchor text in the prompt, after its digest is checked against the capsule's `anchor_digest` | Reference ceiling (exact anchor), not a control |

## 5. Metric

- **Scoring:** for each item, the log-probability of each candidate's first answer token at the
  first answer position. The prediction is the arg-max over candidates. Candidates' first tokens
  are checked to be distinct under the model tokenizer, in the prepare phase and again in the
  receiver; the run aborts otherwise. A tie goes to the first tied candidate in the fixed order
  (the `ATTRIBUTES` order; Yes before No), and ties are logged per arm.
- **Per-document accuracy:** the mean correctness over the document's 4 items in a family (exactly
  4, enforced).
- **Comparisons:** paired over documents, as the per-document difference. A 95% percentile
  bootstrap CI uses 2,000 resamples of documents, seed 0. Both arms must cover the identical
  document set (strict).

## 6. Hypotheses and promotion rule

- **H1 (primary, G-W7 content dependence, unseen phrasings and documents):** on `unseen_template`,
  the lower 95% CI bound of `paired − X` is ≥ 0.10 for every X in {`blank`, `shuffled`,
  `mismatched`, `no_prefix`}. A bound within 1e-9 of 0.10 counts as meeting it (percentile
  interpolation over per-document differences that are multiples of 0.25).
- **H2 (secondary, unseen task family):** the same rule on `yes_no`, with margin 0.10.
- **H3 (descriptive, follow rate):** in `shuffled` and `mismatched`, the share of answers equal to
  the capsule source document's gold.
  - If content is carried, follow should be high for `unseen_template`. Under faithful following,
    own-gold accuracy is about 0.125 for shuffled and exactly 0 for mismatched.
  - For `yes_no`, own-gold accuracy under faithful following is about 0.5 (shuffled) and about 0.43
    (mismatched), so the shuffled and blank comparisons are the binding ones there.
  - Reported, not gated.
- **H4 (integrity, required):** all of the following hold.
  - Every integrity probe is refused with its expected reason code. The 7 probes run on each of
    the first 20 evaluation documents, 140 refusals in all, before any arm is scored:
    - tampered byte → `TENSOR_DIGEST_MISMATCH`
    - grant for another capsule → `CAPSULE_NOT_GRANTED`
    - expired grant → `GRANT_EXPIRED`
    - wrong principal → `PRINCIPAL_MISMATCH`
    - stale fence → `STALE_FENCE`
    - wrong recipient model → `RECIPIENT_MISMATCH`
    - foreign (untrained) codebook → `SOURCE_MISMATCH`
  - Every scored capsule use is INJECTED: 1,600 items × 4 capsule arms = 6,400 receipts. Each
    passes `verify_latent_use` in the receiver and again in the report phase from the saved
    grants and manifests, and every capsule-arm score row points at one of them.
  - All 200 anchors verify against the paired capsules' `anchor_digest`, counted against the
    dataset, before any arm is scored.
  - The base digest is unchanged by training and identical in the train, encode and evaluate
    processes, which all used the same adapter id.
- **Verdicts:**
  - **G-W7 evidence: SUPPORTED** if H1 and H4 hold.
  - **SUPPORTED, including the unseen task family** if H2 also holds.
  - Otherwise **NOT SUPPORTED**. A null result is reported as such, with the same tables.
  - The lane proposes the verdict; the orchestrator owns the gate.
- No other comparison can promote the result. `text_context` is a ceiling, and a capsule arm
  beating `no_prefix` alone is not enough.

## 7. Training budget (fixed before the run)

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0, gradient clip 1.0 (`training.GRAD_CLIP`) |
| Batch | 4 examples, sampled with replacement, generator seed 0 |
| Steps | 3,000 (12,000 examples ≈ 1.5 passes over the 8,000 training items) |
| Learning rate | 1e-3, constant |
| Projector init seed | 0 |
| Scoring batch | 16 items |

These are the `CONFIG` defaults in `bench/latent_transfer/run.py` at the commit that adds this file.
Batch 4 is a memory constraint: batch 16 and batch 8 runs were killed by out-of-memory errors on
this shared host.

These are not tuned on evaluation data. They were set from the training-split calibration pilot
(§9). The run is single-seed: only one training seed fits the CPU budget, which is a limitation
stated in the report.

## 8. Cost accounting (reported, part of the gate)

- **Wall time** per phase (prepare, train incl. the sender cache, encode, evaluate; the report
  phase's own seconds are not counted).
- **Device:** CPU model, logical cores, torch threads, torch/transformers versions, no CUDA. Other
  sessions' processes share this CPU and RAM, so timings are upper bounds.
- **Peak resident memory** per phase.
- **FLOP estimates:**
  - forward = 2·N_blocks·tokens + 2·N_head·head_tokens;
  - training = sender cache (one forward over every training document, all 24 blocks, no head) +
    2 × forward over receiver tokens with the head applied to answer tokens only (activation
    gradients only; frozen weights skip the weight-gradient matmuls);
  - token counts are unpadded and attention-score FLOPs are ignored, so these are lower-bound
    estimates.
- **Totals:** total FLOPs including training; training FLOPs amortised per scored capsule use
  (denominator 6,400: every capsule-arm use, controls included; per paired use is 4× that);
  capsule bytes per handoff; mean receiver tokens per item for `paired` vs `text_context`.

## 9. Calibration pilot (disclosed, not evidence)

**Setup:** dataset seed 999, 500 training documents for fitting and 100 held-out training
documents, training templates only, K = 8, layer 16.

Held-out accuracy is first-token accuracy over the 8 candidates on the 100 held-out training
documents, with training question templates. Chance is 0.125.

| Run | Training loss (first → last logged) | Held-out accuracy: capsule / no prefix / text in prompt | Seconds per step |
|---|---|---|---|
| 60 steps, batch 16, lr 1e-3 | 3.59 → 2.53 | 0.115 / 0.14 / 1.00 | 3.35 |
| 300 steps, batch 4, lr 1e-3 | 4.08 → 2.44 | 0.13 / not scored (killed by OOM) / not scored | 1.09 |

- **Failed attempts:** a 300-step batch-16 run, a 400-step batch-8 run at lr 3e-3 and a 300-step
  batch-8 run at lr 3e-3 were all killed by out-of-memory errors or a memory-mapped weight load
  crash. The batch-8 lr 3e-3 run logged a step-10 loss of 4.35 before it died.
- **Reading:** after about 1,200 examples the capsule carries no measurable content on held-out
  training documents (0.13 against 0.125 chance). The falling loss reflects the answer format and
  answer-word priors. The confirmatory budget is 10× the examples of the 300-step pilot. That is
  as much as the CPU and memory budget allows, and it may still not be enough; a null result is
  an expected outcome.
- **Not evidence:** none of these numbers is G-W7 evidence. They are reported here and in
  `REPORT.md` only as calibration.

## 10. Process and deviations

- The confirmatory run is `python -m bench.latent_transfer.run all` with the settings above, in
  separate processes:
  - the sender process reads documents;
  - the receiver process opens only `questions.json`, `anchors.json`, `capsule_index.json`,
    `capsules/` and `projector.pt` (the last only so its adapter id matches the trained pair; it
    never runs the projector). The anchors hold the exact source records, facts included, because
    the `text_context` ceiling needs the document; the receiver never sees the answer key join,
    the shuffle map, or the blank and mismatched documents, and computes no correctness;
  - the report process joins gold and takes its configuration from the phase records, which must
    all agree.
- If a phase crashes before writing its phase record (this host has killed processes for lack of
  memory), that phase alone is rerun from its saved inputs with an unchanged configuration:
  - `python -m bench.latent_transfer.run <phase> --run-id <same id>`;
  - training restarts from step 0;
  - the crash and rerun are recorded in `REPORT.md`.
- A crash after any score row exists restarts the full run under a new run id (`all` refuses a
  non-empty run directory), and that is recorded too.
- Any change after the first result file exists is listed under "Deviations" in `REPORT.md`, with
  its reason. Hyperparameters are not re-run on evaluation data.

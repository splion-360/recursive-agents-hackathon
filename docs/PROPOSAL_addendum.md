# ChartQA Memory Distillation: Implementation Addendum

## Status and authority

This addendum expands the methodology in `PROPOSAL_revised.html`. It does not replace that proposal, `CONTEXT.md`, or accepted ADRs. If wording conflicts, those sources are authoritative. The contracts below are design commitments, not claims that providers or experiments have already been verified.

## Research question

Can a fixed, smaller multimodal Student improve on unseen ChartQA tasks by retrieving answer-free Behavior Cards distilled from reviewed Collection trajectories, after the online Reviewer is removed and without changing model weights?

The result must distinguish procedural transfer from extra context, example retrieval, answer leakage, human curation, and repeated final-set tuning.

## Canonical artifacts and roles

- **Minimal Harness (H):** the fixed Student prompt, `inspect_image`, `crop`, `run_python`, and `final_answer` action schema, tools, decoding, and budgets used across conditions.
- **Optimized Reference Harness (H-star):** a calibration-only ChartQA harness with empirically accepted tools, checks, skills, middleware, or guards. It never runs measured Collection, Promotion, or Final Held-Out tasks.
- **Kappa Guidance:** the frozen, answer-free private Reviewer artifact adapted from H-star into action recipes, review checks, guidance, and failure patterns executable through H. Kappa is not a model.
- **Student:** the smallest verified Crusoe multimodal model with measurable ChartQA headroom.
- **Reviewer:** the required Crusoe-hosted Nemotron multimodal model. It consumes Kappa Guidance and returns PASS or the smallest valid REPLACE action during calibration, measured Collection, and C1 only.
- **Consolidator:** a schema-reliable Crusoe text model that proposes candidate card deltas from a complete snapshot of eligible Reviewed Episodes. Deterministic validators remain authoritative.
- **H-star Analysis Model:** an optional calibration-only model that may propose one bounded H-star revision. Its exact endpoint is discovered at the provider gate.
- **Promotion Gate:** deterministic code, never an LLM or human, that accepts or rejects a complete Candidate Version.

Exact model IDs, revisions, modalities, quotas, and structured-output behavior are configuration discovered by live contract tests. No named but unverified endpoint is part of the frozen architecture.

## Selective Harness-Zero reuse

The reviewed upstream is [metaevo-ai/harness-zero at `a4a6223241479c750fdab827c3a449bb1d41b99a`](https://github.com/metaevo-ai/harness-zero/tree/a4a6223241479c750fdab827c3a449bb1d41b99a), licensed Apache-2.0.

The project may port or adapt, with attribution and any required notices:

- the proposal-to-review response boundary;
- PASS/REPLACE review contracts;
- H-star-to-Kappa component-mapping ideas;
- trajectory and intervention audit concepts.

The project does not adopt Harness-Zero as its runtime. It excludes Harbor, DeepAgents, SFT, benchmark banks, and the upstream single-`execute` action assumption. ChartQA uses this repository's typed multimodal action schema, OpenShell boundary, MemMachine version lifecycle, and controlled C0-C3 evaluation. Any copied code must be identified during implementation; conceptual inspiration alone does not imply a code dependency.

Relevant pinned references:

- [Evolution workflow](https://github.com/metaevo-ai/harness-zero/blob/a4a6223241479c750fdab827c3a449bb1d41b99a/.agents/skills/student-harness-evolve/SKILL.md)
- [H-star-to-Kappa adaptation](https://github.com/metaevo-ai/harness-zero/blob/a4a6223241479c750fdab827c3a449bb1d41b99a/.agents/skills/teacher-harness-adapt/SKILL.md)
- [Student loop](https://github.com/metaevo-ai/harness-zero/blob/a4a6223241479c750fdab827c3a449bb1d41b99a/src/harness_zero/student.py)
- [Review schema](https://github.com/metaevo-ai/harness-zero/blob/a4a6223241479c750fdab827c3a449bb1d41b99a/src/harness_zero/review.py)
- [Teacher implementation](https://github.com/metaevo-ai/harness-zero/blob/a4a6223241479c750fdab827c3a449bb1d41b99a/src/harness_zero/teacher.py)

## Dataset isolation

All manifests are deterministic, hashed, disjoint by task and image, and fixed before measured work.

| Manifest | Size | Permitted influence | Memory writes |
| --- | ---: | --- | --- |
| Harness Calibration: discovery | 8 | One bounded H-star proposal | None |
| Harness Calibration: regression | 8 | Accept or reject H-star, then validate the adapted Kappa without another revision | None |
| Collection | 40 | Reviewed Episodes and Candidate Version evidence | Reviewed Episodes only |
| Promotion | 30 | Candidate-versus-Accepted gate | None |
| Final Held-Out | 60 | One C0-C3 report after freeze | None |

The Student, Reviewer, retriever, sandbox, and model adapters receive answer-free `TaskInput` values. Gold answers and official ChartQA normalization remain in verifier-only `ScoringReference` values. Only the final-evaluation composition root may load the Final Held-Out manifest.

## Phase A: derive and freeze H-star and Kappa Guidance

1. Run H on the eight discovery items and record failures without creating operational memory.
2. The optional H-star Analysis Model may propose one bounded revision focused on repeated failure families.
3. Evaluate H and the proposed H-star on the separate eight-item regression subset. Accept only measured improvement without leakage or invalid actions.
4. Adapt every accepted H-star component explicitly:
   - tools become H-executable action recipes;
   - middleware becomes review-time evidence checks;
   - skills become review guidance;
   - failure memory becomes generalized failure patterns.
5. Mark every component `transform`, `split`, `merge`, or `drop`. Kappa may not depend on an H-star-only action.
6. Validate the adapted Kappa on the same regression subset without permitting another H-star or Kappa revision. Reject content containing calibration IDs, copied questions, chart values, answers, private paths, or actions outside H.
7. Canonically serialize and hash H-star and Kappa. Freeze both before measured Collection.

If no proposed revision passes, freeze the minimal validated review rubric and narrow the claim. Calibration never produces Behavior Cards.

## Phase B: collect reviewed trajectories

For each Collection task, H retrieves once from the current Accepted Version, then the fixed Student proposes actions. The Reviewer sees the image, task, Student-visible state, proposed action, and private Kappa Guidance before execution.

- **PASS:** execute the proposed action unchanged.
- **REPLACE:** execute one schema-valid Student-native action that is the smallest defensible correction.

Only the accepted action and resulting observation enter continuing Student context. The original proposal, intervention decision, private reason, and Kappa component references are audit evidence, never Student context or public replay content.

A completed task produces one idempotent Reviewed Episode containing task and run identity, proposed and accepted actions, public observations, verifier outcome, intervention metadata, model and configuration identity, Accepted Version identity, and sanitized provenance. Infrastructure failures and invalid or direct-answer replacements remain observable but are ineligible for consolidation.

## Phase C: build one immutable Candidate Version

After all 40 Collection tasks finish, freeze the eligible Reviewed Episode snapshot. The Consolidator receives that complete snapshot and the current Accepted Version, then proposes deltas:

- **CREATE:** add a new generalized Behavior Card supported by grounded evidence.
- **REINFORCE:** add evidence without changing the canonical procedure.
- **REFINE:** replace a broad or incomplete card with a new immutable definition that preserves predecessor provenance.
- **RETIRE:** omit a harmful or superseded card from the candidate manifest without deleting its history.

These are candidate-delta operations, not in-place edits to active memory. Deterministic validation rejects the entire batch for invalid schemas, missing evidence, contradiction, task or image IDs, copied questions, answer options, final answers, raw chart values, or evaluation provenance. A passing batch becomes one complete, immutable Candidate Version with canonical hashes.

No candidate card is retrievable during the Collection pass that created it.

## Phase D: promote or reject atomically

Run the Accepted Version and Candidate Version on the same 30 Promotion tasks with the Student, H, prompts, decoding, tools, budgets, seeds, Reviewer-off policy, and writes-off policy held fixed. Each run receives an explicitly pinned memory view.

The deterministic Promotion Gate applies the predeclared capability, Negative Transfer, latency, token, validity, and contamination thresholds. A passing candidate advances the Accepted Version pointer atomically. A rejected candidate remains immutable for audit and the pointer stays unchanged. Humans may repair infrastructure and rerun only under a predeclared retry policy; they may not edit cards, thresholds, or results.

**Current live gate:** the managed MemMachine public API verified on October 17, 2026 does not expose a conditional update primitive. The adapter stores and retrieves the immutable bootstrap pointer but refuses promotion updates rather than approximating compare-and-set with a non-atomic append. Measured promotion remains blocked until an approved atomic MemMachine mechanism is available.

Accepted v000 is the immutable empty initial version, so a rejected first candidate is representable without fabricating progress.

## Phase E: one Final Held-Out C0-C3 run

After the final Accepted Version is frozen, execute every available condition on all 60 Final Held-Out tasks with identical shared budgets and counterbalanced ordering:

| Condition | Reviewer | Retrieved memory | Purpose |
| --- | --- | --- | --- |
| C0 | Off | None | Minimal H baseline |
| C1 | On with frozen Kappa | None | Online-review ceiling and cost |
| C2 | Off | Relevant cards from the Accepted Version | Primary memory-distilled condition |
| C3 | Off | Trigger-irrelevant cards from the Accepted Version | Extra-context control |

Before Final Held-Out access, freeze the accepted-card donor pool, trigger-mismatch rule, exact card-count rule, 10% token tolerance, and seed `20261016`. After the final-evaluation composition root loads answer-free `TaskInput` values, it materializes C2 retrieval and deterministic C3 assignments before any condition executes or any `ScoringReference` is loaded. If no honest match exists for a task, report C3 unavailable for that task and disclose the eligible paired population; never invent or duplicate a control card merely to fill the cell.

If Accepted v000 remains active, C2 supplies no cards and collapses to the no-memory behavior. C3 is unavailable. Report this as a degenerate negative-result case; it cannot support a relevance claim.

Final outcomes never change prompts, models, thresholds, retrieval, Kappa, cards, or promotion. No selective reruns follow inspection.

## Storage and evidence boundaries

Managed MemMachine is the sole Operational Memory. It stores Reviewed Episodes, immutable card definitions, version manifests, and the Accepted Version pointer behind the repository's `OperationalMemory` interface. Student retrieval is pinned to one version and returns accepted, answer-free Behavior Cards only. Raw Reviewed Episode retrieval is forbidden in the primary condition.

Local JSONL is an append-only Experiment Trace for metrics, debugging, sanitized replay, and labeled offline fixtures. Runtime agents cannot query it. It is not a MemMachine mirror, fallback, training source, or improvement memory.

Every measured run records code revision, dirty-state disclosure, manifest hashes, model identities, Kappa hash, Accepted Version, budgets, seeds, retrieval decisions, request identifiers, latency, tokens, retries, and sanitized outcome provenance.

## Offline-core exception and live gate

Before provider access, implementation may cover typed contracts, pure scoring and lifecycle logic, deterministic fake adapters, synthetic lifecycle tests, the in-memory Operational Memory contract, sanitized replay fixtures, and the replay visualizer.

It may not construct production composition, perform managed writes or paid calls, execute measured Collection or Promotion, or open the Final Held-Out manifest. Those operations require direct evidence that all live seams pass:

- one valid Crusoe Student multimodal action;
- one valid Nemotron PASS or REPLACE decision;
- one schema-valid Consolidator result;
- MemMachine round-trip, pagination, pinned retrieval, idempotency, and pointer semantics;
- NemoClaw-managed OpenShell network denial, filesystem scope, timeout, and resource policy.

Credentials or configuration alone do not satisfy the gate, and no failed production adapter may silently fall back to a local implementation.

## Required evidence

Report paired sample size, ChartQA relaxed accuracy, capability delta, Positive and Negative Transfer, discordant pairs, exact McNemar result, seeded paired-bootstrap interval, Reviewer gap recovery, intervention rate, p50/p95 latency, input/output tokens, calls, optional measured cost, trigger precision, no-retrieval rate, compression ratio, and duplicate/contradiction rate. Every metric must state its denominator and missing-data behavior.

The demo must trace one failure through Reviewer correction, Reviewed Episode persistence, candidate card derivation, deterministic screening, promotion outcome, Reviewer-free retrieval, and C0-C3 evidence. LIVE and REPLAY data must be visibly distinguished.

## Non-goals

- No SFT, LoRA, reinforcement learning, or model-weight update.
- No Harbor, DeepAgents, Harness-Zero benchmark banks, or single-`execute` runtime.
- No raw-episode retrieval in the primary condition.
- No student-operated memory search in the MVP.
- No per-task activation of newly consolidated cards within a Lifecycle Pass.
- No web search in the core ChartQA runtime.
- No human curation of cards, runs, promotion outcomes, or final claims.

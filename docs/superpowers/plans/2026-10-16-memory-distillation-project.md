# Memory-Distilled Agent Team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one complete ChartQA Version N → N+1 lifecycle that derives a frozen ChartQA Kappa, collects reviewed episodes, proposes and promotes immutable Behavior Card versions through MemMachine, runs C0–C3, and presents the evidence in a React visualizer.

**Architecture:** Extend the existing FastAPI/Vite modular monolith with deep Python modules for benchmark access, task execution, operational memory, version lifecycle, and evaluation. All external systems sit behind typed protocols with deterministic test adapters; production orchestration remains disabled until Crusoe, MemMachine, and NemoClaw/OpenShell pass explicit smoke gates. MemMachine is the sole operational memory, while one discriminated `RunEvent` stream serves live SSE, append-only JSONL telemetry, replay, and metrics.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, pytest, mypy, Ruff, React 19, TypeScript strict mode, Vite 7, Vitest/Testing Library, Crusoe OpenAI-compatible inference, NVIDIA Nemotron Reviewer, managed MemMachine, NemoClaw-managed OpenShell, ChartQA.

**Spec:** `docs/PROPOSAL_revised.html`, reconciled with `docs/PROPOSAL_addendum.md`, `CONTEXT.md`, and `docs/adr/` in Task 1.

## Global Constraints

- Do not call paid inference, write managed memory, deploy, or run destructive operations without confirming target, quota, estimated cost, and write scope.
- Offline contracts, pure logic, deterministic adapters, and synthetic orchestration may be built before live access. Production composition roots and measured runs remain disabled until every provider smoke gate passes; Task 1 records this explicit amendment to the earlier phase-zero stop condition.
- ChartQA is required; MathVista, raw-episode retrieval, repeated robustness runs, and weight updates are out of the critical path.
- Student-generated Python executes only through the NemoClaw-managed OpenShell adapter with network denied, scoped filesystem access, a five-second timeout, and bounded resources.
- The primary Student retrieval path accepts only Behavior Cards pinned by the Accepted Version; it never reads Reviewed Episodes or JSONL.
- Collection may write Reviewed Episodes. Harness Calibration, Promotion, and Final Held-Out runs never create improvement memory.
- H-star and Kappa are derived on a disjoint Harness Calibration Set, frozen, hashed, and unchanged throughout measured Collection.
- Candidate cards and manifests are immutable after freeze. The Promotion Gate is deterministic; only the Accepted Version pointer may advance.
- C0–C3 use the same frozen Student, prompts, actions, tools, budgets, decoding, and task manifest except for the stated Reviewer/memory treatment.
- Final Held-Out data is import-inaccessible to calibration, collection, consolidation, retrieval calibration, and promotion code.
- Local JSONL is append-only telemetry and labeled replay only. Runtime and improvement agents receive a write-only event sink.
- Preserve user-owned dirty-worktree changes. Never commit `.env`, credentials, datasets, benchmark images, provider payloads, run traces, or generated evaluation outputs.
- Every implementation task follows red → green → refactor, ends with focused verification, and uses the repository’s uppercase commit-prefix convention only when the user authorizes commits.

## Review Focus

1. **Final-set contamination:** architectural tests must prove that non-final composition roots cannot construct or import final-manifest access.
2. **Memory type confusion:** retrieval tests must fail closed when MemMachine returns an episode, candidate, rejected card, or card outside the pinned accepted manifest.
3. **Unreviewed execution:** collection and C1 must terminate explicitly when Reviewer or replacement budgets fail; no proposal may silently execute without review.
4. **Partial external writes:** task retries must be idempotent by stable run/episode/version IDs and must never duplicate episodes or advance the accepted pointer twice.
5. **Misleading evidence:** replay must be visibly labeled, direct-answer Reviewer replacements must not support procedural cards, and provider/config presence must never count as a successful integration.

---

## Delivery map

| Wave | Components | Can finish without Crusoe credits? | Exit condition |
| --- | --- | ---: | --- |
| A. Contract and deterministic core | Tasks 1–12 | Yes, after Task 1 records the offline-core exception | Full synthetic lifecycle passes offline |
| B. Product surfaces | Tasks 13–15 | Yes | CLI/API/UI replay the same synthetic run |
| C. Provider gate | Task 16 | No | Every real external seam passes one recorded smoke test |
| D. Measured experiment | Tasks 17–18 | No | Frozen Kappa, collection, promotion, and C0–C3 reports complete |
| E. Submission | Task 19 | Mostly | Rendered demo, documentation, verification, and claims agree |

Do not start Wave D merely because credentials exist. Task 16 must record successful behavior at every external seam first.

## Time budget and cut line

| Work | Target hours | Required? |
| --- | ---: | ---: |
| Contract, settings, schemas, scorer, manifests, telemetry | 6 | Yes |
| Deterministic inference/sandbox/memory seams | 4 | Yes |
| Task execution and frozen review-policy machinery | 7 | Yes |
| Consolidation, promotion, metrics, offline lifecycle | 7 | Yes |
| Read-only replay API and minimal visualizer | 4 | Yes |
| Live provider gate and Kappa calibration | 3 | Yes |
| Measured collection, promotion, and C0–C3 | 6 | Yes; wall time may vary |
| Packaging, rendered verification, and submission assets | 3 | Yes |

The 40-hour target is credible only for the narrow implementation described here and depends on prompt provider access plus low failure/retry rates. Cut in this order: live SSE, UI animation/polish, generalized multi-cycle/resume features beyond required idempotency, optional robustness repeats, and extra H-star rounds. Never cut final-set isolation, answer exclusion, OpenShell enforcement, MemMachine record-type filtering, immutable candidate promotion, C0–C3, or truthful replay labeling.

### Task 1: Reconcile the experiment contract

**Files:**
- Modify: `CONTEXT.md`
- Modify: `docs/PROPOSAL_revised.html`
- Modify: `docs/PROPOSAL_addendum.md`
- Modify: `docs/assets/memory-distillation-system-design.drawio`
- Modify: `docs/assets/memory-distillation-system-design.svg`
- Create: `docs/adr/0009-freeze-chartqa-kappa-before-collection.md`
- Create: `docs/adr/0010-use-one-batched-lifecycle-for-the-primary-experiment.md`
- Create: `docs/adr/0011-allow-offline-core-before-live-provider-gates.md`

**Interfaces:**
- Consumes: Existing proposal, addendum, ADRs, and generated lifecycle diagram.
- Produces: One authoritative vocabulary and experiment contract for every later task.

- [ ] Add canonical definitions for `Minimal Harness (H)`, `Optimized Reference Harness (H-star)`, `Kappa Guidance`, `Harness Calibration Set`, and `Lifecycle Pass` without replacing Student/Reviewer terminology.
- [ ] Record that H-star/Kappa use a disjoint calibration manifest, never generate Behavior Cards, and remain frozen during measured Collection.
- [ ] Resolve model roles explicitly: the smallest verified Crusoe multimodal model is the Student, Nemotron is the required Reviewer under the current contract, a schema-reliable text model is the Consolidator, and the optional H-star analysis model is configuration discovered at the provider gate.
- [ ] Record selective Harness-Zero reuse: pin the reviewed upstream revision, port review contracts/adaptation ideas with Apache-2.0 attribution, and exclude its Harbor, DeepAgents, SFT, benchmark-bank, and single-`execute` runtime assumptions.
- [ ] Amend the phase-zero stop condition: provider-independent contracts, pure logic, fake adapters, synthetic lifecycle tests, and replay UI may be implemented offline; no production composition, managed writes, paid calls, or measured orchestration may execute before all live seams pass.
- [ ] Reconcile CREATE/REINFORCE/REFINE/RETIRE as candidate-delta operations under immutable Candidate/Accepted Versions; restore C0–C3 and the deterministic gate everywhere.
- [ ] Freeze C3 semantics: predeclare donor pool, trigger-based irrelevance rule, card-count/token matching tolerance, assignment seed, unavailable-control reporting, and the degenerate accepted-v000 interpretation.
- [ ] Update the editable architecture diagram and SVG so they match the accepted lifecycle; keep the generated PNG as a supplementary illustration.
- [ ] Verify terminology and links with `rg -n "Kappa|H-star|Candidate Version|C0|C1|C2|C3" CONTEXT.md docs/PROPOSAL_revised.html docs/adr docs/assets`.
- [ ] If commits are authorized, commit as `[DOCS]: reconcile the ChartQA review lifecycle`.

### Task 2: Establish configuration, commands, and artifact hygiene

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Create: `backend/.env.example`
- Modify: `.gitignore`
- Create: `backend/src/memory_distillation/settings.py`
- Create: `backend/tests/test_settings.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1’s fixed roles and runtime limits.
- Produces: `Settings.load() -> Settings`, documented offline/live commands, pytest markers, and ignored data/artifact paths.

- [ ] Write tests asserting required live credentials are validated only in live mode, secrets are redacted, and offline mode starts without them.
- [ ] Add only dependencies earned by Tasks 2–5 (including Pillow for image fixtures); defer vendor SDK dependencies until Task 16 confirms the actual APIs.
- [ ] Implement typed settings for model IDs, paths, budgets, timeouts, retrieval defaults, and explicit `offline`/`live` execution mode.
- [ ] Add `integration`, `provider`, and `slow` pytest markers excluded from the default fast suite.
- [ ] Add ignored paths for `.data/`, `.artifacts/`, JSONL traces, benchmark assets, and generated reports without overwriting unrelated `.gitignore` changes.
- [ ] Document exact setup, test, lint, typecheck, backend, and frontend commands in `README.md`.
- [ ] Run `uv run pytest tests/test_settings.py -q`, `uv run ruff check .`, and `uv run mypy` from `backend/`.
- [ ] If commits are authorized, commit as `[CHORE]: configure deterministic experiment environments`.

### Task 3: Define the shared domain and event contracts

**Files:**
- Create: `backend/src/memory_distillation/contracts/__init__.py`
- Create: `backend/src/memory_distillation/contracts/models.py`
- Create: `backend/src/memory_distillation/contracts/events.py`
- Create: `backend/tests/contracts/test_models.py`
- Create: `backend/tests/contracts/test_events.py`

**Interfaces:**
- Produces: answer-free `TaskInput`, verifier-only `ScoringReference`, discriminated `StudentAction`, `ReviewPolicy`, `ReviewDecision`, `TaskRun`, `ReviewedEpisode`, `BehaviorCardDefinition`, `CardEvidence`, `VersionManifest`, `PromotionComparison`, `PromotionDecision`, durable `RunState`, and discriminated `RunEvent`.

- [ ] Write serialization and rejection tests for all action variants, PASS/REPLACE invariants, termination reasons, stable IDs, and provenance fields.
- [ ] Test that inference requests, retrieval queries, sandbox mounts, replay payloads, and Student/Reviewer contexts can serialize `TaskInput` but cannot contain or import `ScoringReference`.
- [ ] Write event round-trip tests proving one schema supports SSE, JSONL, replay, and metrics without provider payloads or secrets.
- [ ] Implement strict Pydantic models with forbidden extra fields and explicit schema versions.
- [ ] Define durable operation IDs and run states for prepared, submitted, uncertain, completed, persisted, and failed operations; later adapters must reconcile uncertain outcomes rather than blindly retry.
- [ ] Define canonical JSON and SHA-256 helpers used by card and manifest hashing; prove key order does not affect hashes.
- [ ] Run `uv run pytest tests/contracts -q`, Ruff, and mypy.
- [ ] If commits are authorized, commit as `[FEAT]: define experiment domain contracts`.

### Task 4: Implement ChartQA access, scoring, and manifest isolation

**Files:**
- Create: `backend/src/memory_distillation/benchmark/__init__.py`
- Create: `backend/src/memory_distillation/benchmark/chartqa.py`
- Create: `backend/src/memory_distillation/benchmark/manifests.py`
- Create: `backend/src/memory_distillation/benchmark/development.py`
- Create: `backend/src/memory_distillation/benchmark/final.py`
- Create: `backend/tests/benchmark/test_chartqa.py`
- Create: `backend/tests/benchmark/test_manifests.py`
- Create: `backend/tests/architecture/test_final_set_isolation.py`
- Create: `backend/tests/fixtures/chartqa/`
- Create when generated: `backend/experiment_manifests/`

**Interfaces:**
- Produces: `DevelopmentBenchmark.load_calibration|load_collection|load_promotion`, `FinalBenchmark.load_final`, `score_chartqa(prediction, reference) -> Score`, and immutable hashed manifests.

- [ ] Port and test official ChartQA normalization/relaxed scoring against known synthetic cases before downloading data.
- [ ] Test deterministic stratified assignment, exact 16/40/30/60 default sizes, zero task overlap, zero image overlap, and stable hashes. The 16-item calibration set is split 8 discovery/8 regression.
- [ ] Implement separate development and final access modules; architectural tests must reject final imports outside the final-evaluation composition root.
- [ ] Mount or expose only the current answer-free task asset to execution code; tests must prove sibling tasks, reference labels, and the full dataset directory are inaccessible.
- [ ] Add a deterministic manifest-generation command that reads a local ignored ChartQA checkout and writes small JSON manifests containing IDs/hashes, never images or full dataset rows.
- [ ] Run benchmark tests offline; then generate and inspect manifests when the dataset is available.
- [ ] If commits are authorized, commit as `[FEAT]: freeze isolated ChartQA experiment manifests`.

### Task 5: Implement append-only telemetry and sanitized replay

**Files:**
- Create: `backend/src/memory_distillation/telemetry/__init__.py`
- Create: `backend/src/memory_distillation/telemetry/sink.py`
- Create: `backend/src/memory_distillation/telemetry/jsonl.py`
- Create: `backend/src/memory_distillation/telemetry/replay.py`
- Create: `backend/tests/telemetry/test_jsonl.py`
- Create: `backend/tests/telemetry/test_replay.py`

**Interfaces:**
- Produces: write-only `EventSink.emit(event: RunEvent) -> None`, `JsonlEventSink`, `InMemoryEventSink`, and visualizer-only `ReplayReader`.

- [ ] Test append ordering, crash-tolerant complete-line reads, secret redaction, oversized payload rejection, and refusal to deserialize provider-private reasoning.
- [ ] Test that runtime protocols expose only `emit`; no task, Reviewer, Consolidator, or memory module can import `ReplayReader`.
- [ ] Implement sanitized replay projection with explicit `LIVE`/`REPLAY` mode metadata.
- [ ] Run telemetry and architectural tests, Ruff, and mypy.
- [ ] If commits are authorized, commit as `[FEAT]: add non-serving experiment telemetry`.

### Task 6: Define inference and structured-output adapters

**Files:**
- Create: `backend/src/memory_distillation/inference/__init__.py`
- Create: `backend/src/memory_distillation/inference/protocol.py`
- Create: `backend/src/memory_distillation/inference/fake.py`
- Create: `backend/src/memory_distillation/inference/crusoe.py`
- Create: `backend/tests/inference/test_fake.py`
- Create: `backend/tests/inference/test_crusoe_contract.py`

**Interfaces:**
- Produces: `InferenceClient.generate(request: InferenceRequest[T]) -> InferenceResult[T]` with model identity, tokens, latency, retry count, request ID, and typed output.

- [ ] Test deterministic scripted responses, image-part serialization, one structured-output repair attempt, bounded retry classification, timeout propagation, and secret-free errors.
- [ ] Implement the fake adapter fully and the Crusoe OpenAI-compatible adapter against a transport stub; do not assert live model compatibility yet.
- [ ] Ensure model aliases resolve once at composition time and every result records the exact provider model ID.
- [ ] Enforce a request-admission budget using configured maximum calls/tokens/cost; quota exhaustion blocks new requests with a recorded terminal reason.
- [ ] Distinguish pre-submit failure, provider-rejected request, uncertain submitted outcome, and completed response so retries cannot silently double-charge or duplicate work.
- [ ] Run inference contract tests offline.
- [ ] If commits are authorized, commit as `[FEAT]: isolate typed inference behind a provider seam`.

### Task 7: Define the OpenShell execution boundary

**Files:**
- Create: `backend/src/memory_distillation/sandbox/__init__.py`
- Create: `backend/src/memory_distillation/sandbox/protocol.py`
- Create: `backend/src/memory_distillation/sandbox/fake.py`
- Create: `backend/src/memory_distillation/sandbox/openshell.py`
- Create: `backend/tests/sandbox/test_policy.py`
- Create: `backend/tests/sandbox/test_openshell_contract.py`

**Interfaces:**
- Produces: `Sandbox.execute(request: SandboxRequest) -> SandboxResult` and immutable `SandboxPolicy`.

- [ ] Test path traversal, network attempts, timeout, output truncation, resource-limit reporting, and the prohibition on any local-Python fallback.
- [ ] Test task-to-task isolation: no inherited credentials, sibling workspace, reference labels, full dataset mount, or prior task state is visible; also bound image decoding dimensions and crop coordinates.
- [ ] Implement deterministic fake execution for fixtures and an OpenShell adapter behind an injectable command/API transport.
- [ ] Record policy version, exit status, timing, stdout/stderr digests, and violations without storing secrets or uncontrolled output.
- [ ] Run sandbox tests offline.
- [ ] If commits are authorized, commit as `[FEAT]: enforce the generated-code execution boundary`.

### Task 8: Implement Operational Memory with a deterministic adapter

**Files:**
- Create: `backend/src/memory_distillation/operational_memory/__init__.py`
- Create: `backend/src/memory_distillation/operational_memory/protocol.py`
- Create: `backend/src/memory_distillation/operational_memory/in_memory.py`
- Create: `backend/src/memory_distillation/operational_memory/retrieval.py`
- Create: `backend/tests/operational_memory/test_records.py`
- Create: `backend/tests/operational_memory/test_retrieval.py`

**Interfaces:**
- Produces: `OperationalMemory.append_episode`, `get_episode`, `snapshot_episodes`, `append_card_definition`, `get_card_definition`, `append_evidence`, `get_evidence_snapshot`, `freeze_candidate`, `get_manifest`, `get_accepted_version`, and `compare_and_set_accepted_version`; serving and promotion receive different typed read capabilities.

- [ ] Test append-only/idempotent writes, immutable definitions/manifests, accepted-v000 empty bootstrap, and atomic compare-and-set pointer behavior.
- [ ] Define `EpisodeSnapshot` with ordered episode IDs, content hashes, provenance checks, pagination/completeness evidence, and a stable snapshot hash; consolidation reads only this capability, never JSONL.
- [ ] Define `AcceptedMemoryView` for collection/final serving and `PromotionMemoryView` for an explicitly pinned frozen Candidate manifest. Only the promotion composition root may construct the latter.
- [ ] Test retrieval filtering by the view’s exact pinned manifest, record type, benchmark, task type, operation, threshold, `top_k=3`, and token budget.
- [ ] Test fail-closed behavior for raw episodes, cards outside the pinned view, stale manifests, duplicate IDs, incomplete snapshots, and malformed provider records. Collection/final code must be unable to request Candidate cards.
- [ ] Implement the in-memory adapter as the authoritative offline behavior contract.
- [ ] Run operational-memory tests, Ruff, and mypy.
- [ ] If commits are authorized, commit as `[FEAT]: define versioned operational memory`.

### Task 9: Implement TaskExecutor in three reviewable slices

**Files:**
- Create: `backend/src/memory_distillation/task_execution/__init__.py`
- Create: `backend/src/memory_distillation/task_execution/executor.py`
- Create: `backend/src/memory_distillation/task_execution/policy.py`
- Create: `backend/src/memory_distillation/task_execution/images.py`
- Create: `backend/tests/task_execution/test_executor.py`
- Create: `backend/tests/task_execution/test_reviewer_boundary.py`

**Interfaces:**
- Produces: `TaskExecutor.run(task: Task, policy: ExecutionPolicy) -> TaskRun`; consumes inference, sandbox, operational-memory, benchmark-verifier, and write-only event-sink protocols.

- [ ] **Slice A — minimal executor:** test and implement all four action types, eight-action/two-Python budgets, deterministic observations, bounded image handling, official termination reasons, and Sandbox-only generated Python. Commit separately if authorized.
- [ ] **Slice B — private review boundary:** test and implement PASS/REPLACE so only the accepted action enters continuing Student context; cover replacement validation, direct-answer labeling, Reviewer failure, uncertain provider outcome, and budget exhaustion. Commit separately if authorized.
- [ ] **Slice C — memory/write policy:** test and implement retrieval-once-at-start through an injected `AcceptedMemoryView` or `PromotionMemoryView`; calibration has no memory writes, collection writes one idempotent episode, promotion/final have writes off, and C2/C3 have Reviewer off. Commit separately if authorized.
- [ ] Run task-execution tests and the complete offline suite after every slice.

### Task 10: Build and freeze ChartQA H-star/Kappa artifacts

**Files:**
- Create: `backend/src/memory_distillation/review_policy/__init__.py`
- Create: `backend/src/memory_distillation/review_policy/models.py`
- Create: `backend/src/memory_distillation/review_policy/evolve.py`
- Create: `backend/src/memory_distillation/review_policy/adapt.py`
- Create: `backend/src/memory_distillation/review_policy/validate.py`
- Create: `backend/review_policy/chartqa_hstar_v1/`
- Create: `backend/review_policy/chartqa_kappa_v1/`
- Create: `backend/tests/review_policy/test_adaptation.py`
- Create: `backend/tests/review_policy/test_privacy.py`

**Interfaces:**
- Produces: immutable `HStarManifest`, `KappaManifest`, and `adapt_hstar(manifest) -> KappaCandidate`; TaskExecutor consumes only frozen Kappa.

- [ ] Define component mappings: H-star tools → action recipes, middleware → review checks, skills → review guidance, failure memory → failure patterns; require an explicit transform/split/merge/drop disposition.
- [ ] Define `propose_hstar(discovery_runs) -> HStarCandidate` and `evaluate_hstar(candidate, regression_runs) -> HStarDecision` with fixed budgets, one revision maximum, paired acceptance thresholds, and executable behavior rather than prose-only scoring.
- [ ] Test that Kappa contains no calibration IDs, copied questions, chart values, answers, private paths, or actions outside H.
- [ ] Test that calibration outcomes cannot call `OperationalMemory.append_episode` and that changing any component changes the frozen hash.
- [ ] Create the minimal v1 artifact structure for axis/unit verification, legend grounding, evidence-before-answer, arithmetic grounding, and stall/budget handling; mark empirical acceptance pending Task 16.
- [ ] Run review-policy tests offline.
- [ ] If commits are authorized, commit as `[FEAT]: define the frozen ChartQA review policy`.

### Task 11: Implement version lifecycle in two reviewable slices

**Files:**
- Create: `backend/src/memory_distillation/version_lifecycle/__init__.py`
- Create: `backend/src/memory_distillation/version_lifecycle/consolidation.py`
- Create: `backend/src/memory_distillation/version_lifecycle/screening.py`
- Create: `backend/src/memory_distillation/version_lifecycle/candidates.py`
- Create: `backend/src/memory_distillation/version_lifecycle/promotion.py`
- Create: `backend/tests/version_lifecycle/test_consolidation.py`
- Create: `backend/tests/version_lifecycle/test_screening.py`
- Create: `backend/tests/version_lifecycle/test_candidates.py`
- Create: `backend/tests/version_lifecycle/test_promotion.py`

**Interfaces:**
- Produces: `VersionLifecycle.build_candidate(input: ConsolidationInput) -> FrozenCandidate` and `decide_promotion(candidate, report) -> PromotionDecision`.

- [ ] **Slice A — consolidation and candidate construction:** test CREATE/REINFORCE/REFINE/RETIRE, infrastructure/direct-answer exclusions, contamination rejection, collection-only retrieval-threshold calibration, duplicate/contradiction metrics, immutable definitions, and deterministic manifests. Consolidator input is a complete `EpisodeSnapshot`; deterministic validators retain authority. Commit separately if authorized.
- [ ] **Slice B — promotion:** consume `PromotionComparison` rather than importing the evaluator; test exact arithmetic, latency/input-token gates, rejection preservation, and atomic compare-and-set pointer advance. For 30 tasks, use at least two net additional correct answers and zero Negative Transfer unless Task 1 changes the threshold. Commit separately if authorized.
- [ ] Freeze the calibrated retrieval threshold and its Collection-only evidence before any Promotion task runs; Promotion and Final outcomes cannot retune it.
- [ ] Run lifecycle tests and the complete offline suite after each slice.

### Task 12: Implement C0–C3 evaluation and statistical reporting

**Files:**
- Create: `backend/src/memory_distillation/evaluation/__init__.py`
- Create: `backend/src/memory_distillation/evaluation/plan.py`
- Create: `backend/src/memory_distillation/evaluation/evaluator.py`
- Create: `backend/src/memory_distillation/evaluation/metrics.py`
- Create: `backend/src/memory_distillation/evaluation/report.py`
- Create: `backend/tests/evaluation/test_conditions.py`
- Create: `backend/tests/evaluation/test_metrics.py`
- Create: `backend/tests/evaluation/test_irrelevant_memory.py`

**Interfaces:**
- Produces: `ExperimentEvaluator.evaluate(plan: ExperimentPlan) -> ExperimentReport` by reusing `TaskExecutor`.

- [ ] Test exact C0–C3 policy differences and identical shared budgets/configuration.
- [ ] Test deterministic counterbalanced condition order and matched irrelevant-card assignment; report tasks without an eligible C3 control rather than inventing one.
- [ ] Freeze the C3 donor pool and irrelevance assignment before final access; report the eligible paired population and label accepted-v000 as a degenerate negative-result case in which C2/C3 collapse rather than demonstrating relevance.
- [ ] Implement paired accuracy, Positive/Negative Transfer, discordant pairs, exact McNemar p-value, seeded paired-bootstrap interval, Reviewer gap recovery, p50/p95 latency, tokens, calls, optional cost, intervention rate, trigger precision, no-retrieval rate, compression ratio, and duplicate/contradiction rate with explicit denominators and missing-data behavior.
- [ ] Test zero-denominator gap recovery, missing usage data, task failures, and empty retrieval without fabricated defaults.
- [ ] Ensure final results are immutable artifacts and never operational-memory input.
- [ ] Run evaluation tests and the complete offline suite.
- [ ] If commits are authorized, commit as `[FEAT]: evaluate paired memory-distillation conditions`.

### Task 13: Add an offline lifecycle runner and acceptance fixture

**Files:**
- Create: `backend/src/memory_distillation/cli.py`
- Create: `backend/src/memory_distillation/composition.py`
- Create: `backend/tests/integration/test_offline_lifecycle.py`
- Create: `backend/tests/fixtures/lifecycle/`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Produces: `memory-distillation dry-run`, `generate-manifests`, `smoke`, `calibrate`, `collect`, `promote`, `evaluate`, and `export-replay` commands using explicit composition roots.

- [ ] Write an end-to-end synthetic test: accepted-v000 → reviewed collection → candidate card → promotion acceptance and a separate rejection case.
- [ ] Assert exact resume/idempotency behavior after interruption at episode persistence, candidate freeze, evaluation completion, and pointer advance.
- [ ] Cover uncertain provider outcomes, quota exhaustion, cancellation, incomplete snapshots, and reconciliation without selective task deletion or duplicate paid submissions.
- [ ] Implement command dry-run/confirmation output showing mode, manifests, model IDs, write scopes, estimated request counts, and artifact paths before live work.
- [ ] Ensure only `evaluate --final` can construct `FinalBenchmark`.
- [ ] Run the offline lifecycle twice and prove byte-stable reports for the same seed/configuration.
- [ ] If commits are authorized, commit as `[FEAT]: orchestrate reproducible lifecycle runs`.

### Task 14: Expose a read-only FastAPI evidence API

**Files:**
- Modify: `backend/src/memory_distillation/main.py`
- Create: `backend/src/memory_distillation/api.py`
- Create: `backend/tests/api/test_runs.py`
- Create: `backend/tests/api/test_events.py`

**Interfaces:**
- Produces: read-only run/report/replay/artifact endpoints. CLI remains the sole execution surface; live SSE is optional after replay works.

- [ ] Test health, run summaries, task details, allowlisted chart-image access, version history, promotion reports, and replay loading.
- [ ] Test that replay is labeled, secrets/provider payloads are absent, and browser clients cannot call provider or memory adapters directly.
- [ ] If time remains, add bounded SSE buffering, last-event-ID resume, completed-run behavior, ordering, and deduplication tests; do not expose browser run-start or managed-write endpoints.
- [ ] Run API tests, Ruff, and mypy.
- [ ] If commits are authorized, commit as `[FEAT]: expose experiment evidence through FastAPI`.

### Task 15: Build the experiment visualizer

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/contracts.ts`
- Create: `frontend/src/components/TaskPanel.tsx`
- Create: `frontend/src/components/InterventionPanel.tsx`
- Create: `frontend/src/components/MemoryPanel.tsx`
- Create: `frontend/src/components/MetricsStrip.tsx`
- Create: `frontend/src/components/VersionTimeline.tsx`
- Create: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: Task 14 HTTP/SSE/replay contracts.
- Produces: responsive LIVE/REPLAY visual evidence for task state, action diff, episode→card provenance, candidate/promotion status, and C0–C3 metrics.

- [ ] Add frontend tests for empty/loading/error states, PASS/REPLACE diff, replay labeling, rejected candidate, accepted pointer advance, and missing C3 control.
- [ ] Implement the three-panel task/intervention/memory layout, top metrics strip, and version timeline without exposing provider controls.
- [ ] Consume replay polling first. If Task 14’s optional SSE is completed, use native `EventSource`, reconnect by last event ID, and deduplicate events.
- [ ] Run `npm run typecheck`, `npm test`, and `npm run build`.
- [ ] Render in a browser at desktop and narrow widths; verify zero console errors and capture clean evidence.
- [ ] If commits are authorized, commit as `[FEAT]: visualize the Version N+1 evidence chain`.

### Task 16: Verify and complete every production adapter

**Files:**
- Modify only as evidence requires: production adapters, `backend/.env.example`, lockfiles, and provider contract tests.
- Create: `backend/src/memory_distillation/operational_memory/memmachine.py`
- Create: `backend/tests/operational_memory/test_memmachine_contract.py`
- Create ignored evidence under: `.artifacts/smoke/<timestamp>/`

**Interfaces:**
- Converts transport-stub compatibility into verified live compatibility. No measured experiment begins until all checks pass.

- [ ] Confirm Crusoe project, exact Student/Reviewer/Consolidator IDs, image modality, structured output, quotas, credits, rate limits, and estimated full-run request count/cost.
- [ ] Run a fixed pilot to estimate actions, review calls, tokens, latency, failure rate, and worst-case full-run requests; set and enforce a hard admission ceiling before measured work.
- [ ] Send one Student multimodal request and record model ID, image handling, typed action validity, usage, latency, and request ID.
- [ ] Send one Nemotron review request and prove PASS plus valid REPLACE behavior on fixed pilot proposals.
- [ ] Send one Consolidator request and prove strict card-operation schema reliability.
- [ ] Confirm MemMachine target/namespace/write scope and implement the production adapter for exact record round-trip, stable IDs, paginated complete episode snapshots, exact card/manifest/evidence reads, search readiness, and accepted-pointer semantics.
- [ ] Add one synthetic episode/card/manifest, search and read it, enforce record-type and pinned-manifest filtering, and verify idempotent IDs. If MemMachine cannot provide atomic compare-and-set, stop for an architectural decision rather than emulating stronger guarantees silently.
- [ ] Execute one harmless bounded Python request through NemoClaw/OpenShell; verify network denial, scoped filesystem, timeout, and no local fallback.
- [ ] Update adapters only from observed APIs, run provider-marked contract tests, and record any changed model/config hashes.
- [ ] Stop and report rather than substituting providers or weakening boundaries if any critical seam fails.
- [ ] If commits are authorized, commit as `[FIX]` or `[FEAT]` according to the observed adapter changes.

### Task 17: Calibrate and freeze H-star/Kappa

**Files:**
- Update generated artifacts only: `backend/review_policy/chartqa_hstar_v1/`, `backend/review_policy/chartqa_kappa_v1/`, and ignored calibration reports.
- Modify docs/ADR only if the accepted policy changes the architectural contract.

**Interfaces:**
- Consumes: 8 discovery and 8 regression calibration tasks plus verified live models.
- Produces: empirically accepted, hashed Kappa consumed by measured Collection and C1.

- [ ] Run H on the eight discovery tasks and classify failure patterns without exposing the eight regression tasks to the proposer.
- [ ] Propose one bounded H-star revision focused on the smallest repeated failure families.
- [ ] A separate evaluator runs paired H/H-star evaluation on the complete fixed eight-item regression subset; its outcomes decide acceptance but never feed another revision.
- [ ] Adapt accepted H-star components into Kappa and produce the transform/split/merge/drop map.
- [ ] Smoke-test 3–5 fixed calibration tasks through PASS/REPLACE; record validity, intervention rate, direct-answer replacements, and outcome delta.
- [ ] Freeze H-star/Kappa content and hashes. If no revision passes, freeze the minimal validated rubric and narrow the claim honestly.
- [ ] Do not manually edit individual review outcomes or measured trajectories.

### Task 18: Run the measured lifecycle and final experiment

**Files:**
- No committed source changes unless a reproducible defect is found and fixed with a regression test.
- Write ignored run artifacts under `.artifacts/runs/`.

**Interfaces:**
- Consumes: frozen code revision, manifests, Kappa, models, policies, and provider gate evidence.
- Produces: one Promotion Decision and one immutable C0–C3 Experiment Report.

- [ ] Record a preflight manifest containing commit, dirty-state disclosure, model IDs, Kappa hash, dataset hashes, budgets, retrieval policy, expected calls, quota, and cost ceiling.
- [ ] Require a user-authorized frozen revision for measured runs; if a dirty tree is unavoidable, record a deterministic diff hash and prohibit source changes until final evaluation completes.
- [ ] Run all 40 Collection tasks with Accepted v000, Reviewer on, and idempotent Reviewed Episode writes.
- [ ] Freeze the exact episode snapshot; run one Consolidator batch; screen and freeze one Candidate Version.
- [ ] Run paired Accepted-versus-Candidate evaluation on all 30 Promotion tasks with Reviewer/writes off; apply the deterministic gate and atomically accept or reject.
- [ ] Freeze the final Accepted Version and run all 60 Final Held-Out tasks under C0–C3 with counterbalanced order and writes off.
- [ ] Export paired reports, intervals, discordant pairs, transfer counts, latency/tokens/calls/cost, retrieval diagnostics, contamination audit, and failure inventory.
- [ ] Never rerun or change configuration to improve the headline result; infrastructure-only reruns must be separately logged and justified.

### Task 19: Package, verify, and rehearse the submission

**Files:**
- Modify: `README.md`
- Modify: `docs/PROPOSAL_revised.html`
- Modify diagrams only if implemented behavior differs.
- Create: one small sanitized replay fixture in a documented frontend/backend fixture path.
- Do not commit raw traces, full datasets, benchmark images, or generated evaluation artifacts.

**Interfaces:**
- Consumes: verified implementation and measured report.
- Produces: independently reproducible repository, honest live/replay demo, and submission evidence.

- [ ] Update README with architecture, setup, exact commands, environment variables, experiment contract, observed—not planned—results, limitations, and sponsor attribution.
- [ ] Embed only measured metrics; distinguish provider smoke, synthetic integration, promotion, final, rendered, and unverified evidence.
- [ ] Sanitize one representative replay and prove it cannot be imported as operational memory.
- [ ] Run full backend test/lint/type suite and frontend test/type/build suite; record exact results.
- [ ] Perform rendered browser checks in LIVE and REPLAY mode with zero console errors and prominent mode labels.
- [ ] Rehearse the three-minute story: failure → Reviewer correction → Reviewed Episode → Behavior Card → promotion → Reviewer-free retrieval → C0–C3 evidence.
- [ ] Prepare the required one-minute video, verify its accessible link, verify the repository/public demo links, and complete the team-membership/submission checklist. Publishing, pushing, uploading, or submitting still requires explicit user authorization.
- [ ] Scan tracked files for secrets, `.env`, datasets, images, provider payloads, traces, and oversized artifacts.
- [ ] Confirm the final documentation, architecture diagrams, configuration samples, code, and claims agree before requesting any commit/push/submission action.
- [ ] Run exact formatter checks in addition to tests, lint, typing, builds, and rendered verification; define deterministic report comparison as a canonical projection that excludes timestamps, provider request IDs, and measured durations.

## Completion matrix

| Required outcome | Owning task(s) | Proof |
| --- | --- | --- |
| Frozen H-star/Kappa provenance | 1, 10, 17 | Hashes, mapping table, calibration report |
| ChartQA isolation and scoring | 4 | Manifest/scorer/architecture tests |
| Reviewed Student action loop | 3, 6–9 | Offline contract and integration tests |
| Safe Python execution | 7, 16 | Policy tests plus real OpenShell smoke |
| MemMachine operational memory | 8, 16 | Contract tests plus real add/search/filter smoke |
| Immutable candidate lifecycle | 8, 11, 13 | Acceptance and rejection integration fixtures |
| C0–C3 controlled evaluation | 12, 18 | Paired final report and contamination audit |
| Live/replay visual evidence | 5, 14, 15, 19 | API tests and rendered browser verification |
| Negative result remains valid | 11, 13, 18 | Rejection fixture and unchanged pointer evidence |
| Reproducible submission | 2, 13, 19 | Commands, lockfiles, full verification record |

## Explicitly deferred

- MathVista generalization.
- Raw Reviewed Episode retrieval.
- LoRA, SFT, or any weight update.
- More than one measured consolidation/promotion cycle.
- Repeated full final runs; only the predeclared robustness subset is optional.
- Hybrid retrieval/reranking beyond deterministic metadata filtering plus the verified MemMachine search primitive.
- Generic SaaS administration, authentication, multi-user tenancy, and deployment automation.


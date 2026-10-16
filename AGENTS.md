# Memory-Distilled Agent Team

## Working contract

This file governs the entire repository. Before changing architecture,
experiments, integrations, evaluation, or the demo, read:

- `docs/PROPOSAL_revised.html` for the current product and experiment contract.
- `CONTEXT.md` for canonical domain language.
- `docs/adr/` for decisions that must not drift silently.
- `docs/DETAILS.md` when work affects hackathon requirements or sponsors.

`docs/PROPOSAL.html` is historical. When documents conflict, the revised
proposal and accepted ADRs are authoritative. Record a changed architectural
decision in an ADR and update every affected document and diagram in the same
change.

The repository currently contains only a minimal runnable FastAPI/Vite shell.
Do not implement agent behavior, provider calls, memory lifecycle, evaluation,
or the experiment visualizer until the user explicitly authorizes that work.

## Non-negotiable experiment boundaries

- Version N+1 is the claim; Shared Brain is its MemMachine-backed mechanism.
- ChartQA is required. MathVista remains stretch work until the primary loop,
  evaluation, and demo are complete.
- Managed MemMachine is the sole operational improvement memory. Local JSONL
  traces are append-only telemetry for measurement, debugging, and labeled
  replay; agents must never retrieve from them.
- The primary condition retrieves accepted, answer-free Behavior Cards only.
  Raw Reviewed Episodes may appear only in a separately labeled ablation.
- The Promotion Gate is deterministic and automated. Human inspection may
  diagnose infrastructure, but it may not curate cards or experimental results.
- Student-generated Python runs only inside the NemoClaw-managed OpenShell
  boundary with scoped filesystem access, denied network access, and bounded
  resources.
- The experiment visualizer uses React/Vite with FastAPI. Streamlit and runtime
  web search are outside the core design.
- The final held-out set cannot influence prompts, thresholds, retrieval,
  models, cards, or promotion. A valid negative result still satisfies the
  technical Definition of Done.

## Codebase structure

Start as a modular monolith with one Python backend and one React/Vite frontend.
Keep the physical tree small until behavior earns a new module:

```text
backend/
  pyproject.toml
  src/memory_distillation/
    __init__.py
    main.py               # FastAPI composition root and transport
  tests/
    test_health.py
frontend/
  src/
    App.tsx
    main.tsx
    styles.css
```

Add a directory only when its first real behavior is implemented. Organize
future backend code around lifecycle capabilities—not agent roles—and grow
these deep modules in dependency order:

- `TaskExecutor.run(task, policy) -> TaskRun` hides the complete typed action
  loop, retrieval, Reviewer mode, sandbox execution, budgets, and trajectory.
- `OperationalMemory` owns Reviewed Episodes, retrieval, immutable Candidate
  Versions, and the Accepted Version pointer. No caller uses MemMachine directly.
- `VersionLifecycle.build_candidate(...)` and `decide_promotion(...)` own
  consolidation, screening, canonical hashing, freezing, and promotion.
- `ExperimentEvaluator.evaluate(plan) -> ExperimentReport` owns paired C0-C3
  execution, official scoring, statistical analysis, and report construction.
- Inference, sandbox, operational memory, and telemetry each have a production
  adapter and a deterministic test adapter at their external seam.

When introduced, each deep module exposes its public interface from
`__init__.py` and keeps provider formats, retries, parsing, and state transitions
private. Keep adapters beside the seam they satisfy. Do not create top-level
`agents/`, `services/`, `managers/`, `repositories/`, `shared/`, or `utils/`
buckets.

Dependency direction is HTTP and CLI → task execution → versioning → evaluation,
with those modules depending on inference, sandbox, operational memory,
benchmark, and telemetry interfaces. Deep modules never import FastAPI. The
frontend never calls provider adapters. Evaluation reuses `TaskExecutor` rather
than recreating the agent loop.

Use one discriminated `RunEvent` schema for live server-sent events, local JSONL
telemetry, sanitized replay, and metric aggregation. Runtime modules receive a
write-only event sink; only visualizer and evaluation paths may read traces.

Expose development and final benchmark access through different interfaces.
Collection, consolidation, calibration, and promotion receive collection and
promotion manifests only. Only the final-evaluation composition root receives
the final manifest; enforce this import rule with an architectural test.

## Change discipline

Investigate the current path before editing it. State assumptions when the
repository cannot establish a fact. Make the smallest cohesive change at the
responsible layer and preserve unrelated work in a dirty worktree.

- Agree on typed inputs, outputs, ownership, and failure semantics before
  parallel work across a boundary.
- Keep provider, storage, sandbox, benchmark, and UI code behind narrow
  adapters. Domain and evaluation logic must remain provider-independent.
- Prefer cohesive modules with explicit dependencies. Aim to keep handwritten
  source files below 500 lines; split only at a real responsibility boundary.
- Extend an existing responsibility before creating a competing abstraction.
  Apply DRY, KISS, SOLID, and YAGNI as design tests, not slogans.
- Validate untrusted data once at its entry boundary, then pass typed internal
  contracts. Reject invalid model output; do not hide it behind permissive
  parsing or silent fallback.
- Keep state transitions, manifests, scoring, promotion, and bookkeeping
  deterministic. Isolate unavoidable randomness behind explicit, recorded
  seeds and frozen configuration.
- Make retries bounded, observable, and safe to repeat. Fail closed at security,
  provenance, contamination, and model-identity boundaries.
- Raise or return contextual errors. Never swallow failures, fabricate defaults,
  or report an integration as working because configuration exists.
- Add dependencies only when the existing stack cannot meet the requirement.
  Pin them through the repository's package manager and commit lockfiles.
- Keep generated files, formatting churn, and opportunistic refactors out of a
  focused change unless they are necessary to complete it.

## Code, documentation, and comments

Optimize for the next engineer being able to verify intent without archaeology.

- Use descriptive names and readable, stepwise control flow. Dense expressions
  and premature helper layers are both defects when they obscure the domain.
- Python must be typed. Use Pydantic or equivalent typed schemas at process and
  provider boundaries. TypeScript must use strict mode; avoid `any` and
  unchecked assertions.
- Public APIs and non-obvious modules require concise documentation. Complex
  functions require docstrings that document contracts, side effects, and
  raised errors; trivial accessors do not need ceremonial prose.
- Comments explain why, invariants, security constraints, or surprising
  tradeoffs. They must not paraphrase the next line or preserve implementation
  history that belongs in Git.
- Update user-facing docs, API contracts, examples, diagrams, configuration
  samples, and ADRs whenever their behavior changes. Documentation is part of
  the implementation, not follow-up work.
- Keep `CONTEXT.md` as the canonical glossary. Reuse its terms exactly and add
  new terms only when they remove genuine ambiguity.

## Security, credentials, and artifacts

Treat model output, benchmark files, provider responses, and replay payloads as
untrusted input.

- Keep credentials in ignored environment files and provide redacted
  placeholders in `.env.example`. Never log, commit, paste, or replay secrets.
- Permit generated code only the explicit OpenShell capabilities required by a
  task. Validate paths, enforce time and resource limits, and record policy
  violations without executing a fallback locally.
- Do not commit model weights, full datasets, benchmark images, raw provider
  payloads, run traces, or generated evaluation artifacts. Commit small,
  synthetic, clearly labeled fixtures when tests need data.
- Record external model identity, configuration version, dataset manifest,
  code revision, and sanitized provenance for every measured run.
- Confirm target, quota, estimated cost, and write scope before paid inference,
  managed-memory writes, deployments, migrations, or destructive operations.

## Testing and evaluation integrity

Every behavior change needs evidence at the narrowest useful layer. Reproduce a
bug before fixing it; preserve that reproduction as a regression test when
practical.

- Unit-test pure transformations, validators, metrics, state transitions, and
  promotion rules with small fixtures.
- Contract-test every external adapter with recorded or stubbed schemas, then
  smoke-test the real service before claiming integration success.
- Integration-test the collection-to-consolidation-to-promotion lifecycle and
  the Student's Behavior Card retrieval path.
- Verify the visualizer in a rendered browser. A frontend build alone is not UI
  verification; replay evidence must be visibly labeled as replay.
- Keep unit tests offline and deterministic. Networked, paid, or long-running
  tests must be explicit and excluded from the default fast suite.
- Assert manifest disjointness and final-set inaccessibility in code. Freeze
  compared prompts, budgets, decoding, tools, and retrieval policy.
- Report paired sample size, capability delta, Negative Transfer, latency, and
  token effects without cherry-picking. Distinguish exploratory, promotion,
  final, and robustness results.

Tooling is not scaffolded yet. Once project configuration exists, treat its
scripts and lockfiles as the command source of truth rather than copying commands
into this file. Before handoff, run the focused tests plus configured formatter,
linter, type checker, and build for every affected language. Report exact
commands, results, and anything that could not run; distinguish static, unit,
integration, rendered, provider, and full-run evidence.

## Git discipline

Keep commits atomic, buildable, and easy to revert. Target no more than 500
handwritten changed lines per commit; generated artifacts and lockfiles are
exceptions when inseparable from the change. Never mix formatting, refactoring,
and behavior changes without a concrete reason.

Use an uppercase intent prefix and imperative subject:

```text
[FEAT]: add reviewed-episode persistence
[FIX]: prevent final-manifest access during promotion
[DOCS]: record MemMachine retention decision
[TEST]: cover negative-transfer rejection
[REFACTOR]: isolate benchmark scoring adapter
[CHORE]: pin frontend toolchain
```

Commit messages and bodies must explain intent and important consequences, not
restate the diff. Never bypass hooks, rewrite shared history, commit, push,
publish, or deploy unless the user explicitly authorizes that action.

## Pull-request discipline

A pull request must be independently reviewable and evidence-first. Use the
same uppercase prefix convention in its title and include:

```markdown
## Objective
Why this change exists and the acceptance condition.

## Summary
The behavior and architecture changed, including explicit non-goals.

## Changed Files
Each changed area and its responsibility.

## Verification
Exact commands and observed results; list tests not run and why.

## Deployment / Configuration
Environment variables, migrations, provider state, rollout, and rollback.

## Risks / Follow-ups
Known limitations, residual risk, and deliberately deferred work.

## Issue / Ticket
Related issue and whether this PR resolves it fully or partially.
```

Before requesting review, state the exact base and head, included commits,
mergeability, CI status, and whether live-provider or rendered verification was
performed. A partial implementation must be labeled partial. Never claim that
checks, deployments, metrics, or external integrations succeeded without direct
evidence.

## Completion gate

Work is ready to hand off only when all applicable items are true:

- Requested behavior and failure paths are implemented within the agreed scope.
- Focused regression, contract, and integration tests pass.
- Formatting, linting, typing, builds, and rendered checks pass where relevant.
- Experiment isolation, provenance, and security invariants remain enforced.
- Documentation, examples, diagrams, configuration samples, and ADRs agree.
- No secrets, large artifacts, hidden fallback paths, or unrelated edits entered
  the change.
- The handoff names changed behavior, exact verification evidence, residual
  risks, and unverified assumptions.

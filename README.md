# Memory-Distilled Agent Team

Memory-Distilled Agent Team is an experimental multimodal agent system that turns
reviewed ChartQA trajectories into versioned, answer-free operating procedures.
The procedures live in MemMachine and can improve a fresh Student without changing
model weights or keeping an online Reviewer in the final serving path.

The project tests one central claim: can a reviewed agent team produce a measurable
**Version N+1** by distilling corrections into external procedural memory?

![Memory distillation lifecycle](docs/assets/memory-distillation-lifecycle-generated.png)

## Current status

The repository currently supports:

- Frozen and disjoint ChartQA manifests for calibration, collection, promotion,
  and final evaluation.
- A typed Student action loop with deterministic scoring and sanitized telemetry.
- Crusoe-hosted Student, Reviewer, and H-star adapters, plus a live Consolidator
  capability probe.
- NemoClaw/OpenShell execution for Student-generated Python.
- Managed MemMachine storage with immutable version and promotion contracts.
- Offline evaluation and promotion logic with deterministic tests.
- A live React dashboard for calibration progress and task-level outcomes.
- A paid Stage 0 command that proposes, regression-screens, freezes, and smoke-tests
  Kappa Reviewer guidance.

Stage 0 calibration is the active live workflow. The full live
collection-to-consolidation-to-promotion command and final measured C0-C3 results
are not complete yet. The repository does not claim a positive benchmark result.

## How it works

1. **Calibrate review guidance.** The Student runs a baseline calibration split.
   H-star proposes one bounded review rule, which is tested on discovery and sealed
   regression tasks before being frozen as Kappa Guidance.
2. **Collect reviewed trajectories.** The Student proposes actions and the Reviewer
   either passes them or supplies the smallest valid replacement before execution.
3. **Distill memory.** Reviewed Episodes are persisted to MemMachine. A Consolidator
   proposes CREATE, REINFORCE, REFINE, or RETIRE operations over an immutable snapshot.
4. **Promote a version.** A deterministic gate compares the Candidate and Accepted
   Versions on identical promotion tasks and budgets.
5. **Evaluate transfer.** A frozen Student is evaluated with and without accepted
   Behavior Cards on the untouched final set.

Calibration never creates Behavior Cards. H-star and Kappa are frozen before
measured collection begins, and local JSONL traces are telemetry rather than agent
memory.

## Quick start

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm
- Docker, NemoClaw, and OpenShell for live sandbox execution
- Crusoe and MemMachine credentials for their respective live workflows

### Install

```bash
git clone <repository-url>
cd recursive-agents-hackathon

cd backend
uv sync

cd ../frontend
npm ci
```

Copy the environment template and add the credentials available to you:

```bash
cp backend/.env.example backend/.env
```

`backend/.env` is ignored by Git. Do not commit provider keys or generated
experiment artifacts.

### Download ChartQA

The dataset is intentionally excluded from the repository:

```bash
git clone https://github.com/vis-nlp/ChartQA.git .local/chartqa-source
```

The default loader expects:

```text
.local/chartqa-source/ChartQA Dataset/
```

Committed manifests bind every experiment task to a dataset revision, task ID, and
image hash.

### Run the application

Start the API:

```bash
cd backend
uv run uvicorn memory_distillation.main:app --reload
```

Start the visualizer in another terminal:

```bash
cd frontend
npm run dev
```

The API defaults to `http://127.0.0.1:8000`. Vite defaults to
`http://127.0.0.1:5173` and proxies `/api` requests to FastAPI.

## Live calibration

Check provider identities, the sandbox, token ceiling, estimated maximum cost, and
artifact destination without starting paid inference:

```bash
cd backend
uv run memory-distillation calibrate
```

Start the paid Stage 0 run only after reviewing that preflight:

```bash
uv run memory-distillation calibrate --confirm-paid-run
```

Calibration uses these fixed roles:

| Role | Default model |
|---|---|
| Student | `Qwen/Qwen3.8-27B` |
| Reviewer | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` |
| H-star analysis | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Consolidator | `nvidia/Nemotron-3.5-Lightning-30B-A3B` |

Live progress is written to `.artifacts/calibration/current/` and streamed to the
visualizer. The directory must be empty before a new run so previous evidence cannot
be overwritten silently. A successful run produces:

```text
.artifacts/calibration/current/
  events.jsonl
  status.json
  hstar.json
  kappa.json
  report.json
```

Private model reasoning, credentials, benchmark answers, and raw provider payloads
are excluded from public telemetry.

## Architecture

![System design](docs/assets/memory-distillation-system-design.svg)

The Python backend is a modular monolith. Provider-specific code is isolated behind
typed inference, sandbox, benchmark, telemetry, and operational-memory adapters. The
React frontend consumes only the FastAPI dashboard API and sanitized server-sent
events.

Student-generated Python can execute only through the NemoClaw-managed OpenShell
boundary. There is no local execution fallback.

The editable architecture source is
[memory-distillation-system-design.drawio](docs/assets/memory-distillation-system-design.drawio).

## Evaluation design

The final experiment compares four paired conditions:

| Condition | Reviewer | Memory | Question answered |
|---|---:|---|---|
| C0 | Off | None | How capable is the fixed Student alone? |
| C1 | On | None | What is the ceiling with online review? |
| C2 | Off | Relevant accepted cards | Does distilled memory improve the Student? |
| C3 | Off | Trigger-irrelevant cards | Is any gain more than extra context? |

The report tracks capability delta, Positive and Negative Transfer, intervention
rate, latency, token usage, cost, and eligible paired sample size. The final held-out
set cannot influence prompts, models, thresholds, retrieval, memory, or promotion.

## Repository layout

```text
backend/
  experiment_manifests/        Frozen ChartQA task assignments
  src/memory_distillation/
    benchmark/                  Dataset access and official scoring
    calibration/                H-star proposal and Kappa freeze
    evaluation/                 Paired C0-C3 execution and reports
    inference/                  Crusoe and deterministic adapters
    operational_memory/         MemMachine and in-memory adapters
    review_policy/              Kappa schemas and screening
    sandbox/                    OpenShell execution boundary
    task_execution/             Student/Reviewer action loop
    telemetry/                  JSONL events and replay
    versioning/                 Consolidation, retrieval, and promotion
    visualizer/                 Dashboard evidence API
  tests/
frontend/                       React/Vite experiment visualizer
docs/                           Proposal, ADRs, diagrams, and evidence
```

## Verification

Run the offline backend checks:

```bash
cd backend
uv run pytest
uv run ruff check .
uv run mypy
```

Run the frontend checks:

```bash
cd frontend
npm run typecheck
npm run build
```

Live integration smoke tests are explicit and opt-in:

```bash
cd backend
uv run verify-crusoe --confirm-live
uv run verify-memmachine --confirm-live
uv run verify-openshell --confirm-live
```

These commands can make external requests or mutate managed test state. Review their
preflight output before confirming a live run.

## Configuration

All application configuration is loaded through
`backend/src/memory_distillation/settings.py`:

- `CRUSOE_*` selects the inference endpoint and fixed model roles.
- `MEMMACHINE_*` selects the managed Operational Memory project.
- `OPENSHELL_*` selects the pre-provisioned execution sandbox.

The redacted defaults and required variable names are documented in
[`backend/.env.example`](backend/.env.example).

## Documentation

- [Revised proposal](docs/PROPOSAL_revised.html)
- [Methodology addendum](docs/PROPOSAL_addendum.md)
- [Canonical terminology](CONTEXT.md)
- [Architecture decisions](docs/adr/)
- [Crusoe capability evidence](docs/verification/2026-10-17-crusoe-capabilities.md)
- [Contributor and agent conventions](AGENTS.md)

## Inspiration

This project is inspired by
[Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974v1).
It keeps response-boundary supervision but changes the transfer mechanism: reviewed
behavior becomes inspectable, versioned external memory rather than fine-tuned model
parameters.

# Memory-Distilled Agent Team

A multimodal agent system that turns reviewed ChartQA trajectories into
versioned, answer-free **Behavior Cards**. The Student model stays frozen; only
its external memory in MemMachine improves, and only when a new version
measurably helps.

## How It Works

<p align="center">
  <img src="docs/assets/memory-distillation-lifecycle-generated.png" alt="Memory distillation lifecycle" width="100%" />
</p>

- **What improves:** versioned Behavior Cards in MemMachine
- **What stays fixed:** Student model, tools, prompts, budgets, and scoring
- **What evaluates improvement:** paired ChartQA accuracy, Negative Transfer,
  latency, and token usage

## Quick Start

### 1. Install the backend

```bash
cd backend
uv sync
cp .env.example .env
```

Add your Crusoe, MemMachine, and OpenShell credentials to `backend/.env`.

### 2. Download ChartQA

```bash
git clone https://github.com/vis-nlp/ChartQA.git .local/chartqa-source
```

### 3. Verify the providers

```bash
cd backend
uv run verify-crusoe
uv run verify-memmachine
uv run verify-openshell
```

### 4. Start the API and dashboard

```bash
cd backend
uv run uvicorn memory_distillation.main:app --reload
```

```bash
cd frontend
npm ci
npm run dev
```

Open the dashboard at http://127.0.0.1:5173.

### 5. Run the measured lifecycle

After calibration finishes and `.artifacts/calibration/current/kappa.json` is
frozen, inspect the Collection/Promotion preflight and then explicitly start
the paid, managed-memory run:

```bash
uv run memory-distillation lifecycle
uv run memory-distillation lifecycle --confirm-paid-run
```

The command runs exactly 40 Collection tasks, one consolidation, and 30 paired
Promotion tasks. It writes ignored evidence under `.artifacts/lifecycle/current`.
If the process fails after all 40 Collection episodes were persisted, resume at
Consolidation without paying to regenerate them:

```bash
uv run memory-distillation lifecycle --resume --confirm-paid-run
```

Resume reads the immutable episodes from MemMachine and accepts them only when
their run, manifest, Kappa, accepted parent, seed schedule, configuration, and
task payloads form one complete pass. If several compatible passes exist, add
`--run-id <lifecycle-id>`. Local JSONL telemetry is never used as memory.

## Project Structure

```text
backend/
├── experiment_manifests/       # Frozen ChartQA task splits
├── src/memory_distillation/
│   ├── benchmark/              # Dataset loading and scoring
│   ├── calibration/            # Kappa Guidance calibration run
│   ├── evaluation/             # C0–C3 runs and statistics
│   ├── inference/              # Crusoe model adapters
│   ├── operational_memory/     # MemMachine adapter
│   ├── review_policy/          # Frozen Kappa Guidance
│   ├── sandbox/                # OpenShell execution boundary
│   ├── task_execution/         # Student and Reviewer action loop
│   ├── telemetry/              # Run events and replay
│   ├── versioning/             # Consolidation and promotion
│   └── visualizer/             # Dashboard API
└── tests/
frontend/                       # React dashboard
docs/                           # Proposal, ADRs, and diagrams
```

## Models

| Role | Default model | Why |
|---|---|---|
| Student | `Qwen/Qwen3.8-27B` | Smaller multimodal model with room to improve |
| Reviewer | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` | Stronger multimodal reviewer for PASS/REPLACE |
| Consolidator | `nvidia/Nemotron-3.5-Lightning-30B-A3B` | Reliable structured output for card proposals |
| H-star analysis | `deepseek-ai/DeepSeek-V4.1-Flash` | Proposes the calibration revision |

All models run on Crusoe and can be changed in `backend/.env`.

## Architecture

<p align="center">
  <img src="docs/assets/memory-distillation-system-design.svg" alt="Memory-Distilled Agent Team architecture" width="100%" />
</p>

## References

- [Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974)
  ([code](https://github.com/metaevo-ai/harness-zero)), the source of the
  PASS/REPLACE review pattern
- [ChartQA benchmark](https://github.com/vis-nlp/ChartQA)

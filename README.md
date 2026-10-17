# Memory-Distilled Agent Team

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![React + Vite](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)
![Benchmark: ChartQA](https://img.shields.io/badge/benchmark-ChartQA-6E56CF)

Can a small vision agent learn from a reviewer's corrections without any weight
updates, and without the reviewer present at test time?

This project turns reviewed ChartQA trajectories into versioned, answer-free
**Behavior Cards** stored in MemMachine. The Student model never changes; only
its external memory does, and only when a new version measurably helps.

> **Status:** the lifecycle and live provider adapters are built. Calibration,
> the measured Collection/Promotion pass, and the final C0–C3 run are in
> progress, so there are no results yet.

<p align="center">
  <img src="docs/assets/memory-distillation-lifecycle-generated.png" alt="Memory distillation lifecycle: calibrate, collect, consolidate, promote, evaluate" width="100%" />
</p>

## How it works

0. **Calibrate.** Derive a private review policy (Kappa Guidance) on 16
   calibration tasks, then freeze it.
1. **Collect.** On 40 tasks, a Nemotron Reviewer checks each Student action and
   either passes it or replaces it with the smallest valid correction.
2. **Consolidate.** The reviewed episodes become a candidate set of answer-free
   Behavior Cards, screened for leakage and frozen.
3. **Promote.** The candidate and the current version run on 30 tasks with the
   Reviewer off. A deterministic gate accepts or rejects the candidate.
4. **Evaluate.** The accepted version is tested once on 60 held-out tasks
   against three controls: no memory, live Reviewer, and irrelevant cards of
   the same size.

## Quick start

Requires Python 3.12 with [uv](https://docs.astral.sh/uv/) and Node.js.

```bash
# Backend API (http://127.0.0.1:8000)
cd backend
uv sync
uv run uvicorn memory_distillation.main:app --reload

# Dashboard (http://127.0.0.1:5173), in a second terminal
cd frontend
npm ci
npm run dev
```

The dashboard shows manifests, integration status, and live run events as they
are written. It reports no results until a measured run has produced them.

<details>
<summary><b>Live mode</b> (paid inference and managed services)</summary>

```bash
cd backend
cp .env.example .env   # add Crusoe, MemMachine, and OpenShell settings
git clone https://github.com/vis-nlp/ChartQA.git ../.local/chartqa-source

uv run verify-crusoe
uv run verify-memmachine
uv run verify-openshell
uv run memory-distillation calibrate --help
```

Model IDs in `.env.example` are configurable defaults. The `verify-*` commands
must pass before any measured run.

</details>

## Development

```bash
cd backend && uv run pytest -q && uv run ruff check src tests && uv run mypy
cd frontend && npm run typecheck && npm run build
```

## Project layout

```text
backend/src/memory_distillation/
  task_execution/      Student action loop and Reviewer boundary
  review_policy/       Kappa Guidance freezing
  versioning/          Candidate building, retrieval, and promotion
  evaluation/          C0–C3 runs and paired statistics
  operational_memory/  MemMachine adapter
  sandbox/             OpenShell execution boundary
frontend/              React dashboard
docs/                  Proposal, glossary, decision records, diagrams
```

## Learn more

- [Revised proposal](docs/PROPOSAL_revised.html): full experiment design
- [Implementation addendum](docs/PROPOSAL_addendum.md): phase-by-phase contracts
- [Glossary](CONTEXT.md) and [decision records](docs/adr/)
- [System architecture diagram](docs/assets/memory-distillation-system-design.svg)

## Acknowledgements

The PASS/REPLACE review pattern is adapted from
[Harness-Zero](https://github.com/metaevo-ai/harness-zero), which distills
behavior into model weights. This project keeps it in versioned external memory
instead. Uses the [ChartQA](https://github.com/vis-nlp/ChartQA) benchmark.

<details>
<summary>Cite Harness-Zero</summary>

```bibtex
@misc{ye2026harnesszeroharnessdistillationagentasharness,
  title         = {Harness-Zero: Harness Distillation via Agent-as-Harness},
  author        = {Haoran Ye and Yuxing Lu and Haonan Dong and Zhaochen Su and Guojie Song},
  year          = {2026},
  eprint        = {2609.24974},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.24974}
}
```

</details>

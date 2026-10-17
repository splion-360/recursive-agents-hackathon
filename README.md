# Memory-assisted Harness-Zero

Inspired by [Harness-Zero](https://arxiv.org/pdf/2609.24974) by Google.

## Backend checks

```bash
cd backend
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

Live provider checks are opt-in:

```bash
uv run verify-crusoe --confirm-live
uv run verify-memmachine --confirm-live
uv run verify-openshell --confirm-live
```

The OpenShell check requires a pre-provisioned NemoClaw sandbox identified by
`OPENSHELL_SANDBOX_NAME` and `OPENSHELL_GATEWAY`. It verifies containment and
does not invoke Student model inference.

The MemMachine check creates an isolated temporary project, verifies immutable
episode and Candidate Version records, pagination, pinned retrieval, and the
accepted-pointer safety gate, then deletes only that temporary project.

## Live calibration

Start the FastAPI backend and Vite dashboard, then open the dashboard URL shown
by Vite. Run the paid calibration separately so its progress remains visible:

```bash
cd backend
uv run memory-distillation calibrate --confirm-paid-run
```

The command prints the configured cost and token ceilings before making any
inference call. It exposes only the eight discovery tasks to H-star, freezes one
proposal, then loads the separate eight-task regression split. It writes public
`RunEvent` telemetry and status under `.artifacts/calibration/current/`; the
dashboard streams those files through FastAPI. Private Reviewer reasoning,
provider payloads, credentials, and scoring references are never written.

Successful runs freeze `hstar.json`, `kappa.json`, and `report.json` in that
ignored artifact directory. A non-empty directory blocks another calibration
run so evidence cannot be silently overwritten or mixed across attempts.

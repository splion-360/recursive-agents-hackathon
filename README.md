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

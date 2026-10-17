# Three-Round Kappa Evolution Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the one-shot 8+8 calibration with three versioned H-star evolution rounds over 12 Discovery tasks, screen the final proposal on 12 sealed Regression tasks, and expose the exact Student, Reviewer, H-star, and Kappa harness definitions in the dashboard.

**Architecture:** Calibration remains provider-independent. The orchestrator owns the deterministic 3x4 schedule and immutable revision history; the Crusoe adapters only translate typed requests and validate provider output. The dashboard reads sanitized configuration and frozen artifacts, never provider state or private reasoning. Existing v1 artifacts remain viewable.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, FastAPI, React 19, TypeScript, Vite.

**Specification:** Discovery contains 12 unique tasks split into three ordered rounds of four. Each round evaluates the current harness, then produces a full replacement H-star proposal bound to all observations available through that round. Only the third proposal is evaluated against 12 sealed Regression tasks and adapted into frozen Kappa. A Reviewer REPLACE of a proposed `final_answer` must select an evidence-gathering action (`inspect_image`, `crop`, or `run_python`), never another `final_answer`.

**Global constraints:** Do not run paid inference. Do not alter MemMachine behavior. Preserve unrelated dirty files and legacy completed artifacts. Do not commit without renewed user authorization.

## Task 1: Encode the 12-task, three-round calibration contract

**Files:** `backend/src/memory_distillation/calibration/models.py`, `backend/src/memory_distillation/review_policy/kappa.py`, `backend/src/memory_distillation/benchmark/manifests.py`, calibration/benchmark tests and generated manifest JSON.

1. Add failing tests for 12+12 split validation, round metadata, and deterministic 12-task manifests.
2. Run focused tests and confirm failures reflect the old 8-task contract.
3. Add typed H-star revision history and dynamic progress totals while retaining v1 artifact compatibility.
4. Regenerate manifests using the existing deterministic generator and rerun focused tests.

## Task 2: Execute three H-star evolution rounds and freeze the final revision

**Files:** `backend/src/memory_distillation/calibration/orchestrator.py`, `backend/src/memory_distillation/calibration/command.py`, `backend/src/memory_distillation/inference/crusoe.py`, calibration tests.

1. Add a failing orchestration test asserting three four-task analyst calls, cumulative answer-free observations, parent revision linkage, sealed Regression access only after the final proposal, and final-revision Kappa adaptation.
2. Implement the smallest orchestration change that passes it.
3. Persist all revisions in `hstar.json`; keep `proposal` as the final revision for compatible readers.
4. Verify focused tests.

## Task 3: Enforce procedural Reviewer replacement

**Files:** `backend/src/memory_distillation/inference/crusoe.py`, `backend/tests/inference/test_crusoe.py`.

1. Add a failing test for rejecting `final_answer -> final_answer` REPLACE output.
2. Tighten the Reviewer prompt and typed boundary validation.
3. Verify inference contract tests.

## Task 4: Show exact harness definitions and evolution in the dashboard

**Files:** `backend/src/memory_distillation/visualizer/dashboard.py`, `backend/tests/visualizer/test_dashboard.py`, `frontend/src/api.ts`, `frontend/src/App.tsx`, `frontend/src/styles.css`.

1. Add failing API tests for Student, Reviewer, H-star, and Kappa definitions including final rule text and revision history.
2. Add a sanitized dashboard projection with dynamic split totals.
3. Render compact expandable harness cards and three-round evolution state.
4. Build and verify the dashboard in a browser at desktop and narrow widths.

## Task 5: Reconcile documentation and verify

**Files:** `CONTEXT.md`, `docs/PROPOSAL_addendum.md`, `docs/PROPOSAL_revised.html`, `docs/adr/0012-use-three-round-hstar-evolution.md` and any diagram source that states obsolete counts.

1. Record the superseding decision and update all authoritative references from one 8-task revision to 3x4 Discovery plus 12 sealed Regression.
2. Run backend tests, Ruff, mypy, frontend build, and rendered dashboard checks.
3. Confirm no paid calls, MemMachine changes, secrets, generated datasets, or commits were introduced.

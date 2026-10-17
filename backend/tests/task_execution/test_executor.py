"""Offline behavior tests for the minimal Student action loop."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.inference import DeterministicInference
from memory_distillation.review_policy import KappaArtifact, KappaRule, RuleEvidence
from memory_distillation.sandbox import DeterministicSandbox, SandboxResult
from memory_distillation.task_execution import (
    CropAction,
    DeterministicReviewer,
    ExecutionPolicy,
    FinalAnswerAction,
    InspectImageAction,
    ReviewResult,
    RunEvent,
    RunPythonAction,
    StudentAction,
    TaskExecutor,
    TaskRun,
)
from memory_distillation.telemetry import InMemoryEventSink


def _kappa(instruction: str = "Prefer exact arithmetic.") -> KappaArtifact:
    rule = KappaRule(
        rule_id="review-arithmetic",
        action_types=("inspect_image", "crop", "run_python", "final_answer"),
        instruction=instruction,
    )
    evidence = RuleEvidence(
        rule_id=rule.rule_id,
        discovery_task_ids=tuple(f"d-{index}" for index in range(8)),
        regression_task_ids=tuple(f"r-{index}" for index in range(8)),
        discovery_helped=1,
        regression_helped=1,
        regression_harmed=0,
        action_compatible=True,
    )
    payload = {
        "manifest_hash": "calibration-manifest",
        "revision": 1,
        "rules": [rule.model_dump(mode="json")],
        "evidence": [evidence.model_dump(mode="json")],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return KappaArtifact(
        artifact_id=f"kappa:{digest[:16]}",
        manifest_hash="calibration-manifest",
        revision=1,
        rules=(rule,),
        guidance=(instruction,),
        evidence=(evidence,),
        content_hash=digest,
    )


class FixtureVerifier:
    """Score only the known fixture answer."""

    def score(self, task_id: str, prediction: str) -> Score:
        assert task_id == "chartqa:fixture:001"
        return Score(correct=prediction == "42", method="text_exact")


def test_offline_baseline_trajectory_is_complete_and_reproducible() -> None:
    """One scripted task exercises every action through public seams."""

    first = _run_fixture()
    second = _run_fixture()

    assert first == second
    assert first.termination_reason == "completed"
    assert first.score == Score(correct=True, method="text_exact")
    assert [action.type for action in first.actions] == [
        "inspect_image",
        "crop",
        "run_python",
        "final_answer",
    ]
    assert first.python_calls == 1
    assert first.tool_calls == 3
    assert [event.sequence for event in first.events] == list(range(len(first.events)))
    assert first.events[0].type == "run_started"
    assert first.events[-1].type == "run_completed"


@pytest.mark.parametrize(
    ("actions", "policy", "expected"),
    [
        ((InspectImageAction(),), ExecutionPolicy(max_actions=1), "action_budget_exhausted"),
        ((InspectImageAction(),), ExecutionPolicy(max_tool_calls=0), "tool_budget_exhausted"),
        (
            (RunPythonAction(code="print(1)"),),
            ExecutionPolicy(max_python_calls=0),
            "python_budget_exhausted",
        ),
        ((InspectImageAction(),), ExecutionPolicy(max_total_tokens=1), "token_budget_exhausted"),
        ((InspectImageAction(),), ExecutionPolicy(max_elapsed_ms=1), "time_budget_exhausted"),
    ],
)
def test_execution_budgets_fail_closed(
    actions: tuple[StudentAction, ...],
    policy: ExecutionPolicy,
    expected: str,
) -> None:
    """Every Minimal Harness budget stops the run with explicit evidence."""

    task, asset = _fixture_input()
    run = TaskExecutor(
        inference=DeterministicInference(actions=actions, model_id="fixture-student-v1"),
        sandbox=DeterministicSandbox(results={}),
        verifier=FixtureVerifier(),
        event_sink=InMemoryEventSink(),
    ).run(task=task, asset=asset, policy=policy, seed=17)

    assert run.termination_reason == expected
    assert run.events[-1].type == "run_completed"


def test_reviewer_passes_or_replaces_before_execution_without_leaking_reason() -> None:
    task, asset = _fixture_input()
    proposed = RunPythonAction(code="print('unsafe proposal')")
    accepted = RunPythonAction(code="print(42)")
    inference: DeterministicInference[StudentAction] = DeterministicInference(
        actions=(proposed, FinalAnswerAction(answer="42")),
        model_id="fixture-student-v1",
    )
    reviewer = DeterministicReviewer(
        results=(
            ReviewResult(
                decision="replace",
                accepted_action=accepted,
                private_reason="hidden reviewer chain of thought",
                model_id="fixture-reviewer-v1",
                request_id="review:1",
                input_tokens=7,
                output_tokens=2,
                latency_ms=3,
            ),
            ReviewResult(
                decision="pass",
                accepted_action=FinalAnswerAction(answer="42"),
                private_reason="hidden final rationale",
                model_id="fixture-reviewer-v1",
                request_id="review:2",
                input_tokens=7,
                output_tokens=2,
                latency_ms=3,
            ),
        )
    )
    sandbox = DeterministicSandbox(
        results={
            accepted.code: SandboxResult(
                status="completed", stdout="42\n", stderr="", exit_code=0, latency_ms=1
            )
        }
    )
    kappa = _kappa()

    run = TaskExecutor(
        inference=inference,
        sandbox=sandbox,
        verifier=FixtureVerifier(),
        event_sink=InMemoryEventSink(),
        reviewer=reviewer,
    ).run(
        task=task,
        asset=asset,
        policy=ExecutionPolicy(
            review_required=True,
            kappa=kappa,
        ),
        seed=17,
    )

    assert run.termination_reason == "completed"
    assert run.actions == (accepted, FinalAnswerAction(answer="42"))
    assert run.reviews[0].proposed_action == proposed
    assert run.reviews[0].accepted_action == accepted
    assert sandbox.requests[0].code == accepted.code
    assert inference.requests[1].observations == ("42\n",)
    assert reviewer.requests[0].guidance == ("Prefer exact arithmetic.",)
    assert reviewer.requests[0].asset == asset
    assert run.input_tokens == 34
    assert run.output_tokens == 10
    assert run.elapsed_ms == 17
    assert "hidden" not in run.model_dump_json()
    assert run.kappa_hash == kappa.content_hash


@pytest.mark.parametrize(
    ("reviewer", "max_review_calls", "expected"),
    [
        (None, 1, "reviewer_failed"),
        (DeterministicReviewer(results=()), 1, "reviewer_failed"),
        (DeterministicReviewer(results=()), 0, "reviewer_budget_exhausted"),
    ],
)
def test_reviewer_failure_and_budget_exhaustion_fail_closed(
    reviewer: DeterministicReviewer | None,
    max_review_calls: int,
    expected: str,
) -> None:
    task, asset = _fixture_input()
    sandbox = DeterministicSandbox(results={})
    run = TaskExecutor(
        inference=DeterministicInference(
            actions=(RunPythonAction(code="print('must not run')"),),
            model_id="fixture-student-v1",
        ),
        sandbox=sandbox,
        verifier=FixtureVerifier(),
        event_sink=InMemoryEventSink(),
        reviewer=reviewer,
    ).run(
        task=task,
        asset=asset,
        policy=ExecutionPolicy(
            review_required=True,
            max_review_calls=max_review_calls,
            kappa=_kappa("Review every action."),
        ),
        seed=17,
    )

    assert run.termination_reason == expected
    assert run.actions == ()
    assert sandbox.requests == []


def test_review_policy_requires_frozen_kappa_identity() -> None:
    with pytest.raises(ValueError, match="frozen Kappa"):
        ExecutionPolicy(review_required=True)

    with pytest.raises(ValueError, match="Reviewer-free"):
        ExecutionPolicy(kappa=_kappa("Should not be visible."))


def test_review_policy_accepts_only_one_validated_kappa_artifact() -> None:
    kappa = _kappa()
    policy = ExecutionPolicy(review_required=True, kappa=kappa)

    assert policy.kappa_hash == kappa.content_hash
    assert policy.kappa_guidance == kappa.guidance
    with pytest.raises(ValidationError):
        ExecutionPolicy.model_validate(
            {
                "review_required": True,
                "kappa_hash": kappa.content_hash,
                "kappa_guidance": ("Substituted guidance.",),
            }
        )


def _run_fixture() -> TaskRun:
    task, asset = _fixture_input()
    inference: DeterministicInference[StudentAction] = DeterministicInference(
        actions=(
            InspectImageAction(),
            CropAction(x0=0, y0=0, x1=50, y1=40),
            RunPythonAction(code="print(40 + 2)"),
            FinalAnswerAction(answer="42"),
        ),
        model_id="fixture-student-v1",
    )
    sandbox = DeterministicSandbox(
        results={
            "print(40 + 2)": SandboxResult(
                status="completed",
                stdout="42\n",
                stderr="",
                exit_code=0,
                latency_ms=2,
            )
        }
    )
    sink: InMemoryEventSink[RunEvent] = InMemoryEventSink()
    executor = TaskExecutor(
        inference=inference,
        sandbox=sandbox,
        verifier=FixtureVerifier(),
        event_sink=sink,
    )

    run = executor.run(
        task=task,
        asset=asset,
        policy=ExecutionPolicy(),
        seed=17,
    )
    assert run.events == sink.events
    assert sandbox.requests[0].code == "print(40 + 2)"
    assert sandbox.requests[0].timeout_ms == 10_000
    assert sandbox.requests[0].network_access is False
    return run


def _fixture_input() -> tuple[TaskInput, TaskAsset]:
    task = TaskInput(
        task_id="chartqa:fixture:001",
        question="What is the total?",
        image_asset_id="train/png/fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    asset = TaskAsset(
        task_id=task.task_id,
        media_type="image/png",
        image_sha256=task.image_sha256,
        content=b"synthetic-png",
    )
    return task, asset

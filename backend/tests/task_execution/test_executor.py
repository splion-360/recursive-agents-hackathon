"""Offline behavior tests for the minimal Student action loop."""

import pytest

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.inference import DeterministicInference
from memory_distillation.sandbox import DeterministicSandbox, SandboxResult
from memory_distillation.task_execution import (
    CropAction,
    ExecutionPolicy,
    FinalAnswerAction,
    InspectImageAction,
    RunEvent,
    RunPythonAction,
    StudentAction,
    TaskExecutor,
    TaskRun,
)
from memory_distillation.telemetry import InMemoryEventSink


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

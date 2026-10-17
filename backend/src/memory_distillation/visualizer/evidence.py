"""Sanitized synthetic evidence used by the read-only visualizer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from memory_distillation.task_execution.models import (
    ActionProposedEvent,
    ActionReview,
    ActionReviewedEvent,
    FinalAnswerAction,
    Observation,
    ObservationEvent,
    RunCompletedEvent,
    RunEvent,
    RunPythonAction,
    RunStartedEvent,
)


class RunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_id: str
    status: Literal["completed", "failed"]
    proposal: str
    review_decision: Literal["PASS", "REPLACE"]
    accepted_action: str
    tool_result: str
    eligible_for_consolidation: bool


class VersionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str
    role: Literal["ACCEPTED", "CANDIDATE", "REJECTED"]
    card_count: int
    content_hash: str


class PromotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["PROMOTE", "REJECT"]
    accepted_before: str
    candidate: str
    accepted_after: str
    capability_delta: float
    negative_transfer_rate: float
    failed_gates: tuple[str, ...]


class MetricEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Literal["C0", "C1", "C2", "C3"]
    accuracy: float
    intervention_rate: float
    mean_latency_ms: float
    mean_tokens: float


class ReplayEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["REPLAY"] = "REPLAY"
    synthetic: Literal[True] = True
    provenance: str
    events: tuple[RunEvent, ...]


def synthetic_evidence() -> tuple[
    tuple[RunEvidence, ...],
    tuple[VersionEvidence, ...],
    PromotionEvidence,
    tuple[MetricEvidence, ...],
    ReplayEvidence,
]:
    """Return a coherent fixture; these values are not measured experiment claims."""

    run = RunEvidence(
        run_id="synthetic-run-001",
        task_id="chartqa:synthetic:001",
        status="completed",
        proposal="Run Python over the extracted values without rechecking labels.",
        review_decision="REPLACE",
        accepted_action="Inspect the chart crop, then calculate the aggregate in OpenShell.",
        tool_result="OpenShell completed with a validated aggregate.",
        eligible_for_consolidation=True,
    )
    versions = (
        VersionEvidence(
            version_id="accepted-v000", role="ACCEPTED", card_count=0, content_hash="0" * 16
        ),
        VersionEvidence(
            version_id="candidate-demo-001",
            role="CANDIDATE",
            card_count=1,
            content_hash="8f41d2e7c6a905bd",
        ),
        VersionEvidence(
            version_id="candidate-rejected-000",
            role="REJECTED",
            card_count=2,
            content_hash="3a10b8f1ec2d4470",
        ),
    )
    promotion = PromotionEvidence(
        decision="PROMOTE",
        accepted_before="accepted-v000",
        candidate="candidate-demo-001",
        accepted_after="candidate-demo-001",
        capability_delta=0.10,
        negative_transfer_rate=0,
        failed_gates=(),
    )
    metrics = (
        MetricEvidence(
            condition="C0", accuracy=0.60, intervention_rate=0, mean_latency_ms=900, mean_tokens=420
        ),
        MetricEvidence(
            condition="C1",
            accuracy=0.65,
            intervention_rate=0.30,
            mean_latency_ms=1120,
            mean_tokens=510,
        ),
        MetricEvidence(
            condition="C2",
            accuracy=0.75,
            intervention_rate=0,
            mean_latency_ms=1080,
            mean_tokens=475,
        ),
        MetricEvidence(
            condition="C3",
            accuracy=0.62,
            intervention_rate=0,
            mean_latency_ms=1090,
            mean_tokens=480,
        ),
    )
    replay = ReplayEvidence(
        provenance="Deterministic fixture generated locally; no provider or benchmark result.",
        events=(
            RunStartedEvent(sequence=0, task_id=run.task_id, seed=17),
            ActionProposedEvent(
                sequence=1,
                task_id=run.task_id,
                action=RunPythonAction(code="print(sum(extracted_values))"),
                model_id="synthetic-student-v1",
                request_id="synthetic:student:1",
            ),
            ActionReviewedEvent(
                sequence=2,
                task_id=run.task_id,
                review=ActionReview(
                    decision="replace",
                    proposed_action=RunPythonAction(code="print(sum(extracted_values))"),
                    accepted_action=RunPythonAction(code="print(sum(verified_values))"),
                    model_id="synthetic-reviewer-v1",
                    request_id="synthetic:reviewer:1",
                    input_tokens=40,
                    output_tokens=12,
                    latency_ms=120,
                ),
            ),
            ObservationEvent(
                sequence=3,
                task_id=run.task_id,
                observation=Observation(
                    action_type="run_python", content="validated aggregate produced"
                ),
            ),
            ActionProposedEvent(
                sequence=4,
                task_id=run.task_id,
                action=FinalAnswerAction(answer="synthetic prediction"),
                model_id="synthetic-student-v1",
                request_id="synthetic:student:2",
            ),
            RunCompletedEvent(
                sequence=5,
                task_id=run.task_id,
                termination_reason="completed",
                correct=True,
            ),
        ),
    )
    return (run,), versions, promotion, metrics, replay

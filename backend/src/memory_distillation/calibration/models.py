"""Provider-independent contracts for one isolated H-star/Kappa calibration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from memory_distillation.benchmark import TaskInput
from memory_distillation.review_policy import CalibrationSplit, HStarProposal, KappaArtifact

if TYPE_CHECKING:
    from memory_distillation.task_execution.models import TaskRun


class DiscoveryObservation(BaseModel):
    """Answer-free discovery evidence exposed to the H-star analysis model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    question: str
    correct: bool
    termination_reason: str
    action_types: tuple[str, ...]


class HStarAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_hash: str
    discovery_task_ids: tuple[str, ...] = Field(min_length=8, max_length=8)
    observations: tuple[DiscoveryObservation, ...] = Field(min_length=8, max_length=8)


@dataclass(frozen=True, slots=True)
class HStarAnalysisResult:
    proposal: HStarProposal
    model_id: str
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    actual_cost_usd: Decimal


class HStarAnalyst(Protocol):
    def propose(self, request: HStarAnalysisRequest, *, seed: int) -> HStarAnalysisResult: ...


class CalibrationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Literal["discovery", "regression", "pilot"]
    condition: Literal["baseline", "candidate", "frozen_kappa"]
    task_id: str
    rule_id: str | None = None
    correct: bool
    termination_reason: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    review_calls: int = Field(ge=0)
    replacements: int = Field(ge=0)


class CalibrationProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal[
        "preflight",
        "discovery",
        "hstar_proposal",
        "regression",
        "kappa_freeze",
        "smoke_test",
        "complete",
        "failed",
    ]
    discovery_completed: int = Field(default=0, ge=0, le=8)
    regression_completed: int = Field(default=0, ge=0, le=8)
    smoke_completed: int = Field(default=0, ge=0, le=5)
    smoke_total: int = Field(default=0, ge=0, le=5)
    current_task_id: str | None = None
    proposal: HStarProposal | None = None
    proposal_hash: str | None = None
    kappa: KappaArtifact | None = None
    rejected_rule_ids: tuple[str, ...] = ()
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: Decimal = Field(default=Decimal(0), ge=0)
    message: str


class ModelIdentities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    student: str
    reviewer: str
    hstar: str


class CalibrationReport(BaseModel):
    """Frozen evidence needed before measured Collection may begin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["calibration-report.v1"] = "calibration-report.v1"
    run_id: str
    split: CalibrationSplit
    proposal: HStarProposal
    proposal_hash: str
    kappa: KappaArtifact
    fallback_reason: str | None
    models: ModelIdentities
    outcomes: tuple[CalibrationOutcome, ...]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_elapsed_ms: int = Field(ge=0)
    total_cost_usd: Decimal = Field(ge=0)


def discovery_observations(
    tasks: tuple[TaskInput, ...], runs: tuple[TaskRun, ...]
) -> tuple[DiscoveryObservation, ...]:
    """Project discovery runs without predictions, references, or private reasoning."""

    if tuple(task.task_id for task in tasks) != tuple(run.task_id for run in runs):
        raise ValueError("discovery tasks and baseline runs are misaligned")
    return tuple(
        DiscoveryObservation(
            task_id=task.task_id,
            question=task.question,
            correct=_require_correct(run),
            termination_reason=run.termination_reason,
            action_types=tuple(action.type for action in run.actions),
        )
        for task, run in zip(tasks, runs, strict=True)
    )


def outcome(
    run: TaskRun,
    *,
    phase: Literal["discovery", "regression", "pilot"],
    condition: Literal["baseline", "candidate", "frozen_kappa"],
    rule_id: str | None = None,
) -> CalibrationOutcome:
    return CalibrationOutcome(
        phase=phase,
        condition=condition,
        task_id=run.task_id,
        rule_id=rule_id,
        correct=_require_correct(run),
        termination_reason=run.termination_reason,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        elapsed_ms=run.elapsed_ms,
        review_calls=run.review_calls,
        replacements=sum(review.decision == "replace" for review in run.reviews),
    )


def _require_correct(run: TaskRun) -> bool:
    if run.termination_reason != "completed" or run.score is None:
        raise RuntimeError(
            f"calibration task did not complete cleanly: {run.task_id} ({run.termination_reason})"
        )
    return run.score.correct

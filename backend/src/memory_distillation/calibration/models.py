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
    round_index: int = Field(ge=1, le=3)
    discovery_task_ids: tuple[str, ...] = Field(min_length=4, max_length=12)
    observations: tuple[DiscoveryObservation, ...] = Field(min_length=4, max_length=12)
    prior_proposal: HStarProposal | None = None


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
    correct: bool | None
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
    discovery_completed: int = Field(default=0, ge=0, le=12)
    regression_completed: int = Field(default=0, ge=0, le=12)
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


class CalibrationProvenance(BaseModel):
    """Non-secret configuration required to reproduce a calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_revision: str
    seed: int
    execution_policy: dict[str, object]
    student_max_output_tokens: int = Field(gt=0)
    reviewer_max_output_tokens: int = Field(gt=0)
    hstar_max_output_tokens: int = Field(gt=0)
    sandbox_name: str
    inference_base_url: str


class PilotIntervention(BaseModel):
    """Public validity evidence for one validated pilot review decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    decision: Literal["pass", "replace"]
    proposed_action_type: str
    accepted_action_type: str
    accepted_action_valid: bool
    direct_answer_replacement: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class PilotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_count: int = Field(ge=3, le=5)
    baseline_correct: int = Field(ge=0)
    kappa_correct: int = Field(ge=0)
    capability_delta: int
    review_calls: int = Field(ge=0)
    replacements: int = Field(ge=0)
    invalid_interventions: int = Field(ge=0)


class AdaptationDecision(BaseModel):
    """Deterministic disposition of one H-star component into frozen Kappa."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str
    source_component: Literal["tools", "middleware", "skills", "failure_memory"]
    disposition: Literal["transform", "split", "merge", "drop"]
    output_rule_ids: tuple[str, ...]


class HStarRevision(BaseModel):
    """One immutable H-star proposal produced after a four-task round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(ge=1, le=3)
    task_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    proposal: HStarProposal
    proposal_hash: str


class CalibrationReport(BaseModel):
    """Frozen evidence needed before measured Collection may begin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["calibration-report.v2"] = "calibration-report.v2"
    run_id: str
    split: CalibrationSplit
    proposal: HStarProposal
    proposal_hash: str
    hstar_revisions: tuple[HStarRevision, ...] = ()
    kappa: KappaArtifact
    fallback_reason: str | None
    models: ModelIdentities
    provenance: CalibrationProvenance
    adaptation_map: tuple[AdaptationDecision, ...]
    outcomes: tuple[CalibrationOutcome, ...]
    pilot_interventions: tuple[PilotIntervention, ...]
    pilot_summary: PilotSummary
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
            correct=(
                run.termination_reason == "completed"
                and run.score is not None
                and run.score.correct
            ),
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
    require_complete: bool = True,
) -> CalibrationOutcome:
    return CalibrationOutcome(
        phase=phase,
        condition=condition,
        task_id=run.task_id,
        rule_id=rule_id,
        correct=(
            _require_correct(run)
            if require_complete
            else run.score.correct if run.score is not None else None
        ),
        termination_reason=run.termination_reason,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        elapsed_ms=run.elapsed_ms,
        review_calls=run.review_calls,
        replacements=sum(review.decision == "replace" for review in run.reviews),
    )


def _require_correct(run: TaskRun) -> bool:
    if run.termination_reason != "completed" or run.score is None:
        failure_kind = getattr(run.events[-1], "failure_kind", None) if run.events else None
        detail = f": {failure_kind}" if failure_kind else ""
        raise RuntimeError(
            "calibration task did not complete cleanly: "
            f"{run.task_id} ({run.termination_reason}{detail})"
        )
    return run.score.correct

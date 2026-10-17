"""Paired promotion metrics, deterministic gates, and atomic pointer updates."""

from dataclasses import dataclass
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.task_execution import ExecutionPolicy, TaskExecutor
from memory_distillation.versioning.models import (
    BehaviorCard,
    CandidateVersion,
    RetrievalPolicy,
    VersionView,
)


@dataclass(frozen=True, slots=True)
class PromotionTask:
    task: TaskInput
    asset: TaskAsset
    seed: int


class PairedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    seed: int
    configuration_hash: str
    accepted_version_id: str
    accepted_content_hash: str
    candidate_version_id: str
    candidate_content_hash: str
    accepted_correct: bool
    candidate_correct: bool
    accepted_latency_ms: int = Field(ge=0)
    candidate_latency_ms: int = Field(ge=0)
    accepted_input_tokens: int = Field(ge=0)
    candidate_input_tokens: int = Field(ge=0)
    accepted_valid: bool = True
    candidate_valid: bool = True


class PromotionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_capability_delta: float = 0.05
    maximum_negative_transfer_rate: float = Field(default=0.02, ge=0, le=1)
    maximum_latency_regression: float = Field(default=0.15, ge=0)
    maximum_token_regression: float = Field(default=0.15, ge=0)
    minimum_validity_rate: float = Field(default=1, ge=0, le=1)


class PromotionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paired_sample_size: int = Field(gt=0)
    capability_delta: float
    negative_transfer_rate: float
    latency_regression: float
    token_regression: float
    validity_rate: float


class PromotionAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    seed: int


class PromotionPlan(BaseModel):
    """Frozen identity of the complete paired Promotion execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_hash: str
    configuration_hash: str
    accepted_version_id: str
    accepted_content_hash: str
    candidate_version_id: str
    candidate_content_hash: str
    manifest_task_ids: tuple[str, ...]
    assignments: tuple[PromotionAssignment, ...]

    @model_validator(mode="after")
    def require_complete_unique_manifest(self) -> "PromotionPlan":
        manifest_ids = set(self.manifest_task_ids)
        assignment_ids = {item.task_id for item in self.assignments}
        if len(self.manifest_task_ids) != 30 or len(manifest_ids) != 30:
            raise ValueError("promotion manifest requires exactly 30 unique tasks")
        if len(self.assignments) != 30 or assignment_ids != manifest_ids:
            raise ValueError("promotion plan must assign every manifest task exactly once")
        return self


class PromotionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["PROMOTE", "REJECT"]
    accepted_version_before: str
    candidate_version_id: str
    accepted_version_after: str
    metrics: PromotionMetrics
    failed_gates: tuple[str, ...]

    @model_validator(mode="after")
    def require_consistent_pointer(self) -> "PromotionDecision":
        expected = (
            self.candidate_version_id
            if self.decision == "PROMOTE"
            else self.accepted_version_before
        )
        if self.accepted_version_after != expected:
            raise ValueError("promotion decision has inconsistent accepted pointer")
        return self


class VersionRegistry:
    """Deterministic memory adapter with immutable views and CAS promotion."""

    def __init__(self) -> None:
        bootstrap = VersionView(
            version_id="accepted-v000",
            cards=(),
            retrieval_policy=RetrievalPolicy(
                top_k=0,
                minimum_score=0,
                token_budget=0,
                calibration_snapshot_hash="0" * 64,
            ),
            content_hash="0" * 64,
        )
        self._versions = {bootstrap.version_id: bootstrap}
        self._accepted_version_id = bootstrap.version_id

    def freeze_candidate(self, candidate: CandidateVersion) -> VersionView:
        view = VersionView(
            version_id=candidate.version_id,
            cards=candidate.cards,
            retrieval_policy=candidate.retrieval_policy,
            content_hash=candidate.content_hash,
        )
        existing = self._versions.get(view.version_id)
        if existing is not None and existing != view:
            raise ValueError("version id already stores different content")
        self._versions[view.version_id] = view
        return view

    def get_view(self, version_id: str) -> VersionView:
        return self._versions[version_id]

    @property
    def accepted_version_id(self) -> str:
        return self._accepted_version_id

    def compare_and_set_accepted(self, *, expected: str, replacement: str) -> bool:
        if replacement not in self._versions:
            raise ValueError("replacement version is not frozen")
        if self._accepted_version_id != expected:
            return False
        self._accepted_version_id = replacement
        return True


def decide_promotion(
    *,
    registry: VersionRegistry,
    candidate: CandidateVersion,
    plan: PromotionPlan,
    outcomes: tuple[PairedOutcome, ...],
    thresholds: PromotionThresholds,
) -> PromotionDecision:
    """Evaluate explicit gates and atomically advance the Accepted Version."""

    accepted_before = registry.accepted_version_id
    if accepted_before != candidate.parent_version_id:
        raise ValueError("candidate parent is not the current Accepted Version")
    accepted_view = registry.get_view(accepted_before)
    if (
        plan.accepted_version_id != accepted_view.version_id
        or plan.accepted_content_hash != accepted_view.content_hash
        or plan.candidate_version_id != candidate.version_id
        or plan.candidate_content_hash != candidate.content_hash
    ):
        raise ValueError("promotion plan does not match the frozen version views")
    expected_assignments = {(item.task_id, item.seed) for item in plan.assignments}
    observed_assignments = {(item.task_id, item.seed) for item in outcomes}
    if len(outcomes) != 30 or observed_assignments != expected_assignments:
        raise ValueError("promotion outcomes do not cover the complete frozen plan")
    for item in outcomes:
        if (
            item.configuration_hash != plan.configuration_hash
            or item.accepted_version_id != plan.accepted_version_id
            or item.accepted_content_hash != plan.accepted_content_hash
            or item.candidate_version_id != plan.candidate_version_id
            or item.candidate_content_hash != plan.candidate_content_hash
        ):
            raise ValueError("promotion outcome provenance does not match the frozen plan")
    registry.freeze_candidate(candidate)
    size = len(outcomes)
    accepted_accuracy = sum(item.accepted_correct for item in outcomes) / size
    candidate_accuracy = sum(item.candidate_correct for item in outcomes) / size
    negative_transfer = (
        sum(item.accepted_correct and not item.candidate_correct for item in outcomes) / size
    )
    accepted_latency = float(median(item.accepted_latency_ms for item in outcomes))
    candidate_latency = float(median(item.candidate_latency_ms for item in outcomes))
    accepted_tokens = float(median(item.accepted_input_tokens for item in outcomes))
    candidate_tokens = float(median(item.candidate_input_tokens for item in outcomes))
    metrics = PromotionMetrics(
        paired_sample_size=size,
        capability_delta=candidate_accuracy - accepted_accuracy,
        negative_transfer_rate=negative_transfer,
        latency_regression=_relative_change(accepted_latency, candidate_latency),
        token_regression=_relative_change(accepted_tokens, candidate_tokens),
        validity_rate=sum(item.candidate_valid for item in outcomes) / size,
    )
    failed: list[str] = []
    if metrics.capability_delta < thresholds.minimum_capability_delta:
        failed.append("capability")
    if metrics.negative_transfer_rate > thresholds.maximum_negative_transfer_rate:
        failed.append("negative_transfer")
    if metrics.latency_regression > thresholds.maximum_latency_regression:
        failed.append("latency")
    if metrics.token_regression > thresholds.maximum_token_regression:
        failed.append("tokens")
    if metrics.validity_rate < thresholds.minimum_validity_rate:
        failed.append("validity")
    if failed:
        return PromotionDecision(
            decision="REJECT",
            accepted_version_before=accepted_before,
            candidate_version_id=candidate.version_id,
            accepted_version_after=accepted_before,
            metrics=metrics,
            failed_gates=tuple(failed),
        )
    if not registry.compare_and_set_accepted(
        expected=accepted_before, replacement=candidate.version_id
    ):
        raise RuntimeError("Accepted Version changed during promotion")
    return PromotionDecision(
        decision="PROMOTE",
        accepted_version_before=accepted_before,
        candidate_version_id=candidate.version_id,
        accepted_version_after=candidate.version_id,
        metrics=metrics,
        failed_gates=(),
    )


def run_paired_promotion(
    *,
    executor: TaskExecutor,
    plan: PromotionPlan,
    tasks: tuple[PromotionTask, ...],
    accepted_view: VersionView,
    candidate_view: VersionView,
    execution_policy: ExecutionPolicy,
) -> tuple[PairedOutcome, ...]:
    """Run the complete frozen Promotion plan through one TaskExecutor seam."""

    if execution_policy.review_required:
        raise ValueError("paired memory promotion must run without the online Reviewer")
    if (
        accepted_view.version_id != plan.accepted_version_id
        or accepted_view.content_hash != plan.accepted_content_hash
        or candidate_view.version_id != plan.candidate_version_id
        or candidate_view.content_hash != plan.candidate_content_hash
    ):
        raise ValueError("promotion views do not match the frozen plan")
    task_assignments = {(item.task.task_id, item.seed) for item in tasks}
    plan_assignments = {(item.task_id, item.seed) for item in plan.assignments}
    if len(tasks) != 30 or task_assignments != plan_assignments:
        raise ValueError("promotion tasks do not cover the complete frozen plan")
    outcomes: list[PairedOutcome] = []
    for item in sorted(tasks, key=lambda value: (value.task.task_id, value.seed)):
        accepted_run = executor.run(
            task=item.task,
            asset=item.asset,
            policy=execution_policy,
            seed=item.seed,
            behavior_cards=retrieve_behavior_cards(accepted_view, item.task),
        )
        candidate_run = executor.run(
            task=item.task,
            asset=item.asset,
            policy=execution_policy,
            seed=item.seed,
            behavior_cards=retrieve_behavior_cards(candidate_view, item.task),
        )
        outcomes.append(
            PairedOutcome(
                task_id=item.task.task_id,
                seed=item.seed,
                configuration_hash=plan.configuration_hash,
                accepted_version_id=accepted_view.version_id,
                accepted_content_hash=accepted_view.content_hash,
                candidate_version_id=candidate_view.version_id,
                candidate_content_hash=candidate_view.content_hash,
                accepted_correct=bool(accepted_run.score and accepted_run.score.correct),
                candidate_correct=bool(candidate_run.score and candidate_run.score.correct),
                accepted_latency_ms=accepted_run.elapsed_ms,
                candidate_latency_ms=candidate_run.elapsed_ms,
                accepted_input_tokens=accepted_run.input_tokens,
                candidate_input_tokens=candidate_run.input_tokens,
                accepted_valid=accepted_run.termination_reason == "completed",
                candidate_valid=candidate_run.termination_reason == "completed",
            )
        )
    return tuple(outcomes)


def retrieve_behavior_cards(view: VersionView, task: TaskInput) -> tuple[str, ...]:
    """Select only cards from one pinned immutable version view."""
    question_terms = set(task.question.casefold().split())
    matching: list[tuple[float, BehaviorCard]] = []
    for card in view.cards:
        trigger_terms = set(card.trigger.casefold().split())
        union = question_terms | trigger_terms
        score = len(question_terms & trigger_terms) / len(union) if union else 0
        if score >= view.retrieval_policy.minimum_score:
            matching.append((score, card))
    selected: list[str] = []
    used_tokens = 0
    for _, card in sorted(matching, key=lambda item: (-item[0], item[1].card_id)):
        if len(selected) >= view.retrieval_policy.top_k:
            break
        text = f"{card.trigger}\n{card.instruction}"
        token_estimate = len(text.split())
        if used_tokens + token_estimate > view.retrieval_policy.token_budget:
            continue
        selected.append(text)
        used_tokens += token_estimate
    return tuple(selected)


def _relative_change(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0 if candidate == 0 else float("inf")
    return (candidate - baseline) / baseline

"""Paired promotion metrics, deterministic gates, and atomic pointer updates."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_distillation.versioning.models import BehaviorCard, CandidateVersion


class VersionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str
    cards: tuple[BehaviorCard, ...]
    content_hash: str


class PairedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    seed: int
    configuration_hash: str
    accepted_correct: bool
    candidate_correct: bool
    accepted_latency_ms: int = Field(ge=0)
    candidate_latency_ms: int = Field(ge=0)
    accepted_tokens: int = Field(ge=0)
    candidate_tokens: int = Field(ge=0)
    accepted_valid: bool = True
    candidate_valid: bool = True


class PromotionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_capability_delta: float = 0
    maximum_negative_transfer_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_latency_regression: float = Field(default=0.2, ge=0)
    maximum_token_regression: float = Field(default=0.2, ge=0)
    minimum_validity_rate: float = Field(default=1, ge=0, le=1)


class PromotionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paired_sample_size: int = Field(gt=0)
    capability_delta: float
    negative_transfer_rate: float
    latency_regression: float
    token_regression: float
    validity_rate: float


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
        bootstrap = VersionView(version_id="accepted-v000", cards=(), content_hash="0" * 64)
        self._versions = {bootstrap.version_id: bootstrap}
        self._accepted_version_id = bootstrap.version_id

    def freeze_candidate(self, candidate: CandidateVersion) -> VersionView:
        view = VersionView(
            version_id=candidate.version_id,
            cards=candidate.cards,
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
    outcomes: tuple[PairedOutcome, ...],
    thresholds: PromotionThresholds,
) -> PromotionDecision:
    """Evaluate explicit gates and atomically advance the Accepted Version."""

    if not outcomes:
        raise ValueError("promotion requires paired outcomes")
    if len({(item.task_id, item.seed, item.configuration_hash) for item in outcomes}) != len(
        outcomes
    ):
        raise ValueError("paired outcomes contain duplicate task assignments")
    accepted_before = registry.accepted_version_id
    if accepted_before != candidate.parent_version_id:
        raise ValueError("candidate parent is not the current Accepted Version")
    registry.freeze_candidate(candidate)
    size = len(outcomes)
    accepted_accuracy = sum(item.accepted_correct for item in outcomes) / size
    candidate_accuracy = sum(item.candidate_correct for item in outcomes) / size
    negative_transfer = (
        sum(item.accepted_correct and not item.candidate_correct for item in outcomes) / size
    )
    accepted_latency = sum(item.accepted_latency_ms for item in outcomes) / size
    candidate_latency = sum(item.candidate_latency_ms for item in outcomes) / size
    accepted_tokens = sum(item.accepted_tokens for item in outcomes) / size
    candidate_tokens = sum(item.candidate_tokens for item in outcomes) / size
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


def _relative_change(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0 if candidate == 0 else float("inf")
    return (candidate - baseline) / baseline

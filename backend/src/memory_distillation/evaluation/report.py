"""Deterministic paired C0-C3 experiment reporting."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_distillation.task_execution import ExecutionPolicy

Condition = Literal["C0", "C1", "C2", "C3"]


class ConditionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Condition
    reviewer_enabled: bool
    memory_view: Literal["none", "accepted", "irrelevant"]
    execution: ExecutionPolicy
    configuration_hash: str


class ConditionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Condition
    task_id: str
    seed: int
    correct: bool
    reviewer_intervened: bool
    latency_ms: int = Field(ge=0)
    tokens: int = Field(ge=0)
    valid: bool = True


class ConditionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Condition
    sample_size: int
    accuracy: float
    intervention_rate: float
    mean_latency_ms: float
    mean_tokens: float
    validity_rate: float


class PairedContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Condition
    baseline: Literal["C0"] = "C0"
    paired_sample_size: int
    capability_delta: float
    capability_ci95: tuple[float, float]
    negative_transfer_rate: float
    latency_effect_ms: float
    token_effect: float


class C3Semantics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["AVAILABLE", "UNAVAILABLE", "DEGENERATE"]
    assignment_seed: int
    reason: str


class ExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["experiment-report.v1"] = "experiment-report.v1"
    manifest_hash: str
    policies: tuple[ConditionPolicy, ...]
    metrics: tuple[ConditionMetrics, ...]
    contrasts: tuple[PairedContrast, ...]
    c3: C3Semantics

    @model_validator(mode="after")
    def require_all_conditions(self) -> "ExperimentReport":
        if {item.condition for item in self.policies} != {"C0", "C1", "C2", "C3"}:
            raise ValueError("report requires one policy for every C0-C3 condition")
        return self


def frozen_condition_policies(
    *, configuration_hash: str, execution: ExecutionPolicy
) -> tuple[ConditionPolicy, ...]:
    """Declare the only allowed experimental differences."""

    return (
        ConditionPolicy(
            condition="C0",
            reviewer_enabled=False,
            memory_view="none",
            execution=execution.model_copy(update={"review_required": False}),
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C1",
            reviewer_enabled=True,
            memory_view="none",
            execution=execution.model_copy(update={"review_required": True}),
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C2",
            reviewer_enabled=True,
            memory_view="accepted",
            execution=execution.model_copy(update={"review_required": True}),
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C3",
            reviewer_enabled=True,
            memory_view="irrelevant",
            execution=execution.model_copy(update={"review_required": True}),
            configuration_hash=configuration_hash,
        ),
    )


def build_experiment_report(
    *,
    manifest_hash: str,
    policies: tuple[ConditionPolicy, ...],
    results: tuple[ConditionResult, ...],
    c3: C3Semantics,
) -> ExperimentReport:
    """Aggregate paired evidence without receiving any writable memory dependency."""

    grouped: dict[Condition, list[ConditionResult]] = {
        condition: [] for condition in ("C0", "C1", "C2", "C3")
    }
    for result in results:
        grouped[result.condition].append(result)
    baseline_keys = {(item.task_id, item.seed) for item in grouped["C0"]}
    if not baseline_keys:
        raise ValueError("C0 cannot be empty")
    for condition, items in grouped.items():
        keys = {(item.task_id, item.seed) for item in items}
        if condition == "C3" and c3.status != "AVAILABLE" and not items:
            continue
        if keys != baseline_keys:
            raise ValueError("conditions must use identical paired task assignments")
    metrics = tuple(
        _condition_metrics(condition, tuple(grouped[condition]))
        for condition in ("C0", "C1", "C2", "C3")
        if grouped[condition]
    )
    baseline = {(item.task_id, item.seed): item for item in grouped["C0"]}
    contrasts = tuple(
        _contrast(condition, baseline, tuple(grouped[condition]))
        for condition in ("C1", "C2", "C3")
        if grouped[condition]
    )
    return ExperimentReport(
        manifest_hash=manifest_hash,
        policies=policies,
        metrics=metrics,
        contrasts=contrasts,
        c3=c3,
    )


def _condition_metrics(
    condition: Condition, items: tuple[ConditionResult, ...]
) -> ConditionMetrics:
    size = len(items)
    return ConditionMetrics(
        condition=condition,
        sample_size=size,
        accuracy=sum(item.correct for item in items) / size,
        intervention_rate=sum(item.reviewer_intervened for item in items) / size,
        mean_latency_ms=sum(item.latency_ms for item in items) / size,
        mean_tokens=sum(item.tokens for item in items) / size,
        validity_rate=sum(item.valid for item in items) / size,
    )


def _contrast(
    condition: Condition,
    baseline: dict[tuple[str, int], ConditionResult],
    items: tuple[ConditionResult, ...],
) -> PairedContrast:
    ordered = sorted(items, key=lambda item: (item.task_id, item.seed))
    differences = [
        int(item.correct) - int(baseline[(item.task_id, item.seed)].correct) for item in ordered
    ]
    size = len(differences)
    mean = sum(differences) / size
    variance = sum((value - mean) ** 2 for value in differences) / (size - 1) if size > 1 else 0
    margin = 1.96 * math.sqrt(variance / size)
    return PairedContrast(
        condition=condition,
        paired_sample_size=size,
        capability_delta=mean,
        capability_ci95=(max(-1, mean - margin), min(1, mean + margin)),
        negative_transfer_rate=sum(
            baseline[(item.task_id, item.seed)].correct and not item.correct for item in ordered
        )
        / size,
        latency_effect_ms=sum(
            item.latency_ms - baseline[(item.task_id, item.seed)].latency_ms for item in ordered
        )
        / size,
        token_effect=sum(
            item.tokens - baseline[(item.task_id, item.seed)].tokens for item in ordered
        )
        / size,
    )

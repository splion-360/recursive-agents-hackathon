"""Deterministic paired C0-C3 experiment reporting."""

import hashlib
import itertools
import math
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_distillation.review_policy import KappaArtifact
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
    mcnemar_exact_p: float


class C3Semantics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["AVAILABLE", "UNAVAILABLE", "DEGENERATE"]
    assignment_seed: int
    reason: str


class IrrelevantDonorCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    trigger_terms: tuple[str, ...]
    token_count: int = Field(gt=0)


class C3Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    card_ids: tuple[str, ...]
    token_count: int


class C2RetrievalTarget(BaseModel):
    """Observed C2 retrieval shape that one C3 assignment must match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card_count: int = Field(ge=0)
    token_count: int = Field(ge=0)


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
        if len(self.policies) != 4 or {item.condition for item in self.policies} != {
            "C0",
            "C1",
            "C2",
            "C3",
        }:
            raise ValueError("report requires one policy for every C0-C3 condition")
        return self


def frozen_condition_policies(
    *,
    configuration_hash: str,
    execution: ExecutionPolicy,
    kappa: KappaArtifact,
) -> tuple[ConditionPolicy, ...]:
    """Declare the only allowed experimental differences."""

    unreviewed = ExecutionPolicy.model_validate(
        execution.model_dump()
        | {"review_required": False, "kappa_hash": None, "kappa_guidance": ()}
    )
    reviewed = ExecutionPolicy.model_validate(
        execution.model_dump()
        | {
            "review_required": True,
            "kappa_hash": kappa.content_hash,
            "kappa_guidance": kappa.guidance,
        }
    )
    return (
        ConditionPolicy(
            condition="C0",
            reviewer_enabled=False,
            memory_view="none",
            execution=unreviewed,
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C1",
            reviewer_enabled=True,
            memory_view="none",
            execution=reviewed,
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C2",
            reviewer_enabled=False,
            memory_view="accepted",
            execution=unreviewed,
            configuration_hash=configuration_hash,
        ),
        ConditionPolicy(
            condition="C3",
            reviewer_enabled=False,
            memory_view="irrelevant",
            execution=unreviewed,
            configuration_hash=configuration_hash,
        ),
    )


def validate_condition_policies(policies: tuple[ConditionPolicy, ...]) -> None:
    """Reject any condition set that changes more than Reviewer and memory access."""

    expected = {
        "C0": (False, "none"),
        "C1": (True, "none"),
        "C2": (False, "accepted"),
        "C3": (False, "irrelevant"),
    }
    if len(policies) != 4 or {item.condition for item in policies} != set(expected):
        raise ValueError("exactly one frozen policy is required per condition")
    if len({item.configuration_hash for item in policies}) != 1:
        raise ValueError("all conditions must use one frozen configuration")
    budget_fields = (
        "max_actions",
        "max_tool_calls",
        "max_python_calls",
        "max_total_tokens",
        "max_elapsed_ms",
        "sandbox_timeout_ms",
        "max_crop_coordinate",
    )
    baseline = policies[0].execution
    for policy in policies:
        if (policy.reviewer_enabled, policy.memory_view) != expected[policy.condition]:
            raise ValueError(f"invalid frozen policy for {policy.condition}")
        if policy.execution.review_required != policy.reviewer_enabled:
            raise ValueError("Reviewer flag and execution policy disagree")
        if policy.condition == "C1":
            if policy.execution.kappa_hash is None or not policy.execution.kappa_guidance:
                raise ValueError("C1 must bind frozen Kappa identity and guidance")
        elif policy.execution.kappa_hash is not None or policy.execution.kappa_guidance:
            raise ValueError("only C1 may receive Kappa guidance")
        if any(
            getattr(policy.execution, field) != getattr(baseline, field) for field in budget_fields
        ):
            raise ValueError("all conditions must use identical execution budgets")


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
    validate_condition_policies(policies)
    row_keys = [(item.condition, item.task_id, item.seed) for item in results]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("duplicate condition result rows are forbidden")
    for result in results:
        grouped[result.condition].append(result)
    baseline_keys = {(item.task_id, item.seed) for item in grouped["C0"]}
    if not baseline_keys:
        raise ValueError("C0 cannot be empty")
    for condition, items in grouped.items():
        keys = {(item.task_id, item.seed) for item in items}
        if condition == "C3":
            if not keys and c3.status in {"UNAVAILABLE", "DEGENERATE"}:
                continue
            if c3.status != "AVAILABLE" or not keys <= baseline_keys:
                raise ValueError("C3 results do not match declared availability")
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
    ci = _paired_bootstrap_interval(differences, condition)
    baseline_wins = sum(
        baseline[(item.task_id, item.seed)].correct and not item.correct for item in ordered
    )
    condition_wins = sum(
        not baseline[(item.task_id, item.seed)].correct and item.correct for item in ordered
    )
    return PairedContrast(
        condition=condition,
        paired_sample_size=size,
        capability_delta=mean,
        capability_ci95=ci,
        negative_transfer_rate=baseline_wins / size,
        latency_effect_ms=sum(
            item.latency_ms - baseline[(item.task_id, item.seed)].latency_ms for item in ordered
        )
        / size,
        token_effect=sum(
            item.tokens - baseline[(item.task_id, item.seed)].tokens for item in ordered
        )
        / size,
        mcnemar_exact_p=_mcnemar_exact(baseline_wins, condition_wins),
    )


def assign_irrelevant_memory(
    *,
    task_terms: dict[str, tuple[str, ...]],
    donor_cards: tuple[IrrelevantDonorCard, ...],
    c2_targets: dict[str, C2RetrievalTarget],
    assignment_seed: int,
) -> tuple[C3Semantics, tuple[C3Assignment, ...]]:
    """Freeze trigger-mismatched, count- and token-matched C3 assignments."""

    if set(c2_targets) != set(task_terms):
        raise ValueError("C2 targets must cover every task exactly once")
    if all(target.card_count <= 0 for target in c2_targets.values()):
        return (
            C3Semantics(
                status="DEGENERATE",
                assignment_seed=assignment_seed,
                reason="Accepted Version has no cards",
            ),
            (),
        )
    assignments: list[C3Assignment] = []
    for task_id, terms in sorted(task_terms.items()):
        target = c2_targets[task_id]
        if target.card_count == 0:
            continue
        normalized_terms = {term.casefold() for term in terms}
        eligible = tuple(
            card
            for card in donor_cards
            if normalized_terms.isdisjoint(term.casefold() for term in card.trigger_terms)
        )
        choices = []
        for cards in itertools.combinations(eligible, target.card_count):
            token_count = sum(card.token_count for card in cards)
            if target.token_count > 0 and abs(token_count - target.token_count) <= (
                target.token_count * 0.10
            ):
                assignment_key = (
                    f"{assignment_seed}:{task_id}:{','.join(card.card_id for card in cards)}"
                )
                digest = hashlib.sha256(assignment_key.encode()).hexdigest()
                choices.append((digest, cards, token_count))
        if choices:
            _, selected, token_count = min(choices, key=lambda item: item[0])
            assignments.append(
                C3Assignment(
                    task_id=task_id,
                    card_ids=tuple(card.card_id for card in selected),
                    token_count=token_count,
                )
            )
    if not assignments:
        return (
            C3Semantics(
                status="UNAVAILABLE",
                assignment_seed=assignment_seed,
                reason="No honest trigger-mismatched donor assignment exists",
            ),
            (),
        )
    return (
        C3Semantics(
            status="AVAILABLE",
            assignment_seed=assignment_seed,
            reason=f"{len(assignments)} tasks have frozen irrelevant-memory assignments",
        ),
        tuple(assignments),
    )


def _paired_bootstrap_interval(differences: list[int], condition: Condition) -> tuple[float, float]:
    if len(differences) == 1:
        value = float(differences[0])
        return value, value
    seed_offset = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}[condition]
    generator = random.Random(20261016 + seed_offset)
    estimates = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(4_000)
    )
    return estimates[99], estimates[3899]


def _mcnemar_exact(baseline_wins: int, condition_wins: int) -> float:
    discordant = baseline_wins + condition_wins
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * 0.5**discordant
        for index in range(min(baseline_wins, condition_wins) + 1)
    )
    return min(1.0, 2 * tail)

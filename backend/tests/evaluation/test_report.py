import pytest

from memory_distillation.evaluation import (
    C3Semantics,
    ConditionResult,
    build_experiment_report,
    frozen_condition_policies,
)
from memory_distillation.task_execution import ExecutionPolicy


def _result(condition: str, task_id: str, correct: bool, *, intervention: bool) -> ConditionResult:
    return ConditionResult.model_validate(
        {
            "condition": condition,
            "task_id": task_id,
            "seed": 17,
            "correct": correct,
            "reviewer_intervened": intervention,
            "latency_ms": 100 if condition == "C0" else 110,
            "tokens": 100 if condition == "C0" else 105,
        }
    )


def test_c0_c3_report_is_paired_deterministic_and_complete() -> None:
    policies = frozen_condition_policies(configuration_hash="frozen", execution=ExecutionPolicy())
    results = tuple(
        _result(condition, task_id, correct=condition != "C0", intervention=condition != "C0")
        for condition in ("C0", "C1", "C2", "C3")
        for task_id in ("t1", "t2")
    )
    semantics = C3Semantics(status="AVAILABLE", assignment_seed=99, reason="matched donor pool")

    first = build_experiment_report(
        manifest_hash="final", policies=policies, results=results, c3=semantics
    )
    second = build_experiment_report(
        manifest_hash="final", policies=policies, results=results, c3=semantics
    )

    assert first == second
    assert first.contrasts[1].condition == "C2"
    assert first.contrasts[1].paired_sample_size == 2
    assert first.contrasts[1].capability_delta == 1
    assert first.contrasts[1].negative_transfer_rate == 0


def test_unavailable_c3_is_reported_without_fabricated_rows() -> None:
    policies = frozen_condition_policies(configuration_hash="frozen", execution=ExecutionPolicy())
    results = tuple(
        _result(condition, "t1", correct=True, intervention=condition != "C0")
        for condition in ("C0", "C1", "C2")
    )
    report = build_experiment_report(
        manifest_hash="final",
        policies=policies,
        results=results,
        c3=C3Semantics(
            status="UNAVAILABLE", assignment_seed=99, reason="no irrelevant donor cards"
        ),
    )
    assert report.c3.status == "UNAVAILABLE"
    assert {item.condition for item in report.metrics} == {"C0", "C1", "C2"}


def test_unpaired_assignments_fail_closed() -> None:
    policies = frozen_condition_policies(configuration_hash="frozen", execution=ExecutionPolicy())
    with pytest.raises(ValueError, match="identical paired"):
        build_experiment_report(
            manifest_hash="final",
            policies=policies,
            results=(
                _result("C0", "t1", True, intervention=False),
                _result("C1", "different", True, intervention=True),
            ),
            c3=C3Semantics(status="UNAVAILABLE", assignment_seed=1, reason="none"),
        )

import pytest

from memory_distillation.evaluation import (
    C2RetrievalTarget,
    C3Semantics,
    ConditionPolicy,
    ConditionResult,
    IrrelevantDonorCard,
    assign_irrelevant_memory,
    build_experiment_report,
    frozen_condition_policies,
)
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.task_execution import ExecutionPolicy


def _kappa() -> KappaArtifact:
    return KappaArtifact.model_construct(
        content_hash="a" * 64,
        guidance=("Inspect relevant chart labels before computing.",),
    )


def _policies() -> tuple[ConditionPolicy, ...]:
    return frozen_condition_policies(
        configuration_hash="frozen",
        execution=ExecutionPolicy(),
        kappa=_kappa(),
    )


def _result(condition: str, task_id: str, correct: bool, *, intervention: bool) -> ConditionResult:
    return ConditionResult.model_validate(
        {
            "condition": condition,
            "task_id": task_id,
            "seed": 17,
            "correct": correct,
            "reviewer_intervened": intervention,
            "latency_ms": 100 if condition == "C0" else 110,
            "input_tokens": 80 if condition == "C0" else 85,
            "output_tokens": 20,
            "model_calls": 1,
            "tool_calls": 0,
            "review_calls": 1 if condition == "C1" else 0,
            "retrieved_card_count": 1 if condition in {"C2", "C3"} else 0,
        }
    )


def test_c0_c3_report_is_paired_deterministic_and_complete() -> None:
    policies = _policies()
    results = tuple(
        _result(condition, task_id, correct=condition != "C0", intervention=condition == "C1")
        for condition in ("C0", "C1", "C2", "C3")
        for task_id in ("t1", "t2")
    )
    semantics = C3Semantics(status="AVAILABLE", assignment_seed=99, reason="matched donor pool")

    first = build_experiment_report(
        manifest_hash="final", policies=policies, results=results, c3=semantics, kappa=_kappa()
    )
    second = build_experiment_report(
        manifest_hash="final", policies=policies, results=results, c3=semantics, kappa=_kappa()
    )

    assert first == second
    assert first.contrasts[1].condition == "C2"
    assert first.contrasts[1].paired_sample_size == 2
    assert first.contrasts[1].capability_delta == 1
    assert first.contrasts[1].negative_transfer_rate == 0
    assert first.contrasts[1].mcnemar_exact_p == 0.5
    assert {policy.condition: policy.reviewer_enabled for policy in policies} == {
        "C0": False,
        "C1": True,
        "C2": False,
        "C3": False,
    }


def test_unavailable_c3_is_reported_without_fabricated_rows() -> None:
    policies = _policies()
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
        kappa=_kappa(),
    )
    assert report.c3.status == "UNAVAILABLE"
    assert {item.condition for item in report.metrics} == {"C0", "C1", "C2"}


def test_unpaired_assignments_fail_closed() -> None:
    policies = _policies()
    with pytest.raises(ValueError, match="identical paired"):
        build_experiment_report(
            manifest_hash="final",
            policies=policies,
            results=(
                _result("C0", "t1", True, intervention=False),
                _result("C1", "different", True, intervention=True),
            ),
            c3=C3Semantics(status="UNAVAILABLE", assignment_seed=1, reason="none"),
            kappa=_kappa(),
        )


def test_condition_policies_reject_budget_or_configuration_drift() -> None:
    policies = _policies()
    drifted_budget = policies[2].model_copy(
        update={"execution": policies[2].execution.model_copy(update={"max_actions": 100})}
    )
    with pytest.raises(ValueError, match="identical execution budgets"):
        build_experiment_report(
            manifest_hash="final",
            policies=(*policies[:2], drifted_budget, policies[3]),
            results=tuple(
                _result(condition, "t1", True, intervention=condition == "C1")
                for condition in ("C0", "C1", "C2", "C3")
            ),
            c3=C3Semantics(status="AVAILABLE", assignment_seed=1, reason="matched"),
            kappa=_kappa(),
        )
    drifted_config: ConditionPolicy = policies[3].model_copy(
        update={"configuration_hash": "different"}
    )
    with pytest.raises(ValueError, match="one frozen configuration"):
        build_experiment_report(
            manifest_hash="final",
            policies=(*policies[:3], drifted_config),
            results=tuple(
                _result(condition, "t1", True, intervention=condition == "C1")
                for condition in ("C0", "C1", "C2", "C3")
            ),
            c3=C3Semantics(status="AVAILABLE", assignment_seed=1, reason="matched"),
            kappa=_kappa(),
        )

    wrong_kappa = KappaArtifact.model_construct(
        content_hash="b" * 64,
        guidance=("Different guidance.",),
    )
    with pytest.raises(ValueError, match="exact frozen Kappa"):
        build_experiment_report(
            manifest_hash="final",
            policies=policies,
            results=tuple(
                _result(condition, "t1", True, intervention=condition == "C1")
                for condition in ("C0", "C1", "C2", "C3")
            ),
            c3=C3Semantics(status="AVAILABLE", assignment_seed=1, reason="matched"),
            kappa=wrong_kappa,
        )


def test_duplicate_rows_fail_closed_and_partial_c3_is_honest() -> None:
    policies = _policies()
    rows = tuple(
        _result(condition, task_id, True, intervention=condition == "C1")
        for condition in ("C0", "C1", "C2")
        for task_id in ("t1", "t2")
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_experiment_report(
            manifest_hash="final",
            policies=policies,
            results=(*rows, rows[-1]),
            c3=C3Semantics(status="UNAVAILABLE", assignment_seed=1, reason="none"),
            kappa=_kappa(),
        )

    report = build_experiment_report(
        manifest_hash="final",
        policies=policies,
        results=(*rows, _result("C3", "t1", True, intervention=False)),
        c3=C3Semantics(status="AVAILABLE", assignment_seed=1, reason="one honest match"),
        kappa=_kappa(),
    )
    assert report.metrics[-1].condition == "C3"
    assert report.metrics[-1].sample_size == 1


def test_irrelevant_memory_assignment_is_seeded_and_trigger_mismatched() -> None:
    donors = (
        IrrelevantDonorCard(card_id="weather", trigger_terms=("weather",), token_count=48),
        IrrelevantDonorCard(card_id="finance", trigger_terms=("finance",), token_count=52),
        IrrelevantDonorCard(card_id="chart", trigger_terms=("chart",), token_count=50),
    )
    first = assign_irrelevant_memory(
        task_terms={"t1": ("chart",)},
        donor_cards=donors,
        c2_targets={"t1": C2RetrievalTarget(card_count=2, token_count=100)},
        assignment_seed=20261016,
    )
    second = assign_irrelevant_memory(
        task_terms={"t1": ("chart",)},
        donor_cards=donors,
        c2_targets={"t1": C2RetrievalTarget(card_count=2, token_count=100)},
        assignment_seed=20261016,
    )
    assert first == second
    assert first[0].status == "AVAILABLE"
    assert first[1][0].card_ids == ("weather", "finance")


def test_irrelevant_memory_matches_each_tasks_c2_shape() -> None:
    donors = (
        IrrelevantDonorCard(card_id="a", trigger_terms=("weather",), token_count=50),
        IrrelevantDonorCard(card_id="b", trigger_terms=("finance",), token_count=50),
    )
    semantics, assignments = assign_irrelevant_memory(
        task_terms={"one": ("chart",), "two": ("chart",)},
        donor_cards=donors,
        c2_targets={
            "one": C2RetrievalTarget(card_count=1, token_count=50),
            "two": C2RetrievalTarget(card_count=2, token_count=100),
        },
        assignment_seed=17,
    )

    assert semantics.status == "AVAILABLE"
    assert {item.task_id: len(item.card_ids) for item in assignments} == {
        "one": 1,
        "two": 2,
    }

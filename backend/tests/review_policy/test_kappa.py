import pytest
from pydantic import ValidationError

from memory_distillation.review_policy import (
    CalibrationSplit,
    HStarProposal,
    RuleEvidence,
    freeze_kappa,
)


def test_kappa_is_deterministic_frozen_and_regression_screened() -> None:
    split = CalibrationSplit(
        manifest_hash="calibration-hash",
        discovery_task_ids=("d1", "d2"),
        regression_task_ids=("r1", "r2"),
    )
    proposal = HStarProposal(
        proposal_id="h-star-1",
        rules=("Verify extracted labels before arithmetic.", "Always replace the answer."),
        discovery_task_ids=("d1", "d2"),
    )
    evidence = (
        RuleEvidence(
            rule=proposal.rules[0], discovery_helped=2, regression_helped=1, regression_harmed=0
        ),
        RuleEvidence(
            rule=proposal.rules[1], discovery_helped=1, regression_helped=0, regression_harmed=1
        ),
    )

    first = freeze_kappa(proposal=proposal, split=split, evidence=evidence)
    second = freeze_kappa(proposal=proposal, split=split, evidence=evidence)

    assert first == second
    assert first.guidance == (proposal.rules[0],)
    assert len(first.content_hash) == 64
    with pytest.raises(ValidationError):
        first.guidance = ()


def test_calibration_isolation_and_privacy_fail_closed() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        CalibrationSplit(
            manifest_hash="hash", discovery_task_ids=("same",), regression_task_ids=("same",)
        )

    split = CalibrationSplit(
        manifest_hash="hash", discovery_task_ids=("d1",), regression_task_ids=("r1",)
    )
    proposal = HStarProposal(
        proposal_id="leak", rules=("The correct answer is 42.",), discovery_task_ids=("d1",)
    )
    evidence = (
        RuleEvidence(
            rule=proposal.rules[0],
            discovery_helped=1,
            regression_helped=1,
            regression_harmed=0,
        ),
    )
    with pytest.raises(ValueError, match=r"leakage|answer"):
        freeze_kappa(proposal=proposal, split=split, evidence=evidence, forbidden_answers=("42",))

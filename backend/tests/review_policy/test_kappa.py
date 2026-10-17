import pytest
from pydantic import ValidationError

from memory_distillation.review_policy import (
    CalibrationExecution,
    CalibrationPrivacyContext,
    CalibrationSplit,
    HStarProposal,
    KappaCalibrationSession,
    KappaRule,
    freeze_kappa,
    hash_hstar_proposal,
)


def _split() -> CalibrationSplit:
    return CalibrationSplit(
        manifest_hash="calibration-hash",
        discovery_task_ids=tuple(f"d{index}" for index in range(8)),
        regression_task_ids=tuple(f"r{index}" for index in range(8)),
    )


def _executions(
    split: CalibrationSplit,
    rule: KappaRule,
    *,
    discovery_helped: int,
    regression_helped: int,
    regression_harmed: int,
) -> tuple[CalibrationExecution, ...]:
    proposal = HStarProposal(
        proposal_id="execution-binding",
        rules=(rule,),
        discovery_task_ids=split.discovery_task_ids,
    )
    return _executions_for_proposal(
        split,
        proposal,
        rule,
        discovery_helped=discovery_helped,
        regression_helped=regression_helped,
        regression_harmed=regression_harmed,
    )


def _executions_for_proposal(
    split: CalibrationSplit,
    proposal: HStarProposal,
    rule: KappaRule,
    *,
    discovery_helped: int,
    regression_helped: int,
    regression_harmed: int,
) -> tuple[CalibrationExecution, ...]:
    rows: list[CalibrationExecution] = []
    for index, task_id in enumerate(split.discovery_task_ids):
        rows.append(
            CalibrationExecution(
                execution_id=f"{rule.rule_id}:d:{index}",
                proposal_hash=hash_hstar_proposal(proposal),
                phase="discovery",
                task_id=task_id,
                rule_id=rule.rule_id,
                action_type=rule.action_types[0],
                baseline_correct=False,
                candidate_correct=index < discovery_helped,
            )
        )
    for index, task_id in enumerate(split.regression_task_ids):
        helped = index < regression_helped
        harmed = regression_helped <= index < regression_helped + regression_harmed
        rows.append(
            CalibrationExecution(
                execution_id=f"{rule.rule_id}:r:{index}",
                proposal_hash=hash_hstar_proposal(proposal),
                phase="regression",
                task_id=task_id,
                rule_id=rule.rule_id,
                action_type=rule.action_types[0],
                baseline_correct=harmed,
                candidate_correct=helped,
            )
        )
    return tuple(rows)


def test_kappa_is_deterministic_frozen_and_regression_screened() -> None:
    split = _split()
    accepted_rule = KappaRule(
        rule_id="verify-labels",
        action_types=("inspect_image", "crop"),
        instruction="Verify extracted labels before arithmetic.",
    )
    rejected_rule = KappaRule(
        rule_id="always-replace",
        action_types=("final_answer",),
        instruction="Always replace the final answer.",
    )
    proposal = HStarProposal(
        proposal_id="h-star-1",
        rules=(accepted_rule, rejected_rule),
        discovery_task_ids=split.discovery_task_ids,
    )
    executions = (
        *_executions_for_proposal(
            split,
            proposal,
            accepted_rule,
            discovery_helped=2,
            regression_helped=1,
            regression_harmed=0,
        ),
        *_executions_for_proposal(
            split,
            proposal,
            rejected_rule,
            discovery_helped=1,
            regression_helped=0,
            regression_harmed=1,
        ),
    )
    privacy = CalibrationPrivacyContext(forbidden_phrases=("fixture.png", "42"))

    session = KappaCalibrationSession(split)
    first = freeze_kappa(session=session, proposal=proposal, executions=executions, privacy=privacy)
    second = freeze_kappa(
        session=session, proposal=proposal, executions=executions, privacy=privacy
    )

    assert first == second
    assert first.rules == (accepted_rule,)
    assert first.guidance == (accepted_rule.instruction,)
    assert len(first.content_hash) == 64
    with pytest.raises(ValidationError):
        first.guidance = ()


def test_calibration_isolation_evidence_and_privacy_fail_closed() -> None:
    with pytest.raises(ValidationError, match="exactly eight"):
        CalibrationSplit(
            manifest_hash="hash", discovery_task_ids=("same",), regression_task_ids=("same",)
        )

    split = _split()
    leaking_rule = KappaRule(
        rule_id="leak",
        action_types=("final_answer",),
        instruction="Open the browser and return 42 for fixture.png.",
    )
    proposal = HStarProposal(
        proposal_id="leak",
        rules=(leaking_rule,),
        discovery_task_ids=split.discovery_task_ids,
    )
    executions = _executions_for_proposal(
        split,
        proposal,
        leaking_rule,
        discovery_helped=1,
        regression_helped=1,
        regression_harmed=0,
    )
    with pytest.raises(ValueError, match=r"leakage|specific"):
        freeze_kappa(
            session=KappaCalibrationSession(split),
            proposal=proposal,
            executions=executions,
            privacy=CalibrationPrivacyContext(forbidden_phrases=("42", "fixture.png")),
        )

    with pytest.raises(ValueError, match="regression split"):
        freeze_kappa(
            session=KappaCalibrationSession(split),
            proposal=proposal,
            executions=executions[:-1],
            privacy=CalibrationPrivacyContext(forbidden_phrases=("never matches",)),
        )


def test_calibration_session_rejects_a_second_proposal() -> None:
    split = _split()
    rule = KappaRule(
        rule_id="verify",
        action_types=("inspect_image",),
        instruction="Verify chart labels.",
    )
    first = HStarProposal(
        proposal_id="first", rules=(rule,), discovery_task_ids=split.discovery_task_ids
    )
    executions = _executions_for_proposal(
        split, first, rule, discovery_helped=1, regression_helped=1, regression_harmed=0
    )
    session = KappaCalibrationSession(split)
    freeze_kappa(
        session=session,
        proposal=first,
        executions=executions,
        privacy=CalibrationPrivacyContext(forbidden_phrases=("never matches",)),
    )
    changed = first.model_copy(
        update={
            "proposal_id": "changed",
            "rules": (rule.model_copy(update={"instruction": "Ignore chart labels."}),),
        }
    )
    with pytest.raises(ValueError, match="different H-star proposal"):
        freeze_kappa(
            session=session,
            proposal=changed,
            executions=executions,
            privacy=CalibrationPrivacyContext(forbidden_phrases=("never matches",)),
        )


def test_kappa_requires_strict_regression_improvement() -> None:
    split = _split()
    rule = KappaRule(
        rule_id="neutral",
        action_types=("inspect_image",),
        instruction="Inspect the chart.",
    )
    proposal = HStarProposal(
        proposal_id="neutral", rules=(rule,), discovery_task_ids=split.discovery_task_ids
    )
    executions = _executions_for_proposal(
        split,
        proposal,
        rule,
        discovery_helped=1,
        regression_helped=0,
        regression_harmed=0,
    )

    with pytest.raises(ValueError, match="measured regression improvement"):
        freeze_kappa(
            session=KappaCalibrationSession(split),
            proposal=proposal,
            executions=executions,
            privacy=CalibrationPrivacyContext(forbidden_phrases=("never matches",)),
        )


def test_failed_kappa_attempt_is_terminal() -> None:
    split = _split()
    rule = KappaRule(
        rule_id="leak",
        action_types=("inspect_image",),
        instruction="Inspect fixture.png before answering.",
    )
    proposal = HStarProposal(
        proposal_id="single-attempt", rules=(rule,), discovery_task_ids=split.discovery_task_ids
    )
    executions = _executions_for_proposal(
        split,
        proposal,
        rule,
        discovery_helped=1,
        regression_helped=1,
        regression_harmed=0,
    )
    session = KappaCalibrationSession(split)
    with pytest.raises(ValueError, match="specific"):
        freeze_kappa(
            session=session,
            proposal=proposal,
            executions=executions,
            privacy=CalibrationPrivacyContext(forbidden_phrases=("fixture.png",)),
        )

    with pytest.raises(ValueError, match="already failed"):
        freeze_kappa(
            session=session,
            proposal=proposal,
            executions=executions,
            privacy=CalibrationPrivacyContext(forbidden_phrases=("different",)),
        )

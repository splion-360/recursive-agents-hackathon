"""Offline acceptance tests for the isolated live-calibration lifecycle."""

from decimal import Decimal

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.calibration import (
    CalibrationProgress,
    HStarAnalysisRequest,
    HStarAnalysisResult,
    ModelIdentities,
    calibrate_review_policy,
)
from memory_distillation.review_policy import HStarProposal, KappaRule
from memory_distillation.task_execution import (
    ActionReview,
    CalibrationExecutionPolicy,
    ExecutionPolicy,
    FinalAnswerAction,
    TaskRun,
)


class FixtureSource:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.discovery = tuple(_task(f"d{index}") for index in range(8))
        self.regression = tuple(_task(f"r{index}") for index in range(8))

    def load_calibration_discovery(self) -> tuple[TaskInput, ...]:
        self.calls.append("discovery")
        return self.discovery

    def load_calibration_regression(self) -> tuple[TaskInput, ...]:
        self.calls.append("regression")
        return self.regression

    def load_asset(self, task: TaskInput) -> TaskAsset:
        return TaskAsset(
            task_id=task.task_id,
            media_type="image/png",
            image_sha256=task.image_sha256,
            content=b"fixture",
        )


class FixtureAnalyst:
    def __init__(self, source: FixtureSource) -> None:
        self.source = source

    def propose(self, request: HStarAnalysisRequest, *, seed: int) -> HStarAnalysisResult:
        assert self.source.calls == ["discovery"]
        assert len(request.observations) == 8
        proposal = HStarProposal(
            proposal_id=f"hstar-{seed}",
            rules=(
                KappaRule(
                    rule_id="verify-labels",
                    action_types=("final_answer",),
                    instruction="Verify labels and arithmetic before the final answer.",
                ),
            ),
            discovery_task_ids=request.discovery_task_ids,
        )
        return HStarAnalysisResult(
            proposal=proposal,
            model_id="fixture-hstar",
            request_id="analysis-1",
            input_tokens=20,
            output_tokens=10,
            latency_ms=5,
            actual_cost_usd=Decimal("0.001"),
        )


class FixtureExecutor:
    def __init__(self, *, candidate_improves: bool) -> None:
        self.candidate_improves = candidate_improves

    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun:
        del asset, seed, behavior_cards
        if policy.review_required:
            return _run(task.task_id, correct=True, reviewed=True, kappa_hash=policy.kappa_hash)
        return _run(task.task_id, correct=False)

    def run_calibration(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: CalibrationExecutionPolicy,
        seed: int,
    ) -> TaskRun:
        del asset, policy, seed
        return _run(task.task_id, correct=self.candidate_improves, reviewed=True)


def test_regression_remains_sealed_until_the_proposal_is_frozen() -> None:
    source = FixtureSource()
    progress: list[CalibrationProgress] = []

    report = calibrate_review_policy(
        run_id="calibration-test",
        source=source,
        executor=FixtureExecutor(candidate_improves=True),
        analyst=FixtureAnalyst(source),
        manifest_hash="m" * 64,
        models=ModelIdentities(
            student="fixture-student",
            reviewer="fixture-reviewer",
            hstar="fixture-hstar",
        ),
        publish=progress.append,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=3,
    )

    assert source.calls == ["discovery", "regression"]
    assert report.kappa.revision == 1
    assert report.kappa.guidance == (
        "Verify labels and arithmetic before the final answer.",
    )
    assert report.fallback_reason is None
    assert len(report.outcomes) == 35
    assert progress[-1].stage == "complete"
    assert progress[-1].smoke_completed == 3


def test_no_positive_regression_evidence_freezes_explicit_minimal_fallback() -> None:
    source = FixtureSource()

    report = calibrate_review_policy(
        run_id="calibration-fallback",
        source=source,
        executor=FixtureExecutor(candidate_improves=False),
        analyst=FixtureAnalyst(source),
        manifest_hash="m" * 64,
        models=ModelIdentities(
            student="fixture-student",
            reviewer="fixture-reviewer",
            hstar="fixture-hstar",
        ),
        publish=lambda progress: None,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=1,
    )

    assert report.kappa.revision == 0
    assert report.kappa.guidance == ()
    assert report.fallback_reason is not None


def _task(task_id: str) -> TaskInput:
    return TaskInput(
        task_id=task_id,
        question=f"Question for {task_id}?",
        image_asset_id=f"{task_id}.png",
        image_sha256="a" * 64,
        source_split="train",
    )


def _run(
    task_id: str,
    *,
    correct: bool,
    reviewed: bool = False,
    kappa_hash: str | None = None,
) -> TaskRun:
    action = FinalAnswerAction(answer="fixture")
    reviews = (
        ActionReview(
            decision="pass",
            proposed_action=action,
            accepted_action=action,
            model_id="fixture-reviewer",
            request_id=f"review:{task_id}",
            input_tokens=2,
            output_tokens=1,
            latency_ms=1,
        ),
    ) if reviewed else ()
    return TaskRun(
        task_id=task_id,
        seed=17,
        termination_reason="completed",
        actions=(action,),
        observations=(),
        reviews=reviews,
        events=(),
        score=Score(correct=correct, method="text_exact"),
        input_tokens=10,
        output_tokens=2,
        elapsed_ms=5,
        tool_calls=0,
        python_calls=0,
        review_calls=len(reviews),
        model_calls=1,
        kappa_hash=kappa_hash,
    )

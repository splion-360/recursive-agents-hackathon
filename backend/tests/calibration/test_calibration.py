"""Offline acceptance tests for the isolated live-calibration lifecycle."""

from decimal import Decimal

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.calibration import (
    CalibrationProgress,
    CalibrationProvenance,
    HStarAnalysisRequest,
    HStarAnalysisResult,
    ModelIdentities,
    calibrate_review_policy,
)
from memory_distillation.review_policy import HStarProposal, KappaRule
from memory_distillation.task_execution import (
    ActionReview,
    CalibrationExecutionPolicy,
    CropAction,
    ExecutionPolicy,
    FinalAnswerAction,
    TaskRun,
)


class FixtureSource:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.discovery = tuple(_task(f"d{index}") for index in range(12))
        self.regression = tuple(_task(f"r{index}") for index in range(12))

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
        self.requests: list[HStarAnalysisRequest] = []

    def propose(self, request: HStarAnalysisRequest, *, seed: int) -> HStarAnalysisResult:
        assert self.source.calls == ["discovery"]
        self.requests.append(request)
        assert len(request.observations) == request.round_index * 4
        proposal = HStarProposal(
            proposal_id=f"hstar-r{request.round_index}",
            parent_revision=request.round_index - 1,
            revision=request.round_index,
            rules=(
                KappaRule(
                    rule_id="verify-labels",
                    source_component="skills",
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
        self.baseline_seeds: dict[str, int] = {}
        self.candidate_seeds: dict[str, int] = {}

    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun:
        del asset, behavior_cards
        if policy.review_required:
            return _run(task.task_id, correct=True, reviewed=True, kappa_hash=policy.kappa_hash)
        self.baseline_seeds[task.task_id] = seed
        return _run(task.task_id, correct=False)

    def run_calibration(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: CalibrationExecutionPolicy,
        seed: int,
    ) -> TaskRun:
        del asset, policy
        self.candidate_seeds[task.task_id] = seed
        return _run(task.task_id, correct=self.candidate_improves, reviewed=True)


class FailingPilotExecutor(FixtureExecutor):
    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun:
        if policy.review_required:
            proposed = FinalAnswerAction(answer="fixture")
            replacement = CropAction(x0=0, y0=0, x1=9000, y1=9000)
            review = ActionReview(
                decision="replace",
                proposed_action=proposed,
                accepted_action=replacement,
                model_id="fixture-reviewer",
                request_id=f"review:{task.task_id}",
                input_tokens=2,
                output_tokens=1,
                latency_ms=1,
            )
            return TaskRun(
                task_id=task.task_id,
                seed=seed,
                termination_reason="invalid_crop",
                actions=(replacement,),
                observations=(),
                reviews=(review,),
                events=(),
                score=None,
                input_tokens=10,
                output_tokens=2,
                elapsed_ms=5,
                tool_calls=1,
                python_calls=0,
                review_calls=1,
                model_calls=1,
                kappa_hash=policy.kappa_hash,
            )
        return super().run(
            task=task,
            asset=asset,
            policy=policy,
            seed=seed,
            behavior_cards=behavior_cards,
        )


class FailingDiscoveryExecutor(FixtureExecutor):
    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun:
        if not policy.review_required and task.task_id == "d9":
            self.baseline_seeds[task.task_id] = seed
            return _failed_run(task.task_id, seed=seed)
        return super().run(
            task=task,
            asset=asset,
            policy=policy,
            seed=seed,
            behavior_cards=behavior_cards,
        )


def test_regression_remains_sealed_until_the_proposal_is_frozen() -> None:
    source = FixtureSource()
    progress: list[CalibrationProgress] = []
    executor = FixtureExecutor(candidate_improves=True)

    analyst = FixtureAnalyst(source)
    report = calibrate_review_policy(
        run_id="calibration-test",
        source=source,
        executor=executor,
        analyst=analyst,
        manifest_hash="m" * 64,
        models=ModelIdentities(
            student="fixture-student",
            reviewer="fixture-reviewer",
            hstar="fixture-hstar",
        ),
        provenance=_provenance(),
        publish=progress.append,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=3,
    )

    assert source.calls == ["discovery", "regression"]
    assert report.kappa.revision == 3
    assert report.kappa.guidance == (
        "Verify labels and arithmetic before the final answer.",
    )
    assert report.fallback_reason is None
    assert [request.round_index for request in analyst.requests] == [1, 2, 3]
    assert [len(request.observations) for request in analyst.requests] == [4, 8, 12]
    assert [revision.proposal.revision for revision in report.hstar_revisions] == [1, 2, 3]
    assert len(report.outcomes) == 51
    assert progress[-1].stage == "complete"
    assert progress[-1].smoke_completed == 3
    assert executor.candidate_seeds == executor.baseline_seeds


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
        provenance=_provenance(),
        publish=lambda progress: None,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=3,
    )

    assert report.kappa.revision == 0
    assert report.kappa.guidance == ()
    assert report.fallback_reason is not None


def test_failed_pilot_interventions_remain_auditable() -> None:
    source = FixtureSource()
    report = calibrate_review_policy(
        run_id="calibration-invalid-pilot",
        source=source,
        executor=FailingPilotExecutor(candidate_improves=True),
        analyst=FixtureAnalyst(source),
        manifest_hash="m" * 64,
        models=ModelIdentities(
            student="fixture-student",
            reviewer="fixture-reviewer",
            hstar="fixture-hstar",
        ),
        provenance=_provenance(),
        publish=lambda progress: None,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=3,
    )

    pilot_outcomes = tuple(item for item in report.outcomes if item.phase == "pilot")
    assert all(item.correct is None for item in pilot_outcomes)
    assert report.pilot_summary.invalid_interventions == 3
    assert all(not item.accepted_action_valid for item in report.pilot_interventions)


def test_failed_discovery_task_is_recorded_and_calibration_continues() -> None:
    source = FixtureSource()
    analyst = FixtureAnalyst(source)

    report = calibrate_review_policy(
        run_id="calibration-failed-discovery",
        source=source,
        executor=FailingDiscoveryExecutor(candidate_improves=True),
        analyst=analyst,
        manifest_hash="m" * 64,
        models=ModelIdentities(
            student="fixture-student",
            reviewer="fixture-reviewer",
            hstar="fixture-hstar",
        ),
        provenance=_provenance(),
        publish=lambda progress: None,
        cost_meter=lambda: Decimal("0.01"),
        pilot_tasks=3,
    )

    failed = next(
        item
        for item in report.outcomes
        if item.task_id == "d9" and item.condition == "baseline"
    )
    assert failed.termination_reason == "sandbox_failed"
    assert failed.correct is None
    assert analyst.requests[-1].observations[-3].correct is False
    assert report.kappa.revision == 3


def _provenance() -> CalibrationProvenance:
    return CalibrationProvenance(
        code_revision="a" * 40,
        seed=17,
        execution_policy=ExecutionPolicy().model_dump(mode="json"),
        student_max_output_tokens=512,
        reviewer_max_output_tokens=512,
        hstar_max_output_tokens=1200,
        sandbox_name="fixture",
        inference_base_url="https://example.test/v1",
    )


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


def _failed_run(task_id: str, *, seed: int) -> TaskRun:
    return TaskRun(
        task_id=task_id,
        seed=seed,
        termination_reason="sandbox_failed",
        actions=(),
        observations=(),
        reviews=(),
        events=(),
        score=None,
        input_tokens=10,
        output_tokens=2,
        elapsed_ms=5,
        tool_calls=1,
        python_calls=1,
        review_calls=0,
        model_calls=1,
        kappa_hash=None,
    )

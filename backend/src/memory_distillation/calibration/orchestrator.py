"""One-revision live H-star/Kappa calibration orchestration."""

import hashlib
import json
from collections.abc import Callable
from decimal import Decimal
from typing import Literal, Protocol

from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.calibration.models import (
    CalibrationOutcome,
    CalibrationProgress,
    CalibrationReport,
    HStarAnalysisRequest,
    HStarAnalyst,
    ModelIdentities,
    discovery_observations,
    outcome,
)
from memory_distillation.review_policy import (
    CalibrationExecution,
    CalibrationPrivacyContext,
    CalibrationSplit,
    HStarProposal,
    KappaCalibrationSession,
    KappaRule,
    freeze_kappa,
    freeze_minimal_kappa,
    hash_hstar_proposal,
)
from memory_distillation.task_execution import (
    CalibrationExecutionPolicy,
    ExecutionPolicy,
    TaskRun,
)


class CalibrationTaskSource(Protocol):
    def load_calibration_discovery(self) -> tuple[TaskInput, ...]: ...
    def load_calibration_regression(self) -> tuple[TaskInput, ...]: ...
    def load_asset(self, task: TaskInput) -> TaskAsset: ...


class CalibrationTaskExecutor(Protocol):
    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun: ...

    def run_calibration(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: CalibrationExecutionPolicy,
        seed: int,
    ) -> TaskRun: ...


ProgressPublisher = Callable[[CalibrationProgress], None]
CostMeter = Callable[[], Decimal]


def calibrate_review_policy(
    *,
    run_id: str,
    source: CalibrationTaskSource,
    executor: CalibrationTaskExecutor,
    analyst: HStarAnalyst,
    manifest_hash: str,
    models: ModelIdentities,
    publish: ProgressPublisher,
    cost_meter: CostMeter,
    seed: int = 17,
    pilot_tasks: int = 3,
) -> CalibrationReport:
    """Run isolated discovery, one proposal, sealed regression, and a frozen pilot."""

    if not 1 <= pilot_tasks <= 5:
        raise ValueError("pilot_tasks must be between one and five")
    base_policy = ExecutionPolicy()
    outcomes: list[CalibrationOutcome] = []
    total_input = total_output = total_elapsed = 0

    def record(run: TaskRun, item: CalibrationOutcome) -> None:
        nonlocal total_input, total_output, total_elapsed
        outcomes.append(item)
        total_input += run.input_tokens
        total_output += run.output_tokens
        total_elapsed += run.elapsed_ms

    def progress(stage: str, message: str, **updates: object) -> None:
        payload: dict[str, object] = {
            "stage": stage,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "elapsed_ms": total_elapsed,
            "estimated_cost_usd": cost_meter(),
            "message": message,
            **updates,
        }
        publish(CalibrationProgress.model_validate(payload))

    progress("preflight", "Loading the isolated discovery manifest.")
    discovery = source.load_calibration_discovery()
    _require_eight("discovery", discovery)
    discovery_baselines: list[TaskRun] = []
    for index, task in enumerate(discovery, start=1):
        progress(
            "discovery",
            f"Running discovery baseline {index} of 8.",
            discovery_completed=index - 1,
            current_task_id=task.task_id,
        )
        run = executor.run(
            task=task,
            asset=source.load_asset(task),
            policy=base_policy,
            seed=seed + index - 1,
        )
        discovery_baselines.append(run)
        record(run, outcome(run, phase="discovery", condition="baseline"))

    analysis = analyst.propose(
        HStarAnalysisRequest(
            manifest_hash=manifest_hash,
            discovery_task_ids=tuple(task.task_id for task in discovery),
            observations=discovery_observations(discovery, tuple(discovery_baselines)),
        ),
        seed=seed,
    )
    proposal = HStarProposal.model_validate(analysis.proposal.model_dump(mode="json"))
    if proposal.discovery_task_ids != tuple(task.task_id for task in discovery):
        raise ValueError("H-star proposal does not bind the exact discovery split")
    proposal_hash = hash_hstar_proposal(proposal)
    total_input += analysis.input_tokens
    total_output += analysis.output_tokens
    total_elapsed += analysis.latency_ms
    progress(
        "hstar_proposal",
        "Proposal frozen; evaluating its rules on discovery evidence.",
        discovery_completed=8,
        proposal=proposal,
        proposal_hash=proposal_hash,
    )

    executions: list[CalibrationExecution] = []
    baseline_by_discovery = {run.task_id: run for run in discovery_baselines}
    for rule in proposal.rules:
        policy = CalibrationExecutionPolicy(
            execution=base_policy,
            proposal_hash=proposal_hash,
            guidance=(rule.instruction,),
        )
        for index, task in enumerate(discovery):
            candidate = executor.run_calibration(
                task=task,
                asset=source.load_asset(task),
                policy=policy,
                seed=seed + index,
            )
            record(
                candidate,
                outcome(
                    candidate,
                    phase="discovery",
                    condition="candidate",
                    rule_id=rule.rule_id,
                ),
            )
            executions.append(
                _execution(
                    phase="discovery",
                    rule=rule,
                    proposal_hash=proposal_hash,
                    baseline=baseline_by_discovery[task.task_id],
                    candidate=candidate,
                )
            )

    # This is intentionally the first point at which regression task inputs are loaded.
    regression = source.load_calibration_regression()
    _require_eight("regression", regression)
    split = CalibrationSplit(
        manifest_hash=manifest_hash,
        discovery_task_ids=tuple(task.task_id for task in discovery),
        regression_task_ids=tuple(task.task_id for task in regression),
    )
    regression_baselines: list[TaskRun] = []
    for index, task in enumerate(regression, start=1):
        progress(
            "regression",
            f"Running sealed regression baseline {index} of 8.",
            discovery_completed=8,
            regression_completed=0,
            proposal=proposal,
            proposal_hash=proposal_hash,
            current_task_id=task.task_id,
        )
        run = executor.run(
            task=task,
            asset=source.load_asset(task),
            policy=base_policy,
            seed=seed + 100 + index - 1,
        )
        regression_baselines.append(run)
        record(run, outcome(run, phase="regression", condition="baseline"))

    baseline_by_regression = {run.task_id: run for run in regression_baselines}
    for rule_index, rule in enumerate(proposal.rules):
        policy = CalibrationExecutionPolicy(
            execution=base_policy,
            proposal_hash=proposal_hash,
            guidance=(rule.instruction,),
        )
        for index, task in enumerate(regression, start=1):
            candidate = executor.run_calibration(
                task=task,
                asset=source.load_asset(task),
                policy=policy,
                seed=seed + 200 + index - 1,
            )
            record(
                candidate,
                outcome(
                    candidate,
                    phase="regression",
                    condition="candidate",
                    rule_id=rule.rule_id,
                ),
            )
            executions.append(
                _execution(
                    phase="regression",
                    rule=rule,
                    proposal_hash=proposal_hash,
                    baseline=baseline_by_regression[task.task_id],
                    candidate=candidate,
                )
            )
            if rule_index == len(proposal.rules) - 1:
                progress(
                    "regression",
                    f"Completed regression task {index} of 8.",
                    discovery_completed=8,
                    regression_completed=index,
                    proposal=proposal,
                    proposal_hash=proposal_hash,
                    current_task_id=task.task_id,
                )

    privacy = CalibrationPrivacyContext(
        forbidden_phrases=tuple(
            item
            for task in (*discovery, *regression)
            for item in (task.task_id, task.question)
            if item.strip()
        )
    )
    fallback_reason: str | None = None
    try:
        kappa = freeze_kappa(
            session=KappaCalibrationSession(split),
            proposal=proposal,
            executions=tuple(executions),
            privacy=privacy,
        )
    except ValueError as error:
        if "no Kappa rule demonstrated" not in str(error):
            raise
        fallback_reason = "No H-star rule produced positive isolated regression evidence."
        kappa = freeze_minimal_kappa(split)
    accepted_ids = {rule.rule_id for rule in kappa.rules}
    rejected = tuple(rule.rule_id for rule in proposal.rules if rule.rule_id not in accepted_ids)
    progress(
        "kappa_freeze",
        fallback_reason or "Regression-screened Kappa guidance is frozen.",
        discovery_completed=8,
        regression_completed=8,
        proposal=proposal,
        proposal_hash=proposal_hash,
        kappa=kappa,
        rejected_rule_ids=rejected,
    )

    pilot_policy = ExecutionPolicy(review_required=True, kappa=kappa)
    for index, task in enumerate(discovery[:pilot_tasks], start=1):
        progress(
            "smoke_test",
            f"Running frozen Kappa pilot {index} of {pilot_tasks}.",
            discovery_completed=8,
            regression_completed=8,
            smoke_completed=index - 1,
            smoke_total=pilot_tasks,
            proposal=proposal,
            proposal_hash=proposal_hash,
            kappa=kappa,
            rejected_rule_ids=rejected,
            current_task_id=task.task_id,
        )
        run = executor.run(
            task=task,
            asset=source.load_asset(task),
            policy=pilot_policy,
            seed=seed + 300 + index - 1,
        )
        record(run, outcome(run, phase="pilot", condition="frozen_kappa"))

    progress(
        "complete",
        fallback_reason or "Calibration complete; frozen Kappa is ready for Collection.",
        discovery_completed=8,
        regression_completed=8,
        smoke_completed=pilot_tasks,
        smoke_total=pilot_tasks,
        proposal=proposal,
        proposal_hash=proposal_hash,
        kappa=kappa,
        rejected_rule_ids=rejected,
    )
    return CalibrationReport(
        run_id=run_id,
        split=split,
        proposal=proposal,
        proposal_hash=proposal_hash,
        kappa=kappa,
        fallback_reason=fallback_reason,
        models=models,
        outcomes=tuple(outcomes),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_elapsed_ms=total_elapsed,
        total_cost_usd=cost_meter(),
    )


def combined_manifest_hash(discovery_hash: str, regression_hash: str) -> str:
    payload = {"discovery": discovery_hash, "regression": regression_hash}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_eight(name: str, tasks: tuple[TaskInput, ...]) -> None:
    if len(tasks) != 8 or len({task.task_id for task in tasks}) != 8:
        raise ValueError(f"{name} calibration split must contain eight unique tasks")


def _execution(
    *,
    phase: Literal["discovery", "regression"],
    rule: KappaRule,
    proposal_hash: str,
    baseline: TaskRun,
    candidate: TaskRun,
) -> CalibrationExecution:
    baseline_outcome = outcome(baseline, phase=phase, condition="baseline")
    candidate_outcome = outcome(candidate, phase=phase, condition="candidate")
    return CalibrationExecution(
        execution_id=f"{proposal_hash[:12]}:{rule.rule_id}:{phase}:{baseline.task_id}",
        proposal_hash=proposal_hash,
        phase=phase,
        task_id=baseline.task_id,
        rule_id=rule.rule_id,
        action_type=rule.action_types[0],
        baseline_correct=baseline_outcome.correct,
        candidate_correct=candidate_outcome.correct,
    )

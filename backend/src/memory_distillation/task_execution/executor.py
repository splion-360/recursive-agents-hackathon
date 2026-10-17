"""Minimal deterministic Student executor."""

from typing import Protocol

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.inference.protocol import InferenceClient, InferenceRequest
from memory_distillation.sandbox import Sandbox, SandboxRequest
from memory_distillation.task_execution.models import (
    ActionProposedEvent,
    ActionReview,
    ActionReviewedEvent,
    CropAction,
    ExecutionPolicy,
    FinalAnswerAction,
    InferenceRequestedEvent,
    InspectImageAction,
    Observation,
    ObservationEvent,
    RunCompletedEvent,
    RunEvent,
    RunStartedEvent,
    SandboxFailureEvent,
    StudentAction,
    TaskRun,
    TerminationReason,
)
from memory_distillation.task_execution.review import Reviewer, ReviewRequest
from memory_distillation.telemetry.sink import EventSink


class Verifier(Protocol):
    def score(self, task_id: str, prediction: str) -> Score: ...


class TaskExecutor:
    """Run the complete typed action loop behind one stable interface."""

    def __init__(
        self,
        *,
        inference: InferenceClient[StudentAction],
        sandbox: Sandbox,
        verifier: Verifier,
        event_sink: EventSink[RunEvent],
        reviewer: Reviewer | None = None,
    ) -> None:
        self._inference = inference
        self._sandbox = sandbox
        self._verifier = verifier
        self._event_sink = event_sink
        self._reviewer = reviewer

    def run(
        self,
        *,
        task: TaskInput,
        asset: TaskAsset,
        policy: ExecutionPolicy,
        seed: int,
        behavior_cards: tuple[str, ...] = (),
    ) -> TaskRun:
        """Execute one answer-free task with deterministic budgets and evidence."""

        policy = ExecutionPolicy.model_validate(policy.model_dump(mode="json"))
        if asset.task_id != task.task_id or asset.image_sha256 != task.image_sha256:
            raise ValueError("task asset does not match task input")

        actions: list[StudentAction] = []
        observations: list[Observation] = []
        reviews: list[ActionReview] = []
        events: list[RunEvent] = []
        input_tokens = output_tokens = elapsed_ms = tool_calls = python_calls = review_calls = 0

        def emit(event: RunEvent) -> None:
            events.append(event)
            self._event_sink.emit(event)

        emit(
            RunStartedEvent(
                sequence=0,
                task_id=task.task_id,
                seed=seed,
                kappa_hash=policy.kappa_hash,
            )
        )

        for turn_index in range(policy.max_actions):
            emit(
                InferenceRequestedEvent(
                    sequence=len(events),
                    task_id=task.task_id,
                    turn_index=turn_index,
                )
            )
            try:
                result = self._inference.generate(
                    InferenceRequest(
                        task=task,
                        asset=asset,
                        observations=tuple(item.content for item in observations),
                        seed=seed,
                        turn_index=turn_index,
                        behavior_cards=behavior_cards,
                    )
                )
            except Exception:
                return self._finish(
                    "inference_failed",
                    task,
                    seed,
                    actions,
                    observations,
                    events,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                    tool_calls,
                    python_calls,
                    None,
                )

            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            elapsed_ms += result.latency_ms
            if input_tokens + output_tokens > policy.max_total_tokens:
                return self._finish(
                    "token_budget_exhausted",
                    task,
                    seed,
                    actions,
                    observations,
                    events,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                    tool_calls,
                    python_calls,
                    None,
                )
            if elapsed_ms > policy.max_elapsed_ms:
                return self._finish(
                    "time_budget_exhausted",
                    task,
                    seed,
                    actions,
                    observations,
                    events,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                    tool_calls,
                    python_calls,
                    None,
                )

            proposed_action = result.output
            emit(
                ActionProposedEvent(
                    sequence=len(events),
                    task_id=task.task_id,
                    action=proposed_action,
                    model_id=result.model_id,
                    request_id=result.request_id,
                )
            )

            action = proposed_action
            if policy.review_required:
                if review_calls >= policy.max_review_calls:
                    return self._finish(
                        "reviewer_budget_exhausted",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )
                if self._reviewer is None:
                    return self._finish(
                        "reviewer_failed",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )
                review_calls += 1
                try:
                    result_review = self._reviewer.review(
                        ReviewRequest(
                            task=task,
                            asset=asset,
                            proposed_action=proposed_action,
                            observations=tuple(observations),
                            guidance=policy.kappa_guidance,
                            turn_index=turn_index,
                        )
                    )
                except Exception:
                    return self._finish(
                        "reviewer_failed",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )
                pass_changed_action = (
                    result_review.decision == "pass"
                    and result_review.accepted_action != proposed_action
                )
                if pass_changed_action:
                    return self._finish(
                        "reviewer_failed",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )
                action = result_review.accepted_action
                public_review = ActionReview(
                    decision=result_review.decision,
                    proposed_action=proposed_action,
                    accepted_action=action,
                    model_id=result_review.model_id,
                    request_id=result_review.request_id,
                    input_tokens=result_review.input_tokens,
                    output_tokens=result_review.output_tokens,
                    latency_ms=result_review.latency_ms,
                )
                reviews.append(public_review)
                emit(
                    ActionReviewedEvent(
                        sequence=len(events),
                        task_id=task.task_id,
                        review=public_review,
                    )
                )
                input_tokens += result_review.input_tokens
                output_tokens += result_review.output_tokens
                elapsed_ms += result_review.latency_ms
                if input_tokens + output_tokens > policy.max_total_tokens:
                    return self._finish(
                        "token_budget_exhausted",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )
                if elapsed_ms > policy.max_elapsed_ms:
                    return self._finish(
                        "time_budget_exhausted",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                        reviews,
                        review_calls,
                    )

            actions.append(action)

            if isinstance(action, FinalAnswerAction):
                score = self._verifier.score(task.task_id, action.answer)
                return self._finish(
                    "completed",
                    task,
                    seed,
                    actions,
                    observations,
                    events,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                    tool_calls,
                    python_calls,
                    score,
                )

            if tool_calls >= policy.max_tool_calls:
                return self._finish(
                    "tool_budget_exhausted",
                    task,
                    seed,
                    actions,
                    observations,
                    events,
                    input_tokens,
                    output_tokens,
                    elapsed_ms,
                    tool_calls,
                    python_calls,
                    None,
                )
            tool_calls += 1

            if isinstance(action, InspectImageAction):
                observation = Observation(
                    action_type="inspect_image",
                    content=f"image_sha256={asset.image_sha256}; bytes={len(asset.content)}",
                )
            elif isinstance(action, CropAction):
                if max(action.x0, action.y0, action.x1, action.y1) > policy.max_crop_coordinate:
                    return self._finish(
                        "invalid_crop",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                    )
                observation = Observation(
                    action_type="crop",
                    content=f"crop=({action.x0},{action.y0},{action.x1},{action.y1})",
                )
            else:
                if python_calls >= policy.max_python_calls:
                    return self._finish(
                        "python_budget_exhausted",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                    )
                python_calls += 1
                try:
                    sandbox_result = self._sandbox.execute(
                        SandboxRequest(
                            task_id=task.task_id,
                            code=action.code,
                            timeout_ms=policy.sandbox_timeout_ms,
                        )
                    )
                except Exception:
                    emit(
                        SandboxFailureEvent(
                            sequence=len(events),
                            task_id=task.task_id,
                            status="failed",
                            failure_kind="transport_failed",
                            exit_code=None,
                            latency_ms=0,
                        )
                    )
                    return self._finish(
                        "sandbox_failed",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                    )
                elapsed_ms += sandbox_result.latency_ms
                if elapsed_ms > policy.max_elapsed_ms:
                    return self._finish(
                        "time_budget_exhausted",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                    )
                if sandbox_result.status != "completed" or sandbox_result.exit_code != 0:
                    failure_kind = sandbox_result.failure_kind
                    if failure_kind is None:
                        failure_kind = (
                            "timed_out"
                            if sandbox_result.status == "timed_out"
                            else "command_failed"
                        )
                    emit(
                        SandboxFailureEvent(
                            sequence=len(events),
                            task_id=task.task_id,
                            status=sandbox_result.status,
                            failure_kind=failure_kind,
                            exit_code=sandbox_result.exit_code,
                            latency_ms=sandbox_result.latency_ms,
                        )
                    )
                    return self._finish(
                        "sandbox_failed",
                        task,
                        seed,
                        actions,
                        observations,
                        events,
                        input_tokens,
                        output_tokens,
                        elapsed_ms,
                        tool_calls,
                        python_calls,
                        None,
                    )
                observation = Observation(
                    action_type="run_python",
                    content=sandbox_result.stdout,
                )

            observations.append(observation)
            emit(
                ObservationEvent(
                    sequence=len(events),
                    task_id=task.task_id,
                    observation=observation,
                )
            )

        return self._finish(
            "action_budget_exhausted",
            task,
            seed,
            actions,
            observations,
            events,
            input_tokens,
            output_tokens,
            elapsed_ms,
            tool_calls,
            python_calls,
            None,
        )

    def _finish(
        self,
        reason: TerminationReason,
        task: TaskInput,
        seed: int,
        actions: list[StudentAction],
        observations: list[Observation],
        events: list[RunEvent],
        input_tokens: int,
        output_tokens: int,
        elapsed_ms: int,
        tool_calls: int,
        python_calls: int,
        score: Score | None,
        reviews: list[ActionReview] | None = None,
        review_calls: int = 0,
    ) -> TaskRun:
        if reviews is None:
            reviews = [event.review for event in events if isinstance(event, ActionReviewedEvent)]
            review_calls = len(reviews)
        event = RunCompletedEvent(
            sequence=len(events),
            task_id=task.task_id,
            termination_reason=reason,
            correct=score.correct if score is not None else None,
        )
        events.append(event)
        self._event_sink.emit(event)
        return TaskRun(
            task_id=task.task_id,
            seed=seed,
            termination_reason=reason,
            actions=tuple(actions),
            observations=tuple(observations),
            reviews=tuple(reviews),
            events=tuple(events),
            score=score,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed_ms,
            tool_calls=tool_calls,
            python_calls=python_calls,
            review_calls=review_calls,
            model_calls=sum(isinstance(item, InferenceRequestedEvent) for item in events),
            kappa_hash=(
                events[0].kappa_hash if events and isinstance(events[0], RunStartedEvent) else None
            ),
        )

"""Offline C0-C3 execution through the shared TaskExecutor."""

from dataclasses import dataclass

from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.evaluation.report import (
    C3Assignment,
    Condition,
    ConditionPolicy,
    ConditionResult,
    validate_condition_policies,
)
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.task_execution import TaskExecutor
from memory_distillation.versioning import (
    VersionView,
    retrieve_behavior_cards,
)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    task: TaskInput
    asset: TaskAsset
    seed: int


def run_offline_conditions(
    *,
    executor: TaskExecutor,
    tasks: tuple[EvaluationTask, ...],
    policies: tuple[ConditionPolicy, ...],
    accepted_view: VersionView,
    c3_assignments: tuple[C3Assignment, ...],
    kappa: KappaArtifact,
) -> tuple[ConditionResult, ...]:
    """Execute frozen conditions without accepting any writable memory capability."""

    if not tasks:
        raise ValueError("evaluation requires at least one task")
    validate_condition_policies(policies, kappa=kappa)
    policy_by_condition = {policy.condition: policy for policy in policies}
    assignment_by_task = {item.task_id: item for item in c3_assignments}
    if len(assignment_by_task) != len(c3_assignments):
        raise ValueError("C3 assignments contain duplicate task identifiers")
    donor_by_id = {card.card_id: card for card in accepted_view.cards}
    if len(donor_by_id) != len(accepted_view.cards):
        raise ValueError("Accepted Version contains duplicate card identifiers")

    # Resolve and validate every context before the first model request. A bad late C3
    # assignment must not leave a partial C0-C2 experiment behind.
    contexts: dict[tuple[str, str], tuple[str, ...]] = {}
    for item in tasks:
        task_id = item.task.task_id
        contexts[(task_id, "C0")] = ()
        contexts[(task_id, "C1")] = ()
        accepted_context = retrieve_behavior_cards(accepted_view, item.task)
        contexts[(task_id, "C2")] = accepted_context
        assignment = assignment_by_task.get(task_id)
        if assignment is None:
            continue
        try:
            selected = tuple(donor_by_id[card_id] for card_id in assignment.card_ids)
        except KeyError as error:
            raise ValueError("C3 assignment references an unknown donor card") from error
        task_terms = set(item.task.question.casefold().split())
        if any(task_terms & set(card.trigger.casefold().split()) for card in selected):
            raise ValueError("C3 assignment contains a trigger-matched donor card")
        irrelevant_context = tuple(f"{card.trigger}\n{card.instruction}" for card in selected)
        accepted_tokens = sum(len(text.split()) for text in accepted_context)
        assigned_tokens = sum(len(text.split()) for text in irrelevant_context)
        if assignment.token_count != assigned_tokens:
            raise ValueError("C3 assignment token provenance is invalid")
        if len(irrelevant_context) != len(accepted_context):
            raise ValueError("C3 assignment does not match C2 card count")
        if accepted_tokens == 0 or abs(assigned_tokens - accepted_tokens) > (
            accepted_tokens * 0.10
        ):
            raise ValueError("C3 assignment does not match C2 token count")
        contexts[(task_id, "C3")] = irrelevant_context

    results: list[ConditionResult] = []
    conditions: tuple[Condition, ...] = ("C0", "C1", "C2", "C3")
    for task_index, item in enumerate(sorted(tasks, key=lambda value: value.task.task_id)):
        # Rotate condition order across tasks to prevent warm-cache/order effects from
        # consistently favoring a single condition.
        offset = task_index % len(conditions)
        for condition in conditions[offset:] + conditions[:offset]:
            context = contexts.get((item.task.task_id, condition))
            if context is None:
                continue
            policy = policy_by_condition[condition]
            run = executor.run(
                task=item.task,
                asset=item.asset,
                policy=policy.execution,
                seed=item.seed,
                behavior_cards=context,
            )
            results.append(
                ConditionResult(
                    condition=condition,
                    task_id=item.task.task_id,
                    seed=item.seed,
                    correct=bool(run.score and run.score.correct),
                    reviewer_intervened=any(review.decision == "replace" for review in run.reviews),
                    latency_ms=run.elapsed_ms,
                    input_tokens=run.input_tokens,
                    output_tokens=run.output_tokens,
                    model_calls=len(run.actions),
                    tool_calls=run.tool_calls,
                    review_calls=run.review_calls,
                    retrieved_card_count=len(context),
                    valid=run.termination_reason == "completed",
                )
            )
    return tuple(results)

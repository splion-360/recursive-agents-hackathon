"""Offline C0-C3 execution through the shared TaskExecutor."""

from dataclasses import dataclass

from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.evaluation.report import (
    C3Assignment,
    ConditionPolicy,
    ConditionResult,
    validate_condition_policies,
)
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
) -> tuple[ConditionResult, ...]:
    """Execute frozen conditions without accepting any writable memory capability."""

    if not tasks:
        raise ValueError("evaluation requires at least one task")
    validate_condition_policies(policies)
    policy_by_condition = {policy.condition: policy for policy in policies}
    assignment_by_task = {item.task_id: item for item in c3_assignments}
    if len(assignment_by_task) != len(c3_assignments):
        raise ValueError("C3 assignments contain duplicate task identifiers")
    donor_by_id = {card.card_id: card for card in accepted_view.cards}
    if len(donor_by_id) != len(accepted_view.cards):
        raise ValueError("Accepted Version contains duplicate card identifiers")

    results: list[ConditionResult] = []
    for condition in ("C0", "C1", "C2", "C3"):
        policy = policy_by_condition[condition]
        for item in tasks:
            if condition == "C3" and item.task.task_id not in assignment_by_task:
                continue
            context: tuple[str, ...] = ()
            if condition == "C2":
                context = retrieve_behavior_cards(accepted_view, item.task)
            elif condition == "C3":
                assignment = assignment_by_task[item.task.task_id]
                try:
                    selected = tuple(donor_by_id[card_id] for card_id in assignment.card_ids)
                except KeyError as error:
                    raise ValueError("C3 assignment references an unknown donor card") from error
                accepted_context = retrieve_behavior_cards(accepted_view, item.task)
                task_terms = set(item.task.question.casefold().split())
                if any(task_terms & set(card.trigger.casefold().split()) for card in selected):
                    raise ValueError("C3 assignment contains a trigger-matched donor card")
                context = tuple(f"{card.trigger}\n{card.instruction}" for card in selected)
                accepted_tokens = sum(len(text.split()) for text in accepted_context)
                assigned_tokens = sum(len(text.split()) for text in context)
                if assignment.token_count != assigned_tokens:
                    raise ValueError("C3 assignment token provenance is invalid")
                if len(context) != len(accepted_context):
                    raise ValueError("C3 assignment does not match C2 card count")
                if accepted_tokens == 0 or abs(assigned_tokens - accepted_tokens) > (
                    accepted_tokens * 0.10
                ):
                    raise ValueError("C3 assignment does not match C2 token count")
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
                    tokens=run.input_tokens + run.output_tokens,
                    valid=run.termination_reason == "completed",
                )
            )
    return tuple(results)

"""Deterministic Behavior Card retrieval over one immutable version view."""

from memory_distillation.benchmark import TaskInput
from memory_distillation.versioning.models import BehaviorCard, VersionView


def select_behavior_cards(view: VersionView, task: TaskInput) -> tuple[BehaviorCard, ...]:
    """Return the exact card records selected by the frozen retrieval policy."""

    question_terms = set(task.question.casefold().split())
    matching: list[tuple[float, BehaviorCard]] = []
    for card in view.cards:
        trigger_terms = set(card.trigger.casefold().split())
        union = question_terms | trigger_terms
        score = len(question_terms & trigger_terms) / len(union) if union else 0
        if score >= view.retrieval_policy.minimum_score:
            matching.append((score, card))
    selected: list[BehaviorCard] = []
    used_tokens = 0
    for _, card in sorted(matching, key=lambda item: (-item[0], item[1].card_id)):
        if len(selected) >= view.retrieval_policy.top_k:
            break
        token_estimate = len(render_behavior_card(card).split())
        if used_tokens + token_estimate > view.retrieval_policy.token_budget:
            continue
        selected.append(card)
        used_tokens += token_estimate
    return tuple(selected)


def retrieve_behavior_cards(view: VersionView, task: TaskInput) -> tuple[str, ...]:
    """Render selected cards for Student context without exposing storage records."""

    return tuple(render_behavior_card(card) for card in select_behavior_cards(view, task))


def render_behavior_card(card: BehaviorCard) -> str:
    return f"{card.trigger}\n{card.instruction}"

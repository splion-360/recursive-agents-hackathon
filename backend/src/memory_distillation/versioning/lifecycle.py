"""Deterministic consolidation into a complete immutable Candidate Version."""

import hashlib
import json

from memory_distillation.operational_memory import ReviewedEpisode
from memory_distillation.versioning.models import (
    BehaviorCard,
    CandidateVersion,
    CardOperation,
    EpisodeSnapshot,
    RetrievalPolicy,
)


def build_candidate(
    *,
    episodes: tuple[ReviewedEpisode, ...],
    operations: tuple[CardOperation, ...],
    kappa_hash: str,
    parent_version_id: str = "accepted-v000",
) -> CandidateVersion:
    """Apply a validated operation batch to an empty bootstrap version and freeze it."""

    if not episodes or any(not episode.eligible_for_consolidation for episode in episodes):
        raise ValueError("candidate requires a complete eligible episode snapshot")
    ordered_episodes = tuple(sorted(episodes, key=lambda item: item.episode_id))
    if len({item.episode_id for item in ordered_episodes}) != len(ordered_episodes):
        raise ValueError("episode snapshot contains duplicate identifiers")
    episode_hashes = tuple(_hash_json(item.model_dump(mode="json")) for item in ordered_episodes)
    snapshot_hash = _hash_json(
        {"ids": [item.episode_id for item in ordered_episodes], "hashes": episode_hashes}
    )
    snapshot = EpisodeSnapshot(
        episode_ids=tuple(item.episode_id for item in ordered_episodes),
        episode_hashes=episode_hashes,
        snapshot_hash=snapshot_hash,
    )
    allowed_evidence = set(snapshot.episode_ids)
    cards: dict[str, BehaviorCard] = {}
    for operation in operations:
        if not set(operation.evidence_episode_ids) <= allowed_evidence:
            raise ValueError("card operation cites evidence outside the frozen snapshot")
        _reject_contamination(operation)
        existing = cards.get(operation.card_id)
        if operation.kind == "CREATE":
            if existing is not None or operation.trigger is None or operation.instruction is None:
                raise ValueError("invalid CREATE operation")
            cards[operation.card_id] = _card_from_operation(operation)
        elif operation.kind == "REINFORCE":
            if (
                existing is None
                or operation.trigger is not None
                or operation.instruction is not None
            ):
                raise ValueError("invalid REINFORCE operation")
            cards[operation.card_id] = existing.model_copy(
                update={
                    "evidence_episode_ids": tuple(
                        sorted(
                            set(existing.evidence_episode_ids) | set(operation.evidence_episode_ids)
                        )
                    )
                }
            )
        elif operation.kind == "REFINE":
            if existing is None or operation.trigger is None or operation.instruction is None:
                raise ValueError("invalid REFINE operation")
            cards[operation.card_id] = _card_from_operation(operation)
        else:
            if existing is None:
                raise ValueError("invalid RETIRE operation")
            del cards[operation.card_id]
    frozen_cards = tuple(cards[key] for key in sorted(cards))
    retrieval_policy = RetrievalPolicy(
        top_k=min(3, len(frozen_cards)),
        minimum_score=0.25 if frozen_cards else 0,
        token_budget=600 if frozen_cards else 0,
    )
    content = {
        "parent_version_id": parent_version_id,
        "kappa_hash": kappa_hash,
        "snapshot": snapshot.model_dump(mode="json"),
        "cards": [card.model_dump(mode="json") for card in frozen_cards],
        "operations": [item.model_dump(mode="json") for item in operations],
        "retrieval_policy": retrieval_policy.model_dump(mode="json"),
    }
    digest = _hash_json(content)
    return CandidateVersion(
        version_id=f"candidate:{digest[:16]}",
        parent_version_id=parent_version_id,
        kappa_hash=kappa_hash,
        snapshot=snapshot,
        cards=frozen_cards,
        operations=operations,
        retrieval_policy=retrieval_policy,
        content_hash=digest,
    )


def _card_from_operation(operation: CardOperation) -> BehaviorCard:
    assert operation.trigger is not None
    assert operation.instruction is not None
    return BehaviorCard(
        card_id=operation.card_id,
        trigger=operation.trigger,
        instruction=operation.instruction,
        evidence_episode_ids=tuple(sorted(set(operation.evidence_episode_ids))),
    )


def _reject_contamination(operation: CardOperation) -> None:
    content = f"{operation.trigger or ''} {operation.instruction or ''}".casefold()
    markers = ("correct answer", "ground truth", "hidden label", "chartqa:", "task_id")
    if any(marker in content for marker in markers):
        raise ValueError("candidate batch contains answer or benchmark leakage")


def _hash_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

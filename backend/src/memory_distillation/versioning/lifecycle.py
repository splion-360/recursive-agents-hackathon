"""Deterministic consolidation into a complete immutable Candidate Version."""

import hashlib
import json
import re

from memory_distillation.benchmark import ExperimentManifest
from memory_distillation.operational_memory import ReviewedEpisode
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.versioning.models import (
    BehaviorCard,
    CandidateVersion,
    CardOperation,
    EpisodeSnapshot,
    RetrievalCalibration,
    RetrievalPolicy,
    VersionView,
)
from memory_distillation.versioning.retrieval import select_behavior_cards


def build_candidate(
    *,
    episodes: tuple[ReviewedEpisode, ...],
    collection_manifest: ExperimentManifest,
    operations: tuple[CardOperation, ...],
    kappa: KappaArtifact,
    retrieval_calibration: RetrievalCalibration,
    contamination_phrases: tuple[str, ...],
    parent_view: VersionView,
) -> CandidateVersion:
    """Apply a validated operation batch to one pinned parent view and freeze it."""

    if collection_manifest.name != "collection":
        raise ValueError("candidate construction requires the Collection manifest")
    if not episodes:
        raise ValueError("candidate requires a complete Collection pass")
    ordered_pass = tuple(sorted(episodes, key=lambda item: item.episode_id))
    if len({item.episode_id for item in ordered_pass}) != len(ordered_pass):
        raise ValueError("Collection pass contains duplicate episode identifiers")
    provenances = [item.provenance for item in ordered_pass]
    first_provenance = provenances[0]
    pass_identity = {
        (
            item.lifecycle_pass_id,
            item.manifest_hash,
            item.configuration_hash,
            item.code_revision,
            item.accepted_version_id,
            item.accepted_content_hash,
            item.manifest_task_ids,
            item.purpose,
            item.kappa_hash,
        )
        for item in provenances
    }
    if len(pass_identity) != 1:
        raise ValueError("candidate snapshot mixes Collection lifecycle provenance")
    if (
        first_provenance.accepted_version_id != parent_view.version_id
        or first_provenance.accepted_content_hash != parent_view.content_hash
    ):
        raise ValueError("candidate parent does not match the Collection Accepted Version")
    manifest_ids = tuple(entry.task_id for entry in collection_manifest.entries)
    if (
        first_provenance.manifest_hash != collection_manifest.content_sha256
        or first_provenance.manifest_task_ids != manifest_ids
    ):
        raise ValueError("Collection provenance does not match the authoritative manifest")
    if first_provenance.purpose != "collection":
        raise ValueError("candidate construction accepts only Collection episodes")
    if first_provenance.kappa_hash != kappa.content_hash:
        raise ValueError("Collection Kappa does not match the frozen Kappa artifact")
    task_ids = [item.task.task_id for item in ordered_pass]
    if len(task_ids) != len(manifest_ids) or set(task_ids) != set(manifest_ids):
        raise ValueError("Collection pass does not cover the complete Collection manifest")
    if not contamination_phrases:
        raise ValueError("candidate requires explicit contamination phrases")
    ordered_episodes = tuple(item for item in ordered_pass if item.eligible_for_consolidation)
    if not ordered_episodes:
        raise ValueError("Collection pass contains no eligible consolidation evidence")
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
    if set(retrieval_calibration.episode_ids) != allowed_evidence:
        raise ValueError("retrieval calibration must cover the frozen Collection snapshot")
    if len({card.card_id for card in parent_view.cards}) != len(parent_view.cards):
        raise ValueError("parent view contains duplicate card identifiers")
    cards = {card.card_id: card for card in parent_view.cards}
    forbidden_phrases = _snapshot_forbidden_phrases(ordered_pass) + contamination_phrases
    for operation in operations:
        if not operation.evidence_episode_ids:
            raise ValueError("every card operation requires Reviewed Episode evidence")
        if not set(operation.evidence_episode_ids) <= allowed_evidence:
            raise ValueError("card operation cites evidence outside the frozen snapshot")
        _reject_contamination(operation, forbidden_phrases)
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
            replacement = _card_from_operation(operation)
            cards[operation.card_id] = replacement.model_copy(
                update={
                    "evidence_episode_ids": tuple(
                        sorted(
                            set(existing.evidence_episode_ids)
                            | set(replacement.evidence_episode_ids)
                        )
                    )
                }
            )
        else:
            if existing is None:
                raise ValueError("invalid RETIRE operation")
            del cards[operation.card_id]
    frozen_cards = tuple(cards[key] for key in sorted(cards))
    _reject_contradictory_cards(frozen_cards)
    card_ids = set(cards)
    if any(
        not set(trial.relevant_card_ids) <= card_ids
        or not set(trial.retrieved_card_ids) <= card_ids
        for trial in retrieval_calibration.trials
    ):
        raise ValueError("retrieval calibration references cards outside the frozen candidate")
    episode_by_id = {episode.episode_id: episode for episode in ordered_episodes}
    for trial in retrieval_calibration.trials:
        episode = episode_by_id[trial.episode_id]
        if trial.query_id != episode.task.task_id:
            raise ValueError("retrieval trial query does not match its Collection episode")
        trial_view = VersionView(
            version_id="retrieval-calibration",
            cards=frozen_cards,
            retrieval_policy=RetrievalPolicy(
                top_k=trial.top_k,
                minimum_score=trial.minimum_score,
                token_budget=trial.token_budget,
                calibration_snapshot_hash=retrieval_calibration.evidence_hash,
            ),
            content_hash=retrieval_calibration.evidence_hash,
        )
        observed = tuple(card.card_id for card in select_behavior_cards(trial_view, episode.task))
        if observed != trial.retrieved_card_ids:
            raise ValueError("retrieval trial does not reproduce through the frozen retriever")
    retrieval_policy = RetrievalPolicy(
        top_k=min(retrieval_calibration.top_k, len(frozen_cards)),
        minimum_score=retrieval_calibration.minimum_score,
        token_budget=retrieval_calibration.token_budget,
        calibration_snapshot_hash=retrieval_calibration.evidence_hash,
    )
    content = {
        "parent_version_id": parent_view.version_id,
        "kappa_hash": kappa.content_hash,
        "snapshot": snapshot.model_dump(mode="json"),
        "cards": [card.model_dump(mode="json") for card in frozen_cards],
        "operations": [item.model_dump(mode="json") for item in operations],
        "retrieval_policy": retrieval_policy.model_dump(mode="json"),
    }
    digest = _hash_json(content)
    return CandidateVersion(
        version_id=f"candidate:{digest[:16]}",
        parent_version_id=parent_view.version_id,
        kappa_hash=kappa.content_hash,
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


def _reject_contradictory_cards(cards: tuple[BehaviorCard, ...]) -> None:
    """Reject divergent instructions for the same normalized retrieval trigger."""

    instruction_by_trigger: dict[str, str] = {}
    for card in cards:
        trigger = " ".join(card.trigger.casefold().split())
        instruction = " ".join(card.instruction.casefold().split())
        previous = instruction_by_trigger.setdefault(trigger, instruction)
        if previous != instruction:
            raise ValueError("candidate contains contradictory cards for one trigger")


def _reject_contamination(operation: CardOperation, forbidden_phrases: tuple[str, ...]) -> None:
    content = f"{operation.trigger or ''} {operation.instruction or ''}".casefold()
    markers = ("correct answer", "ground truth", "hidden label", "chartqa:", "task_id")
    if any(marker in content for marker in markers):
        raise ValueError("candidate batch contains answer or benchmark leakage")
    if any(_contains_phrase(content, phrase) for phrase in forbidden_phrases):
        raise ValueError("candidate batch contains snapshot-specific content")


def _snapshot_forbidden_phrases(
    episodes: tuple[ReviewedEpisode, ...],
) -> tuple[str, ...]:
    phrases: list[str] = []
    for episode in episodes:
        phrases.extend((episode.task.task_id, episode.task.question, episode.task.image_asset_id))
        phrases.extend(action.answer for action in episode.actions if action.type == "final_answer")
        for observation in episode.observations:
            phrases.append(observation.content)
            phrases.extend(_sensitive_values(observation.content))
    return tuple(phrases)


def _sensitive_values(content: str) -> tuple[str, ...]:
    """Extract chart-like numeric values that must not survive consolidation."""

    return tuple(
        match.group(0)
        for match in re.finditer(r"(?<!\w)[$+-]?\d[\d,.]*(?:%|[a-zA-Z]+)?(?!\w)", content)
    )


def _contains_phrase(content: str, phrase: str) -> bool:
    normalized = " ".join(phrase.casefold().split())
    if not normalized:
        return False
    normalized_content = " ".join(content.split())
    if len(normalized) <= 3:
        return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", normalized_content) is not None
    return normalized in normalized_content


def _hash_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

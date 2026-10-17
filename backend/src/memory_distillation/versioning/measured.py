"""One frozen Collection-to-Promotion lifecycle pass."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from memory_distillation.benchmark import ExperimentManifest, TaskAsset, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.task_execution import ExecutionPolicy, TaskExecutor
from memory_distillation.versioning.lifecycle import build_candidate
from memory_distillation.versioning.models import (
    BehaviorCard,
    CandidateVersion,
    CardOperation,
    ConsolidationBatch,
    RetrievalPolicy,
    RetrievalTrial,
    VersionView,
    calibrate_retrieval,
)
from memory_distillation.versioning.promotion import (
    PairedOutcome,
    PromotionAssignment,
    PromotionDecision,
    PromotionPlan,
    PromotionRegistry,
    PromotionTask,
    PromotionThresholds,
    decide_promotion,
    run_paired_promotion,
)
from memory_distillation.versioning.retrieval import (
    retrieve_behavior_cards,
    select_behavior_cards,
)


class MeasuredTaskSource(Protocol):
    def load_collection(self) -> tuple[TaskInput, ...]: ...
    def load_promotion(self) -> tuple[TaskInput, ...]: ...
    def load_asset(self, task: TaskInput) -> TaskAsset: ...


class OperationalMemory(PromotionRegistry, Protocol):
    def initialize(self) -> None: ...
    def append_episode(self, episode: ReviewedEpisode) -> None: ...


class Consolidator(Protocol):
    def consolidate(
        self,
        *,
        episodes: tuple[ReviewedEpisode, ...],
        parent_view: VersionView,
        seed: int,
    ) -> ConsolidationBatch: ...


class LifecycleReport(BaseModel):
    """Immutable evidence emitted before the final held-out set is opened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "measured-lifecycle.v1"
    run_id: str
    code_revision: str
    configuration_hash: str
    collection_manifest_hash: str
    promotion_manifest_hash: str
    kappa_hash: str
    collection_episode_ids: tuple[str, ...] = Field(min_length=40, max_length=40)
    eligible_episode_ids: tuple[str, ...]
    consolidator_model_id: str
    consolidator_request_id: str
    candidate: CandidateVersion
    promotion_outcomes: tuple[PairedOutcome, ...] = Field(min_length=30, max_length=30)
    decision: PromotionDecision


def run_measured_lifecycle(
    *,
    run_id: str,
    code_revision: str,
    source: MeasuredTaskSource,
    collection_manifest: ExperimentManifest,
    promotion_manifest: ExperimentManifest,
    executor: TaskExecutor,
    consolidator: Consolidator,
    memory: OperationalMemory,
    kappa: KappaArtifact,
    seed: int = 17,
    thresholds: PromotionThresholds | None = None,
) -> LifecycleReport:
    """Execute exactly one immutable Collection, consolidation, and Promotion pass."""

    if collection_manifest.name != "collection" or len(collection_manifest.entries) != 40:
        raise ValueError("measured lifecycle requires the complete 40-task Collection manifest")
    if promotion_manifest.name != "promotion" or len(promotion_manifest.entries) != 30:
        raise ValueError("measured lifecycle requires the complete 30-task Promotion manifest")
    memory.initialize()
    accepted_id = memory.accepted_version_id
    accepted_view = memory.get_view(accepted_id)
    configuration_hash = _hash_json(
        {
            "code_revision": code_revision,
            "collection_manifest": collection_manifest.content_sha256,
            "promotion_manifest": promotion_manifest.content_sha256,
            "kappa_hash": kappa.content_hash,
            "accepted_version_id": accepted_id,
            "accepted_content_hash": accepted_view.content_hash,
            "seed": seed,
            "thresholds": (thresholds or PromotionThresholds()).model_dump(mode="json"),
        }
    )
    collection_tasks = source.load_collection()
    _require_task_membership(collection_tasks, collection_manifest)
    provenance = EpisodeProvenance(
        code_revision=code_revision,
        manifest_hash=collection_manifest.content_sha256,
        configuration_hash=configuration_hash,
        lifecycle_pass_id=run_id,
        purpose="collection",
        manifest_task_ids=tuple(entry.task_id for entry in collection_manifest.entries),
        accepted_version_id=accepted_id,
        accepted_content_hash=accepted_view.content_hash,
        kappa_hash=kappa.content_hash,
    )
    collection_policy = ExecutionPolicy(review_required=True, kappa=kappa)
    episodes: list[ReviewedEpisode] = []
    for index, task in enumerate(collection_tasks):
        run = executor.run(
            task=task,
            asset=source.load_asset(task),
            policy=collection_policy,
            seed=seed + index,
            behavior_cards=retrieve_behavior_cards(accepted_view, task),
        )
        episode = episode_from_run(task=task, run=run, provenance=provenance)
        memory.append_episode(episode)
        episodes.append(episode)
    frozen_episodes = tuple(episodes)
    eligible = tuple(item for item in frozen_episodes if item.eligible_for_consolidation)
    if not eligible:
        raise RuntimeError("Collection produced no eligible Reviewed Episodes")
    batch = consolidator.consolidate(
        episodes=eligible,
        parent_view=accepted_view,
        seed=seed,
    )
    _validate_operation_coverage(batch.operations, eligible, accepted_view)
    retrieval_calibration = _calibrate_candidate_retrieval(
        operations=batch.operations,
        episodes=eligible,
    )
    candidate = build_candidate(
        episodes=frozen_episodes,
        collection_manifest=collection_manifest,
        operations=batch.operations,
        kappa=kappa,
        retrieval_calibration=retrieval_calibration,
        contamination_phrases=(collection_manifest.dataset_name,),
        parent_view=accepted_view,
    )
    memory.freeze_candidate(candidate)

    promotion_tasks = source.load_promotion()
    _require_task_membership(promotion_tasks, promotion_manifest)
    assignments = tuple(
        PromotionAssignment(task_id=task.task_id, seed=seed + 10_000 + index)
        for index, task in enumerate(promotion_tasks)
    )
    plan = PromotionPlan(
        manifest=promotion_manifest,
        configuration_hash=configuration_hash,
        accepted_version_id=accepted_id,
        accepted_content_hash=accepted_view.content_hash,
        candidate_version_id=candidate.version_id,
        candidate_content_hash=candidate.content_hash,
        assignments=assignments,
    )
    paired = run_paired_promotion(
        executor=executor,
        plan=plan,
        tasks=tuple(
            PromotionTask(
                task=task,
                asset=source.load_asset(task),
                seed=assignment.seed,
            )
            for task, assignment in zip(promotion_tasks, assignments, strict=True)
        ),
        registry=memory,
        candidate=candidate,
        execution_policy=ExecutionPolicy(review_required=False),
    )
    decision = decide_promotion(
        registry=memory,
        candidate=candidate,
        plan=plan,
        outcomes=paired,
        thresholds=thresholds or PromotionThresholds(),
    )
    return LifecycleReport(
        run_id=run_id,
        code_revision=code_revision,
        configuration_hash=configuration_hash,
        collection_manifest_hash=collection_manifest.content_sha256,
        promotion_manifest_hash=promotion_manifest.content_sha256,
        kappa_hash=kappa.content_hash,
        collection_episode_ids=tuple(item.episode_id for item in frozen_episodes),
        eligible_episode_ids=tuple(item.episode_id for item in eligible),
        consolidator_model_id=batch.model_id,
        consolidator_request_id=batch.request_id,
        candidate=candidate,
        promotion_outcomes=paired,
        decision=decision,
    )


def _validate_operation_coverage(
    operations: tuple[CardOperation, ...],
    episodes: tuple[ReviewedEpisode, ...],
    parent_view: VersionView,
) -> None:
    if not operations:
        raise ValueError("Consolidator returned no card operations")
    if not parent_view.cards and any(operation.kind != "CREATE" for operation in operations):
        raise ValueError("empty parent accepts only CREATE operations")
    evidence = {
        episode_id
        for operation in operations
        for episode_id in operation.evidence_episode_ids
    }
    expected = {episode.episode_id for episode in episodes}
    if evidence != expected:
        raise ValueError("Consolidator operations must cover every eligible episode exactly by ID")


def _calibrate_candidate_retrieval(
    *,
    operations: tuple[CardOperation, ...],
    episodes: tuple[ReviewedEpisode, ...],
):
    cards = tuple(
        BehaviorCard(
            card_id=operation.card_id,
            trigger=operation.trigger or "",
            instruction=operation.instruction or "",
            evidence_episode_ids=operation.evidence_episode_ids,
        )
        for operation in operations
        if operation.kind == "CREATE"
    )
    if not cards:
        raise ValueError("initial Candidate Version requires at least one CREATE operation")
    top_k = min(3, len(cards))
    policy = RetrievalPolicy(
        top_k=top_k,
        minimum_score=0,
        token_budget=1_200,
        calibration_snapshot_hash="pending",
    )
    view = VersionView(
        version_id="retrieval-calibration",
        cards=cards,
        retrieval_policy=policy,
        content_hash="pending",
    )
    trials = []
    for episode in episodes:
        relevant = tuple(
            card.card_id for card in cards if episode.episode_id in card.evidence_episode_ids
        )
        trials.append(
            RetrievalTrial(
                query_id=episode.task.task_id,
                episode_id=episode.episode_id,
                top_k=top_k,
                minimum_score=0,
                token_budget=1_200,
                relevant_card_ids=relevant,
                retrieved_card_ids=tuple(
                    card.card_id for card in select_behavior_cards(view, episode.task)
                ),
            )
        )
    return calibrate_retrieval(
        episode_ids=tuple(episode.episode_id for episode in episodes),
        trials=tuple(trials),
    )


def _require_task_membership(
    tasks: tuple[TaskInput, ...], manifest: ExperimentManifest
) -> None:
    if tuple(task.task_id for task in tasks) != tuple(entry.task_id for entry in manifest.entries):
        raise ValueError(f"loaded tasks do not match the frozen {manifest.name} manifest")


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

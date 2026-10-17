import hashlib
import json

import pytest
from pydantic import ValidationError

from memory_distillation.benchmark import (
    ExperimentManifest,
    ManifestEntry,
    Score,
    TaskAsset,
    TaskInput,
)
from memory_distillation.inference import DeterministicInference
from memory_distillation.sandbox import DeterministicSandbox
from memory_distillation.task_execution import ExecutionPolicy, FinalAnswerAction, TaskExecutor
from memory_distillation.telemetry import InMemoryEventSink
from memory_distillation.versioning import (
    BehaviorCard,
    CandidateVersion,
    EpisodeSnapshot,
    PairedOutcome,
    PromotionAssignment,
    PromotionPlan,
    PromotionTask,
    PromotionThresholds,
    RetrievalPolicy,
    VersionRegistry,
    decide_promotion,
    run_paired_promotion,
)


class AnyAnswerVerifier:
    def score(self, task_id: str, prediction: str) -> Score:
        return Score(correct=prediction == "ok", method="text_exact")


def _question(index: int) -> str:
    return f"What chart value {index} should be checked?"


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.create(
        name="promotion",
        seed=17,
        dataset_revision="fixture",
        source_sha256="f" * 64,
        entries=tuple(
            ManifestEntry(
                task_id=f"task-{index:02d}",
                image_asset_id=f"task-{index:02d}.png",
                image_sha256=f"{index + 1:064x}",
                question_sha256=hashlib.sha256(_question(index).encode()).hexdigest(),
                source_split="val",
                question_form="wh",
                operation_type="lookup",
            )
            for index in range(30)
        ),
    )


def _candidate(
    *, with_card: bool = False, parent_version_id: str = "accepted-v000"
) -> CandidateVersion:
    snapshot = EpisodeSnapshot(episode_ids=("e1",), episode_hashes=("h1",), snapshot_hash="s")
    cards = (
        (
            BehaviorCard(
                card_id="card:what",
                trigger="What chart",
                instruction="Verify the label before arithmetic.",
                evidence_episode_ids=("episode:1",),
            ),
        )
        if with_card
        else ()
    )
    retrieval_policy = RetrievalPolicy(
        top_k=1 if cards else 0,
        minimum_score=0,
        token_budget=100 if cards else 0,
        calibration_snapshot_hash="s",
    )
    kappa_hash = "b" * 64
    payload = {
        "parent_version_id": parent_version_id,
        "kappa_hash": kappa_hash,
        "snapshot": snapshot.model_dump(mode="json"),
        "cards": [card.model_dump(mode="json") for card in cards],
        "operations": [],
        "retrieval_policy": retrieval_policy.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CandidateVersion(
        version_id=f"candidate:{digest[:16]}",
        parent_version_id=parent_version_id,
        kappa_hash=kappa_hash,
        snapshot=snapshot,
        cards=cards,
        operations=(),
        retrieval_policy=retrieval_policy,
        content_hash=digest,
    )


def _plan(candidate: CandidateVersion, manifest: ExperimentManifest | None = None) -> PromotionPlan:
    frozen_manifest = manifest or _manifest()
    return PromotionPlan(
        manifest=frozen_manifest,
        configuration_hash="frozen-config",
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        candidate_version_id=candidate.version_id,
        candidate_content_hash=candidate.content_hash,
        assignments=tuple(
            PromotionAssignment(task_id=entry.task_id, seed=17) for entry in frozen_manifest.entries
        ),
    )


def _outcomes(
    candidate: CandidateVersion, *, improve_count: int, regress_count: int = 0
) -> tuple[PairedOutcome, ...]:
    rows = []
    for index in range(30):
        accepted = index >= improve_count
        candidate_correct = True
        if improve_count <= index < improve_count + regress_count:
            accepted = True
            candidate_correct = False
        rows.append(
            PairedOutcome(
                task_id=f"task-{index:02d}",
                seed=17,
                configuration_hash="frozen-config",
                accepted_version_id="accepted-v000",
                accepted_content_hash="0" * 64,
                candidate_version_id=candidate.version_id,
                candidate_content_hash=candidate.content_hash,
                accepted_correct=accepted,
                candidate_correct=candidate_correct,
                accepted_latency_ms=100,
                candidate_latency_ms=110,
                accepted_input_tokens=100,
                candidate_input_tokens=110,
            )
        )
    return tuple(rows)


def _tasks(manifest: ExperimentManifest) -> tuple[PromotionTask, ...]:
    return tuple(
        PromotionTask(
            task=TaskInput(
                task_id=entry.task_id,
                question=_question(index),
                image_asset_id=entry.image_asset_id,
                image_sha256=entry.image_sha256,
                source_split="val",
            ),
            asset=TaskAsset(
                task_id=entry.task_id,
                media_type="image/png",
                image_sha256=entry.image_sha256,
                content=b"fixture",
            ),
            seed=17,
        )
        for index, entry in enumerate(manifest.entries)
    )


def test_promotion_advances_pointer_only_when_all_gates_pass() -> None:
    registry = VersionRegistry()
    candidate = _candidate()
    decision = decide_promotion(
        registry=registry,
        candidate=candidate,
        plan=_plan(candidate),
        outcomes=_outcomes(candidate, improve_count=2),
        thresholds=PromotionThresholds(),
    )
    assert decision.decision == "PROMOTE"
    assert registry.accepted_version_id == candidate.version_id


def test_rejection_preserves_pointer_and_frozen_candidate() -> None:
    registry = VersionRegistry()
    candidate = _candidate()
    decision = decide_promotion(
        registry=registry,
        candidate=candidate,
        plan=_plan(candidate),
        outcomes=_outcomes(candidate, improve_count=0, regress_count=1),
        thresholds=PromotionThresholds(),
    )
    assert decision.decision == "REJECT"
    assert registry.accepted_version_id == "accepted-v000"
    assert registry.get_view(candidate.version_id).content_hash == candidate.content_hash


def test_promotion_plan_rejects_non_promotion_manifest_and_repeated_assignments() -> None:
    candidate = _candidate()
    collection_data = _manifest().model_dump()
    collection_data["name"] = "collection"
    collection_data["entries"] = tuple(collection_data["entries"]) + tuple(
        collection_data["entries"][:10]
    )
    with pytest.raises((ValidationError, ValueError)):
        PromotionPlan.model_validate(
            {
                "manifest": collection_data,
                "configuration_hash": "frozen-config",
                "accepted_version_id": "accepted-v000",
                "accepted_content_hash": "0" * 64,
                "candidate_version_id": candidate.version_id,
                "candidate_content_hash": candidate.content_hash,
                "assignments": (),
            }
        )

    manifest = _manifest()
    with pytest.raises(ValidationError, match="every manifest task exactly once"):
        _plan(candidate, manifest).model_copy(
            update={
                "assignments": tuple(
                    PromotionAssignment(task_id="task-00", seed=index) for index in range(30)
                )
            }
        ).__class__.model_validate(
            _plan(candidate, manifest)
            .model_copy(
                update={
                    "assignments": tuple(
                        PromotionAssignment(task_id="task-00", seed=index) for index in range(30)
                    )
                }
            )
            .model_dump()
        )


def test_paired_runner_resolves_views_and_validates_manifest_content() -> None:
    manifest = _manifest()
    candidate = _candidate(with_card=True)
    registry = VersionRegistry()
    inference = DeterministicInference(
        actions=(FinalAnswerAction(answer="ok"),), model_id="student-v1"
    )
    executor = TaskExecutor(
        inference=inference,
        sandbox=DeterministicSandbox(results={}),
        verifier=AnyAnswerVerifier(),
        event_sink=InMemoryEventSink(),
    )
    outcomes = run_paired_promotion(
        executor=executor,
        plan=_plan(candidate, manifest),
        tasks=_tasks(manifest),
        registry=registry,
        candidate=candidate,
        execution_policy=ExecutionPolicy(review_required=False),
    )
    assert len(outcomes) == 30
    assert inference.requests[0].behavior_cards == ()
    assert inference.requests[1].behavior_cards == (
        "What chart\nVerify the label before arithmetic.",
    )

    altered = list(_tasks(manifest))
    altered[0] = PromotionTask(
        task=altered[0].task.model_copy(update={"question": "Changed question"}),
        asset=altered[0].asset,
        seed=17,
    )
    with pytest.raises(ValueError, match="authoritative manifest"):
        run_paired_promotion(
            executor=executor,
            plan=_plan(candidate, manifest),
            tasks=tuple(altered),
            registry=registry,
            candidate=candidate,
            execution_policy=ExecutionPolicy(review_required=False),
        )


def test_paired_runner_rejects_unvalidated_candidate_copy_before_mutation() -> None:
    manifest = _manifest()
    candidate = _candidate()
    injected = BehaviorCard(
        card_id="card:injected",
        trigger="What chart",
        instruction="Use unverified injected guidance.",
        evidence_episode_ids=("episode:forged",),
    )
    forged = candidate.model_copy(update={"cards": (injected,)})
    registry = VersionRegistry()
    inference = DeterministicInference(
        actions=(FinalAnswerAction(answer="ok"),), model_id="student-v1"
    )

    with pytest.raises(ValidationError, match="canonical identity"):
        run_paired_promotion(
            executor=TaskExecutor(
                inference=inference,
                sandbox=DeterministicSandbox(results={}),
                verifier=AnyAnswerVerifier(),
                event_sink=InMemoryEventSink(),
            ),
            plan=_plan(candidate, manifest),
            tasks=_tasks(manifest),
            registry=registry,
            candidate=forged,
            execution_policy=ExecutionPolicy(),
        )

    assert inference.requests == []
    with pytest.raises(KeyError):
        registry.get_view(candidate.version_id)


def test_paired_runner_rejects_noncurrent_parent_before_mutation() -> None:
    manifest = _manifest()
    candidate = _candidate(parent_version_id="accepted-v999")
    registry = VersionRegistry()
    inference = DeterministicInference(
        actions=(FinalAnswerAction(answer="ok"),), model_id="student-v1"
    )

    with pytest.raises(ValueError, match="parent"):
        run_paired_promotion(
            executor=TaskExecutor(
                inference=inference,
                sandbox=DeterministicSandbox(results={}),
                verifier=AnyAnswerVerifier(),
                event_sink=InMemoryEventSink(),
            ),
            plan=_plan(candidate, manifest),
            tasks=_tasks(manifest),
            registry=registry,
            candidate=candidate,
            execution_policy=ExecutionPolicy(),
        )

    assert inference.requests == []
    with pytest.raises(KeyError):
        registry.get_view(candidate.version_id)

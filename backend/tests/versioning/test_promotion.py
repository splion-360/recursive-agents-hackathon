import hashlib
import json

import pytest
from pydantic import ValidationError

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
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
    VersionView,
    decide_promotion,
    run_paired_promotion,
)


class AnyAnswerVerifier:
    def score(self, task_id: str, prediction: str) -> Score:
        return Score(correct=prediction == "ok", method="text_exact")


def _candidate() -> CandidateVersion:
    snapshot = EpisodeSnapshot(episode_ids=("e1",), episode_hashes=("h1",), snapshot_hash="s")
    retrieval_policy = RetrievalPolicy(
        top_k=0,
        minimum_score=0,
        token_budget=0,
        calibration_snapshot_hash="s",
    )
    payload = {
        "parent_version_id": "accepted-v000",
        "kappa_hash": "kappa",
        "snapshot": snapshot.model_dump(mode="json"),
        "cards": [],
        "operations": [],
        "retrieval_policy": retrieval_policy.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CandidateVersion(
        version_id=f"candidate:{digest[:16]}",
        parent_version_id="accepted-v000",
        kappa_hash="kappa",
        snapshot=snapshot,
        cards=(),
        operations=(),
        retrieval_policy=retrieval_policy,
        content_hash=digest,
    )


def _plan(candidate: CandidateVersion) -> PromotionPlan:
    return PromotionPlan(
        manifest_hash="promotion-manifest",
        configuration_hash="frozen-config",
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        candidate_version_id=candidate.version_id,
        candidate_content_hash=candidate.content_hash,
        manifest_task_ids=tuple(f"task-{index:02d}" for index in range(30)),
        assignments=tuple(
            PromotionAssignment(task_id=f"task-{index:02d}", seed=17) for index in range(30)
        ),
    )


def _outcomes(
    candidate: CandidateVersion,
    *,
    improve_count: int,
    regress_count: int = 0,
    latency: int = 115,
    tokens: int = 115,
) -> tuple[PairedOutcome, ...]:
    rows: list[PairedOutcome] = []
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
                candidate_latency_ms=latency,
                accepted_input_tokens=100,
                candidate_input_tokens=tokens,
            )
        )
    return tuple(rows)


def test_promotion_advances_pointer_once_at_inclusive_boundaries() -> None:
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
    assert (
        registry.compare_and_set_accepted(
            expected="accepted-v000", replacement=candidate.version_id
        )
        is False
    )


def test_rejection_preserves_both_versions_and_pointer() -> None:
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
    assert "capability" in decision.failed_gates
    assert "negative_transfer" in decision.failed_gates
    assert registry.accepted_version_id == "accepted-v000"
    assert registry.get_view(candidate.version_id).content_hash == candidate.content_hash


def test_promotion_rejects_incomplete_or_mixed_provenance() -> None:
    registry = VersionRegistry()
    candidate = _candidate()
    outcomes = _outcomes(candidate, improve_count=2)
    with pytest.raises(ValueError, match="complete frozen plan"):
        decide_promotion(
            registry=registry,
            candidate=candidate,
            plan=_plan(candidate),
            outcomes=outcomes[:-1],
            thresholds=PromotionThresholds(),
        )
    mixed = (outcomes[0].model_copy(update={"configuration_hash": "changed"}), *outcomes[1:])
    with pytest.raises(ValueError, match="provenance"):
        decide_promotion(
            registry=registry,
            candidate=candidate,
            plan=_plan(candidate),
            outcomes=mixed,
            thresholds=PromotionThresholds(),
        )


def test_promotion_plan_rejects_repeated_task_with_different_seeds() -> None:
    candidate = _candidate()
    with pytest.raises(ValidationError, match="every manifest task exactly once"):
        PromotionPlan(
            manifest_hash="promotion-manifest",
            configuration_hash="frozen-config",
            accepted_version_id="accepted-v000",
            accepted_content_hash="0" * 64,
            candidate_version_id=candidate.version_id,
            candidate_content_hash=candidate.content_hash,
            manifest_task_ids=tuple(f"task-{index:02d}" for index in range(30)),
            assignments=tuple(
                PromotionAssignment(task_id="task-00", seed=index) for index in range(30)
            ),
        )


def test_paired_runner_uses_same_executor_and_pinned_views() -> None:
    tasks = tuple(
        PromotionTask(
            task=TaskInput(
                task_id=f"task-{index:02d}",
                question="What chart value should be checked?",
                image_asset_id=f"task-{index:02d}.png",
                image_sha256=f"{index + 1:064x}",
                source_split="val",
            ),
            asset=TaskAsset(
                task_id=f"task-{index:02d}",
                media_type="image/png",
                image_sha256=f"{index + 1:064x}",
                content=b"fixture",
            ),
            seed=17,
        )
        for index in range(30)
    )
    candidate_view = VersionView(
        version_id="candidate:view",
        cards=(
            BehaviorCard(
                card_id="card:what",
                trigger="What chart",
                instruction="Verify the label before arithmetic.",
                evidence_episode_ids=("episode:1",),
            ),
        ),
        retrieval_policy=RetrievalPolicy(
            top_k=1,
            minimum_score=0,
            token_budget=100,
            calibration_snapshot_hash="snapshot",
        ),
        content_hash="c" * 64,
    )
    accepted_view = VersionRegistry().get_view("accepted-v000")
    plan = PromotionPlan(
        manifest_hash="promotion-manifest",
        configuration_hash="frozen-config",
        accepted_version_id=accepted_view.version_id,
        accepted_content_hash=accepted_view.content_hash,
        candidate_version_id=candidate_view.version_id,
        candidate_content_hash=candidate_view.content_hash,
        manifest_task_ids=tuple(item.task.task_id for item in tasks),
        assignments=tuple(
            PromotionAssignment(task_id=item.task.task_id, seed=item.seed) for item in tasks
        ),
    )
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
        plan=plan,
        tasks=tasks,
        accepted_view=accepted_view,
        candidate_view=candidate_view,
        execution_policy=ExecutionPolicy(review_required=False),
    )

    assert len(outcomes) == 30
    assert inference.requests[0].behavior_cards == ()
    assert inference.requests[1].behavior_cards == (
        "What chart\nVerify the label before arithmetic.",
    )

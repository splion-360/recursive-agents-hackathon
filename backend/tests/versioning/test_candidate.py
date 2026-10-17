import pytest
from pydantic import ValidationError

from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.task_execution import ActionReview, FinalAnswerAction, TaskRun
from memory_distillation.versioning import (
    BehaviorCard,
    CardOperation,
    RetrievalCalibration,
    RetrievalPolicy,
    RetrievalTrial,
    VersionView,
    build_candidate,
    calibrate_retrieval,
)


def _eligible_episode() -> ReviewedEpisode:
    task = TaskInput(
        task_id="chartqa:fixture:001",
        question="Total?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    action = FinalAnswerAction(answer="42")
    run = TaskRun(
        task_id=task.task_id,
        seed=1,
        termination_reason="completed",
        actions=(action,),
        observations=(),
        reviews=(
            ActionReview(
                decision="pass",
                proposed_action=action,
                accepted_action=action,
                model_id="reviewer-v1",
                request_id="review:1",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            ),
        ),
        events=(),
        score=Score(correct=True, method="text_exact"),
        input_tokens=1,
        output_tokens=1,
        elapsed_ms=1,
        tool_calls=0,
        python_calls=0,
        review_calls=1,
    )
    return episode_from_run(
        task=task,
        run=run,
        provenance=EpisodeProvenance(
            code_revision="abc",
            manifest_hash="manifest",
            configuration_hash="config",
            lifecycle_pass_id="pass-1",
            purpose="collection",
            manifest_task_ids=(task.task_id,),
            accepted_version_id="accepted-v000",
            accepted_content_hash="0" * 64,
        ),
    )


def _parent_view(*cards: BehaviorCard) -> VersionView:
    return VersionView(
        version_id="accepted-v000",
        cards=cards,
        retrieval_policy=RetrievalPolicy(
            top_k=0,
            minimum_score=0,
            token_budget=0,
            calibration_snapshot_hash="0" * 64,
        ),
        content_hash="0" * 64,
    )


def _calibration(episode: ReviewedEpisode, card_id: str) -> RetrievalCalibration:
    return calibrate_retrieval(
        episode_ids=(episode.episode_id,),
        trials=(
            RetrievalTrial(
                query_id=episode.task.task_id,
                episode_id=episode.episode_id,
                top_k=1,
                minimum_score=0.4,
                token_budget=200,
                relevant_card_ids=(card_id,),
                retrieved_card_ids=(card_id,),
            ),
        ),
    )


def test_candidate_freezes_exact_snapshot_and_deterministic_operations() -> None:
    episode = _eligible_episode()
    operations = (
        CardOperation(
            kind="CREATE",
            card_id="card:verify-labels",
            trigger="When chart labels are visually ambiguous",
            instruction="Inspect the relevant chart region before computing.",
            evidence_episode_ids=(episode.episode_id,),
        ),
        CardOperation(
            kind="REINFORCE",
            card_id="card:verify-labels",
            evidence_episode_ids=(episode.episode_id,),
        ),
    )
    calibration = _calibration(episode, "card:verify-labels")
    first = build_candidate(
        episodes=(episode,),
        operations=operations,
        kappa_hash="kappa",
        retrieval_calibration=calibration,
        contamination_phrases=("fixture-answer",),
        parent_view=_parent_view(),
    )
    second = build_candidate(
        episodes=(episode,),
        operations=operations,
        kappa_hash="kappa",
        retrieval_calibration=calibration,
        contamination_phrases=("fixture-answer",),
        parent_view=_parent_view(),
    )

    assert first == second
    assert first.snapshot.episode_ids == (episode.episode_id,)
    assert first.cards[0].evidence_episode_ids == (episode.episode_id,)
    assert len(first.content_hash) == 64
    assert first.retrieval_policy.minimum_score == 0.4


def test_candidate_rejects_entire_contaminated_batch() -> None:
    episode = _eligible_episode()
    operations = (
        CardOperation(
            kind="CREATE",
            card_id="safe",
            trigger="When arithmetic is needed",
            instruction="Use Python for arithmetic.",
            evidence_episode_ids=(episode.episode_id,),
        ),
        CardOperation(
            kind="CREATE",
            card_id="leak",
            trigger="For chartqa:fixture:001",
            instruction="Use the correct answer.",
            evidence_episode_ids=(episode.episode_id,),
        ),
    )
    with pytest.raises(ValueError, match="leakage"):
        build_candidate(
            episodes=(episode,),
            operations=operations,
            kappa_hash="kappa",
            retrieval_calibration=_calibration(episode, "safe"),
            contamination_phrases=("fixture-answer",),
            parent_view=_parent_view(),
        )


def test_candidate_preserves_parent_and_rejects_copied_snapshot_content() -> None:
    episode = _eligible_episode()
    parent = BehaviorCard(
        card_id="existing",
        trigger="When labels are ambiguous",
        instruction="Inspect the relevant region.",
        evidence_episode_ids=("older-episode",),
    )
    calibration = _calibration(episode, parent.card_id)
    refined = build_candidate(
        episodes=(episode,),
        parent_view=_parent_view(parent),
        operations=(
            CardOperation(
                kind="REFINE",
                card_id=parent.card_id,
                trigger="When chart labels are ambiguous",
                instruction="Inspect, crop, and verify the relevant region.",
                evidence_episode_ids=(episode.episode_id,),
            ),
        ),
        kappa_hash="kappa",
        retrieval_calibration=calibration,
        contamination_phrases=("fixture-answer",),
    )
    assert refined.cards[0].evidence_episode_ids == (
        episode.episode_id,
        "older-episode",
    )
    assert parent.instruction == "Inspect the relevant region."

    with pytest.raises(ValueError, match="snapshot-specific"):
        build_candidate(
            episodes=(episode,),
            operations=(
                CardOperation(
                    kind="CREATE",
                    card_id="copied",
                    trigger=episode.task.question,
                    instruction="For fixture.png, return 42.",
                    evidence_episode_ids=(episode.episode_id,),
                ),
            ),
            kappa_hash="kappa",
            retrieval_calibration=calibration,
            contamination_phrases=("42",),
            parent_view=_parent_view(),
        )


def test_retrieval_calibration_is_derived_and_candidate_binds_parent() -> None:
    episode = _eligible_episode()
    calibration = _calibration(episode, "card")
    with pytest.raises(ValidationError, match="deterministic trial selection"):
        calibration.model_copy(update={"top_k": 2}).__class__.model_validate(
            calibration.model_copy(update={"top_k": 2}).model_dump()
        )

    operation = CardOperation(
        kind="CREATE",
        card_id="card",
        trigger="When labels are ambiguous",
        instruction="Inspect the relevant region.",
        evidence_episode_ids=(episode.episode_id,),
    )
    wrong_parent = _parent_view().model_copy(update={"content_hash": "f" * 64})
    with pytest.raises(ValueError, match="candidate parent"):
        build_candidate(
            episodes=(episode,),
            operations=(operation,),
            kappa_hash="kappa",
            retrieval_calibration=calibration,
            contamination_phrases=("fixture-answer",),
            parent_view=wrong_parent,
        )


def test_complete_collection_pass_can_contain_ineligible_runs() -> None:
    eligible = _eligible_episode()
    manifest_ids = (eligible.task.task_id, "chartqa:fixture:002")
    provenance = eligible.provenance.model_copy(update={"manifest_task_ids": manifest_ids})
    eligible = eligible.model_copy(update={"provenance": provenance})
    failed = eligible.model_copy(
        update={
            "episode_id": "episode:failed",
            "task": eligible.task.model_copy(
                update={"task_id": manifest_ids[1], "image_asset_id": "fixture-2.png"}
            ),
            "eligible_for_consolidation": False,
            "termination_reason": "inference_failed",
            "provenance": provenance,
        }
    )
    operation = CardOperation(
        kind="CREATE",
        card_id="card",
        trigger="When labels are ambiguous",
        instruction="Inspect the relevant region.",
        evidence_episode_ids=(eligible.episode_id,),
    )
    candidate = build_candidate(
        episodes=(eligible, failed),
        operations=(operation,),
        kappa_hash="kappa",
        retrieval_calibration=_calibration(eligible, "card"),
        contamination_phrases=("fixture-answer",),
        parent_view=_parent_view(),
    )
    assert candidate.snapshot.episode_ids == (eligible.episode_id,)


def test_retrieval_calibration_rejects_duplicate_policy_query_trials() -> None:
    episode = _eligible_episode()
    trial = RetrievalTrial(
        query_id=episode.task.task_id,
        episode_id=episode.episode_id,
        top_k=1,
        minimum_score=0.4,
        token_budget=100,
        relevant_card_ids=("card",),
        retrieved_card_ids=("card",),
    )
    with pytest.raises(ValueError, match="duplicate policy-query"):
        calibrate_retrieval(
            episode_ids=(episode.episode_id,),
            trials=(trial, trial.model_copy(update={})),
        )

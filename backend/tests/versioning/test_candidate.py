import pytest

from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.task_execution import ActionReview, FinalAnswerAction, TaskRun
from memory_distillation.versioning import CardOperation, build_candidate


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
        reviews=(ActionReview(decision="pass", proposed_action=action, accepted_action=action),),
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
            code_revision="abc", manifest_hash="manifest", configuration_hash="config"
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
    first = build_candidate(episodes=(episode,), operations=operations, kappa_hash="kappa")
    second = build_candidate(episodes=(episode,), operations=operations, kappa_hash="kappa")

    assert first == second
    assert first.snapshot.episode_ids == (episode.episode_id,)
    assert first.cards[0].evidence_episode_ids == (episode.episode_id,)
    assert len(first.content_hash) == 64


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
        build_candidate(episodes=(episode,), operations=operations, kappa_hash="kappa")

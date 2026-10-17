from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    InMemoryOperationalMemory,
    episode_from_run,
)
from memory_distillation.task_execution import (
    ActionReview,
    FinalAnswerAction,
    InspectImageAction,
    TaskRun,
)

KAPPA_HASH = "b" * 64


def test_episode_write_is_idempotent_and_failed_runs_are_ineligible() -> None:
    task = TaskInput(
        task_id="chartqa:fixture:001",
        question="What is the total?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    action = FinalAnswerAction(answer="42")
    run = TaskRun(
        task_id=task.task_id,
        seed=17,
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
        input_tokens=10,
        output_tokens=3,
        elapsed_ms=5,
        tool_calls=0,
        python_calls=0,
        review_calls=1,
        kappa_hash=KAPPA_HASH,
    )
    provenance = EpisodeProvenance(
        code_revision="abc123",
        manifest_hash="manifest",
        configuration_hash="config",
        lifecycle_pass_id="pass-1",
        purpose="collection",
        manifest_task_ids=(task.task_id,),
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        kappa_hash=KAPPA_HASH,
    )
    episode = episode_from_run(task=task, run=run, provenance=provenance)
    memory = InMemoryOperationalMemory()
    memory.append_episode(episode)
    memory.append_episode(episode)

    assert memory.get_episode(episode.episode_id) == episode
    assert memory.snapshot_eligible_episodes() == (episode,)

    failed = TaskRun.model_validate(
        {**run.model_dump(), "termination_reason": "reviewer_failed", "score": None}
    )
    failed_episode = episode_from_run(task=task, run=failed, provenance=provenance)
    assert failed_episode.eligible_for_consolidation is False

    for purpose in ("calibration", "promotion", "final"):
        excluded = episode_from_run(
            task=task,
            run=run,
            provenance=provenance.model_copy(update={"purpose": purpose}),
        )
        assert excluded.eligible_for_consolidation is False


def test_direct_answer_replacement_is_ineligible_for_consolidation() -> None:
    task = TaskInput(
        task_id="chartqa:fixture:direct",
        question="What is the total?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    proposed = InspectImageAction()
    accepted = FinalAnswerAction(answer="42")
    run = TaskRun(
        task_id=task.task_id,
        seed=17,
        termination_reason="completed",
        actions=(accepted,),
        observations=(),
        reviews=(
            ActionReview(
                decision="replace",
                proposed_action=proposed,
                accepted_action=accepted,
                model_id="reviewer-v1",
                request_id="review:direct",
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
        kappa_hash=KAPPA_HASH,
    )
    provenance = EpisodeProvenance(
        code_revision="abc",
        manifest_hash="manifest",
        configuration_hash="config",
        lifecycle_pass_id="pass-1",
        purpose="collection",
        manifest_task_ids=(task.task_id,),
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        kappa_hash=KAPPA_HASH,
    )

    assert (
        episode_from_run(task=task, run=run, provenance=provenance).eligible_for_consolidation
        is False
    )

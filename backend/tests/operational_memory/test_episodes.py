from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    InMemoryOperationalMemory,
    episode_from_run,
)
from memory_distillation.task_execution import (
    ActionReview,
    FinalAnswerAction,
    TaskRun,
)


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
        reviews=(ActionReview(decision="pass", proposed_action=action, accepted_action=action),),
        events=(),
        score=Score(correct=True, method="text_exact"),
        input_tokens=10,
        output_tokens=3,
        elapsed_ms=5,
        tool_calls=0,
        python_calls=0,
        review_calls=1,
    )
    provenance = EpisodeProvenance(
        code_revision="abc123", manifest_hash="manifest", configuration_hash="config"
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

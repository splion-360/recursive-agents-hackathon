import pytest

from memory_distillation.benchmark import Score, TaskAsset, TaskInput
from memory_distillation.evaluation import (
    C3Assignment,
    EvaluationTask,
    frozen_condition_policies,
    run_offline_conditions,
)
from memory_distillation.inference import DeterministicInference
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.sandbox import DeterministicSandbox
from memory_distillation.task_execution import (
    DeterministicReviewer,
    ExecutionPolicy,
    FinalAnswerAction,
    ReviewResult,
    TaskExecutor,
)
from memory_distillation.telemetry import InMemoryEventSink
from memory_distillation.versioning import BehaviorCard, RetrievalPolicy, VersionView


class AnyAnswerVerifier:
    def score(self, task_id: str, prediction: str) -> Score:
        return Score(correct=prediction == "ok", method="text_exact")


def _kappa() -> KappaArtifact:
    return KappaArtifact.model_construct(
        content_hash="a" * 64,
        guidance=("Inspect relevant chart labels before computing.",),
    )


def test_conditions_share_executor_but_isolate_reviewer_and_memory() -> None:
    task = TaskInput(
        task_id="final-task-1",
        question="What chart label should be checked?",
        image_asset_id="final-task-1.png",
        image_sha256="a" * 64,
        source_split="test",
    )
    evaluation_task = EvaluationTask(
        task=task,
        asset=TaskAsset(
            task_id=task.task_id,
            media_type="image/png",
            image_sha256=task.image_sha256,
            content=b"fixture",
        ),
        seed=17,
    )
    inference = DeterministicInference(
        actions=(FinalAnswerAction(answer="ok"),), model_id="student-v1"
    )
    final_action = FinalAnswerAction(answer="ok")
    reviewer = DeterministicReviewer(
        results=(
            ReviewResult(
                decision="pass",
                accepted_action=final_action,
                private_reason="private",
                model_id="reviewer-v1",
                request_id="review:1",
                input_tokens=2,
                output_tokens=1,
                latency_ms=2,
            ),
        )
    )
    executor = TaskExecutor(
        inference=inference,
        sandbox=DeterministicSandbox(results={}),
        verifier=AnyAnswerVerifier(),
        event_sink=InMemoryEventSink(),
        reviewer=reviewer,
    )
    accepted_card = BehaviorCard(
        card_id="accepted-card",
        trigger="What chart",
        instruction="Verify the relevant label.",
        evidence_episode_ids=("episode:accepted",),
    )
    donor_card = BehaviorCard(
        card_id="donor-card",
        trigger="Weather table",
        instruction="Check units before arithmetic.",
        evidence_episode_ids=("episode:donor",),
    )
    accepted_view = VersionView(
        version_id="accepted-v001",
        cards=(accepted_card, donor_card),
        retrieval_policy=RetrievalPolicy(
            top_k=1,
            minimum_score=0.1,
            token_budget=100,
            calibration_snapshot_hash="snapshot",
        ),
        content_hash="a" * 64,
    )

    results = run_offline_conditions(
        executor=executor,
        tasks=(evaluation_task,),
        policies=frozen_condition_policies(
            configuration_hash="frozen",
            execution=ExecutionPolicy(),
            kappa=_kappa(),
        ),
        accepted_view=accepted_view,
        c3_assignments=(
            C3Assignment(task_id=task.task_id, card_ids=(donor_card.card_id,), token_count=6),
        ),
    )

    assert [item.condition for item in results] == ["C0", "C1", "C2", "C3"]
    assert len(reviewer.requests) == 1
    assert inference.requests[0].behavior_cards == ()
    assert inference.requests[1].behavior_cards == ()
    assert inference.requests[2].behavior_cards == ("What chart\nVerify the relevant label.",)
    assert inference.requests[3].behavior_cards == (
        "Weather table\nCheck units before arithmetic.",
    )

    inference.requests.clear()
    with pytest.raises(ValueError, match="token provenance"):
        run_offline_conditions(
            executor=executor,
            tasks=(evaluation_task,),
            policies=frozen_condition_policies(
                configuration_hash="frozen",
                execution=ExecutionPolicy(),
                kappa=_kappa(),
            ),
            accepted_view=accepted_view,
            c3_assignments=(
                C3Assignment(
                    task_id=task.task_id,
                    card_ids=(donor_card.card_id,),
                    token_count=999,
                ),
            ),
        )
    assert inference.requests == []

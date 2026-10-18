import hashlib
import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    MemMachineError,
    MemMachineOperationalMemory,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.task_execution import ActionReview, FinalAnswerAction, TaskRun
from memory_distillation.versioning import (
    BehaviorCard,
    CandidateVersion,
    CardOperation,
    EpisodeSnapshot,
    RetrievalPolicy,
)


def _episode() -> tuple[TaskInput, ReviewedEpisode]:
    task = TaskInput(
        task_id="chartqa:fixture:managed",
        question="Which category has the largest value?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    action = FinalAnswerAction(answer="North")
    run = TaskRun(
        task_id=task.task_id,
        seed=7,
        termination_reason="completed",
        actions=(action,),
        observations=(),
        reviews=(
            ActionReview(
                decision="pass",
                proposed_action=action,
                accepted_action=action,
                model_id="reviewer-v1",
                request_id="review-1",
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
        kappa_hash="b" * 64,
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
        kappa_hash="b" * 64,
    )
    return task, episode_from_run(task=task, run=run, provenance=provenance)


def _transport() -> tuple[httpx2.MockTransport, list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content or b"{}")
        if request.url.path == "/v2/projects":
            return httpx2.Response(201, json={"project_id": payload["project_id"]})
        if request.url.path == "/v2/memories":
            message = payload["messages"][0]
            uid = f"uid-{len(memories) + 1}"
            memories.append(
                {
                    "uid": uid,
                    "content": message["content"],
                    "filterable_metadata": message["metadata"],
                }
            )
            return httpx2.Response(200, json={"results": [{"uid": uid}]})
        if request.url.path == "/v2/memories/list":
            record_type = payload["filter"].split("=", 1)[1]
            filtered = [
                item
                for item in memories
                if item["filterable_metadata"]["md_record_type"] == record_type
            ]
            start = payload["page_num"] * payload["page_size"]
            content = filtered[start : start + payload["page_size"]]
            return httpx2.Response(
                200,
                json={"status": 0, "content": {"episodic_memory": content}},
            )
        raise AssertionError(f"unexpected request: {request.url.path}")

    return httpx2.MockTransport(handler), memories


def _client(transport: httpx2.MockTransport) -> MemMachineOperationalMemory:
    return MemMachineOperationalMemory(
        base_url="https://memory.example.test",
        api_key=SecretStr("fixture-secret"),
        project_id="fixture-project",
        page_size=1,
        transport=transport,
    )


def _candidate(episode_id: str) -> CandidateVersion:
    card = BehaviorCard(
        card_id="card:largest",
        trigger="largest value category",
        instruction="Compare all plotted values before answering.",
        evidence_episode_ids=(episode_id,),
    )
    operation = CardOperation(
        kind="CREATE",
        card_id=card.card_id,
        trigger=card.trigger,
        instruction=card.instruction,
        evidence_episode_ids=card.evidence_episode_ids,
    )
    snapshot = EpisodeSnapshot(
        episode_ids=(episode_id,),
        episode_hashes=("e" * 64,),
        snapshot_hash="f" * 64,
    )
    retrieval_policy = RetrievalPolicy(
        top_k=1,
        minimum_score=0,
        token_budget=20,
        calibration_snapshot_hash="c" * 64,
    )
    payload = {
        "parent_version_id": "accepted-v000",
        "kappa_hash": "b" * 64,
        "snapshot": snapshot.model_dump(mode="json"),
        "cards": [card.model_dump(mode="json")],
        "operations": [operation.model_dump(mode="json")],
        "retrieval_policy": retrieval_policy.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CandidateVersion(
        version_id=f"candidate:{digest[:16]}",
        parent_version_id="accepted-v000",
        kappa_hash="b" * 64,
        snapshot=snapshot,
        cards=(card,),
        operations=(operation,),
        retrieval_policy=retrieval_policy,
        content_hash=digest,
    )


def test_managed_records_round_trip_idempotently_across_pages() -> None:
    transport, memories = _transport()
    task, episode = _episode()

    with _client(transport) as memory:
        memory.initialize()
        memory.append_episode(episode)
        memory.append_episode(episode)
        assert memory.get_episode(episode.episode_id) == episode
        assert memory.snapshot_collection_episodes() == (episode,)
        assert memory.snapshot_eligible_episodes() == (episode,)

        candidate = _candidate(episode.episode_id)
        view = memory.freeze_candidate(candidate)
        memory.freeze_candidate(candidate)
        assert memory.get_candidate(candidate.version_id) == candidate
        assert memory.get_view(view.version_id) == view
        assert memory.retrieve(version_id=view.version_id, task=task) == (
            "largest value category\nCompare all plotted values before answering.",
        )
        assert memory.accepted_version_id == "accepted-v000"
        assert memory.compare_and_set_accepted(
            expected="accepted-v000", replacement=view.version_id
        )
        assert memory.accepted_version_id == view.version_id
        assert not memory.compare_and_set_accepted(
            expected="accepted-v000", replacement=view.version_id
        )

    assert len(memories) == 6


def test_managed_record_identity_rejects_changed_content_and_duplicates() -> None:
    transport, memories = _transport()
    _, episode = _episode()

    with _client(transport) as memory:
        memory.initialize()
        memory.append_episode(episode)
        changed = episode.model_copy(update={"correct": False})
        with pytest.raises(ValueError, match="different immutable content"):
            memory.append_episode(changed)

        memories.append(dict(memories[-1], uid="duplicate"))
        with pytest.raises(MemMachineError, match="identity is duplicated"):
            memory.get_episode(episode.episode_id)


def test_provider_failures_are_sanitized() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, text="secret provider payload")

    with (
        _client(httpx2.MockTransport(handler)) as memory,
        pytest.raises(MemMachineError) as error,
    ):
        memory.initialize()

    assert "secret provider payload" not in str(error.value)

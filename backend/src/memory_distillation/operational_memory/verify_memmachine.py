"""Live round-trip verification for managed MemMachine Operational Memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Sequence

import httpx2
from pydantic import BaseModel, ConfigDict

from memory_distillation.benchmark import Score, TaskInput
from memory_distillation.operational_memory.memmachine import (
    MemMachineAtomicityError,
    MemMachineOperationalMemory,
)
from memory_distillation.operational_memory.models import (
    EpisodeProvenance,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.settings import get_settings
from memory_distillation.task_execution import ActionReview, FinalAnswerAction, TaskRun
from memory_distillation.versioning import (
    BehaviorCard,
    CandidateVersion,
    CardOperation,
    EpisodeSnapshot,
    RetrievalPolicy,
)


class MemMachineReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def verify_live(memory: MemMachineOperationalMemory) -> MemMachineReport:
    """Exercise typed storage, pagination, retrieval, immutability, and CAS blocking."""

    checks: dict[str, bool] = {}
    memory.initialize()
    checks["project_initialized"] = memory.accepted_version_id == "accepted-v000"

    first_task, first = _episode(1)
    _, second = _episode(2)
    memory.append_episode(first)
    memory.append_episode(first)
    memory.append_episode(second)
    checks["episode_round_trip"] = memory.get_episode(first.episode_id) == first
    snapshot = memory.snapshot_eligible_episodes()
    checks["idempotent_paginated_snapshot"] = snapshot == tuple(
        sorted((first, second), key=lambda item: item.episode_id)
    )

    candidate = _candidate(first.episode_id)
    view = memory.freeze_candidate(candidate)
    memory.freeze_candidate(candidate)
    checks["immutable_version_round_trip"] = (
        memory.get_candidate(candidate.version_id) == candidate
        and memory.get_view(view.version_id) == view
    )
    checks["pinned_answer_free_retrieval"] = memory.retrieve(
        version_id=view.version_id,
        task=first_task,
    ) == ("largest bar category\nCompare every plotted bar before choosing the maximum.",)
    try:
        memory.compare_and_set_accepted(
            expected="accepted-v000",
            replacement=view.version_id,
        )
    except MemMachineAtomicityError:
        checks["non_atomic_promotion_blocked"] = True
    else:
        checks["non_atomic_promotion_blocked"] = False
    return MemMachineReport(checks=checks)


def _candidate(episode_id: str) -> CandidateVersion:
    card = BehaviorCard(
        card_id="card:compare-bars",
        trigger="largest bar category",
        instruction="Compare every plotted bar before choosing the maximum.",
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
        minimum_score=0.1,
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


def _episode(index: int) -> tuple[TaskInput, ReviewedEpisode]:
    task = TaskInput(
        task_id=f"chartqa:ticket13:{index}",
        question="Which category has the largest bar?",
        image_asset_id=f"ticket13-{index}.png",
        image_sha256=f"{index:x}" * 64,
        source_split="train",
    )
    action = FinalAnswerAction(answer="North")
    run = TaskRun(
        task_id=task.task_id,
        seed=index,
        termination_reason="completed",
        actions=(action,),
        observations=(),
        reviews=(
            ActionReview(
                decision="pass",
                proposed_action=action,
                accepted_action=action,
                model_id="ticket13-reviewer",
                request_id=f"ticket13-review-{index}",
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
        code_revision="ticket13-smoke",
        manifest_hash="ticket13-smoke-manifest",
        configuration_hash="ticket13-smoke-config",
        lifecycle_pass_id="ticket13-smoke",
        purpose="collection",
        manifest_task_ids=(task.task_id,),
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        kappa_hash="b" * 64,
    )
    return task, episode_from_run(task=task, run=run, provenance=provenance)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_live:
        parser.error("--confirm-live is required")

    settings = get_settings()
    if settings.memmachine_api_key is None:
        parser.error("MEMMACHINE_API_KEY is required")
    smoke_project = f"md-ticket13-smoke-{uuid.uuid4().hex[:12]}"
    try:
        with MemMachineOperationalMemory(
            base_url=str(settings.memmachine_base_url),
            api_key=settings.memmachine_api_key,
            project_id=smoke_project,
            page_size=1,
        ) as memory:
            report = verify_live(memory)
    finally:
        _delete_smoke_project(
            base_url=str(settings.memmachine_base_url),
            api_key=settings.memmachine_api_key.get_secret_value(),
            project_id=smoke_project,
        )
    print(json.dumps(report.model_dump(), sort_keys=True))
    return 0 if report.passed else 1


def _delete_smoke_project(*, base_url: str, api_key: str, project_id: str) -> None:
    if not project_id.startswith("md-ticket13-smoke-"):
        raise ValueError("refusing to delete a non-smoke MemMachine project")
    with httpx2.Client(
        base_url=base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    ) as client:
        response = client.post("v2/projects/delete", json={"project_id": project_id})
    if response.status_code not in {204, 404}:
        raise RuntimeError(f"MemMachine smoke cleanup failed (status={response.status_code})")


if __name__ == "__main__":
    raise SystemExit(main())

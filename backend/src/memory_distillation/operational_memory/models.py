"""Typed records owned by Operational Memory."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from memory_distillation.benchmark import TaskInput
from memory_distillation.task_execution.models import (
    ActionReview,
    Observation,
    StudentAction,
    TaskRun,
)


class EpisodeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code_revision: str
    manifest_hash: str
    configuration_hash: str


class ReviewedEpisode(BaseModel):
    """Sanitized, answer-free improvement record for one TaskRun."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["reviewed-episode.v1"] = "reviewed-episode.v1"
    episode_id: str
    task: TaskInput
    actions: tuple[StudentAction, ...]
    reviews: tuple[ActionReview, ...]
    observations: tuple[Observation, ...]
    correct: bool | None
    termination_reason: str
    model_ids: tuple[str, ...]
    eligible_for_consolidation: bool
    provenance: EpisodeProvenance


def episode_from_run(
    *, task: TaskInput, run: TaskRun, provenance: EpisodeProvenance
) -> ReviewedEpisode:
    """Project a public run into an idempotently addressed memory record."""

    if task.task_id != run.task_id:
        raise ValueError("task and run identifiers differ")
    model_ids = tuple(
        dict.fromkeys(event.model_id for event in run.events if event.type == "action_proposed")
    )
    identity = json.dumps(
        {
            "task_id": task.task_id,
            "seed": run.seed,
            "configuration_hash": provenance.configuration_hash,
            "manifest_hash": provenance.manifest_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    episode_id = "episode:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    eligible = run.termination_reason == "completed" and len(run.reviews) == len(run.actions)
    return ReviewedEpisode(
        episode_id=episode_id,
        task=task,
        actions=run.actions,
        reviews=run.reviews,
        observations=run.observations,
        correct=run.score.correct if run.score else None,
        termination_reason=run.termination_reason,
        model_ids=model_ids,
        eligible_for_consolidation=eligible,
        provenance=provenance,
    )

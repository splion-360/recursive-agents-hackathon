"""Typed records owned by Operational Memory."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    lifecycle_pass_id: str = Field(min_length=1)
    purpose: Literal["calibration", "collection", "promotion", "final"]
    manifest_task_ids: tuple[str, ...]
    accepted_version_id: str = Field(min_length=1)
    accepted_content_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_manifest_membership(self) -> "EpisodeProvenance":
        if not self.manifest_task_ids or len(set(self.manifest_task_ids)) != len(
            self.manifest_task_ids
        ):
            raise ValueError("episode provenance requires unique manifest task identifiers")
        return self


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
    student_models = (event.model_id for event in run.events if event.type == "action_proposed")
    reviewer_models = (
        event.review.model_id for event in run.events if event.type == "action_reviewed"
    )
    model_ids = tuple(dict.fromkeys((*student_models, *reviewer_models)))
    identity = json.dumps(
        {
            "task_id": task.task_id,
            "seed": run.seed,
            "configuration_hash": provenance.configuration_hash,
            "manifest_hash": provenance.manifest_hash,
            "lifecycle_pass_id": provenance.lifecycle_pass_id,
            "purpose": provenance.purpose,
            "accepted_version_id": provenance.accepted_version_id,
            "accepted_content_hash": provenance.accepted_content_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    episode_id = "episode:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    eligible = (
        provenance.purpose == "collection"
        and task.task_id in provenance.manifest_task_ids
        and run.termination_reason == "completed"
        and len(run.reviews) == len(run.actions)
    )
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

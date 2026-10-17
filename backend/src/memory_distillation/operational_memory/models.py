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
    kappa_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

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
    seed: int
    task: TaskInput
    actions: tuple[StudentAction, ...]
    reviews: tuple[ActionReview, ...]
    observations: tuple[Observation, ...]
    correct: bool | None
    termination_reason: str
    model_ids: tuple[str, ...]
    eligible_for_consolidation: bool
    provenance: EpisodeProvenance

    @model_validator(mode="after")
    def verify_derived_fields(self) -> "ReviewedEpisode":
        if self.episode_id != _episode_id(self.task.task_id, self.seed, self.provenance):
            raise ValueError("Reviewed Episode identity does not match its provenance")
        expected_eligibility = _is_eligible(
            task=self.task,
            actions=self.actions,
            reviews=self.reviews,
            termination_reason=self.termination_reason,
            provenance=self.provenance,
        )
        if self.eligible_for_consolidation != expected_eligibility:
            raise ValueError("Reviewed Episode eligibility does not match its evidence")
        return self


def episode_from_run(
    *, task: TaskInput, run: TaskRun, provenance: EpisodeProvenance
) -> ReviewedEpisode:
    """Project a public run into an idempotently addressed memory record."""

    if task.task_id != run.task_id:
        raise ValueError("task and run identifiers differ")
    if run.kappa_hash != provenance.kappa_hash:
        raise ValueError("run Kappa identity does not match episode provenance")
    student_models = (event.model_id for event in run.events if event.type == "action_proposed")
    reviewer_models = (
        event.review.model_id for event in run.events if event.type == "action_reviewed"
    )
    model_ids = tuple(dict.fromkeys((*student_models, *reviewer_models)))
    episode_id = _episode_id(task.task_id, run.seed, provenance)
    eligible = _is_eligible(
        task=task,
        actions=run.actions,
        reviews=run.reviews,
        termination_reason=run.termination_reason,
        provenance=provenance,
    )
    return ReviewedEpisode(
        episode_id=episode_id,
        seed=run.seed,
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


def _episode_id(task_id: str, seed: int, provenance: EpisodeProvenance) -> str:
    identity = json.dumps(
        {
            "task_id": task_id,
            "seed": seed,
            "configuration_hash": provenance.configuration_hash,
            "manifest_hash": provenance.manifest_hash,
            "lifecycle_pass_id": provenance.lifecycle_pass_id,
            "purpose": provenance.purpose,
            "accepted_version_id": provenance.accepted_version_id,
            "accepted_content_hash": provenance.accepted_content_hash,
            "kappa_hash": provenance.kappa_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "episode:" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _is_eligible(
    *,
    task: TaskInput,
    actions: tuple[StudentAction, ...],
    reviews: tuple[ActionReview, ...],
    termination_reason: str,
    provenance: EpisodeProvenance,
) -> bool:
    return (
        provenance.purpose == "collection"
        and task.task_id in provenance.manifest_task_ids
        and termination_reason == "completed"
        and len(reviews) == len(actions)
        and not any(
            review.decision == "replace"
            and (
                review.accepted_action.type == "final_answer"
                or review.accepted_action == review.proposed_action
            )
            for review in reviews
        )
    )

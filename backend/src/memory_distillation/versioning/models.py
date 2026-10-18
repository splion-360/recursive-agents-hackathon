"""Immutable Behavior Card and Candidate Version contracts."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BehaviorCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    trigger: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=800)
    evidence_episode_ids: tuple[str, ...]


class CardOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CREATE", "REINFORCE", "REFINE", "RETIRE"]
    card_id: str = Field(min_length=1)
    trigger: str | None = Field(default=None, min_length=1, max_length=300)
    instruction: str | None = Field(default=None, min_length=1, max_length=800)
    evidence_episode_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_kind_specific_payload(self) -> "CardOperation":
        if self.kind in {"CREATE", "REFINE"}:
            if (
                self.trigger is None
                or self.instruction is None
                or not self.trigger.strip()
                or not self.instruction.strip()
            ):
                raise ValueError(f"{self.kind} requires a non-empty trigger and instruction")
        elif self.trigger is not None or self.instruction is not None:
            raise ValueError(f"{self.kind} must not include a trigger or instruction")
        return self


@dataclass(frozen=True, slots=True)
class ConsolidationBatch:
    """One provider result before deterministic candidate screening."""

    operations: tuple[CardOperation, ...]
    model_id: str
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class EpisodeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_ids: tuple[str, ...]
    episode_hashes: tuple[str, ...]
    snapshot_hash: str


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(ge=0, le=3)
    minimum_score: float = Field(ge=0, le=1)
    token_budget: int = Field(ge=0)
    calibration_snapshot_hash: str


class VersionView(BaseModel):
    """One immutable, explicitly pinned operational-memory version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str
    cards: tuple[BehaviorCard, ...]
    retrieval_policy: RetrievalPolicy
    content_hash: str


class RetrievalCalibration(BaseModel):
    """Collection-only trial evidence and its deterministically selected policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_ids: tuple[str, ...] = Field(min_length=1)
    trials: tuple["RetrievalTrial", ...] = Field(min_length=1)
    top_k: int = Field(ge=0, le=3)
    minimum_score: float = Field(ge=0, le=1)
    token_budget: int = Field(ge=0)
    evidence_hash: str

    @model_validator(mode="after")
    def verify_derived_selection(self) -> "RetrievalCalibration":
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("retrieval calibration episode identifiers must be unique")
        if {trial.episode_id for trial in self.trials} != set(self.episode_ids):
            raise ValueError("retrieval trials must cover the complete Collection snapshot")
        selected = _select_retrieval_settings(self.trials)
        if selected != (self.top_k, self.minimum_score, self.token_budget):
            raise ValueError("retrieval settings do not match deterministic trial selection")
        payload = {
            "episode_ids": self.episode_ids,
            "trials": [trial.model_dump(mode="json") for trial in self.trials],
            "selected": selected,
        }
        if self.evidence_hash != _hash_json(payload):
            raise ValueError("retrieval calibration hash does not match its evidence")
        return self


class RetrievalTrial(BaseModel):
    """Observed retrieval result for one Collection query and candidate policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    episode_id: str
    top_k: int = Field(ge=0, le=3)
    minimum_score: float = Field(ge=0, le=1)
    token_budget: int = Field(ge=0)
    relevant_card_ids: tuple[str, ...] = Field(min_length=1)
    retrieved_card_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_valid_observation(self) -> "RetrievalTrial":
        if len(set(self.relevant_card_ids)) != len(self.relevant_card_ids):
            raise ValueError("relevant retrieval labels must be unique")
        if len(set(self.retrieved_card_ids)) != len(self.retrieved_card_ids):
            raise ValueError("retrieved card identifiers must be unique")
        if len(self.retrieved_card_ids) > self.top_k:
            raise ValueError("retrieval trial exceeds its top-k policy")
        return self


def calibrate_retrieval(
    *, episode_ids: tuple[str, ...], trials: tuple[RetrievalTrial, ...]
) -> RetrievalCalibration:
    """Select retrieval settings from recorded Collection outcomes and freeze evidence."""

    selected = _select_retrieval_settings(trials)
    payload = {
        "episode_ids": episode_ids,
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "selected": selected,
    }
    return RetrievalCalibration(
        episode_ids=episode_ids,
        trials=trials,
        top_k=selected[0],
        minimum_score=selected[1],
        token_budget=selected[2],
        evidence_hash=_hash_json(payload),
    )


def _select_retrieval_settings(
    trials: tuple[RetrievalTrial, ...],
) -> tuple[int, float, int]:
    if not trials:
        raise ValueError("retrieval calibration requires recorded trials")
    grouped: dict[tuple[int, float, int], list[float]] = {}
    query_sets: dict[tuple[int, float, int], set[str]] = {}
    trial_keys = [
        (trial.top_k, trial.minimum_score, trial.token_budget, trial.query_id) for trial in trials
    ]
    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("retrieval calibration contains duplicate policy-query trials")
    query_identity: dict[str, tuple[str, tuple[str, ...]]] = {}
    for trial in trials:
        identity = (trial.episode_id, tuple(sorted(trial.relevant_card_ids)))
        previous = query_identity.setdefault(trial.query_id, identity)
        if previous != identity:
            raise ValueError(
                "retrieval query provenance or relevance labels changed across policies"
            )
    for trial in trials:
        key = (trial.top_k, trial.minimum_score, trial.token_budget)
        relevant = set(trial.relevant_card_ids)
        retrieved = set(trial.retrieved_card_ids)
        true_positive = len(relevant & retrieved)
        precision = true_positive / len(retrieved) if retrieved else 0.0
        recall = true_positive / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        grouped.setdefault(key, []).append(f1)
        query_sets.setdefault(key, set()).add(trial.query_id)
    expected_queries = next(iter(query_sets.values()))
    if any(queries != expected_queries for queries in query_sets.values()):
        raise ValueError("every retrieval policy must evaluate the same Collection queries")
    return min(
        grouped,
        key=lambda key: (-sum(grouped[key]) / len(grouped[key]), key[2], key[0], -key[1]),
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class CandidateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["candidate-version.v1"] = "candidate-version.v1"
    version_id: str
    parent_version_id: str
    kappa_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: EpisodeSnapshot
    cards: tuple[BehaviorCard, ...]
    operations: tuple[CardOperation, ...]
    retrieval_policy: RetrievalPolicy
    content_hash: str

    @model_validator(mode="after")
    def verify_canonical_identity(self) -> "CandidateVersion":
        payload = {
            "parent_version_id": self.parent_version_id,
            "kappa_hash": self.kappa_hash,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "cards": [card.model_dump(mode="json") for card in self.cards],
            "operations": [item.model_dump(mode="json") for item in self.operations],
            "retrieval_policy": self.retrieval_policy.model_dump(mode="json"),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.content_hash != digest or self.version_id != f"candidate:{digest[:16]}":
            raise ValueError("candidate canonical identity does not match its content")
        return self

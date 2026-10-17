"""Immutable Behavior Card and Candidate Version contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BehaviorCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    trigger: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=800)
    evidence_episode_ids: tuple[str, ...]


class CardOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CREATE", "REINFORCE", "REFINE", "RETIRE"]
    card_id: str
    trigger: str | None = None
    instruction: str | None = None
    evidence_episode_ids: tuple[str, ...] = ()


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


class CandidateVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["candidate-version.v1"] = "candidate-version.v1"
    version_id: str
    parent_version_id: str
    kappa_hash: str
    snapshot: EpisodeSnapshot
    cards: tuple[BehaviorCard, ...]
    operations: tuple[CardOperation, ...]
    retrieval_policy: RetrievalPolicy
    content_hash: str

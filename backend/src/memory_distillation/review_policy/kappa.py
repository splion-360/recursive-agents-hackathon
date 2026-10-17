"""Deterministic H-star evidence to frozen Kappa Guidance adaptation."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_hash: str
    discovery_task_ids: tuple[str, ...]
    regression_task_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_isolation(self) -> "CalibrationSplit":
        if set(self.discovery_task_ids) & set(self.regression_task_ids):
            raise ValueError("discovery and regression tasks overlap")
        return self


class HStarProposal(BaseModel):
    """Executable review rules proposed from discovery tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    rules: tuple[str, ...] = Field(min_length=1, max_length=8)
    discovery_task_ids: tuple[str, ...]


class RuleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    discovery_helped: int = Field(ge=0)
    regression_helped: int = Field(ge=0)
    regression_harmed: int = Field(ge=0)


class KappaArtifact(BaseModel):
    """Immutable, canonically hashed review guidance used during Collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["kappa.v1"] = "kappa.v1"
    artifact_id: str
    manifest_hash: str
    revision: Literal[0, 1]
    guidance: tuple[str, ...]
    evidence: tuple[RuleEvidence, ...]
    content_hash: str


def freeze_kappa(
    *,
    proposal: HStarProposal,
    split: CalibrationSplit,
    evidence: tuple[RuleEvidence, ...],
    revision: Literal[0, 1] = 0,
    forbidden_answers: tuple[str, ...] = (),
) -> KappaArtifact:
    """Accept rules with non-negative regression evidence and freeze the result."""

    if set(proposal.discovery_task_ids) - set(split.discovery_task_ids):
        raise ValueError("H-star proposal used tasks outside discovery split")
    by_rule = {item.rule: item for item in evidence}
    if set(proposal.rules) != set(by_rule):
        raise ValueError("every proposed rule requires exactly one evidence record")
    guidance = tuple(
        rule
        for rule in proposal.rules
        if by_rule[rule].regression_helped >= by_rule[rule].regression_harmed
        and by_rule[rule].discovery_helped > 0
    )
    _validate_privacy(guidance, forbidden_answers)
    canonical = json.dumps(
        {
            "manifest_hash": split.manifest_hash,
            "revision": revision,
            "guidance": guidance,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return KappaArtifact(
        artifact_id=f"kappa:{digest[:16]}",
        manifest_hash=split.manifest_hash,
        revision=revision,
        guidance=guidance,
        evidence=evidence,
        content_hash=digest,
    )


def _validate_privacy(guidance: tuple[str, ...], forbidden_answers: tuple[str, ...]) -> None:
    joined = " ".join(guidance).casefold()
    leak_markers = ("hidden label", "ground truth", "correct answer", "task_id")
    if any(marker in joined for marker in leak_markers):
        raise ValueError("Kappa guidance contains benchmark leakage")
    for answer in forbidden_answers:
        normalized = answer.strip().casefold()
        if normalized and normalized in joined:
            raise ValueError("Kappa guidance contains a task-specific answer")

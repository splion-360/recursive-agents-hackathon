"""Deterministic H-star evidence to frozen Kappa Guidance adaptation."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal["inspect_image", "crop", "run_python", "final_answer"]


class CalibrationSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_hash: str
    discovery_task_ids: tuple[str, ...]
    regression_task_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_isolation(self) -> "CalibrationSplit":
        if len(self.discovery_task_ids) != 8 or len(self.regression_task_ids) != 8:
            raise ValueError(
                "calibration requires exactly eight discovery and eight regression tasks"
            )
        if len(set(self.discovery_task_ids)) != 8 or len(set(self.regression_task_ids)) != 8:
            raise ValueError("calibration task identifiers must be unique")
        if set(self.discovery_task_ids) & set(self.regression_task_ids):
            raise ValueError("discovery and regression tasks overlap")
        return self


class KappaRule(BaseModel):
    """One executable rule constrained to Minimal Harness action types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    action_types: tuple[ActionType, ...] = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=500)


class HStarProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    parent_revision: Literal[0] = 0
    revision: Literal[1] = 1
    rules: tuple[KappaRule, ...] = Field(min_length=1, max_length=8)
    discovery_task_ids: tuple[str, ...]


class RuleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    discovery_task_ids: tuple[str, ...]
    regression_task_ids: tuple[str, ...]
    discovery_helped: int = Field(ge=0)
    regression_helped: int = Field(ge=0)
    regression_harmed: int = Field(ge=0)
    action_compatible: bool


class CalibrationExecution(BaseModel):
    """Recorded baseline/candidate outcome used to evaluate one proposed rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    proposal_hash: str
    phase: Literal["discovery", "regression"]
    task_id: str
    rule_id: str
    action_type: ActionType
    baseline_correct: bool
    candidate_correct: bool


class CalibrationPrivacyContext(BaseModel):
    """Verifier-built strings that must never appear in Kappa Guidance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    forbidden_phrases: tuple[str, ...] = Field(min_length=1)


class KappaArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["kappa.v1"] = "kappa.v1"
    artifact_id: str
    manifest_hash: str
    revision: Literal[0, 1]
    rules: tuple[KappaRule, ...]
    guidance: tuple[str, ...]
    evidence: tuple[RuleEvidence, ...]
    content_hash: str

    @model_validator(mode="after")
    def verify_canonical_identity(self) -> "KappaArtifact":
        if self.guidance != tuple(rule.instruction for rule in self.rules):
            raise ValueError("Kappa guidance does not match its frozen executable rules")
        payload = {
            "manifest_hash": self.manifest_hash,
            "revision": self.revision,
            "rules": [item.model_dump(mode="json") for item in self.rules],
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
        }
        digest = _hash_json(payload)
        if self.content_hash != digest or self.artifact_id != f"kappa:{digest[:16]}":
            raise ValueError("Kappa canonical identity does not match its content")
        return self


class KappaCalibrationSession:
    """One in-process calibration state machine permitting one H-star proposal."""

    def __init__(self, split: CalibrationSplit) -> None:
        self._split = split
        self._proposal_hash: str | None = None
        self._artifact: KappaArtifact | None = None

    def freeze(
        self,
        *,
        proposal: HStarProposal,
        executions: tuple[CalibrationExecution, ...],
        privacy: CalibrationPrivacyContext,
    ) -> KappaArtifact:
        proposal_hash = hash_hstar_proposal(proposal)
        if self._proposal_hash is not None and self._proposal_hash != proposal_hash:
            raise ValueError("calibration session already started a different H-star proposal")
        self._proposal_hash = proposal_hash
        artifact = _freeze_kappa(
            proposal=proposal,
            split=self._split,
            executions=executions,
            privacy=privacy,
        )
        if self._artifact is not None and self._artifact != artifact:
            raise ValueError("calibration session is already frozen")
        self._artifact = artifact
        return artifact


def freeze_kappa(
    *,
    session: KappaCalibrationSession,
    proposal: HStarProposal,
    executions: tuple[CalibrationExecution, ...],
    privacy: CalibrationPrivacyContext,
) -> KappaArtifact:
    """Freeze the sole proposal allowed by a calibration session."""

    return session.freeze(proposal=proposal, executions=executions, privacy=privacy)


def _freeze_kappa(
    *,
    proposal: HStarProposal,
    split: CalibrationSplit,
    executions: tuple[CalibrationExecution, ...],
    privacy: CalibrationPrivacyContext,
) -> KappaArtifact:
    """Regression-screen, privacy-check, canonically hash, and freeze Kappa."""

    if tuple(proposal.discovery_task_ids) != tuple(split.discovery_task_ids):
        raise ValueError("H-star proposal must use the complete ordered discovery split")
    proposal_hash = hash_hstar_proposal(proposal)
    if any(item.proposal_hash != proposal_hash for item in executions):
        raise ValueError("calibration executions do not match the exact H-star proposal")
    rule_ids = [rule.rule_id for rule in proposal.rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("Kappa rule identifiers must be unique")
    evidence = _derive_evidence(proposal=proposal, split=split, executions=executions)
    by_rule = {item.rule_id: item for item in evidence}
    accepted = tuple(
        rule
        for rule in proposal.rules
        if by_rule[rule.rule_id].action_compatible
        and by_rule[rule.rule_id].regression_helped >= by_rule[rule.rule_id].regression_harmed
        and by_rule[rule.rule_id].discovery_helped > 0
    )
    if not accepted:
        raise ValueError("no Kappa rule passed deterministic regression screening")
    _validate_privacy(accepted, privacy)
    canonical_payload = {
        "manifest_hash": split.manifest_hash,
        "revision": proposal.revision,
        "rules": [item.model_dump(mode="json") for item in accepted],
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    digest = _hash_json(canonical_payload)
    return KappaArtifact(
        artifact_id=f"kappa:{digest[:16]}",
        manifest_hash=split.manifest_hash,
        revision=proposal.revision,
        rules=accepted,
        guidance=tuple(rule.instruction for rule in accepted),
        evidence=evidence,
        content_hash=digest,
    )


def hash_hstar_proposal(proposal: HStarProposal) -> str:
    """Bind calibration executions to one exact, one-revision proposal."""

    return _hash_json(proposal.model_dump(mode="json"))


def _derive_evidence(
    *,
    proposal: HStarProposal,
    split: CalibrationSplit,
    executions: tuple[CalibrationExecution, ...],
) -> tuple[RuleEvidence, ...]:
    if len({item.execution_id for item in executions}) != len(executions):
        raise ValueError("calibration execution identifiers must be unique")
    rules = {rule.rule_id: rule for rule in proposal.rules}
    if any(item.rule_id not in rules for item in executions):
        raise ValueError("calibration execution references an unknown rule")
    evidence: list[RuleEvidence] = []
    for rule in proposal.rules:
        rule_runs = tuple(item for item in executions if item.rule_id == rule.rule_id)
        discovery = tuple(item for item in rule_runs if item.phase == "discovery")
        regression = tuple(item for item in rule_runs if item.phase == "regression")
        if {item.task_id for item in discovery} != set(split.discovery_task_ids):
            raise ValueError("rule executions do not cover the discovery split")
        if {item.task_id for item in regression} != set(split.regression_task_ids):
            raise ValueError("rule executions do not cover the regression split")
        if len(discovery) != 8 or len(regression) != 8:
            raise ValueError("every calibration task must execute exactly once per rule")
        compatible = all(item.action_type in rule.action_types for item in rule_runs)
        evidence.append(
            RuleEvidence(
                rule_id=rule.rule_id,
                discovery_task_ids=split.discovery_task_ids,
                regression_task_ids=split.regression_task_ids,
                discovery_helped=sum(
                    not item.baseline_correct and item.candidate_correct for item in discovery
                ),
                regression_helped=sum(
                    not item.baseline_correct and item.candidate_correct for item in regression
                ),
                regression_harmed=sum(
                    item.baseline_correct and not item.candidate_correct for item in regression
                ),
                action_compatible=compatible,
            )
        )
    return tuple(evidence)


def _validate_privacy(rules: tuple[KappaRule, ...], privacy: CalibrationPrivacyContext) -> None:
    content = " ".join(rule.instruction for rule in rules).casefold()
    leak_markers = (
        "hidden label",
        "ground truth",
        "correct answer",
        "task_id",
        "open the browser",
        "search the web",
        "network request",
    )
    if any(marker in content for marker in leak_markers):
        raise ValueError("Kappa guidance contains leakage or unsupported behavior")
    for phrase in privacy.forbidden_phrases:
        normalized = phrase.strip().casefold()
        if normalized and normalized in content:
            raise ValueError("Kappa guidance contains calibration-specific content")


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

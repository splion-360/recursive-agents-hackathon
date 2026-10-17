"""Truthful read-only dashboard projections for local experiment runs."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from memory_distillation.settings import Settings
from memory_distillation.task_execution.models import RunEvent

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_ARTIFACT_DIR = _REPOSITORY_ROOT / ".artifacts" / "calibration" / "current"
CALIBRATION_STATUS_PATH = CALIBRATION_ARTIFACT_DIR / "status.json"
CALIBRATION_EVENTS_PATH = CALIBRATION_ARTIFACT_DIR / "events.jsonl"
_MANIFEST_DIR = _REPOSITORY_ROOT / "backend" / "experiment_manifests"
_RUN_EVENT_ADAPTER = TypeAdapter(RunEvent)


class CalibrationStatus(BaseModel):
    """Public progress written by the calibration command and displayed verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["calibration-status.v1"] = "calibration-status.v1"
    run_id: str | None = None
    state: Literal["idle", "running", "complete", "failed"] = "idle"
    stage: Literal[
        "waiting",
        "preflight",
        "discovery",
        "hstar_proposal",
        "regression",
        "kappa_freeze",
        "smoke_test",
        "complete",
        "failed",
    ] = "waiting"
    discovery_completed: int = Field(default=0, ge=0, le=8)
    regression_completed: int = Field(default=0, ge=0, le=8)
    smoke_completed: int = Field(default=0, ge=0, le=5)
    smoke_total: int = Field(default=0, ge=0, le=5)
    current_task_id: str | None = None
    proposal_id: str | None = None
    proposal_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    kappa_artifact_id: str | None = None
    kappa_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    accepted_rule_ids: tuple[str, ...] = ()
    rejected_rule_ids: tuple[str, ...] = ()
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    message: str = "Waiting for the calibration command."
    updated_at: datetime | None = None


class ManifestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    task_count: int
    content_sha256: str


class IntegrationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    state: Literal["configured", "missing", "limited"]
    detail: str


class ModelRoster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    student: str | None
    reviewer: str | None
    consolidator: str | None
    hstar: str | None


class DashboardSnapshot(BaseModel):
    """Single truthful snapshot used to hydrate the experiment dashboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["LIVE"] = "LIVE"
    generated_at: datetime
    accepted_version: Literal["accepted-v000"] = "accepted-v000"
    accepted_card_count: Literal[0] = 0
    measured_results_available: Literal[False] = False
    calibration: CalibrationStatus
    manifests: tuple[ManifestSummary, ...]
    integrations: tuple[IntegrationStatus, ...]
    models: ModelRoster
    event_count: int
    recent_events: tuple[RunEvent, ...]


def load_dashboard_snapshot(settings: Settings) -> DashboardSnapshot:
    """Load only public local artifacts; malformed artifacts fail closed."""

    events = read_run_events(CALIBRATION_EVENTS_PATH)
    return DashboardSnapshot(
        generated_at=datetime.now(UTC),
        calibration=_load_calibration_status(),
        manifests=tuple(
            _load_manifest(name)
            for name in (
                "calibration_discovery",
                "calibration_regression",
                "collection",
                "promotion",
                "final",
            )
        ),
        integrations=_integration_statuses(settings),
        models=ModelRoster(
            student=settings.crusoe_student_model,
            reviewer=settings.crusoe_reviewer_model,
            consolidator=settings.crusoe_consolidator_model,
            hstar=settings.crusoe_hstar_model,
        ),
        event_count=len(events),
        recent_events=events[-40:],
    )


def read_run_events(path: Path) -> tuple[RunEvent, ...]:
    """Validate complete JSONL records as the canonical public RunEvent union."""

    if not path.exists():
        return ()
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("live event trace contains a truncated JSONL record")
    return tuple(
        parse_run_event(json.loads(line))
        for line in content.splitlines()
        if line.strip()
    )


def parse_run_event(payload: object) -> RunEvent:
    """Validate one live JSON value before it crosses the dashboard boundary."""

    return _RUN_EVENT_ADAPTER.validate_python(payload)


def write_calibration_status(status: CalibrationStatus) -> None:
    """Atomically publish public progress for a shell-driven calibration run."""

    CALIBRATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CALIBRATION_STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(status.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(CALIBRATION_STATUS_PATH)


def _load_calibration_status() -> CalibrationStatus:
    if not CALIBRATION_STATUS_PATH.exists():
        return CalibrationStatus()
    return CalibrationStatus.model_validate_json(CALIBRATION_STATUS_PATH.read_text())


def _load_manifest(name: str) -> ManifestSummary:
    payload = json.loads((_MANIFEST_DIR / f"{name}.json").read_text())
    return ManifestSummary(
        name=name,
        task_count=len(payload["entries"]),
        content_sha256=payload["content_sha256"],
    )


def _integration_statuses(settings: Settings) -> tuple[IntegrationStatus, ...]:
    return (
        IntegrationStatus(
            name="Crusoe Intelligence Foundry",
            state="configured" if settings.crusoe_api_key is not None else "missing",
            detail="Credentials configured; dashboard does not imply a live provider check.",
        ),
        IntegrationStatus(
            name="NemoClaw / OpenShell",
            state="configured" if settings.openshell_sandbox_name is not None else "missing",
            detail=(
                f"Sandbox {settings.openshell_sandbox_name} via {settings.openshell_gateway}."
                if settings.openshell_sandbox_name is not None
                else "No sandbox name configured."
            ),
        ),
        IntegrationStatus(
            name="MemMachine",
            state="limited" if settings.memmachine_api_key is not None else "missing",
            detail=(
                "Managed memory configured; atomic accepted-pointer update is unavailable."
                if settings.memmachine_api_key is not None
                else "No managed-memory credential configured."
            ),
        ),
    )

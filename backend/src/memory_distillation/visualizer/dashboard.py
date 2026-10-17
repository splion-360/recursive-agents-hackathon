"""Truthful read-only dashboard projections for local experiment runs."""

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from memory_distillation.benchmark import DevelopmentBenchmark, TaskAsset, TaskInput
from memory_distillation.settings import Settings
from memory_distillation.task_execution.models import RunEvent

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_ARTIFACT_DIR = _REPOSITORY_ROOT / ".artifacts" / "calibration" / "current"
CALIBRATION_STATUS_PATH = CALIBRATION_ARTIFACT_DIR / "status.json"
CALIBRATION_EVENTS_PATH = CALIBRATION_ARTIFACT_DIR / "events.jsonl"
_MANIFEST_DIR = _REPOSITORY_ROOT / "backend" / "experiment_manifests"
_DATASET_ROOT = _REPOSITORY_ROOT / ".local" / "chartqa-source" / "ChartQA Dataset"
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
    discovery_completed: int = Field(default=0, ge=0, le=12)
    regression_completed: int = Field(default=0, ge=0, le=12)
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


class CalibrationTaskSummary(BaseModel):
    """Answer-free calibration membership used to label dashboard traces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    phase: Literal["discovery", "regression"]


class SplitTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery: int
    regression: int


class HarnessDefinition(BaseModel):
    """Sanitized executable contract for one calibration participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["student", "reviewer", "hstar", "kappa"]
    model_id: str | None
    purpose: str
    sees: tuple[str, ...]
    actions: tuple[str, ...]
    guidance: tuple[str, ...] = ()
    revisions: tuple[str, ...] = ()


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
    split_totals: SplitTotals
    harnesses: tuple[HarnessDefinition, ...]
    calibration_tasks: tuple[CalibrationTaskSummary, ...]
    run_events: tuple[RunEvent, ...]


class CalibrationTaskDetail(BaseModel):
    """Answer-free task context displayed beside a calibration chart."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    question: str


def load_dashboard_snapshot(settings: Settings) -> DashboardSnapshot:
    """Load only public local artifacts; malformed artifacts fail closed."""

    events = read_run_events(CALIBRATION_EVENTS_PATH)
    _, _, task_phases = _calibration_source()
    manifests = tuple(
        _load_manifest(name)
        for name in (
            "calibration_discovery",
            "calibration_regression",
            "collection",
            "promotion",
            "final",
        )
    )
    totals = {manifest.name: manifest.task_count for manifest in manifests}
    return DashboardSnapshot(
        generated_at=datetime.now(UTC),
        calibration=_load_calibration_status(),
        manifests=manifests,
        integrations=_integration_statuses(settings),
        models=ModelRoster(
            student=settings.crusoe_student_model,
            reviewer=settings.crusoe_reviewer_model,
            consolidator=settings.crusoe_consolidator_model,
            hstar=settings.crusoe_hstar_model,
        ),
        event_count=len(events),
        split_totals=SplitTotals(
            discovery=totals["calibration_discovery"],
            regression=totals["calibration_regression"],
        ),
        harnesses=_harness_definitions(settings),
        calibration_tasks=tuple(
            CalibrationTaskSummary(task_id=task_id, phase=phase)
            for task_id, phase in task_phases.items()
        ),
        run_events=events,
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


def load_calibration_task_asset(task_id: str) -> TaskAsset:
    """Load an answer-free image only when it belongs to a calibration manifest."""

    benchmark, _, _ = _calibration_source()
    task = load_calibration_task(task_id)
    return benchmark.load_asset(task)


def load_calibration_task(task_id: str) -> TaskInput:
    """Load answer-free context only for a task in the calibration manifests."""

    _, tasks, _ = _calibration_source()
    try:
        return tasks[task_id]
    except KeyError as error:
        raise KeyError(f"unknown calibration task: {task_id}") from error


@lru_cache(maxsize=1)
def _calibration_source() -> tuple[
    DevelopmentBenchmark,
    dict[str, TaskInput],
    dict[str, Literal["discovery", "regression"]],
]:
    benchmark = DevelopmentBenchmark(dataset_root=_DATASET_ROOT, manifest_dir=_MANIFEST_DIR)
    calibration = benchmark.load_calibration()
    tasks = {task.task_id: task for task in (*calibration.discovery, *calibration.regression)}
    phases: dict[str, Literal["discovery", "regression"]] = {
        task.task_id: "discovery" for task in calibration.discovery
    }
    phases.update({task.task_id: "regression" for task in calibration.regression})
    return benchmark, tasks, phases


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


def _harness_definitions(settings: Settings) -> tuple[HarnessDefinition, ...]:
    guidance, revisions = _calibration_rule_summary()
    return (
        HarnessDefinition(
            role="student",
            model_id=settings.crusoe_student_model,
            purpose="Solve the ChartQA task through a bounded typed action loop.",
            sees=("question", "chart image", "public tool observations"),
            actions=("inspect_image", "crop", "run_python", "final_answer"),
        ),
        HarnessDefinition(
            role="reviewer",
            model_id=settings.crusoe_reviewer_model,
            purpose="Privately PASS a sound action or REPLACE it with the smallest valid action.",
            sees=("Student proposal", "chart image", "observations", "frozen Kappa"),
            actions=("pass", "replace with inspect_image/crop/run_python"),
            guidance=("Never replace final_answer directly with another final_answer.",),
        ),
        HarnessDefinition(
            role="hstar",
            model_id=settings.crusoe_hstar_model,
            purpose="Evolve answer-free review guidance over three four-task Discovery rounds.",
            sees=("Discovery outcomes", "action types", "prior H-star revision"),
            actions=("propose full next harness revision",),
            revisions=revisions,
        ),
        HarnessDefinition(
            role="kappa",
            model_id=settings.crusoe_reviewer_model,
            purpose="Frozen regression-screened guidance used only by the private Reviewer.",
            sees=("accepted H-star components", "12-task Regression evidence"),
            actions=("constrain Reviewer PASS/REPLACE decisions",),
            guidance=guidance,
        ),
    )


def _calibration_rule_summary() -> tuple[tuple[str, ...], tuple[str, ...]]:
    report_path = CALIBRATION_ARTIFACT_DIR / "report.json"
    if not report_path.exists():
        return (), ()
    payload = json.loads(report_path.read_text())
    guidance = tuple(str(item) for item in payload.get("kappa", {}).get("guidance", ()))
    revision_items = payload.get("hstar_revisions", ())
    revisions = tuple(
        f"R{item['round_index']}: "
        + " | ".join(rule["instruction"] for rule in item["proposal"].get("rules", ()))
        for item in revision_items
    )
    if not revisions and payload.get("proposal"):
        rules = payload["proposal"].get("rules", ())
        revisions = tuple(f"Final: {rule['instruction']}" for rule in rules)
    return guidance, revisions


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

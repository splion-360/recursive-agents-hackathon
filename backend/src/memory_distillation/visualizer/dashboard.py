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
from memory_distillation.versioning.promotion import PairedOutcome

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_ARTIFACT_DIR = _REPOSITORY_ROOT / ".artifacts" / "calibration" / "current"
CALIBRATION_STATUS_PATH = CALIBRATION_ARTIFACT_DIR / "status.json"
CALIBRATION_EVENTS_PATH = CALIBRATION_ARTIFACT_DIR / "events.jsonl"
LIFECYCLE_ARTIFACT_DIR = _REPOSITORY_ROOT / ".artifacts" / "lifecycle" / "current"
LIFECYCLE_EVENTS_PATH = LIFECYCLE_ARTIFACT_DIR / "events.jsonl"
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


class BehaviorCardSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    trigger: str
    instruction: str
    evidence_episode_ids: tuple[str, ...]


class PromotionPairSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    accepted_correct: bool
    candidate_correct: bool
    accepted_latency_ms: int
    candidate_latency_ms: int
    accepted_input_tokens: int
    candidate_input_tokens: int


def _promotion_pair_summary(value: object) -> PromotionPairSummary:
    """Validate full promotion evidence, then expose only dashboard-safe fields."""

    outcome = PairedOutcome.model_validate(value)
    return PromotionPairSummary(
        task_id=outcome.task_id,
        accepted_correct=outcome.accepted_correct,
        candidate_correct=outcome.candidate_correct,
        accepted_latency_ms=outcome.accepted_latency_ms,
        candidate_latency_ms=outcome.candidate_latency_ms,
        accepted_input_tokens=outcome.accepted_input_tokens,
        candidate_input_tokens=outcome.candidate_input_tokens,
    )


class LifecycleDashboard(BaseModel):
    """Sanitized live projection of the Collection-to-Promotion pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["idle", "running", "complete"]
    stage: Literal["waiting", "collection", "consolidation", "promotion", "complete"]
    run_id: str | None = None
    collection_completed: int = Field(ge=0, le=40)
    collection_total: Literal[40] = 40
    promotion_runs_completed: int = Field(ge=0, le=60)
    promotion_runs_total: Literal[60] = 60
    reviewed_episode_count: int = Field(ge=0)
    eligible_episode_count: int = Field(ge=0)
    parent_version_id: str = "accepted-v000"
    candidate_version_id: str | None = None
    accepted_version_id: str = "accepted-v000"
    cards: tuple[BehaviorCardSummary, ...] = ()
    decision: Literal["PROMOTE", "REJECT"] | None = None
    failed_gates: tuple[str, ...] = ()
    capability_delta: float | None = None
    negative_transfer_rate: float | None = None
    latency_regression: float | None = None
    token_regression: float | None = None
    promotion_pairs: tuple[PromotionPairSummary, ...] = ()
    run_events: tuple[RunEvent, ...] = ()


class DashboardSnapshot(BaseModel):
    """Single truthful snapshot used to hydrate the experiment dashboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["LIVE"] = "LIVE"
    generated_at: datetime
    accepted_version: str = "accepted-v000"
    accepted_card_count: int = Field(default=0, ge=0)
    measured_results_available: bool = False
    calibration: CalibrationStatus
    manifests: tuple[ManifestSummary, ...]
    integrations: tuple[IntegrationStatus, ...]
    models: ModelRoster
    event_count: int
    split_totals: SplitTotals
    harnesses: tuple[HarnessDefinition, ...]
    calibration_tasks: tuple[CalibrationTaskSummary, ...]
    run_events: tuple[RunEvent, ...]
    lifecycle: LifecycleDashboard


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
    lifecycle = _load_lifecycle_dashboard()
    return DashboardSnapshot(
        generated_at=datetime.now(UTC),
        accepted_version=lifecycle.accepted_version_id,
        accepted_card_count=(
            len(lifecycle.cards) if lifecycle.decision == "PROMOTE" else 0
        ),
        measured_results_available=lifecycle.state == "complete",
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
        lifecycle=lifecycle,
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


def load_development_task(task_id: str) -> TaskInput:
    """Load answer-free context for any non-final development task."""

    _, tasks = _development_source()
    try:
        return tasks[task_id]
    except KeyError as error:
        raise KeyError(f"unknown development task: {task_id}") from error


def load_development_task_asset(task_id: str) -> TaskAsset:
    """Load a chart only when it belongs to an approved development manifest."""

    benchmark, _ = _development_source()
    return benchmark.load_asset(load_development_task(task_id))


@lru_cache(maxsize=1)
def _development_source() -> tuple[DevelopmentBenchmark, dict[str, TaskInput]]:
    benchmark = DevelopmentBenchmark(dataset_root=_DATASET_ROOT, manifest_dir=_MANIFEST_DIR)
    calibration = benchmark.load_calibration()
    tasks = (
        *calibration.discovery,
        *calibration.regression,
        *benchmark.load_collection(),
        *benchmark.load_promotion(),
    )
    return benchmark, {task.task_id: task for task in tasks}


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


def _load_lifecycle_dashboard() -> LifecycleDashboard:
    events = read_run_events(LIFECYCLE_EVENTS_PATH)
    completed_runs = sum(event.type == "run_completed" for event in events)
    collection_completed = min(completed_runs, 40)
    promotion_runs_completed = max(0, completed_runs - 40)
    report_path = LIFECYCLE_ARTIFACT_DIR / "report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = payload["candidate"]
        decision = payload["decision"]
        metrics = decision["metrics"]
        return LifecycleDashboard(
            state="complete",
            stage="complete",
            run_id=str(payload["run_id"]),
            collection_completed=40,
            promotion_runs_completed=60,
            reviewed_episode_count=len(payload["collection_episode_ids"]),
            eligible_episode_count=len(payload["eligible_episode_ids"]),
            parent_version_id=str(candidate["parent_version_id"]),
            candidate_version_id=str(candidate["version_id"]),
            accepted_version_id=str(decision["accepted_version_after"]),
            cards=tuple(BehaviorCardSummary.model_validate(card) for card in candidate["cards"]),
            decision=decision["decision"],
            failed_gates=tuple(decision["failed_gates"]),
            capability_delta=float(metrics["capability_delta"]),
            negative_transfer_rate=float(metrics["negative_transfer_rate"]),
            latency_regression=float(metrics["latency_regression"]),
            token_regression=float(metrics["token_regression"]),
            promotion_pairs=tuple(
                _promotion_pair_summary(item) for item in payload["promotion_outcomes"]
            ),
            run_events=events,
        )
    if not events:
        return LifecycleDashboard(
            state="idle",
            stage="waiting",
            collection_completed=0,
            promotion_runs_completed=0,
            reviewed_episode_count=0,
            eligible_episode_count=0,
        )
    stage: Literal["collection", "consolidation", "promotion"]
    if completed_runs < 40:
        stage = "collection"
    elif completed_runs == 40:
        stage = "consolidation"
    else:
        stage = "promotion"
    return LifecycleDashboard(
        state="running",
        stage=stage,
        collection_completed=collection_completed,
        promotion_runs_completed=promotion_runs_completed,
        reviewed_episode_count=collection_completed,
        eligible_episode_count=0,
        run_events=events,
    )


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

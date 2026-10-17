"""HTTP composition root for the experiment runner and read-only dashboard."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from memory_distillation.operational_memory import MemMachineOperationalMemory
from memory_distillation.sandbox import OpenShellSandbox
from memory_distillation.settings import Settings, get_settings
from memory_distillation.visualizer import (
    CALIBRATION_EVENTS_PATH,
    DashboardSnapshot,
    MetricEvidence,
    PromotionEvidence,
    ReplayEvidence,
    RunEvidence,
    VersionEvidence,
    load_dashboard_snapshot,
    parse_run_event,
    synthetic_evidence,
)
from memory_distillation.visualizer.dashboard import (
    CalibrationTaskDetail,
    load_calibration_task,
    load_calibration_task_asset,
)


class HealthResponse(BaseModel):
    """Health response returned by the scaffold."""

    status: Literal["ok"]
    application: str


def create_production_sandbox(settings: Settings) -> OpenShellSandbox | None:
    """Compose the live sandbox adapter only when a sandbox is configured."""

    if settings.openshell_sandbox_name is None:
        return None
    return OpenShellSandbox(
        sandbox_name=settings.openshell_sandbox_name,
        gateway=settings.openshell_gateway,
    )


def create_production_memory(settings: Settings) -> MemMachineOperationalMemory | None:
    """Compose managed Operational Memory only when its credential is configured."""

    if settings.memmachine_api_key is None:
        return None
    return MemMachineOperationalMemory(
        base_url=str(settings.memmachine_base_url),
        api_key=settings.memmachine_api_key,
        project_id=settings.memmachine_project_id,
    )


app = FastAPI(title="Memory-Distilled Agent Team")
app.state.settings = get_settings()
app.state.sandbox = create_production_sandbox(app.state.settings)
app.state.operational_memory = create_production_memory(app.state.settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the backend process is ready to receive requests."""

    return HealthResponse(status="ok", application="memory-distillation")


@app.get("/api/dashboard", response_model=DashboardSnapshot)
def dashboard() -> DashboardSnapshot:
    """Return current local experiment state without synthetic measurements."""

    return load_dashboard_snapshot(app.state.settings)


@app.get("/api/dashboard/tasks/{task_id}/image", response_class=Response)
def dashboard_task_image(task_id: str) -> Response:
    """Serve a calibration chart without exposing dataset paths or answer labels."""

    try:
        asset = load_calibration_task_asset(task_id)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration chart not found") from error
    return Response(
        content=asset.content,
        media_type=asset.media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/dashboard/tasks/{task_id}", response_model=CalibrationTaskDetail)
def dashboard_task(task_id: str) -> CalibrationTaskDetail:
    """Return answer-free task context for a calibration-table row."""

    try:
        task = load_calibration_task(task_id)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration task not found") from error
    return CalibrationTaskDetail(task_id=task.task_id, question=task.question)


@app.get("/api/dashboard/events")
async def dashboard_events() -> StreamingResponse:
    """Stream validated public RunEvents appended by the shell-driven runner."""

    async def stream() -> AsyncIterator[str]:
        offset = 0
        while True:
            if CALIBRATION_EVENTS_PATH.exists():
                with CALIBRATION_EVENTS_PATH.open(encoding="utf-8") as source:
                    source.seek(offset)
                    while line := source.readline():
                        if not line.endswith("\n"):
                            break
                        offset = source.tell()
                        if not line.strip():
                            continue
                        event = parse_run_event(json.loads(line))
                        payload = event.model_dump(mode="json")
                        encoded = json.dumps(payload, separators=(",", ":"))
                        yield f"event: run_event\ndata: {encoded}\n\n"
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/evidence/runs", response_model=list[RunEvidence])
def evidence_runs() -> tuple[RunEvidence, ...]:
    return synthetic_evidence()[0]


@app.get("/api/evidence/versions", response_model=list[VersionEvidence])
def evidence_versions() -> tuple[VersionEvidence, ...]:
    return synthetic_evidence()[1]


@app.get("/api/evidence/promotion", response_model=PromotionEvidence)
def evidence_promotion() -> PromotionEvidence:
    return synthetic_evidence()[2]


@app.get("/api/evidence/metrics", response_model=list[MetricEvidence])
def evidence_metrics() -> tuple[MetricEvidence, ...]:
    return synthetic_evidence()[3]


@app.get("/api/evidence/replay", response_model=ReplayEvidence)
def evidence_replay() -> ReplayEvidence:
    return synthetic_evidence()[4]

"""Minimal HTTP entry point for the project scaffold."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from memory_distillation.sandbox import OpenShellSandbox
from memory_distillation.settings import Settings, get_settings
from memory_distillation.visualizer import (
    MetricEvidence,
    PromotionEvidence,
    ReplayEvidence,
    RunEvidence,
    VersionEvidence,
    synthetic_evidence,
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


app = FastAPI(title="Memory-Distilled Agent Team")
app.state.settings = get_settings()
app.state.sandbox = create_production_sandbox(app.state.settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the backend process is ready to receive requests."""

    return HealthResponse(status="ok", application="memory-distillation")


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

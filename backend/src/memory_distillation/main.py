"""Minimal HTTP entry point for the project scaffold."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health response returned by the scaffold."""

    status: Literal["ok"]
    application: str


app = FastAPI(title="Memory-Distilled Agent Team")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the backend process is ready to receive requests."""

    return HealthResponse(status="ok", application="memory-distillation")

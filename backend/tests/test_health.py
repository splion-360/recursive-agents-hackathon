"""Smoke tests for the minimal HTTP scaffold."""

from fastapi.testclient import TestClient

from memory_distillation.main import app


def test_health_reports_ready() -> None:
    """The scaffold exposes an explicit readiness response."""

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "memory-distillation",
    }

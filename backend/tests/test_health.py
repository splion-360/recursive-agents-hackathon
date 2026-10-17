"""Smoke tests for the minimal HTTP scaffold."""

import subprocess
import sys

from fastapi.testclient import TestClient

from memory_distillation.main import app


def test_composition_root_imports_in_a_fresh_interpreter() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from memory_distillation.main import app; assert app"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_health_reports_ready() -> None:
    """The scaffold exposes an explicit readiness response."""

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "memory-distillation",
    }


def test_read_only_evidence_endpoints_are_synthetic_and_sanitized() -> None:
    client = TestClient(app)
    paths = ("runs", "versions", "promotion", "metrics", "replay")
    payloads = [client.get(f"/api/evidence/{path}") for path in paths]

    assert all(response.status_code == 200 for response in payloads)
    replay = payloads[-1].json()
    assert replay["mode"] == "REPLAY"
    assert replay["synthetic"] is True
    combined = "".join(response.text for response in payloads).casefold()
    assert "private_reason" not in combined
    assert "api_key" not in combined
    assert {item["role"] for item in payloads[1].json()} == {"ACCEPTED", "PREVIOUS"}
    for method in (client.post, client.put, client.delete):
        assert method("/api/evidence/runs").status_code == 405

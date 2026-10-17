"""Dashboard projection and shell-runner artifact contract tests."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memory_distillation import main as main_module
from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.main import app
from memory_distillation.task_execution.models import RunStartedEvent
from memory_distillation.visualizer import dashboard as dashboard_module
from memory_distillation.visualizer.dashboard import (
    CalibrationStatus,
    read_run_events,
    write_calibration_status,
)


def test_dashboard_starts_truthfully_without_measured_results() -> None:
    payload = TestClient(app).get("/api/dashboard").json()

    assert payload["mode"] == "LIVE"
    assert payload["accepted_version"] == "accepted-v000"
    assert payload["accepted_card_count"] == 0
    assert payload["measured_results_available"] is False
    assert {item["name"]: item["task_count"] for item in payload["manifests"]} == {
        "calibration_discovery": 8,
        "calibration_regression": 8,
        "collection": 40,
        "promotion": 30,
        "final": 60,
    }
    serialized = json.dumps(payload).casefold()
    assert "api_key" not in serialized
    assert "private_reason" not in serialized


def test_shell_runner_status_is_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(dashboard_module, "CALIBRATION_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(dashboard_module, "CALIBRATION_STATUS_PATH", status_path)
    status = CalibrationStatus(
        run_id="calibration-001",
        state="running",
        stage="discovery",
        discovery_completed=2,
        message="Running discovery task 3 of 8.",
    )

    write_calibration_status(status)

    assert CalibrationStatus.model_validate_json(status_path.read_text()) == status
    assert not status_path.with_suffix(".json.tmp").exists()


def test_live_trace_accepts_only_complete_typed_run_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    event = RunStartedEvent(sequence=0, task_id="chartqa:calibration:001", seed=17)
    path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    assert read_run_events(path) == (event,)

    path.write_text(event.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        read_run_events(path)


def test_dashboard_serves_only_sanitized_calibration_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "chartqa:train:human:fixture"
    monkeypatch.setattr(
        main_module,
        "load_calibration_task_asset",
        lambda requested: TaskAsset(
            task_id=requested,
            media_type="image/png",
            image_sha256="a" * 64,
            content=b"fixture-png",
        ),
    )

    response = TestClient(app).get(f"/api/dashboard/tasks/{task_id}/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"fixture-png"


def test_dashboard_serves_answer_free_calibration_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "chartqa:train:human:fixture"
    monkeypatch.setattr(
        main_module,
        "load_calibration_task",
        lambda requested: TaskInput(
            task_id=requested,
            question="Which category has the highest value?",
            image_asset_id="fixture.png",
            image_sha256="a" * 64,
            source_split="train",
        ),
    )

    response = TestClient(app).get(f"/api/dashboard/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "question": "Which category has the highest value?",
    }

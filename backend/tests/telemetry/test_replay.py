from pathlib import Path

import pytest

from memory_distillation.task_execution import FinalAnswerAction
from memory_distillation.task_execution.models import ActionProposedEvent, RunStartedEvent
from memory_distillation.telemetry import ReplayReader, ReplayTraceWriter


def _writer(path: Path, *, phrases: tuple[str, ...] = ("private-reference",)) -> ReplayTraceWriter:
    return ReplayTraceWriter(
        path,
        origin="SYNTHETIC_FIXTURE",
        forbidden_phrases=phrases,
    )


def test_jsonl_is_append_only_and_replay_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    events = (RunStartedEvent(sequence=0, task_id="fixture", seed=17),)
    sink = _writer(path)
    for event in events:
        sink.emit(event)
    for event in events:
        sink.emit(event)

    replay = ReplayReader().read(sink.freeze())
    assert replay.mode == "REPLAY"
    assert replay.synthetic is True
    assert replay.events == (*events, *events)
    assert "private_reason" not in replay.model_dump_json()


def test_replay_rejects_crash_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    event = RunStartedEvent(sequence=0, task_id="fixture", seed=17)
    writer = _writer(path)
    writer.emit(event)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"broken"')

    with pytest.raises(ValueError, match="truncated"):
        ReplayReader().read(writer.freeze())


def test_replay_rejects_answer_bearing_reference_content(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    event = ActionProposedEvent(
        sequence=0,
        task_id="fixture",
        action=FinalAnswerAction(answer="secret-reference-answer"),
        model_id="student-v1",
        request_id="request-1",
    )
    writer = _writer(path, phrases=("secret-reference-answer",))
    writer.emit(event)

    with pytest.raises(ValueError, match="answer-bearing"):
        ReplayReader().read(writer.freeze())


def test_replay_redacts_predictions_and_detects_trace_mutation(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    event = ActionProposedEvent(
        sequence=0,
        task_id="fixture",
        action=FinalAnswerAction(answer="model-prediction"),
        model_id="student-v1",
        request_id="request-1",
    )
    writer = _writer(path)
    writer.emit(event)
    artifact = writer.freeze()

    replay = ReplayReader().read(artifact)
    projected = replay.events[0]
    assert isinstance(projected, ActionProposedEvent)
    assert isinstance(projected.action, FinalAnswerAction)
    assert projected.action.answer == "[REDACTED]"

    writer.emit(RunStartedEvent(sequence=1, task_id="changed", seed=1))
    with pytest.raises(ValueError, match="immutable provenance"):
        ReplayReader().read(artifact)

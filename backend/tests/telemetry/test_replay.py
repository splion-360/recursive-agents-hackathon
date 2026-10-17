from pathlib import Path

import pytest

from memory_distillation.task_execution import FinalAnswerAction
from memory_distillation.task_execution.models import ActionProposedEvent, RunStartedEvent
from memory_distillation.telemetry import JsonlEventSink, ReplayReader


def test_jsonl_is_append_only_and_replay_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    events = (RunStartedEvent(sequence=0, task_id="fixture", seed=17),)
    sink = JsonlEventSink(path)
    for event in events:
        sink.emit(event)
    for event in events:
        sink.emit(event)

    replay = ReplayReader().read(path, synthetic=True)
    assert replay.mode == "REPLAY"
    assert replay.synthetic is True
    assert replay.events == (*events, *events)
    assert "private_reason" not in replay.model_dump_json()


def test_replay_ignores_crash_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    event = RunStartedEvent(sequence=0, task_id="fixture", seed=17)
    JsonlEventSink(path).emit(event)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"broken"')

    assert ReplayReader().read(path, synthetic=True).events == (event,)


def test_replay_rejects_answer_bearing_reference_content(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    event = ActionProposedEvent(
        sequence=0,
        task_id="fixture",
        action=FinalAnswerAction(answer="secret-reference-answer"),
        model_id="student-v1",
        request_id="request-1",
    )
    JsonlEventSink(path).emit(event)

    with pytest.raises(ValueError, match="answer-bearing"):
        ReplayReader().read(
            path,
            synthetic=True,
            forbidden_phrases=("secret-reference-answer",),
        )

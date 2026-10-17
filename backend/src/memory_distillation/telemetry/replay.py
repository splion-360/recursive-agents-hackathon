"""Visualizer-only sanitized replay reader."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from memory_distillation.task_execution.models import RunEvent


class Replay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["REPLAY"] = "REPLAY"
    synthetic: bool
    events: tuple[RunEvent, ...]


class ReplayReader:
    def read(
        self,
        path: Path,
        *,
        synthetic: bool,
        forbidden_phrases: tuple[str, ...] = (),
    ) -> Replay:
        events: list[RunEvent] = []
        adapter = TypeAdapter(RunEvent)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                event = adapter.validate_python(json.loads(line))
                _validate_sanitized(event.model_dump_json(), forbidden_phrases)
                events.append(event)
        return Replay(synthetic=synthetic, events=tuple(events))


def _validate_sanitized(payload: str, forbidden_phrases: tuple[str, ...]) -> None:
    lowered = payload.casefold()
    private_markers = (
        "private_reason",
        "authorization",
        "api_key",
        "bearer ",
        "provider_payload",
        "scoring_reference",
    )
    if any(marker in lowered for marker in private_markers):
        raise ValueError("replay contains private or provider data")
    if any(phrase.strip().casefold() in lowered for phrase in forbidden_phrases if phrase.strip()):
        raise ValueError("replay contains answer-bearing reference data")

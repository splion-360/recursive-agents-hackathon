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
    def read(self, path: Path, *, synthetic: bool) -> Replay:
        events: list[RunEvent] = []
        adapter = TypeAdapter(RunEvent)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                events.append(adapter.validate_python(json.loads(line)))
        return Replay(synthetic=synthetic, events=tuple(events))

"""Append-only local telemetry. This module exposes no operational query API."""

import json
from pathlib import Path

from memory_distillation.task_execution.models import RunEvent


class JsonlEventSink:
    def __init__(self, path: Path, *, max_event_bytes: int = 64_000) -> None:
        self._path = path
        self._max_event_bytes = max_event_bytes

    def emit(self, event: RunEvent) -> None:
        payload = event.model_dump_json()
        lowered = payload.casefold()
        if any(term in lowered for term in ("api_key", "authorization", "private_reason")):
            raise ValueError("telemetry event contains forbidden private data")
        encoded = payload.encode()
        if len(encoded) > self._max_event_bytes:
            raise ValueError("telemetry event exceeds configured size limit")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json.loads(payload), separators=(",", ":")) + "\n")

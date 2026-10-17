"""Visualizer-only, provenance-bound sanitized replay projection."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from memory_distillation.task_execution.models import RunEvent
from memory_distillation.telemetry.jsonl import JsonlEventSink


class ReplayArtifact(BaseModel):
    """Immutable metadata that binds replay origin, bytes, and privacy policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    origin: Literal["SYNTHETIC_FIXTURE", "SANITIZED_RUN"]
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forbidden_phrases: tuple[str, ...] = Field(min_length=1)
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_metadata(self) -> Self:
        if not all(phrase.strip() for phrase in self.forbidden_phrases):
            raise ValueError("replay privacy phrases must be non-empty")
        expected = _metadata_hash(self.origin, self.trace_sha256, self.forbidden_phrases)
        if self.metadata_sha256 != expected:
            raise ValueError("replay metadata identity does not match its content")
        return self


class ReplayTraceWriter:
    """Trace producer that fixes replay provenance before the first event."""

    def __init__(
        self,
        path: Path,
        *,
        origin: Literal["SYNTHETIC_FIXTURE", "SANITIZED_RUN"],
        forbidden_phrases: tuple[str, ...],
    ) -> None:
        if not forbidden_phrases or not all(phrase.strip() for phrase in forbidden_phrases):
            raise ValueError("replay privacy phrases must be fixed and non-empty")
        self._path = path
        self._origin = origin
        self._forbidden_phrases = forbidden_phrases
        self._sink = JsonlEventSink(path)

    def emit(self, event: RunEvent) -> None:
        self._sink.emit(event)

    def freeze(self) -> ReplayArtifact:
        """Bind completed bytes to the provenance fixed by the trace producer."""

        trace_sha256 = hashlib.sha256(self._path.read_bytes()).hexdigest()
        return ReplayArtifact(
            path=self._path,
            origin=self._origin,
            trace_sha256=trace_sha256,
            forbidden_phrases=self._forbidden_phrases,
            metadata_sha256=_metadata_hash(self._origin, trace_sha256, self._forbidden_phrases),
        )


class Replay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["REPLAY"] = "REPLAY"
    synthetic: bool
    trace_sha256: str
    events: tuple[RunEvent, ...]


class ReplayReader:
    def read(self, artifact: ReplayArtifact) -> Replay:
        """Verify trace identity and return only its typed public projection."""

        content = artifact.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.trace_sha256:
            raise ValueError("replay trace bytes do not match immutable provenance")
        events: list[RunEvent] = []
        adapter = TypeAdapter(RunEvent)
        if content and not content.endswith(b"\n"):
            raise ValueError("replay trace contains a truncated JSONL record")
        for line in content.splitlines(keepends=True):
            raw = json.loads(line)
            _validate_sanitized(json.dumps(raw), artifact.forbidden_phrases)
            events.append(adapter.validate_python(_public_projection(raw)))
        return Replay(
            synthetic=artifact.origin == "SYNTHETIC_FIXTURE",
            trace_sha256=artifact.trace_sha256,
            events=tuple(events),
        )


def _public_projection(value: Any) -> Any:
    """Remove answer labels and correctness while preserving the event shape."""

    if isinstance(value, list):
        return [_public_projection(item) for item in value]
    if not isinstance(value, dict):
        return value
    projected = {key: _public_projection(item) for key, item in value.items()}
    if projected.get("type") == "final_answer" and "answer" in projected:
        projected["answer"] = "[REDACTED]"
    if projected.get("type") == "run_completed":
        projected["correct"] = None
    return projected


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
    if any(phrase.casefold() in lowered for phrase in forbidden_phrases):
        raise ValueError("replay contains answer-bearing reference data")


def _metadata_hash(origin: str, trace_sha256: str, forbidden_phrases: tuple[str, ...]) -> str:
    payload = {
        "origin": origin,
        "trace_sha256": trace_sha256,
        "forbidden_phrases": list(forbidden_phrases),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

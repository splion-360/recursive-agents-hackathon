"""Minimal write-only event sink contracts."""

from typing import Protocol


class EventSink[EventT](Protocol):
    """Accept ordered events without exposing a runtime read path."""

    def emit(self, event: EventT) -> None: ...


class InMemoryEventSink[EventT]:
    """Collect deterministic events for offline verification."""

    def __init__(self) -> None:
        self.events: tuple[EventT, ...] = ()

    def emit(self, event: EventT) -> None:
        self.events = (*self.events, event)

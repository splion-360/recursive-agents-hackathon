"""Private Reviewer seam for PASS or REPLACE decisions."""

from dataclasses import dataclass
from typing import Literal, Protocol

from memory_distillation.benchmark import TaskInput
from memory_distillation.task_execution.models import Observation, StudentAction


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    """Private review input for one proposed Student action."""

    task: TaskInput
    proposed_action: StudentAction
    observations: tuple[Observation, ...]
    guidance: tuple[str, ...]
    turn_index: int


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Private result; rationale is intentionally absent from public evidence."""

    decision: Literal["pass", "replace"]
    accepted_action: StudentAction
    private_reason: str


class Reviewer(Protocol):
    """Review one proposed action before it can execute."""

    def review(self, request: ReviewRequest) -> ReviewResult: ...


class DeterministicReviewer:
    """Return fixed decisions for offline Reviewer-boundary tests."""

    def __init__(self, *, results: tuple[ReviewResult, ...]) -> None:
        self._results = results
        self.requests: list[ReviewRequest] = []

    def review(self, request: ReviewRequest) -> ReviewResult:
        self.requests.append(request)
        try:
            return self._results[len(self.requests) - 1]
        except IndexError as error:
            raise RuntimeError("deterministic Reviewer script exhausted") from error

"""Provider-independent inference contracts."""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from memory_distillation.benchmark import TaskAsset, TaskInput

OutputT = TypeVar("OutputT", covariant=True)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One Student turn with answer-free task context."""

    task: TaskInput
    asset: TaskAsset
    observations: tuple[str, ...]
    seed: int
    turn_index: int


@dataclass(frozen=True, slots=True)
class InferenceResult[OutputT]:
    """Typed model output and deterministic usage evidence."""

    output: OutputT
    model_id: str
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class InferenceClient(Protocol[OutputT]):
    """Generate one typed output without exposing provider transport details."""

    def generate(self, request: InferenceRequest) -> InferenceResult[OutputT]: ...

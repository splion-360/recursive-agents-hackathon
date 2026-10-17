"""Deterministic inference for offline trajectories."""

from hashlib import sha256
from typing import TypeVar

from memory_distillation.inference.protocol import InferenceRequest, InferenceResult

OutputT = TypeVar("OutputT")


class DeterministicInference[OutputT]:
    """Return a fixed action sequence with stable model evidence."""

    def __init__(
        self,
        *,
        actions: tuple[OutputT, ...],
        model_id: str,
        input_tokens_per_call: int = 10,
        output_tokens_per_call: int = 3,
        latency_ms_per_call: int = 5,
    ) -> None:
        self._actions = actions
        self._model_id = model_id
        self._input_tokens = input_tokens_per_call
        self._output_tokens = output_tokens_per_call
        self._latency_ms = latency_ms_per_call

    def generate(self, request: InferenceRequest) -> InferenceResult[OutputT]:
        """Return the action at the current turn or fail when the script is exhausted."""

        try:
            action = self._actions[request.turn_index]
        except IndexError as error:
            raise RuntimeError("deterministic inference script exhausted") from error
        request_id = sha256(
            f"{request.task.task_id}:{request.seed}:{request.turn_index}".encode()
        ).hexdigest()[:24]
        return InferenceResult(
            output=action,
            model_id=self._model_id,
            request_id=f"fixture:{request_id}",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=self._latency_ms,
        )

"""Deterministic sandbox adapter for offline tests."""

from memory_distillation.sandbox.protocol import SandboxRequest, SandboxResult


class DeterministicSandbox:
    """Return configured results without ever executing local Python."""

    def __init__(self, *, results: dict[str, SandboxResult]) -> None:
        self._results = dict(results)
        self.requests: list[SandboxRequest] = []

    def execute(self, request: SandboxRequest) -> SandboxResult:
        """Record the request and return its exact configured result."""

        self.requests.append(request)
        try:
            return self._results[request.code]
        except KeyError as error:
            raise RuntimeError("no deterministic sandbox result configured") from error

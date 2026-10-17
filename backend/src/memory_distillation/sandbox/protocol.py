"""Contracts for sandbox-only generated Python execution."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

SandboxStatus = Literal["completed", "failed", "timed_out"]
SandboxFailureKind = Literal[
    "policy_denied", "command_failed", "transport_failed", "timed_out"
]


class SandboxRequest(BaseModel):
    """A bounded Python request with no network or host-path capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    code: str = Field(min_length=1)
    timeout_ms: int = Field(gt=0)
    network_access: Literal[False] = False


class SandboxResult(BaseModel):
    """Sanitized execution result returned by a sandbox adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SandboxStatus
    stdout: str
    stderr: str
    exit_code: int | None
    latency_ms: int = Field(ge=0)
    failure_kind: SandboxFailureKind | None = None


class Sandbox(Protocol):
    """Execute generated code only inside the configured boundary."""

    def execute(self, request: SandboxRequest) -> SandboxResult: ...

from collections.abc import Sequence

from memory_distillation.sandbox.openshell import CommandOutcome, OpenShellSandbox
from memory_distillation.sandbox.protocol import SandboxRequest


class RecordingRunner:
    def __init__(self, outcome: CommandOutcome | Exception) -> None:
        self._outcome = outcome
        self.calls: list[tuple[tuple[str, ...], str, float]] = []

    def run(self, args: Sequence[str], *, stdin: str, timeout_seconds: float) -> CommandOutcome:
        self.calls.append((tuple(args), stdin, timeout_seconds))
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def request(code: str = "print(42)") -> SandboxRequest:
    return SandboxRequest(task_id="chartqa-001", code=code, timeout_ms=5_000)


def test_executes_only_through_openshell_with_a_clean_environment() -> None:
    runner = RecordingRunner(
        CommandOutcome(returncode=0, stdout="42\n", stderr="", latency_ms=17)
    )
    sandbox = OpenShellSandbox(
        sandbox_name="memory-distill",
        gateway="nemoclaw",
        cli_path="/opt/homebrew/bin/openshell",
        runner=runner,
    )

    result = sandbox.execute(request())

    assert result.status == "completed"
    assert result.failure_kind is None
    assert result.stdout == "42\n"
    assert result.latency_ms == 17
    args, stdin, host_timeout = runner.calls[0]
    assert args[:8] == (
        "/opt/homebrew/bin/openshell",
        "--gateway",
        "nemoclaw",
        "sandbox",
        "exec",
        "--name",
        "memory-distill",
        "--timeout",
    )
    assert "/usr/bin/env" in args
    assert "-i" in args
    assert "/bin/sh" in args
    assert stdin == "print(42)"
    assert host_timeout > 5


def test_remote_timeout_is_typed() -> None:
    runner = RecordingRunner(
        CommandOutcome(
            returncode=124,
            stdout="partial",
            stderr="deadline exceeded",
            latency_ms=5_010,
        )
    )
    sandbox = OpenShellSandbox(sandbox_name="memory-distill", runner=runner)

    result = sandbox.execute(request("while True: pass"))

    assert result.status == "timed_out"
    assert result.failure_kind == "timed_out"
    assert result.exit_code == 124


def test_policy_denial_is_typed_without_exposing_provider_output() -> None:
    runner = RecordingRunner(
        CommandOutcome(
            returncode=1,
            stdout="",
            stderr="OpenShell policy denied outbound connection",
            latency_ms=8,
        )
    )
    sandbox = OpenShellSandbox(sandbox_name="memory-distill", runner=runner)

    result = sandbox.execute(request("import socket"))

    assert result.status == "failed"
    assert result.failure_kind == "policy_denied"


def test_missing_cli_fails_closed_and_never_executes_code_locally() -> None:
    runner = RecordingRunner(FileNotFoundError("openshell missing"))
    sandbox = OpenShellSandbox(sandbox_name="memory-distill", runner=runner)

    result = sandbox.execute(request("raise RuntimeError('must not run locally')"))

    assert result.status == "failed"
    assert result.failure_kind == "transport_failed"
    assert result.exit_code is None
    assert result.stdout == ""
    assert "openshell" in result.stderr.lower()


def test_output_is_bounded() -> None:
    runner = RecordingRunner(
        CommandOutcome(returncode=0, stdout="x" * 100_000, stderr="y" * 100_000, latency_ms=2)
    )
    sandbox = OpenShellSandbox(sandbox_name="memory-distill", runner=runner, output_limit=256)

    result = sandbox.execute(request())

    assert len(result.stdout.encode()) <= 256
    assert len(result.stderr.encode()) <= 256
    assert result.stdout.endswith("[truncated]\n")

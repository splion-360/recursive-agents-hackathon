import json
from collections.abc import Sequence

from memory_distillation.sandbox.openshell import CommandOutcome
from memory_distillation.sandbox.protocol import SandboxRequest, SandboxResult
from memory_distillation.sandbox.verify_openshell import verify_boundary, verify_external_policy


class VerificationSandbox:
    """Return typed evidence by verification task identity."""

    def __init__(self) -> None:
        self.requests: list[SandboxRequest] = []

    def execute(self, request: SandboxRequest) -> SandboxResult:
        self.requests.append(request)
        if (
            "denial" in request.task_id
            or "sensitive-path" in request.task_id
            or "path-escape" in request.task_id
            or request.task_id == "verify-filesystem-scope"
        ):
            return SandboxResult(
                status="failed",
                stdout="",
                stderr="MD_POLICY_DENIED",
                exit_code=77,
                latency_ms=1,
                failure_kind="policy_denied",
            )
        if request.task_id == "verify-timeout":
            return SandboxResult(
                status="timed_out",
                stdout="",
                stderr="",
                exit_code=124,
                latency_ms=1_000,
                failure_kind="timed_out",
            )
        if request.task_id == "verify-output-limit":
            return SandboxResult(
                status="completed",
                stdout="x" * (64 * 1024 - 13) + "\n[truncated]\n",
                stderr="",
                exit_code=0,
                latency_ms=1,
            )
        markers = {
            "verify-boundary_ok": "BOUNDARY_OK",
            "verify-workspace_writable": "WORKSPACE_WRITABLE",
            "verify-wrote": "WROTE",
            "verify-task_isolated": "TASK_ISOLATED",
            "verify-credentials_absent": "CREDENTIALS_ABSENT",
            "verify-memory_limited": "MEMORY_LIMITED",
        }
        return SandboxResult(
            status="completed",
            stdout=f"{markers[request.task_id]}\n",
            stderr="",
            exit_code=0,
            latency_ms=1,
        )


class QueueRunner:
    def __init__(self, outcomes: tuple[CommandOutcome, ...]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str,
        timeout_seconds: float,
    ) -> CommandOutcome:
        self.calls.append(tuple(args))
        assert stdin == ""
        assert timeout_seconds == 10
        return next(self._outcomes)


def test_boundary_report_requires_every_security_control() -> None:
    sandbox = VerificationSandbox()

    report = verify_boundary(sandbox, sandbox_name="memory-distill")

    assert report.passed
    assert report.checks["provider_route_denied"]
    assert report.checks["filesystem_path_escapes_denied"]
    assert report.checks["sensitive_paths_denied"]
    assert report.checks["task_isolated"]
    assert report.checks["output_limited"]
    assert len(sandbox.requests) == 22


def test_external_policy_requires_effective_deny_by_default_evidence() -> None:
    policy = {
        "status": "effective",
        "scope": "sandbox",
        "policy": {
            "filesystem_policy": {"read_write": ["/sandbox", "/tmp"]},
            "network_policies": {
                "managed": {
                    "endpoints": [
                        {
                            "host": "inference.local",
                            "port": 8443,
                            "enforcement": "enforce",
                        }
                    ]
                }
            },
        },
    }
    runner = QueueRunner(
        (
            CommandOutcome(
                returncode=0,
                stdout=json.dumps(policy),
                stderr="",
                latency_ms=1,
            ),
            CommandOutcome(
                returncode=1,
                stdout="",
                stderr="ConnectionRefusedError",
                latency_ms=1,
            ),
            CommandOutcome(
                returncode=1,
                stdout="",
                stderr="socket.gaierror",
                latency_ms=1,
            ),
        )
    )

    checks = verify_external_policy(
        sandbox_name="memory-distill",
        gateway="nemoclaw",
        runner=runner,
    )

    assert all(checks.values())
    assert len(runner.calls) == 3
    assert "('inference.local', 8443)" in runner.calls[2][-1]

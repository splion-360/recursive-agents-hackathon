"""Explicit live verification for the NemoClaw/OpenShell execution boundary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from memory_distillation.sandbox.openshell import (
    CommandOutcome,
    CommandRunner,
    OpenShellSandbox,
    SubprocessCommandRunner,
    build_isolated_python_command,
)
from memory_distillation.sandbox.protocol import Sandbox, SandboxRequest
from memory_distillation.settings import get_settings


class BoundaryReport(BaseModel):
    """Secret-free evidence from a live OpenShell boundary verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sandbox_name: str
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class _PolicyEndpoint(BaseModel):
    host: str
    port: int = Field(ge=1, le=65_535)
    enforcement: str


class _NetworkPolicy(BaseModel):
    endpoints: tuple[_PolicyEndpoint, ...] = ()


class _FilesystemPolicy(BaseModel):
    read_write: tuple[str, ...]


class _EffectivePolicy(BaseModel):
    filesystem_policy: _FilesystemPolicy
    network_policies: dict[str, _NetworkPolicy] = Field(default_factory=dict)


class _PolicyResponse(BaseModel):
    status: str
    scope: str
    policy: _EffectivePolicy


def verify_boundary(sandbox: Sandbox, *, sandbox_name: str) -> BoundaryReport:
    """Exercise execution, isolation, denial, timeout, and resource controls."""

    checks: dict[str, bool] = {}
    checks["bounded_python"] = _marker_check(
        sandbox,
        "print('BOUNDARY_OK')",
        marker="BOUNDARY_OK",
    )
    checks["network_denied"] = _policy_denial_check(
        sandbox,
        "verify-network-denial",
        """
import socket
socket.create_connection(("example.com", 443), timeout=1)
""",
    )
    checks["provider_route_denied"] = _policy_denial_check(
        sandbox,
        "verify-provider-route-denial",
        """
import socket
socket.create_connection(("inference.local", 443), timeout=1)
""",
    )
    checks["filesystem_scoped"] = _policy_denial_check(
        sandbox,
        "verify-filesystem-scope",
        """
from pathlib import Path
Path("/tmp/cross-task-shared").write_text("forbidden")
""",
    )
    path_escape_programs = (
        "import os\nopen('source', 'w').close()\nos.rename('source', '/tmp/escaped')",
        "import os\nopen('source', 'w').close()\nos.link('source', '/tmp/escaped')",
        "import os\nos.symlink('/tmp/escaped', 'link')",
        "import os\nos.utime('/tmp', None)",
        "import os\nos.setxattr('/tmp', b'user.test', b'value')",
        "import os\nos.removexattr('/tmp', b'user.test')",
    )
    checks["filesystem_path_escapes_denied"] = all(
        _policy_denial_check(sandbox, f"verify-path-escape-{index}", code)
        for index, code in enumerate(path_escape_programs)
    )
    checks["workspace_writable"] = _marker_check(
        sandbox,
        """
from pathlib import Path
Path("task-local.txt").write_text("ok")
print("WORKSPACE_WRITABLE")
""",
        marker="WORKSPACE_WRITABLE",
    )
    _marker_check(
        sandbox,
        "from pathlib import Path\nPath('private.txt').write_text('first task')\nprint('WROTE')",
        marker="WROTE",
    )
    checks["task_isolated"] = _marker_check(
        sandbox,
        """
from pathlib import Path
print("TASK_ISOLATED" if not Path("private.txt").exists() else "TASK_LEAKED")
""",
        marker="TASK_ISOLATED",
    )
    sensitive_paths = ("/sandbox", "/app", "/workspace", "/data", "/Users")
    checks["sensitive_paths_denied"] = all(
        _policy_denial_check(
            sandbox,
            f"verify-sensitive-path-{index}",
            f"import os\nos.listdir({path!r})",
        )
        for index, path in enumerate(sensitive_paths)
    )
    checks["credentials_absent"] = _marker_check(
        sandbox,
        """
import os
secret_names = [
    name
    for name in os.environ
    if any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET"))
]
print("CREDENTIALS_ABSENT" if not secret_names else "CREDENTIALS_VISIBLE")
""",
        marker="CREDENTIALS_ABSENT",
    )
    checks["memory_limited"] = _marker_check(
        sandbox,
        """
try:
    bytearray(700 * 1024 * 1024)
except MemoryError:
    print("MEMORY_LIMITED")
else:
    print("MEMORY_UNLIMITED")
""",
        marker="MEMORY_LIMITED",
    )
    timeout_result = sandbox.execute(
        SandboxRequest(
            task_id="verify-timeout",
            code="while True:\n    pass",
            timeout_ms=1_000,
        )
    )
    checks["timeout_enforced"] = (
        timeout_result.status == "timed_out" and timeout_result.failure_kind == "timed_out"
    )
    output_result = sandbox.execute(
        SandboxRequest(task_id="verify-output-limit", code="print('x' * 100_000)", timeout_ms=5_000)
    )
    checks["output_limited"] = (
        output_result.status == "completed"
        and len(output_result.stdout.encode()) <= 64 * 1024
        and output_result.stdout.endswith("[truncated]\n")
    )
    return BoundaryReport(sandbox_name=sandbox_name, checks=checks)


def verify_external_policy(
    *,
    sandbox_name: str,
    gateway: str,
    cli_path: str = "openshell",
    runner: CommandRunner | None = None,
) -> dict[str, bool]:
    """Verify the effective OpenShell policy independently of the Python audit hook."""

    command_runner = runner or SubprocessCommandRunner()
    policy_result = command_runner.run(
        (
            cli_path,
            "--gateway",
            gateway,
            "policy",
            "get",
            sandbox_name,
            "--full",
            "--output",
            "json",
        ),
        stdin="",
        timeout_seconds=10,
    )
    policy_valid = False
    declared_endpoints: tuple[tuple[str, int], ...] = ()
    if policy_result.returncode == 0:
        try:
            policy = _PolicyResponse.model_validate_json(policy_result.stdout)
        except ValueError:
            pass
        else:
            policy_valid = _is_deny_by_default(policy)
            declared_endpoints = tuple(
                sorted(
                    {
                        (endpoint.host, endpoint.port)
                        for network_policy in policy.policy.network_policies.values()
                        for endpoint in network_policy.endpoints
                    }
                )
            )

    public_probe = command_runner.run(
        _raw_socket_probe_command(cli_path, gateway, sandbox_name, "1.1.1.1"),
        stdin="",
        timeout_seconds=10,
    )
    endpoint_probes = {
        (host, port): command_runner.run(
            _raw_socket_probe_command(cli_path, gateway, sandbox_name, host, port),
            stdin="",
            timeout_seconds=10,
        )
        for host, port in declared_endpoints
    }
    provider_probes = [
        outcome
        for (host, _port), outcome in endpoint_probes.items()
        if host == "inference.local"
    ]
    return {
        "external_policy_deny_by_default": policy_valid,
        "external_public_network_denied": _has_network_denial(public_probe),
        "external_declared_endpoints_denied": policy_valid
        and bool(endpoint_probes)
        and all(_has_network_denial(outcome) for outcome in endpoint_probes.values()),
        "external_provider_route_denied": bool(provider_probes)
        and all(_has_network_denial(outcome) for outcome in provider_probes),
    }


def _marker_check(sandbox: Sandbox, code: str, *, marker: str) -> bool:
    result = sandbox.execute(
        SandboxRequest(task_id=f"verify-{marker.casefold()}", code=code, timeout_ms=5_000)
    )
    return result.status == "completed" and result.stdout.strip() == marker


def _policy_denial_check(sandbox: Sandbox, task_id: str, code: str) -> bool:
    result = sandbox.execute(SandboxRequest(task_id=task_id, code=code, timeout_ms=5_000))
    return result.status == "failed" and result.failure_kind == "policy_denied"


def _is_deny_by_default(response: _PolicyResponse) -> bool:
    endpoints = [
        endpoint
        for policy in response.policy.network_policies.values()
        for endpoint in policy.endpoints
    ]
    allowed_writes = {"/sandbox", "/tmp", "/dev/null", "/dev/pts", "/sandbox/.hermes"}
    return (
        response.status.casefold() == "effective"
        and response.scope.casefold() == "sandbox"
        and set(response.policy.filesystem_policy.read_write) <= allowed_writes
        and all(
            endpoint.enforcement == "enforce"
            and endpoint.host not in {"*", "0.0.0.0", "1.1.1.1"}
            for endpoint in endpoints
        )
    )


def _raw_socket_probe_command(
    cli_path: str,
    gateway: str,
    sandbox_name: str,
    host: str,
    port: int = 443,
) -> tuple[str, ...]:
    code = f"import socket; socket.create_connection(({host!r}, {port}), 2)"
    return build_isolated_python_command(
        cli_path=cli_path,
        gateway=gateway,
        sandbox_name=sandbox_name,
        timeout_seconds=5,
        program=code,
    )


def _has_network_denial(outcome: CommandOutcome | None) -> bool:
    if outcome is None or outcome.returncode in {None, 0}:
        return False
    return any(
        signature in outcome.stderr
        for signature in ("ConnectionRefusedError", "gaierror", "Network is unreachable")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run paid/live verification only after an explicit confirmation flag."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="acknowledge execution against the configured live OpenShell sandbox",
    )
    args = parser.parse_args(argv)
    if not args.confirm_live:
        parser.error("--confirm-live is required")

    settings = get_settings()
    if settings.openshell_sandbox_name is None:
        parser.error("OPENSHELL_SANDBOX_NAME is required")
    sandbox = OpenShellSandbox(
        sandbox_name=settings.openshell_sandbox_name,
        gateway=settings.openshell_gateway,
    )
    report = verify_boundary(sandbox, sandbox_name=settings.openshell_sandbox_name)
    external_checks = verify_external_policy(
        sandbox_name=settings.openshell_sandbox_name,
        gateway=settings.openshell_gateway,
    )
    report = report.model_copy(update={"checks": {**report.checks, **external_checks}})
    print(json.dumps(report.model_dump(), sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

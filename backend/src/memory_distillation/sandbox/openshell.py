"""NemoClaw-managed OpenShell adapter for untrusted generated Python."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from typing import BinaryIO, Protocol

from pydantic import BaseModel, ConfigDict, Field

from memory_distillation.sandbox.protocol import (
    SandboxFailureKind,
    SandboxRequest,
    SandboxResult,
    SandboxStatus,
)

_DEFAULT_OUTPUT_LIMIT = 64 * 1024
_HOST_TIMEOUT_GRACE_SECONDS = 5.0
_TRUNCATION_MARKER = "\n[truncated]\n"

_RUNTIME_WRAPPER = """\
set -u
runtime_dir="$1"
bootstrap="$2"
/usr/bin/mkdir -m 700 "$runtime_dir"
/usr/bin/cp /usr/bin/python3 "$runtime_dir/student-python"
/usr/bin/chmod 500 "$runtime_dir/student-python"
"$runtime_dir/student-python" -I -u -c "$bootstrap"
status=$?
/usr/bin/unlink "$runtime_dir/student-python"
/usr/bin/rmdir "$runtime_dir"
exit "$status"
"""

_PYTHON_BOOTSTRAP = """\
import os
import resource
import shutil
import sys
from pathlib import Path

cpu_seconds = __CPU_SECONDS__
workspace = Path("/tmp/__WORKSPACE_NAME__")
limits = (
    (resource.RLIMIT_CORE, 0),
    (resource.RLIMIT_CPU, cpu_seconds),
    (resource.RLIMIT_FSIZE, 1_048_576),
    (resource.RLIMIT_NOFILE, 64),
    (resource.RLIMIT_NPROC, 32),
    (resource.RLIMIT_AS, 536_870_912),
)
for resource_kind, maximum in limits:
    resource.setrlimit(resource_kind, (maximum, maximum))

if workspace.is_symlink():
    workspace.unlink()
elif workspace.exists():
    shutil.rmtree(workspace)
workspace.mkdir(mode=0o700)
os.chdir(workspace)

class SandboxPolicyViolation(PermissionError):
    pass

def is_within(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

def checked_path(raw_path):
    if isinstance(raw_path, int):
        return None
    return Path(os.path.abspath(os.fspath(raw_path)))

def audit(event, args):
    if event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.exec", "os.spawn",
        "pty.spawn"
    }:
        raise SandboxPolicyViolation("process or network capability denied")
    if event == "import" and str(args[0]).partition(".")[0] in {
        "ctypes", "multiprocessing", "resource", "subprocess"
    }:
        raise SandboxPolicyViolation("unsafe module denied")
    if event == "open":
        path = checked_path(args[0])
        if path is None or is_within(path, workspace):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = (
            isinstance(mode, str) and any(character in mode for character in "wax+")
        ) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        readable_system_path = any(
            is_within(path, root)
            for root in (Path("/usr"), Path("/lib"), Path("/lib64"), Path("/etc/ssl"))
        )
        if writing or not readable_system_path:
            raise SandboxPolicyViolation("filesystem capability denied")
    if event in {"os.link", "os.symlink"}:
        raise SandboxPolicyViolation("filesystem link capability denied")
    if event == "os.rename":
        paths = (checked_path(args[0]), checked_path(args[1]))
        if any(path is not None and not is_within(path, workspace) for path in paths):
            raise SandboxPolicyViolation("filesystem capability denied")
    if event in {
        "os.chdir", "os.chmod", "os.chown", "os.listdir", "os.mkdir",
        "os.remove", "os.removexattr", "os.rmdir", "os.scandir", "os.setxattr",
        "os.truncate", "os.utime",
    }:
        path = checked_path(args[0])
        if path is not None and not is_within(path, workspace):
            raise SandboxPolicyViolation("filesystem capability denied")

sys.addaudithook(audit)
try:
    source = sys.stdin.read()
    exec(compile(source, "<student-code>", "exec"), {"__name__": "__main__"})
except SandboxPolicyViolation as error:
    print(f"MD_POLICY_DENIED: {error}", file=sys.stderr)
    raise SystemExit(77) from None
finally:
    shutil.rmtree(workspace, ignore_errors=True)
"""


class CommandOutcome(BaseModel):
    """Sanitized result of invoking the OpenShell CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    returncode: int | None
    stdout: str
    stderr: str
    latency_ms: int = Field(ge=0)
    timed_out: bool = False


class CommandRunner(Protocol):
    """Host command seam used to contract-test the OpenShell invocation."""

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str,
        timeout_seconds: float,
    ) -> CommandOutcome: ...


class SubprocessCommandRunner:
    """Invoke only the configured OpenShell CLI with bounded host waiting."""

    def __init__(self, *, output_limit: int = _DEFAULT_OUTPUT_LIMIT) -> None:
        self._output_limit = output_limit

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str,
        timeout_seconds: float,
    ) -> CommandOutcome:
        started = time.monotonic()
        safe_environment = {
            key: value
            for key in ("HOME", "PATH", "SSL_CERT_FILE", "SSL_CERT_DIR")
            if (value := os.environ.get(key)) is not None
        }
        process = subprocess.Popen(
            list(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=safe_environment,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("OpenShell process pipes were not created")
        stdout_capture = _BoundedCapture(self._output_limit)
        stderr_capture = _BoundedCapture(self._output_limit)
        stdout_thread = threading.Thread(
            target=stdout_capture.drain,
            args=(process.stdout,),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=stderr_capture.drain,
            args=(process.stderr,),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            process.stdin.write(stdin.encode())
            process.stdin.close()
        except BrokenPipeError:
            pass
        timed_out = False
        try:
            returncode: int | None = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
            returncode = None
        stdout_thread.join()
        stderr_thread.join()
        return CommandOutcome(
            returncode=returncode,
            stdout=stdout_capture.text(),
            stderr=stderr_capture.text(),
            latency_ms=_elapsed_ms(started),
            timed_out=timed_out,
        )


class _BoundedCapture:
    """Drain a child pipe while retaining at most the configured byte count."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._content = bytearray()
        self._truncated = False

    def drain(self, stream: BinaryIO) -> None:
        while chunk := stream.read(8_192):
            remaining = self._limit - len(self._content)
            if remaining > 0:
                self._content.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self._truncated = True

    def text(self) -> str:
        if not self._truncated:
            return self._content.decode(errors="replace")
        marker = _TRUNCATION_MARKER.encode()
        content = self._content[: self._limit - len(marker)]
        return (content + marker).decode(errors="replace")


class OpenShellSandbox:
    """Execute Python in one pre-provisioned NemoClaw/OpenShell sandbox.

    The adapter never invokes a local Python interpreter. OpenShell receives a
    clean remote environment, and a bootstrap applies per-request process limits
    plus an ephemeral working directory before compiling the untrusted source.
    """

    def __init__(
        self,
        *,
        sandbox_name: str,
        gateway: str = "nemoclaw",
        cli_path: str = "openshell",
        runner: CommandRunner | None = None,
        output_limit: int = _DEFAULT_OUTPUT_LIMIT,
    ) -> None:
        if not sandbox_name:
            raise ValueError("sandbox_name is required")
        if output_limit < len(_TRUNCATION_MARKER.encode()):
            raise ValueError("output_limit is too small for the truncation marker")
        self._sandbox_name = sandbox_name
        self._gateway = gateway
        self._cli_path = cli_path
        self._runner = runner or SubprocessCommandRunner(output_limit=output_limit)
        self._output_limit = output_limit
        self._execution_lock = threading.Lock()

    def execute(self, request: SandboxRequest) -> SandboxResult:
        """Execute a request through OpenShell or return typed failure evidence."""

        with self._execution_lock:
            return self._execute_serially(request)

    def _execute_serially(self, request: SandboxRequest) -> SandboxResult:
        """Keep namespaced workspace cleanup isolated from concurrent requests."""

        remote_timeout = max(1, math.ceil(request.timeout_ms / 1_000))
        command = self._command(remote_timeout)
        try:
            outcome = self._runner.run(
                command,
                stdin=request.code,
                timeout_seconds=remote_timeout + _HOST_TIMEOUT_GRACE_SECONDS,
            )
        except OSError as error:
            return SandboxResult(
                status="failed",
                stdout="",
                stderr=f"OpenShell transport unavailable: {type(error).__name__}",
                exit_code=None,
                latency_ms=0,
                failure_kind="transport_failed",
            )

        stdout = _truncate(outcome.stdout, self._output_limit)
        stderr = _truncate(outcome.stderr, self._output_limit)
        if outcome.timed_out or outcome.returncode == 124:
            status: SandboxStatus = "timed_out"
            failure_kind: SandboxFailureKind | None = "timed_out"
        elif outcome.returncode == 0:
            status = "completed"
            failure_kind = None
        else:
            status = "failed"
            failure_kind = (
                "policy_denied" if _looks_like_policy_denial(stderr) else "command_failed"
            )
        return SandboxResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=outcome.returncode,
            latency_ms=outcome.latency_ms,
            failure_kind=failure_kind,
        )

    def _command(self, remote_timeout: int) -> tuple[str, ...]:
        execution_id = uuid.uuid4().hex
        workspace_name = f"memory-distillation-{execution_id}"
        bootstrap = _PYTHON_BOOTSTRAP.replace(
            "__CPU_SECONDS__", str(remote_timeout)
        ).replace(
            "__WORKSPACE_NAME__", workspace_name
        )
        return build_isolated_python_command(
            cli_path=self._cli_path,
            gateway=self._gateway,
            sandbox_name=self._sandbox_name,
            timeout_seconds=remote_timeout,
            program=bootstrap,
            execution_id=execution_id,
        )


def build_isolated_python_command(
    *,
    cli_path: str,
    gateway: str,
    sandbox_name: str,
    timeout_seconds: int,
    program: str,
    execution_id: str | None = None,
) -> tuple[str, ...]:
    """Build a fixed OpenShell command using a non-allowlisted Python copy."""

    identifier = execution_id or uuid.uuid4().hex
    runtime_directory = f"/tmp/memory-distillation-runtime-{identifier}"
    return (
        cli_path,
        "--gateway",
        gateway,
        "sandbox",
        "exec",
        "--name",
        sandbox_name,
        "--timeout",
        str(timeout_seconds),
        "--no-tty",
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED=0",
        "TMPDIR=/tmp",
        "/bin/sh",
        "-c",
        _RUNTIME_WRAPPER,
        "student-runtime",
        runtime_directory,
        program,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1_000))


def _truncate(value: str, limit: int) -> str:
    encoded = value.encode(errors="replace")
    if len(encoded) <= limit:
        return value
    marker = _TRUNCATION_MARKER.encode()
    return (encoded[: limit - len(marker)] + marker).decode(errors="replace")


def _looks_like_policy_denial(stderr: str) -> bool:
    normalized = stderr.casefold()
    return (
        "md_policy_denied:" in normalized
        or "policy denied" in normalized
        or "network access denied" in normalized
    )

# Use OpenShell as the generated-code execution boundary

The student’s `run_python` actions will execute inside an OpenShell sandbox managed through NemoClaw. This makes the NemoClaw and OpenShell integration necessary for safe, auditable tool execution while keeping their responsibility narrow enough that the rest of the experiment orchestration is not coupled to the sandbox framework.

The production adapter invokes `openshell sandbox exec` against a pre-provisioned,
NemoClaw-owned sandbox. It does not create sandboxes, attach providers, or fall
back to a host Python process. Each request receives a clean environment, an
ephemeral namespaced workspace, bounded CPU, memory, file, process, descriptor,
output, and wall-clock resources, and typed failure evidence. Every request uses
a unique workspace, and the Python audit boundary denies reads and writes outside
that workspace. Normal exits remove it; a timed-out process may leave an
inaccessible namespaced directory that later host maintenance can remove.

The verified local sandbox is named `memory-distill` on the `nemoclaw` gateway.
It has no host mounts or optional package/network presets. Crusoe inference is
attached through OpenShell provider storage for the managed Hermes runtime; the
Student Python command receives no credential environment variables and cannot
use that provider route. Runtime configuration therefore records only the
sandbox and gateway names, never an OpenShell API key.

Run the explicit live verification after provisioning or changing the policy:

```bash
cd backend
uv run verify-openshell --confirm-live
```

The command fails unless bounded execution, network denial, filesystem scope,
host-file absence, credential absence, memory limits, timeout enforcement, and
cross-task workspace isolation all pass.

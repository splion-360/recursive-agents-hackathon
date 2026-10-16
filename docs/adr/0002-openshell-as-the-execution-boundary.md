# Use OpenShell as the generated-code execution boundary

The student’s `run_python` actions will execute inside an OpenShell sandbox managed through NemoClaw. This makes the NemoClaw and OpenShell integration necessary for safe, auditable tool execution while keeping their responsibility narrow enough that the rest of the experiment orchestration is not coupled to the sandbox framework.

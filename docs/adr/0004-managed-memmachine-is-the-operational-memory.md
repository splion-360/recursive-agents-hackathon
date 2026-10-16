# Use managed MemMachine as the sole operational memory

Reviewed episodes, candidate behaviors, accepted versions, and Student retrieval will use the managed MemMachine service so the project demonstrates a real cross-session memory integration without spending the hackathon operating its databases. A local gitignored JSONL experiment trace may record sanitized telemetry for metrics, debugging, and replay, but no agent may read it as context or use it as a fallback memory system.

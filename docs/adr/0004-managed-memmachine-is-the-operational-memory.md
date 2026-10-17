# Use managed MemMachine as the sole operational memory

Reviewed episodes, candidate behaviors, accepted versions, and Student retrieval will use the managed MemMachine service so the project demonstrates a real cross-session memory integration without spending the hackathon operating its databases. A local gitignored JSONL experiment trace may record sanitized telemetry for metrics, debugging, and replay, but no agent may read it as context or use it as a fallback memory system.

The managed adapter stores canonically serialized Reviewed Episodes, complete
Candidate Versions, derived retrieval views, and the bootstrap Accepted Version
pointer as idempotent episodic records with stable identities and content
hashes. Reads exhaust API pagination and validate every record before returning
it. Retrieval loads an explicitly pinned immutable view and renders Behavior
Cards locally through the frozen retrieval policy; MemMachine's unconstrained
semantic search is not a serving fallback.

Live verification on October 17, 2026 confirmed record round trips,
idempotency, pagination, pinned retrieval, and project isolation. It also
confirmed that the current public API exposes append, list/search, and delete,
but no conditional update primitive. The adapter therefore reports
`atomic_pointer_updates_supported = False`. ADR 0013 supersedes the original
promotion-blocking conclusion by defining a narrower, explicitly single-writer
immutable transition protocol. Native compare-and-set remains unavailable.

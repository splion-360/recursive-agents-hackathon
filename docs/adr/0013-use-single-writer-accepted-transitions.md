# Use immutable single-writer Accepted Version transitions

MemMachine's verified public API does not expose conditional updates. The one-shot measured
lifecycle nevertheless has exactly one orchestrator and one Promotion Decision. For this bounded
run, promotion records one immutable transition keyed by the expected Accepted Version. The
effective Accepted Version is derived by validating and traversing that append-only chain.

This is not presented as native compare-and-set. The lifecycle enforces one writer, re-reads the
transition after writing, validates both version content hashes, and fails closed on duplicate,
ambiguous, cyclic, stale, or conflicting transitions. A rejected candidate writes no transition.
Concurrent promotion writers remain unsupported.

This decision supersedes only the promotion-blocking conclusion in ADR 0004. Managed MemMachine
remains the sole Operational Memory, and immutable versions, pinned retrieval, and local-telemetry
restrictions remain unchanged.

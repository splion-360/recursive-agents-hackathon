from memory_distillation.versioning import (
    CandidateVersion,
    EpisodeSnapshot,
    PairedOutcome,
    PromotionThresholds,
    RetrievalPolicy,
    VersionRegistry,
    decide_promotion,
)


def _candidate() -> CandidateVersion:
    return CandidateVersion(
        version_id="candidate:test",
        parent_version_id="accepted-v000",
        kappa_hash="kappa",
        snapshot=EpisodeSnapshot(episode_ids=("e1",), episode_hashes=("h1",), snapshot_hash="s"),
        cards=(),
        operations=(),
        retrieval_policy=RetrievalPolicy(top_k=0, minimum_score=0, token_budget=0),
        content_hash="c" * 64,
    )


def _outcome(
    *, accepted: bool, candidate: bool, latency: int = 100, tokens: int = 100
) -> PairedOutcome:
    return PairedOutcome(
        task_id="task-1",
        seed=17,
        configuration_hash="frozen-config",
        accepted_correct=accepted,
        candidate_correct=candidate,
        accepted_latency_ms=100,
        candidate_latency_ms=latency,
        accepted_tokens=100,
        candidate_tokens=tokens,
    )


def test_promotion_advances_pointer_once_at_inclusive_boundaries() -> None:
    registry = VersionRegistry()
    decision = decide_promotion(
        registry=registry,
        candidate=_candidate(),
        outcomes=(_outcome(accepted=True, candidate=True, latency=120, tokens=120),),
        thresholds=PromotionThresholds(
            maximum_latency_regression=0.2, maximum_token_regression=0.2
        ),
    )

    assert decision.decision == "PROMOTE"
    assert registry.accepted_version_id == "candidate:test"
    assert (
        registry.compare_and_set_accepted(expected="accepted-v000", replacement="candidate:test")
        is False
    )


def test_rejection_preserves_both_versions_and_pointer() -> None:
    registry = VersionRegistry()
    candidate = _candidate()
    decision = decide_promotion(
        registry=registry,
        candidate=candidate,
        outcomes=(_outcome(accepted=True, candidate=False),),
        thresholds=PromotionThresholds(maximum_negative_transfer_rate=0),
    )

    assert decision.decision == "REJECT"
    assert "capability" in decision.failed_gates
    assert "negative_transfer" in decision.failed_gates
    assert registry.accepted_version_id == "accepted-v000"
    assert registry.get_view(candidate.version_id).content_hash == candidate.content_hash

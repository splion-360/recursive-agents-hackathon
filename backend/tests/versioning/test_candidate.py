import hashlib
import json
from typing import Any

import pytest

from memory_distillation.benchmark import ExperimentManifest, ManifestEntry, Score, TaskInput
from memory_distillation.operational_memory import (
    EpisodeProvenance,
    ReviewedEpisode,
    episode_from_run,
)
from memory_distillation.review_policy import KappaArtifact, KappaRule, RuleEvidence
from memory_distillation.task_execution import ActionReview, FinalAnswerAction, TaskRun
from memory_distillation.versioning import (
    CandidateVersion,
    CardOperation,
    RetrievalCalibration,
    RetrievalPolicy,
    RetrievalTrial,
    VersionView,
    build_candidate,
    calibrate_retrieval,
)


def _question(index: int) -> str:
    return f"What chart label {index} should be checked?"


@pytest.mark.parametrize("field", ("trigger", "instruction"))
@pytest.mark.parametrize("value", ("", "   "))
def test_create_operation_rejects_empty_behavior_text(field: str, value: str) -> None:
    payload = {
        "kind": "CREATE",
        "card_id": "card:invalid",
        "trigger": "inspect the chart",
        "instruction": "verify the relevant marks",
        "evidence_episode_ids": ("episode-1",),
    }
    payload[field] = value

    with pytest.raises(ValueError, match=r"trigger|instruction"):
        CardOperation.model_validate(payload)


def _manifest() -> ExperimentManifest:
    entries = tuple(
        ManifestEntry(
            task_id=f"collection-{index:02d}",
            image_asset_id=f"collection-{index:02d}.png",
            image_sha256=f"{index + 1:064x}",
            question_sha256=hashlib.sha256(_question(index).encode()).hexdigest(),
            source_split="train",
            question_form="wh",
            operation_type="lookup",
        )
        for index in range(40)
    )
    return ExperimentManifest.create(
        name="collection",
        seed=17,
        dataset_revision="fixture",
        source_sha256="f" * 64,
        entries=entries,
    )


def _kappa() -> KappaArtifact:
    rule = KappaRule(
        rule_id="verify-labels",
        action_types=("inspect_image",),
        instruction="Inspect relevant chart labels before computing.",
    )
    evidence = RuleEvidence(
        rule_id=rule.rule_id,
        discovery_task_ids=tuple(f"d-{index}" for index in range(12)),
        regression_task_ids=tuple(f"r-{index}" for index in range(12)),
        discovery_helped=1,
        regression_helped=1,
        regression_harmed=0,
        action_compatible=True,
    )
    payload = {
        "manifest_hash": "calibration-manifest",
        "revision": 1,
        "rules": [rule.model_dump(mode="json")],
        "evidence": [evidence.model_dump(mode="json")],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return KappaArtifact(
        artifact_id=f"kappa:{digest[:16]}",
        manifest_hash="calibration-manifest",
        revision=1,
        rules=(rule,),
        guidance=(rule.instruction,),
        evidence=(evidence,),
        content_hash=digest,
    )


def _episodes(manifest: ExperimentManifest, kappa: KappaArtifact) -> tuple[ReviewedEpisode, ...]:
    manifest_ids = tuple(entry.task_id for entry in manifest.entries)
    provenance = EpisodeProvenance(
        code_revision="abc",
        manifest_hash=manifest.content_sha256,
        configuration_hash="config",
        lifecycle_pass_id="pass-1",
        purpose="collection",
        manifest_task_ids=manifest_ids,
        accepted_version_id="accepted-v000",
        accepted_content_hash="0" * 64,
        kappa_hash=kappa.content_hash,
    )
    episodes = []
    for index, entry in enumerate(manifest.entries):
        task = TaskInput(
            task_id=entry.task_id,
            question=_question(index),
            image_asset_id=entry.image_asset_id,
            image_sha256=entry.image_sha256,
            source_split="train",
        )
        action = FinalAnswerAction(answer="42")
        eligible = index == 0
        run = TaskRun(
            task_id=task.task_id,
            seed=1,
            termination_reason="completed" if eligible else "inference_failed",
            actions=(action,) if eligible else (),
            observations=(),
            reviews=(
                ActionReview(
                    decision="pass",
                    proposed_action=action,
                    accepted_action=action,
                    model_id="reviewer-v1",
                    request_id="review:1",
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1,
                ),
            )
            if eligible
            else (),
            events=(),
            score=Score(correct=True, method="text_exact") if eligible else None,
            input_tokens=1,
            output_tokens=1,
            elapsed_ms=1,
            tool_calls=0,
            python_calls=0,
            review_calls=1 if eligible else 0,
            kappa_hash=kappa.content_hash,
        )
        episodes.append(episode_from_run(task=task, run=run, provenance=provenance))
    return tuple(episodes)


def _parent_view() -> VersionView:
    return VersionView(
        version_id="accepted-v000",
        cards=(),
        retrieval_policy=RetrievalPolicy(
            top_k=0,
            minimum_score=0,
            token_budget=0,
            calibration_snapshot_hash="0" * 64,
        ),
        content_hash="0" * 64,
    )


def _candidate_inputs() -> tuple[
    ExperimentManifest,
    KappaArtifact,
    tuple[ReviewedEpisode, ...],
    CardOperation,
    RetrievalCalibration,
]:
    manifest = _manifest()
    kappa = _kappa()
    episodes = _episodes(manifest, kappa)
    eligible = episodes[0]
    operation = CardOperation(
        kind="CREATE",
        card_id="card:verify-labels",
        trigger="What chart label should be checked",
        instruction="Inspect the relevant chart region before computing.",
        evidence_episode_ids=(eligible.episode_id,),
    )
    calibration = calibrate_retrieval(
        episode_ids=(eligible.episode_id,),
        trials=(
            RetrievalTrial(
                query_id=eligible.task.task_id,
                episode_id=eligible.episode_id,
                top_k=1,
                minimum_score=0,
                token_budget=200,
                relevant_card_ids=(operation.card_id,),
                retrieved_card_ids=(operation.card_id,),
            ),
        ),
    )
    return manifest, kappa, episodes, operation, calibration


def _build(**overrides: Any) -> CandidateVersion:
    manifest, kappa, episodes, operation, calibration = _candidate_inputs()
    values = {
        "episodes": episodes,
        "collection_manifest": manifest,
        "operations": (operation,),
        "kappa": kappa,
        "retrieval_calibration": calibration,
        "contamination_phrases": ("fixture-answer",),
        "parent_view": _parent_view(),
    }
    values.update(overrides)
    return build_candidate(**values)  # type: ignore[arg-type]


def test_candidate_binds_complete_manifest_kappa_and_actual_retrieval() -> None:
    manifest, kappa, episodes, _, _ = _candidate_inputs()
    candidate = _build(episodes=episodes, collection_manifest=manifest, kappa=kappa)

    assert candidate.kappa_hash == kappa.content_hash
    assert candidate.snapshot.episode_ids == (episodes[0].episode_id,)


def test_candidate_rejects_incomplete_collection_and_wrong_kappa() -> None:
    _, kappa, episodes, _, _ = _candidate_inputs()
    with pytest.raises(ValueError, match="complete Collection manifest"):
        _build(episodes=episodes[:-1])
    with pytest.raises(ValueError, match="Collection Kappa"):
        _build(kappa=kappa.model_copy(update={"content_hash": "a" * 64}))

    forged = episodes[0].model_copy(
        update={"task": episodes[0].task.model_copy(update={"question": "Forged question"})}
    )
    with pytest.raises(ValueError, match="authoritative manifest"):
        _build(episodes=(forged, *episodes[1:]))


def test_candidate_rejects_fabricated_retrieval_trials() -> None:
    _, _, _, _, calibration = _candidate_inputs()
    false_trial = calibration.trials[0].model_copy(update={"retrieved_card_ids": ()})
    false_calibration = calibrate_retrieval(
        episode_ids=calibration.episode_ids,
        trials=(false_trial,),
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        _build(retrieval_calibration=false_calibration)


def test_candidate_rejects_contaminated_batch() -> None:
    _, _, _, operation, _ = _candidate_inputs()
    leak = operation.model_copy(update={"instruction": "Use the correct answer."})
    with pytest.raises(ValueError, match="leakage"):
        _build(operations=(leak,))


def test_candidate_rejects_contradictory_cards() -> None:
    _, _, episodes, operation, _ = _candidate_inputs()
    conflict = operation.model_copy(
        update={
            "card_id": "card:conflict",
            "instruction": "Never inspect the chart region before computing.",
        }
    )
    calibration = calibrate_retrieval(
        episode_ids=(episodes[0].episode_id,),
        trials=(
            RetrievalTrial(
                query_id=episodes[0].task.task_id,
                episode_id=episodes[0].episode_id,
                top_k=2,
                minimum_score=0,
                token_budget=400,
                relevant_card_ids=(operation.card_id,),
                retrieved_card_ids=(operation.card_id, conflict.card_id),
            ),
        ),
    )
    with pytest.raises(ValueError, match="contradictory"):
        _build(operations=(operation, conflict), retrieval_calibration=calibration)

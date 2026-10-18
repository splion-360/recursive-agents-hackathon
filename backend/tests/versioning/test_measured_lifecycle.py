import hashlib

from memory_distillation.benchmark import (
    ExperimentManifest,
    ManifestEntry,
    Score,
    TaskAsset,
    TaskInput,
)
from memory_distillation.inference import DeterministicInference
from memory_distillation.operational_memory import InMemoryOperationalMemory, ReviewedEpisode
from memory_distillation.review_policy import CalibrationSplit, freeze_minimal_kappa
from memory_distillation.sandbox import DeterministicSandbox
from memory_distillation.task_execution import (
    DeterministicReviewer,
    FinalAnswerAction,
    ReviewResult,
    TaskExecutor,
)
from memory_distillation.telemetry import InMemoryEventSink
from memory_distillation.versioning import CardOperation, VersionRegistry
from memory_distillation.versioning.command import _select_resume_collection
from memory_distillation.versioning.measured import (
    ConsolidationBatch,
    run_measured_lifecycle,
)


class FixtureSource:
    def __init__(self, collection: ExperimentManifest, promotion: ExperimentManifest) -> None:
        self.collection = _tasks(collection)
        self.promotion = _tasks(promotion)

    def load_collection(self) -> tuple[TaskInput, ...]:
        return self.collection

    def load_promotion(self) -> tuple[TaskInput, ...]:
        return self.promotion

    def load_asset(self, task: TaskInput) -> TaskAsset:
        return TaskAsset(
            task_id=task.task_id,
            media_type="image/png",
            image_sha256=task.image_sha256,
            content=b"fixture",
        )


class FixtureMemory(VersionRegistry, InMemoryOperationalMemory):
    def __init__(self) -> None:
        VersionRegistry.__init__(self)
        InMemoryOperationalMemory.__init__(self)

    def initialize(self) -> None:
        return None


class FixtureConsolidator:
    def consolidate(
        self,
        *,
        episodes: tuple[ReviewedEpisode, ...],
        parent_view: object,
        seed: int,
    ) -> ConsolidationBatch:
        del parent_view, seed
        return ConsolidationBatch(
            operations=(
                CardOperation(
                    kind="CREATE",
                    card_id="card:visual-check",
                    trigger="visual comparison procedure",
                    instruction="Inspect all relevant marks before calculating.",
                    evidence_episode_ids=tuple(item.episode_id for item in episodes),
                ),
            ),
            model_id="fixture-consolidator",
            request_id="fixture-request",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


class AlwaysCorrect:
    def score(self, task_id: str, prediction: str) -> Score:
        del task_id, prediction
        return Score(correct=True, method="text_exact")


def test_one_measured_pass_collects_consolidates_and_decides_promotion() -> None:
    collection = _manifest("collection", "train", 40)
    promotion = _manifest("promotion", "val", 30)
    action = FinalAnswerAction(answer="fixture")
    reviewer = DeterministicReviewer(
        results=tuple(
            ReviewResult(
                decision="pass",
                accepted_action=action,
                private_reason="fixture",
                model_id="fixture-reviewer",
                request_id=f"review-{index}",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )
            for index in range(40)
        )
    )
    memory = FixtureMemory()
    progress: list[str] = []
    kappa = freeze_minimal_kappa(
        CalibrationSplit(
            manifest_hash="m" * 64,
            discovery_task_ids=tuple(f"d-{index}" for index in range(12)),
            regression_task_ids=tuple(f"r-{index}" for index in range(12)),
        )
    )
    report = run_measured_lifecycle(
        run_id="fixture-pass",
        code_revision="fixture-revision",
        source=FixtureSource(collection, promotion),
        collection_manifest=collection,
        promotion_manifest=promotion,
        executor=TaskExecutor(
            inference=DeterministicInference(actions=(action,), model_id="fixture-student"),
            reviewer=reviewer,
            sandbox=DeterministicSandbox(results={}),
            verifier=AlwaysCorrect(),
            event_sink=InMemoryEventSink(),
        ),
        consolidator=FixtureConsolidator(),
        memory=memory,
        kappa=kappa,
        progress=progress.append,
    )

    assert len(report.collection_episode_ids) == 40
    assert len(report.eligible_episode_ids) == 40
    assert len(report.promotion_outcomes) == 30
    assert report.decision.decision == "REJECT"
    assert memory.accepted_version_id == "accepted-v000"
    assert [message[:5] for message in progress] == [
        "[1/4]",
        "[1/4]",
        "[2/4]",
        "[2/4]",
        "[3/4]",
        "[3/4]",
        "[4/4]",
        "[4/4]",
    ]


def test_measured_pass_resumes_complete_collection_without_reviewer_calls() -> None:
    collection = _manifest("collection", "train", 40)
    promotion = _manifest("promotion", "val", 30)
    action = FinalAnswerAction(answer="fixture")
    source = FixtureSource(collection, promotion)
    kappa = freeze_minimal_kappa(
        CalibrationSplit(
            manifest_hash="m" * 64,
            discovery_task_ids=tuple(f"d-{index}" for index in range(12)),
            regression_task_ids=tuple(f"r-{index}" for index in range(12)),
        )
    )
    collected_memory = FixtureMemory()
    run_measured_lifecycle(
        run_id="resumable-pass",
        code_revision="fixture-revision",
        source=source,
        collection_manifest=collection,
        promotion_manifest=promotion,
        executor=TaskExecutor(
            inference=DeterministicInference(actions=(action,), model_id="fixture-student"),
            reviewer=DeterministicReviewer(
                results=tuple(
                    ReviewResult(
                        decision="pass",
                        accepted_action=action,
                        private_reason="fixture",
                        model_id="fixture-reviewer",
                        request_id=f"review-{index}",
                        input_tokens=1,
                        output_tokens=1,
                        latency_ms=1,
                    )
                    for index in range(40)
                )
            ),
            sandbox=DeterministicSandbox(results={}),
            verifier=AlwaysCorrect(),
            event_sink=InMemoryEventSink(),
        ),
        consolidator=FixtureConsolidator(),
        memory=collected_memory,
        kappa=kappa,
    )
    resumed_memory = FixtureMemory()
    for episode in collected_memory.snapshot_collection_episodes():
        resumed_memory.append_episode(episode)
    selected_run_id, selected_episodes = _select_resume_collection(
        episodes=resumed_memory.snapshot_collection_episodes(),
        requested_run_id=None,
        collection_manifest=collection,
        kappa_hash=kappa.content_hash,
    )

    resumed = run_measured_lifecycle(
        run_id=selected_run_id,
        code_revision="post-failure-fix",
        source=source,
        collection_manifest=collection,
        promotion_manifest=promotion,
        executor=TaskExecutor(
            inference=DeterministicInference(actions=(action,), model_id="fixture-student"),
            reviewer=None,
            sandbox=DeterministicSandbox(results={}),
            verifier=AlwaysCorrect(),
            event_sink=InMemoryEventSink(),
        ),
        consolidator=FixtureConsolidator(),
        memory=resumed_memory,
        kappa=kappa,
        resume_collection=selected_episodes,
    )

    assert resumed.run_id == "resumable-pass"
    assert resumed.resumed_code_revision == "post-failure-fix"
    assert len(resumed.collection_episode_ids) == 40
    assert len(resumed.promotion_outcomes) == 30


def _manifest(name: str, split: str, size: int) -> ExperimentManifest:
    return ExperimentManifest.create(
        name=name,  # type: ignore[arg-type]
        seed=17,
        dataset_revision="fixture",
        source_sha256="f" * 64,
        entries=tuple(
            ManifestEntry(
                task_id=f"{name}-{index:02d}",
                image_asset_id=f"{name}-{index:02d}.png",
                image_sha256=f"{index + 1:064x}",
                question_sha256=hashlib.sha256(_question(name, index).encode()).hexdigest(),
                source_split=split,  # type: ignore[arg-type]
                question_form="wh",
                operation_type="lookup",
            )
            for index in range(size)
        ),
    )


def _tasks(manifest: ExperimentManifest) -> tuple[TaskInput, ...]:
    return tuple(
        TaskInput(
            task_id=entry.task_id,
            question=_question(manifest.name, index),
            image_asset_id=entry.image_asset_id,
            image_sha256=entry.image_sha256,
            source_split=entry.source_split,
        )
        for index, entry in enumerate(manifest.entries)
    )


def _question(name: str, index: int) -> str:
    return f"Which category is shown in {name} example {index}?"

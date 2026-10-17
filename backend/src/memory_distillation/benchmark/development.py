"""Development-only access to calibration, collection, and promotion tasks."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from memory_distillation.benchmark.chartqa import (
    Score,
    ScoringReference,
    TaskInput,
    score_chartqa,
)
from memory_distillation.benchmark.manifests import (
    ChartQARecord,
    ManifestEntry,
    ManifestName,
    load_chartqa_records,
    load_manifest,
    validate_manifest_entry,
)


class CalibrationTasks(BaseModel):
    """Disjoint discovery and regression tasks for H-star and Kappa."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery: tuple[TaskInput, ...]
    regression: tuple[TaskInput, ...]


class DevelopmentBenchmark:
    """Resolve answer-free development tasks while retaining labels privately."""

    def __init__(self, *, dataset_root: Path, manifest_dir: Path) -> None:
        self._manifest_dir = manifest_dir
        self._records = {
            record.task_id: record
            for record in load_chartqa_records(dataset_root, splits=("train", "val"))
        }

    def load_calibration(self) -> CalibrationTasks:
        """Load the fixed discovery and regression calibration subsets."""

        return CalibrationTasks(
            discovery=self._load_tasks("calibration_discovery"),
            regression=self._load_tasks("calibration_regression"),
        )

    def load_collection(self) -> tuple[TaskInput, ...]:
        """Load tasks eligible to produce Reviewed Episodes."""

        return self._load_tasks("collection")

    def load_promotion(self) -> tuple[TaskInput, ...]:
        """Load validation-only promotion tasks."""

        return self._load_tasks("promotion")

    def score(self, task_id: str, prediction: str) -> Score:
        """Score a development prediction without exposing its gold answer."""

        record = self._record(task_id)
        return score_chartqa(
            prediction,
            ScoringReference(task_id=record.task_id, answer=record.answer),
        )

    def _load_tasks(self, name: ManifestName) -> tuple[TaskInput, ...]:
        manifest = load_manifest(self._manifest_dir / f"{name}.json")
        if manifest.name != name:
            raise ValueError(f"expected {name} manifest, found {manifest.name}")
        return tuple(self._task(entry) for entry in manifest.entries)

    def _task(self, entry: ManifestEntry) -> TaskInput:
        record = self._record(entry.task_id)
        validate_manifest_entry(entry, record)
        return TaskInput(
            task_id=record.task_id,
            question=record.question,
            image_asset_id=record.image_asset_id,
            image_sha256=record.image_sha256,
            source_split=record.source_split,
        )

    def _record(self, task_id: str) -> ChartQARecord:
        try:
            return self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown development ChartQA task: {task_id}") from error

"""Development-only access to calibration, collection, and promotion tasks."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from memory_distillation.benchmark.chartqa import (
    Score,
    ScoringReference,
    TaskAsset,
    TaskInput,
    score_chartqa,
)
from memory_distillation.benchmark.manifests import (
    ChartQARecord,
    ManifestName,
    load_chartqa_records,
    load_manifest,
    load_task_asset,
    validate_manifest_entry,
    validate_manifest_source,
)


class CalibrationTasks(BaseModel):
    """Disjoint discovery and regression tasks for H-star and Kappa."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery: tuple[TaskInput, ...]
    regression: tuple[TaskInput, ...]


class DevelopmentBenchmark:
    """Resolve answer-free development tasks while retaining labels privately."""

    def __init__(self, *, dataset_root: Path, manifest_dir: Path) -> None:
        self._dataset_root = dataset_root
        self._manifest_dir = manifest_dir
        self._records = {
            record.task_id: record
            for record in load_chartqa_records(dataset_root, splits=("train", "val"))
        }
        self._visible_records: dict[str, ChartQARecord] = {}

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

        record = self._visible_record(task_id)
        return score_chartqa(
            prediction,
            ScoringReference(task_id=record.task_id, answer=record.answer),
        )

    def load_asset(self, task: TaskInput) -> TaskAsset:
        """Expose the current task's bytes without exposing any dataset path."""

        record = self._visible_record(task.task_id)
        if task != self._task(record):
            raise ValueError(f"task input does not match manifest source: {task.task_id}")
        return load_task_asset(self._dataset_root, record)

    def _load_tasks(self, name: ManifestName) -> tuple[TaskInput, ...]:
        manifest = load_manifest(self._manifest_dir / f"{name}.json")
        if manifest.name != name:
            raise ValueError(f"expected {name} manifest, found {manifest.name}")
        source_records = [
            record
            for record in self._records.values()
            if record.source_split == manifest.entries[0].source_split
        ]
        validate_manifest_source(manifest, source_records)

        tasks: list[TaskInput] = []
        for entry in manifest.entries:
            record = self._record(entry.task_id)
            validate_manifest_entry(entry, record)
            self._visible_records[record.task_id] = record
            tasks.append(self._task(record))
        return tuple(tasks)

    def _task(self, record: ChartQARecord) -> TaskInput:
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

    def _visible_record(self, task_id: str) -> ChartQARecord:
        try:
            return self._visible_records[task_id]
        except KeyError as error:
            raise KeyError(f"ChartQA task is not loaded for development: {task_id}") from error

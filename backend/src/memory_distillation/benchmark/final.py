"""Held-out ChartQA access for the final evaluation composition root only."""

from pathlib import Path

from memory_distillation.benchmark.chartqa import (
    Score,
    ScoringReference,
    TaskAsset,
    TaskInput,
    score_chartqa,
)
from memory_distillation.benchmark.manifests import (
    ChartQARecord,
    load_chartqa_records,
    load_manifest,
    load_task_asset,
    validate_manifest_entry,
    validate_manifest_source,
)


class FinalBenchmark:
    """Resolve and score only the immutable final manifest."""

    def __init__(self, *, dataset_root: Path, manifest_dir: Path) -> None:
        self._dataset_root = dataset_root
        self._manifest_path = manifest_dir / "final.json"
        self._records: dict[str, ChartQARecord] | None = None
        self._tasks: tuple[TaskInput, ...] | None = None

    def load_final(self) -> tuple[TaskInput, ...]:
        """Load answer-free held-out tasks after checking source integrity."""

        if self._tasks is not None:
            return self._tasks

        manifest = load_manifest(self._manifest_path)
        if manifest.name != "final":
            raise ValueError(f"expected final manifest, found {manifest.name}")
        source_records = load_chartqa_records(self._dataset_root, splits=("test",))
        validate_manifest_source(manifest, source_records)
        records_by_id = {record.task_id: record for record in source_records}

        tasks: list[TaskInput] = []
        selected_records: dict[str, ChartQARecord] = {}
        for entry in manifest.entries:
            try:
                record = records_by_id[entry.task_id]
            except KeyError as error:
                raise KeyError(f"unknown final ChartQA task: {entry.task_id}") from error
            validate_manifest_entry(entry, record)
            selected_records[record.task_id] = record
            tasks.append(
                TaskInput(
                    task_id=record.task_id,
                    question=record.question,
                    image_asset_id=record.image_asset_id,
                    image_sha256=record.image_sha256,
                    source_split=record.source_split,
                )
            )
        self._records = selected_records
        self._tasks = tuple(tasks)
        return self._tasks

    def score(self, task_id: str, prediction: str) -> Score:
        """Score a held-out prediction without exposing its gold answer."""

        record = self._record(task_id)
        return score_chartqa(
            prediction,
            ScoringReference(task_id=record.task_id, answer=record.answer),
        )

    def load_asset(self, task: TaskInput) -> TaskAsset:
        """Expose one opened final task's bytes without exposing the dataset root."""

        record = self._record(task.task_id)
        expected = TaskInput(
            task_id=record.task_id,
            question=record.question,
            image_asset_id=record.image_asset_id,
            image_sha256=record.image_sha256,
            source_split=record.source_split,
        )
        if task != expected:
            raise ValueError(f"task input does not match manifest source: {task.task_id}")
        return load_task_asset(self._dataset_root, record)

    def _record(self, task_id: str) -> ChartQARecord:
        if self._records is None:
            raise RuntimeError("final benchmark has not been opened with load_final()")
        try:
            return self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown final ChartQA task: {task_id}") from error

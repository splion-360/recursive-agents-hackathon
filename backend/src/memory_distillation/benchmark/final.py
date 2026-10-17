"""Held-out ChartQA access for the final evaluation composition root only."""

from pathlib import Path

from memory_distillation.benchmark.chartqa import (
    Score,
    ScoringReference,
    TaskInput,
    score_chartqa,
)
from memory_distillation.benchmark.manifests import (
    ChartQARecord,
    load_chartqa_records,
    load_manifest,
    validate_manifest_entry,
)


class FinalBenchmark:
    """Resolve and score only the immutable final manifest."""

    def __init__(self, *, dataset_root: Path, manifest_dir: Path) -> None:
        self._manifest_path = manifest_dir / "final.json"
        self._records = {
            record.task_id: record
            for record in load_chartqa_records(dataset_root, splits=("test",))
        }

    def load_final(self) -> tuple[TaskInput, ...]:
        """Load answer-free held-out tasks after checking source integrity."""

        manifest = load_manifest(self._manifest_path)
        if manifest.name != "final":
            raise ValueError(f"expected final manifest, found {manifest.name}")

        tasks: list[TaskInput] = []
        for entry in manifest.entries:
            record = self._record(entry.task_id)
            validate_manifest_entry(entry, record)
            tasks.append(
                TaskInput(
                    task_id=record.task_id,
                    question=record.question,
                    image_asset_id=record.image_asset_id,
                    image_sha256=record.image_sha256,
                    source_split=record.source_split,
                )
            )
        return tuple(tasks)

    def score(self, task_id: str, prediction: str) -> Score:
        """Score a held-out prediction without exposing its gold answer."""

        record = self._record(task_id)
        return score_chartqa(
            prediction,
            ScoringReference(task_id=record.task_id, answer=record.answer),
        )

    def _record(self, task_id: str) -> ChartQARecord:
        try:
            return self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown final ChartQA task: {task_id}") from error

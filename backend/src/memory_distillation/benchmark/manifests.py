"""Deterministic, answer-free ChartQA experiment manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceSplit = Literal["train", "val", "test"]
ManifestName = Literal[
    "calibration_discovery",
    "calibration_regression",
    "collection",
    "promotion",
    "final",
]
QuestionStratum = Literal["boolean", "count", "arithmetic", "lookup"]

CHARTQA_REPOSITORY: Literal["https://github.com/vis-nlp/ChartQA"] = (
    "https://github.com/vis-nlp/ChartQA"
)
CHARTQA_REVIEWED_REVISION = "044eabfc306abfe9340c5741f0093aefc5973d06"
MANIFEST_REQUIREMENTS: dict[ManifestName, tuple[SourceSplit, int]] = {
    "calibration_discovery": ("train", 8),
    "calibration_regression": ("train", 8),
    "collection": ("train", 40),
    "promotion": ("val", 30),
    "final": ("test", 60),
}


class ChartQARecord(BaseModel):
    """Internal source row kept behind the benchmark boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str
    image_asset_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_split: SourceSplit


class ManifestEntry(BaseModel):
    """Answer-free identity and integrity data for one selected task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    image_asset_id: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_split: SourceSplit
    stratum: QuestionStratum


class ExperimentManifest(BaseModel):
    """Immutable, self-verifying task membership for one experiment stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["chartqa-manifest.v1"] = "chartqa-manifest.v1"
    dataset_name: Literal["ChartQA"] = "ChartQA"
    dataset_repository: Literal["https://github.com/vis-nlp/ChartQA"] = CHARTQA_REPOSITORY
    name: ManifestName
    seed: int
    dataset_revision: str = Field(min_length=1)
    entries: tuple[ManifestEntry, ...]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        name: ManifestName,
        seed: int,
        dataset_revision: str,
        entries: tuple[ManifestEntry, ...],
    ) -> Self:
        """Create a manifest whose digest covers every behavior-bearing field."""

        content_sha256 = _manifest_digest(
            name=name,
            seed=seed,
            dataset_revision=dataset_revision,
            entries=entries,
        )
        return cls(
            name=name,
            seed=seed,
            dataset_revision=dataset_revision,
            entries=entries,
            content_sha256=content_sha256,
        )

    @model_validator(mode="after")
    def verify_content_digest(self) -> Self:
        """Reject edited membership or provenance instead of trusting stale hashes."""

        required_split, required_size = MANIFEST_REQUIREMENTS[self.name]
        if len(self.entries) != required_size:
            raise ValueError(f"{self.name} manifest requires exactly {required_size} entries")
        if any(entry.source_split != required_split for entry in self.entries):
            raise ValueError(f"{self.name} manifest requires only the {required_split} split")
        if len({entry.task_id for entry in self.entries}) != len(self.entries):
            raise ValueError(f"{self.name} manifest contains duplicate tasks")
        if len({entry.image_asset_id for entry in self.entries}) != len(self.entries):
            raise ValueError(f"{self.name} manifest contains duplicate images")

        expected = _manifest_digest(
            name=self.name,
            seed=self.seed,
            dataset_revision=self.dataset_revision,
            entries=self.entries,
        )
        if self.content_sha256 != expected:
            raise ValueError("manifest content_sha256 does not match its contents")
        return self


def build_default_manifests(
    records: list[ChartQARecord],
    *,
    dataset_revision: str,
    seed: int,
) -> dict[ManifestName, ExperimentManifest]:
    """Build the fixed 8/8/40/30/60 split with task and image isolation."""

    _require_unique_task_ids(records)
    pools = {
        split: _one_record_per_image(record for record in records if record.source_split == split)
        for split in ("train", "val", "test")
    }

    selected: dict[ManifestName, list[ChartQARecord]] = {}
    train_pool = pools["train"]
    development_specs: tuple[tuple[ManifestName, int], ...] = (
        ("calibration_discovery", 8),
        ("calibration_regression", 8),
        ("collection", 40),
    )
    for name, size in development_specs:
        chosen = _select_stratified(train_pool, size=size, seed=seed, namespace=name)
        selected[name] = chosen
        chosen_ids = {record.task_id for record in chosen}
        train_pool = [record for record in train_pool if record.task_id not in chosen_ids]

    selected["promotion"] = _select_stratified(
        pools["val"], size=30, seed=seed, namespace="promotion"
    )
    selected["final"] = _select_stratified(
        pools["test"], size=60, seed=seed, namespace="final"
    )

    _assert_disjoint(selected)
    return {
        name: ExperimentManifest.create(
            name=name,
            seed=seed,
            dataset_revision=dataset_revision,
            entries=tuple(_to_manifest_entry(record) for record in chosen),
        )
        for name, chosen in selected.items()
    }


def generate_default_manifest_files(
    *,
    dataset_root: Path,
    output_dir: Path,
    dataset_revision: str,
    seed: int,
) -> dict[ManifestName, Path]:
    """Read a local ChartQA checkout and write only answer-free manifest metadata."""

    records = load_chartqa_records(dataset_root)
    manifests = build_default_manifests(
        records,
        dataset_revision=dataset_revision,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[ManifestName, Path] = {}
    for name, manifest in manifests.items():
        path = output_dir / f"{name}.json"
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[name] = path
    return written


def load_chartqa_records(
    dataset_root: Path,
    *,
    splits: tuple[SourceSplit, ...] = ("train", "val", "test"),
) -> list[ChartQARecord]:
    """Load human-authored ChartQA rows and verify their referenced image bytes."""

    records: list[ChartQARecord] = []
    for split in splits:
        annotation_path = dataset_root / split / f"{split}_human.json"
        raw_rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError(f"{annotation_path} must contain a JSON array")
        image_hashes: dict[str, str] = {}
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                raise ValueError(f"{annotation_path} row {index} must be an object")
            image_name = _required_string(raw_row, "imgname", annotation_path, index)
            question = _required_string(raw_row, "query", annotation_path, index)
            answer = _required_string(raw_row, "label", annotation_path, index, allow_empty=True)
            if Path(image_name).name != image_name:
                raise ValueError(f"{annotation_path} row {index} has an invalid imgname")
            image_path = dataset_root / split / "png" / image_name
            image_sha256 = image_hashes.get(image_name)
            if image_sha256 is None:
                image_sha256 = sha256(image_path.read_bytes()).hexdigest()
                image_hashes[image_name] = image_sha256
            task_digest = sha256(
                f"{split}\0human\0{image_name}\0{question}".encode()
            ).hexdigest()
            records.append(
                ChartQARecord(
                    task_id=f"chartqa:{split}:human:{task_digest[:24]}",
                    question=question,
                    answer=answer,
                    image_asset_id=image_name,
                    image_sha256=image_sha256,
                    source_split=split,
                )
            )
    return records


def load_manifest(path: Path) -> ExperimentManifest:
    """Load a manifest and verify its canonical digest."""

    return ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_manifest_entry(entry: ManifestEntry, record: ChartQARecord) -> None:
    """Verify that answer-free manifest integrity data matches local source bytes."""

    if entry.source_split != record.source_split:
        raise ValueError(f"manifest source split mismatch for {entry.task_id}")
    if entry.image_asset_id != record.image_asset_id:
        raise ValueError(f"manifest image identity mismatch for {entry.task_id}")
    if entry.image_sha256 != record.image_sha256:
        raise ValueError(f"manifest image hash mismatch for {entry.task_id}")
    if entry.question_sha256 != sha256(record.question.encode()).hexdigest():
        raise ValueError(f"manifest question hash mismatch for {entry.task_id}")


def _one_record_per_image(records: Iterable[ChartQARecord]) -> list[ChartQARecord]:
    by_image: dict[str, list[ChartQARecord]] = defaultdict(list)
    for record in records:
        by_image[record.image_asset_id].append(record)
    return [
        min(candidates, key=lambda record: record.task_id)
        for _, candidates in sorted(by_image.items())
    ]


def _select_stratified(
    records: list[ChartQARecord],
    *,
    size: int,
    seed: int,
    namespace: str,
) -> list[ChartQARecord]:
    if len(records) < size:
        raise ValueError(f"{namespace} requires {size} unique images; found {len(records)}")

    strata: dict[QuestionStratum, list[ChartQARecord]] = defaultdict(list)
    for record in records:
        strata[_question_stratum(record.question)].append(record)
    for candidates in strata.values():
        candidates.sort(key=lambda record: _selection_key(seed, namespace, record.task_id))

    selected: list[ChartQARecord] = []
    ordered_strata = sorted(strata)
    while len(selected) < size:
        progressed = False
        for stratum in ordered_strata:
            if strata[stratum] and len(selected) < size:
                selected.append(strata[stratum].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"{namespace} could not select {size} stratified records")
    return selected


def _question_stratum(question: str) -> QuestionStratum:
    normalized = " ".join(question.casefold().split())
    if normalized.startswith("how many"):
        return "count"
    if normalized.startswith(("is ", "are ", "do ", "does ", "did ", "was ", "were ")):
        return "boolean"
    arithmetic_terms = (
        "sum",
        "difference",
        "average",
        "ratio",
        "median",
        "product",
        " add ",
        "subtract",
        "deduct",
        "multiply",
    )
    if any(term in f" {normalized} " for term in arithmetic_terms):
        return "arithmetic"
    return "lookup"


def _to_manifest_entry(record: ChartQARecord) -> ManifestEntry:
    return ManifestEntry(
        task_id=record.task_id,
        image_asset_id=record.image_asset_id,
        image_sha256=record.image_sha256,
        question_sha256=sha256(record.question.encode()).hexdigest(),
        source_split=record.source_split,
        stratum=_question_stratum(record.question),
    )


def _selection_key(seed: int, namespace: str, task_id: str) -> str:
    return sha256(f"{seed}:{namespace}:{task_id}".encode()).hexdigest()


def _require_unique_task_ids(records: list[ChartQARecord]) -> None:
    task_ids = [record.task_id for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("ChartQA source contains duplicate task IDs")


def _assert_disjoint(selected: dict[ManifestName, list[ChartQARecord]]) -> None:
    records = [record for stage in selected.values() for record in stage]
    if len({record.task_id for record in records}) != len(records):
        raise ValueError("experiment manifests overlap by task")
    if len({record.image_asset_id for record in records}) != len(records):
        raise ValueError("experiment manifests overlap by image")


def _manifest_digest(
    *,
    name: ManifestName,
    seed: int,
    dataset_revision: str,
    entries: tuple[ManifestEntry, ...],
) -> str:
    payload = {
        "schema_version": "chartqa-manifest.v1",
        "dataset_name": "ChartQA",
        "dataset_repository": CHARTQA_REPOSITORY,
        "name": name,
        "seed": seed,
        "dataset_revision": dataset_revision,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode()).hexdigest()


def _required_string(
    row: dict[object, object],
    field: str,
    path: Path,
    index: int,
    *,
    allow_empty: bool = False,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{path} row {index} requires string field {field!r}")
    return value

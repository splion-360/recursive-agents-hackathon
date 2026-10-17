"""Behavior tests for deterministic ChartQA experiment manifests."""

import json
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from memory_distillation.benchmark.development import DevelopmentBenchmark
from memory_distillation.benchmark.final import FinalBenchmark
from memory_distillation.benchmark.manifests import (
    ChartQARecord,
    ExperimentManifest,
    build_default_manifests,
    generate_default_manifest_files,
    load_chartqa_records,
)

EXPECTED_SIZES = {
    "calibration_discovery": 12,
    "calibration_regression": 12,
    "collection": 40,
    "promotion": 30,
    "final": 60,
}


def test_default_manifests_are_deterministic_disjoint_and_answer_free() -> None:
    """The frozen experiment split is reproducible and cannot leak labels."""

    records = [
        *_records("train", 70),
        *_records("val", 40),
        *_records("test", 70),
    ]

    manifests = build_default_manifests(
        records,
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )
    reversed_manifests = build_default_manifests(
        list(reversed(records)),
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )

    assert {name: len(manifest.entries) for name, manifest in manifests.items()} == EXPECTED_SIZES
    assert {
        name: manifest.content_sha256 for name, manifest in manifests.items()
    } == {
        name: manifest.content_sha256 for name, manifest in reversed_manifests.items()
    }

    entries = [entry for manifest in manifests.values() for entry in manifest.entries]
    assert len({entry.task_id for entry in entries}) == sum(EXPECTED_SIZES.values())
    assert len({entry.image_asset_id for entry in entries}) == sum(EXPECTED_SIZES.values())
    assert all(
        len({entry.operation_type for entry in manifest.entries}) == 4
        for manifest in manifests.values()
    )
    assert all(
        len({entry.question_form for entry in manifest.entries}) >= 3
        for manifest in manifests.values()
    )

    serialized = "".join(manifest.model_dump_json() for manifest in manifests.values())
    assert "answer" not in serialized
    assert "question\"" not in serialized
    assert "fixture-answer" not in serialized

    tampered = manifests["collection"].model_dump(mode="json")
    tampered["entries"][0]["image_asset_id"] = "tampered.png"
    with pytest.raises(ValidationError, match="content_sha256"):
        ExperimentManifest.model_validate(tampered)


def test_development_benchmark_loads_answer_free_tasks_and_scores_privately(
    tmp_path: Path,
) -> None:
    """Development access resolves manifest IDs without returning gold labels."""

    dataset_root = tmp_path / "ChartQA Dataset"
    answers = _write_fixture_dataset(dataset_root)
    manifest_dir = tmp_path / "manifests"
    written = generate_default_manifest_files(
        dataset_root=dataset_root,
        output_dir=manifest_dir,
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )

    assert set(written) == set(EXPECTED_SIZES)
    assert all(path.is_file() for path in written.values())

    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    calibration = benchmark.load_calibration()
    collection = benchmark.load_collection()
    promotion = benchmark.load_promotion()

    assert len(calibration.discovery) == 12
    assert len(calibration.regression) == 12
    assert len(collection) == 40
    assert len(promotion) == 30
    task = collection[0]
    assert "answer" not in task.model_dump_json()
    assert benchmark.score(task.task_id, answers[task.question]).correct is True
    asset = benchmark.load_asset(task)
    assert {field.name for field in fields(asset)} == {
        "task_id",
        "media_type",
        "image_sha256",
        "content",
    }
    assert asset.task_id == task.task_id
    assert asset.content.startswith(b"synthetic-train-")


def test_final_benchmark_is_separate_and_keeps_answers_private(tmp_path: Path) -> None:
    """The held-out loader resolves only the final manifest and never returns labels."""

    dataset_root = tmp_path / "ChartQA Dataset"
    answers = _write_fixture_dataset(dataset_root)
    manifest_dir = tmp_path / "manifests"
    generate_default_manifest_files(
        dataset_root=dataset_root,
        output_dir=manifest_dir,
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )

    benchmark = FinalBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    with pytest.raises(RuntimeError, match="has not been opened"):
        benchmark.score("chartqa:test:not-open", "answer")
    tasks = benchmark.load_final()

    assert len(tasks) == 60
    assert {task.source_split for task in tasks} == {"test"}
    assert all("answer" not in task.model_dump_json() for task in tasks)
    assert benchmark.score(tasks[0].task_id, answers[tasks[0].question]).correct is True
    assert benchmark.load_asset(tasks[0]).task_id == tasks[0].task_id
    excluded = next(
        record
        for record in load_chartqa_records(dataset_root, splits=("test",))
        if record.task_id not in {task.task_id for task in tasks}
    )
    with pytest.raises(KeyError, match="unknown final"):
        benchmark.score(excluded.task_id, excluded.answer)


def test_final_benchmark_does_not_touch_data_until_opened(tmp_path: Path) -> None:
    """Constructing final access cannot open held-out questions or labels."""

    FinalBenchmark(dataset_root=tmp_path / "missing", manifest_dir=tmp_path / "missing")


def test_loader_rejects_manifest_that_disagrees_with_dataset(tmp_path: Path) -> None:
    """A self-consistent but wrong manifest cannot silently select changed data."""

    dataset_root = tmp_path / "ChartQA Dataset"
    _write_fixture_dataset(dataset_root)
    manifest_dir = tmp_path / "manifests"
    generate_default_manifest_files(
        dataset_root=dataset_root,
        output_dir=manifest_dir,
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )
    manifest_path = manifest_dir / "collection.json"
    manifest = ExperimentManifest.model_validate_json(manifest_path.read_text())
    first = manifest.entries[0].model_copy(update={"question_sha256": "0" * 64})
    altered = ExperimentManifest.create(
        name=manifest.name,
        seed=manifest.seed,
        dataset_revision=manifest.dataset_revision,
        source_sha256=manifest.source_sha256,
        entries=(first, *manifest.entries[1:]),
    )
    manifest_path.write_text(altered.model_dump_json(), encoding="utf-8")

    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    with pytest.raises(ValueError, match="question hash"):
        benchmark.load_collection()


def test_loader_rejects_changed_private_scoring_reference(tmp_path: Path) -> None:
    """A label change invalidates source provenance without putting labels in manifests."""

    dataset_root = tmp_path / "ChartQA Dataset"
    _write_fixture_dataset(dataset_root)
    manifest_dir = tmp_path / "manifests"
    generate_default_manifest_files(
        dataset_root=dataset_root,
        output_dir=manifest_dir,
        dataset_revision="chartqa@fixture-v1",
        seed=20261016,
    )
    annotation_path = dataset_root / "train" / "train_human.json"
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows[0]["label"] = "changed-reference"
    annotation_path.write_text(json.dumps(rows), encoding="utf-8")

    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    with pytest.raises(ValueError, match="source hash"):
        benchmark.load_collection()


def test_manifest_generation_rejects_duplicate_image_content() -> None:
    """Different filenames cannot hide the same chart across experiment stages."""

    records = [*_records("train", 70), *_records("val", 40), *_records("test", 70)]
    records = [
        record.model_copy(update={"image_sha256": records[0].image_sha256})
        for record in records
    ]

    with pytest.raises(ValueError, match="overlap by image"):
        build_default_manifests(
            records,
            dataset_revision="chartqa@fixture-v1",
            seed=20261016,
        )


def test_exact_duplicate_source_rows_are_collapsed(tmp_path: Path) -> None:
    """Known exact duplicates in upstream ChartQA do not create duplicate tasks."""

    dataset_root = tmp_path / "ChartQA Dataset"
    _write_fixture_dataset(dataset_root)
    annotation_path = dataset_root / "train" / "train_human.json"
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows.append(dict(rows[0]))
    annotation_path.write_text(json.dumps(rows), encoding="utf-8")

    assert len(load_chartqa_records(dataset_root, splits=("train",))) == 70


def _records(source_split: Literal["train", "val", "test"], count: int) -> list[ChartQARecord]:
    question_templates = (
        "Is the blue bar larger?",
        "How many bars are visible?",
        "What is the sum of the values?",
        "What year is shown?",
    )
    return [
        ChartQARecord(
            task_id=f"chartqa:{source_split}:{index:03d}",
            question=question_templates[index % len(question_templates)],
            answer=f"fixture-answer-{index}",
            image_asset_id=f"{source_split}-image-{index:03d}.png",
            image_sha256=sha256(f"{source_split}-image-{index}".encode()).hexdigest(),
            source_split=source_split,
        )
        for index in range(count)
    ]


def _write_fixture_dataset(dataset_root: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for split, count in (("train", 70), ("val", 40), ("test", 70)):
        split_dir = dataset_root / split
        image_dir = split_dir / "png"
        image_dir.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        for index in range(count):
            image_name = f"{split}-image-{index:03d}.png"
            question = _question(split, index)
            answer = f"fixture-answer-{split}-{index}"
            rows.append({"imgname": image_name, "query": question, "label": answer})
            answers[question] = answer
            (image_dir / image_name).write_bytes(f"synthetic-{split}-{index}".encode())
        (split_dir / f"{split}_human.json").write_text(json.dumps(rows), encoding="utf-8")
    return answers


def _question(split: str, index: int) -> str:
    templates = (
        "Is the blue bar larger in fixture {index}?",
        "How many bars are visible in fixture {index}?",
        "What is the sum for fixture {index}?",
        "What year is shown in fixture {index}?",
    )
    return f"{templates[index % len(templates)].format(index=index)} Source {split}."

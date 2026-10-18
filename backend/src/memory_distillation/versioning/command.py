"""Composition root for one paid Collection-to-Promotion lifecycle pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from memory_distillation.benchmark import DevelopmentBenchmark, ExperimentManifest
from memory_distillation.benchmark.manifests import load_manifest
from memory_distillation.calibration import CalibrationReport
from memory_distillation.inference import (
    CrusoeConsolidator,
    CrusoeReviewer,
    CrusoeStudentInference,
)
from memory_distillation.inference.crusoe import CrusoeModel, build_crusoe_client
from memory_distillation.operational_memory import MemMachineOperationalMemory
from memory_distillation.operational_memory.models import ReviewedEpisode
from memory_distillation.review_policy import KappaArtifact
from memory_distillation.sandbox import OpenShellSandbox
from memory_distillation.settings import Settings, get_settings
from memory_distillation.task_execution import TaskExecutor
from memory_distillation.telemetry import JsonlEventSink
from memory_distillation.versioning.measured import LifecycleReport, run_measured_lifecycle

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_ROOT = _REPOSITORY_ROOT / ".local" / "chartqa-source" / "ChartQA Dataset"
_DEFAULT_MANIFEST_DIR = _REPOSITORY_ROOT / "backend" / "experiment_manifests"
_DEFAULT_CALIBRATION_DIR = _REPOSITORY_ROOT / ".artifacts" / "calibration" / "current"
_LIFECYCLE_DIR = _REPOSITORY_ROOT / ".artifacts" / "lifecycle" / "current"


def run_live_lifecycle(
    settings: Settings,
    *,
    run_id: str | None,
    dataset_root: Path,
    manifest_dir: Path,
    calibration_dir: Path,
    seed: int,
    code_revision: str,
    resume: bool = False,
) -> LifecycleReport:
    """Compose verified adapters and execute one measured lifecycle pass."""

    _prepare_artifact_directory(resume=resume)
    report, kappa = _load_calibration(calibration_dir)
    collection_manifest = load_manifest(manifest_dir / "collection.json")
    promotion_manifest = load_manifest(manifest_dir / "promotion.json")
    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    student_model = _required(settings.crusoe_student_model, "CRUSOE_STUDENT_MODEL")
    reviewer_model = _required(settings.crusoe_reviewer_model, "CRUSOE_REVIEWER_MODEL")
    consolidator_model = _required(
        settings.crusoe_consolidator_model, "CRUSOE_CONSOLIDATOR_MODEL"
    )
    sandbox_name = _required(settings.openshell_sandbox_name, "OPENSHELL_SANDBOX_NAME")
    if settings.memmachine_api_key is None:
        raise ValueError("MEMMACHINE_API_KEY is required")
    with (
        build_crusoe_client(settings) as client,
        MemMachineOperationalMemory(
            base_url=str(settings.memmachine_base_url),
            api_key=settings.memmachine_api_key,
            project_id=settings.memmachine_project_id,
        ) as memory,
    ):
        memory.initialize()
        resume_collection: tuple[ReviewedEpisode, ...] | None = None
        if resume:
            run_id, resume_collection = _select_resume_collection(
                episodes=memory.snapshot_collection_episodes(),
                requested_run_id=run_id,
                collection_manifest=collection_manifest,
                kappa_hash=kappa.content_hash,
            )
            print(
                f"Resume verified: {run_id}; loading 40 Collection episodes from MemMachine."
            )
        elif run_id is None:
            raise ValueError("a run_id is required for a new lifecycle pass")
        executor = TaskExecutor(
            inference=CrusoeStudentInference(
                client=client,
                model_id=student_model,
                max_output_tokens=settings.crusoe_student_max_output_tokens,
            ),
            reviewer=CrusoeReviewer(
                client=client,
                model_id=reviewer_model,
                max_output_tokens=settings.crusoe_reviewer_max_output_tokens,
            ),
            sandbox=OpenShellSandbox(
                sandbox_name=sandbox_name,
                gateway=settings.openshell_gateway,
            ),
            verifier=benchmark,
            event_sink=JsonlEventSink(_LIFECYCLE_DIR / "events.jsonl"),
        )
        lifecycle = run_measured_lifecycle(
            run_id=run_id,
            code_revision=code_revision,
            source=benchmark,
            collection_manifest=collection_manifest,
            promotion_manifest=promotion_manifest,
            executor=executor,
            consolidator=CrusoeConsolidator(
                client=client,
                model_id=consolidator_model,
                max_output_tokens=settings.crusoe_consolidator_max_output_tokens,
            ),
            memory=memory,
            kappa=kappa,
            seed=seed,
            progress=print,
            resume_collection=resume_collection,
        )
        _write_json("report.json", lifecycle.model_dump(mode="json"))
        _write_json("candidate.json", lifecycle.candidate.model_dump(mode="json"))
        _write_json("promotion.json", lifecycle.decision.model_dump(mode="json"))
        _write_json(
            "cost.json",
            {
                "actual_cost_usd": str(client.spent_cost_usd),
                "calibration_run_id": report.run_id,
            },
        )
        return lifecycle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="memory-distillation lifecycle", description=__doc__)
    parser.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=_DEFAULT_MANIFEST_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=_DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a complete Collection pass from verified MemMachine records",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    calibration_report, kappa = _load_calibration(args.calibration_dir)
    collection = load_manifest(args.manifest_dir / "collection.json")
    promotion = load_manifest(args.manifest_dir / "promotion.json")
    model_ids = (
        _required(settings.crusoe_student_model, "CRUSOE_STUDENT_MODEL"),
        _required(settings.crusoe_reviewer_model, "CRUSOE_REVIEWER_MODEL"),
        _required(settings.crusoe_consolidator_model, "CRUSOE_CONSOLIDATOR_MODEL"),
        _required(settings.crusoe_hstar_model, "CRUSOE_HSTAR_MODEL"),
    )
    with build_crusoe_client(settings) as client:
        models = tuple(client.get_model(model_id) for model_id in model_ids)
    estimated_cost = _estimated_upper_cost(settings, models)
    calibration_identities = (
        calibration_report.models.student,
        calibration_report.models.reviewer,
        calibration_report.models.hstar,
    )
    live_identities = (
        _model_identity(models[0]),
        _model_identity(models[1]),
        _model_identity(models[3]),
    )
    if live_identities != calibration_identities:
        raise ValueError("live Student, Reviewer, or H-star identity differs from calibration")
    source_identity = _source_identity()
    print(
        "Lifecycle preflight\n"
        f"  Mode: {'resume after Collection' if args.resume else 'new pass'}"
        f"{f'; run={args.run_id}' if args.run_id else ''}\n"
        f"  Calibration: {calibration_report.run_id}; Kappa={kappa.content_hash[:12]}\n"
        f"  Work: Collection={len(collection.entries)} ({collection.content_sha256[:12]}), "
        f"Promotion={len(promotion.entries)} paired ({promotion.content_sha256[:12]})\n"
        f"  Models: {_model_identity(models[0])} / {_model_identity(models[1])} / "
        f"{_model_identity(models[2])}\n"
        f"  Limits: <= {settings.crusoe_max_total_tokens:,} tokens, "
        f"estimated <= ${estimated_cost}; artifacts={_LIFECYCLE_DIR}\n"
        f"  Memory: MemMachine/{settings.memmachine_project_id}; source={source_identity}"
    )
    if not args.confirm_paid_run:
        print("Dry preflight only; rerun with --confirm-paid-run to execute paid calls and writes.")
        return
    lifecycle = run_live_lifecycle(
        settings,
        run_id=(
            args.run_id
            if args.resume
            else args.run_id or datetime.now(UTC).strftime("lifecycle-%Y%m%dT%H%M%SZ")
        ),
        dataset_root=args.dataset_root,
        manifest_dir=args.manifest_dir,
        calibration_dir=args.calibration_dir,
        seed=args.seed,
        code_revision=source_identity,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "run_id": lifecycle.run_id,
                "candidate_version": lifecycle.candidate.version_id,
                "decision": lifecycle.decision.decision,
                "accepted_version": lifecycle.decision.accepted_version_after,
                "report_path": str(_LIFECYCLE_DIR / "report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_calibration(path: Path) -> tuple[CalibrationReport, KappaArtifact]:
    report_path = path / "report.json"
    kappa_path = path / "kappa.json"
    if not report_path.is_file() or not kappa_path.is_file():
        raise FileNotFoundError("completed calibration report.json and kappa.json are required")
    report = CalibrationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    kappa = KappaArtifact.model_validate_json(kappa_path.read_text(encoding="utf-8"))
    if report.kappa != kappa or report.kappa.content_hash != kappa.content_hash:
        raise ValueError("calibration report and Kappa artifact do not match")
    return report, kappa


def _prepare_artifact_directory(*, resume: bool = False) -> None:
    _LIFECYCLE_DIR.mkdir(parents=True, exist_ok=True)
    conflicts = tuple(_LIFECYCLE_DIR.iterdir())
    if conflicts and not resume:
        raise FileExistsError("lifecycle artifact directory is not empty; archive it first")
    unexpected = tuple(path.name for path in conflicts if path.name != "events.jsonl")
    if unexpected:
        raise FileExistsError(
            "resume requires an incomplete lifecycle directory containing only events.jsonl; "
            f"found {', '.join(sorted(unexpected))}"
        )


def _select_resume_collection(
    *,
    episodes: tuple[ReviewedEpisode, ...],
    requested_run_id: str | None,
    collection_manifest: ExperimentManifest,
    kappa_hash: str,
) -> tuple[str, tuple[ReviewedEpisode, ...]]:
    """Select one complete pass; full provenance is revalidated before inference."""

    task_ids = tuple(entry.task_id for entry in collection_manifest.entries)
    task_id_set = set(task_ids)
    grouped: dict[str, list[ReviewedEpisode]] = {}
    for episode in episodes:
        provenance = episode.provenance
        if (
            provenance.purpose == "collection"
            and provenance.manifest_hash == collection_manifest.content_sha256
            and provenance.kappa_hash == kappa_hash
        ):
            grouped.setdefault(provenance.lifecycle_pass_id, []).append(episode)
    candidates: dict[str, tuple[ReviewedEpisode, ...]] = {}
    for candidate_run_id, records in grouped.items():
        by_task = {record.task.task_id: record for record in records}
        if len(records) == len(task_ids) and set(by_task) == task_id_set:
            candidates[candidate_run_id] = tuple(by_task[task_id] for task_id in task_ids)
    if requested_run_id is not None:
        selected = candidates.get(requested_run_id)
        if selected is None:
            raise ValueError(
                f"MemMachine has no complete compatible Collection pass for {requested_run_id}"
            )
        return requested_run_id, selected
    if len(candidates) != 1:
        ids = ", ".join(sorted(candidates)) or "none"
        raise ValueError(
            "resume requires exactly one complete compatible Collection pass; "
            f"found {ids}. Use --run-id to disambiguate."
        )
    selected_run_id = next(iter(candidates))
    return selected_run_id, candidates[selected_run_id]


def _write_json(name: str, payload: object) -> None:
    destination = _LIFECYCLE_DIR / name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _source_identity() -> str:
    head = _git("rev-parse", "HEAD").strip()
    diff = _git("diff", "--binary", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    digest = hashlib.sha256(diff.encode())
    for relative in sorted(untracked):
        path = _REPOSITORY_ROOT / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return head if not diff and not untracked else f"{head}+dirty:{digest.hexdigest()}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _estimated_upper_cost(settings: Settings, models: tuple[CrusoeModel, ...]) -> Decimal:
    maximum = max(
        price for model in models for price in (model.pricing.prompt, model.pricing.completion)
    )
    return Decimal(settings.crusoe_max_total_tokens) * maximum / Decimal(1_000_000)


def _model_identity(model: CrusoeModel) -> str:
    return f"{model.id}@{model.model_revision}" if model.model_revision else model.id


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()

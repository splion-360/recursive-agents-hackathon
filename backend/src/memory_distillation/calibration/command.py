"""Composition root for the paid, shell-driven ChartQA calibration run."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from memory_distillation.benchmark import DevelopmentBenchmark
from memory_distillation.benchmark.manifests import load_manifest
from memory_distillation.calibration.models import (
    CalibrationProgress,
    CalibrationProvenance,
    CalibrationReport,
    ModelIdentities,
)
from memory_distillation.calibration.orchestrator import (
    calibrate_review_policy,
    combined_manifest_hash,
)
from memory_distillation.inference import (
    CrusoeHStarAnalyst,
    CrusoeReviewer,
    CrusoeStudentInference,
)
from memory_distillation.inference.crusoe import CrusoeModel, build_crusoe_client
from memory_distillation.sandbox import OpenShellSandbox
from memory_distillation.settings import Settings, get_settings
from memory_distillation.task_execution import ExecutionPolicy, TaskExecutor
from memory_distillation.telemetry import JsonlEventSink
from memory_distillation.visualizer import (
    CALIBRATION_EVENTS_PATH,
    CalibrationStatus,
    write_calibration_status,
)
from memory_distillation.visualizer.dashboard import CALIBRATION_ARTIFACT_DIR

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_ROOT = _REPOSITORY_ROOT / ".local" / "chartqa-source" / "ChartQA Dataset"
_DEFAULT_MANIFEST_DIR = _REPOSITORY_ROOT / "backend" / "experiment_manifests"


class DashboardProgressPublisher:
    """Translate calibration lifecycle state into the public dashboard contract."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._last: CalibrationProgress | None = None

    def publish(self, progress: CalibrationProgress) -> None:
        self._last = progress
        kappa = progress.kappa
        write_calibration_status(
            CalibrationStatus(
                run_id=self._run_id,
                state="complete" if progress.stage == "complete" else "running",
                stage=progress.stage,
                discovery_completed=progress.discovery_completed,
                regression_completed=progress.regression_completed,
                smoke_completed=progress.smoke_completed,
                smoke_total=progress.smoke_total,
                current_task_id=progress.current_task_id,
                proposal_id=progress.proposal.proposal_id if progress.proposal else None,
                proposal_hash=progress.proposal_hash,
                kappa_artifact_id=kappa.artifact_id if kappa else None,
                kappa_hash=kappa.content_hash if kappa else None,
                accepted_rule_ids=tuple(rule.rule_id for rule in kappa.rules) if kappa else (),
                rejected_rule_ids=progress.rejected_rule_ids,
                input_tokens=progress.input_tokens,
                output_tokens=progress.output_tokens,
                elapsed_ms=progress.elapsed_ms,
                estimated_cost_usd=float(progress.estimated_cost_usd),
                message=progress.message,
                updated_at=datetime.now(UTC),
            )
        )

    def fail(self, error: Exception) -> None:
        previous = self._last
        write_calibration_status(
            CalibrationStatus(
                run_id=self._run_id,
                state="failed",
                stage="failed",
                discovery_completed=previous.discovery_completed if previous else 0,
                regression_completed=previous.regression_completed if previous else 0,
                smoke_completed=previous.smoke_completed if previous else 0,
                smoke_total=previous.smoke_total if previous else 0,
                current_task_id=previous.current_task_id if previous else None,
                proposal_id=(
                    previous.proposal.proposal_id
                    if previous is not None and previous.proposal is not None
                    else None
                ),
                proposal_hash=previous.proposal_hash if previous else None,
                kappa_artifact_id=(
                    previous.kappa.artifact_id
                    if previous is not None and previous.kappa is not None
                    else None
                ),
                kappa_hash=(
                    previous.kappa.content_hash
                    if previous is not None and previous.kappa is not None
                    else None
                ),
                accepted_rule_ids=(
                    tuple(rule.rule_id for rule in previous.kappa.rules)
                    if previous is not None and previous.kappa is not None
                    else ()
                ),
                rejected_rule_ids=previous.rejected_rule_ids if previous else (),
                input_tokens=previous.input_tokens if previous else 0,
                output_tokens=previous.output_tokens if previous else 0,
                elapsed_ms=previous.elapsed_ms if previous else 0,
                estimated_cost_usd=(
                    float(previous.estimated_cost_usd) if previous else 0
                ),
                message=f"Calibration failed closed: {type(error).__name__}: {error}",
                updated_at=datetime.now(UTC),
            )
        )


def run_live_calibration(
    settings: Settings,
    *,
    run_id: str,
    dataset_root: Path,
    manifest_dir: Path,
    seed: int,
    pilot_tasks: int,
) -> CalibrationReport:
    """Compose verified live adapters and execute the paid calibration exactly once."""

    student_model = _required(settings.crusoe_student_model, "CRUSOE_STUDENT_MODEL")
    reviewer_model = _required(settings.crusoe_reviewer_model, "CRUSOE_REVIEWER_MODEL")
    hstar_model = _required(settings.crusoe_hstar_model, "CRUSOE_HSTAR_MODEL")
    sandbox_name = _required(settings.openshell_sandbox_name, "OPENSHELL_SANDBOX_NAME")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ChartQA dataset root not found: {dataset_root}")
    _prepare_artifact_directory()
    publisher = DashboardProgressPublisher(run_id)
    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    discovery_manifest = load_manifest(manifest_dir / "calibration_discovery.json")
    regression_manifest = load_manifest(manifest_dir / "calibration_regression.json")
    manifest_hash = combined_manifest_hash(
        discovery_manifest.content_sha256,
        regression_manifest.content_sha256,
    )
    try:
        with build_crusoe_client(settings) as client:
            identities = ModelIdentities(
                student=_model_identity(client.get_model(student_model)),
                reviewer=_model_identity(client.get_model(reviewer_model)),
                hstar=_model_identity(client.get_model(hstar_model)),
            )
            student = CrusoeStudentInference(
                client=client,
                model_id=student_model,
                max_output_tokens=settings.crusoe_student_max_output_tokens,
            )
            reviewer = CrusoeReviewer(
                client=client,
                model_id=reviewer_model,
                max_output_tokens=settings.crusoe_reviewer_max_output_tokens,
            )
            analyst = CrusoeHStarAnalyst(
                client=client,
                model_id=hstar_model,
                max_output_tokens=settings.crusoe_hstar_max_output_tokens,
            )
            executor = TaskExecutor(
                inference=student,
                reviewer=reviewer,
                sandbox=OpenShellSandbox(
                    sandbox_name=sandbox_name,
                    gateway=settings.openshell_gateway,
                ),
                verifier=benchmark,
                event_sink=JsonlEventSink(CALIBRATION_EVENTS_PATH),
            )
            report = calibrate_review_policy(
                run_id=run_id,
                source=benchmark,
                executor=executor,
                analyst=analyst,
                manifest_hash=manifest_hash,
                models=identities,
                provenance=CalibrationProvenance(
                    code_revision=_code_revision(),
                    seed=seed,
                    execution_policy=ExecutionPolicy().model_dump(mode="json"),
                    student_max_output_tokens=settings.crusoe_student_max_output_tokens,
                    reviewer_max_output_tokens=settings.crusoe_reviewer_max_output_tokens,
                    hstar_max_output_tokens=settings.crusoe_hstar_max_output_tokens,
                    sandbox_name=sandbox_name,
                    inference_base_url=str(settings.crusoe_base_url),
                ),
                publish=publisher.publish,
                cost_meter=lambda: client.spent_cost_usd,
                seed=seed,
                pilot_tasks=pilot_tasks,
            )
        _write_artifacts(report)
        return report
    except Exception as error:
        publisher.fail(error)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="memory-distillation calibrate", description=__doc__)
    parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Acknowledge paid Crusoe calls and managed OpenShell execution.",
    )
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=_DEFAULT_MANIFEST_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pilot-tasks", type=int, choices=range(3, 6), default=3)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable run identity; defaults to a UTC timestamp.",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    student_model = _required(settings.crusoe_student_model, "CRUSOE_STUDENT_MODEL")
    reviewer_model = _required(settings.crusoe_reviewer_model, "CRUSOE_REVIEWER_MODEL")
    hstar_model = _required(settings.crusoe_hstar_model, "CRUSOE_HSTAR_MODEL")
    sandbox_name = _required(settings.openshell_sandbox_name, "OPENSHELL_SANDBOX_NAME")
    with build_crusoe_client(settings) as preflight_client:
        preflight_models = tuple(
            preflight_client.get_model(model_id)
            for model_id in (student_model, reviewer_model, hstar_model)
        )
    resolved_identities = tuple(_model_identity(model) for model in preflight_models)
    max_token_price = max(
        price
        for model in preflight_models
        for price in (model.pricing.prompt, model.pricing.completion)
    )
    estimated_upper_cost = (
        Decimal(settings.crusoe_max_total_tokens)
        * max_token_price
        / Decimal(1_000_000)
    )
    print(
        "Calibration preflight\n"
        f"  Models: {resolved_identities[0]} -> {resolved_identities[1]} "
        f"(H-star: {resolved_identities[2]})\n"
        f"  Limits: <= {settings.crusoe_max_total_tokens:,} tokens, "
        f"estimated <= ${estimated_upper_cost}, {args.pilot_tasks} pilot tasks\n"
        f"  Runtime: OpenShell/{sandbox_name}; artifacts={CALIBRATION_ARTIFACT_DIR}"
    )
    if not args.confirm_paid_run:
        print("Dry preflight only; rerun with --confirm-paid-run to execute paid calls.")
        return
    run_id = args.run_id or datetime.now(UTC).strftime("calibration-%Y%m%dT%H%M%SZ")
    report = run_live_calibration(
        settings,
        run_id=run_id,
        dataset_root=args.dataset_root,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        pilot_tasks=args.pilot_tasks,
    )
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "proposal_hash": report.proposal_hash,
                "kappa_hash": report.kappa.content_hash,
                "kappa_revision": report.kappa.revision,
                "fallback_reason": report.fallback_reason,
                "total_cost_usd": str(report.total_cost_usd),
                "report_path": str(CALIBRATION_ARTIFACT_DIR / "report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _prepare_artifact_directory() -> None:
    CALIBRATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    conflicts = tuple(CALIBRATION_ARTIFACT_DIR.iterdir())
    if conflicts:
        names = ", ".join(path.name for path in conflicts)
        raise FileExistsError(
            f"calibration artifact directory is not empty ({names}); archive it before rerunning"
        )


def _write_artifacts(report: CalibrationReport) -> None:
    _atomic_write(
        "hstar.json",
        json.dumps(
            {
                "schema_version": "hstar-evolution.v1",
                "final_proposal_hash": report.proposal_hash,
                "revisions": [
                    revision.model_dump(mode="json")
                    for revision in report.hstar_revisions
                ],
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _atomic_write("kappa.json", report.kappa.model_dump_json(indent=2))
    _atomic_write("report.json", report.model_dump_json(indent=2))


def _atomic_write(name: str, content: str) -> None:
    destination = CALIBRATION_ARTIFACT_DIR / name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _model_identity(model: CrusoeModel) -> str:
    model_id = model.id
    revision = model.model_revision
    return f"{model_id}@{revision}" if revision else model_id


def _code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
